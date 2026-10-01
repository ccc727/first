"""Qwen chat template 与 loss masking。

- format_messages：套 Qwen 官方 ChatML 模板
    <|im_start|>system\n...<|im_end|>\n
    <|im_start|>user\n...<|im_end|>\n
    <|im_start|>assistant\n...<|im_end|>\n
- build_labels：把 input_ids 里「user / system 内容 + 模板控制符」标成 -100，
  只对 assistant 的正文 token 保留原 id 计算 loss。
"""
from __future__ import annotations

from pathlib import Path

import torch

_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>\n"

_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "Qwen2.5-0.5B"

_tok_cache = None


_SIMPLE_CHAT_TEMPLATE = (
    "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + "
    "message['content'] + '<|im_end|>' + '\\n'}}{% endfor %}"
    "{% if add_generation_prompt %}{{'<|im_start|>assistant\\n'}}{% endif %}"
)


def _get_tokenizer():
    global _tok_cache
    if _tok_cache is None:
        from transformers import AutoTokenizer
        _tok_cache = AutoTokenizer.from_pretrained(str(_MODEL_DIR))
        # 统一成与 format_messages 完全一致的 ChatML 渲染（含 generation 标记）
        _tok_cache.chat_template = _SIMPLE_CHAT_TEMPLATE
    return _tok_cache


def format_messages(messages: list[dict]) -> str:
    """把 [{'role':..., 'content':...}] 渲染为 Qwen ChatML 字符串。"""
    text = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        text += f"{_IM_START}{role}\n{content}{_IM_END}"
    return text


def _build_message_parts(messages) -> list[tuple[str, str]]:
    """把消息切分成 (kind, text) 片段：kind in {'meta','sys','user','assistant'}。"""
    parts: list[tuple[str, str]] = []
    for msg in messages:
        role = msg.get("role", "user")
        content = str(msg.get("content", ""))
        parts.append(("meta", f"{_IM_START}{role}\n"))
        if role == "assistant":
            parts.append(("assistant", content))
        elif role == "system":
            parts.append(("sys", content))
        else:
            parts.append(("user", content))
        parts.append(("meta", _IM_END))
    return parts


def build_labels(input_ids, messages) -> torch.Tensor:
    """labels 与 input_ids 同形状；仅 assistant 正文 token 保留，其余 -100。

    用 tokenizer.apply_chat_template 的 assistant_tokens_mask 精确定位
    assistant 正文（与整段 tokenization 天然对齐，避免跨边界 BPE 的偏差）。
    """
    tok = _get_tokenizer()
    try:
        out = tok.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            return_assistant_tokens_mask=True)
        template_ids = out["input_ids"]
        assist_mask = out["assistant_tokens_mask"]
    except (TypeError, ValueError, KeyError):
        # 兜底：退化为启发式（对给定 ids 本身不做改动）
        template_ids = None

    ids = input_ids.detach().cpu().tolist()
    labels = torch.full_like(input_ids, -100)
    if template_ids is not None and len(template_ids) == len(ids):
        # 语言建模位移：用位置 i 的 logits 预测 token[i+1]。
        # 仅当 i+1 是 assistant 正文时保留 target=token[i+1]。
        for i in range(len(ids) - 1):
            if assist_mask[i + 1] and template_ids[i + 1] == ids[i + 1]:
                labels[i] = ids[i + 1]
        return labels

    # 老路径兜底（同样做位移）：assistant 片段 p..p+m-1 对应 target 放在 p-1..p+m-2
    parts = _build_message_parts(messages)
    pos = 0
    for kind, text in parts:
        if not text:
            continue
        piece_ids = tok(text, add_special_tokens=False).input_ids
        if not piece_ids:
            continue
        start = _find_subseq(ids, piece_ids, pos)
        if start is None:
            continue
        if kind == "assistant":
            for j, t in enumerate(piece_ids):
                p = start + j
                if p > 0 and p < len(ids) and ids[p] == t:
                    labels[p - 1] = ids[p]  # 用 p-1 位置的 logits 预测 content token p
        pos = start + len(piece_ids)
    return labels


def _find_subseq(hay, needle, start=0):
    n, m = len(hay), len(needle)
    if m == 0:
        return None
    for i in range(start, n - m + 1):
        if hay[i:i + m] == needle:
            return i
    return None
