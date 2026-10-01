"""Generator：把检索片段拼 prompt，调本地 Qwen（OpenAI 兼容）；不可用时返回 None。"""
from __future__ import annotations

from .rag import _call_llm

SYSTEM = ("你是一个严谨的中文问答助手。只能依据给定的参考资料回答问题，"
          "不要使用资料之外的知识；如果资料不足以回答，请直接说明不知道。")


def generate(prompt: str) -> str | None:
    return _call_llm(prompt)
