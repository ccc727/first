"""MiniGPT：decoder-only 语言模型（手写，集成 RoPE + KV cache）。

自检契约（eval/run.py 依赖）：
    class MiniGPT(nn.Module)
        forward(ids, kv_cache=None, return_cache=False)
            无 cache:  返回 logits (B, T, vocab)
            有 cache:  返回 (logits, new_cache)
        generate(prompt_ids, max_new_tokens, top_k, top_p, temperature)
        block_size / max_seq_len 属性
    load_for_eval(ckpt_path) -> (model, tokenizer)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .rope import precompute_rope


class MLP(nn.Module):
    """Position-wise FFN：d -> 4d -> d，GELU 激活。"""

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * d_model, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Pre-LN decoder block：self-attention + FFN，各带残差。"""

    def __init__(self, d_model: int, n_heads: int, max_seq_len: int,
                 dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)

    def forward(self, x: torch.Tensor, past_kv=None):
        h = self.ln1(x)
        attn_out, kv = self.attn(h, past_kv)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, kv


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 6,
        block_size: int = 128,
        dropout: float = 0.1,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size
        self.max_seq_len = block_size  # 自检两个属性名都读得到

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, block_size, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, ids, kv_cache=None, return_cache=False):
        """ids: (B, T) LongTensor。返回 logits 或 (logits, cache)。"""
        ids = ids.to(self.tok_emb.weight.device)
        x = self.drop(self.tok_emb(ids))
        new_cache = [] if (return_cache or kv_cache is not None) else None
        for i, blk in enumerate(self.blocks):
            past = kv_cache[i] if kv_cache is not None else None
            x, kv = blk(x, past)
            if new_cache is not None:
                new_cache.append(kv)
        logits = self.lm_head(self.ln_f(x))
        if return_cache:
            return logits, new_cache
        return logits

    # ------------------------------------------------------------------ 生成
    @torch.no_grad()
    def generate(
        self,
        prompt_ids,
        max_new_tokens: int = 100,
        top_k: int = 0,
        top_p: float = 0.0,
        temperature: float = 1.0,
    ) -> list[int]:
        """自回归生成：从 prompt token ids 续写 max_new_tokens 个 token。

        内部自动使用 KV cache 增量解码（只喂最后一个 token + 缓存）。
        返回完整 token id 序列（prompt + 新生成）。
        """
        from .sampling import sample_logits

        device = next(self.parameters()).device
        if not isinstance(prompt_ids, torch.Tensor):
            prompt = torch.tensor(prompt_ids, dtype=torch.long, device=device)
        else:
            prompt = prompt_ids.to(device)
        prompt = prompt.view(1, -1)[:, -self.block_size:]

        generated = prompt.tolist()[0]
        cache = None
        for _ in range(max_new_tokens):
            if cache is None:
                logits, cache = self(prompt, return_cache=True)
            else:
                cur = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
                logits, cache = self(cur, kv_cache=cache, return_cache=True)
            next_id = sample_logits(
                logits[:, -1, :], temperature=temperature,
                top_k=top_k, top_p=top_p,
            )
            generated.append(int(next_id.item()))
        return generated


# ---------------------------------------------------------------------------
# 自检契约：load_for_eval
# ---------------------------------------------------------------------------
def load_for_eval(ckpt_path: str):
    """从 best.pt 恢复 (model, tokenizer)。"""
    ckpt_path = str(ckpt_path)
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = data["config"]
    model = MiniGPT(**cfg)
    model.load_state_dict(data["state_dict"])
    # 保持在 CPU：eval/run.py 的 ppl 测试在 CPU 上构造 target chunk，
    # 若模型在 cuda 上会让 harness 内的 cross_entropy 出现设备不一致。
    model.eval()
    model.to("cpu")

    from pathlib import Path

    from .tokenizer import BPETokenizer

    tok_path = data.get("tokenizer_path") or str(Path(ckpt_path).parent / "tokenizer.json")
    tok = BPETokenizer.from_pretrained(tok_path)
    return model, tok

