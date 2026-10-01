"""因果多头注意力 + KV cache（推理加速）。

对比任务一的 encoder attention，这里是 decoder-only 用的 **causal attention**：
- 注意力只能看到当前及之前的 token（上三角 mask）；
- 训练/全量前向时不带 cache，一次性算整段；
- 增量解码时把历史 K/V 缓存起来，每次只对**新 token** 做 QK^T，
  省去对历史序列的重复计算——这正是 KV cache「用显存换算力」的本质。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import apply_rope, precompute_rope

# 每个 cache 项：(K, V)，K/V 形状 (B, n_heads, T, head_dim)
KVCache = tuple[torch.Tensor, torch.Tensor]


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int,
                 dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.max_seq_len = max_seq_len

        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 预计算 RoPE 表：留出冗余（自检分窗会喂 block_size+1 长度，位置索引可到 block_size）
        _rope_len = max_seq_len + 8
        cos, sin = precompute_rope(self.head_dim, _rope_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, 3*d) -> 3 x (B, H, T, head_dim)
        B, T, _ = x.shape
        qkv = x.view(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * D)

    def forward(
        self,
        x: torch.Tensor,
        past_kv: KVCache | None = None,
    ) -> tuple[torch.Tensor, KVCache]:
        """x: (B, T, d_model)；返回 (输出, 更新后的 (K, V) cache)。"""
        B, T, _ = x.shape
        q, k, v = self._split(self.c_attn(x))       # (B,H,T,Dh)

        n_past = 0 if past_kv is None else past_kv[0].size(2)
        # RoPE 作用在 Q/K 上；增量解码时新 token 的绝对位置从 n_past 续起
        q = apply_rope(q, self.cos, self.sin, start=n_past)
        k = apply_rope(k, self.cos, self.sin, start=n_past)

        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        cache = (k, v)

        # 因果 mask：当前这批 T 个 query 只能看到自己及之前的 key。
        # 列 j 被屏蔽当且仅当 j > n_past + row。
        T_total = n_past + T
        mask = torch.triu(
            torch.ones(T, T_total, dtype=torch.bool, device=x.device),
            diagonal=1 + n_past,
        )

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)   # (B,H,T,T_total)
        att = att.masked_fill(mask, float("-inf"))
        att_w = F.softmax(att, dim=-1)
        att_w = self.dropout(att_w)
        out = att_w @ v                                            # (B,H,T,Dh)
        out = self._merge(out)
        out = self.resid_dropout(self.c_proj(out))                 # (B,T,d)
        return out, cache
