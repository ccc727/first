"""手写 scaled dot-product attention 与 multi-head attention（不允许 nn.MultiheadAttention）。

约定：
- `scaled_dot_product_attention(Q, K, V, mask=None)`
  Q/K/V 形状 (B, H, T, D)；mask 广播到 (B, H, T, T)，`True` = 被屏蔽。
  与 torch 官方 `F.scaled_dot_product_attention` 数值误差 < 1e-5。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scaled dot-product attention：Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V。

    Args:
        Q, K, V: 形状 (B, H, T, D) 的查询 / 键 / 值。
        mask: 可选布尔 mask，`True` 表示屏蔽该位置。形状可广播到 (B, H, T, T)，
            例如 (T, T)（causal 上三角）或 (B, 1, 1, T)（padding key）。
            屏蔽位置在 softmax 前填充为 -inf，保证不泄漏信息。
    """
    d_k = Q.size(-1)
    # (B, H, T_q, T_k)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V)


class MultiHeadAttention(nn.Module):
    """多头自注意力：d_model -> n_heads 个 d_k 维子空间并行做 attention 再拼接。

    结构：Wq/Wk/Wv 各自把输入投影到 d_model，拆成 n_heads 个头，
    每个头独立做 scaled dot-product attention，拼接后过 Wo 输出投影。
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        # 最近一次 forward 的注意力权重（用于可视化），形状 (B, H, T, T)
        self.attn_weights: torch.Tensor | None = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) -> (B, n_heads, T, head_dim)
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_heads, T, head_dim) -> (B, T, d_model)
        B, _, T, _ = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        """x: (B, T, d_model)；padding_mask: (B, T)，True = pad 位置。"""
        B, T, _ = x.shape

        q = self._split_heads(self.wq(x))  # (B, H, T, d_k)
        k = self._split_heads(self.wk(x))
        v = self._split_heads(self.wv(x))

        if padding_mask is not None:
            # key 侧的 padding 位置不可被 attend：(B, 1, 1, T) -> 广播到 (B, H, T, T)
            attn_mask = padding_mask.unsqueeze(1).unsqueeze(2)
        else:
            attn_mask = None

        context = scaled_dot_product_attention(q, k, v, mask=attn_mask)

        # 需要可视化时，回算并缓存 softmax 权重（不计梯度）
        if return_weights:
            d_k = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask, float("-inf"))
            self.attn_weights = F.softmax(scores, dim=-1)

        out = self._merge_heads(context)          # (B, T, d_model)
        out = self.dropout(self.wo(out))
        return out
