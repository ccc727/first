"""手写 Transformer encoder block：attention + FFN + 两个 residual + 两个 LayerNorm。

采用 Pre-LN 结构（先 LayerNorm 再进子层，残差在最后相加），这是 GPT/BERT 系
现代实现的主流做法，训练更稳（参考任务 README「知识点：Pre-LN vs Post-LN」）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention


class FeedForward(nn.Module):
    """Position-wise FFN：Linear(d_model -> d_ff) -> GELU -> Linear(d_ff -> d_model)。"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(nn.Module):
    """一个 Pre-LN Transformer encoder block。

    x = x + Attn(LN1(x))
    x = x + FFN(LN2(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        return_weights: bool = False,
    ) -> torch.Tensor:
        # 子层 1：多头自注意力 + 残差
        h = self.ln1(x)
        h = self.attn(h, padding_mask=padding_mask, return_weights=return_weights)
        x = x + h
        # 子层 2：FFN + 残差
        x = x + self.ffn(self.ln2(x))
        return x
