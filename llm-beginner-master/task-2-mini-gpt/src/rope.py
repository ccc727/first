"""RoPE（Rotary Position Embedding）—— 旋转位置编码。

只作用于 Q/K（绝不动 V）。实现为相邻两维一组的旋转：
    (x_{2i}, x_{2i+1}) -> (x_{2i}*cos - x_{2i+1}*sin, x_{2i}*sin + x_{2i+1}*cos)
角度按 token 的**绝对位置**计算，天然把位置信息编码进点积，使 attention 分数只依赖
相对位置差（relative position），从而具备比绝对位置编码更好的长度外推能力。
"""
from __future__ import annotations

import torch

__all__ = ["precompute_rope", "apply_rope"]


def precompute_rope(
    head_dim: int,
    max_seq_len: int,
    base: float = 10000.0,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """预计算 cos / sin 表。

    Returns:
        cos, sin: 形状都是 (max_seq_len, head_dim // 2)。
        cos[p, i] / sin[p, i] 对应第 p 个位置、第 i 组相邻维 (2i, 2i+1) 的角度。
    """
    assert head_dim % 2 == 0
    # 每个维组的频率：theta_i = base^(-2i / head_dim)
    i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    theta = 1.0 / (base ** (i / head_dim))          # (head_dim/2,)
    pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(pos, theta)                # (max_seq_len, head_dim/2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start: int = 0,
) -> torch.Tensor:
    """把旋转位置编码作用到 Q/K。

    Args:
        x: 形状 (B, H, L, D)，D = head_dim。
        cos / sin: 预计算表 (max_seq_len, D/2)。
        start: 这批 token 的起始绝对位置（KV cache 增量解码时用历史长度续上）。

    Returns:
        旋转后的张量，形状不变。位置 p 使用 cos[start+p] / sin[start+p]。
    """
    B, H, L, D = x.shape
    half = D // 2
    c = cos[start:start + L].unsqueeze(0).unsqueeze(0)   # (1,1,L,half)
    s = sin[start:start + L].unsqueeze(0).unsqueeze(0)

    x = x.contiguous().view(B, H, L, half, 2)            # 相邻两维一组
    x0 = x[..., 0]                                       # (B,H,L,half)
    x1 = x[..., 1]
    r0 = x0 * c - x1 * s
    r1 = x0 * s + x1 * c
    return torch.stack([r0, r1], dim=-1).view(B, H, L, D)
