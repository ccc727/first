"""采样策略：greedy / top-k / top-p / temperature。

约定（model.generate 内部调用）：
    sample_logits(logits, temperature, top_k, top_p) -> 下一个 token id 的 1-D 张量
- temperature <= 0 退化为 greedy（argmax），避免除零；
- top_k <= 0 表示不做 top-k 截断；
- top_p <= 0 表示不做 nucleus 截断（调用方传 0 即关闭）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["top_k_filtering", "top_p_filtering", "sample_logits"]


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """只保留概率最大的 top_k 个词，其余置 -inf。"""
    if top_k <= 0:
        return logits
    if top_k == 1:
        return logits
    k = min(top_k, logits.size(-1))
    topk_vals, _ = torch.topk(logits, k)
    threshold = topk_vals[..., -1].unsqueeze(-1)
    return torch.where(logits >= threshold, logits, torch.tensor(float("-inf"),
                                                                device=logits.device))


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """nucleus sampling：按概率降序累加，累积概率超过 top_p 之后的词置 -inf。"""
    if top_p <= 0.0 or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    probs = F.softmax(sorted_logits, dim=-1)
    cumsum = torch.cumsum(probs, dim=-1)

    # 累积概率超过 top_p 的位置移除（至少保留一个词）
    remove_mask = cumsum - probs > top_p
    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

    # 还原到原始顺序
    return sorted_logits.scatter(-1, sorted_indices, sorted_logits)


def sample_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    """logits: (..., V)；返回一个形状同 logits 前几维的采样结果。"""
    if temperature is not None and temperature <= 0:
        return torch.argmax(logits, dim=-1)

    t = temperature if temperature and temperature > 0 else 1.0
    logits = logits / t
    logits = top_k_filtering(logits, top_k)
    logits = top_p_filtering(logits, top_p)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
