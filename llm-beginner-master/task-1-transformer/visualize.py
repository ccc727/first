"""任务一可视化：把训练好的模型的注意力权重画成热图（M5）。

产出 figures/ 下 ≥3 张图：
    attn_pos.png   一条正面样本（含「不错/赞」等关键词）
    attn_neg.png   一条负面样本（含「失望/差」等关键词）
    attn_long.png  一条较长句子样本
用法：python visualize.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from src.model import load_for_eval, tokenize

ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
CKPT = ROOT / "ckpt" / "best.pt"

# 中文字体回退链（Windows 常见中文字体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

POS_SENT = "物流很快，包装也很仔细，卖家服务态度不错，东西质量非常好，值得推荐！"
NEG_SENT = "收到货非常失望，质量差得离谱，客服态度也不好，以后再也不会买了，差评！"
LONG_SENT = (
    "手机用了三天才到货，屏幕有一个亮点，申请换货后客服一直不回复，"
    "联系了好几次才处理，前后折腾了差不多两个星期，体验真的很一般，"
    "建议大家谨慎购买，不如去实体店看看。"
)


def collect_weights(model, sentence):
    """对单条句子做一次前向，收集每个 block 的 attn_weights。

    Returns: weights[layer][head] 形状 (T, T) numpy。
    """
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    # 先清掉旧缓存
    for block in model.blocks:
        block.attn.attn_weights = None

    text = sentence[: model.max_len]
    with torch.no_grad():
        ids = torch.tensor(tokenize(text, model.max_len), dtype=torch.long).unsqueeze(0)
        ids = ids.to(next(model.parameters()).device)
        _ = model(ids, return_weights=True)

    weights = []
    for block in model.blocks:
        w = block.attn.attn_weights  # (1, n_heads, T, T)
        weights.append(w[0].cpu().numpy())
    return weights, list(text)


def plot_one(ax, mat, tokens, title, layer, head):
    T = mat.shape[0]
    ax.imshow(mat, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(T))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticks(range(T))
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_title(f"{title} | layer {layer} head {head}", fontsize=10)
    ax.set_xlabel("key")
    ax.set_ylabel("query")


def main():
    if not CKPT.exists():
        raise SystemExit("ckpt/best.pt 不存在，先跑 python train.py")
    FIG_DIR.mkdir(exist_ok=True)
    model, _ = load_for_eval(str(CKPT))

    cases = [
        ("attn_pos", POS_SENT, "正面样本"),
        ("attn_neg", NEG_SENT, "负面样本"),
        ("attn_long", LONG_SENT, "长句样本"),
    ]

    for fname, sentence, title in cases:
        weights, tokens = collect_weights(model, sentence)
        # 挑一个看起来有结构的层 / 头：取最后两层各一个 head
        layer = len(weights) - 1
        head = 0
        mat = weights[layer][head][: len(tokens), : len(tokens)]
        fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.28), max(5, len(tokens) * 0.24)))
        plot_one(ax, mat, tokens, title, layer, head)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{fname}.png", dpi=160)
        plt.close(fig)
        print(f"[figure] {FIG_DIR / (fname + '.png')}  shape={mat.shape}")

    # 额外：对一个样本画同一 layer 所有 head 并排对比（2x2 grid）
    weights, tokens = collect_weights(model, POS_SENT)
    n_heads = weights[0].shape[0]
    layer = len(weights) - 1
    fig, axes = plt.subplots(2, max(2, (n_heads + 1) // 2), figsize=(14, 6))
    axes = np.array(axes).reshape(-1)
    for h in range(n_heads):
        mat = weights[layer][h][: len(tokens), : len(tokens)]
        plot_one(axes[h], mat, tokens, "正面样本 heads", layer, h)
    for h in range(n_heads, len(axes)):
        axes[h].axis("off")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "attn_heads_grid.png", dpi=150)
    plt.close(fig)
    print("[figure] figures/attn_heads_grid.png")


if __name__ == "__main__":
    main()
