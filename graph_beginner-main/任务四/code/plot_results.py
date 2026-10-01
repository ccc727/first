# -*- coding: utf-8 -*-
"""plot_results.py —— 任务四结果图（MRR / Hits@10 / 训练耗时）。"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm
import matplotlib.pyplot as plt
for _f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
    if _f in {f.name for f in _fm.fontManager.ttflist}:
        plt.rcParams["font.sans-serif"] = [_f]
        break
plt.rcParams["axes.unicode_minus"] = False
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CODE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(CODE, "results", "results.csv")
OUT = os.path.join(CODE, "results", "figures")
M = ["transe", "rotate", "conve"]
LABEL = {"transe": "TransE", "rotate": "RotatE", "conve": "ConvE"}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", os.path.join(OUT, name))


def main():
    df = pd.read_csv(CSV)
    print(df[["model", "test_mrr", "hits@10", "hits@3", "hits@1",
              "train_sec"]].to_string(index=False))

    x = np.arange(len(M))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    metrics = [("test_mrr", "测试 MRR"), ("hits@10", "测试 Hits@10"),
               ("train_sec", "训练总耗时(秒)")]
    for ax, (col, ylab) in zip(axes, metrics):
        vals = [df[df.model == m][col].mean() if len(df[df.model == m]) else 0
                for m in M]
        ax.bar(x, vals, 0.55, color=["#4C72B0", "#DD8452", "#55A868"])
        ax.set_xticks(x)
        ax.set_xticklabels([LABEL[m] for m in M])
        ax.set_ylabel(ylab)
        ax.set_title("知识图谱补全：%s" % ylab)
        for xi, v in zip(x, vals):
            ax.text(xi, v, "%.3f" % v, ha="center", va="bottom", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("任务四：TransE / RotatE / ConvE 在 FB15k-237 上的表现", y=1.03)
    fig.tight_layout()
    save(fig, "kge_comparison.png")


if __name__ == "__main__":
    main()
