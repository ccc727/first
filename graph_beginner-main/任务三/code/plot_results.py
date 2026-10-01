# -*- coding: utf-8 -*-
"""plot_results.py —— 任务三结果图（池化对比 + 模型对比）。"""
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
M_ORDER = ["gcn", "gat", "sage", "gin"]
M_LABEL = {"gcn": "GCN", "gat": "GAT", "sage": "SAGE", "gin": "GIN"}
POOL_ORDER = ["mean", "max", "min"]
POOL_LABEL = {"mean": "AvgPool", "max": "MaxPool", "min": "MinPool"}
C_POOL = {"mean": "#4C72B0", "max": "#DD8452", "min": "#55A868"}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", os.path.join(OUT, name))


def metric_col(df):
    if "test_acc" in df.columns:
        return "test_acc", "分类准确率"
    return "test_mae", "MAE(还原尺度)"


def col_for(df, ds):
    sub = df[df["dataset"] == ds]
    if "test_acc" in sub and sub["test_acc"].notna().any():
        return "test_acc", "分类准确率"
    return "test_mae", "MAE(原始尺度)"


def plot_pooling(df):
    """每个数据集一张图：x=model，分组柱=三种池化。"""
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        col, ylabel = col_for(df, ds)
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(len(M_ORDER))
        for j, p in enumerate(POOL_ORDER):
            vals = []
            for m in M_ORDER:
                s2 = sub[(sub["model"] == m) & (sub["pool"] == p)]
                vals.append(s2[col].mean() if len(s2) else np.nan)
            ax.bar(x + (j - 0.8) * 0.27, vals, 0.27, label=POOL_LABEL[p],
                   color=C_POOL[p])
        ax.set_xticks(x)
        ax.set_xticklabels([M_LABEL[m] for m in M_ORDER])
        ax.set_title(f"{ds.upper()}：不同池化方法")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        save(fig, f"pooling_comparison_{ds}.png")


def plot_model(df):
    """分类数据集模型对比（池化取平均，仅 mutag 等含 acc 的数据集）。"""
    cls = [d for d in df["dataset"].unique()
           if "test_acc" in df[df["dataset"] == d]
           and df[df["dataset"] == d]["test_acc"].notna().any()]
    if not cls:
        return
    sub = df[df["dataset"].isin(cls)]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    x = np.arange(len(cls))
    for j, m in enumerate(M_ORDER):
        vals = [sub[(sub["dataset"] == d) & (sub["model"] == m)]["test_acc"].mean()
                if len(sub[(sub["dataset"] == d) & (sub["model"] == m)]) else np.nan
                for d in cls]
        ax.bar(x + (j - 1.5) * 0.2, vals, 0.2, label=M_LABEL[m])
    ax.set_xticks(x)
    ax.set_xticklabels([d.upper() for d in cls])
    ax.set_ylabel("测试准确率")
    ax.set_title("任务三：模型对比（池化取平均）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, "model_comparison.png")


def main():
    df = pd.read_csv(CSV)
    print(df.to_string(index=False))
    plot_pooling(df)
    plot_model(df)


if __name__ == "__main__":
    main()
