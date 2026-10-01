# -*- coding: utf-8 -*-
"""plot_results.py —— 任务二结果图（AUC / 时间对比 / 时间-AUC 权衡）。"""
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

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(CODE_DIR, "results", "results.csv")
OUT = os.path.join(CODE_DIR, "results", "figures")

M_ORDER = ["gcn", "gat", "sage", "gin"]
M_LABEL = {"gcn": "GCN", "gat": "GAT", "sage": "SAGE", "gin": "GIN"}
MODE_LABEL = {"full": "全图", "sample": "采样"}
COLOR = {"full": "#4C72B0", "sample": "#DD8452"}


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", os.path.join(OUT, name))


def bar(df, col, ylabel, title, fname, log=False, ylim=None):
    order = []
    for ds in sorted(df["dataset"].unique()):
        for m in M_ORDER:
            if ((df.dataset == ds) & (df.model == m)).any():
                order.append((ds, m))
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.9), 5.4))
    for j, mode in enumerate(["full", "sample"]):
        vals = []
        for ds, m in order:
            sub = df[(df.dataset == ds) & (df.model == m) &
                     (df["mode"] == mode)]
            vals.append(sub[col].mean() if len(sub) else np.nan)
        ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, label=MODE_LABEL[mode],
               color=COLOR[mode])
    ax.set_xticks(x)
    ax.set_xticklabels(["%s-%s" % (d.capitalize(), M_LABEL[m])
                        for d, m in order], rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, fname)


def scatter(df):
    fig, ax = plt.subplots(figsize=(8.5, 6))
    mk = {"full": "o", "sample": "s"}
    for ds in sorted(df["dataset"].unique()):
        for mode in ["full", "sample"]:
            sub = df[(df.dataset == ds) & (df["mode"] == mode)]
            if not len(sub):
                continue
            ax.scatter(sub["train_sec"], sub["test_auc"], s=95,
                       marker=mk[mode],
                       label="%s-%s" % (ds.capitalize(), MODE_LABEL[mode]))
            for _, r in sub.iterrows():
                ax.annotate(M_LABEL[r["model"]], (r["train_sec"],
                                                  r["test_auc"]),
                            textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("训练总耗时 (秒, log)")
    ax.set_ylabel("测试 AUC")
    ax.set_title("任务二：链路预测 训练总耗时 vs 测试 AUC")
    ax.legend()
    ax.grid(alpha=0.3)
    save(fig, "total_time_vs_auc.png")


def main():
    df = pd.read_csv(CSV)
    print(df.groupby(["dataset", "model", "mode"])["test_auc"].mean())
    bar(df, "test_auc", "测试集 AUC", "任务二：链路预测测试 AUC（全图 vs 采样）",
        "test_auc_comparison.png", ylim=(0.4, 1.0))
    bar(df, "avg_epoch_sec", "平均每轮训练耗时(s, log)",
        "任务二：平均每轮训练耗时（全图 vs 采样）",
        "epoch_time_comparison.png", log=True)
    scatter(df)


if __name__ == "__main__":
    main()
