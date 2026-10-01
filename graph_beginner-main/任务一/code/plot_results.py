# -*- coding: utf-8 -*-
"""
plot_results.py —— 读取 results.csv 生成任务一结果报告图

输出到 <code>/results/figures/：
  1. test_acc_comparison.png  ：各 (数据集, 模型) 全图 vs 采样 的测试准确率对比
  2. epoch_time_comparison.png：各 (数据集, 模型) 全图 vs 采样 的每轮训练耗时(对数轴)
  3. total_time_vs_acc.png    ：训练总耗时 vs 测试准确率 散点（时间-精度权衡）
  4. param_sweep.png          ：若有 lr/layers 扫描数据则输出参数影响图

用法：
    python plot_results.py
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(CODE_DIR, "results", "results.csv")
OUT_DIR = os.path.join(CODE_DIR, "results", "figures")

for _f in ["Microsoft YaHei", "SimHei", "WenQuanYi Zen Hei",
           "Noto Sans CJK SC", "Arial Unicode MS"]:
    try:
        if _f in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [_f]
            break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

MODEL_ORDER = ["gcn", "gat", "sage", "gin"]
MODEL_LABEL = {"gcn": "GCN", "gat": "GAT", "sage": "GraphSAGE", "gin": "GIN"}
MODE_LABEL = {"full": "全图训练", "sample": "采样训练"}
COLORS = {"full": "#4C72B0", "sample": "#DD8452"}


def _save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", path)


def _groups(df):
    """按 (dataset, model) 排序返回全部组合。"""
    order = []
    for ds in sorted(df["dataset"].unique()):
        for m in MODEL_ORDER:
            if ((df["dataset"] == ds) & (df["model"] == m)).any():
                order.append((ds, m))
    return order


def plot_acc(df):
    """图1：测试准确率 grouped bar。"""
    order = _groups(df)
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.9), 5.2))
    x = np.arange(len(order))
    for j, mode in enumerate(["full", "sample"]):
        vals = []
        for ds, m in order:
            sub = df[(df.dataset == ds) & (df.model == m) &
                     (df["mode"] == mode)]
            vals.append(sub["test_acc"].mean() if len(sub) else np.nan)
        ax.bar(x + (j - 0.5) * width, vals, width,
               label=MODE_LABEL[mode], color=COLORS[mode])
    ax.set_xticks(x)
    labels = ["%s-%s" % (ds.capitalize(), MODEL_LABEL[m]) for ds, m in order]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("测试集准确率")
    ax.set_ylim(0, 1.0)
    ax.set_title("任务一：节点分类 测试准确率（全图 vs 采样训练）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "test_acc_comparison.png")


def plot_time(df):
    """图2：平均每轮训练耗时（对数轴）。"""
    order = _groups(df)
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(8, len(order) * 0.9), 5.2))
    x = np.arange(len(order))
    for j, mode in enumerate(["full", "sample"]):
        vals = []
        for ds, m in order:
            sub = df[(df.dataset == ds) & (df.model == m) &
                     (df["mode"] == mode)]
            vals.append(sub["avg_epoch_sec"].mean() if len(sub) else np.nan)
        ax.bar(x + (j - 0.5) * width, vals, width,
               label=MODE_LABEL[mode], color=COLORS[mode])
    ax.set_xticks(x)
    labels = ["%s-%s" % (ds.capitalize(), MODEL_LABEL[m]) for ds, m in order]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylim(bottom=5e-4)
    ax.set_ylabel("平均每轮训练耗时 (秒, log)")
    ax.set_title("任务一：平均每轮训练耗时（全图 vs 采样训练）")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, which="both")
    _save(fig, "epoch_time_comparison.png")


def plot_time_acc(df):
    """图3：训练总耗时 vs 测试准确率 散点。"""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    markers = {"full": "o", "sample": "s"}
    for ds in sorted(df["dataset"].unique()):
        sub = df[df.dataset == ds]
        for mode in ["full", "sample"]:
            s2 = sub[sub["mode"] == mode]
            if len(s2) == 0:
                continue
            ax.scatter(s2["train_sec"], s2["test_acc"], s=90, alpha=0.85,
                       marker=markers[mode],
                       label="%s-%s" % (ds.capitalize(), MODE_LABEL[mode]))
            for _, row in s2.iterrows():
                ax.annotate(MODEL_LABEL[row["model"]],
                            (row["train_sec"], row["test_acc"]),
                            textcoords="offset points", xytext=(4, 4),
                            fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("训练总耗时 (秒, log)")
    ax.set_ylabel("测试集准确率")
    ax.set_title("任务一：训练总耗时 vs 测试准确率")
    ax.legend()
    ax.grid(alpha=0.3)
    _save(fig, "total_time_vs_acc.png")


def plot_sweep(df):
    """图4：学习率 / 层数 扫描（需先跑过 --sweep-lr/--sweep-layers）。"""
    if not {"lr", "layers"}.issubset(df.columns):
        return
    cols = [c for c in ["dataset", "model", "mode", "lr", "layers"] if c in df]
    g = df.groupby(cols, dropna=False)["test_acc"].mean().reset_index()
    if len(g) == 0 or g["lr"].nunique() < 2:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for (ds, model, mode), sub in g.groupby(["dataset", "model", "mode"]):
        axes[0].plot(sub["lr"], sub["test_acc"], marker="o",
                     label="%s-%s-%s" % (ds, MODEL_LABEL.get(model, model), mode))
    axes[0].set_xscale("log")
    axes[0].set_xlabel("学习率 (log)")
    axes[0].set_ylabel("测试准确率")
    axes[0].set_title("学习率对性能的影响")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)
    for (ds, model, mode), sub in g.groupby(["dataset", "model", "mode"]):
        if sub["layers"].nunique() < 2:
            continue
        axes[1].plot(sub["layers"], sub["test_acc"], marker="o",
                     label="%s-%s-%s" % (ds, MODEL_LABEL.get(model, model), mode))
    axes[1].set_xlabel("网络层数")
    axes[1].set_ylabel("测试准确率")
    axes[1].set_title("层数对性能的影响")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, "param_sweep.png")


def main():
    parser = argparse.ArgumentParser(description="任务一结果绘图")
    parser.add_argument("--results", type=str, default=DEFAULT_CSV)
    args = parser.parse_args()
    if not os.path.exists(args.results):
        print(f"找不到结果文件 {args.results}，请先运行训练脚本。")
        return
    df = pd.read_csv(args.results)
    print("read rows:", len(df))
    plot_acc(df)
    plot_time(df)
    plot_time_acc(df)
    plot_sweep(df)
    print("全部图片已输出到", OUT_DIR)


if __name__ == "__main__":
    main()
