# -*- coding: utf-8 -*-
"""
analysis.py —— 汇总 results.csv 并打印分析表格

用于报告：
  1) 不同模型 + 全图/采样训练的 测试准确率 与 训练耗时对比
  2) 不同学习率 / 层数 对性能的影响

用法:
    python analysis.py                       # 默认读取 results/results.csv
    python analysis.py --results xxx.csv
"""
import argparse
import os

import pandas as pd

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(CODE_DIR, "results", "results.csv")


def fmt(v):
    if pd.isna(v):
        return "N/A"
    return f"{float(v):.4f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default=DEFAULT_CSV)
    args = parser.parse_args()

    if not os.path.exists(args.results):
        print(f"找不到结果文件 {args.results}，请先运行训练脚本。")
        return

    df = pd.read_csv(args.results)

    print("=" * 90)
    print("表1：各 (数据集, 模型, 训练方式) 的测试准确率 与 训练耗时")
    print("=" * 90)
    cols = ["dataset", "model", "mode"]
    g = df.groupby(cols, dropna=False).agg(
        测试准确率=("test_acc", "mean"),
        最优验证=("best_val_acc", "mean"),
        平均每轮秒=("avg_epoch_sec", "mean"),
        训练总秒=("train_sec", "mean"),
        运行次数=("test_acc", "count"),
    ).reset_index()
    with pd.option_context("display.width", 120, "display.max_rows", None):
        print(g.to_string(index=False))

    print("\n" + "=" * 90)
    print("表2：学习率 / 层数 扫描结果（若存在）")
    print("=" * 90)
    if {"lr", "layers"}.issubset(df.columns):
        g2 = df.groupby(["dataset", "model", "lr", "layers"], dropna=False).agg(
            测试准确率=("test_acc", "mean"),
            平均每轮秒=("avg_epoch_sec", "mean"),
        ).reset_index()
        with pd.option_context("display.width", 120, "display.max_rows", None):
            print(g2.to_string(index=False))
    else:
        print("（无）")

    # 全图 vs 采样：每个 (dataset, model) 的差值
    print("\n" + "=" * 90)
    print("表3：全图训练 vs 采样训练 对比（测试准确率差 = sample - full）")
    print("=" * 90)
    pivot = df.pivot_table(
        index=["dataset", "model"], columns="mode",
        values=["test_acc", "avg_epoch_sec", "train_sec"], aggfunc="mean")
    for ds in sorted(df["dataset"].unique()):
        print(f"\n--- {ds} ---")
        sub = pivot.loc[ds] if ds in pivot.index.get_level_values(0) else None
        if sub is None:
            continue
        with pd.option_context("display.width", 120):
            print(sub.to_string())
        # 手动打印差值
        for model in sorted(df[df["dataset"] == ds]["model"].unique()):
            try:
                full_acc = pivot.loc[(ds, model), ("test_acc", "full")]
                samp_acc = pivot.loc[(ds, model), ("test_acc", "sample")]
                full_t = pivot.loc[(ds, model), ("avg_epoch_sec", "full")]
                samp_t = pivot.loc[(ds, model), ("avg_epoch_sec", "sample")]
                print(f"  {model:6s}: acc full={fmt(full_acc)} "
                      f"sample={fmt(samp_acc)} "
                      f"(Δ={fmt(samp_acc - full_acc)}) | "
                      f"epoch_sec full={fmt(full_t)} sample={fmt(samp_t)}")
            except KeyError:
                continue


if __name__ == "__main__":
    main()
