# -*- coding: utf-8 -*-
"""
run_experiments.py —— 批量跑实验

对 datatsets × models × modes 全组合调用 train_full.py / train_sampling.py，
并可对学习率 / 层数做网格扫描（用于报告中的参数分析）。所有结果统一写入
results.csv。

示例:
    # 基本组合（推荐先用小数据集验证）
    python run_experiments.py --datasets cora,citeseer \
        --models gcn,gat,sage,gin --modes full,sample

    # 只跑部分模型/模式
    python run_experiments.py --datasets cora --models gcn,gin \
        --modes full --epochs 200

    # 扫学习率 + 层数（cora 上，GCN）
    python run_experiments.py --datasets cora --models gcn --modes full \
        --sweep-lr 0.001,0.005,0.01,0.05 --sweep-layers 2,3

    # 全数据集大实验（比较耗时，建议有 GPU 或耐心等待）
    python run_experiments.py --datasets cora,citeseer,flickr \
        --models gcn,gat,sage,gin --modes full,sample
"""
import argparse
import itertools
import os
import subprocess
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable

SCRIPT = {
    "full": os.path.join(CODE_DIR, "train_full.py"),
    "sample": os.path.join(CODE_DIR, "train_sampling.py"),
}


def parse_args():
    p = argparse.ArgumentParser(description="批量运行任务一实验")
    p.add_argument("--datasets", type=str, default="cora,citeseer,flickr")
    p.add_argument("--models", type=str, default="gcn,gat,sage,gin")
    p.add_argument("--modes", type=str, default="full,sample")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--fanouts", type=str, default="10,10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--data-dir", type=str, default=None,
                   help="默认 任务一/data")
    p.add_argument("--results", type=str, default=None,
                   help="默认 任务一/code/results/results.csv")
    # 网格扫描（可同时给出；与上面固定参数做笛卡尔积）
    p.add_argument("--sweep-lr", type=str, default="",
                   help="例如 0.001,0.005,0.01")
    p.add_argument("--sweep-layers", type=str, default="",
                   help="例如 2,3")
    return p.parse_args()


def split_csv(s: str):
    return [x.strip() for x in s.split(",") if x.strip()]


def build_commands(args):
    """返回 (模式, 命令行参数列表) 的列表。"""
    datasets = split_csv(args.datasets)
    models = split_csv(args.models)
    modes = split_csv(args.modes)
    lrs = [float(x) for x in split_csv(args.sweep_lr)] or [args.lr]
    layers_list = [int(x) for x in split_csv(args.sweep_layers)] or [args.layers]

    data_dir = args.data_dir or os.path.abspath(
        os.path.join(CODE_DIR, "..", "data"))
    results = args.results or os.path.join(CODE_DIR, "results", "results.csv")

    cmds = []
    for dataset, model, mode, lr, layers in itertools.product(
            datasets, models, modes, lrs, layers_list):
        cmd = [SCRIPT[mode],
               "--dataset", dataset,
               "--model", model,
               "--lr", str(lr),
               "--layers", str(layers),
               "--hidden", str(args.hidden),
               "--dropout", str(args.dropout),
               "--seed", str(args.seed),
               "--device", args.device,
               "--data-dir", data_dir,
               "--results", results]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if mode == "sample":
            cmd += ["--batch-size", str(args.batch_size),
                    "--fanouts", args.fanouts]
        cmds.append(cmd)
    return cmds


def main():
    args = parse_args()
    cmds = build_commands(args)
    print(f"[plan] 共 {len(cmds)} 个实验: ")
    for c in cmds:
        print("   ", "python " + " ".join(c))
    print("-" * 80)

    for idx, cmd in enumerate(cmds, 1):
        tag = f"[{idx}/{len(cmds)}]"
        print(f"\n{tag} 开始运行: {os.path.basename(cmd[0])} "
              f"dataset={cmd[cmd.index('--dataset') + 1]} "
              f"model={cmd[cmd.index('--model') + 1]}")
        sys.stdout.flush()
        ret = subprocess.call([PYTHON] + cmd)
        if ret != 0:
            print(f"{tag} 运行失败，返回码 {ret}，继续下一个实验。")
        sys.stdout.flush()

    print("\n[all-done] 所有实验结束，结果见 "
          + (args.results or os.path.join(CODE_DIR, "results", "results.csv")))


if __name__ == "__main__":
    main()
