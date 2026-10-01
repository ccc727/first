# -*- coding: utf-8 -*-
"""run_experiments.py —— 任务二批量实验（数据集×模型×模式）。"""
import argparse
import itertools
import os
import subprocess
import sys

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SCRIPT = os.path.join(CODE_DIR, "train.py")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, default="cora,citeseer,flickr")
    p.add_argument("--models", type=str, default="gcn,gat,sage,gin")
    p.add_argument("--modes", type=str, default="full,sample")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--fanouts", type=str, default="10,10")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--results", type=str, default=None)
    p.add_argument("--sweep-lr", type=str, default="")
    p.add_argument("--sweep-layers", type=str, default="")
    return p.parse_args()


def split_csv(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def build(args):
    data_dir = os.path.abspath(os.path.join(CODE_DIR, "..", "data"))
    res = args.results or os.path.join(CODE_DIR, "results", "results.csv")
    lrs = [float(x) for x in split_csv(args.sweep_lr)] or [args.lr]
    layers = [int(x) for x in split_csv(args.sweep_layers)] or [args.layers]
    cmds = []
    for ds, model, mode, lr, nl in itertools.product(
            split_csv(args.datasets), split_csv(args.models),
            split_csv(args.modes), lrs, layers):
        cmd = [SCRIPT, "--dataset", ds, "--model", model, "--mode", mode,
               "--lr", str(lr), "--layers", str(nl), "--hidden",
               str(args.hidden), "--seed", str(args.seed),
               "--device", args.device, "--data-dir", data_dir,
               "--results", res]
        if args.epochs is not None:
            cmd += ["--epochs", str(args.epochs)]
        if mode == "sample":
            cmd += ["--batch-size", str(args.batch_size),
                    "--fanouts", args.fanouts]
        cmds.append(cmd)
    return cmds


def main():
    args = parse()
    cmds = build(args)
    print(f"[plan] {len(cmds)} 个实验")
    for i, c in enumerate(cmds, 1):
        print(f"[{i}/{len(cmds)}] {os.path.basename(c[0])} "
              + " ".join(c[2:6]))
        sys.stdout.flush()
        if subprocess.call([PY] + c) != 0:
            print("  !! 运行失败")
        sys.stdout.flush()
    print("[all-done]")


if __name__ == "__main__":
    main()
