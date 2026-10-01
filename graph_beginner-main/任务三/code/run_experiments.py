# -*- coding: utf-8 -*-
"""run_experiments.py —— 任务三批量实验。"""
import argparse
import itertools
import os
import subprocess
import sys

CODE = os.path.dirname(os.path.abspath(__file__))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", type=str, default="mutag,zinc")
    p.add_argument("--models", type=str, default="gcn,gat,sage,gin")
    p.add_argument("--pools", type=str, default="mean,max,min")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--limit-train", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    args = parse()
    for ds, model, pool in itertools.product(
            args.datasets.split(","), args.models.split(","),
            args.pools.split(",")):
        cmd = [sys.executable, os.path.join(CODE, "train.py"),
               "--dataset", ds.strip(), "--model", model.strip(),
               "--pool", pool.strip(), "--epochs", str(args.epochs),
               "--seed", str(args.seed), "--device", args.device]
        if args.limit_train:
            cmd += ["--limit-train", str(args.limit_train)]
        print("run:", " ".join(c[2:6]))
        sys.stdout.flush()
        if subprocess.call(cmd) != 0:
            print(" !! fail")
        sys.stdout.flush()
    print("[all-done]")


if __name__ == "__main__":
    main()
