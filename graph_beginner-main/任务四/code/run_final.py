# -*- coding: utf-8 -*-
"""任务四调参后重跑：用更低 lr + 更大评测集，得到更有代表性的 KGE 结果。

用法: python run_final.py [--phase all|transe|rotate|conve]
结果写入 results/results.csv（追加）。
"""
import argparse
import datetime
import os
import subprocess
import sys

CODE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(CODE, "results", "results.csv")
PY = sys.executable

# (标签, 模型, dim, lr, epochs, batch, eval-n)
PLAN = {
    "transe": [
        ("transe dim100 lr1e-4 ep15", "transe", 100, 1e-4, 15, 1024, 2000),
        ("transe dim200 lr1e-4 ep15", "transe", 200, 1e-4, 15, 1024, 2000),
        ("transe dim100 lr3e-4 ep15", "transe", 100, 3e-4, 15, 1024, 2000),
    ],
    "rotate": [
        # RotatE 的关系相位是可学习参数（见 models.py 修复），需要比 TransE 更大的步长
        ("rotate dim100 lr1e-3 ep15", "rotate", 100, 1e-3, 15, 1024, 2000),
        ("rotate dim200 lr1e-3 ep15", "rotate", 200, 1e-3, 15, 512, 2000),
    ],
    "conve": [
        ("conve dim100 lr1e-4 ep15", "conve", 100, 1e-4, 15, 256, 2000),
        ("conve dim100 lr3e-4 ep15", "conve", 100, 3e-4, 15, 256, 2000),
    ],
}


def run(label, model, dim, lr, epochs, batch, eval_n):
    cmd = [PY, os.path.join(CODE, "train.py"),
           "--model", model, "--dim", str(dim), "--lr", str(lr),
           "--epochs", str(epochs), "--batch-size", str(batch),
           "--eval-n", str(eval_n), "--eval-every", "1",
           "--results", RESULTS]
    print(f"\n[{datetime.datetime.now():%H:%M:%S}] {label}", flush=True)
    print("  " + " ".join(cmd[1:]), flush=True)
    ret = subprocess.call(cmd)
    if ret != 0:
        print(f"  [warn] 返回码 {ret}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all",
                    choices=["all", "transe", "rotate", "conve"])
    args = ap.parse_args()
    phases = list(PLAN) if args.phase == "all" else [args.phase]
    for ph in phases:
        for item in PLAN[ph]:
            run(*item)
    print(f"\n[all-done] {datetime.datetime.now():%H:%M:%S}")


if __name__ == "__main__":
    main()
