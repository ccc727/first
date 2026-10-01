# -*- coding: utf-8 -*-
"""run_experiments.py —— 任务四批量跑三个模型。"""
import os
import subprocess
import sys

CODE = os.path.dirname(os.path.abspath(__file__))


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=str, default="transe,rotate,conve")
    p.add_argument("--epochs-transe", type=int, default=40)
    p.add_argument("--epochs-rotate", type=int, default=40)
    p.add_argument("--epochs-conve", type=int, default=20)
    p.add_argument("--eval-n", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    epochs = {"transe": a.epochs_transe, "rotate": a.epochs_rotate,
              "conve": a.epochs_conve}
    for model in [m.strip() for m in a.models.split(",")]:
        cmd = [sys.executable, os.path.join(CODE, "train.py"),
               "--model", model, "--epochs", str(epochs[model]),
               "--eval-n", str(a.eval_n), "--seed", str(a.seed)]
        print("run:", " ".join(cmd))
        sys.stdout.flush()
        if subprocess.call(cmd) != 0:
            print(" !! fail")
        sys.stdout.flush()
    print("[all-done]")


if __name__ == "__main__":
    main()
