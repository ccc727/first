# -*- coding: utf-8 -*-
"""Make small data subsets for task6: QA (DuReaderQG) & filling model."""
import os
import random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "_smoke", "data")
os.makedirs(DATA, exist_ok=True)
random.seed(42)


def subsample(src, dst, n):
    with open(src, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    random.shuffle(lines)
    lines = lines[:n]
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    QG = os.path.join(ROOT, "answer_generation", "data", "DuReaderQG")
    subsample(os.path.join(QG, "train.json"), os.path.join(DATA, "qg_train.json"), 300)
    subsample(os.path.join(QG, "dev.json"), os.path.join(DATA, "qg_dev.json"), 80)

    FM = os.path.join(ROOT, "data_augment", "filling_model", "data")
    subsample(os.path.join(FM, "train.tsv"), os.path.join(DATA, "fm_train.tsv"), 400)
    subsample(os.path.join(FM, "dev.tsv"), os.path.join(DATA, "fm_dev.tsv"), 80)
    print("task6 data ready")


if __name__ == "__main__":
    main()
