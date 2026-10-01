# -*- coding: utf-8 -*-
"""Make small train/dev subsets used by the smoke runs."""
import os
import random
import shutil

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
    TM = os.path.join(ROOT, "text_matching", "supervised", "data", "comment_classify")
    subsample(os.path.join(TM, "train.txt"), os.path.join(DATA, "tm_train.txt"), 400)
    subsample(os.path.join(TM, "dev.txt"), os.path.join(DATA, "tm_dev.txt"), 150)
    shutil.copy(os.path.join(TM, "types_desc.txt"), os.path.join(DATA, "tm_types_desc.txt"))

    TC = os.path.join(ROOT, "text_classification", "data", "comment_classify")
    subsample(os.path.join(TC, "train.txt"), os.path.join(DATA, "tc_train.txt"), 402)
    subsample(os.path.join(TC, "dev.txt"), os.path.join(DATA, "tc_dev.txt"), 63)
    shutil.copy(os.path.join(TC, "label_mapping.txt"), os.path.join(DATA, "tc_label_mapping.txt"))

    for task in ("PET", "p-tuning"):
        src = os.path.join(ROOT, "prompt_tasks", task, "data", "comment_classify")
        out = os.path.join(DATA, task.lower())
        os.makedirs(out, exist_ok=True)
        subsample(os.path.join(src, "train.txt"), os.path.join(out, "train.txt"), 63)
        subsample(os.path.join(src, "dev.txt"), os.path.join(out, "dev.txt"), 120)
        shutil.copy(os.path.join(src, "verbalizer.txt"), os.path.join(out, "verbalizer.txt"))
        if task == "PET":
            shutil.copy(os.path.join(src, "prompt.txt"), os.path.join(out, "prompt.txt"))

    DU = os.path.join(ROOT, "UIE", "data", "DuIE")
    subsample(os.path.join(DU, "train.txt"), os.path.join(DATA, "uie_train.txt"), 300)
    subsample(os.path.join(DU, "dev.txt"), os.path.join(DATA, "uie_dev.txt"), 80)

    RD = os.path.join(ROOT, "RLHF", "data", "reward_datasets", "sentiment_analysis")
    subsample(os.path.join(RD, "train.tsv"), os.path.join(DATA, "rm_train.tsv"), 300)
    subsample(os.path.join(RD, "dev.tsv"), os.path.join(DATA, "rm_dev.tsv"), 80)
    print("data ready")


if __name__ == "__main__":
    main()
