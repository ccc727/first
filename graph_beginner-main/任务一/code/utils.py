# -*- coding: utf-8 -*-
"""
utils.py

公共工具：命令行参数、随机种子、评估、结果记录等。
"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch

from models import build_model, count_parameters

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA_DIR = os.path.abspath(os.path.join(CODE_DIR, "..", "data"))
DEFAULT_RESULTS = os.path.abspath(
    os.path.join(CODE_DIR, "results", "results.csv"))


# ----------------------------------------------------------------------------
# 参数解析
# ----------------------------------------------------------------------------

def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["cora", "citeseer", "flickr"],
                        help="数据集")
    parser.add_argument("--model", type=str, required=True,
                        choices=["gcn", "gat", "sage", "gin"],
                        help="模型: gcn/gat/sage/gin")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--device", type=str, default="auto",
                        help="auto/cpu/cuda[:0]")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数（默认: cora/citeseer=300, flickr=60）")
    parser.add_argument("--lr", type=float, default=0.01, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--hidden", type=int, default=64, help="隐藏层维度")
    parser.add_argument("--layers", type=int, default=2, help="GNN 层数")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=None,
                        help="早停耐心值（默认: cora/citeseer=100, flickr=20）")
    parser.add_argument("--eval-every", type=int, default=1,
                        help="每 N 个 epoch 在验证集上评估一次")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="数据集存放根目录")
    parser.add_argument("--results", type=str, default=DEFAULT_RESULTS,
                        help="结果 CSV 路径")
    parser.add_argument("--no-self-loop", action="store_true",
                        help="不加自环（默认会为每个节点加一条自环）")
    parser.add_argument("--verbose", action="store_true",
                        help="逐 epoch 打印日志")
    # GAT / SAGE / GIN 专属参数
    parser.add_argument("--num-heads", type=int, default=8,
                        help="GAT 注意力头数")
    parser.add_argument("--attn-dropout", type=float, default=0.0)
    parser.add_argument("--sage-aggregator", type=str, default="mean",
                        choices=["mean", "gcn", "pool"],
                        help="GraphSAGE 聚合方式")
    parser.add_argument("--gin-aggregator", type=str, default="sum",
                        choices=["sum", "mean", "max"],
                        help="GIN 聚合方式")
    return parser

def make_parser(desc: str, sampling: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=desc)
    add_common_args(parser)
    if sampling:
        parser.add_argument("--batch-size", type=int, default=512,
                            help="每个 mini-batch 的种子节点数")
        parser.add_argument("--fanouts", type=str, default="10,10",
                            help="每层采样邻居数，逗号分隔，个数应等于层数")
        parser.add_argument("--num-workers", type=int, default=0)
    return parser


def resolve_device(name: str) -> str:
    if name and name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


# ----------------------------------------------------------------------------
# 随机种子 / 评估 / 计时
# ----------------------------------------------------------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_full(model, g, feats, labels, idx, device):
    """整图前向，返回 (accuracy, 耗时)。"""
    model.eval()
    t0 = time.time()
    g = g.to(device)
    logits = model.forward_graph(g, feats.to(device))
    pred = logits.argmax(dim=1)
    acc = (pred[idx] == labels[idx]).float().mean().item()
    return acc, time.time() - t0


def epoch_time(seconds_list):
    """由逐 epoch 秒数列表返回 (平均秒, 总秒)。"""
    if not seconds_list:
        return 0.0, 0.0
    return float(np.mean(seconds_list)), float(np.sum(seconds_list))


# ----------------------------------------------------------------------------
# 结果记录
# ----------------------------------------------------------------------------

RESULT_FIELDS = [
    "time", "dataset", "model", "mode", "seed", "lr", "layers", "hidden",
    "dropout", "fanouts", "batch_size", "n_params", "best_val_acc",
    "best_epoch", "test_acc", "train_epochs", "avg_epoch_sec",
    "train_sec", "eval_sec", "device", "note",
]


def append_result(csv_path: str, row: dict):
    """把一行结果追加到 CSV；首次运行时自动创建表头。"""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS,
                                extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def build_and_log_model(args, data, device):
    """构造模型并返回 (model, n_params)。"""
    model_kwargs = {}
    if args.model == "gat":
        model_kwargs = dict(num_heads=args.num_heads,
                            attn_dropout=args.attn_dropout)
    elif args.model == "sage":
        model_kwargs = dict(aggregator_type=args.sage_aggregator)
    elif args.model == "gin":
        model_kwargs = dict(aggregator_type=args.gin_aggregator)
    model = build_model(args.model, data.feats.shape[1], args.hidden,
                        data.num_classes, args.layers, args.dropout,
                        **model_kwargs)
    n_params = count_parameters(model)
    return model.to(device), n_params

