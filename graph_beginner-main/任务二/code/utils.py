# -*- coding: utf-8 -*-
"""utils.py —— 任务二公共工具。"""
import argparse
import csv
import os
import random
import time

import numpy as np
import torch

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.abspath(os.path.join(CODE_DIR, "..", "data"))
DEFAULT_RESULTS = os.path.abspath(os.path.join(CODE_DIR, "results",
                                               "results.csv"))


def make_parser():
    p = argparse.ArgumentParser(description="任务二：链路预测")
    p.add_argument("--dataset", type=str, required=True,
                   choices=["cora", "citeseer", "flickr"])
    p.add_argument("--model", type=str, required=True,
                   choices=["gcn", "gat", "sage", "gin"])
    p.add_argument("--mode", type=str, default="full",
                   choices=["full", "sample"], help="full=全图训练, sample=采样训练")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=0.005)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=1024,
                   help="全图模式=每条正边批次大小；采样模式=每批种子边数")
    p.add_argument("--fanouts", type=str, default="10,10")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA)
    p.add_argument("--results", type=str, default=DEFAULT_RESULTS)
    p.add_argument("--verbose", action="store_true")
    return p


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name):
    if name and name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_fanouts(s, layers):
    vals = [int(x) for x in str(s).split(",") if x.strip()]
    if len(vals) == 1:
        vals = vals * layers
    assert len(vals) == layers, f"fanouts {s} 长度须等于层数 {layers}"
    return vals


def defaults(args):
    if args.epochs is None:
        args.epochs = 300 if args.dataset in ("cora", "citeseer") else 50
    if args.patience is None:
        args.patience = 100 if args.dataset in ("cora", "citeseer") else 20
    return args


@torch.no_grad()
def evaluate_auc(encoder, predictor, train_g, feats, pos, neg, device):
    """整图编码后，计算正/负样本打分结果的 ROC-AUC。"""
    from sklearn.metrics import roc_auc_score
    encoder.eval()
    emb = encoder.encode_graph(train_g.to(device), feats.to(device))
    pos_s, pos_d = pos[0].to(device), pos[1].to(device)
    neg_s, neg_d = neg[0].to(device), neg[1].to(device)
    p_score = predictor.forward_pair(emb[pos_s], emb[pos_d])
    n_score = predictor.forward_pair(emb[neg_s], emb[neg_d])
    y = np.concatenate([np.ones(len(p_score)), np.zeros(len(n_score))])
    x = torch.cat([p_score, n_score]).cpu().numpy()
    return float(roc_auc_score(y, x))



def append_result(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = ["time", "dataset", "model", "mode", "seed", "lr", "layers",
              "hidden", "fanouts", "batch_size", "n_params", "val_auc",
              "best_epoch", "test_auc", "train_epochs", "avg_epoch_sec",
              "train_sec", "eval_sec", "device"]
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
