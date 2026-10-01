# -*- coding: utf-8 -*-
"""train.py —— 任务三：图分类(TUDataset) / 图回归(ZINC) 训练。"""
import argparse
import copy
import csv
import datetime
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import load_graph_dataset
from models import GNNEncoder, count_parameters


def make_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--model", type=str, required=True,
                   choices=["gcn", "gat", "sage", "gin"])
    p.add_argument("--pool", type=str, default="mean",
                   choices=["mean", "max", "min"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--limit-train", type=int, default=0,
                   help=">0 时只取前 N 个训练图（加速）")
    p.add_argument("--data-dir", type=str,
                   default=os.path.abspath(os.path.join(
                       os.path.dirname(os.path.abspath(__file__)),
                       "..", "data")))
    p.add_argument("--results", type=str,
                   default=os.path.abspath(os.path.join(
                       os.path.dirname(os.path.abspath(__file__)),
                       "results", "results.csv")))
    p.add_argument("--verbose", action="store_true")
    return p


def device_of(name):
    if name and name != "auto":
        return name
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, data, idx, opt, device, batch_size, train):
    model.train() if train else model.eval()
    idx = idx[torch.randperm(len(idx))] if train else idx
    loss_sum, score_sum = 0.0, 0.0
    cnt = 0
    with torch.set_grad_enabled(train):
        for i in range(0, len(idx), batch_size):
            bg, node_ids, y = data.batch(idx[i:i + batch_size].tolist())
            bg = bg.to(device)
            y = y.to(device)
            out = model(bg, node_ids)
            if data.task == "cls":
                loss = F.cross_entropy(out, y.long())
                score_sum += (out.argmax(1) == y.long()).sum().item()
            else:
                loss = F.l1_loss(out.squeeze(1), y)
                score_sum += (out.squeeze(1) - y).abs().sum().item()
            loss_sum += loss.item() * len(y)
            cnt += len(y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
    return loss_sum / max(cnt, 1), score_sum / max(cnt, 1)


def main():
    args = make_parser().parse_args()
    args.device = device_of(args.device)
    set_seed(args.seed)
    print(f"[task3] dataset={args.dataset} model={args.model} "
          f"pool={args.pool} device={args.device}")

    data = load_graph_dataset(args.dataset, args.data_dir, seed=args.seed,
                              limit_train=args.limit_train)
    print("[data]", data)
    n_cls = int(data.labels.max().item()) + 1 if data.task == "cls" else 1
    out_dim = n_cls if data.task == "cls" else 1

    model = GNNEncoder(args.model, args.hidden, out_dim, args.layers,
                       args.dropout, pool=args.pool, use_node_id=True,
                       n_node_types=data.n_node_types).to(args.device)
    print(f"[model] params={count_parameters(model)}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    best, best_ep, bad = None, 0, 0
    best_state = None
    ep_times, ev_times = [], []
    better = (lambda a, b: a < b) if data.task == "reg" \
        else (lambda a, b: a > b)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        run_epoch(model, data, data.idx_train, opt, args.device,
                  args.batch_size, True)
        ep_times.append(time.time() - t0)
        e0 = time.time()
        _, val_score = run_epoch(model, data, data.idx_val, None,
                                 args.device, args.batch_size, False)
        ev_times.append(time.time() - e0)
        if best is None or better(val_score, best):
            best, best_ep, bad = val_score, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
        if args.verbose:
            metric = "acc" if data.task == "cls" else "mae"
            print(f"  epoch {epoch:4d} val_{metric}={val_score:.4f}",
                  flush=True)
        if bad >= args.patience:
            break

    model.load_state_dict(best_state)
    _, test_score = run_epoch(model, data, data.idx_test, None,
                              args.device, args.batch_size, False)
    metric = "acc" if data.task == "cls" else "mae"
    if data.task == "reg":
        test_score = test_score * data.y_std   # 还原到原始尺度

    row = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
           "dataset": args.dataset, "model": args.model, "pool": args.pool,
           "seed": args.seed, "lr": args.lr, "layers": args.layers,
           "hidden": args.hidden, "n_params": count_parameters(model),
           "best_val": round(best, 4), "best_epoch": best_ep,
           "test_" + metric: round(test_score, 4),
           "train_epochs": len(ep_times),
           "avg_epoch_sec": round(sum(ep_times) / max(len(ep_times), 1), 3),
           "train_sec": round(sum(ep_times), 3), "device": args.device}
    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)
    new = not os.path.exists(args.results)
    with open(args.results, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)

    print("-" * 70)
    print(f"[done] best_val({metric})={best:.4f} @ epoch {best_ep} | "
          f"test_{metric}={test_score:.4f}")


if __name__ == "__main__":
    main()

