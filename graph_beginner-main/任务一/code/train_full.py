# -*- coding: utf-8 -*-
"""
train_full.py —— 全图（整批）节点分类训练与测试

对整张图做一次前向/反向（Full-batch Training），
支持 GCN / GAT / GraphSAGE / GIN 四种模型与 cora / citeseer / flickr 三个数据集。

示例:
    python train_full.py --dataset cora --model gcn
    python train_full.py --dataset citeseer --model gat --lr 0.005
    python train_full.py --dataset flickr --model sage --epochs 30
"""
import copy
import datetime
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import load_dataset
from utils import (append_result, build_and_log_model, epoch_time,
                   evaluate_full, make_parser, resolve_device, set_seed)


def train(args):
    # ----------------------------- 初始化 ---------------------------------
    args.device = resolve_device(args.device)
    args.epochs = args.epochs or (300 if args.dataset in ("cora", "citeseer") else 60)
    args.patience = args.patience or (100 if args.dataset in ("cora", "citeseer") else 20)
    set_seed(args.seed)

    print(f"[task1/full] dataset={args.dataset} model={args.model} "
          f"device={args.device} seed={args.seed}")
    print(f"[cfg] lr={args.lr} wd={args.weight_decay} layers={args.layers} "
          f"hidden={args.hidden} dropout={args.dropout} epochs={args.epochs}")

    # ----------------------------- 数据 -----------------------------------
    data = load_dataset(args.dataset, args.data_dir, seed=args.seed,
                        self_loop=not args.no_self_loop)
    print(f"[data] {data}")

    g = data.g.to(args.device)
    feats = data.feats.to(args.device)
    labels = data.labels.to(args.device)
    train_idx = data.train_idx.to(args.device)
    val_idx = data.val_idx.to(args.device)
    test_idx = data.test_idx.to(args.device)

    # ----------------------------- 模型 -----------------------------------
    model, n_params = build_and_log_model(args, data, args.device)
    print(f"[model] {args.model.upper()} 参数量={n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # ----------------------------- 训练循环 --------------------------------
    best_val, best_epoch, bad_counter = -1.0, 0, 0
    best_state = None
    epoch_times, eval_times = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        logits = model.forward_graph(g, feats)
        loss = loss_fn(logits[train_idx], labels[train_idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_times.append(time.time() - t0)

        if epoch % args.eval_every == 0:
            val_acc, e_t = evaluate_full(model, g, feats, labels,
                                         val_idx, args.device)
            eval_times.append(e_t)
            improved = val_acc > best_val
            if improved:
                best_val, best_epoch = val_acc, epoch
                best_state = copy.deepcopy(model.state_dict())
                bad_counter = 0
            else:
                bad_counter += 1

            if args.verbose:
                print(f"  epoch {epoch:4d} loss={loss.item():.4f} "
                      f"val_acc={val_acc:.4f} ({time.time()-t0:.2f}s)",
                      flush=True)

            if bad_counter >= args.patience:
                print(f"[early-stop] epoch {epoch}，验证集 {args.patience} "
                      f"轮未提升，停止训练。")
                break

    # --------------------------- 测试集评估 ---------------------------------
    if best_state is not None:
        model.load_state_dict(best_state)
    test_acc, test_t = evaluate_full(model, g, feats, labels, test_idx,
                                     args.device)
    avg_sec, total_sec = epoch_time(epoch_times)

    row = {
        "time": datetime.datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset, "model": args.model, "mode": "full",
        "seed": args.seed, "lr": args.lr, "layers": args.layers,
        "hidden": args.hidden, "dropout": args.dropout,
        "fanouts": "", "batch_size": "", "n_params": n_params,
        "best_val_acc": round(best_val, 4), "best_epoch": best_epoch,
        "test_acc": round(test_acc, 4), "train_epochs": len(epoch_times),
        "avg_epoch_sec": round(avg_sec, 3), "train_sec": round(total_sec, 3),
        "eval_sec": round(float(sum(eval_times)), 3),
        "device": args.device, "note": "full-batch",
    }
    append_result(args.results, row)

    print("-" * 80)
    print(f"[done] 训练 {len(epoch_times)} 轮，最优验证准确率 "
          f"{best_val:.4f} @ epoch {best_epoch}")
    print(f"[done] 测试集准确率 {test_acc:.4f}")
    print(f"[time] 平均每轮训练耗时 {avg_sec:.3f}s，训练总耗时 {total_sec:.3f}s")
    print(f"[time] 评估总耗时 {sum(eval_times):.3f}s")
    return test_acc


if __name__ == "__main__":
    parser = make_parser("任务一：节点分类 —— 全图训练")
    args = parser.parse_args()
    train(args)
