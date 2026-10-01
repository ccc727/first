# -*- coding: utf-8 -*-
"""
train_sampling.py —— 基于 DGL NeighborSampler 的子图（邻居采样）节点分类训练

每个 epoch 按 batch 采样节点的 L 跳邻居子图（block），只在该子图上做前向/反向，
支持 GCN / GAT / GraphSAGE / GIN 与 cora / citeseer / flickr。

每个 epoch 结束后仍在整张图上做一次验证/测试评估（与全图训练公平对比）。

示例:
    python train_sampling.py --dataset cora --model gcn
    python train_sampling.py --dataset flickr --model sage \
        --batch-size 1000 --fanouts 15,10
"""
import copy
import datetime
import time

import torch
import torch.nn as nn

from dgl.dataloading import DataLoader, NeighborSampler

from data import load_dataset
from utils import (append_result, build_and_log_model, epoch_time,
                   evaluate_full, make_parser, resolve_device, set_seed)


def parse_fanouts(fanouts_str: str, n_layers: int):
    """'10,25' -> [10, 25]。若只给一个数则每层复用。"""
    vals = [int(x) for x in str(fanouts_str).split(",") if x.strip()]
    if len(vals) == 1:
        vals = vals * n_layers
    if len(vals) != n_layers:
        raise ValueError(
            f"fanouts 长度({len(vals)})必须等于网络层数({n_layers})，"
            f"当前 --fanouts={fanouts_str}"
        )
    return vals


def train(args):
    # ----------------------------- 初始化 ---------------------------------
    args.device = resolve_device(args.device)
    args.epochs = args.epochs or (300 if args.dataset in ("cora", "citeseer") else 60)
    args.patience = args.patience or (100 if args.dataset in ("cora", "citeseer") else 20)
    set_seed(args.seed)

    print(f"[task1/sample] dataset={args.dataset} model={args.model} "
          f"device={args.device} seed={args.seed}")
    print(f"[cfg] lr={args.lr} wd={args.weight_decay} layers={args.layers} "
          f"hidden={args.hidden} dropout={args.dropout} epochs={args.epochs} "
          f"batch_size={args.batch_size} fanouts={args.fanouts}")

    # ----------------------------- 数据 -----------------------------------
    data = load_dataset(args.dataset, args.data_dir, seed=args.seed,
                        self_loop=not args.no_self_loop)
    print(f"[data] {data}")

    g = data.g
    feats = data.feats.to(args.device)
    labels = data.labels.to(args.device)
    train_idx = data.train_idx.to(args.device)
    val_idx = data.val_idx.to(args.device)
    test_idx = data.test_idx.to(args.device)

    fanouts = parse_fanouts(args.fanouts, args.layers)
    print(f"[sampler] 每层邻居采样数 fanouts={fanouts}")

    # ----------------------------- 模型 -----------------------------------
    model, n_params = build_and_log_model(args, data, args.device)
    print(f"[model] {args.model.upper()} 参数量={n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # --------------------------- 采样 DataLoader ---------------------------
    sampler = NeighborSampler(fanouts)
    train_loader = DataLoader(
        g, train_idx, sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        device=args.device,
    )

    # ----------------------------- 训练循环 --------------------------------
    best_val, best_epoch, bad_counter = -1.0, 0, 0
    best_state = None
    epoch_times, eval_times = [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        total_loss, total_nodes = 0.0, 0

        for step, (input_nodes, output_nodes, blocks) in enumerate(train_loader):
            batch_feats = feats[input_nodes]
            batch_labels = labels[output_nodes]
            logits = model.forward_blocks(blocks, batch_feats)
            loss = loss_fn(logits, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_labels)
            total_nodes += len(batch_labels)

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
                mean_loss = total_loss / max(total_nodes, 1)
                print(f"  epoch {epoch:4d} loss={mean_loss:.4f} "
                      f"val_acc={val_acc:.4f} "
                      f"({time.time()-t0:.2f}s, {step+1} batches)", flush=True)

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
        "dataset": args.dataset, "model": args.model, "mode": "sample",
        "seed": args.seed, "lr": args.lr, "layers": args.layers,
        "hidden": args.hidden, "dropout": args.dropout,
        "fanouts": ",".join(str(x) for x in fanouts),
        "batch_size": args.batch_size, "n_params": n_params,
        "best_val_acc": round(best_val, 4), "best_epoch": best_epoch,
        "test_acc": round(test_acc, 4), "train_epochs": len(epoch_times),
        "avg_epoch_sec": round(avg_sec, 3), "train_sec": round(total_sec, 3),
        "eval_sec": round(float(sum(eval_times)), 3),
        "device": args.device, "note": f"neighbor-sampling fanouts={fanouts}",
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
    parser = make_parser("任务一：节点分类 —— 邻居采样子图训练", sampling=True)
    args = parser.parse_args()
    train(args)

