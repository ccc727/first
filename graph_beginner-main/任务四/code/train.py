# -*- coding: utf-8 -*-
"""
train.py —— 任务四：知识图谱补全训练与评估（TransE / RotatE / ConvE）

过滤式评估：固定 (h,r) 破环尾实体（1-to-N），报告 MRR 与 Hits@1/3/10。
用法：
    python train.py --model transe --epochs 60
    python train.py --model conve --epochs 30
"""
import argparse
import csv
import datetime
import os
import time

import torch
import torch.nn.functional as F

from data import load_kg
from models import build_kge, count_parameters


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="FB15k-237")
    p.add_argument("--model", type=str, required=True,
                   choices=["transe", "rotate", "conve"])
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--dim", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--eval-n", type=int, default=1000,
                   help="评测三元组上限（加快 CPU 评测）")
    p.add_argument("--max-train", type=int, default=0,
                   help=">0 时每轮只随机取前 N 条训练三元组（加速）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--data-dir", type=str,
                   default=os.path.abspath(os.path.join(
                       os.path.dirname(os.path.abspath(__file__)),
                       "..", "data")))
    p.add_argument("--results", type=str,
                   default=os.path.abspath(os.path.join(
                       os.path.dirname(os.path.abspath(__file__)),
                       "results", "results.csv")))
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


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


def build_filter(data):
    """(h,r) -> 所有合法尾实体集合（train+valid+test 并集，供过滤）。"""
    f = {}
    for arr in (data.train, data.valid, data.test):
        for h, r, t in arr.tolist():
            f.setdefault((h, r), set()).add(t)
    return f


@torch.no_grad()
def filtered_tail_metrics(model, triples, idx, filt, max_n, device):
    """尾实体过滤式排名指标（MRR / Hits@1/3/10）。

    CPU 上逐个候选实体写 score 向量非常慢（FB15k-237 有 14541 个实体），
    这里改为**按批向量化**：把批内所有 (h,r) 的过滤实体收集成扁平索引后，
    一次性 index_fill 屏蔽，排名也整批算。语义与逐条循环一致：被屏蔽的
    其它合法尾实体置 -1e9、不参与排名，自身真值 t 永不被屏蔽。
    """
    model.eval()
    idx = idx[torch.randperm(len(idx))[:max_n]]
    n = len(idx)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0

    mrr = h10 = h3 = h1 = 0.0
    for i in range(0, n, 128):
        batch = triples[idx[i:i + 128]]
        h = batch[:, 0].to(device)
        r = batch[:, 1].to(device)
        t = batch[:, 2].to(device)
        bs = h.shape[0]
        scores = model.tail_scores(h, r)                 # (B, E)
        true = scores.gather(1, t.unsqueeze(1)).squeeze(1)

        rows, cols = [], []
        for j in range(bs):
            cands = filt.get((int(h[j]), int(r[j])))
            if not cands:
                continue
            tj = int(t[j])
            for c in cands:
                if c != tj:                              # 真值本身不屏蔽
                    rows.append(j)
                    cols.append(c)
        if rows:
            rows_t = torch.as_tensor(rows, dtype=torch.long, device=scores.device)
            cols_t = torch.as_tensor(cols, dtype=torch.long, device=scores.device)
            scores[rows_t, cols_t] = -1e9

        rank = (scores > true.unsqueeze(1)).sum(dim=1) + 1     # (B,)
        mrr += (1.0 / rank.float()).sum().item()
        h1 += (rank <= 1).sum().item()
        h3 += (rank <= 3).sum().item()
        h10 += (rank <= 10).sum().item()
    return mrr / n, h10 / n, h3 / n, h1 / n


def main():
    args = parse()
    args.device = device_of(args.device)
    set_seed(args.seed)
    print(f"[task4] dataset={args.dataset} model={args.model} "
          f"device={args.device} dim={args.dim}")

    data = load_kg(args.dataset, args.data_dir, seed=args.seed)
    print("[data]", data)
    if args.max_train > 0 and args.max_train < len(data.train):
        data.train = data.train[torch.randperm(len(data.train))[:args.max_train]]
        print(f"[data] 每轮训练三元组截断为 {len(data.train)}")

    model = build_kge(args.model, data.num_entities, data.num_relations,
                      args.dim).to(args.device)
    print(f"[model] {args.model} params={count_parameters(model)}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    filt = build_filter(data)
    best, best_ep, bad = -1.0, 0, 0
    ep_times, ev_times = [], []
    best_path = os.path.join(os.path.dirname(args.results),
                             f"best_{args.model}.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        perm = torch.randperm(len(data.train))
        for i in range(0, len(perm), args.batch_size):
            eids = perm[i:i + args.batch_size]
            h, r, t = data.train[eids].T.to(args.device)
            neg_t = torch.randint(0, data.num_entities, h.shape,
                                  device=args.device)
            s_pos = model.score_batch(h, r, t)
            s_neg = model.score_batch(h, r, neg_t)
            y = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)])
            loss = F.binary_cross_entropy_with_logits(
                torch.cat([s_pos, s_neg]), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        ep_times.append(time.time() - t0)

        if epoch % args.eval_every == 0:
            e0 = time.time()
            mrr, h10, _, _ = filtered_tail_metrics(
                model, data.valid, torch.arange(len(data.valid)),
                filt, args.eval_n, args.device)
            ev_times.append(time.time() - e0)
            if mrr > best:
                best, best_ep, bad = mrr, epoch, 0
                torch.save(model.state_dict(), best_path)
            else:
                bad += 1
            if args.verbose:
                print(f"  epoch {epoch:4d} val MRR={mrr:.4f} H@10={h10:.4f}",
                      flush=True)
            if bad >= args.patience:
                print(f"[early-stop] epoch {epoch}")
                break

    model.load_state_dict(torch.load(best_path))
    t_mrr, t_h10, t_h3, t_h1 = filtered_tail_metrics(
        model, data.test, torch.arange(len(data.test)), filt, args.eval_n,
        args.device)

    row = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
           "dataset": args.dataset, "model": args.model,
           "dim": args.dim, "batch_size": args.batch_size,
           "seed": args.seed, "n_params": count_parameters(model),
           "val_mrr": round(best, 4), "best_epoch": best_ep,
           "test_mrr": round(t_mrr, 4), "hits@10": round(t_h10, 4),
           "hits@3": round(t_h3, 4), "hits@1": round(t_h1, 4),
           "train_epochs": len(ep_times),
           "avg_epoch_sec": round(sum(ep_times) / max(len(ep_times), 1), 3),
           "train_sec": round(sum(ep_times), 3),
           "eval_sec": round(sum(ev_times), 3), "device": args.device}
    os.makedirs(os.path.dirname(os.path.abspath(args.results)), exist_ok=True)
    new = not os.path.exists(args.results)
    with open(args.results, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)

    print("-" * 70)
    print(f"[done] valid MRR={best:.4f} | test MRR={t_mrr:.4f} "
          f"Hits@1={t_h1:.4f} Hits@3={t_h3:.4f} Hits@10={t_h10:.4f}")


if __name__ == "__main__":
    main()

