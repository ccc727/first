# -*- coding: utf-8 -*-
"""
train.py —— 任务二：链路预测训练（全图 / NeighborSampler 子图两种方式）

全图模式：每个 epoch 在训练子图上整图编码一次，训练正边分小批，
          为每条正边随机采样 1 条负边（随机换尾），用 BCE 损失更新。
采样模式：每个 epoch 取一批训练正边，把边两端节点作为种子，
          用 DGL NeighborSampler 采 2 跳子图（block），得到局部嵌入后，
          用“批内打乱尾部”构造负样本计算损失。
评估：两种模式统一使用“训练子图整图编码 + 验证集 AUC”选模型，再报测试 AUC。

用法:
    python train.py --dataset cora --model gcn --mode full
    python train.py --dataset flickr --model sage --mode sample --batch-size 512
"""
import copy
import datetime
import time

import torch
import torch.nn as nn

from dgl.dataloading import DataLoader as DGLDataLoader
from dgl.dataloading import NeighborSampler

from data import load_lp_data
from models import LinkPredictor, build_encoder, count_parameters
from utils import (append_result, defaults, evaluate_auc, make_parser,
                   parse_fanouts, resolve_device, set_seed)


def _build_model(args, data, device):
    kwargs = {}
    if args.model == "gat":
        kwargs = dict(num_heads=4)
    elif args.model == "sage":
        kwargs = dict(aggregator_type="mean")
    elif args.model == "gin":
        kwargs = dict(aggregator_type="sum")
    enc = build_encoder(args.model, data.feats.shape[1], args.hidden,
                        args.layers, args.dropout, **kwargs)
    dec = LinkPredictor(args.hidden)
    n_params = count_parameters(enc) + count_parameters(dec)
    return enc.to(device), dec.to(device), n_params


def _bce_loss(dec, emb_h, emb_t, emb_n):
    pos = dec.forward_pair(emb_h, emb_t)
    neg = dec.forward_pair(emb_h, emb_n)
    y = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
    return nn.functional.binary_cross_entropy_with_logits(
        torch.cat([pos, neg]), y)


def train_full(args, data, enc, dec, device):
    g = data.train_g.to(device)
    feats = data.feats.to(device)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)
    best, best_ep, bad = -1.0, 0, 0
    best_state = None
    ep_times, ev_times = [], []

    for epoch in range(1, args.epochs + 1):
        enc.train()
        dec.train()
        t0 = time.time()
        emb = enc.encode_graph(g, feats)          # 整图编码 = 全图训练
        perm = torch.randperm(len(data.train_eids))
        opt.zero_grad()
        total_loss = torch.zeros((), device=device)
        for i in range(0, len(perm), args.batch_size):
            eids = data.train_eids[perm[i:i + args.batch_size]]
            s = data.train_src[eids].to(device)
            d = data.train_dst[eids].to(device)
            neg_d = torch.randint(0, data.num_nodes, s.shape, device=device)
            pos_l = dec.forward_pair(emb[s], emb[d])
            neg_l = dec.forward_pair(emb[s], emb[neg_d])
            y = torch.cat([torch.ones_like(pos_l), torch.zeros_like(neg_l)])
            loss = nn.functional.binary_cross_entropy_with_logits(
                torch.cat([pos_l, neg_l]), y)
            total_loss = total_loss + loss
        total_loss.backward()
        opt.step()
        ep_times.append(time.time() - t0)

        if epoch % args.eval_every == 0:
            a_t = time.time()
            val_auc = evaluate_auc(enc, dec, g, feats, data.val_pos,
                                   data.val_neg, device)
            ev_times.append(time.time() - a_t)
            if val_auc > best:
                best, best_ep, bad = val_auc, epoch, 0
                best_state = (copy.deepcopy(enc.state_dict()),
                              copy.deepcopy(dec.state_dict()))
            else:
                bad += 1
            if args.verbose:
                print(f"  epoch {epoch:4d} val_auc={val_auc:.4f}", flush=True)
            if bad >= args.patience:
                print(f"[early-stop] epoch {epoch}")
                break
    return best, best_ep, best_state, ep_times, ev_times


def _gather(emb, nodes, sorted_nodes):
    """把 output_nodes 的嵌入按全局节点 id 取回。"""
    order = torch.argsort(sorted_nodes)
    srt = sorted_nodes[order]
    pos = torch.searchsorted(srt, nodes)
    return emb[order][pos]


def train_sample(args, data, enc, dec, device):
    g = data.train_g
    feats = data.feats.to(device)
    fanouts = parse_fanouts(args.fanouts, args.layers)
    sampler = NeighborSampler(fanouts)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()),
                           lr=args.lr, weight_decay=args.weight_decay)
    best, best_ep, bad = -1.0, 0, 0
    best_state = None
    ep_times, ev_times = [], []
    n_train = len(data.train_eids)

    for epoch in range(1, args.epochs + 1):
        enc.train()
        dec.train()
        t0 = time.time()
        n_batches = max(1, n_train // args.batch_size)
        for _ in range(n_batches):
            eids = data.train_eids[torch.randint(0, n_train,
                                                 (args.batch_size,))]
            heads = data.train_src[eids].to(device)
            tails = data.train_dst[eids].to(device)
            seeds = torch.unique(torch.cat([heads, tails]))
            loader = DGLDataLoader(g, seeds, sampler,
                                   batch_size=len(seeds), shuffle=False,
                                   drop_last=False,
                                   num_workers=args.num_workers,
                                   device=device)
            input_nodes, output_nodes, blocks = next(iter(loader))
            h = enc.encode_blocks(blocks, feats[input_nodes])
            emb_h = _gather(h, heads, output_nodes)
            emb_t = _gather(h, tails, output_nodes)
            emb_n = emb_t[torch.randperm(emb_t.shape[0])]   # 批内换尾负采样
            loss = _bce_loss(dec, emb_h, emb_t, emb_n)
            opt.zero_grad()
            loss.backward()
            opt.step()
        ep_times.append(time.time() - t0)

        if epoch % args.eval_every == 0:
            a_t = time.time()
            val_auc = evaluate_auc(enc, dec, g, data.feats, data.val_pos,
                                   data.val_neg, device)
            ev_times.append(time.time() - a_t)
            if val_auc > best:
                best, best_ep, bad = val_auc, epoch, 0
                best_state = (copy.deepcopy(enc.state_dict()),
                              copy.deepcopy(dec.state_dict()))
            else:
                bad += 1
            if args.verbose:
                print(f"  epoch {epoch:4d} val_auc={val_auc:.4f}", flush=True)
            if bad >= args.patience:
                print(f"[early-stop] epoch {epoch}")
                break
    return best, best_ep, best_state, ep_times, ev_times


def main():
    args = make_parser().parse_args()
    args = defaults(args)
    args.device = resolve_device(args.device)
    set_seed(args.seed)
    print(f"[task2/{args.mode}] dataset={args.dataset} model={args.model} "
          f"device={args.device} lr={args.lr} layers={args.layers} "
          f"mode={args.mode}")

    data = load_lp_data(args.dataset, args.data_dir, seed=args.seed)
    print("[data]", data)

    enc, dec, n_params = _build_model(args, data, args.device)
    print(f"[model] encoder={args.model.upper()} params={n_params}")

    if args.mode == "full":
        best, best_ep, best_state, ep_t, ev_t = train_full(
            args, data, enc, dec, args.device)
    else:
        best, best_ep, best_state, ep_t, ev_t = train_sample(
            args, data, enc, dec, args.device)

    if best_state is not None:
        enc.load_state_dict(best_state[0])
        dec.load_state_dict(best_state[1])
    test_auc = evaluate_auc(enc, dec, data.train_g, data.feats,
                            data.test_pos, data.test_neg, args.device)

    avg = sum(ep_t) / max(len(ep_t), 1)
    row = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
           "dataset": args.dataset, "model": args.model, "mode": args.mode,
           "seed": args.seed, "lr": args.lr, "layers": args.layers,
           "hidden": args.hidden, "fanouts": args.fanouts,
           "batch_size": args.batch_size, "n_params": n_params,
           "val_auc": round(best, 4), "best_epoch": best_ep,
           "test_auc": round(test_auc, 4), "train_epochs": len(ep_t),
           "avg_epoch_sec": round(avg, 3), "train_sec": round(sum(ep_t), 3),
           "eval_sec": round(sum(ev_t), 3), "device": args.device}
    append_result(args.results, row)

    print("-" * 70)
    print(f"[done] best val_auc={best:.4f} @ epoch {best_ep} | "
          f"test_auc={test_auc:.4f}")
    print(f"[time] avg_epoch={avg:.3f}s total={sum(ep_t):.3f}s")


if __name__ == "__main__":
    main()

