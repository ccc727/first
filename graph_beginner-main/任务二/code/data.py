# -*- coding: utf-8 -*-
"""
data.py —— 任务二：链路预测数据准备

数据来源与任务一相同（Cora / Citeseer / Flickr），在“整图 + 无自环”的图上，
把全部边随机划分为 train / valid / test 三部分（85% / 5% / 10%），
并为 valid / test 生成等量的负样本（图中不存在的节点对）。

返回对象 LPData：
    train_g       训练用子图（只含 train 边）
    feats / num_nodes / num_rels
    train_src/train_dst/train_eids（训练边与索引，用于全图训练逐批采样）
    val_pos / val_neg / test_pos / test_neg
"""
import os
import sys
import importlib.util

import dgl
import torch

_T1_DATA_PY = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "..", "任务一", "code", "data.py")


def _import_task1_data():
    spec = importlib.util.spec_from_file_location("task1_data", _T1_DATA_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_t1 = _import_task1_data()
load_dataset = _t1.load_dataset



class LPData:
    def __init__(self, name, train_g, feats, train_src, train_dst, val_pos,
                 val_neg, test_pos, test_neg, num_nodes):
        self.name = name
        self.train_g = train_g
        self.feats = feats
        self.train_src = train_src
        self.train_dst = train_dst
        self.train_eids = torch.arange(train_src.shape[0])
        self.val_pos = val_pos
        self.val_neg = val_neg
        self.test_pos = test_pos
        self.test_neg = test_neg
        self.num_nodes = int(num_nodes)

    def __repr__(self):
        return (f"LPData({self.name}: nodes={self.num_nodes}, "
                f"train_edges={len(self.train_src)}, "
                f"val_pos={len(self.val_pos[0])}, "
                f"test_pos={len(self.test_pos[0])})")


def _split_edge_ids(n_edges, val_ratio, test_ratio, seed):
    """随机切分边，返回 (train_idx, val_idx, test_idx)。"""
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_edges, generator=gen)
    n_val = int(n_edges * val_ratio)
    n_test = int(n_edges * test_ratio)
    return perm[:n_edges - n_val - n_test], \
        perm[n_edges - n_val - n_test:n_edges - n_test], perm[n_edges - n_test:]


def _pack(u, v, n):
    return (u * n + v).tolist()


def sample_negatives(num_nodes, pos_src, pos_dst, forbidden, n_neg=1, seed=0):
    """为每条正边采样 n_neg 条负边（尾实体随机替换，不在 forbidden 中）。"""
    gen = torch.Generator().manual_seed(seed)
    k = len(pos_src)
    forbidden_set = set(forbidden)
    neg_src, neg_dst = [], []
    attempts = 0
    while len(neg_dst) < k * n_neg and attempts < 200:
        src = torch.randint(0, num_nodes, (k * n_neg,), generator=gen)
        dst = torch.randint(0, num_nodes, (k * n_neg,), generator=gen)
        mask = src != dst
        cand_src = src[mask]
        cand_dst = dst[mask]
        keys = _pack(cand_src, cand_dst, num_nodes)
        seen = set()
        for i, key in enumerate(keys):
            if key in forbidden_set or key in seen:
                continue
            seen.add(key)
            neg_src.append(cand_src[i].item())
            neg_dst.append(cand_dst[i].item())
            if len(neg_dst) >= k * n_neg:
                break
        attempts += 1
    return torch.tensor(neg_src[:k * n_neg], dtype=torch.long), \
        torch.tensor(neg_dst[:k * n_neg], dtype=torch.long)


def load_lp_data(name, data_dir, seed=0, val_ratio=0.05, test_ratio=0.10):
    """加载图并划分边，返回 LPData。"""
    nd = load_dataset(name, data_dir, seed=seed, self_loop=False)
    g = nd.g
    feats = nd.feats
    n = g.num_nodes()

    u, v = g.edges()
    train_idx, val_idx, test_idx = _split_edge_ids(len(u), val_ratio,
                                                   test_ratio, seed)

    train_g = dgl.graph((u[train_idx], v[train_idx]), num_nodes=n)
    # 为训练子图加自环，避免消息传递时出现 0 入度节点（不影响训练/评测边）
    train_g = dgl.add_self_loop(train_g)
    train_g.ndata["feat"] = feats

    # 全部现有边（含划分后正边），负样本必须避开
    forbidden = _pack(u, v, n)

    val_pos = (u[val_idx], v[val_idx])
    test_pos = (u[test_idx], v[test_idx])
    val_neg = sample_negatives(n, val_pos[0], val_pos[1], forbidden,
                               n_neg=1, seed=seed + 1)
    test_neg = sample_negatives(n, test_pos[0], test_pos[1], forbidden,
                                n_neg=1, seed=seed + 2)

    return LPData(name, train_g, feats, u[train_idx], v[train_idx],
                  val_pos, val_neg, test_pos, test_neg, n)
