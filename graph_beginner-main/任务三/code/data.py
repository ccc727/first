# -*- coding: utf-8 -*-
"""
data.py —— 任务三：图分类 / 图回归数据集

分类任务使用 TUDataset（MUTAG 等）；回归任务使用 ZINC-12K。
TUDataset 无官方划分 -> 随机 80/10/10；
ZINC 自带 train/valid/test。
"""
import os
import random

import dgl
import torch


class GraphData:
    """统一的图数据集容器。"""

    def __init__(self, name, graphs, labels, idx_train, idx_val, idx_test,
                 n_node_types, task, node_key="node_labels",
                 y_mean=0.0, y_std=1.0):
        self.name = name
        self.graphs = graphs          # list[DGLGraph]
        self.labels = labels          # Tensor: float(回归/归一化后) 或 long(分类)
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test
        self.n_node_types = n_node_types
        self.task = task              # "cls" | "reg"
        self.node_key = node_key
        self.y_mean = y_mean
        self.y_std = y_std

    def __repr__(self):
        return (f"GraphData({self.name}, task={self.task}, "
                f"graphs={len(self.graphs)}, "
                f"train={len(self.idx_train)}, val={len(self.idx_val)}, "
                f"test={len(self.idx_test)}, node_types={self.n_node_types})")

    def batch(self, idx):
        idx = list(idx)
        bg = dgl.batch([self.graphs[i] for i in idx])
        node_ids = bg.ndata[self.node_key].reshape(-1).long()
        return bg, node_ids, self.labels[idx].reshape(-1)


def _tu_graph_node_types(ds):
    mx = 0
    for g, _ in ds:
        mx = max(mx, int(g.ndata["node_labels"].max().item()))
    return mx + 1


def load_tu(name, data_dir, seed=0, limit=0):
    """加载 TUDataset 并随机划分。"""
    from dgl.data import TUDataset

    root = os.path.join(data_dir, "tu")
    ds = TUDataset(name=name.upper(), raw_dir=root)
    n = min(len(ds), limit) if limit and limit > 0 else len(ds)

    graphs, labels = [], []
    for i in range(n):
        g, y = ds[i]
        if isinstance(y, (list, tuple)):
            y = y[0]
        y = torch.as_tensor(y)
        graphs.append(g)
        labels.append(y)
    labels = torch.stack(labels).reshape(-1)  # (n,)
    n_node_types = _tu_graph_node_types(ds) if n == len(ds) else _max_type(graphs)

    if labels.dtype == torch.float and (labels == labels.long()).all():
        labels = labels.long()
    # 处理类别标签不在 0 开始的情况（如 MUTAG 为 -1/+1）
    if labels.min() < 0:
        labels = labels - labels.min()

    rng = random.Random(seed)
    perm = list(range(n))
    rng.shuffle(perm)
    ntr = int(n * 0.8)
    nva = int(n * 0.1)
    return GraphData(name, graphs, labels,
                     torch.tensor(perm[:ntr]), torch.tensor(perm[ntr:ntr + nva]),
                     torch.tensor(perm[ntr + nva:]), n_node_types, "cls")


def _max_type(graphs, key="node_labels"):
    mx = 0
    for g in graphs:
        mx = max(mx, int(g.ndata[key].max().item()))
    return mx + 1


def load_zinc(data_dir, seed=0, limit_train=0):
    """加载 ZINC-12K（回归）。默认对标签做 (y-mean)/std 归一化。"""
    from dgl.data import ZINCDataset

    root = os.path.join(data_dir, "zinc")
    sets = {}
    for mode in ["train", "valid", "test"]:
        ds = ZINCDataset(mode=mode, raw_dir=root)
        graphs, labels = [], []
        lim = limit_train if mode == "train" else 0
        n = min(len(ds), lim) if lim and lim > 0 else len(ds)
        for i in range(n):
            g, y = ds[i]
            graphs.append(g)
            labels.append(float(y))
        sets[mode] = (graphs, torch.tensor(labels).float())

    tr_g, tr_y = sets["train"]
    n_node_types = _max_type(tr_g, key="feat")  # ZINC 的节点 feat 为原子类型 id
    y_mean = float(tr_y.mean())
    y_std = float(max(tr_y.std(), 1e-6))

    def _norm(labels):
        return (labels - y_mean) / y_std

    va_g, va_y = sets["valid"]
    te_g, te_y = sets["test"]
    return GraphData("zinc", tr_g + va_g + te_g,
                     torch.cat([_norm(tr_y), _norm(va_y), _norm(te_y)]),
                     torch.arange(len(tr_g)), torch.arange(len(tr_g),
                                                           len(tr_g) + len(va_g)),
                     torch.arange(len(tr_g) + len(va_g), len(tr_g) + len(va_g)
                                  + len(te_g)),
                     n_node_types, "reg", node_key="feat",
                     y_mean=y_mean, y_std=y_std)


def load_graph_dataset(name, data_dir, seed=0, limit_train=0):
    name = name.lower()
    if name in ("mutag", "proteins", "imdb-binary", "imdb-multi", "collab"):
        return load_tu(name, data_dir, seed, limit_train)
    if name == "zinc":
        return load_zinc(data_dir, seed, limit_train)
    raise ValueError(name)
