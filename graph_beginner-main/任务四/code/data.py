# -*- coding: utf-8 -*-
"""
data.py —— 任务四：知识图谱数据集

使用 DGL 提供的 FB15k-237（含标准 train/valid/test 三元组划分）。
返回：
    train / valid / test : 三元组 (h, r, t) 索引张量
    num_entities / num_relations
"""
import os

import torch


class KGData:
    def __init__(self, name, train, valid, test, num_entities, num_relations):
        self.name = name
        self.train = train
        self.valid = valid
        self.test = test
        self.num_entities = int(num_entities)
        self.num_relations = int(num_relations)

    def __repr__(self):
        return (f"KGData({self.name}: entities={self.num_entities}, "
                f"relations={self.num_relations}, "
                f"train={len(self.train)}, valid={len(self.valid)}, "
                f"test={len(self.test)})")


def _tensor(a):
    return torch.as_tensor(a, dtype=torch.long)


def load_kg(name, data_dir, seed=0):
    """加载 dgl 内置 FB15k-237（其它可换 FB15k/wn18）。"""
    from dgl.data import FB15k237Dataset

    raw_dir = os.path.join(data_dir, "dgl")
    ds = FB15k237Dataset(raw_dir=raw_dir)
    n_ent = ds.num_nodes
    n_rel = ds.num_rels
    train = _tensor(ds._train)
    valid = _tensor(ds._valid)
    test = _tensor(ds._test)
    return KGData(name, train, valid, test, n_ent, n_rel)
