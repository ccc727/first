# -*- coding: utf-8 -*-
"""
models.py —— 任务三：图分类 / 图回归模型

结构：节点特征(可选嵌入) -> GNN 编码器 -> 图级 Readout(池化) -> MLP 头
池化方式支持平均池化(AvgPooling)/最大池化(MaxPooling)/最小池化(MinPooling)。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch import GATConv, GINConv, GraphConv, SAGEConv


def _mlp2(a, b, c):
    return nn.Sequential(nn.Linear(a, b), nn.ReLU(), nn.Linear(b, c))


class GNNEncoder(nn.Module):
    """四选一编码器 + 图级池化 + MLP 头。"""

    def __init__(self, model_name, hidden, out_dim, n_layers=3,
                 dropout=0.5, pool="mean", use_node_id=False,
                 n_node_types=None, readout_hidden=128):
        super().__init__()
        self.model_name = model_name
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_node_id = use_node_id

        if use_node_id:
            assert n_node_types is not None
            self.node_emb = nn.Embedding(n_node_types, hidden)
            in_dim = hidden
        else:
            self.node_emb = None
            in_dim = hidden

        self.convs = nn.ModuleList()
        if model_name == "gcn":
            for i in range(n_layers):
                self.convs.append(GraphConv(in_dim if i == 0 else hidden,
                                            hidden, norm="both"))
        elif model_name in ("sage", "graphsage"):
            for i in range(n_layers):
                self.convs.append(SAGEConv(in_dim if i == 0 else hidden,
                                           hidden, "mean"))
        elif model_name == "gin":
            for i in range(n_layers):
                apply = _mlp2(in_dim if i == 0 else hidden, hidden, hidden)
                self.convs.append(GINConv(apply, aggregator_type="sum"))
        elif model_name == "gat":
            for i in range(n_layers):
                heads = 4 if i < n_layers - 1 else 1
                d_in = in_dim if i == 0 else hidden
                out_ph = hidden // heads if i < n_layers - 1 else hidden
                conv = GATConv(d_in, out_ph, num_heads=heads,
                               feat_drop=dropout, attn_drop=0.0,
                               residual=False, activation=None)
                self.convs.append(conv)
        else:
            raise ValueError(model_name)

        self.pool = pool  # mean / max / min
        self.head = nn.Sequential(nn.Linear(hidden, readout_hidden),
                                  nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(readout_hidden, out_dim))

    def _combine_gat(self, h, last):
        if h.dim() <= 2:
            return h
        if last:
            return h.mean(dim=-2)
        return h.flatten(start_dim=-2)

    def forward(self, g, node_ids):
        """g 是 dgl.batch 图；node_ids 为 (sumN,) 的节点类型 id。返回图级输出。"""
        if self.node_emb is not None:
            h = self.node_emb(node_ids.to(g.device))
        else:
            h = node_ids.float()
        for i, conv in enumerate(self.convs):
            if self.model_name == "gat":
                h = self._combine_gat(conv(g, h), i == self.n_layers - 1)
            else:
                h = conv(g, h)
            if i < self.n_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        g.ndata["h"] = h
        if self.pool == "mean":
            out = dgl.readout_nodes(g, "h", op="mean")
        elif self.pool == "max":
            out = dgl.readout_nodes(g, "h", op="max")
        elif self.pool == "min":
            out = dgl.readout_nodes(g, "h", op="min")
        else:
            raise ValueError(self.pool)
        return self.head(out)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
