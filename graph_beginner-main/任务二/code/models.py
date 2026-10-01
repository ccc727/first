# -*- coding: utf-8 -*-
"""
models.py —— 任务二：链路预测模型（编码器 + 打分器）

编码器为 GCN / GAT / GraphSAGE / GIN 四种，与任务一共用 DGL 卷积层实现，
但输出的是 d 维节点嵌入（不接分类头）；打分器把 (h_u, h_v) 拼接后过 MLP，
输出该边存在的 logit。

接口：
- encode_graph(graph, feat)  -> (N, d) 整图嵌入
- encode_blocks(blocks, feat)-> (B, d) 采样得到的输出节点嵌入（用于子图训练）
- forward_pair(h_src, h_dst) -> logit
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv, GINConv, GraphConv, SAGEConv


def _mlp(in_dim, hid, out_dim):
    return nn.Sequential(nn.Linear(in_dim, hid), nn.ReLU(),
                         nn.Linear(hid, out_dim))


class BaseEncoder(nn.Module):
    def __init__(self, in_feats, hidden, n_layers=2, dropout=0.5):
        super().__init__()
        self.in_feats = in_feats
        self.hidden = hidden
        self.n_layers = n_layers
        self.dropout = dropout
        self.layers = nn.ModuleList()

    def _post(self, h, i):
        if i < self.n_layers - 1:
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def encode_graph(self, graph, feat):
        h = feat
        for i, conv in enumerate(self.layers):
            h = conv(graph, h)
            h = self._post(h, i)
        return h

    def encode_blocks(self, blocks, feat):
        h = feat
        for i, conv in enumerate(self.layers):
            h = conv(blocks[i], h)
            h = self._post(h, i)
        return h


class GCNEncoder(BaseEncoder):
    def __init__(self, in_feats, hidden, n_layers=2, dropout=0.5):
        super().__init__(in_feats, hidden, n_layers, dropout)
        for i in range(n_layers):
            d_in = in_feats if i == 0 else hidden
            self.layers.append(GraphConv(d_in, hidden, norm="both"))


class SAGEEncoder(BaseEncoder):
    def __init__(self, in_feats, hidden, n_layers=2, dropout=0.5,
                 aggregator_type="mean"):
        super().__init__(in_feats, hidden, n_layers, dropout)
        for i in range(n_layers):
            d_in = in_feats if i == 0 else hidden
            self.layers.append(SAGEConv(d_in, hidden, aggregator_type))


class GATEncoder(BaseEncoder):
    def __init__(self, in_feats, hidden, n_layers=2, dropout=0.5,
                 num_heads=4, attn_dropout=0.0):
        super().__init__(in_feats, hidden, n_layers, dropout)
        self.num_heads = num_heads
        cur_in = in_feats
        for i in range(n_layers):
            per_head = max(1, hidden // num_heads)
            total = per_head * num_heads if i < n_layers - 1 else hidden
            heads = num_heads if i < n_layers - 1 else 1
            out_ph = per_head if i < n_layers - 1 else hidden
            self.layers.append(
                GATConv(cur_in, out_ph, num_heads=heads,
                        feat_drop=dropout, attn_drop=attn_dropout,
                        residual=False, activation=None))
            cur_in = total

    def _combine(self, rst, last):
        if rst.dim() <= 2:
            return rst
        if last:
            return rst.mean(dim=-2)
        return rst.flatten(start_dim=-2)

    def encode_graph(self, graph, feat):
        h = feat
        for i, conv in enumerate(self.layers):
            h = self._combine(conv(graph, h), i == self.n_layers - 1)
            h = self._post(h, i)
        return h

    def encode_blocks(self, blocks, feat):
        h = feat
        for i, conv in enumerate(self.layers):
            h = self._combine(conv(blocks[i], h), i == self.n_layers - 1)
            h = self._post(h, i)
        return h


class GINEncoder(BaseEncoder):
    def __init__(self, in_feats, hidden, n_layers=2, dropout=0.5,
                 aggregator_type="sum"):
        super().__init__(in_feats, hidden, n_layers, dropout)
        self.first_proj = nn.Linear(in_feats, hidden)
        for _ in range(n_layers):
            self.layers.append(GINConv(_mlp(hidden, hidden, hidden),
                                       aggregator_type=aggregator_type))

    def encode_graph(self, graph, feat):
        h = F.relu(self.first_proj(feat))
        h = F.dropout(h, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.layers):
            h = conv(graph, h)
            h = self._post(h, i)
        return h

    def encode_blocks(self, blocks, feat):
        h = F.relu(self.first_proj(feat))
        h = F.dropout(h, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.layers):
            h = conv(blocks[i], h)
            h = self._post(h, i)
        return h


def build_encoder(name, in_feats, hidden, n_layers=2, dropout=0.5, **kw):
    name = name.lower()
    if name == "gcn":
        return GCNEncoder(in_feats, hidden, n_layers, dropout)
    if name in ("sage", "graphsage"):
        return SAGEEncoder(in_feats, hidden, n_layers, dropout,
                           kw.get("aggregator_type", "mean"))
    if name == "gat":
        return GATEncoder(in_feats, hidden, n_layers, dropout,
                          kw.get("num_heads", 4), kw.get("attn_dropout", 0.0))
    if name == "gin":
        return GINEncoder(in_feats, hidden, n_layers, dropout,
                          kw.get("aggregator_type", "sum"))
    raise ValueError(name)


class LinkPredictor(nn.Module):
    """把两端嵌入拼接过 MLP 的边打分器。"""

    def __init__(self, emb_dim, hidden=256):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(emb_dim * 2, hidden),
                                nn.ReLU(),
                                nn.Dropout(0.1),
                                nn.Linear(hidden, 1))

    def forward_pair(self, h_src, h_dst):
        return self.fc(torch.cat([h_src, h_dst], dim=-1)).squeeze(-1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
