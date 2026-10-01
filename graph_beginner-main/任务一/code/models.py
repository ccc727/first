# -*- coding: utf-8 -*-
"""
models.py

基于 DGL 实现的四种主流 GNN 模型：GCN / GAT / GraphSAGE / GIN。

每个模型都提供两个前向接口，因此既能做「全图训练」，也能配合 DGL 自带的
NeighborSampler 做「子图（分块）训练」：

- forward_graph(graph, feat): 在整张图上做前向传播，feat 是全部节点特征，
  返回全部节点的 logits。
- forward_blocks(blocks, feat): 在采样得到的 block 序列上逐层消息传递
  （邻居采样训练的标准写法），feat 是 blocks[0] 源节点(input_nodes)的特征，
  返回 blocks[-1] 目标节点(output_nodes/种子节点) 的 logits。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv, GINConv, GraphConv, SAGEConv


def _make_mlp(in_dim: int, out_dim: int, hidden_dim: int = None):
    """两层 MLP，用作 GIN 卷积的更新函数(apply_func)。"""
    h = hidden_dim if hidden_dim is not None else out_dim
    return nn.Sequential(
        nn.Linear(in_dim, h),
        nn.ReLU(),
        nn.Linear(h, out_dim),
    )


class BaseGNN(nn.Module):
    """所有模型的公共父类：统一管理超参数与层间激活/dropout。"""

    def __init__(self, in_feats: int, hidden: int, n_classes: int,
                 n_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.in_feats = in_feats
        self.hidden = hidden
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout = dropout

    def _post_process(self, h, layer_idx: int):
        if layer_idx < self.n_layers - 1:
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h

    def forward_blocks(self, blocks, feat):
        """在采样得到的 block 序列上逐层前向（NeighborSampler 子图训练）。"""
        h = feat
        for i, layer in enumerate(self.layers):
            h = layer(blocks[i], h)
            h = self._post_process(h, i)
        return h


class GCN(BaseGNN):
    """图卷积网络 (Kipf & Welling, 2017)，norm='both' 为对称归一化。"""

    def __init__(self, in_feats: int, hidden: int, n_classes: int,
                 n_layers: int = 2, dropout: float = 0.5):
        super().__init__(in_feats, hidden, n_classes, n_layers, dropout)
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = in_feats if i == 0 else hidden
            out_dim = n_classes if i == n_layers - 1 else hidden
            self.layers.append(GraphConv(in_dim, out_dim, norm="both",
                                         weight=True, bias=True))

    def forward_graph(self, graph, feat):
        h = feat
        for i, conv in enumerate(self.layers):
            h = conv(graph, h)
            h = self._post_process(h, i)
        return h


class GraphSAGE(BaseGNN):
    """GraphSAGE (Hamilton et al., 2017)，支持 mean/gcn/pool 等聚合器。"""

    def __init__(self, in_feats: int, hidden: int, n_classes: int,
                 n_layers: int = 2, dropout: float = 0.5,
                 aggregator_type: str = "mean"):
        super().__init__(in_feats, hidden, n_classes, n_layers, dropout)
        self.aggregator_type = aggregator_type
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_dim = in_feats if i == 0 else hidden
            out_dim = n_classes if i == n_layers - 1 else hidden
            self.layers.append(SAGEConv(in_dim, out_dim, aggregator_type))

    def forward_graph(self, graph, feat):
        h = feat
        for i, conv in enumerate(self.layers):
            h = conv(graph, h)
            h = self._post_process(h, i)
        return h


class GAT(BaseGNN):
    """图注意力网络 (Veličković et al., 2018)，带多头注意力。"""

    def __init__(self, in_feats: int, hidden: int, n_classes: int,
                 n_layers: int = 2, dropout: float = 0.5, num_heads: int = 8,
                 attn_dropout: float = 0.0, residual: bool = False):
        super().__init__(in_feats, hidden, n_classes, n_layers, dropout)
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.residual = residual
        self.layers = nn.ModuleList()
        cur_in = in_feats
        for i in range(n_layers):
            last = (i == n_layers - 1)
            heads = 1 if last else num_heads
            out_per_head = n_classes if last else max(1, hidden // num_heads)
            out_total = n_classes if last else out_per_head * heads
            use_res = residual and (not last) and (cur_in == out_total)
            self.layers.append(
                GATConv(cur_in, out_per_head, num_heads=heads,
                        feat_drop=dropout, attn_drop=attn_dropout,
                        residual=use_res, activation=None)
            )
            cur_in = out_total
        self.out_dim = cur_in

    @staticmethod
    def _combine(rst, last: bool):
        """兼容不同 DGL 版本的 GATConv 输出。

        - 某些版本 concat=True 时返回 (N, H*D)，concat=False 时返回 (N, D)
        - dgl 2.2 的 GATConv 恒返回 (N, H, D)
        因此按维度数统一处理。
        """
        if rst.dim() <= 2:          # 已经是 (N, F)，无需处理
            return rst
        if last:                     # 最后一层单头：各头取平均
            return rst.mean(dim=-2)
        return rst.flatten(start_dim=-2)   # 隐藏层：拼接各头

    def forward_graph(self, graph, feat):
        h = feat
        for i, conv in enumerate(self.layers):
            last = (i == len(self.layers) - 1)
            h = self._combine(conv(graph, h), last)
            if not last:
                h = self._post_process(h, i)
        return h

    def forward_blocks(self, blocks, feat):
        h = feat
        for i, conv in enumerate(self.layers):
            last = (i == len(self.layers) - 1)
            h = self._combine(conv(blocks[i], h), last)
            if not last:
                h = self._post_process(h, i)
        return h



class GIN(BaseGNN):
    """图同构网络 (Xu et al., 2019)：sum 聚合 + 每层 MLP 更新。"""

    def __init__(self, in_feats: int, hidden: int, n_classes: int,
                 n_layers: int = 2, dropout: float = 0.5,
                 aggregator_type: str = "sum", learn_eps: bool = False):
        super().__init__(in_feats, hidden, n_classes, n_layers, dropout)
        self.aggregator_type = aggregator_type
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            apply_func = _make_mlp(hidden, hidden, hidden)
            self.layers.append(GINConv(apply_func, aggregator_type=aggregator_type,
                                       learn_eps=learn_eps))
        self.first_proj = nn.Linear(in_feats, hidden)
        self.classifier = nn.Linear(hidden, n_classes)

    def forward_graph(self, graph, feat):
        h = F.relu(self.first_proj(feat))
        h = F.dropout(h, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.layers):
            h = conv(graph, h)
            h = self._post_process(h, i)
        return self.classifier(h)

    def forward_blocks(self, blocks, feat):
        h = F.relu(self.first_proj(feat))
        h = F.dropout(h, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.layers):
            h = conv(blocks[i], h)
            h = self._post_process(h, i)
        return self.classifier(h)


def build_model(name: str, in_feats: int, hidden: int, n_classes: int,
                n_layers: int = 2, dropout: float = 0.5, **kwargs):
    """按名称构造模型。"""
    name = name.lower()
    if name == "gcn":
        model = GCN(in_feats, hidden, n_classes, n_layers, dropout)
    elif name == "gat":
        model = GAT(in_feats, hidden, n_classes, n_layers, dropout,
                    num_heads=int(kwargs.get("num_heads", 8)),
                    attn_dropout=float(kwargs.get("attn_dropout", 0.0)),
                    residual=bool(kwargs.get("residual", False)))
    elif name in ("sage", "graphsage"):
        model = GraphSAGE(in_feats, hidden, n_classes, n_layers, dropout,
                          aggregator_type=kwargs.get("aggregator_type", "mean"))
    elif name == "gin":
        model = GIN(in_feats, hidden, n_classes, n_layers, dropout,
                    aggregator_type=kwargs.get("aggregator_type", "sum"),
                    learn_eps=bool(kwargs.get("learn_eps", False)))
    else:
        raise ValueError(f"Unknown model: {name}. 可选 gcn/gat/sage/gin")
    return model


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

