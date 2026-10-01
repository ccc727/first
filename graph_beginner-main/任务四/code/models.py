# -*- coding: utf-8 -*-
"""
models.py —— 任务四：知识图谱补全模型

TransE / RotatE / ConvE 三种打分函数。
统一接口：
    score_batch(h, r, t)  -> (B,)  batch 三元组得分（越大越正确）
    tail_scores(h, r)     -> (B, E) 固定 (h,r) 时所有尾实体的得分
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, dim=100, margin=0.0,
                 p_norm=2):
        super().__init__()
        self.name = "transe"
        self.dim = dim
        self.p = p_norm
        self.entity = nn.Embedding(num_entities, dim)
        self.relation = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.relation.weight)

    def score_batch(self, h, r, t):
        s = self.entity(h) + self.relation(r) - self.entity(t)
        return -torch.norm(s, p=self.p, dim=-1)

    def tail_scores(self, h, r):
        hr = self.entity(h) + self.relation(r)          # (B, d)
        return -(hr.unsqueeze(1) - self.entity.weight.unsqueeze(0)
                 ).norm(p=self.p, dim=-1)


class RotatE(nn.Module):
    """RotatE：复数空间旋转 h ∘ r ≈ t。

    关系是**可学习**的相位角（原论文做法）：r 用相位 θ_r 表示，旋转由 cos/sin 给出，
    因此需要把相位当作参数优化，而不是固定的随机常数。
    score = -||h∘r − t||²（取负，越大越正确）。
    """

    def __init__(self, num_entities, num_relations, dim=100):
        super().__init__()
        self.name = "rotate"
        self.dim = dim
        # 实体复数 = [实部(d), 虚部(d)]
        self.entity = nn.Embedding(num_entities, 2 * dim)
        nn.init.normal_(self.entity.weight, std=0.05)
        # 关系相位角：可学习参数，初始为 [0, 2π) 上的均匀值
        init_phase = torch.rand(num_relations, dim) * 2 * torch.pi
        self.phase = nn.Parameter(init_phase)

    def _rotate(self, h_re, h_im, r):
        """h∘r 的实部与虚部。"""
        c = torch.cos(self.phase[r])
        s = torch.sin(self.phase[r])
        return h_re * c - h_im * s, h_re * s + h_im * c

    def score_batch(self, h, r, t):
        e = self.entity
        h_re, h_im = e(h)[:, :self.dim], e(h)[:, self.dim:]
        re, im = self._rotate(h_re, h_im, r)
        t_re, t_im = e(t)[:, :self.dim], e(t)[:, self.dim:]
        return -(torch.pow(re - t_re, 2) + torch.pow(im - t_im, 2)).sum(-1)

    def tail_scores(self, h, r):
        e = self.entity
        h_re, h_im = e(h)[:, :self.dim], e(h)[:, self.dim:]
        re, im = self._rotate(h_re, h_im, r)              # (B, d)
        w = self.entity.weight                            # (E, 2d)
        w_re, w_im = w[:, :self.dim], w[:, self.dim:]
        d2 = (re.unsqueeze(1) - w_re.unsqueeze(0)) ** 2 + \
             (im.unsqueeze(1) - w_im.unsqueeze(0)) ** 2
        return -d2.sum(-1)


class ConvE(nn.Module):
    """简化 ConvE：把 (h, r) 堆成 2×d 的图，用 2D 卷积提取特征后与尾实体点积。"""

    def __init__(self, num_entities, num_relations, dim=200, conv_out=32,
                 kernel=(2, 3)):
        super().__init__()
        self.name = "conve"
        self.dim = dim
        self.entity = nn.Embedding(num_entities, dim)     # 尾实体矩阵
        self.rel = nn.Embedding(num_relations, dim)
        nn.init.xavier_uniform_(self.entity.weight)
        nn.init.xavier_uniform_(self.rel.weight)
        pad = (0, 1)
        self.conv = nn.Conv2d(1, conv_out, kernel_size=kernel, padding=pad)
        self.bn = nn.BatchNorm2d(conv_out)
        self.fc = nn.Linear(conv_out * dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def _hidden(self, h, r):
        x = torch.cat([self.entity(h).unsqueeze(1),
                       self.rel(r).unsqueeze(1)], dim=1)  # (B, 2, d)
        x = x.unsqueeze(1)                                 # (B,1,2,d)
        x = self.conv(x)
        x = self.bn(F.relu(x))
        x = x.view(x.size(0), -1)                          # (B, C*d)
        x = self.fc(x)
        return self.bn2(F.relu(x))                         # (B, d)

    def score_batch(self, h, r, t):
        hidden = self._hidden(h, r)
        return (hidden * self.entity(t)).sum(-1)

    def tail_scores(self, h, r):
        hidden = self._hidden(h, r)                        # (B, d)
        return hidden @ self.entity.weight.t()             # (B, E)


def build_kge(name, num_entities, num_relations, dim, **kw):
    name = name.lower()
    if name == "transe":
        return TransE(num_entities, num_relations, dim)
    if name == "rotate":
        return RotatE(num_entities, num_relations, dim)
    if name == "conve":
        return ConvE(num_entities, num_relations, dim)
    raise ValueError(name)


def count_parameters(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
