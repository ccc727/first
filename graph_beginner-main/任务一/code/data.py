# -*- coding: utf-8 -*-
"""
data.py

加载任务一所需的三个数据集：
  - Cora / Citeseer：DGL 内置 citation 数据集（自带标准 train/val/test 划分）
  - Flickr        ：GraphSAINT Flickr（节点约 8.9 万、500 维特征、7 类），
                    本模块直接读取/下载其原始文件(adj_full.npz, feats.npy,
                    class_map.json, role.json)并构造 DGL 图，
                    划分取自 role.json（官方随机 50/25/25）。

统一返回一个 NodeData 对象，包含：
  g, feats, labels, train_idx, val_idx, test_idx, num_classes

说明：
  1) Cora/Citeseer/Flickr 的划分掩码均由数据集自带；
  2) 加载后统一把图处理为「无向图 + 每个节点一条自环」，
     保证 GCN 的对称归一化有意义，且对四种模型使用完全一致的图结构。
"""
import os

import dgl
import torch

# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------

def _as_index(mask: torch.Tensor) -> torch.Tensor:
    """把 bool 掩码或索引数组统一转成 1 维索引张量。"""
    mask = mask.reshape(-1)
    if mask.dtype == torch.bool:
        return mask.nonzero(as_tuple=False).reshape(-1)
    return mask.long()


def _ensure_self_loop(g: dgl.DGLGraph) -> dgl.DGLGraph:
    """每个节点恰好保留一条自环（先删再加，避免重复自环）。"""
    g = dgl.remove_self_loop(g)
    return dgl.add_self_loop(g)


def _make_undirected(g: dgl.DGLGraph) -> dgl.DGLGraph:
    """若图为有向图则补上反向边；已是无向图则保持不变（节点 id 不变）。"""
    g = g.to("cpu")
    u, v = g.edges()
    n = g.num_nodes()
    m = u.numel()
    if m == 0 or n == 0:
        return g
    # 快速判断每条边 (u->v) 的反向边 (v->u) 是否都在边集中
    key = u * n + v
    rev = v * n + u
    skey, _ = torch.sort(key)
    pos = torch.searchsorted(skey, rev).clamp(max=m - 1)
    if (skey[pos] == rev).all().item():
        return g
    return dgl.add_reverse_edges(g, copy_ndata=True)


def _download_file(url: str, dest: str, timeout: int = 60):
    """带进度与临时文件的简单下载（可断点续传）。"""
    import urllib.request

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    existing = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": "bytes=%d-" % existing} if existing else {}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = existing + int(resp.headers.get("Content-Length", 0))
        with open(tmp, "ab") as f:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done = os.path.getsize(tmp)
                if done < total:
                    print(f"\r下载中 {done / 1e6:.1f}/{total / 1e6:.1f} MB",
                          end="", flush=True)
    os.replace(tmp, dest)
    print(f"\n下载完成: {dest}")


def _random_split(n_nodes: int, seed: int = 0):
    """生成 60/20/20 的随机划分（仅当数据集本身没有划分时使用）。"""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_nodes, generator=g)
    n_tr, n_va = int(n_nodes * 0.6), int(n_nodes * 0.2)
    return perm[:n_tr], perm[n_tr:n_tr + n_va], perm[n_tr + n_va:]


# ----------------------------------------------------------------------------
# 数据对象
# ----------------------------------------------------------------------------

class NodeData:
    """节点分类数据集的统一容器。"""

    def __init__(self, name, g, feats, labels, train_idx, val_idx, test_idx,
                 num_classes):
        self.name = name
        self.g = g
        self.feats = feats
        self.labels = labels
        self.train_idx = train_idx
        self.val_idx = val_idx
        self.test_idx = test_idx
        self.num_classes = int(num_classes)
        self.num_nodes = int(g.num_nodes())
        self.num_edges = int(g.num_edges())

    def __repr__(self):
        return (f"NodeData({self.name}: nodes={self.num_nodes}, "
                f"edges={self.num_edges}, features={self.feats.shape[1]}, "
                f"classes={self.num_classes}, "
                f"train={len(self.train_idx)}, val={len(self.val_idx)}, "
                f"test={len(self.test_idx)})")


# ----------------------------------------------------------------------------
# 各数据集加载函数
# ----------------------------------------------------------------------------

def _load_citation(name: str, data_dir: str, seed: int = 0,
                   self_loop: bool = True) -> NodeData:
    """加载 DGL 内置的 Cora / Citeseer。"""
    if name == "cora":
        from dgl.data import CoraGraphDataset as Ctor
    elif name == "citeseer":
        from dgl.data import CiteseerGraphDataset as Ctor
    else:
        raise ValueError(f"Unknown citation dataset: {name}")

    raw_dir = os.path.join(data_dir, "dgl")
    dataset = Ctor(raw_dir=raw_dir)
    item = dataset[0]
    g = item[0] if isinstance(item, (list, tuple)) else item
    g = g.to("cpu")

    feats = g.ndata["feat"].float()
    labels = g.ndata["label"].long()

    # 标准划分掩码（若数据集未提供，则随机划分）
    if all(k in g.ndata for k in ("train_mask", "val_mask", "test_mask")):
        train_idx = _as_index(g.ndata["train_mask"])
        val_idx = _as_index(g.ndata["val_mask"])
        test_idx = _as_index(g.ndata["test_mask"])
    else:
        train_idx, val_idx, test_idx = _random_split(g.num_nodes(), seed)

    num_classes = getattr(dataset, "num_classes", None)
    if num_classes is None:
        num_classes = int(labels.max().item()) + 1

    if self_loop:
        g = _ensure_self_loop(g)

    return NodeData(name, g, feats, labels, train_idx, val_idx, test_idx,
                    num_classes)


def _load_flickr(data_dir: str, seed: int = 0,
                 self_loop: bool = True) -> NodeData:
    """加载 Flickr 数据集（GraphSAINT，约 8.9 万节点 / 500 维特征 / 7 类）。

    直接下载/读取原始文件并构造 DGL 图，绕开 DGL 内置数据集的缓存问题，
    保证 train/val/test 划分（role.json，50/25/25）每次加载都准确。
    原始文件：adj_full.npz（邻接矩阵）、feats.npy（特征）、
              class_map.json（标签）、role.json（划分）。
    """
    import json
    import zipfile

    import numpy as np
    import scipy.sparse as sp

    base = os.path.join(data_dir, "dgl")
    raw = os.path.join(base, "flickr")
    zip_path = os.path.join(base, "flickr.zip")
    os.makedirs(base, exist_ok=True)

    needed = ["adj_full.npz", "feats.npy", "class_map.json", "role.json"]
    if not all(os.path.exists(os.path.join(raw, f)) for f in needed):
        if not os.path.exists(zip_path) or os.path.getsize(zip_path) == 0:
            _download_file("https://data.dgl.ai/dataset/flickr.zip", zip_path)
        os.makedirs(raw, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw)

    coo = sp.load_npz(os.path.join(raw, "adj_full.npz"))
    g = dgl.from_scipy(coo)

    feats = torch.from_numpy(np.load(os.path.join(raw, "feats.npy"),
                                     allow_pickle=False)).float()

    with open(os.path.join(raw, "class_map.json")) as f:
        class_map = json.load(f)
    n = feats.shape[0]
    labels = torch.tensor([class_map[str(i)] for i in range(n)],
                          dtype=torch.long)

    with open(os.path.join(raw, "role.json")) as f:
        role = json.load(f)

    def _mask(key):
        m = torch.zeros(n, dtype=torch.bool)
        m[torch.as_tensor(role[key], dtype=torch.long)] = True
        return m

    g.ndata["feat"] = feats
    g.ndata["label"] = labels
    g.ndata["train_mask"] = _mask("tr")
    g.ndata["val_mask"] = _mask("va")
    g.ndata["test_mask"] = _mask("te")

    train_idx = _as_index(g.ndata["train_mask"])
    val_idx = _as_index(g.ndata["val_mask"])
    test_idx = _as_index(g.ndata["test_mask"])

    g = _make_undirected(g)
    if self_loop:
        g = _ensure_self_loop(g)

    return NodeData("flickr", g, feats, labels, train_idx, val_idx, test_idx,
                    num_classes=7)




# ----------------------------------------------------------------------------
# 统一入口
# ----------------------------------------------------------------------------

def load_dataset(name: str, data_dir: str, seed: int = 0,
                 self_loop: bool = True) -> NodeData:
    """按名称加载数据集。name: cora | citeseer | flickr"""
    name = name.strip().lower()
    os.makedirs(data_dir, exist_ok=True)
    if name in ("cora", "citeseer"):
        return _load_citation(name, data_dir, seed, self_loop)
    if name in ("flickr", "ogbn-flickr"):
        return _load_flickr(data_dir, seed, self_loop)
    raise ValueError(
        f"Unknown dataset '{name}'，可选: cora / citeseer / flickr"
    )
