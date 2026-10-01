# 任务二：图上的链路预测

基于 **DGL**，在 **Cora / Citeseer / Flickr** 上完成链路预测（Link Prediction）。
编码器支持 **GCN / GAT / GraphSAGE / GIN**，打分器为“两端嵌入拼接 → MLP”。
提供两种训练方式：
- **全图训练 (full)**：每轮先在训练子图上整图编码一次，再分小批处理训练正边 +
  随机负尾边；
- **采样训练 (sample)**：每轮抽取一批训练边，以两端点为种子，用 DGL
  `NeighborSampler` 采 2 跳子图并在 block 上编码，批内换尾构造负样本。

## 目录

```
任务二/
├── data/            # 数据集（复用任务一 loader，首次自动下载）
├── code/
│   ├── data.py      # 边划分(85/5/10) + 负采样
│   ├── models.py    # GCN/GAT/SAGE/GIN 编码器 + MLP 边打分器
│   ├── utils.py     # 参数 / AUC 评估 / 计时 / 结果 CSV
│   ├── train.py     # 训练（--mode full|sample）
│   ├── run_experiments.py
│   ├── plot_results.py   # 结果图 → results/figures/
│   └── results/
└── README.md
```

## 运行

```bash
# 全图训练（验证集 AUC 选优，报测试 AUC）
python train.py --dataset cora --model gcn --mode full --epochs 300

# 邻居采样子图训练
python train.py --dataset cora --model gcn --mode sample \
    --epochs 200 --batch-size 512 --fanouts 10,10

# 批量（数据 集×模型×方式）
python run_experiments.py --datasets cora,citeseer,flickr \
    --models gcn,gat,sage,gin --modes full,sample

# 出图
python plot_results.py
```

主要参数：`--lr`(默认0.005)、`--layers`、`--hidden`、`--patience`、
`--eval-every`；采样模式还有 `--batch-size`、`--fanouts`。

## 说明
- 评测指标为 **ROC-AUC**：验证/测试各用 1:1 的“正边 + 随机负边”，
  用训练子图整图编码打分。
- 负采样：训练时随机换尾（全图模式在全体节点中随机；采样模式为批内换尾）；
  验证/测试负样本已排除图中已有边。
- 训练子图会补自环，避免 0 入度节点导致消息传递异常。

## 本机快速验证结果（CPU，cora，25 轮，演示用）
| 模型 | mode | val AUC | test AUC |
| --- | --- | --- | --- |
| GCN | full | 0.59 | 0.58 |
（完整版请按上面脚本跑 150~300 轮，AUC 会明显上升）
