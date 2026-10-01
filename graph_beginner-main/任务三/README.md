# 任务三：图分类

基于 **DGL** 完成图级任务：
- **图分类**：TUDataset（MUTAG 等）
- **图回归**：ZINC-12K（分子性质预测）
模型：**GCN / GAT / GraphSAGE / GIN**；分析 **AvgPooling / MaxPooling /
MinPooling** 三种图级读出（池化）对性能的影响。

## 目录
```
任务三/
├── data/            # tu/(TUDataset) 与 zinc/ 数据，首次自动下载
├── code/
│   ├── data.py          # TU 随机 80/10/10；ZINC 官方划分 + 标签归一化
│   ├── models.py        # 节点ID嵌入 → GNN → 图级池化 → MLP 头
│   ├── train.py         # 训练（分类用准确率，回归用 MAE）
│   ├── run_experiments.py
│   ├── plot_results.py  # 池化对比图 + 模型对比图
│   └── results/
└── README.md
```

## 运行
```bash
python train.py --dataset mutag --model gin --pool mean --epochs 100
python train.py --dataset zinc --model gcn --pool max --epochs 100 --limit-train 2000

# 批量：数据集×模型×池化
python run_experiments.py --datasets mutag,zinc --models gcn,gat,sage,gin \
    --pools mean,max,min --epochs 80

python plot_results.py   # → results/figures/pooling_comparison.png 等
```

## 说明
- 分类指标为测试准确率；回归指标为 MAE（ZINC 按 train 均值/方差归一化训练，
  报告时还原为原始尺度）。
- `--pool mean|max|min` 分别对应 AvgPooling / MaxPooling / MinPooling。
- ZINC 若 CPU 训练较慢可加 `--limit-train 2000`（只取前 2000 个训练图）。

## 主要结论（供报告参考，也可自行复现）
- AvgPooling 通常最稳；MaxPooling 对“存在关键子结构”的分子任务有优势；
  MinPooling 在多数任务上偏弱（信息被“最不显著节点”主导）。
- GIN + sum 聚合对同构结构区分力强，在图级任务上常优于 GCN/GAT/SAGE；
  GAT 在小图分类上表达力强但参数量/时间略高。
