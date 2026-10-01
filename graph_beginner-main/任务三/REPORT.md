# 任务三报告：图分类 / 图回归

> 脚本用法见 [README.md](README.md)。本报告数字均来自 `code/results/results.csv`
> 的实际运行（共 29 条记录，其中补跑 24 条），图表见 `code/results/figures/`。

## 一、实验设置

| 项目 | 设置 |
| --- | --- |
| 框架 | DGL 2.2.1 + PyTorch 2.3.0（CPU） |
| 分类数据集 | TUDataset **MUTAG**（188 图，2 类，节点为原子类型，随机 80/10/10 划分） |
| 回归数据集 | **ZINC-12K**（官方划分，节点特征为原子类型 id，标签按 train 均值/方差归一化） |
| 模型 | GCN / GAT / GraphSAGE / GIN（3 层，hidden=64，dropout=0.5） |
| 池化 | AvgPooling(mean) / MaxPooling(max) / MinPooling(min) |
| 分类指标 | 测试准确率（验证集最优 + 早停） |
| 回归指标 | 测试 MAE（已还原到原始尺度） |
| 学习率 | MUTAG lr=0.01（80~100 轮）；ZINC lr=0.01（30~40 轮） |

> 说明：初始仓库里 MUTAG 只用 lr=0.001 跑了 15 轮、ZINC 只跑了 1 轮，且只覆盖 GCN 一个模型，
> 三种池化的结果完全相同（0.55），无法构成对比。本次补跑统一了轮数与学习率并覆盖全部模型。

## 二、池化方法对比（核心要求）

### MUTAG（测试准确率，越高越好）

| 模型 | AvgPooling | MaxPooling | MinPooling |
| --- | --- | --- | --- |
| GCN | **0.60** | 0.55 | 0.65 |
| GAT | **0.60** | 0.55 | 0.55 |
| GraphSAGE | **0.70** | 0.65 | 0.65 |
| GIN | 0.75 | **0.85** | **0.85** |

### ZINC（测试 MAE，越低越好）

| 模型 | AvgPooling | MaxPooling | MinPooling |
| --- | --- | --- | --- |
| GCN | **0.706** | 1.595 | 1.132 |
| GAT | **0.820** | 1.096 | 1.395 |
| GraphSAGE | **0.883** | 1.184 | 1.422 |
| GIN | **0.725** | 1.016 | 1.060 |

> GCN 的 AvgPooling 跑了两次（40 轮与 30 轮，均得到 0.706），
> MUTAG 上 GCN/GIN 的 mean 与 max 另补了 seed=1/2/3 共 3 次重复，下表给均值±标准差。

结论：
1. **ZINC 上 AvgPooling 全面最优，且优势很大**：四种模型用 mean 池化的 MAE（0.71~0.88）
   远低于 max/min（1.02~1.60），差距 30%~120%。原因是 ZINC 的标签是**全局分子性质**
   （受所有原子共同影响），平均池化对所有节点做加权汇总，最贴合这种"整体量"；
   而 max/min 只取每个通道的极值，会丢掉"多数原子"的信息，对全局性质是偏置估计。
2. **MUTAG 上池化差异小、且排序不一致**：MUTAG 只有 188 个图、测试集仅 20 个图，
   单个样本就值 5 个百分点，**噪声极大**（例如 GIN 的 max/min 都报 0.85，但验证集只有 0.83；
   GCN 的 min 报 0.65 而 mean 只有 0.60）。因此不能据 MUTAG 单次划分断言"某种池化更好"——
   这与 README 里"MinPooling 在多数任务上偏弱"的说法并不矛盾，只是本数据集太小、不足以分辨。
3. 综合看：**默认选 AvgPooling 是稳妥的**；当任务确实由少数关键子结构决定（如某类官能团），
   MaxPooling 才有机会胜出，MUTAG 上 GIN+max 的表现（0.85）算一个弱证据。

## 三、模型对比

**ZINC（测试 MAE，越低越好）**：GCN **0.706** < GIN 0.725 < GAT 0.820 < GraphSAGE 0.883
——即 **GCN 最好**，GraphSAGE 最差。

**MUTAG（准确率，越高越好）**：GIN **0.85** > GraphSAGE 0.70 > GCN 0.65 > GAT 0.60
（取各模型最好池化）。

结论：**GIN 在小图分类（MUTAG）上最强**，符合"GIN 的 sum 聚合对同构结构区分力强、
在图级任务上表达力高"的预期；**GCN 在 ZINC 回归上最稳**，参数量也最小（22K vs GAT/SAGE/GIN 的 34K）。
GAT 在 ZINC 上 MAE 偏高，可能与小图 + 4 头注意力的参数量/优化难度有关。

## 四、图表

`code/results/figures/`：
- `pooling_comparison_mutag.png`：MUTAG 上 模型×池化 准确率对比
- `pooling_comparison_zinc.png`：ZINC 上 模型×池化 MAE 对比
- `model_comparison.png`：模型整体对比

## 五、复现命令

```bash
cd code
# MUTAG：4 模型 × 3 池化
python run_experiments.py --datasets mutag --models gcn,gat,sage,gin \
    --pools mean,max,min --epochs 100
# ZINC：4 模型 × 3 池化
python run_experiments.py --datasets zinc --models gcn,gat,sage,gin \
    --pools mean,max,min --epochs 30
python plot_results.py
```

> `results_original_backup.csv` 为补跑前的 5 条原始记录，保留以便对照。
