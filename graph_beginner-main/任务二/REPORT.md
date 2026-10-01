# 任务二报告：图上的链路预测

> 脚本用法见 [README.md](README.md)。本报告数字均来自 `code/results/results.csv`
> 的实际运行（共 30 条记录，其中补跑 24 条），图表见 `code/results/figures/`。

## 一、实验设置

| 项目 | 设置 |
| --- | --- |
| 框架 | DGL 2.2.1 + PyTorch 2.3.0（CPU） |
| 数据集 | Cora / Citeseer / Flickr（复用任务一 loader） |
| 边划分 | 全部边随机 85% / 5% / 10% 作 train/valid/test；valid、test 各配 1:1 负样本（排除图中已有边） |
| 模型 | 编码器 GCN / GAT / GraphSAGE / GIN + 打分器「两端嵌入拼接 → MLP」 |
| 训练方式 | **全图 full**（每轮训练子图整图编码一次，正边分小批 + 随机换尾负采样）<br>**采样 sample**（NeighborSampler 采 2 跳子图，批内换尾构造负样本） |
| 评价指标 | **ROC-AUC**；两种模式统一用"训练子图整图编码 + 验证集 AUC"选模型，再报测试 AUC |
| 公共超参 | hidden=64, layers=2, lr=0.005, seed=0 |
| 轮数 | Cora/Citeseer：full 200 / sample 100；Flickr：full 60 / sample 30 |

覆盖度：3 数据集 × 4 模型 × 2 训练方式 = **24 组**。

> 说明：补跑前仓库里只有 4 条 cora-full（25 轮，AUC 0.58~0.80）+ 1 条临时 sample 记录，
> Citeseer/Flickr 完全没跑、sample 模式一次没跑过。本次全部补齐。

## 二、模型对比（测试 AUC）

| 数据集 | 模式 | GCN | GAT | GraphSAGE | GIN |
| --- | --- | --- | --- | --- | --- |
| Cora | full | 0.898 | 0.955 | **0.971** | 0.921 |
| Cora | sample | 0.980 | 0.978 | **0.986** | 0.932 |
| Citeseer | full | 0.894 | 0.940 | **0.967** | 0.927 |
| Citeseer | sample | 0.968 | 0.973 | **0.985** | 0.945 |
| Flickr | full | **0.810** | 0.588 | 0.528 | 0.635 |
| Flickr | sample | 0.835 | **0.852** | 0.718 | 0.744 |

结论：
1. **Cora/Citeseer 上 GraphSAGE 最好**（0.97~0.99），GAT 紧随，GIN 略低。GraphSAGE 的
   均值聚合在"预测两节点是否有边"这种需要平滑邻域表示的任务上表现稳定。
2. **Flickr 上 GCN/GAT 明显优于 SAGE/GIN**。注意这与任务一（节点分类）的排序不同——
   链路预测依赖**成对**表示，SAGE/GIN 在大图上的聚合更容易把不同节点的表示"抹平"，
   反而不如 GCN 的对称归一化。
3. 全图模式下 Flickr 的 GCN 最好（0.810），GAT 只有 0.588，说明大图上"每轮一次整图
   梯度"对注意力模型的优化很不充分。

## 三、全图训练 vs 采样训练（核心对比）

**AUC 差 Δ = sample − full**：

| 数据集 | GCN | GAT | GraphSAGE | GIN |
| --- | --- | --- | --- | --- |
| Cora | **+0.082** | **+0.023** | **+0.015** | **+0.011** |
| Citeseer | **+0.074** | **+0.032** | **+0.018** | **+0.018** |
| Flickr | **+0.025** | **+0.264** | **+0.190** | **+0.109** |

**每轮训练耗时（秒）**：

| 数据集 | GCN full→sample | GAT full→sample | SAGE full→sample | GIN full→sample |
| --- | --- | --- | --- | --- |
| Cora | 0.07 → 0.68 | 0.11 → 0.97 | 0.12 → 0.81 | 0.12 → 0.81 |
| Citeseer | 0.11 → 0.71 | 0.16 → 0.99 | 0.16 → 0.88 | 0.13 → 0.81 |
| Flickr | 15.2 → 84.1 | 20.9 → 86.8 | 23.6 → 58.7 | 19.9 → 54.1 |

结论：
1. **采样训练在所有 12 个组合上都优于全图训练**（Δ 全为正），这是本任务最干净的结论。
   与小图节点分类（任务一）不同，链路预测的采样优势在小图上也很明显
   （Cora +0.082、Citeseer +0.074）——因为采样让每个 epoch 产生数十个 mini-batch，
   梯度更新次数提升了一到两个数量级。
2. **Flickr 上提升最大**：GAT +0.264（0.588 → 0.852）、SAGE +0.190、GIN +0.109。
   全图训练在 Flickr 上连验证集 AUC 都上不去（GAT 60 轮只到 0.587），而采样训练 30 轮
   就达到 0.85，说明**大图上全图训练的优化效率是瓶颈**，不是模型容量。
3. **代价是每轮耗时增加约 3~10 倍**（采样要在 CPU 上做邻居采样与子图构造）。
   但采样训练的内存占用与图规模无关，可扩展到单机放不下的大图；全图训练在
   Flickr 上单次整图编码已需 15~24 秒且需要把 8.9 万节点特征全部驻留内存。

## 四、参数影响分析

**学习率（Cora + GCN + full，150 轮）**

| lr | 0.001 | 0.005 | 0.01 |
| --- | --- | --- | --- |
| 验证 AUC | 0.7737 | 0.8723 | **0.8937** |
| 测试 AUC | 0.7782 | 0.8816 | **0.9140** |

结论：**链路预测对学习率比节点分类更敏感**。lr 从 0.001 提到 0.01，测试 AUC 从 0.778
升到 0.914（+0.136）。原因是链路预测的打分器只靠点积/MLP 区分正负边，损失信号比
节点分类弱，步长过小时 150 轮仍未收敛（lr=0.001 时验证 AUC 在第 150 轮还在爬升）。
本任务默认 lr=0.005 偏保守，**0.01 更好**。

## 五、图表

`code/results/figures/`：
- `test_auc_comparison.png`：各数据集 模型×训练方式 测试 AUC 对比
- `epoch_time_comparison.png`：平均每轮训练耗时对比
- `total_time_vs_auc.png`：训练总耗时 vs AUC 散点

## 六、已修复的问题

本次补跑同时修掉了脚本的一个缺陷：`train.py` 里 `os.makedirs(os.path.dirname(args.results))`
在 `--results` 传裸文件名时会因 `dirname('')` 抛 `FileNotFoundError`，已改为
`os.path.dirname(os.path.abspath(...))`（任务三、任务四同类问题一并修复）。

## 七、复现命令

```bash
cd code
# 全量：3 数据集 × 4 模型 × full/sample（本机 CPU 约 3 小时，Flickr 占大头）
python run_experiments.py --datasets cora,citeseer,flickr \
    --models gcn,gat,sage,gin --modes full,sample
# 学习率扫描
for lr in 0.001 0.005 0.01; do
  python train.py --dataset cora --model gcn --mode full --lr $lr --epochs 150
done
python plot_results.py
```

> `results_original_backup.csv` 为补跑前的 4 条原始记录，保留以便对照。
