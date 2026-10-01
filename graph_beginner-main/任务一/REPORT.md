# 任务一报告：图神经网络节点分类

> 脚本用法见 [README.md](README.md)。本报告所有数字均来自 `code/results/results.csv`
> 的实际运行结果（共 31 条实验记录），可用 `python analysis.py` 复现表格。

## 一、实验设置

| 项目 | 设置 |
| --- | --- |
| 框架 | DGL 2.2.1 + PyTorch 2.3.0（CPU） |
| 数据集 | Cora (2708 节点/13264 边/1433 维)、Citeseer (3327/12431/3703)、Flickr (89250/989006/500) |
| 模型 | GCN / GAT / GraphSAGE / GIN |
| 训练方式 | 全图（full-batch）vs 邻居采样子图（`NeighborSampler`, fanouts 10,10 或 15,10） |
| 公共超参 | hidden=64, layers=2, dropout=0.5, lr=0.01, Adam(wd=5e-4), seed=0 |
| 轮数/早停 | Cora/Citeseer 300 轮/patience 100；Flickr 60 轮/patience 20（采样 30 轮） |
| 选模型 | 验证集最优 + 早停；报对应模型的**测试集**准确率 |

覆盖度：3 数据集 × 4 模型 × 2 训练方式 = **24 组**，另加学习率/层数扫描 7 组。

## 二、模型对比（测试准确率）

| 数据集 | GCN | GAT | GraphSAGE | GIN |
| --- | --- | --- | --- | --- |
| Cora (full) | 0.804 | **0.821** | 0.796 | 0.751 |
| Cora (sample) | 0.791 | **0.813** | 0.788 | 0.754 |
| Citeseer (full) | **0.712** | **0.712** | 0.706 | 0.572 |
| Citeseer (sample) | 0.694 | **0.711** | 0.674 | 0.654 |
| Flickr (full) | **0.498** | 0.423 | 0.434 | 0.445 |
| Flickr (sample) | 0.423 | 0.507 | **0.510** | 0.507 |

结论：
1. **引文网络（Cora/Citeseer）上 GAT ≈ GCN > GraphSAGE > GIN**。GAT 的注意力在 Cora 上
   最好（0.821），GCN 紧随其后；GIN 在 Citeseer 全图上明显偏低（0.572），与 GIN 的 sum
   聚合 + 无归一化在异质引文图上易过拟合/难收敛有关。
2. **Flickr 上排序反转**：全图训练 GCN 最好（0.498），但 GAT/SAGE/GIN 全图只有 0.42~0.45；
   改用采样训练后 GAT/SAGE/GIN 都升到 0.51 左右反超 GCN。说明大图上"每轮只更新一次整图"
   对注意力/聚合类模型的优化并不充分。

## 三、全图训练 vs 采样训练（核心对比）

**精度差 Δ = sample − full**：

| 数据集 | GCN | GAT | GraphSAGE | GIN |
| --- | --- | --- | --- | --- |
| Cora | −0.013 | −0.008 | −0.008 | **+0.003** |
| Citeseer | −0.018 | −0.001 | −0.032 | **+0.082** |
| Flickr | −0.075 | **+0.083** | **+0.076** | **+0.063** |

**每轮训练耗时（秒，仅前向+反向，不含验证）**：

| 数据集 | GCN full→sample | GAT full→sample | SAGE full→sample | GIN full→sample |
| --- | --- | --- | --- | --- |
| Cora | 0.016 → 0.010 | 0.017 → 0.015 | 0.012 → 0.011 | 0.014 → 0.013 |
| Citeseer | 0.018 → 0.012 | 0.030 → 0.017 | 0.023 → 0.017 | 0.037 → 0.016 |
| Flickr | 0.121 → 1.756 | 0.377 → 3.251 | 0.165 → 1.936 | 0.240 → 2.000 |

结论：
1. **小图（Cora/Citeseer）全图训练更划算**：图只有几千节点，整图前向本身就极快，
   采样反而引入了 DataLoader 与子图构造的额外开销（每轮 0.010~0.017s，与全图同量级），
   精度还普遍略低（Δ 多为负）。例外是 Citeseer-GIN，采样把 0.572 提到 0.654（+0.082），
   说明小图上采样带来的**正则化/随机性**对易过拟合的 GIN 有帮助。
2. **大图（Flickr，8.9 万节点/98.9 万边）上采样训练在精度上明显更好**：
   GAT/SAGE/GIN 分别 +0.083/+0.076/+0.063。这与"全图每轮只做一次梯度更新"有关——
   采样训练每轮有 ~45 个 mini-batch（44625/1000），梯度更新次数多两个数量级。
3. **但本机上采样训练每轮"更慢"**：Flickr 上采样每轮 1.76~3.25s，是全图（0.12~0.38s）的
   10~20 倍。原因是 CPU 上邻居采样的子图构造是 Python/DGL 侧开销，且 batch=1000 时要跑
   约 45 个 batch 才算一个 epoch；全图则是一次稀疏矩阵乘。
   注意这不否定采样训练的价值：它**每轮显存/内存占用固定**，可扩展到单卡放不下的大图，
   而全图训练在大图上会直接 OOM。本机是 64GB 内存的 CPU，全图才"跑得动"。

## 四、参数影响分析

**学习率扫描（Cora + GCN + full，layers=2）**

| lr | 0.001 | 0.005 | 0.01 | 0.05 |
| --- | --- | --- | --- | --- |
| test acc | 0.815 | **0.816** | 0.803 | 0.807 |
| 实际训练轮数 | 300 | 192 | 159 | 148 |

结论：lr 在 0.001~0.005 时精度最高且曲线最稳；0.01（默认）略低；0.05 虽然精度没崩，
但收敛所需的评估轮次更少（148 轮即早停），说明步长过大时验证集准确率更早进入平台/震荡。
本任务默认 lr=0.01 是个略偏大的选择，**0.005 是更好的默认值**。

**层数扫描（Cora + GCN + full，lr=0.01）**

| layers | 2 | 3 | 4 |
| --- | --- | --- | --- |
| test acc | **0.807** | 0.798 | 0.775 |
| 每轮秒 | 0.016 | 0.016 | 0.021 |

结论：**2 层最优，加深持续掉点**（4 层 −0.032）。这是典型的**过平滑（over-smoothing）**：
Cora 是异质引文图，多层邻域平均会让不同类节点的表示趋同。同时加深还增加了每轮耗时。

## 五、图表

`code/results/figures/` 下：
- `test_acc_comparison.png`：各数据集 模型×训练方式 的测试准确率对比
- `epoch_time_comparison.png`：平均每轮训练耗时对比
- `total_time_vs_acc.png`：训练总耗时 vs 准确率散点
- `param_sweep.png`：学习率 / 层数扫描曲线

## 六、复现命令

```bash
cd code
# 全量：3 数据集 × 4 模型 × 2 训练方式（本机 CPU 约 15 分钟）
python run_experiments.py --datasets cora,citeseer,flickr \
    --models gcn,gat,sage,gin --modes full,sample
# 参数扫描
python run_experiments.py --datasets cora --models gcn --modes full \
    --sweep-lr 0.001,0.005,0.01,0.05
python run_experiments.py --datasets cora --models gcn --modes full \
    --sweep-layers 2,3,4
python analysis.py && python plot_results.py
```

> 说明：`results_original_backup.csv` 是补实验前仓库里原有的 14 条记录，保留以便对照。
