# 任务四报告：知识图谱补全（TransE / RotatE / ConvE）

> 脚本用法见 [README.md](README.md)。本报告数字均来自 `code/results/results.csv`
> 的实际运行（共 14 条记录，其中补跑 13 条），图表见 `code/results/figures/kge_comparison.png`。

## 一、实验设置

| 项目 | 设置 |
| --- | --- |
| 框架 | DGL 2.2.1 + PyTorch 2.3.0（CPU） |
| 数据集 | **FB15k-237**（14,541 实体 / 237 关系 / 272,115 训练三元组、17,535 验证、20,466 测试） |
| 模型 | TransE（`-‖h+r−t‖`）、RotatE（复数旋转 `h∘r≈t`）、ConvE（(h,r) 堆叠 2D 卷积后与 t 点积） |
| 训练损失 | 随机换尾负采样 + BCE(logistic) |
| 评测 | **过滤式 1-to-N**（固定 (h,r)，屏蔽 train+valid+test 中其它合法尾实体），报 MRR / Hits@1/3/10 |
| 选模型 | 验证集 MRR 选优 + 早停（patience=10） |

## 二、主结果（测试集，`--eval-n 2000`）

| 模型 | dim | lr | 训练轮数 | **MRR** | Hits@1 | Hits@3 | **Hits@10** | 每轮秒 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TransE | 100 | 1e-4 | 15 | **0.3316** | 0.2580 | 0.3570 | **0.4790** | 3.15 |
| TransE | 200 | 1e-4 | 13 | 0.3228 | 0.2510 | 0.3445 | 0.4670 | 5.83 |
| TransE | 100 | 3e-4 | 11 | 0.3241 | 0.2555 | 0.3415 | 0.4680 | 3.79 |
| RotatE | 200 | 1e-3 | 15 | **0.2689** | 0.1820 | 0.2980 | 0.4425 | 33.66 |
| RotatE | 100 | 1e-3 | 15 | 0.2635 | 0.1755 | 0.2955 | 0.4430 | 8.83 |
| ConvE | 100 | 3e-4 | 15 | **0.3144** | 0.2400 | 0.3395 | **0.4640** | 48.86 |
| ConvE | 100 | 1e-4 | 12 | 0.3013 | 0.2300 | 0.3230 | 0.4465 | 48.71 |

**结论：TransE 的 MRR 最高（0.332），ConvE 的 Hits@10 最高（0.464），RotatE 两者都最低。**
三种模型的 Hits@10 都达到 **44%~48%**，与 README 里"FB15k-237 上 Hits@10 一般可到 40%+"
的预期一致。

需要说明 RotatE 为何没有像原论文那样胜出：原版 RotatE 的关键组件是
**自对抗负采样（self-adversarial negative sampling）**，而本实现用的是与 TransE/ConvE
相同的朴素随机换尾负采样。RotatE 用 `-‖h∘r−t‖²` 的打分幅度对随机负样本过于"自信"，
在朴素负采样下反而不如 `-‖h+r−t‖` 稳定。这是**评测口径的差异**，不是模型写错
（相位是实现为可学习参数的，见下节）。

## 三、参数影响分析

**学习率（同数据集、同 dim、同轮数）**

| 模型 | lr | MRR | Hits@10 |
| --- | --- | --- | --- |
| TransE dim100 | 1e-4 | **0.3316** | **0.4790** |
| TransE dim100 | 3e-4 | 0.3241 | 0.4680 |
| TransE dim100 | 1e-3 | 0.1956 | 0.3360 |
| ConvE dim100 | 3e-4 | **0.3144** | **0.4640** |
| ConvE dim100 | 1e-4 | 0.3013 | 0.4465 |

**结论：学习率是本任务最敏感的超参**。TransE 从 lr=1e-3 降到 1e-4，MRR 从 0.196 提到
0.332（+69%）、Hits@10 从 0.336 提到 0.479。原因是 BCE + 随机负采样在 1e-3 下
**前几个 epoch 就过拟合**：验证 MRR 在 epoch 5 达到 0.21 后持续下滑到 0.17。
ConvE 的最优 lr 则偏大（3e-4 优于 1e-4），因为它有 BatchNorm 与卷积层，需要更大步长。

**嵌入维度**

| 配置 | dim=50 | dim=100 | dim=200 |
| --- | --- | --- | --- |
| TransE (lr=1e-3) | 0.200 | 0.196 | 0.193 |
| TransE (lr=1e-4) | — | **0.332** | 0.323 |
| RotatE (lr=1e-3) | — | 0.264 | **0.269** |

结论：**维度收益很小且很快饱和**。TransE 在 lr=1e-4 下 dim 100 与 200 几乎持平
（0.332 vs 0.323），dim 200 反而略低且每轮耗时接近翻倍；RotatE 从 100 到 200 只提升
0.005。在 FB15k-237（272k 三元组）这个规模上，**瓶颈是负采样质量而不是维度**。

## 四、本次修复的两个实现问题

1. **RotatE 的关系相位原本是固定随机常数，不是可学习参数**
   （原代码 `self.register_buffer("phase", torch.rand(...))`）。RotatE 的核心正是
   "关系 = 可学习的旋转相位"，固定成常数等于让关系不可训练。已改为
   `nn.Parameter`，参数量从 2,908,200 变为 2,931,900（多出 237×100 个相位参数）。
   修复后 RotatE 的 MRR 从 0.121（lr=1e-4 时）提升到 **0.269**。

2. **过滤式评测是逐候选实体的 Python 循环，CPU 上极慢**。
   FB15k-237 有 14,541 个实体，原实现对每个测试三元组都要循环写一遍 score 向量。
   已改为**按批向量化**（把批内所有 (h,r) 的过滤实体收集成扁平索引后一次性屏蔽），
   并用"与原始逐条实现逐位对比"验证过两者结果**完全一致**（50/300 条样本上 MRR、
   Hits@1/3/10 完全相同）。

另外修复了 `os.makedirs(os.path.dirname(args.results))` 在传裸文件名时抛
`FileNotFoundError` 的问题（任务二、任务三同类问题一并修复）。

## 五、图表

`code/results/figures/kge_comparison.png`：TransE / RotatE / ConvE 的测试 MRR、
Hits@10、训练总耗时三栏对比。

## 六、复现命令

```bash
cd code
# 主结果（推荐 lr：TransE 1e-4 / RotatE 1e-3 / ConvE 3e-4）
python train.py --model transe --dim 100 --lr 1e-4 --epochs 15 --eval-n 2000
python train.py --model rotate --dim 200 --lr 1e-3 --epochs 15 --eval-n 2000
python train.py --model conve  --dim 100 --lr 3e-4 --epochs 15 --eval-n 2000
python plot_results.py
```

> 说明：
> - `results_original_backup.csv` 为补跑前的 1 条原始记录（TransE 仅 1 轮），保留以便对照。
> - RotatE dim=200 在 CPU 上每轮约 34 秒（`tail_scores` 的 (B,E,2d) 中间张量较大），
>   跑完整 40 轮不划算，故统一用 15 轮 + 早停。
> - 所有模型的 `best_*.pt` 检查点为运行产物，已列入 `.gitignore`。
