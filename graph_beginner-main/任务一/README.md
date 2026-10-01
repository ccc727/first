# 任务一：图神经网络节点分类

基于 **DGL**（Deep Graph Library）实现图上的 **节点分类**（Node Classification）。
在 **Cora / Citeseer / Flickr(ogbn-flickr)** 三个数据集上测试 **GCN / GAT /
GraphSAGE / GIN** 四种主流 GNN 模型，并对比 **全图训练** 与 **邻居采样子图训练**
（DGL `NeighborSampler` + `DataLoader`）的性能与运行时间。

---

## 一、目录结构

```
任务一/
├── data/                       # 数据集存放目录（脚本会自动下载，见 data/README.md）
├── code/
│   ├── models.py               # GCN / GAT / GraphSAGE / GIN 模型实现（DGL）
│   ├── data.py                 # 数据集加载与预处理
│   ├── utils.py                # 公共工具（参数/评估/计时/结果落盘）
│   ├── train_full.py           # 全图（整批）训练 + 测试
│   ├── train_sampling.py       # NeighborSampler 子图采样训练 + 测试
│   ├── run_experiments.py      # 批量跑 数据集×模型×训练方式（可扫学习率/层数）
│   ├── analysis.py             # 汇总 results.csv，输出分析表格
│   └── results/                # 自动生成：results.csv（每次运行追加一行）
└── README.md                   # 本说明
```

## 二、环境安装

Python 3.10 ~ 3.12。项目根目录提供了 `requirements.txt`。

> ⚠️ **重要（Windows 用户）**：PyPI 上 `dgl==2.2.1` 的 Windows wheel 内置的
> `graphbolt` 编译库最高只对应 `torch 2.3.0`，因此**请先安装 torch 2.3.0 再装 DGL**。
> Linux 环境可参考 [DGL 官方安装页](https://www.dgl.ai/pages/start.html)选择与
> torch 匹配的安装方式（本代码对 torch 2.1~2.5 均兼容，Windows 下建议 2.3.0）。

```bash
# CPU 环境（Windows / Linux 均可）
pip install torch==2.3.0
pip install -r requirements.txt        # 含 dgl==2.2.1、ogb 等

# GPU 环境（示例 CUDA 12.1；Linux）
# pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121
```

## 三、数据集说明

| 数据集 | 来源 | 节点/边 | 特征维 | 类别 | 划分 |
| --- | --- | --- | --- | --- | --- |
| Cora | DGL 内置 | 2708 / 5429(双向) | 1433 | 7 | 官方 140/500/1000 |
| Citeseer | DGL 内置 | 3327 / 4732(双向) | 3703 | 6 | 官方 120/500/1000 |
| Flickr | DGL 内置 `FlickrDataset`(GraphSAINT) | 89250 / 899756(无向) | 500 | 7 | 官方随机 50/25/25 |

- 三个数据集均使用 DGL 自带的标准划分掩码；
- 为保证 GCN 对称归一化有意义，加载时统一对图做“无向化 + 每节点一条自环”
  （可用 `--no-self-loop` 关闭）。


首次运行脚本会自动把数据下载到 `data/` 目录；如网络受限，可手动下载后放入
`data/`（详见 `data/README.md`）。

## 四、训练与测试脚本

以下命令均在 `任务一/code/` 目录下执行（推荐：`cd code`）。

### 1. 全图训练

```bash
python train_full.py --dataset cora --model gcn
python train_full.py --dataset citeseer --model gat --lr 0.005
python train_full.py --dataset flickr --model sage --epochs 30 --verbose
```

### 2. 邻居采样子图训练（Sampler）

```bash
python train_sampling.py --dataset cora --model gcn --batch-size 512 --fanouts 10,10
python train_sampling.py --dataset flickr --model sage --batch-size 1000 --fanouts 15,10
```

参数说明（两种模式通用）：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--dataset` | cora / citeseer / flickr | 必填 |
| `--model` | gcn / gat / sage / gin | 必填 |
| `--lr` | 学习率 | 0.01 |
| `--layers` | GNN 层数 | 2 |
| `--hidden` | 隐藏层维度 | 64 |
| `--dropout` | dropout | 0.5 |
| `--epochs` | 训练轮数（cora/citeseer 默认 300，flickr 默认 60） | 自动 |
| `--patience` | 早停耐心（cora/citeseer 默认 100，flickr 默认 20） | 自动 |
| `--device` | auto / cpu / cuda:0 | auto |
| `--num-heads` | GAT 头数 | 8 |
| `--no-self-loop` | 关闭自环 | 关 |

采样模式专属参数：`--batch-size`（默认 512）、`--fanouts`（默认 10,10，逗号分隔，
个数=层数）、`--num-workers`（默认 0）。

### 3. 批量实验（数据集 × 模型 × 训练方式）

```bash
# 小规模验证：cora/citeseer × 4 模型 × 2 种训练方式（约 16 个实验）
python run_experiments.py --datasets cora,citeseer \
    --models gcn,gat,sage,gin --modes full,sample

# 全量实验（含 flickr，时间较长，建议 GPU）
python run_experiments.py --datasets cora,citeseer,flickr \
    --models gcn,gat,sage,gin --modes full,sample

# 参数扫描：学习率 × 层数（用于报告中的参数影响分析）
python run_experiments.py --datasets cora --models gcn --modes full \
    --sweep-lr 0.001,0.005,0.01,0.05 --sweep-layers 2,3
```

### 4. 结果汇总与对比

每次运行都会向 `code/results/results.csv` 追加一行，字段包括：
`dataset / model / mode / seed / lr / layers / hidden / dropout /
fanouts / batch_size / n_params / best_val_acc / best_epoch / test_acc /
train_epochs / avg_epoch_sec / train_sec / eval_sec / device / note`。

```bash
python analysis.py     # 打印 3 张表：模型×训练方式对比 / 参数扫描 / full-vs-sample 差值
```

## 五、计时口径说明（重要）

- `avg_epoch_sec` = 每个 epoch **仅训练（前向+反向）的平均耗时**，不含验证；
- `train_sec` = 训练部分总耗时（不含评估）；
- `eval_sec` = 整个训练过程中验证/测试评估的总耗时；
- 两种训练方式每轮结束都用**同一整图前向**做验证（并据此早停、选最优模型），
  保证“精度对比”与“训练耗时对比”互不干扰、公平可比；
- 测试集准确率 = 验证集最优模型的测试准确率。

## 六、主要结论（供报告参考）

1. **模型对比**：在 Cora/Citeseer 上 GAT 与 GCN 通常相当或略高，GraphSAGE 与 GIN
   稍低；在 Flickr（大数据、强特征）上四种模型差距较小，SAGE/GIN 训练更快。
2. **全图 vs 采样**：
   - 精度：采样训练在小数据集（Cora/Citeseer）上与全图训练相当（差约 0~2%），
     在 Flickr 上通过调大 batch/fanout 可逼近全图；
   - 时间：Cora/Citeseer 图太小，全图训练通常更快；Flickr 上采样训练每轮
     耗时显著更短、显存占用固定，是大规模图训练的主流方案。
3. **学习率**：lr 过小收敛慢，过大易发散/震荡；GAT 通常需要比 GCN 更小的
   学习率（0.005 附近），GCN/SAGE 在 0.01 附近表现较好。
4. **层数**：2 层是引文网络上的常见最优；加深到 3~4 层时，无残差/归一化辅助的
   模型容易出现性能下降或收敛变慢（可对比 `--layers 2/3` 观察过平滑）。
5. **评估方式**：使用“验证集最优 + 早停”选模型，避免直接用测试集调参。

## 七、常见问题

| 问题 | 处理 |
| --- | --- |
| `Cannot find DGL C++ graphbolt library ... torch_2.x.dll` | torch 与 dgl 版本不匹配，Windows 下安装 `torch==2.3.0` 后重装 dgl |
| 数据集下载慢 / 失败 | 参见 `data/README.md` 手动下载，或配置代理 |
| 采样训练 loss 波动大 | 增大 `--batch-size`、`--fanouts`，或调小学习率 |
| 对比显存占用 | 采样训练每个 batch 图大小固定，可用小 batch 在单卡训练超大图 |

