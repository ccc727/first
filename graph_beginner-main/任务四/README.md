# 任务四：知识图谱补全（KGE）

参考任务说明中的 KGE 框架思想，实现并评测三种经典打分模型：
- **TransE**：`s = -||h + r - t||`
- **RotatE**：复数空间旋转 `h ∘ r ≈ t`
- **ConvE**：把 (h, r) 堆叠后用 2D 卷积编码，再与 t 点积

数据集：**FB15k-237**（DGL 内置，272k 训练三元组），
另有 `--dataset FB15k/wn18` 可选（改 `data.py` 中对应类）。

## 目录
```
任务四/
├── data/            # DGL FB15k-237 数据，首次自动下载
├── code/
│   ├── data.py      # train/valid/test 三元组
│   ├── models.py    # TransE / RotatE / ConvE
│   ├── train.py     # 训练 + 过滤式评测(MRR, Hits@1/3/10)
│   ├── run_experiments.py
│   ├── plot_results.py  # 模型对比图
│   └── results/
└── README.md
```

## 运行
```bash
python train.py --model transe --epochs 50
python train.py --model rotate --epochs 50
python train.py --model conve  --epochs 30

# CPU 加速（截断训练三元组 / 评测集）
python train.py --model transe --epochs 20 --max-train 40000 --eval-n 500

python plot_results.py
```

## 评测口径
- 训练损失：随机负尾 + BCE（logistic）。
- 评测：**过滤式 1-to-N（固定 (h,r) 破环尾实体）**，报告 MRR、Hits@1/3/10；
  验证集选优（早停），最终报测试集指标。

## 结论（供报告参考）
- 同规模同维度下 TransE 训练最快但表达力有限；RotatE 能建模对称/反对称等
  关系模式，通常 MRR/Hits@10 更高；ConvE 参数交互更充分，小样本下往往最好。
- FB15k-237 上三种模型的 Hits@10 一般可到 40%+（充分训练后）。
