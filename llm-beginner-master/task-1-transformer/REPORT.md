# 任务一：熟悉 Transformer — 完成报告

## 一句话总结
从零手写 scaled dot-product attention / MultiHeadAttention / Transformer encoder block，
在 ChnSentiCorp 中文情感二分类上达到 **dev 准确率 0.8392**（门槛 0.80，参考基线 0.85），
三项自检全部 `[通过]`。

## DoD Checklist
- [x] M1 手写 `scaled_dot_product_attention`，`attention_correctness` 通过（与官方误差 9.5e-07 < 1e-5）
- [x] M2 手写 `MultiHeadAttention` + `TransformerBlock`（Pre-LN，残差 + LayerNorm + FFN）
- [x] M3 ChnSentiCorp 训练分类器，dev 准确率 **0.8392** ≥ 0.80
- [x] M4 causal mask 自检通过（未来词元 V 改动，过去位置输出泄漏 diff = 0.0）
- [x] M5 `figures/` 下 4 张注意力热图（正样本 / 负样本 / 长句 / 同层多 head 对比）
- [ ] S1 head / 层数消融（本报告附层数消融，head 数消融留待扩展）
- [ ] S2 拆除 residual / LayerNorm 收敛对比（留待扩展）
- [ ] S3 dev 准确率 > 0.88（当前 0.8392，未达）
- [ ] S4 绝对 PE 换 RoPE 对比（留待任务二热身）

## 复现命令
```bash
python data/download.py                      # 数据（datasets>=5 时脚本型数据集受限，用 data/fallback_build.py）
python train.py --epochs 10 --batch 128 --lr 4e-4 --n-layers 8   # 训练
python eval/run.py                           # 自检 -> eval/result.json
python visualize.py                          # 注意力热图 -> figures/
```

## 实验记录与观察（层数消融）
模型统一：d_model=128、n_heads=4、d_ff=512、字符级分词、max_len=128、AdamW + OneCycle cosine。

| 配置 | epochs | best dev acc |
|---|---|---|
| 4 层 | 6 | 0.8117 |
| 6 层 | 8 | 0.8333 |
| **8 层** | **10** | **0.8392** |

训练在 RTX 3080 Ti 上仅需 ~5.5 s/epoch，全程 55 s。前 1-2 个 epoch 准确率 < 0.70 属于
warmup 阶段（lr 从 0 线性爬升），第 3 epoch 起快速收敛到 0.79 附近，之后每层/每 epoch
缓慢爬升到 0.84 并趋于平稳，8 层到第 9 epoch 达到峰值后开始轻微过拟合（train loss 仍降、
dev acc 掉 0.002），符合小数据 + 较小模型的典型曲线。

实验观察（约 300 字）：从热图能直观看到模型确实"看对了词"——正面样本（"物流很快，卖家
服务态度不错，东西质量非常好"）上，最后一个 attention 层的多个 head 把高权重集中到"很快、
不错、好、推荐"等情感词；负面样本则在"失望、差、不好、差评"上形成对角带。层数加深时
（4→8）dev 准确率稳定抬升约 2.8 个百分点，说明该数据规模下加深编码层带来的容量收益尚未
饱和；但超过约 9 epoch 后 train/dev gap 开始拉大，需要 early stop。causal mask 测试泄漏 diff
为 0，说明 -inf 填充的 mask 实现正确、未来信息没有回流。整体看，手写 Transformer 的数值
行为（attention 输出与 torch 官方一致到 1e-6 量级）与训练收敛都验证了实现的正确性。
