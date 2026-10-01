# 任务三：指令微调与偏好对齐（SFT + DPO）— 完成报告

## 一句话总结
在 Qwen2.5-0.5B 上手写 LoRA（低秩注入 + merge），实现 Qwen ChatML 模板与 loss masking
（只算 assistant 正文），跑通 SFT（自建中文指令集）与 DPO（自建偏好对）两阶段对齐，
三个自检项全部 `[通过]`，并能看到 base / SFT / DPO 在同一指令上的可观察差异。

## DoD Checklist
- [x] M1 手写 LoRA（`inject_lora` / `merge_lora`），`lora_param_count` 通过（可训练占比 0.109%）
- [x] M2 Qwen chat template + loss masking，`loss_masking` 通过（mock 多轮 -100 占比 0.556）
- [x] M3 SFT 产出非空 `ckpt/sft/`（LoRA adapter lora.pt），`sft_vs_base` 通过
- [x] M4 DPO 产出 `ckpt/dpo/`，`compare.py` 对比 base / SFT / DPO 输出
- [ ] S1/S2/S3/S4 加分项留待扩展

## 说明（数据适配）
仓库 README 推荐 MOSS-003-sft 数据；其 no-tools 版单个 jsonl.zip 达 2.6GB，with-tools 版
全部为带 `<|Commands|>` 的工具类样本，不适合纯指令 SFT。为在有限环境下跑通整条对齐链路，
使用**自建中文指令集**（深度学习概念 / Python / 通识，36 条 SFT + 54 条 DPO 偏好对，
见 `data/build_synthetic.py`），格式与 Qwen ChatML 一致。LoRA / 模板 / masking / SFT / DPO
机制均为手写实现，与本任务教学点一致。

## 自检结果
```
[通过] lora_param_count: trainable=540672 / 494573440, ratio=0.00109
[通过] loss_masking:     mask_ratio=0.556
[通过] sft_vs_base:      ckpt/sft 非空
```

## 训练与观察
- SFT：r=8, alpha=16，只训 q_proj/v_proj；60 epochs 后 loss ~5e-4（36 条小数据收敛快）。
- DPO：在 SFT 合并权重上冻结 reference（只 forward 预计算 logprob），policy 再注入 r=4 LoRA，
  β=0.1；40 epochs，dpo loss 0.30 → 0.0019。
- 关键修复：最初 SFT 的 labels 未做位移（用位置 i 预测 token[i]），模型学成"复制当前词"，
  生成退化成换行复读；改为标准位移 labels[i]=token[i+1] 后，SFT 能正确按指令输出答案。
  DPO 的 seq_logprob 也随之改为直接对"有效目标位置"取 log_softmax(logits[i])。
- 局限说明：0.5B 基座 + 54 条极小偏好数据 + r=4，DPO 后文本易出现标点复读退化；
  更大的偏好数据/rank 会显著改善，但本环境已足以演示 DPO 的机制与 ckpt/dpo 产物。
- 其余修复：LoRA 矩阵方向、注入后 A/B 对齐基座 device/dtype、merge 前向数值一致性验证等。

（对比样例见 `ckpt/compare_results.txt`：base 复读乱码 → SFT 高质量作答 → DPO 分布偏移）
