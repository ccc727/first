# 五个任务本地跑通报告（transformers_tasks）

- 机器：Windows / CPU（无 GPU）、Python 3.12.7
- 网络限制：只能访问 pypi 清华镜像与 download.pytorch.org；huggingface.co / hf-mirror / ModelScope 均不通
- 影响：无法下载 `ernie-3.0`、`bert-base-chinese`、`uie-base-zh`、`gpt2-chinese` 等官方预训练权重
- 解决方案：在 `_smoke/models/` 下自建**随机初始化 tiny BERT（2 层 hidden768）+ 字符级中文 tokenizer**，
  以及 tiny GPT2、2 分类情感模型，用于跑通每个任务的完整「数据→训练→评估→保存→推理」链路。

## 结果速览

| 任务 | 目录 | 冒烟训练结果 | 推理演示 |
|---|---|---|---|
| 1. 文本匹配 PointWise | `text_matching/supervised` | best F1 0.466（400 训练/150 dev）| 输出两句话的相似 logits |
| 1. 文本匹配 DSSM | 同上 | loss 0.50→0.24 | 输出各类别 cos 相似度排序 |
| 1. 文本匹配 Sentence Transformer | 同上 | best F1 0.432 | 输出匹配类别列表（[] 表示都低于阈值，属正常）|
| 2. 信息抽取 UIE | `UIE` | loss 0.29→0.04 | NER/SPO 抽取结果（空列表，模型置信度低）|
| 3. Prompt: PET | `prompt_tasks/PET` | best F1 0.33 | 输出 `['洗浴','洗浴']` |
| 3. Prompt: P-tuning | `prompt_tasks/p-tuning` | best F1 0.32 | `苹果→水果`（正确）、`笔记本→洗浴` |
| 4. 文本分类 BERT-CLS | `text_classification` | best F1 0.56 | `[4,2,5]=衣服/平板/酒店`（合理命中）|
| 5. RLHF Reward Model | `RLHF` | dev acc 0.26（上升中）| 正向句 -3.43 > 负向句 -6.27（方向正确）|
| 5. RLHF PPO+GPT2 | `RLHF` | 3 epoch reward 循环完成 | GPT2 续写推理跑通 |

## 关键修复（使旧代码在新环境可运行）

1. matplotlib 新版删除 `seaborn-darkgrid` → 各任务 `iTrainingLogger.py` 加 fallback。
2. `transformers` 新版移除 `top_k_top_p_filtering` → `trl/gpt2.py` 内置等价实现；
   `GPT2HeadWithValueModel` 增加 `GenerationMixin` 继承以获得 `.generate`。
3. `trl/core.py` `collections.Mapping` 在 Py3.12 已移除 → 改用 `collections.abc.Mapping`。
4. `evaluate`(0.4.x) 指标模块需联网 → 文本匹配三个训练脚本改用本地 sklearn 计算 P/R/F1。
5. `UIE/train.py` 原 `torch.load(pytorch_model.bin)` 属 bug → 改为 `AutoModel+UIE()` 组装。
6. torch 2.14 默认 `weights_only=True` → 本地整模型 `torch.load(..., weights_only=False)`。
7. 各推理脚本硬编码 `cuda:0/cuda:1` → 自动 `cpu`，checkpoint 路径支持环境变量覆盖。
8. `UIE` 依赖 `jieba` → 已安装；rich emoji 输出 → 运行设 `PYTHONIOENCODING=utf-8`。
9. 数据集 map 缓存锁冲突 → 每次训练单独设置 `HF_DATASETS_CACHE`。

## 产物位置

- `_smoke/models/tiny_bert_char` —— tiny BERT backbone + 字符级中文 tokenizer
- `_smoke/models/tiny_gpt2_char` —— tiny GPT2
- `_smoke/models/senti_2cls` / `senti_ft` —— PPO 用情感打分代理
- `_smoke/checkpoints/*` —— 各任务 checkpoint
- `_smoke/runs/*.log` —— 训练/推理日志

## 如何在有网/GPU 环境按官方方式复现

把 `--model`/`--pretrained_model` 换回官方模型名，device 换成 `cuda:0`，并采用原 `.sh` 参数即可。
（说明：由于离线只能用随机初始化小模型 + 少样本训练，指标远低于官方预训练模型，仅供链路验证。）
