# 任务 6–9 完成报告

## 任务6：文本生成（Text Generation）— 已本机跑通

背景：官方用 `uer/t5-base-chinese-cluecorpussmall`（无法联网下载）。
方案：`_smoke/models/tiny_t5` = 随机初始化 tiny T5（d_model=128, 2层）+ 本地字符级 tokenizer（兼容仓库“Bert-tokenizer 风格”的 T5 代码）。

### 6.1 生成式问答（answer_generation / DuReaderQG）
- 数据：子集 300/80 条（`_smoke/data/qg_*.json`）
- 命令：`python train.py --pretrained_model _smoke/models/tiny_t5 --batch_size 4 --max_source_seq_len 96 --max_target_seq_len 32 --learning_rate 3e-4 --num_train_epochs 2 --device cpu`
- 结果：loss 8.52 → 6.08，BLEU-4 0→0.0011（小模型少样本，指标仅代表链路正常）
- 推理：`inference.py`（已适配 CPU/路径环境变量）输出 `Q/C/A`（tiny 模型只学出“答案：”前缀）

### 6.2 Mask→Fill Filling 模型（data_augment/filling_model）
- 数据：子集 400/80 行（`_smoke/data/fm_*.tsv`）
- 命令同上（tiny T5 复用，`max_source_seq_len 128`）
- 结果：loss 8.95 → 8.23，BLEU-4 0；推理管线输出还原片段（模型未收敛故为空串）

补充：`UEI/web_da.py` 中引用 filling 模型路径，训练出可用模型后填入即可实现 Mask Then Fill 数据增强。

---

## 任务9：工具类（tokenizer_viewer）— 已本机启动验证

- 已安装 `streamlit 1.63.0`（清华源）
- 修改 `tools/tokenizer_viewer/web_ui.py`：tokenizer 默认值支持环境变量 `VIEWER_TOKENIZER`，
  本机验证时指向 `_smoke/models/tiny_bert_char`（本地字符级 tokenizer）
- 启动：`python -m streamlit run web_ui.py --server.headless=true --server.port 8906`
- 验证：`http://127.0.0.1:8906` 返回 **HTTP 200**，日志显示 `You can now view your Streamlit app`
- 功能：词表查看/搜索、encode/decode 测试、两个 tokenizer diff、merge 界面

---

## 任务7：大模型应用（LLM Zero-shot / MBTI）— 讲解 + 无法离线运行

### 目录
- `LLM/zero-shot/llm_classification.py`：文本分类 zero-shot（In-Context Learning）
- `LLM/zero-shot/llm_text_matching.py`：文本匹配 zero-shot
- `LLM/zero-shot/llm_information_extraction.py`：信息抽取 zero-shot（JSON 输出约束）
- `LLM/zero-shot/playground.py`：prompt 调试平台（streamlit）
- `LLM/llms_mbti/get_llms_mbti.py`：给 Baichuan-7B / Bloom-7B 做大模型 MBTI 测试

### 代码要点（可离线阅读的部分）
- prompt 设计：先给模型系统提示词，再用 few-shot 示例构建 `pre_history`；
  最终调用 `model.chat(tokenizer, prompt, history=...)`（ChatGLM 专属接口）。
- 无法在离线/无 GPU 本机运行的原因：
  1. `AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)` 需要联网下载 ~12GB 权重，且要求 ~13GB 显存（或量化）；
  2. `llms_mbti` 需要 `baichuan-inc/Baichuan-7B`、`bigscience/bloom-7b1` 等 7B 权重；
  3. 这些脚本强依赖 ChatGLM 的 remote-code 模型类与 `.chat()` 接口，无法用本地 tiny 模型直接替换语义。

---

## 任务8：大模型训练（LLM Training）— 讲解 + 可运行的数据环节已跑通

### 8.1 chatglm_finetune（ChatGLM-6B LoRA / P-Tuning 微调）
- 代码：`train.py`（`--use_lora / --use_ptuning`）、`train_multi_gpu.py`（accelerate）、`inference.py`
- 依赖仓库内置的 `peft-chatglm`，基座 `THUDM/chatglm-6b`（remote code + fp16）
- 无法离线运行原因：无 chatglm-6b 权重、无 GPU（代码大量使用 `torch.cuda.amp.autocast`），且 peft-chatglm 与新版环境不兼容
- 数据 `data/mixed_train_dataset.jsonl`（信息抽取+文本分类混合指令数据）已随仓库提供，可阅读/自行整理

### 8.2 LLMsTrainer（从零训练：继续预训练/指令微调/Reward Model）
- `train_llms.py` + accelerate + DeepSpeed 配置；基座 open-llama-7b → 离线不可运行训练本身
- **已离线跑通的数据环节**（均在 `LLM/LLMsTrainer/data/`）：
  - SFT 数据压缩：`python compress_data.py` → sharegpt 9637 行 → 964 个 `.jsonl.zst`
  - 预训练数据压缩：MNBVC_news/qa/wiki（100/50/100 行）→ 10/5/10 个 `.jsonl.zst`
  - 修复：`compress_data.py` 中 `open()` 未指定编码，中文 Windows 默认 GBK 读 utf-8 报错 → 改为 `encoding="utf-8"`
- RM 训练数据 `data/reward_model_data/sentiment_comments.jsonl` 已就绪（prompt/selected/rejected 偏序对）
- 需要联网/GPU 才能继续的环节：tokenizer 扩充训练、词表融合、continue-pretrain / SFT / RM 训练

---

## 本轮修复清单（任务6/8/9 相关）
1. `answer_generation/iTrainingLogger.py`、`data_augment/filling_model/iTrainingLogger.py`：
   matplotlib `seaborn-darkgrid` → fallback `ggplot`
2. `answer_generation/inference.py`、`filling_model/inference.py`：
   硬编码 `cuda:0` → CPU 自适应，模型路径支持环境变量（QA_MODEL_PATH / FM_MODEL_PATH）
3. `tools/tokenizer_viewer/web_ui.py`：默认 tokenizer 支持 `VIEWER_TOKENIZER` 环境变量（离线可用）
4. `LLM/LLMsTrainer/data/compress_data.py`：`open()` 显式 `encoding="utf-8"`（修复中文 Windows GBK 读取崩溃）

## 产物位置
- `_smoke/models/tiny_t5`：tiny T5（QA/Filling 共用）
- `_smoke/checkpoints/qg_t5`、`_smoke/checkpoints/fm_t5`
- `LLM/LLMsTrainer/data/{sft_data,pretrain_data}/`：zstd 压缩分片
- `_smoke/runs/qg_*.log、fm_*.log、viewer_*.log、llmt_*compress*.log`：日志
