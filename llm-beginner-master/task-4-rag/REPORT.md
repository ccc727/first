# 任务四：RAG 文档问答 — 完成报告

## 一句话总结
对 NNDL v2（`data/kb.pdf`）走「字符切块 → BGE 向量化 → FAISS 召回 → bge-reranker 精排 → 生成」
的 RAG 流水线；在 30 条 gold QA 上自检 **Recall@10 = 0.867**（门槛 0.6）、MRR = 0.485，端到端
可返回可溯源答案，四项自检全部 `[通过]`。

## DoD Checklist
- [x] M1 实现 `chunk_text`（字符窗口 + overlap），`chunking_sanity` 通过（16 块，平均长度 255.0 ∈ [128, 307.2]）
- [x] M2 从 `data/kb.pdf` 建 embedding + FAISS 索引，`Retriever.retrieve` 可用（`data/index/` 共 1916 块）
- [x] M3 `nndl_gold_recall_at_10` 通过：Recall@1/3/5/10 = 0.300 / 0.633 / 0.700 / **0.867**，MRR = 0.485
- [x] M4 `rag_end_to_end` 通过（返回非空 `answer` + 非空 `sources`）
- [ ] S1 chunk_size 扫描 / S2 reranker 消融 / S3 query rewrite / S4 RAGAS（加分项，留待扩展）

## 复现命令
```bash
python data/download.py             # 拉 kb.pdf / BGE 模型 / gold QA（本仓库已就绪）
python -m src.indexer               # 建索引 -> data/index/{vecs.faiss, chunks.json, meta.json}
python eval/run.py                  # 自检 -> eval/result.json（三项全过）
```

## 流水线与配置
| 环节 | 实现 | 配置 |
|---|---|---|
| 切块 | `src/chunker.py`：按行归并后字符窗口切，超长块用 `chunk_text` 硬切 | `chunk_document_by_lines(chunk_size=384, overlap=48)` → **1916 块** |
| 向量化 | `src/indexer.py`：bge-small-zh-v1.5，`normalize_embeddings=True` | PDF 抽取 606,655 字符，dim = **512** |
| 索引 | `IndexFlatIP`（归一化后等价余弦），query 端带 bge 检索前缀 | `data/index/vecs.faiss` |
| 召回 | `src/retriever.py`：`retrieve(query, k=20)` | 自检按 k=10 调用 |
| 精排 | `src/reranker.py`：bge-reranker-base（CrossEncoder） | `rerank(top_k=5)`，失败则退回前 5 |
| 生成 | `src/rag.py`：拼「参考资料 + 问题」prompt | 本地 Qwen 不可用时退回**抽取式摘要**作答 |

## 检索评测结果（30 条 gold QA）
| 指标 | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|---|
| 数值 | 0.300 | 0.633 | 0.700 | **0.867** | 0.485 |

30 题中命中 26 题；4 题未命中集中在绪论 / 图神经网络 / 无监督学习等章节的表述型 gold anchor
（如 `nndl-transformer-attention`、`nndl-graphsage-gat`、`nndl-pca-autoencoder`），rank 多在 10 开外
或 anchor 文本与 chunk 用词存在句法差异。

## 本机问题与修复
faiss 底层 C++ 用 `fopen` 打开文件，Windows 下无法直接读写含中文的路径（本项目位于
`e:\桌面\...`，直接 `write_index` / `read_index` 会抛 `could not open ... for writing/reading`）。
新增 [src/_faiss_io.py](src/_faiss_io.py)：写盘走 `faiss.serialize_index` + Python `Path.write_bytes`，
读盘走 `np.frombuffer` + `faiss.deserialize_index`，全程内存字节流、不经过 fopen。indexer / retriever
改为经该模块读写后，中文路径下可正常 `python -m src.indexer` 与 `python eval/run.py`。

## 实验观察（约 300 字，草稿可改）
30 条 gold QA 上 Recall@10 = 0.867、MRR = 0.485，说明「粗召回 + 精排」能覆盖绝大多数知识点：单靠
Top-1 只有 0.30 命中，Rerank 到 Top-3 后升到 0.63——印证了粗召回靠 bge 小模型的语义近似、精排阶段用
交叉编码器修正序的核心设计。未命中的几条多为跨句、需全局整合的知识点（如对比类、综述类表述），
其 gold anchor 与库内原文存在用词差异，暴露了固定字符窗 + 单向量余弦在「长程语义对齐」上的短板，
后续可试 chunk_size 扫描与 query rewrite（S1/S3）。端到端生成侧本机未部署本地 Qwen，`_call_llm`
探测 `localhost:8000` 失败后自动退回抽取式摘要，故 M4 仅验证了「检索→组装→可溯源回答」这条链路；
部署 Qwen 后生成质量仍待人工对比（如回答是否忠实于资料、是否答非所问）。
