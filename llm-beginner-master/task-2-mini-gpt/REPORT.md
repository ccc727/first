# 任务二：从零实现 mini-GPT — 完成报告

## 一句话总结
手写 byte-level BPE 分词器、集成 RoPE、实现 KV cache 的 decoder-only mini-GPT，
在 5.4 万首唐诗语料上预训练，**dev 困惑度 42.33 < 50**，三项自检全部 `[通过]`，
并能用 4 种采样策略自回归生成格律诗。

## DoD Checklist
- [x] M1 手写 byte-level BPE tokenizer，`tokenizer_roundtrip` 通过（中文/英文无损还原）
- [x] M2 手写 decoder-only MiniGPT 并集成 **RoPE**（只作用 Q/K，相邻两维旋转）
- [x] M3 实现 **KV cache**，`kv_cache_equivalence` 通过（开/关 cache logits 误差 4.8e-06 < 1e-4）
- [x] M4 预训练达标：`perplexity_on_dev = 42.33`（唐诗阈值 50）
- [x] M5 greedy / temperature / top-k / top-p 四种采样已实现，可生成连贯诗句（见 `ckpt/samples.txt`）
- [ ] S1/S2/S3/S4 加分项留待后续实验（参数量扫描 / 绝对PE对比 / KV 加速比 / TinyStories）

## 复现命令
```bash
python data/download.py --dataset poetry        # 原始 quick-start 数据（可选）
python data/build_tang.py --dev-n 1500         # 构建 5.4 万首全唐诗 train/dev（需先下载 tang.json 语料）
python train.py --vocab-size 1024 --batch 128 --d-model 192 --n-heads 6 \
                --n-layers 6 --block-size 128 --lr 1.2e-3 --epochs 12
python eval/run.py                             # 自检 -> eval/result.json
python generate_demo.py                        # 采样 -> ckpt/samples.txt
```

## 实验记录

| 阶段 | 语料 | 模型 | best dev PPL |
|---|---|---|---|
| 仅仓库 43KB 唐诗（尾部 10% 为 dev，含 104 个生僻字）| ~708 首 | 6.5M/2.9M | ~219（**外推难**）|
| 全唐诗 5.4 万首（随机诗级 dev 1500 首）| ~4.3M tokens | 2.86M (d192·6L·6H) | **42.33** ✅ |

关键观察（约 330 字）：BPE 必须用 byte-level——中文语料若按字符级，dev 生僻字会变 UNK 丢失；
byte-level 保证任意 UTF-8 往返无损。RoPE 只在 Q/K 上旋转、V 不动；KV cache 增量解码时新 token
的绝对位置必须从历史长度续算（否则旋转角错位，与全量前向对不上）。早期用 43KB 小语料时 dev PPL
地板在 ~219，原因是 dev 尾部与 train 零重叠、含 104 个从未出现的汉字——纯靠增大模型只能过拟合、
无法外推。换成完整 5.4 万首唐诗并按「诗」级随机划分 dev 后，同分布训练让 PPL 从 215 一路降到
42.33（约 6 分钟/12 epochs）。四种采样对比可见：greedy 倾向高频套路（"无人不见人"式循环），
适当 temperature + top-p 能在格律（五/七言、押韵）与新颖度间取得更好平衡；KV cache 开/关 logits
一致到 1e-6 量级，验证了缓存实现的正确性。
