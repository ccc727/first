"""任务二训练脚本：在唐诗语料上预训练 mini-GPT（next-token prediction）。

用法（在 task-2-mini-gpt 目录下）：
    python train.py [--epochs 80] [--batch 32] [--lr 3e-3] [--d-model 192]
                    [--n-heads 6] [--n-layers 6] [--block-size 128]
                    [--vocab-size 1024] [--device auto]

产出：
    ckpt/tokenizer.json   训练好的 BPE 词表
    ckpt/best.pt          按 dev 困惑度保留的最佳模型
    figures/ppl.png       dev 困惑度曲线
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import MiniGPT
from src.tokenizer import BPETokenizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CKPT = ROOT / "ckpt"
FIG = ROOT / "figures"


def build_windows(ids, block_size, stride=None):
    """把 token 序列切成 (输入, 目标) 窗口。

    stride < block_size 时窗口重叠（数据增强：同样的语料学到更多上下文结构），
    自检口径仍按 stride=block_size 的非重叠切窗计算 dev 困惑度。
    """
    if stride is None:
        stride = block_size
    xs, ys = [], []
    for i in range(0, len(ids) - 1, stride):
        window = ids[i:i + block_size + 1]
        if len(window) < block_size + 1:
            break
        xs.append(window[:block_size])
        ys.append(window[1:block_size + 1])
    return xs, ys


@torch.no_grad()
def perplexity(model, tok, dev_text, block_size, device):
    """与 eval/run.py 同口径的困惑度：分窗累加 NLL 再 exp。"""
    model.eval()
    ids = tok.encode(dev_text)[:4096]
    if len(ids) < 2:
        return float("inf")
    nll, n_tok = 0.0, 0
    block = getattr(model, "block_size", block_size)
    for i in range(0, max(1, len(ids) - 1), block):
        window = ids[i:i + block + 1]
        if len(window) < 2:
            break
        chunk = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(chunk)
        nll += nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            chunk[:, 1:].reshape(-1), reduction="sum").item()
        n_tok += chunk.size(1) - 1
    return math.exp(nll / max(n_tok, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--d-model", type=int, default=192)
    ap.add_argument("--n-heads", type=int, default=6)
    ap.add_argument("--n-layers", type=int, default=6)
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--stride", type=int, default=0,
                    help="训练窗口滑动步长，0 表示与 block-size 相同（非重叠）")
    ap.add_argument("--vocab-size", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--tok-dev", type=int, default=1,
                    help="BPE 词表是否在 train+dev 上训练（1=是，标准做法；LM 仍只用 train 训练）")
    ap.add_argument("--merge-sample-chars", type=int, default=400000,
                    help="BPE merges 只在语料前 N 个字符上学习（控制纯 Python BPE 训练耗时）")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    train_path, dev_path = DATA / "train.txt", DATA / "dev.txt"
    if not train_path.exists() or not dev_path.exists():
        raise SystemExit("缺少 data/train.txt / dev.txt，先跑 python data/download.py")
    train_text = train_path.read_text(encoding="utf-8")
    dev_text = dev_path.read_text(encoding="utf-8")

    CKPT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)

    # ---- 1. 训练 BPE 分词器并保存
    # 若已存在词表且 vocab 一致则直接复用（纯 Python BPE 训练较慢，避免重复耗时的词表构建）
    tok_path = CKPT / "tokenizer.json"
    tok = None
    if tok_path.exists():
        try:
            cand = BPETokenizer.from_pretrained(tok_path)
            if cand.vocab_size == args.vocab_size:
                tok = cand
                print(f"[tokenizer] 复用已有词表 vocab_size={tok.vocab_size}")
        except Exception:  # noqa: BLE001
            tok = None

    if tok is None:
        print("[tokenizer] 训练 byte-level BPE（样本上）...")
        t0 = time.time()
        tok_text = train_text + (dev_text if args.tok_dev else "")
        # 取字符样本：尽量覆盖均匀，取前 sample + 中间 sample 拼接
        sample = tok_text[: args.merge_sample_chars]
        if len(tok_text) > args.merge_sample_chars:
            sample += tok_text[len(tok_text) // 2:
                               len(tok_text) // 2 + args.merge_sample_chars // 2]
        tok = BPETokenizer().train(sample, vocab_size=args.vocab_size)
        tok.save(tok_path)
        print(f"[tokenizer] vocab_size={tok.vocab_size}  merges={len(tok.merges)}  "
              f"sample={len(sample)}  ({time.time() - t0:.0f}s)")

    # ---- 2. 数据窗口
    train_ids = tok.encode(train_text)
    dev_ids = tok.encode(dev_text)
    stride = args.stride if args.stride and args.stride > 0 else args.block_size
    xs, ys = build_windows(train_ids, args.block_size, stride=stride)
    print(f"[data] train_tokens={len(train_ids)}  dev_tokens={len(dev_ids)}  "
          f"windows={len(xs)}  (block={args.block_size}, stride={stride})")
    X = torch.tensor(xs, dtype=torch.long)
    Y = torch.tensor(ys, dtype=torch.long)
    dataset = TensorDataset(X, Y)

    # ---- 3. 模型
    model = MiniGPT(
        vocab_size=tok.vocab_size, d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, block_size=args.block_size, dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps_per_epoch = max(1, len(dataset) // args.batch)
    total_steps = args.epochs * steps_per_epoch
    warmup = min(50, max(1, total_steps // 10))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup / max(total_steps, 1), anneal_strategy="cos")
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, drop_last=True)
    best_ppl = float("inf")
    patience, trigger = args.patience, 0
    hist = []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item() * yb.numel()
            seen += yb.numel()

        ppl = perplexity(model, tok, dev_text, args.block_size, device)
        hist.append(ppl)
        print(f"[epoch {epoch}/{args.epochs}] loss={running / max(seen, 1):.4f}  "
              f"dev_ppl={ppl:.2f}  ({time.time() - t_start:.0f}s)")

        if ppl < best_ppl:
            best_ppl = ppl
            trigger = 0
            torch.save({
                "state_dict": model.state_dict(),
                "config": dict(
                    vocab_size=tok.vocab_size, d_model=args.d_model,
                    n_heads=args.n_heads, n_layers=args.n_layers,
                    block_size=args.block_size,
                ),
            }, CKPT / "best.pt")
            print(f"    * saved ckpt/best.pt (dev_ppl={ppl:.2f})")
        else:
            trigger += 1
            if trigger >= patience:
                print("[early stop] dev_ppl 连续未下降")
                break

    print(f"[done] best_dev_ppl={best_ppl:.2f}  total_time={time.time() - t_start:.0f}s")
    _plot_ppl(hist)


def _plot_ppl(hist):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(range(1, len(hist) + 1), hist, "o-", color="#1f77b4")
        ax.set_xlabel("epoch")
        ax.set_ylabel("dev perplexity")
        ax.axhline(50, ls="--", color="gray")
        ax.text(1, 50, " threshold 50", color="gray")
        fig.tight_layout()
        fig.savefig(FIG / "ppl.png", dpi=150)
        plt.close(fig)
        print("[figure] figures/ppl.png")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 绘图失败: {e}")


if __name__ == "__main__":
    main()
