"""任务一训练脚本：在 ChnSentiCorp 中文情感二分类上训练 TransformerClassifier。

用法（在 task-1-transformer 目录下）：
    python train.py [--epochs 5] [--batch 64] [--lr 3e-4] [--d-model 128]
                    [--n-layers 4] [--max-len 128] [--device auto]

产出：
    ckpt/best.pt            按 dev 准确率保留的最佳模型（含 config）
    figures/train_loss.png  训练 loss 曲线
    eval/result.json        由 python eval/run.py 生成
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.model import PAD_ID, VOCAB_SIZE, TransformerClassifier, tokenize

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "ckpt"
FIG_DIR = ROOT / "figures"


class ChnSentiDataset(Dataset):
    def __init__(self, texts, labels, max_len):
        self.texts = list(texts)
        self.labels = list(labels)
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        ids = torch.tensor(tokenize(str(self.texts[i]), self.max_len), dtype=torch.long)
        return ids, torch.tensor(int(self.labels[i]), dtype=torch.long)


def collate_fn(batch):
    ids_list, labels = zip(*batch)
    max_t = max(x.size(0) for x in ids_list)
    padded = torch.full((len(batch), max_t), PAD_ID, dtype=torch.long)
    for i, x in enumerate(ids_list):
        padded[i, : x.size(0)] = x
    return padded, torch.stack(labels)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for ids, labels in loader:
        ids, labels = ids.to(device), labels.to(device)
        logits = model(ids)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    train_path = DATA_DIR / "train.parquet"
    val_path = DATA_DIR / "validation.parquet"
    if not train_path.exists() or not val_path.exists():
        raise SystemExit(
            "缺少 data/train.parquet 或 data/validation.parquet，先跑 python data/download.py"
        )

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    print(f"[data] train={len(train_df)}  val={len(val_df)}")

    train_ds = ChnSentiDataset(train_df["text"], train_df["label"], args.max_len)
    val_ds = ChnSentiDataset(val_df["text"], val_df["label"], args.max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate_fn, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=collate_fn, num_workers=0)

    model = TransformerClassifier(
        vocab_size=VOCAB_SIZE,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        num_classes=2,
        max_len=args.max_len,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] params={n_params / 1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps,
        pct_start=min(args.warmup_steps / total_steps, 0.9), anneal_strategy="cos",
    )
    criterion = nn.CrossEntropyLoss()

    CKPT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)

    best_acc = 0.0
    hist = {"epoch": [], "loss": [], "dev_acc": []}
    patience, trigger = 4, 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for ids, labels in train_loader:
            ids, labels = ids.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(ids)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)
        avg_loss = running / seen

        dev_acc = evaluate(model, val_loader, device)
        hist["epoch"].append(epoch)
        hist["loss"].append(round(avg_loss, 4))
        hist["dev_acc"].append(round(dev_acc, 4))
        print(f"[epoch {epoch}/{args.epochs}] loss={avg_loss:.4f}  "
              f"dev_acc={dev_acc:.4f}  ({time.time() - t0:.0f}s)")

        if dev_acc > best_acc:
            best_acc = dev_acc
            trigger = 0
            cfg = dict(
                vocab_size=VOCAB_SIZE, d_model=args.d_model, n_heads=args.n_heads,
                n_layers=args.n_layers, d_ff=args.d_ff, num_classes=2,
                max_len=args.max_len,
            )
            torch.save(
                {"state_dict": model.state_dict(), "config": cfg},
                CKPT_DIR / "best.pt",
            )
            print(f"    * saved ckpt/best.pt (dev_acc={best_acc:.4f})")
        else:
            trigger += 1
            if trigger >= patience:
                print("[early stop] dev acc 连续未提升")
                break

    print(f"[done] best_dev_acc={best_acc:.4f}  total_time={time.time() - t0:.0f}s")
    _plot_history(hist)


def _plot_history(hist):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(hist["epoch"], hist["loss"], "o-", color="#d62728", label="train loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss", color="#d62728")
        ax2 = ax1.twinx()
        ax2.plot(hist["epoch"], hist["dev_acc"], "s-", color="#1f77b4", label="dev acc")
        ax2.set_ylabel("dev acc", color="#1f77b4")
        fig.legend(loc="center right")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "train_loss.png", dpi=150)
        plt.close(fig)
        print("[figure] figures/train_loss.png")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] 绘图失败: {e}")


if __name__ == "__main__":
    main()

