"""任务三 SFT：在 Qwen2.5-0.5B 上手写 LoRA 做监督微调。

用法（在 task-3-sft-dpo 目录下）：
    python train_sft.py --epochs 3 --batch 8 --lr 2e-4 --max-len 512
产出 ckpt/sft/ 下 LoRA adapter（lora.json 等）。
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat import build_labels, format_messages
from src.lora import inject_lora, save_lora
from src.sft_data import iter_conversations

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B"
SFT_OUT = ROOT / "ckpt" / "sft"


def prepare_text(tok, msgs, max_len):
    """把 messages 渲染成 ChatML 并得到 (input_ids, labels)。"""
    text = format_messages(msgs)
    ids = tok(text, add_special_tokens=False).input_ids
    if not ids:
        return None
    labels = build_labels(torch.tensor(ids, dtype=torch.long), msgs)
    if len(ids) > max_len:
        ids = ids[:max_len]
        labels = labels[:max_len]
    if (labels != -100).sum().item() == 0:
        return None
    return ids, labels.tolist()


class SFTDataset(Dataset):
    def __init__(self, tok, samples, max_len):
        self.tok = tok
        self.max_len = max_len
        self.items = []
        for s in samples:
            prep = prepare_text(tok, s["messages"], max_len)
            if prep is None:
                continue
            self.items.append((prep[0], prep[1], s.get("id")))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, labels, _ = self.items[i]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate(batch, pad_id):
    ids_l = [b[0] for b in batch]
    lab_l = [b[1] for b in batch]
    max_t = max(x.size(0) for x in ids_l)
    ids = torch.full((len(batch), max_t), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), max_t), -100, dtype=torch.long)
    for i, (x, y) in enumerate(zip(ids_l, lab_l)):
        ids[i, : x.size(0)] = x
        lab[i, : y.size(0)] = y
    return ids, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--n-samples", type=int, default=2600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device)

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    samples = list(iter_conversations(ROOT / "data" / "sft_samples.jsonl",
                                      max_samples=args.n_samples))
    random.Random(args.seed).shuffle(samples)
    print(f"[data] samples={len(samples)}")

    ds = SFTDataset(tok, samples, args.max_len)
    print(f"[data] after filter={len(ds)}")
    if len(ds) == 0:
        raise SystemExit("没有可用 SFT 样本")
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=False,
                        collate_fn=lambda b: collate(b, pad_id))

    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR))
    model.to(device)
    inject_lora(model, target_modules=["q_proj", "v_proj"], r=args.r, alpha=args.alpha)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[lora] trainable params={ntr}")

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    steps = len(loader) * args.epochs
    sched = torch.optim.lr_scheduler.LinearLR(optim, start_factor=1.0, end_factor=0.1,
                                              total_iters=steps)
    model.train()
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        n = 0
        for ids, lab in loader:
            ids, lab = ids.to(device), lab.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(ids).logits
            loss = criterion(logits.view(-1, logits.size(-1)), lab.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                     1.0)
            optim.step()
            sched.step()
            total_loss += loss.item() * ids.size(0)
            n += ids.size(0)
        print(f"[epoch {epoch}/{args.epochs}] loss={total_loss / max(n, 1):.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    SFT_OUT.mkdir(parents=True, exist_ok=True)
    save_lora(model, SFT_OUT / "lora.pt")
    (SFT_OUT / "args.json").write_text(
        __import__("json").dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8")
    (SFT_OUT / "DONE.txt").write_text("SFT finished", encoding="utf-8")
    print("[save] ckpt/sft/lora.pt")


if __name__ == "__main__":
    main()
