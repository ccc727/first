"""任务三 DPO：在 SFT 之上用偏好数据做直接偏好优化（手写 DPO 损失）。

用法：python train_dpo.py --epochs 40 --lr 1e-4 --beta 0.1 --r 4
产出 ckpt/dpo/（LoRA adapter）。
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat import build_labels, format_messages
from src.lora import inject_lora, load_lora, merge_lora, save_lora

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B"
SFT_DIR = ROOT / "ckpt" / "sft"
DPO_OUT = ROOT / "ckpt" / "dpo"
BETA_DEFAULT = 0.1


def tokenize_response(tok, prompt, response, max_len):
    msgs = [{"role": "user", "content": prompt},
            {"role": "assistant", "content": response}]
    text = format_messages(msgs)
    ids = tok(text, add_special_tokens=False).input_ids
    labels = build_labels(torch.tensor(ids, dtype=torch.long), msgs)
    if len(ids) > max_len:
        ids = ids[:max_len]
        labels = labels[:max_len]
    if (labels != -100).sum().item() == 0:
        return None
    return torch.tensor(ids, dtype=torch.long), labels


def seq_logprob(model, tok, ids, labels, device):
    """assistant token 的平均 log 概率。

    labels 采用「位移约定」：labels[i] = 位置 i+1 的 assistant 正文 token id，
    即模型在位置 i 应预测的下一 token。直接对有效位置取 log_softmax(logits[i])。
    """
    ids = ids.to(device).unsqueeze(0)
    labels = labels.to(device).unsqueeze(0)
    logits = model(ids).logits                      # (B, T, V)
    lg = F.log_softmax(logits, dim=-1)
    valid = (labels != -100) & (labels != -100)
    if valid.sum().item() == 0:
        return torch.tensor(0.0, device=device)
    safe = labels.clamp(min=0)
    tok_logps = lg.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    return tok_logps[valid].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=BETA_DEFAULT)
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--alpha", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device)

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    pairs = []
    with (ROOT / "data" / "dpo_pairs.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    print("[data] dpo pairs", len(pairs))

    # ---- 1. SFT 模型（base + SFT LoRA 合并）作为 reference
    base = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR)).to(device)
    inject_lora(base, ["q_proj", "v_proj"], r=8, alpha=16)
    load_lora(base, str(SFT_DIR / "lora.pt"))
    merge_lora(base)          # base 现在是纯 SFT 权重
    ref_model = copy.deepcopy(base).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ---- 2. policy：在 SFT 之上再注入新的 LoRA
    inject_lora(base, ["q_proj", "v_proj"], r=args.r, alpha=args.alpha)
    policy = base.train()
    ntr = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print("[lora] DPO trainable", ntr)

    # ---- 3. 预计算每个 pair 的 ids/labels 与 reference logprob
    items = []
    for i, p in enumerate(pairs):
        pc = tokenize_response(tok, p["prompt"], p["chosen"], args.max_len)
        pr = tokenize_response(tok, p["prompt"], p["rejected"], args.max_len)
        if pc is None or pr is None:
            continue
        with torch.no_grad():
            ref_c = seq_logprob(ref_model, tok, pc[0], pc[1], device).item()
            ref_r = seq_logprob(ref_model, tok, pr[0], pr[1], device).item()
        items.append((p["id"], pc, pr, ref_c, ref_r))
    print("[data] valid dpo items", len(items))
    del ref_model
    torch.cuda.empty_cache()

    optim = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=args.lr)
    total = len(items) * args.epochs
    sched = torch.optim.lr_scheduler.LinearLR(optim, 1.0, 0.1, total_iters=total)
    t0 = time.time()
    step = 0
    for epoch in range(1, args.epochs + 1):
        loss_sum, n = 0.0, 0
        for pid, pc, pr, ref_c, ref_r in items:
            logp_c = seq_logprob(policy, tok, pc[0], pc[1], device)
            logp_r = seq_logprob(policy, tok, pr[0], pr[1], device)
            logits = args.beta * (
                logp_c - logp_r - (torch.tensor(ref_c, device=device)
                                   - torch.tensor(ref_r, device=device)))
            loss = -F.logsigmoid(logits)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad], 1.0)
            optim.step()
            sched.step()
            loss_sum += loss.item()
            n += 1
            step += 1
        print(f"[epoch {epoch}/{args.epochs}] dpo_loss={loss_sum / max(n,1):.4f}  "
              f"({time.time()-t0:.0f}s)", flush=True)

    DPO_OUT.mkdir(parents=True, exist_ok=True)
    save_lora(policy, DPO_OUT / "lora.pt")
    (DPO_OUT / "args.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    (DPO_OUT / "DONE.txt").write_text("DPO finished", encoding="utf-8")
    print("[save] ckpt/dpo/lora.pt")


if __name__ == "__main__":
    main()
