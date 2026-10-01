"""任务三对比：同一批指令上看 base / SFT / DPO 三个模型的回答。

用法：python compare.py   -> 输出 ckpt/compare_results.txt 并在终端打印摘要。
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.chat import format_messages
from src.lora import inject_lora, load_lora, merge_lora

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "Qwen2.5-0.5B"
CKPT = ROOT / "ckpt"

PROMPTS = [
    "什么是反向传播？",
    "LoRA 的原理是什么？",
    "“床前明月光”的作者是谁？",
    "Python 中列表和元组的区别？",
]


def load_variant(tok, kind: str):
    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR))
    if kind == "base":
        return model
    # SFT / DPO：先在 base 上挂 SFT LoRA
    inject_lora(model, ["q_proj", "v_proj"], r=8, alpha=16)
    load_lora(model, str(CKPT / "sft" / "lora.pt"))
    if kind == "sft":
        return model  # LoRA 激活式推理（与训练状态一致，已验证）
    # DPO：SFT 权重合入 base 后，再注入 DPO LoRA
    merge_lora(model)
    import json
    dpo_args = json.loads((CKPT / "dpo" / "args.json").read_text(encoding="utf-8"))
    inject_lora(model, ["q_proj", "v_proj"], r=dpo_args["r"], alpha=dpo_args["alpha"])
    load_lora(model, str(CKPT / "dpo" / "lora.pt"))
    return model


@torch.no_grad()
def generate(model, tok, prompt, max_new=90, device="cpu"):
    model.to(device)
    model.eval()
    msgs = [{"role": "user", "content": prompt}]
    text = format_messages(msgs) + "<|im_start|>assistant\n"
    ids = tok(text, add_special_tokens=False).input_ids
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        input_ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id
    )
    return tok.decode(out[0][len(ids):], skip_special_tokens=True).strip()


def main():
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lines = ["## base / SFT / DPO 输出对比", "=" * 70]
    for prompt in PROMPTS:
        lines.append("")
        lines.append(f"### 指令：{prompt}")
        for kind in ("base", "sft", "dpo"):
            if kind == "dpo" and not (CKPT / "dpo" / "lora.pt").exists():
                lines.append(f"- [{kind}] (未训练，跳过)")
                continue
            model = load_variant(tok, kind)
            ans = generate(model, tok, prompt)
            lines.append(f"- [{kind}]")
            lines.append(f"  {ans}")
            print(f"[{kind}] {prompt[:16]} -> {ans[:60]}...", flush=True)
            del model
            torch.cuda.empty_cache()
    text = "\n".join(lines)
    (CKPT / "compare_results.txt").write_text(text, encoding="utf-8")
    print("结果写入 ckpt/compare_results.txt")


if __name__ == "__main__":
    main()
