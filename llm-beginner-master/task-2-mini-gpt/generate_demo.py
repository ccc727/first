"""生成样例脚本（DoD M5）：对比 greedy / temperature / top-k / top-p 采样。

用法：python generate_demo.py
输出：ckpt/samples.txt（几段不同采样策略的生成结果）
"""
from __future__ import annotations

from pathlib import Path

from src.model import load_for_eval

ROOT = Path(__file__).resolve().parent
CKPT = ROOT / "ckpt" / "best.pt"

PROMPTS = ["床前明月光", "春眠不觉晓", "白日依山尽"]

CONFIGS = [
    ("greedy",            dict(temperature=1e-9)),
    ("temp=0.8",          dict(temperature=0.8)),
    ("top-k=40 temp=0.9", dict(temperature=0.9, top_k=40)),
    ("top-p=0.9 temp=0.9", dict(temperature=0.9, top_p=0.9)),
    ("top-k=100 top-p=0.9", dict(temperature=0.9, top_k=100, top_p=0.9)),
]


def main():
    if not CKPT.exists():
        raise SystemExit("ckpt/best.pt 不存在，先跑 python train.py")
    model, tok = load_for_eval(str(CKPT))
    model.to("cuda" if __import__("torch").cuda.is_available() else "cpu")
    out_lines = [f"MiniGPT 生成样例  max_new_tokens=60", "=" * 60]

    for prompt in PROMPTS:
        out_lines.append("")
        out_lines.append(f"## prompt: {prompt}")
        for name, cfg in CONFIGS:
            ids = model.generate(
                tok.encode(prompt), max_new_tokens=60,
                top_k=cfg.get("top_k", 0), top_p=cfg.get("top_p", 0.0),
                temperature=cfg.get("temperature", 1.0),
            )
            text = tok.decode(ids)
            out_lines.append(f"--- [{name}]")
            out_lines.append(text)

    # 先写文件，再打印（打印仅做摘要，避免控制台编码把正文搞崩）
    text = "\n".join(out_lines)
    (ROOT / "ckpt" / "samples.txt").write_text(text, encoding="utf-8")
    print(f"生成完成，共 {len(out_lines)} 行，写入 ckpt/samples.txt")


if __name__ == "__main__":
    main()
