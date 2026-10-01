"""ChnSentiCorp 备用下载脚本（datasets>=5 不支持脚本型数据集时的 fallback）。

seamew/ChnSentiCorp 是「脚本型」数据集，新版 `datasets` 已禁止执行远程脚本，
但仓库里已经带有三份 Arrow 缓存文件。本脚本直接下载它们并转成与
download.py 约定一致的 parquet：data/{train,validation,test}.parquet（text/label）。

用法：python data/fallback_build.py [--mirror https://hf-mirror.com]
"""
from __future__ import annotations

import argparse
import sys
import urllib.request as u
from pathlib import Path

import pyarrow as pa

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FILES = {
    "train": "chn_senti_corp-train.arrow",
    "validation": "chn_senti_corp-validation.arrow",
    "test": "chn_senti_corp-test.arrow",
}


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  已存在 {dest.name}，跳过下载")
        return
    print(f"  下载 {url} -> {dest.name}")
    req = u.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with u.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)


def arrow_to_df(path: Path):
    """datasets 的 .arrow 缓存是 Arrow IPC stream 格式。"""
    with pa.ipc.open_stream(path) as reader:
        return reader.read_all()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mirror", default="https://hf-mirror.com")
    args = ap.parse_args()

    DATA.mkdir(exist_ok=True)
    base = f"{args.mirror}/datasets/seamew/ChnSentiCorp/resolve/main"
    arrow_paths = {}
    for split, fname in FILES.items():
        dest = DATA / fname
        download(f"{base}/{fname}", dest)
        arrow_paths[split] = dest

    for split, path in arrow_paths.items():
        table = arrow_to_df(path)
        df = table.to_pandas()
        print(f"[{split}] arrow rows={len(df)} cols={list(df.columns)}")
        # 统一列名 text / label（label 可能是 int 或字符串）
        if "text" not in df.columns and "sentence" in df.columns:
            df = df.rename(columns={"sentence": "text"})
        df = df.rename(columns={"label": "label"})
        df["label"] = df["label"].astype(str).str.strip()
        df["label"] = df["label"].replace({"positive": "1", "negative": "0"})
        df["label"] = df["label"].map(lambda x: 1 if x in ("1", "True", "true") else 0)
        out = DATA / f"{split}.parquet"
        df[["text", "label"]].to_parquet(out, index=False)
        print(f"  -> {out.name}: {len(df)} 条")


if __name__ == "__main__":
    main()
