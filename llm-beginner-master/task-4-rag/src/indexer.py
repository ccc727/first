"""索引构建：PDF -> 文本 -> chunk -> BGE embedding -> FAISS 索引。

产物：data/index/{chunks.json, vecs.faiss, docstore.json}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
IDX = DATA / "index"
MODELS = ROOT / "models"

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def embed_texts(model, texts, batch_size=64):
    vecs = []
    for i in range(0, len(texts), batch_size):
        out = model.encode(texts[i:i + batch_size], normalize_embeddings=True)
        vecs.append(np.asarray(out, dtype="float32"))
    return np.concatenate(vecs, axis=0)


def extract_pdf_text(pdf_path):
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def build_index():
    import faiss
    from sentence_transformers import SentenceTransformer

    from ._faiss_io import write_index
    from .chunker import chunk_document_by_lines

    IDX.mkdir(parents=True, exist_ok=True)
    pdf = DATA / "kb.pdf"
    if not pdf.exists():
        raise SystemExit("缺少 data/kb.pdf")
    text = extract_pdf_text(pdf)
    print(f"[pdf] chars={len(text)}")
    chunks = chunk_document_by_lines(text, chunk_size=384, overlap=48)
    # 记录每个 chunk 的来源页/顺序
    chunks_meta = [{"i": i, "text": c, "source": f"kb.pdf#chunk{i}"}
                   for i, c in enumerate(chunks)]
    print(f"[chunk] n={len(chunks)}")

    model = SentenceTransformer(str(MODELS / "bge-small-zh-v1.5"))
    vecs = embed_texts(model, [c["text"] for c in chunks_meta])
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)   # normalize 后等价 cosine
    index.add(vecs)
    write_index(index, IDX / "vecs.faiss")   # 兼容中文路径，见 _faiss_io.py
    (IDX / "chunks.json").write_text(
        json.dumps(chunks_meta, ensure_ascii=False), encoding="utf-8")
    (IDX / "meta.json").write_text(
        json.dumps({"n": len(chunks), "dim": dim}), encoding="utf-8")
    print(f"[index] n={len(chunks)} dim={dim} -> {IDX}")


if __name__ == "__main__":
    build_index()
