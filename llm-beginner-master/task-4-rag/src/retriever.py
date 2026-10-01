"""Retriever：query embedding + FAISS top-k 召回。

自检契约：class Retriever() 无参实例化；retrieve(query, k) -> List[dict]，
每个 dict 含 text / score / source。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
IDX = ROOT / "data" / "index"
MODELS = ROOT / "models"

QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


class Retriever:
    def __init__(self, index_dir: Path | None = None):
        self.index_dir = Path(index_dir) if index_dir else IDX
        self._index = None
        self._chunks = None
        self._model = None

    def _load(self):
        if self._chunks is not None:
            return
        from ._faiss_io import read_index
        if not (self.index_dir / "vecs.faiss").exists():
            raise FileNotFoundError(
                "索引不存在，先运行 python -m src.indexer （需要 data/kb.pdf）")
        self._index = read_index(self.index_dir / "vecs.faiss")
        self._chunks = json.loads((self.index_dir / "chunks.json").read_text(
            encoding="utf-8"))

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(str(MODELS / "bge-small-zh-v1.5"))
        return self._model

    def retrieve(self, query: str, k: int = 10) -> list[dict]:
        self._load()
        model = self._get_model()
        q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)
        scores, idxs = self._index.search(np.asarray(q, dtype="float32"), k)
        out = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0 or i >= len(self._chunks):
                continue
            meta = self._chunks[int(i)]
            out.append({"text": meta["text"], "score": float(score),
                        "source": meta.get("source", "kb.pdf")})
        return out
