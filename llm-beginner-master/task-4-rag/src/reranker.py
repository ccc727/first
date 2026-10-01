"""Reranker：bge-reranker-base 精排 [query, doc] 文本对。"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


class Reranker:
    def __init__(self):
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            from sentence_transformers import CrossEncoder
            self._pipe = CrossEncoder(str(MODELS / "bge-reranker-base"))
        return self._pipe

    def rerank(self, query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
        if not docs:
            return []
        pipe = self._load()
        pairs = [(query, d["text"]) for d in docs]
        scores = pipe.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: -float(x[1]))
        out = []
        for d, s in ranked[:top_k]:
            out.append({**d, "score": float(s)})
        return out
