"""端到端 RAG：answer(query) -> {answer, sources}。

生成侧：
1) 若配置了 OpenAI 兼容的本地 Qwen 服务（环境变量 RAG_LLM_BASE_URL，
   或默认 http://localhost:8000/v1 可用），用检索上下文让模型生成；
2) 否则退回"基于检索片段的抽取式回答"，保证端到端可运行、答案可溯源。
"""
from __future__ import annotations

import os

from .reranker import Reranker
from .retriever import Retriever

SYSTEM = ("你是一个严谨的中文问答助手。只能依据给定的参考资料回答问题，"
          "不要使用资料之外的知识；如果资料不足以回答，请直接说明不知道。")


def _dedupe(docs, max_chars=3000):
    seen = set()
    out = []
    total = 0
    for d in docs:
        t = d.get("text", "")
        if t in seen:
            continue
        seen.add(t)
        if total + len(t) > max_chars:
            break
        out.append(t)
        total += len(t)
    return out


def answer(query: str) -> dict:
    retriever = Retriever()
    docs = retriever.retrieve(query, k=20)
    if not docs:
        return {"answer": "未检索到相关资料，无法回答。", "sources": []}
    try:
        docs = Reranker().rerank(query, docs, top_k=5)
    except Exception:
        docs = docs[:5]
    contexts = _dedupe(docs)

    prompt = "参考资料：\n" + "\n---\n".join(contexts) + \
        f"\n\n问题：{query}\n请基于资料作答，不知道就明说。"

    text = _call_llm(prompt)
    if not text or not text.strip():
        text = f"根据检索资料摘要：{contexts[0][:400]}"
    sources = [{"text": d["text"], "source": d.get("source", "kb.pdf"),
                "score": d.get("score")} for d in docs]
    return {"answer": text.strip(), "sources": sources}


def _call_llm(prompt: str, max_tokens: int = 400) -> str | None:
    base_url = os.environ.get("RAG_LLM_BASE_URL", "http://localhost:8000/v1")
    api_key = os.environ.get("RAG_LLM_API_KEY", "EMPTY")
    model_name = os.environ.get("RAG_LLM_MODEL", "qwen2.5-7b-instruct")
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=60)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": prompt}],
            max_tokens=max_tokens, temperature=0.3,
        )
        return resp.choices[0].message.content
    except Exception:
        return None
