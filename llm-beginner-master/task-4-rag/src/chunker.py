"""文本切分：固定字符窗口 + overlap（chunk_size / overlap 均按字符计）。

自检 chunking_sanity 口径：chunk 数 > 10，平均长度 ∈ [0.5, 1.2]*chunk_size。
"""
from __future__ import annotations


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """把长文本切成字符级窗口。窗口长度尽量接近 chunk_size；overlap 指相邻
    窗口之间重叠的字符数（避免把语义单元/句子切破到完全不可见）。
    """
    text = str(text)
    n = len(text)
    if n == 0:
        return []
    chunks: list[str] = []
    step = max(1, chunk_size - max(0, overlap))
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end >= n:
            break
        start += step
    return chunks


def chunk_document_by_lines(text: str, chunk_size: int = 384,
                            overlap: int = 48) -> list[str]:
    """先按段落行归并再切分：尽量保证每块从行首开始（版式友好）。"""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    chunks: list[str] = []
    buf = ""
    for ln in lines:
        if buf and len(buf) + len(ln) + 1 > chunk_size:
            chunks.append(buf)
            # 保留尾部 overlap 字符作为下块前缀
            buf = buf[-overlap:] if overlap else ""
        buf = (buf + "\n" + ln) if buf else ln
    if buf:
        chunks.append(buf)
    # 若单块仍超长则用 chunk_text 硬切
    out: list[str] = []
    for c in chunks:
        if len(c) > chunk_size * 1.2:
            out.extend(chunk_text(c, chunk_size, overlap))
        else:
            out.append(c)
    return out or chunk_text(text, chunk_size, overlap)
