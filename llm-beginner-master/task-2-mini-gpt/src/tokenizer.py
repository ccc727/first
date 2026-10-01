"""手写简化版 BPE tokenizer（不依赖 tiktoken / sentencepiece）。

采用 **byte-level BPE**（GPT-2 风格）：
- 词表以 0..255 的原始字节为基底，任何 UTF-8 文本都可无损编码；
- 训练时统计相邻 token 对频率、迭代地把最高频对 merge 成新 token；
- 因此 encode -> decode 对任意中英文都能逐字节还原（自检 tokenizer_roundtrip 的
  中文 / 英文样例天然满足），跨 UTF-8 多字节边界也安全。

对外契约（自检 eval/run.py 依赖）：
    BPETokenizer.encode(text) -> List[int]
    BPETokenizer.decode(ids)  -> str
    BPETokenizer.vocab_size   -> int
    BPETokenizer.from_pretrained(path)
"""
from __future__ import annotations

import json
from pathlib import Path

# ---------------------------------------------------------------------------
# 字节级底层工具
# ---------------------------------------------------------------------------
def bytes_to_ids(text: str) -> list[int]:
    """把字符串编码为 UTF-8 字节列表（0..255）。"""
    return list(text.encode("utf-8"))


def ids_to_text(ids: list[int], table: dict[int, bytes]) -> str:
    """把 token id 序列还原为字符串。

    跨 UTF-8 边界由整体 decode 保证：只要 token 序列来自合法文本，整段拼接就仍是
    合法 UTF-8，可精确还原；用 errors='replace' 仅为兜底"随机生成出的非法字节拼接"。
    """
    raw = b"".join(table[i] for i in ids)
    return raw.decode("utf-8", errors="replace")


def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """统计相邻 token 对的出现次数。"""
    counts: dict[tuple[int, int], int] = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


class BPETokenizer:
    """Byte-level BPE：支持 train / encode / decode / 导出与加载。"""

    BASE_VOCAB = 256  # 字节基底

    def __init__(self, merges: list[tuple[int, int]] | None = None):
        # merges 按训练顺序排列：先学到的新 token id 更小（256, 257, ...）
        self.merges: list[tuple[int, int]] = merges or []
        # pair -> 新 token id
        self.merge_rank: dict[tuple[int, int], int] = {
            pair: self.BASE_VOCAB + i for i, pair in enumerate(self.merges)
        }
        # 每个 token id -> 它的字节内容（用于 decode）
        self._table: dict[int, bytes] = {i: bytes([i]) for i in range(self.BASE_VOCAB)}
        for (a, b), i in self.merge_rank.items():
            self._table[i] = self._table[a] + self._table[b]

    # ------------------------------------------------------------------ 属性
    @property
    def vocab_size(self) -> int:
        return self.BASE_VOCAB + len(self.merges)

    # ------------------------------------------------------------------ 训练
    def train(self, text: str, vocab_size: int | None = None) -> "BPETokenizer":
        """在语料上迭代 merge 直到词表达到 vocab_size（默认 256 + 1000 merge）。

        每轮取当前出现频率最高（并列时取更早学到）的相邻对，合并成新 token。
        语料 ~几十 KB，全量重扫的复杂度可以接受。
        """
        target = vocab_size or 1256
        num_merges = max(0, target - self.BASE_VOCAB)

        ids = bytes_to_ids(text)
        for m in range(num_merges):
            counts = get_stats(ids)
            # 只考虑尚未 merge 过的相邻对；取最高频者（并列时取先出现的）
            candidates = [
                (p, c) for p, c in counts.items() if p not in self.merge_rank
            ]
            if not candidates:
                break
            best_pair = max(candidates, key=lambda pc: pc[1])[0]
            self.merges.append(best_pair)
            new_id = self.BASE_VOCAB + m
            self.merge_rank[best_pair] = new_id
            self._table[new_id] = self._table[best_pair[0]] + self._table[best_pair[1]]

            # 单次扫描合并所有出现位置
            new_ids: list[int] = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) == best_pair:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids
        return self

    # ------------------------------------------------------------------ 编解码
    def encode(self, text: str) -> list[int]:
        """编码（O(n) 栈式贪心）：逐个字节入栈，入栈后尝试把栈顶相邻对合并为
        已学到的 merge token，能合并就一直合并到不能为止。

        decode 的精确还原由 merge 树保证（token 内容是子 token 字节拼接），
        所以这里的贪心分段即使与训练顺序不完全一致，往返依旧逐字节精确。
        """
        ids: list[int] = []
        for byte in bytes_to_ids(text):
            ids.append(byte)
            while len(ids) >= 2:
                rank = self.merge_rank.get((ids[-2], ids[-1]))
                if rank is None:
                    break
                ids[-2:] = [rank]
        return ids

    def decode(self, ids) -> str:
        if isinstance(ids, (list, tuple)):
            pass
        else:  # numpy / torch 张量
            ids = ids.tolist()
        return ids_to_text([int(i) for i in ids], self._table)

    # ------------------------------------------------------------------ 保存
    def save(self, path: str | Path) -> None:
        obj = {
            "version": 1,
            "type": "byte-level-bpe",
            "merges": [[int(a), int(b)] for (a, b) in self.merges],
        }
        Path(path).write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def from_pretrained(cls, path: str | Path) -> "BPETokenizer":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [tuple(pair) for pair in obj["merges"]]
        return cls(merges=merges)
