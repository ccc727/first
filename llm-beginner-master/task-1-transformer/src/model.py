"""TransformerClassifier：手写 Transformer encoder + padding mask + 分类头。

对外契约（自检 eval/run.py 依赖）：
- `class TransformerClassifier`
- `load_for_eval(ckpt_path) -> (model, tokenize_fn)`
  tokenize_fn(text: str) -> LongTensor 形状 (T,)
  model(ids) 接受 (B, T) 返回 (B, num_classes) logits

分词采用**确定性字符级**方案：覆盖 ASCII 可打印字符 + 中日韩标点 + 全部
CJK 统一表意文字 + 全角字符。任何合法中文/英文句子都可以被稳定编码，
无需额外词表文件，保证训练与自检（对 dev 集逐条编码）完全一致。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .block import TransformerBlock

# ---------------------------------------------------------------------------
# 确定性字符级 tokenizer
# ---------------------------------------------------------------------------
PAD_ID = 0
UNK_ID = 1

# 各字符区间在词表中的排列顺序（id 从 _BASE_ID 开始递增分配）
_ASCII_START = 0x20        # 空格 ~ ~    （0x20-0x7E 共 95 个可打印字符）
_CJK_PUNCT_START = 0x3000  # 中日韩符号和标点（0x3000-0x303F 共 64 个）
_FULLWIDTH_START = 0xFF00   # 全角字符（0xFF00-0xFFEF 共 240 个）
_CJK_START = 0x4E00        # CJK 统一表意文字（0x4E00-0x9FFF 共 20992 个）

_N_ASCII = 0x7F - _ASCII_START          # 95
_N_CJK_PUNCT = 0x3040 - _CJK_PUNCT_START  # 64
_N_FULLWIDTH = 0xFFF0 - _FULLWIDTH_START  # 240
_N_CJK = 0xA000 - _CJK_START             # 20992


def _char_id(ch: str) -> int:
    """把单个字符映射到词表 id（未覆盖字符 -> UNK）。"""
    code = ord(ch)
    if _ASCII_START <= code < 0x7F:
        return 2 + (code - _ASCII_START)
    if _CJK_PUNCT_START <= code < 0x3040:
        return 2 + _N_ASCII + (code - _CJK_PUNCT_START)
    if _FULLWIDTH_START <= code < 0xFFF0:
        return 2 + _N_ASCII + _N_CJK_PUNCT + (code - _FULLWIDTH_START)
    if _CJK_START <= code < 0xA000:
        return 2 + _N_ASCII + _N_CJK_PUNCT + _N_FULLWIDTH + (code - _CJK_START)
    return UNK_ID


VOCAB_SIZE = 2 + _N_ASCII + _N_CJK_PUNCT + _N_FULLWIDTH + _N_CJK  # 21393


def tokenize(text: str, max_len: int = 128) -> list[int]:
    """将文本编码为 id 序列（最长 max_len）。"""
    chars = text[:max_len]
    return [_char_id(c) for c in chars] if chars else [PAD_ID]


# ---------------------------------------------------------------------------
# 位置编码（正弦-余弦固定编码）
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    """Sinusoidal absolute positional encoding（与 Transformer 原论文一致）。"""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
class TransformerClassifier(nn.Module):
    """N 层手写 Transformer encoder + 排除 padding 的 mean pooling + 分类头。"""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        num_classes: int = 2,
        max_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pad_id = PAD_ID

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(
        self,
        ids: torch.Tensor,
        return_weights: bool = False,
    ) -> torch.Tensor:
        """ids: (B, T) LongTensor；返回 (B, num_classes) logits。"""
        x = self.tok_emb(ids) * math.sqrt(self.d_model)
        x = self.drop(self.pos_enc(x))

        padding_mask = ids == self.pad_id  # (B, T)，True = pad

        for block in self.blocks:
            x = block(x, padding_mask=padding_mask, return_weights=return_weights)

        x = self.final_ln(x)

        # mean pooling：只对非 padding 位置求平均
        keep = (~padding_mask).float().unsqueeze(-1)          # (B, T, 1)
        denom = keep.sum(dim=1).clamp(min=1.0)                # (B, 1)
        pooled = (x * keep).sum(dim=1) / denom                # (B, d_model)

        return self.head(pooled)


# ---------------------------------------------------------------------------
# 自检契约：load_for_eval
# ---------------------------------------------------------------------------
def load_for_eval(ckpt_path: str) -> tuple[nn.Module, object]:
    """从 best.pt 恢复 (model, tokenize_fn)。

    约定：tokenize_fn(text) -> LongTensor (T,)；model(ids) -> (B, num_classes)。
    """
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = data["config"]
    model = TransformerClassifier(**cfg)
    model.load_state_dict(data["state_dict"])
    model.eval()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model.to(device)

    max_len = cfg.get("max_len", 128)

    def tokenize_fn(text: str) -> torch.Tensor:
        return torch.tensor(tokenize(text, max_len), dtype=torch.long, device=device)

    return model, tokenize_fn
