"""SFT 数据加载：MOSS 会话 -> messages 列表。
"""
from __future__ import annotations

import json
from pathlib import Path


def iter_conversations(path: str | Path, max_samples: int = -1):
    """逐条产出 {'id','messages'}；跳过带工具调用标记或结构异常的记录。"""
    path = Path(path)
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = obj.get("conversation")
            msgs = obj.get("messages")
            if isinstance(msgs, list):
                conv = msgs  # 已是 [{'role','content'}...] 形式
            if not isinstance(conv, list):
                continue
            messages = []
            ok = False
            bad = False
            for turn in conv:
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("role", "")).lower()
                if role not in ("user", "assistant", "system"):
                    continue
                content = turn.get("content")
                if isinstance(content, list):
                    content = "".join(
                        str(c) if isinstance(c, str) else "" for c in content
                    )
                content = "" if content is None else str(content)
                if "<|plugin|>" in content or "<|MOSS|>" in content:
                    bad = True
                    break
                if content.strip():
                    messages.append({"role": role, "content": content.strip()})
                    if role == "assistant":
                        ok = True
            if ok and not bad and messages:
                yield {"id": obj.get("id"), "messages": messages}
                count += 1
                if 0 < max_samples <= count:
                    return
