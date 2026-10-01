"""本地文件检索工具：按文件名模式或内容关键词检索，返回匹配文件及内容片段。

- ``pattern`` 支持通配符（``*.md``）与内容关键词两种用法；
- ``dir`` 收到文件路径时直接返回该文件内容（用于「读取某个文件写了什么」类任务）。
"""
from __future__ import annotations

import fnmatch
from pathlib import Path

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "file_search",
        "description": "在指定目录内按文件名通配模式（如 *.md）或内容关键词检索，"
                       "返回匹配文件及内容片段；dir 传文件路径时直接返回该文件内容。",
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "文件名通配模式（如 *.md）或内容关键词"},
                "dir": {"type": "string", "description": "要搜索的目录（或要读取的文件）路径"},
            },
            "required": ["pattern", "dir"],
        },
    },
}

MAX_READ_CHARS = 2000


def _read_file(p: Path) -> str:
    """读取单个文件内容（截断），用于「读取某文件」类调用。"""
    text = p.read_text(encoding="utf-8", errors="ignore")
    if len(text) > MAX_READ_CHARS:
        text = text[:MAX_READ_CHARS] + "…(已截断)"
    return f"{p}\n{text}"


def run(args: dict) -> str:
    pattern = str(args.get("pattern", "")).strip()
    base = str(args.get("dir", "."))
    root = Path(base).resolve()
    if root.is_file():      # dir 指向文件：直接当作「读取文件内容」
        return _read_file(root)
    if not root.is_dir():
        # 允许传不存在的路径时退到父目录，尽量给出有用结果
        root = root.parent
    matches = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            # 通配模式（fnmatch 支持 *.md / todo_*.md；也是纯文件名子串匹配的超集）
            if pattern and fnmatch.fnmatch(p.name.lower(), pattern.lower()):
                matches.append(str(p))
                continue
            if pattern and pattern.lower() in p.name.lower():
                matches.append(str(p))
                continue
            if pattern:
                try:
                    txt = p.read_text(encoding="utf-8", errors="ignore")[:60000]
                    if pattern in txt:
                        i = txt.find(pattern)
                        matches.append(f"{p}\n...{txt[max(0, i-60):i+len(pattern)+160]}...")
                except Exception:   # noqa: BLE001 —— 单个文件读失败不影响整体检索
                    pass
        if len(matches) >= 20:
            break
    if not matches:
        return f"未在 {root} 找到匹配 {pattern!r} 的文件"
    return "\n".join(matches)
