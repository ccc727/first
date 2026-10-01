"""路径安全：所有工具只能在工作 repo 内读写。

安全要点（对应 README 第 2 周「安全」三条）：
- 路径先 `Path.resolve()` 规整，再校验落在目标 repo 内，拒绝 `..` 越界与绝对路径逃逸；
- 不把 issue 文本 / 文件名拼进 shell 字符串（调用方一律用 subprocess 的 list 形式）；
- 工作 repo 根本身也做 resolve，避免符号链接绕过。
"""
from __future__ import annotations

import os
from pathlib import Path

# 仓库根（task-6-coding-agent/）
TASK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = TASK_ROOT / "data" / "toy-repo"


class PathEscapeError(ValueError):
    """路径逃出工作 repo。"""


def default_repo_root() -> Path:
    """MCP server 独立启动时的工作 repo：环境变量 CODING_AGENT_REPO 优先。"""
    env = os.environ.get("CODING_AGENT_REPO")
    if env:
        return Path(env).expanduser().resolve()
    return DEFAULT_REPO.resolve()


def safe_resolve(root: str | Path, target: str, must_exist: bool = False) -> Path:
    """把 `target` 规整到 `root` 内的绝对路径，越界就抛 PathEscapeError。

    Args:
        root: 工作 repo 根目录。
        target: 用户 / 模型给出的相对（或绝对）路径。
        must_exist: 为 True 时要求目标必须已存在（读文件场景）。
    """
    root_path = Path(root).resolve()
    raw = Path(str(target))
    # 绝对路径一律按 root 内的相对路径解释，避免直接写到仓库外
    candidate = raw if raw.is_absolute() else root_path / raw
    resolved = candidate.resolve()

    if resolved != root_path and root_path not in resolved.parents:
        raise PathEscapeError(
            f"路径越界：{target!r} 不在工作 repo {root_path} 内（拒绝访问）")

    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"路径不存在：{target!r}")
    return resolved


def relative_to_root(root: str | Path, path: Path) -> str:
    """尽量返回相对路径，便于 trace / 输出可读。"""
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)
