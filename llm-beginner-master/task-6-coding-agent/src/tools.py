"""原子工具层（Tools）：MCP server 与主 agent 共用同一份实现。

五个工具（对应 README 第 2 周）：
- `read_file`   读 repo 内文本文件
- `write_file`  写 / 覆盖 repo 内文本文件
- `run_tests`   在 repo 内跑 `python -m pytest`
- `git_diff`    看当前工作区 diff
- `git_apply`   把 unified diff 应用到工作区（不碰 git 历史）

安全约定（逐条对应 README 第 2 周「安全」）：
- 所有路径经 `paths.safe_resolve()`，拒绝 `..` 越界与绝对路径逃逸；
- git / pytest 一律 `subprocess.run([...])` list 形式，不用 `shell=True`，并限定 `cwd`；
- 拒绝任何会丢未提交改动 / 删文件的 git 子命令（`reset --hard` / `clean` / `checkout --`）；
- 工具异常一律转成结构化 error 返回，不让调用方（MCP server / agent 循环）崩掉。

`SPECS` 是唯一事实来源：`list_specs()` 供 MCP server 与自检枚举，
`call_tool()` 是统一调用入口。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from .paths import PathEscapeError, relative_to_root, safe_resolve

# 单次返回给模型的最大字符数，避免一个工具把上下文撑爆
MAX_OUTPUT_CHARS = 8000

# git 里会丢数据 / 删文件的危险子命令，一律拒绝
FORBIDDEN_GIT_TOKENS = (
    "reset --hard", "clean -fd", "clean -f", "checkout --", "checkout .",
    "restore .", "restore --staged .", "filter-branch", "push --force",
)


def _clip(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[已截断，共 {len(text)} 字符]"


def _run(cmd: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess:
    """list 形式 + 限定时限执行子进程（绝不 shell=True）。"""
    return subprocess.run(
        cmd, cwd=str(cwd), text=True, capture_output=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )


def _pytest_python() -> str:
    """优先使用本仓库 .venv 的解释器（自带 pytest），否则退回当前解释器。"""
    venv_py = Path(sys.executable)
    candidate = venv_py.parent / "python.exe"
    if candidate.exists() and (venv_py.parent.parent / "Lib" / "site-packages").exists():
        return str(candidate)
    return sys.executable


# --------------------------------------------------------------------------
# 工具实现：签名统一为 fn(root: Path, args: dict) -> dict
# --------------------------------------------------------------------------

def _read_file(root: Path, args: dict) -> dict:
    target = safe_resolve(root, args.get("path", ""), must_exist=True)
    if target.is_dir():
        raise IsADirectoryError(f"{args.get('path')!r} 是目录，不是文件")
    text = target.read_text(encoding="utf-8", errors="replace")
    return {
        "path": relative_to_root(root, target),
        "content": _clip(text),
        "lines": text.count("\n") + 1,
    }


def _write_file(root: Path, args: dict) -> dict:
    target = safe_resolve(root, args.get("path", ""))
    if target.is_dir():
        raise IsADirectoryError(f"{args.get('path')!r} 是目录，不能当文件写")
    content = args.get("content")
    if content is None:
        raise ValueError("缺少参数 content")
    existed = target.exists()
    before = target.read_text(encoding="utf-8", errors="replace") if existed else ""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(content), encoding="utf-8", newline="\n")
    return {
        "path": relative_to_root(root, target),
        "created": not existed,
        "changed": before != str(content),
        "bytes": len(str(content).encode("utf-8")),
    }


def _run_tests(root: Path, args: dict) -> dict:
    test_path = args.get("path")
    cmd = [_pytest_python(), "-m", "pytest", "-q"]
    if test_path:
        # 只允许 repo 内的路径；把绝对路径转成相对 root 的形式再传给 pytest
        cmd.append(relative_to_root(root, safe_resolve(root, test_path, must_exist=True)))
    try:
        proc = _run(cmd, cwd=root, timeout=int(args.get("timeout", 180)))
    except subprocess.TimeoutExpired:
        return {"passed": False, "returncode": -1, "output": "[超时] pytest 超过时限"}
    output = _clip((proc.stdout or "") + (proc.stderr or ""))
    return {"passed": proc.returncode == 0, "returncode": proc.returncode,
            "output": output}


def _git_diff(root: Path, args: dict) -> dict:
    path = args.get("path")
    cmd = ["git", "diff", "--no-color"]
    if path:
        cmd += ["--", relative_to_root(root, safe_resolve(root, path))]
    proc = _run(cmd, cwd=root, timeout=60)
    if proc.returncode != 0 and "not a git repository" in (proc.stderr or ""):
        return {"is_git": False, "diff": "", "note": "不是 git 仓库"}
    return {"is_git": True, "diff": _clip(proc.stdout or ""),
            "stderr": (proc.stderr or "").strip()[:500]}


def _git_apply(root: Path, args: dict) -> dict:
    patch = str(args.get("patch", ""))
    if not patch.strip():
        raise ValueError("缺少参数 patch")

    normalized = " ".join(patch.split()).lower()
    for token in FORBIDDEN_GIT_TOKENS:
        if token in normalized:
            raise ValueError(f"拒绝执行含危险操作的补丁：{token}")

    def _apply(extra: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "apply", *extra, "-"], cwd=str(root), input=patch, text=True,
            capture_output=True, encoding="utf-8", errors="replace", timeout=60)

    check = _apply(["--check"])
    if check.returncode == 0:
        proc = _apply([])
        return {"applied": proc.returncode == 0, "check_output": "",
                "output": _clip((proc.stdout or "") + (proc.stderr or "")),
                "mode": "direct"}

    # 回退：--3way 能处理"上下文能对齐、但目标文件尚未进 index"的部分情况
    check3 = _apply(["--check", "--3way"])
    if check3.returncode == 0:
        proc = _apply(["--3way"])
        return {"applied": proc.returncode == 0, "check_output": "",
                "output": _clip((proc.stdout or "") + (proc.stderr or "")),
                "mode": "3way"}

    # 两者都失败：多半是目标文件根本不在 index 里（untracked 新文件 git apply 拒绝处理）。
    # 这里如实回报，不静默"假装成功"——需要改新文件时请直接用 write_file 写完整内容。
    return {"applied": False, "check_output": _clip(check.stderr or ""),
            "check_output_3way": _clip(check3.stderr or ""),
            "note": ("补丁无法应用（--check 与 --3way 均失败）。"
                     "若目标是尚未入 index 的新文件，git apply 不会处理，请改用 write_file。")}


def _list_dir(root: Path, args: dict) -> dict:
    target = safe_resolve(root, args.get("path", "."), must_exist=True)
    if not target.is_dir():
        raise NotADirectoryError(f"{args.get('path')!r} 不是目录")
    entries = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir())
    return {"path": relative_to_root(root, target), "entries": entries[:200],
            "count": len(entries)}


# --------------------------------------------------------------------------
# 工具规格表：name / description / input_schema + 实现
# --------------------------------------------------------------------------
SPECS: list[dict] = [
    {
        "name": "read_file",
        "description": "读取工作 repo 内某个文本文件的内容。参数 path 为 repo 内相对路径。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "repo 内相对路径，如 calculator.py"},
            },
            "required": ["path"],
            "additionalProperties": False,
        },
        "fn": _read_file,
    },
    {
        "name": "write_file",
        "description": "把 content 写入工作 repo 内的 path（整文件覆盖，父目录自动创建）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "repo 内相对路径"},
                "content": {"type": "string", "description": "要写入的完整文件内容"},
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
        "fn": _write_file,
    },
    {
        "name": "run_tests",
        "description": "在工作 repo 内运行 python -m pytest -q，返回是否通过（passed）与输出。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "可选：只跑某个测试文件"},
                "timeout": {"type": "integer", "description": "超时秒数，默认 180"},
            },
            "additionalProperties": False,
        },
        "fn": _run_tests,
    },
    {
        "name": "git_diff",
        "description": "查看工作 repo 当前未提交改动的 unified diff。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "可选：只看某个文件"},
            },
            "additionalProperties": False,
        },
        "fn": _git_diff,
    },
    {
        "name": "git_apply",
        "description": "把一段 unified diff 补丁应用到工作 repo（只改工作区，不动 git 历史）。",
        "input_schema": {
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "unified diff 文本"},
            },
            "required": ["patch"],
            "additionalProperties": False,
        },
        "fn": _git_apply,
    },
    {
        "name": "list_dir",
        "description": "列出工作 repo 内某个目录的文件与子目录名。",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "repo 内相对目录，默认 ."},
            },
            "additionalProperties": False,
        },
        "fn": _list_dir,
    },
]

TOOLS_BY_NAME = {spec["name"]: spec for spec in SPECS}


def list_specs() -> list[dict]:
    """只返回 name / description / input_schema（不含实现），供 MCP 与自检使用。"""
    return [{"name": s["name"], "description": s["description"],
             "input_schema": s["input_schema"]} for s in SPECS]


def call_tool(name: str, args: dict | None = None, root: str | Path | None = None) -> dict:
    """统一调用入口：永远返回结构化 dict，工具异常转成 `{"ok": False, "error": ...}`。

    这样 MCP server 与 agent 循环都不需要自己 try/except，也不会被工具异常打断。
    """
    from .paths import default_repo_root

    args = args or {}
    repo = Path(root).resolve() if root else default_repo_root()
    spec = TOOLS_BY_NAME.get(name)
    if spec is None:
        return {"ok": False, "error": f"未知工具 {name!r}",
                "available": sorted(TOOLS_BY_NAME)}
    try:
        result = spec["fn"](repo, args)
        return {"ok": True, "tool": name, "result": result}
    except (PathEscapeError, FileNotFoundError, IsADirectoryError,
            NotADirectoryError, ValueError, KeyError, TypeError) as e:
        return {"ok": False, "tool": name, "error": f"{type(e).__name__}: {e}",
                "error_type": type(e).__name__}
    except subprocess.TimeoutExpired as e:
        return {"ok": False, "tool": name, "error": f"命令超时：{e}",
                "error_type": "TimeoutExpired"}
    except Exception as e:  # 兜底：任何工具异常都不能让 server 崩
        return {"ok": False, "tool": name, "error": f"{type(e).__name__}: {e}",
                "error_type": type(e).__name__}


def format_observation(payload: dict) -> str:
    """把 call_tool 的结构化结果压成一段给模型看的 Observation 文本。"""
    import json

    if not payload.get("ok"):
        return f"[工具错误] {payload.get('error')}"
    result = payload.get("result", {})
    if isinstance(result, dict) and "content" in result:
        return f"文件 {result.get('path')}（{result.get('lines')} 行）：\n{result['content']}"
    if isinstance(result, dict) and "output" in result and "passed" in result:
        status = "通过" if result["passed"] else "失败"
        return f"pytest {status}（returncode={result['returncode']}）：\n{result['output']}"
    if isinstance(result, dict) and "diff" in result:
        return result["diff"] or "[工作区无改动]"
    return json.dumps(result, ensure_ascii=False)[:MAX_OUTPUT_CHARS]
