"""受限 Python 沙箱工具（教学级防护：白名单 builtins + 禁止 import + 超时）。

⚠️ 仅用于可信/自产代码；黑名单式防护不是真正的隔离。
"""
from __future__ import annotations

import contextlib
import io
import threading

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "python_sandbox",
        "description": "在受限环境中执行 Python 代码并返回 stdout（不允许 import 文件系统等）。",
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "Python 代码"}},
            "required": ["code"],
        },
    },
}

_ALLOWED_BUILTINS = {
    "print", "len", "range", "sum", "min", "max", "abs", "round", "int", "float",
    "str", "list", "dict", "set", "tuple", "bool", "enumerate", "zip", "sorted",
    "reversed", "any", "all", "map", "filter", "pow", "divmod", "isinstance",
    "chr", "ord", "True", "False", "None", "Exception", "ValueError", "TypeError",
}


def _safe_exec(code: str) -> str:
    buf = io.StringIO()
    builtins_ = {k: __builtins__[k] for k in _ALLOWED_BUILTINS if k in __builtins__}
    g = {"__builtins__": builtins_}
    with contextlib.redirect_stdout(buf):
        exec(code, g)
    return buf.getvalue() or "(无输出)"


def run(args: dict) -> str:
    code = str(args.get("code", ""))
    if not code.strip():
        raise ValueError("code 不能为空")
    result = {}

    def worker():
        try:
            result["ok"] = True
            result["out"] = _safe_exec(code)
        except Exception as e:  # noqa: BLE001
            result["ok"] = False
            result["err"] = f"{type(e).__name__}: {e}"

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout=5)
    if t.is_alive():
        raise TimeoutError("代码执行超时（>5s）")
    if not result.get("ok"):
        raise RuntimeError(result.get("err", "exec 失败"))
    return str(result["out"])
