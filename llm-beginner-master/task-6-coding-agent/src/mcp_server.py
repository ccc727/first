"""任务六的 MCP server：把 `tools.py` 的原子工具通过 MCP 协议暴露出去。

两种启动方式：
    python src/mcp_server.py                  # 官方 MCP SDK（stdio，默认）
    python src/mcp_server.py --plain-jsonrpc  # 内置 JSON-RPC 回退（不依赖 SDK）
    python src/mcp_server.py --list           # 只打印工具清单后退出（排查用）

模块顶层额外导出 `list_tools()`（返回 name / description / input_schema 列表），
自检 `mcp_server_lists_tools` 正是靠它枚举工具，因此走 import 这条路不需要起进程。

工作 repo 由 `--repo` 或环境变量 `CODING_AGENT_REPO` 指定，默认 `data/toy-repo`。

关于 SDK 版本：requirements 写的是 `mcp>=0.9`，但 mcp 2.x 是一次破坏性重写
（`FastMCP` 改名 `MCPServer`，`Server.list_tools()` 装饰器改为 `on_list_tools=`
构造参数）。这里按 **2.x 的低层 API** 实现，并对协议错误返回结构化错误、
对工具异常做兜底，保证 server 不会因为一次坏调用而崩。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):                       # 支持 python src/mcp_server.py 直接跑
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paths import default_repo_root              # noqa: E402
from src.tools import call_tool, list_specs          # noqa: E402

SERVER_NAME = "coding-agent-mcp"
SERVER_VERSION = "1.0.0"


def list_tools() -> list[dict]:
    """模块顶层导出：返回工具清单（每项至少含 name）。

    自检 `mcp_server_lists_tools` 直接 import 本函数；`tools.py` 的 SPECS 是唯一事实来源。
    """
    return list_specs()


# --------------------------------------------------------------------------
# 官方 MCP SDK（mcp 2.x 低层 Server）
# --------------------------------------------------------------------------

async def _run_sdk_server(repo_root: Path) -> None:
    import mcp.types as types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    server = Server(
        SERVER_NAME,
        version=SERVER_VERSION,
        instructions="本地代码仓库工具集：读写文件、跑测试、看/应用 diff。",
        on_list_tools=_make_list_tools_handler(types),
        on_call_tool=_make_call_tool_handler(types, repo_root),
    )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream,
                         server.create_initialization_options())


def _make_list_tools_handler(types):
    async def handle_list_tools(ctx, params):
        tools = [
            types.Tool(
                name=s["name"],
                description=s["description"],
                input_schema=s["input_schema"],
            )
            for s in list_tools()
        ]
        return types.ListToolsResult(tools=tools)
    return handle_list_tools


def _make_call_tool_handler(types, repo_root: Path):
    async def handle_call_tool(ctx, params):
        name = getattr(params, "name", None)
        arguments = getattr(params, "arguments", None) or {}
        payload = call_tool(name, arguments, root=repo_root)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=text)],
            structured_content=payload,
            is_error=not payload.get("ok", False),
        )
    return handle_call_tool


# --------------------------------------------------------------------------
# 回退：内置 JSON-RPC 2.0 over stdio（NDJSON 分帧），不依赖 SDK 是否装好
# --------------------------------------------------------------------------

def _plain_jsonrpc_server(repo_root: Path) -> int:
    """最小 MCP 子集：initialize / tools/list / tools/call。

    每行一个 JSON-RPC 消息（NDJSON）。这条路径用于 SDK 缺失或版本不兼容时
    仍能完成 stdio 握手与工具枚举，保证 `python src/mcp_server.py --plain-jsonrpc`
    始终可用。
    """
    log(f"[plain-jsonrpc] 工作 repo = {repo_root}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _write_message({"jsonrpc": "2.0", "id": None,
                            "error": {"code": -32700, "message": f"解析失败：{e}"}})
            continue

        req_id = req.get("id")
        method = req.get("method")
        params = req.get("params") or {}

        if method == "initialize":
            result = {
                "protocolVersion": params.get("protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            }
        elif method == "notifications/initialized":
            continue                                   # 通知，无需响应
        elif method == "tools/list":
            result = {"tools": [
                {"name": s["name"], "description": s["description"],
                 "inputSchema": s["input_schema"]} for s in list_tools()
            ]}
        elif method == "tools/call":
            payload = call_tool(params.get("name"), params.get("arguments") or {},
                                root=repo_root)
            result = {"content": [{"type": "text",
                                   "text": json.dumps(payload, ensure_ascii=False)}],
                      "structuredContent": payload,
                      "isError": not payload.get("ok", False)}
        elif method == "ping":
            result = {}
        else:
            _write_message({"jsonrpc": "2.0", "id": req_id,
                            "error": {"code": -32601,
                                      "message": f"未实现的方法：{method}"}})
            continue

        if req_id is not None:
            _write_message({"jsonrpc": "2.0", "id": req_id, "result": result})
    return 0


def _write_message(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def log(*parts) -> None:
    """日志一律走 stderr —— stdout 是协议通道，不能混入任何非协议内容。"""
    print(*parts, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="任务六 MCP server")
    parser.add_argument("--repo", default=None,
                        help="工作 repo 根目录（默认 data/toy-repo）")
    parser.add_argument("--plain-jsonrpc", action="store_true",
                        help="使用内置 JSON-RPC 回退实现（不依赖 mcp SDK）")
    parser.add_argument("--list", action="store_true", help="打印工具清单后退出")
    args = parser.parse_args(argv)

    repo_root = Path(args.repo).resolve() if args.repo else default_repo_root()

    if args.list:
        print(json.dumps(list_tools(), ensure_ascii=False, indent=2))
        return 0

    if args.plain_jsonrpc or os.environ.get("MCP_PLAIN_JSONRPC") == "1":
        return _plain_jsonrpc_server(repo_root)

    try:
        asyncio.run(_run_sdk_server(repo_root))
        return 0
    except ImportError as e:
        log(f"[警告] 未安装官方 mcp SDK（{e}），回退到内置 JSON-RPC 实现。")
        return _plain_jsonrpc_server(repo_root)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
