"""任务六：Mini Coding Agent 的实现包。

三层栈：
- Tools / MCP：`tools.py`（原子工具）+ `mcp_server.py`（stdio MCP server）
- Skills：`skill_loader.py` + `skills/*/SKILL.md`
- Subagents：`subagents/`（独立 context 的子 agent）

顶层 `agent.py` 的 `CodingAgent.run()` 把三层串成 agentic loop。

注意：这里用 `__getattr__` 做惰性导出，**不要**在包导入时 eager-import `agent`。
`python src/mcp_server.py` 会先导入 `src` 包，若此处 eager 导入 agent，
一旦 agent 依赖（openai 等）有问题，MCP server 的独立启动就会连带失败。
"""
from __future__ import annotations

__all__ = ["CodingAgent", "Trace"]


def __getattr__(name: str):
    if name in __all__:
        from . import agent as _agent
        return getattr(_agent, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
