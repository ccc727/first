"""Subagents：独立 context 的子 agent。

README 第 4-5 周要求「1-2 个独立 context 的子 agent，主 agent 只看摘要、不吞全部 trace」。
这里的隔离是**真的隔离**，不是换个函数名：

- 每个 subagent 有**自己的 message 列表**（不共用主 agent 的历史）；
- **自己的步数上限**（防止子任务跑飞拖垮主循环）；
- **自己的工具子集**（只拿需要的工具，减小误用面）；
- 对主 agent 只返回**结构化摘要**（工具调用记录 + 结论），原始 observation 不回传。

`Subagent` 基类负责这套隔离与执行；`CodeSearchSubagent` / `TestRunnerSubagent`
是可复用的两个实例，也作为「派发」模式的最小示例。
"""
from __future__ import annotations

import json
from pathlib import Path

from ..tools import call_tool, format_observation


class Subagent:
    """带独立 context / 工具子集 / 步数上限的子 agent 基类。"""

    name = "subagent"
    description = ""
    allowed_tools: tuple[str, ...] = ()
    max_steps = 6

    def __init__(self, repo_root: str | Path | None = None, model_client=None):
        self.repo_root = Path(repo_root).resolve() if repo_root else None
        self.model_client = model_client
        # 独立 message 列表：与主 agent 完全隔离，主 agent 拿不到它的中间过程
        self.messages: list[dict] = []

    # ------------------------------------------------------------------
    def run(self, task: str) -> dict:
        """执行子任务，返回**摘要**（不含完整 observation 流）。

        有可用模型时走 `_run_with_model`（真正的独立 context + 独立步数循环）；
        没有模型时走 `_run_offline`（确定性策略），二者都只回摘要。
        """
        self.messages = [{"role": "system", "content": self.system_prompt()}]
        self.messages.append({"role": "user", "content": task})

        if self.model_client is not None and getattr(self.model_client, "available", False):
            return self._run_with_model(task)
        return self._run_offline(task)

    # ------------------------------------------------------------------
    def system_prompt(self) -> str:
        return (f"你是 {self.name} 子 agent。{self.description}\n"
                f"你只能使用这些工具：{', '.join(self.allowed_tools)}。\n"
                "每次只输出一个动作，完成后用 Final Answer: 给出结论。")

    def _call(self, tool: str, args: dict) -> dict:
        """子 agent 调工具：强制限定在 allowed_tools 内。"""
        if tool not in self.allowed_tools:
            return {"ok": False, "tool": tool,
                    "error": f"{self.name} 无权调用工具 {tool!r}（可用：{self.allowed_tools}）"}
        return call_tool(tool, args, root=self.repo_root)

    # ------------------------------------------------------------------
    def _run_with_model(self, task: str) -> dict:
        """独立步数循环：自己的 messages、自己的 max_steps。"""
        tool_calls: list[dict] = []
        conclusion = ""
        for _ in range(self.max_steps):
            reply = self.model_client.chat(self.messages)
            self.messages.append({"role": "assistant", "content": reply})
            final = _extract(reply, "Final Answer")
            if final:
                conclusion = final
                break
            tool = _extract(reply, "Action")
            raw_args = _extract(reply, "Action Input")
            if not tool:
                self.messages.append({"role": "user",
                                      "content": "格式错误：请输出 Action / Action Input 或 Final Answer。"})
                continue
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                self.messages.append({"role": "user", "content": "Action Input 不是合法 JSON。"})
                continue
            payload = self._call(tool, args)
            tool_calls.append({"tool": tool, "args": args, "ok": payload.get("ok", False)})
            # observation 只留在子 agent 自己的 messages 里，不回传主 agent
            self.messages.append({"role": "user",
                                  "content": f"Observation: {format_observation(payload)}"})
        return self.summary(tool_calls, conclusion)

    def _run_offline(self, task: str) -> dict:
        raise NotImplementedError

    def summary(self, tool_calls: list[dict], conclusion: str) -> dict:
        """对主 agent 暴露的摘要：只有工具调用记录与结论。"""
        return {
            "subagent": self.name,
            "tool_calls": tool_calls,
            "n_steps": len(self.messages),
            "conclusion": conclusion,
            "isolated_context": True,      # 明示：原始 messages 不随摘要外传
        }


# ----------------------------------------------------------------------
class CodeSearchSubagent(Subagent):
    """代码搜索子 agent：定位与 issue 相关的文件与行。"""

    name = "code-search"
    description = "在仓库里定位与问题描述相关的源文件与关键行。"
    allowed_tools = ("list_dir", "read_file")
    max_steps = 4

    def _run_offline(self, task: str) -> dict:
        keywords = extract_keywords(task)
        hits: list[dict] = []
        tool_calls: list[dict] = []

        listing = self._call("list_dir", {"path": "."})
        tool_calls.append({"tool": "list_dir", "args": {"path": "."},
                           "ok": listing.get("ok", False)})
        entries = (listing.get("result") or {}).get("entries", []) if listing.get("ok") else []

        for entry in entries:
            if not entry.endswith(".py") or entry.endswith("test_calculator.py"):
                continue
            payload = self._call("read_file", {"path": entry})
            tool_calls.append({"tool": "read_file", "args": {"path": entry},
                               "ok": payload.get("ok", False)})
            if not payload.get("ok"):
                continue
            content = payload["result"]["content"]
            for lineno, line in enumerate(content.splitlines(), 1):
                # 命中的行：包含 issue 关键词，或像"有问题的实现"（返回/赋值）
                if any(k in line for k in keywords) or _looks_suspicious(line):
                    hits.append({"file": entry, "line": lineno, "text": line.strip()})

        conclusion = (f"在 {len(entries)} 个条目中定位到 {len(hits)} 个候选位置："
                      + "; ".join(f"{h['file']}:{h['line']} {h['text']}" for h in hits[:8]))
        if not hits:
            conclusion = "未定位到明显可疑位置，建议人工检查。"
        return self.summary(tool_calls, conclusion)


class TestRunnerSubagent(Subagent):
    """测试执行子 agent：跑测试并把失败摘要（而非完整日志）交回主 agent。"""

    name = "test-runner"
    description = "运行 pytest 并总结失败用例，不把完整日志带回主上下文。"
    allowed_tools = ("run_tests",)
    max_steps = 3

    def _run_offline(self, task: str) -> dict:
        payload = self._call("run_tests", {})
        tool_calls = [{"tool": "run_tests", "args": {}, "ok": payload.get("ok", False)}]
        if not payload.get("ok"):
            return self.summary(tool_calls, f"测试执行出错：{payload.get('error')}")
        result = payload["result"]
        output = result.get("output", "")
        failing = [ln.strip() for ln in output.splitlines()
                   if ln.strip().startswith(("FAILED", "E  ", "assert"))]
        conclusion = ("pytest 全部通过。" if result.get("passed")
                      else "pytest 失败，关键行：" + " | ".join(failing[:6]))
        return self.summary(tool_calls, conclusion)


# ----------------------------------------------------------------------
def default_subagents() -> dict[str, Subagent]:
    return {"code-search": CodeSearchSubagent(), "test-runner": TestRunnerSubagent()}


def extract_keywords(task: str) -> list[str]:
    """从 issue 文本里抽关键词：反引号包裹的标识符优先，其次英文标识符。"""
    import re

    quoted = re.findall(r"`([^`]+)`", task)
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_.]{2,}", task)
    out = [w for w in quoted]
    for w in words:
        if w not in out:
            out.append(w)
    return out


def _looks_suspicious(line: str) -> bool:
    """粗判一行是否像"被写坏的实现"：return / 赋值里出现算术运算。"""
    stripped = line.strip()
    if stripped.startswith("#"):
        return False
    return ("return " in stripped or "= " in stripped) and any(
        op in stripped for op in (" - ", " + ", " * ", " / ", "//"))


def _extract(reply: str, label: str) -> str:
    import re

    m = re.search(rf"{re.escape(label)}:\s*(.+)", reply or "")
    return m.group(1).strip() if m else ""
