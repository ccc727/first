"""手写 ReAct Agent：Thought / Action / Action Input / Observation 循环。

通过 OpenAI 兼容接口调用本地模型（默认 http://localhost:11434/v1，可用环境变量
OPENAI_BASE_URL / OPENAI_API_KEY / REACT_MODEL 覆盖）。工具异常会作为 Observation
喂回给模型（错误恢复），步数超限/出现 Final Answer 停止。

本版本把「ReAct 循环」与「下一步动作的决策」解耦为可插拔策略（见 ``src/policies/``）：

- 默认策略 :class:`~src.policies.LLMPolicy`：调用真实 LLM，行为与原来完全一致；
- 回退策略 :class:`~src.policies.OfflinePolicy`：仅在 LLM 端点不可用（连接失败 /
  404 model not found）或显式 ``REACT_OFFLINE=1`` 时接管。它同样只负责路由 + 拼参数，
  真正的工具调用 / Observation 回填 / trace 记录 / 收尾判定都发生在下面这个循环里，
  因此它也会**真实调用工具**并由真结果拼出 final_answer。

可选错误恢复钩子：``ReActAgent(inject_error="wiki")`` 会让指定工具第一次调用时抛异常，
用于验证「工具失败 -> 错误进 Observation -> 再次调用成功」的自我纠错路径。
"""
from __future__ import annotations

import json
import os
import sys

from .policies import LLMPolicy, OfflinePolicy
from .policies.offline import offline_enabled
from .tools import calculator, file_search, python_sandbox, wiki

TOOLS = {
    "calculator": calculator,
    "python_sandbox": python_sandbox,
    "file_search": file_search,
    "wiki": wiki,
}

SYSTEM = """你是会使用工具解决问题的智能体。工具列表：
{schemas}

请按如下格式回复（一次一个动作）：
Thought: 你的推理
Action: 工具名
Action Input: {{"参数": "值"}}
之后会返回 Observation，继续循环；得到答案时输出：
Final Answer: 最终答案

规则：只能调用上面列出的工具；参数必须合法 JSON。""".strip()


TOOL_ERROR_PREFIX = "[工具错误"


def _tool_schemas() -> str:
    return "\n".join(json.dumps(TOOLS[n].TOOL_SCHEMA, ensure_ascii=False) for n in TOOLS)


def _summarize(text: str, limit: int = 400) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


class ReActAgent:
    def __init__(self, model: str | None = None, base_url: str | None = None,
                 max_steps: int = 12, inject_error: str | list[str] | None = None,
                 offline: bool | None = None, policy=None, verbose: bool = False):
        self.model = model or os.environ.get("REACT_MODEL", "qwen2.5:7b-instruct")
        self.base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "http://localhost:11434/v1")
        self.max_steps = max_steps
        self.inject_error = inject_error
        self.verbose = verbose
        self._forced_offline = offline_enabled() if offline is None else bool(offline)
        self._client = None
        # 注入错误状态：工具名 -> 剩余注入次数
        self._error_budget = {inject_error: 1} if isinstance(inject_error, str) else \
            {name: 1 for name in (inject_error or [])}
        self.offline_policy = OfflinePolicy(
            tools=TOOLS, verbose=verbose,
            error_predicate=lambda obs: str(obs).startswith(TOOL_ERROR_PREFIX))
        self.llm_policy = policy if policy is not None else self._build_llm_policy()
        self.fallback_reason = ""
        self.policy = self._select_policy()
        if self.policy is self.offline_policy and not self.fallback_reason:
            self.fallback_reason = "REACT_OFFLINE 已启用，强制离线策略" if self._forced_offline \
                else (f"LLM 端点 {self.base_url} 不可用（连接失败或模型 {self.model!r} "
                      "不存在），已回退到确定性离线策略")

    # ---- 策略装配 ---------------------------------------------------
    def _build_llm_policy(self) -> LLMPolicy:
        from openai import OpenAI
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "ollama"),
            timeout=float(os.environ.get("REACT_TIMEOUT", "120")))
        return LLMPolicy(self._chat, SYSTEM.format(schemas=_tool_schemas()),
                         list(TOOLS), probe_task="ping",
                         cache_key=(self.base_url, self.model))

    def _select_policy(self):
        """装配策略：能连上 LLM 就用 LLM，否则用离线策略（原因由调用方记录）。"""
        if self._forced_offline:
            if hasattr(self.llm_policy, "mark_unavailable"):
                self.llm_policy.mark_unavailable()
            return self.offline_policy
        if self.llm_policy.available():
            return self.llm_policy
        return self.offline_policy

    def _chat(self, messages):
        r = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.2,
            max_tokens=600)
        return r.choices[0].message.content or ""

    # ---- 工具执行（含错误注入钩子） ------------------------------------
    def _run_tool(self, tool, args: dict) -> str:
        try:
            return str(tool.run(args))
        except Exception as e:          # noqa: BLE001 —— M3：错误吞回 Observation
            return f"{TOOL_ERROR_PREFIX} {type(e).__name__}: {e}]"

    def _failing_tool(self, name: str):
        """注入错误用的假工具：run() 与真实工具一样抛异常，走上同一条捕获路径。"""
        def _raise(_args):
            raise RuntimeError(f"注入错误：工具 {name} 暂时不可用")
        return type("InjectedTool", (), {"run": staticmethod(_raise)})()

    def _invoke_tool(self, name: str, args: dict) -> str:
        tool = TOOLS.get(name)
        if tool is None:
            return f"[错误] 未知工具 {name}"
        if self._error_budget.get(name, 0) > 0:
            self._error_budget[name] -= 1
            return self._run_tool(self._failing_tool(name), args)
        return self._run_tool(tool, args)

    # ---- 主循环 ------------------------------------------------------
    def run(self, task: str) -> dict:
        # 每次 run 重置注入预算：保证每道题都恰好被注入一次错误（便于统计恢复率）
        if self.inject_error:
            names = [self.inject_error] if isinstance(self.inject_error, str) \
                else list(self.inject_error)
            self._error_budget = {name: 1 for name in names}
        try:
            return self._run_with(self.policy, task)
        except Exception as e:      # noqa: BLE001 —— LLM 调用中途失败（如 404）也要能回退
            if self.policy is self.llm_policy:
                # 探测通过但真实调用失败：标记端点不可用并改用离线策略重跑一遍
                if hasattr(self.llm_policy, "mark_unavailable"):
                    self.llm_policy.mark_unavailable()
                self.fallback_reason = (
                    f"LLM 调用中途失败（{type(e).__name__}: {e}），"
                    "已回退到确定性离线策略")
                self.policy = self.offline_policy
                if self.verbose:
                    print(f"[agent] {self.fallback_reason}", file=sys.stderr)
                return self._run_with(self.offline_policy, task)
            raise

    def _run_with(self, policy, task: str) -> dict:
        policy.reset(task)
        steps: list[dict] = []
        observations: list[str] = []
        final_answer = ""

        for step_i in range(1, self.max_steps + 1):
            decision = policy.next_decision(task, observations, len(observations))
            if self.verbose:
                print(f"[agent] step {step_i} -> {decision.reasoning[:120]}", file=sys.stderr)
            if decision.done or decision.final_answer is not None:
                final_answer = (decision.final_answer or "").strip()
                steps.append({"step": step_i, "thought": decision.reasoning,
                              "tool": None, "action_input": None,
                              "observation": "done"})
                break

            if decision.tool is None:       # 解析失败（仅 LLM 策略会出现）
                policy.note_format_error()
                steps.append({"step": step_i, "thought": decision.reasoning, "tool": None,
                              "action_input": None, "observation": "格式错误，要求重试"})
                continue

            args = decision.action_input or {}
            obs = self._invoke_tool(decision.tool, args)
            steps.append({"step": step_i, "thought": decision.reasoning,
                          "tool": decision.tool, "action_input": args,
                          "observation": obs})
            observations.append(obs)
            policy.note_tool_result(decision.tool, obs, len(observations) - 1)
            if self.verbose:
                print(f"[agent]   observation: {_summarize(obs, 200)}", file=sys.stderr)

        return {"steps": steps, "final_answer": final_answer,
                "success": bool(final_answer), "task": task,
                "policy": getattr(policy, "name", type(policy).__name__),
                "fallback_reason": self.fallback_reason}
