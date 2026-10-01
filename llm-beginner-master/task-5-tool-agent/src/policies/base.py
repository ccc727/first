"""策略协议与默认的 LLM 策略（保留原 ReAct 提示词 / 解析逻辑）。"""
from __future__ import annotations

import json
import re
from typing import Any


class PolicyDecision:
    """策略对「下一步」的决策。

    三选一：
    - ``tool=None, done=False``：格式错误，要求模型重试（只在 LLM 策略里出现）；
    - ``tool=<工具名>``：调用该工具，``action_input`` 为参数字典；
    - ``done=True``：本轮已经产出 ``final_answer``，循环结束。
    """

    __slots__ = ("tool", "action_input", "final_answer", "reasoning", "done")

    def __init__(self, tool: str | None = None, action_input: dict[str, Any] | None = None,
                 final_answer: str | None = None, reasoning: str = "",
                 done: bool = False):
        self.tool = tool
        self.action_input = action_input or {}
        self.final_answer = final_answer
        self.reasoning = reasoning
        self.done = done


def parse_react_reply(reply: str) -> PolicyDecision:
    """把模型回复解析成 Action / Final Answer（原实现的解析逻辑，行为保持不变）。"""
    m_final = re.search(r"Final Answer:\s*(.+)", reply, re.S)
    m_action = re.search(r"Action:\s*(\w+)", reply)
    m_input = re.search(r"Action Input:\s*(\{.*?\}|\[.*?\]|\S+)", reply, re.S)
    if m_final:
        return PolicyDecision(final_answer=m_final.group(1).strip(),
                              reasoning=reply, done=True)
    if not m_action or not m_input:
        return PolicyDecision(reasoning=reply)
    try:
        args = json.loads(m_input.group(1))
    except json.JSONDecodeError:
        return PolicyDecision(reasoning=reply)      # 参数不是合法 JSON -> 要求重试
    return PolicyDecision(tool=m_action.group(1), action_input=args, reasoning=reply)


class LLMPolicy:
    """默认策略：调用 OpenAI 兼容接口，按 Thought/Action 格式解析出下一步动作。

    ``available()`` 用于一次性探测端点是否真的能用；发现不可用时由 Agent 回退到
    OfflinePolicy。探测结果按「端点 + 模型」在**进程内缓存**：同一端点只探测一次，
    避免自检里每个 Agent 实例都重复等一遍连接超时。
    """

    # (base_url, model) -> 可用性；仅缓存失败结论（避免服务后起来时永久卡在离线）
    _probe_cache: dict[tuple[str, str], bool] = {}

    def __init__(self, chat_fn, system_prompt: str, tool_names: list[str],
                 probe_task: str | None = None, cache_key: tuple[str, str] | None = None):
        self.chat_fn = chat_fn
        self.system_prompt = system_prompt
        self.tool_names = list(tool_names)
        self.messages: list[dict] = []
        self._available: bool | None = None
        self._probe_task = probe_task
        self._cache_key = cache_key

    # ---- 可用性探测 -------------------------------------------------
    def available(self) -> bool:
        if self._available is None:
            if self._cache_key is not None and self._cache_key in self._probe_cache:
                self._available = self._probe_cache[self._cache_key]
            else:
                self._available = self._probe()
        return bool(self._available)

    def mark_unavailable(self) -> None:
        self._available = False
        if self._cache_key is not None:
            self._probe_cache[self._cache_key] = False

    def _probe(self) -> bool:
        try:
            reply = self.chat_fn(self._probe_messages())
        except Exception:       # noqa: BLE001 —— 连接失败 / 404 模型不存在 => 不可用
            ok = False
        else:
            ok = bool(reply and reply.strip())
        if self._cache_key is not None:
            self._probe_cache[self._cache_key] = ok
        return ok

    def _probe_messages(self) -> list[dict]:
        task = self._probe_task or "ping"
        return self.build_messages(task)

    # ---- 策略协议 ---------------------------------------------------
    def reset(self, task: str) -> list[dict]:
        messages = self.build_messages(task)
        self.messages = messages
        return messages

    def build_messages(self, task: str) -> list[dict]:
        return [{"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task}]

    def note_assistant(self, reply: str) -> None:
        self.messages.append({"role": "assistant", "content": reply})

    def note_observation(self, observation: str) -> None:
        self.messages.append({"role": "user", "content": f"Observation: {observation}"})

    def note_tool_result(self, tool: str, observation: str,
                         obs_index: int | None = None) -> None:
        """工具结果回填：LLM 策略只需把 Observation 拼回对话（保持原行为）。"""
        self.note_observation(observation)

    def note_format_error(self) -> None:
        self.messages.append({"role": "user",
                              "content": "格式错误：请按 Thought/Action/Action Input 输出。"})

    def note_bad_json(self) -> None:
        self.messages.append({"role": "user",
                              "content": "Action Input 不是合法 JSON，请重试。"})

    def next_decision(self, task: str, observations: list[str],
                      step_index: int) -> PolicyDecision:
        reply = self.chat_fn(self.messages)
        self.note_assistant(reply)
        return parse_react_reply(reply)
