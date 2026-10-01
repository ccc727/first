"""Agent 策略层：把「ReAct 循环」与「下一步该做什么」的决策解耦。

- :class:`LLMPolicy`  —— 默认策略，走 OpenAI 兼容接口（本地 Qwen 等），行为与原始
  手写 ReAct 完全一致；
- :class:`offline.OfflinePolicy` —— 确定性离线策略，仅在 LLM 端点不可用（连接失败 /
  404 model not found / 显式 ``REACT_OFFLINE=1``）时接管。

两种策略都只负责「下一步动作」，真正的工具调用、Observation 回填、trace 记录与
终止判定仍然由 :class:`src.agent.ReActAgent` 的同一个循环完成。
"""
from __future__ import annotations

from .base import LLMPolicy, PolicyDecision
from .offline import OfflinePolicy

__all__ = ["LLMPolicy", "OfflinePolicy", "PolicyDecision"]
