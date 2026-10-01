"""离线路径冒烟测试（可复现的调试脚本，不参与评分）。

用法：
    & .venv\\Scripts\\python.exe eval/offline_smoke.py

做的事：
1. 强制 ``REACT_OFFLINE=1``，确认 Agent 走的是确定性离线策略；
2. 跑一遍 ``data/tasks.json`` 的 10 道题，打印每题命中的规则、真实调用的工具链、
   Observation 摘要与最终答案预览；
3. 复用 ``eval/run.py`` 的 ``answer_matches``（判定口径与自检完全一致）打印命中率。

注意：它只读 ``tasks.json`` 里的 ``task`` / ``expected_*`` 字段用于**报告**，
不把任何答案喂给 Agent —— Agent 的答案全部由真实工具输出拼出。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["REACT_OFFLINE"] = "1"

from eval.run import answer_matches, extract_used_tools      # noqa: E402
from src.agent import ReActAgent                             # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    tasks = json.loads((ROOT / "data" / "tasks.json").read_text(encoding="utf-8"))
    agent = ReActAgent()
    print(f"策略 = {agent.policy.name} | 回退原因 = {agent.fallback_reason or '(未回退)'}")
    ok = 0
    for t in tasks:
        trace = agent.run(t["task"])
        hit = answer_matches(trace["final_answer"], t.get("expected_answer_contains", []))
        ok += int(hit)
        print(f"\n=== 任务 {t['id']} {'PASS' if hit else 'FAIL'} ===")
        print(f"任务：{t['task']}")
        print(f"工具链：{extract_used_tools(trace)}")
        for s in trace["steps"]:
            if s.get("tool"):
                print(f"  step{s['step']} {s['tool']}({json.dumps(s['action_input'], ensure_ascii=False)})"
                      f" -> {str(s['observation'])[:160]}...")
        print(f"final_answer：{trace['final_answer'][:400]}")
    print(f"\n离线策略命中率：{ok}/{len(tasks)} = {ok / len(tasks):.3f}")
    return 0 if ok > 0.6 * len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
