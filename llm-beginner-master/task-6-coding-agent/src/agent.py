"""主 agent：agentic loop `while not done: model → tool → observation → loop`。

数据流（对应 README 第 4-5 周）：

    issue ──> SkillLoader.match()      选中的 skill 正文注入上下文（渐进式披露）
          ──> dispatch Subagent(s)     代码搜索 / 测试执行（独立 context，只回摘要）
          ──> agent loop               模型提出 tool_call → 执行 → observation 回填
          ──> 停机                     测试全绿 / 模型显式声明完成
          ──> Trace                    steps + patch + tests_passed

两种策略：
- **模型驱动**（`ModelClient.available` 为 True）：调本地 Qwen2.5-Coder-7B-Instruct 的
  OpenAI 兼容端点，模型自己决定每步调哪个工具。
- **离线确定性回退**：本机没有可用本地模型时启用。它不是"假装调了模型"——它走**同一套
  工具**（`tools.call_tool`）与**同一条 trace 结构**，只是由确定性策略挑工具与参数：
  先派 code-search 子 agent 定位嫌疑行，再对候选修复逐个跑 `run_tests` 验证，最后产出 patch。
  覆盖常见机械型缺陷（运算符写反、边界错），语义型缺陷仍需要真实模型。

Trace 是 `dict` 子类，含自检要求的 `steps` / `patch` / `tests_passed` 三个键。
"""
from __future__ import annotations

import difflib
import json
import os
import re
from pathlib import Path

from .skill_loader import SkillLoader
from .subagents import CodeSearchSubagent, TestRunnerSubagent
from .tools import call_tool, format_observation

TASK_ROOT = Path(__file__).resolve().parents[1]
SKILLS_DIR = TASK_ROOT / "src" / "skills"

# 离线回退时尝试的算术运算符替换，按"常见程度"排序
CANDIDATE_OPERATORS = ["+", "-", "*", "/", "//", "%"]
ARITH_RE = re.compile(r"(?<=[\w\)\]]) ?([+\-*/%]) ?(?=[\w\(\[.])")


class Trace(dict):
    """自检按 `trace.get(...)` 取值，因此做成 dict 子类并保证三个键都在。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setdefault("steps", [])
        self.setdefault("patch", "")
        self.setdefault("tests_passed", False)

    def add_step(self, thought: str, tool: str | None, args: dict | None,
                 observation: str, **extra) -> None:
        step = {
            "step": len(self["steps"]) + 1,
            "thought": thought,
            "tool_call": {"name": tool, "arguments": args or {}} if tool else None,
            "observation": _clip(observation, 1200),
        }
        step.update(extra)
        self["steps"].append(step)


def _clip(text: str, limit: int) -> str:
    text = str(text)
    return text if len(text) <= limit else text[:limit] + "…[截断]"


# ----------------------------------------------------------------------
# 模型客户端
# ----------------------------------------------------------------------
class ModelClient:
    """OpenAI 兼容客户端（本地 Qwen2.5-Coder-7B-Instruct）。

    可用性在 `probe()` 里判定：连不上 / 模型不存在 → `available = False`，
    主 agent 自动切到离线确定性策略。
    """

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 timeout: int = 120):
        self.base_url = base_url or os.environ.get(
            "OPENAI_BASE_URL", "http://localhost:11434/v1")
        self.api_key = os.environ.get("OPENAI_API_KEY", "ollama")
        self.model = model or os.environ.get("CODING_AGENT_MODEL",
                                             "qwen2.5-coder:7b-instruct")
        self.timeout = timeout
        self.available = False
        self.unavailable_reason = "未探测"
        self._client = None

    def probe(self) -> bool:
        """探测端点：连得上且模型列表包含目标模型才算可用。"""
        if os.environ.get("CODING_AGENT_OFFLINE") == "1":
            self.unavailable_reason = "环境变量 CODING_AGENT_OFFLINE=1，强制离线"
            self.available = False
            return False
        try:
            from openai import OpenAI
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key,
                                  timeout=self.timeout)
            models = self._client.models.list()
            ids = [getattr(m, "id", "") for m in getattr(models, "data", []) or []]
            if not ids:
                self.unavailable_reason = f"{self.base_url} 无可用模型（模型列表为空）"
                self.available = False
                return False
            if self.model not in ids:
                self.unavailable_reason = (
                    f"{self.base_url} 没有模型 {self.model!r}（已有：{ids}）")
                self.available = False
                return False
            self.available = True
            self.unavailable_reason = ""
            return True
        except Exception as e:
            self.unavailable_reason = f"{type(e).__name__}: {e}"
            self.available = False
            return False

    def chat(self, messages: list[dict]) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=0.1, max_tokens=800)
        return resp.choices[0].message.content or ""


# ----------------------------------------------------------------------
# 主 agent
# ----------------------------------------------------------------------
class CodingAgent:
    """Mini Coding Agent：读 issue → 定位 → 改码 → 跑测试 → 出 patch。"""

    def __init__(self, model_client: ModelClient | None = None,
                 skills_dir: str | Path | None = None,
                 max_steps: int = 14, use_subagents: bool = True):
        self.model_client = model_client or ModelClient()
        self.skills = SkillLoader(str(skills_dir or SKILLS_DIR))
        self.max_steps = max_steps
        self.use_subagents = use_subagents
        self._subagents: dict = {}

    # ------------------------------------------------------------------
    def run(self, repo_path: str, issue: str) -> Trace:
        repo = Path(repo_path).resolve()
        trace = Trace(repo=str(repo), issue=issue, strategy="", skill=None,
                      subagent_summaries=[])
        if not repo.is_dir():
            trace["error"] = f"repo 不存在：{repo}"
            return trace

        # 1) Skill：只把命中的 skill 正文注入上下文（渐进式披露）
        hits = self.skills.match(issue, top_k=1)
        trace["skill"] = hits[0]["name"] if hits else None
        skill_body = hits[0]["body"] if hits else ""
        if hits:
            trace.add_step(
                thought=f"任务描述命中 skill「{hits[0]['name']}」，加载其正文（score={hits[0]['score']}）",
                tool=None, args=None,
                observation=f"已加载 skill 正文 {len(skill_body)} 字符（仅命中项，未全量读入）",
                kind="skill_load")

        # 2) 探测模型：决定走模型驱动还是离线确定性策略
        self.model_client.probe()
        trace["model_available"] = self.model_client.available
        trace["model_unavailable_reason"] = self.model_client.unavailable_reason
        trace["strategy"] = "model" if self.model_client.available else "offline-heuristic"
        trace.add_step(
            thought=("本地模型可用，走模型驱动的 ReAct 循环。"
                     if self.model_client.available else
                     f"本地模型不可用（{self.model_client.unavailable_reason}），"
                     "切换到离线确定性策略（同一套工具与 trace 结构）。"),
            tool=None, args=None, observation=f"strategy={trace['strategy']}",
            kind="strategy_choice")

        # 3) 派发 Subagent 定位与基线（主 agent 只拿摘要）
        if self.use_subagents:
            self._dispatch_subagents(repo, issue, trace)

        # 4) agent loop
        if self.model_client.available:
            self._run_model_loop(repo, issue, skill_body, trace)
        else:
            self._run_offline_loop(repo, issue, trace)

        # 5) 收尾：终检 + 生成 patch
        self._finalize(repo, trace)
        return trace

    # ------------------------------------------------------------------
    # Subagent 派发
    # ------------------------------------------------------------------
    def _dispatch_subagents(self, repo: Path, issue: str, trace: Trace) -> None:
        search = CodeSearchSubagent(repo_root=repo, model_client=self.model_client)
        summary = search.run(issue)
        trace["subagent_summaries"].append(summary)
        trace.add_step(
            thought="派发 code-search 子 agent（独立 context）定位相关文件与嫌疑行",
            tool=None, args=None,
            observation=f"code-search 摘要：{summary['conclusion']}",
            kind="subagent", subagent="code-search")

        tester = TestRunnerSubagent(repo_root=repo, model_client=self.model_client)
        t_summary = tester.run(issue)
        trace["subagent_summaries"].append(t_summary)
        trace["baseline_tests_passed"] = "全部通过" in t_summary["conclusion"]
        trace.add_step(
            thought="派发 test-runner 子 agent（独立 context）建立测试基线",
            tool=None, args=None,
            observation=f"test-runner 摘要：{t_summary['conclusion']}",
            kind="subagent", subagent="test-runner")

    # ------------------------------------------------------------------
    # 策略 A：模型驱动
    # ------------------------------------------------------------------
    def _run_model_loop(self, repo: Path, issue: str, skill_body: str,
                        trace: Trace) -> None:
        system = self._system_prompt(skill_body)
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": f"仓库：{repo}\n任务：{issue}"}]
        done = False
        for _ in range(self.max_steps):
            reply = self.model_client.chat(messages)
            messages.append({"role": "assistant", "content": reply})

            final = _field(reply, "Final Answer")
            if final and not _field(reply, "Action"):
                trace.add_step(thought=reply, tool=None, args=None,
                               observation=f"模型声明完成：{final}", kind="final_answer")
                done = True
                break

            tool = _field(reply, "Action")
            raw = _field(reply, "Action Input")
            if not tool:
                messages.append({"role": "user",
                                 "content": "格式错误：请给出 Action / Action Input，或 Final Answer。"})
                trace.add_step(thought=reply, tool=None, args=None,
                               observation="格式错误，要求模型重试", kind="format_retry")
                continue
            try:
                args = json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                messages.append({"role": "user", "content": "Action Input 不是合法 JSON。"})
                trace.add_step(thought=reply, tool=tool, args={},
                               observation="Action Input 非法 JSON，要求重试", kind="format_retry")
                continue

            payload = call_tool(tool, args, root=repo)
            observation = format_observation(payload)
            trace.add_step(thought=reply, tool=tool, args=args,
                           observation=observation, ok=payload.get("ok"))
            messages.append({"role": "user", "content": f"Observation: {observation}"})

            if trace.get("tests_passed"):
                done = True
                break
        if not done and not trace.get("tests_passed"):
            trace.add_step(thought="达到步数上限", tool=None, args=None,
                           observation="loop 因步数上限退出", kind="max_steps")

    def _system_prompt(self, skill_body: str) -> str:
        from .tools import list_specs

        tools_desc = "\n".join(
            f"- {s['name']}: {s['description']}" for s in list_specs())
        prompt = (
            "你是一个本地代码仓库的 coding agent。用工具读代码、改代码、跑测试，"
            "直到 `run_tests` 全绿。不要修改测试文件。\n\n"
            f"可用工具：\n{tools_desc}\n\n"
            "每一步只输出一个动作，严格用如下格式：\n"
            "Thought: 你的推理\n"
            "Action: 工具名\n"
            "Action Input: {\"参数\": \"值\"}\n"
            "得到 Observation 后继续；任务完成时输出：\n"
            "Final Answer: 结论\n")
        if skill_body:
            prompt += f"\n【已加载 Skill】\n{skill_body}\n"
        return prompt

    # ------------------------------------------------------------------
    # 策略 B：离线确定性
    # ------------------------------------------------------------------
    def _run_offline_loop(self, repo: Path, issue: str, trace: Trace) -> None:
        # 1) 找候选目标文件（优先用子 agent 的结论里提到的文件）
        targets = self._candidate_files(repo, issue, trace)
        if not targets:
            trace.add_step(thought="未找到任何候选源文件", tool=None, args=None,
                           observation="离线策略放弃", kind="give_up")
            return

        for fname in targets:
            payload = call_tool("read_file", {"path": fname}, root=repo)
            trace.add_step(thought=f"读取候选文件 {fname}", tool="read_file",
                           args={"path": fname},
                           observation=format_observation(payload),
                           ok=payload.get("ok"))
            if not payload.get("ok"):
                continue
            original = payload["result"]["content"]

            # 2) 先确认它确实是坏的（避免在已经修好的文件上瞎改）
            check = call_tool("run_tests", {"path": "test_calculator.py"}, root=repo)
            if check.get("ok") and check["result"].get("passed"):
                trace.add_step(thought=f"{fname} 相关测试已通过，无需修改",
                               tool="run_tests", args={"path": "test_calculator.py"},
                               observation="pytest 已通过", ok=True)
                continue

            # 3) 对候选修复逐个真实验证
            for lineno, buggy_line in _suspicious_lines(original):
                for candidate in _candidate_fixes(buggy_line):
                    if candidate == buggy_line:
                        continue
                    fixed = _replace_line(original, lineno, candidate)
                    trace.add_step(
                        thought=(f"{fname}:{lineno} 的 `{buggy_line.strip()}` 疑似缺陷，"
                                 f"尝试改为 `{candidate.strip()}`"),
                        tool="write_file",
                        args={"path": fname, "content": fixed},
                        observation="已写入候选修复，准备跑测试验证",
                        kind="hypothesis", line=lineno)
                    call_tool("write_file", {"path": fname, "content": fixed}, root=repo)

                    result = call_tool("run_tests", {}, root=repo)
                    passed = bool(result.get("ok") and result["result"].get("passed"))
                    trace.add_step(
                        thought=f"验证候选修复（{fname}:{lineno}）",
                        tool="run_tests", args={},
                        observation=format_observation(result), ok=result.get("ok"),
                        verified=passed, kind="verify")
                    if passed:
                        trace["fix"] = {"file": fname, "line": lineno,
                                        "from": buggy_line.strip(),
                                        "to": candidate.strip()}
                        return
                    # 回滚，试下一个候选
                    call_tool("write_file", {"path": fname, "content": original}, root=repo)

        trace.add_step(thought="所有候选修复都未让测试通过", tool=None, args=None,
                       observation="离线策略未能修复；语义型缺陷需要真实模型",
                       kind="give_up")

    def _candidate_files(self, repo: Path, issue: str, trace: Trace) -> list[str]:
        """候选源文件：优先子 agent 定位到的，其次 issue 里反引号提到的标识符。"""
        names: list[str] = []
        for summary in trace.get("subagent_summaries", []):
            for call in summary.get("tool_calls", []):
                if call.get("tool") == "read_file":
                    p = call.get("args", {}).get("path")
                    if p and p not in names:
                        names.append(p)
        for ident in re.findall(r"`([^`]+)`", issue):
            base = ident.split(".")[0]
            for cand in (f"{base}.py", ident):
                if cand not in names and (repo / cand).is_file():
                    names.append(cand)
        return names

    # ------------------------------------------------------------------
    def _finalize(self, repo: Path, trace: Trace) -> None:
        test_payload = call_tool("run_tests", {}, root=repo)
        passed = bool(test_payload.get("ok") and test_payload["result"].get("passed"))
        trace["tests_passed"] = passed
        trace.add_step(thought="收尾：跑一次完整测试确认最终状态", tool="run_tests", args={},
                       observation=format_observation(test_payload),
                       ok=test_payload.get("ok"), kind="final_test")

        diff_payload = call_tool("git_diff", {}, root=repo)
        patch = ""
        if diff_payload.get("ok"):
            patch = diff_payload["result"].get("diff", "")
        if not patch.strip():
            # git diff 为空（例如所有改动都被提交过）时，用快照补一个 unified diff
            patch = _snapshot_diff(repo)
        trace["patch"] = patch
        trace["done"] = passed
        trace.add_step(thought="生成最终 patch", tool="git_diff", args={},
                       observation=(patch or "[无改动]"), ok=diff_payload.get("ok"),
                       kind="patch")


# ----------------------------------------------------------------------
# 离线策略的辅助函数
# ----------------------------------------------------------------------
def _suspicious_lines(content: str) -> list[tuple[int, str]]:
    """返回像"实现体里带算术运算"的行号与内容。"""
    out = []
    for i, line in enumerate(content.splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#") or s.startswith(('"""', "'''")):
            continue
        body = s.split("#", 1)[0]
        if ("return" in body or "=" in body) and ARITH_RE.search(body):
            out.append((i, line))
    return out


def _candidate_fixes(line: str) -> list[str]:
    """对一行里的第一个算术运算符，依次替换成其它运算符。"""
    m = ARITH_RE.search(line)
    if not m:
        return []
    current = m.group(1)
    out = [line]
    for op in CANDIDATE_OPERATORS:
        if op == current:
            continue
        out.append(line[:m.start(1)] + op + line[m.end(1):])
    return out


def _replace_line(content: str, lineno: int, new_line: str) -> str:
    lines = content.splitlines(keepends=True)
    idx = lineno - 1
    if idx < 0 or idx >= len(lines):
        return content
    ending = "\n" if lines[idx].endswith("\n") else ""
    lines[idx] = new_line.rstrip("\n") + ending
    return "".join(lines)


def _snapshot_diff(repo: Path) -> str:
    """git diff 为空时的兜底：把工作区与 `<file>.orig` 快照对比成 unified diff。"""
    chunks = []
    for orig in sorted(repo.glob("*.orig")):
        current = orig.with_suffix("")
        if not current.is_file():
            continue
        chunks.extend(difflib.unified_diff(
            orig.read_text(encoding="utf-8").splitlines(keepends=True),
            current.read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile=f"a/{current.name}", tofile=f"b/{current.name}"))
    return "".join(chunks)


def _field(reply: str, label: str) -> str:
    m = re.search(rf"{re.escape(label)}:\s*(.+)", reply or "")
    return m.group(1).strip() if m else ""
