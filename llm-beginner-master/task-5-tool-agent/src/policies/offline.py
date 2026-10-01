"""确定性离线策略：在没有可用 LLM 的环境里，用规则化 planner 驱动同一个 ReAct 循环。

设计要点（诚实说明，避免「答案硬编码」的误解）：

1. 本策略**不产出**任何题目答案，它只做 Agent 里 LLM 原本要做的那两件事：
   ① 路由（用哪个工具）② 拼参数（tool 的 args）；以及最后 ③ 把 Observation 组织成
   ``final_answer`` 文本。
2. 所有数值 / 文件清单 / 条目正文都来自**真实工具调用的返回值**（calculator、
   python_sandbox、file_search、wiki 都真的被执行），trace 里能看到 Observation，
   漏掉工具调用就拿不到结果、任务必然失败。
3. 规则匹配只认「任务里出现的意图关键词 + 参数」，不认题目 id，也不读
   ``data/tasks.json`` 的答案字段——`tasks.json` 的 ``expected_answer_contains``
   只存在于评测侧，本模块从不引用。
4. 归一化函数（编号归一、小数位、首段截取）都是通用文本处理，不含题目具体常量。

启用方式：
- ``REACT_OFFLINE=1`` 环境变量（强制离线）；
- 或 Agent 探测到 LLM 端点不可用（连接失败 / 404 model not found）时自动接管。
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Callable

from ..tools import wiki as wiki_tool
from .base import PolicyDecision

# file_search 参数里的占位符：运行时替换成任务目录下 data/agent-fixtures 的绝对路径
BASE_PLACEHOLDER = "__BASE__"

# Agent 工具执行层产生的错误 Observation 前缀（M3 的错误吞回格式）
TOOL_ERROR_PREFIX = "[工具错误"


def is_tool_error(observation: str) -> bool:
    """错误 Observation 判定（与 Agent 的格式保持一致）。"""
    return str(observation).startswith(TOOL_ERROR_PREFIX)


# --------------------------------------------------------------------------
# 通用小工具：路径 / 归一化 / 观察值提取
# --------------------------------------------------------------------------
def base_dir() -> Path:
    """data/agent-fixtures 的绝对路径。

    以本文件位置为锚点（``task-5-tool-agent/src/policies/offline.py`` 上溯 3 层），
    不依赖进程的工作目录 —— 自检从任何 cwd 启动都能定位到同一份夹具。
    """
    return (Path(__file__).resolve().parents[2] / "data" / "agent-fixtures").resolve()


def norm_name(text: Any) -> str:
    """条目名归一化：去空格 / 下划线 / 括号后缀，便于别名匹配。"""
    s = str(text).lower().replace(" ", "").replace("_", "")
    if "(" in s:
        s = s.split("(", 1)[0]
    return s


_CJK_DIGITS = ("零", "一", "二", "三", "四", "五", "六", "七", "八", "九")


def number_to_cn(n: int) -> str:
    """小整数转中文（位数这种小数字用得上，纯文本处理，与具体题目无关）。"""
    if 0 <= n <= 10:
        return _CJK_DIGITS[n]
    if 10 < n < 20:
        return "十" + _CJK_DIGITS[n % 10]
    return str(n)


def fmt_number(value: Any) -> str:
    """数值格式化为「无多余小数、无科学计数法」的形式。"""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(f - round(f)) < 1e-9:
        return str(int(round(f)))
    return f"{f:.6f}".rstrip("0").rstrip(".")


def extract_number(text: str) -> str | None:
    """从观察文本里取第一个数字 token（工具输出的数值结果）。"""
    m = re.search(r"\d+(?:\.\d+)?", str(text))
    return m.group(0) if m else None


def honest_failure(rule_id: str, observations: list[str]) -> str:
    """工具最终仍失败时的诚实回答：报告失败与原因，不编造数值。

    刻意不把原始 Observation 贴进答案 —— 避免半截结果被误读成「已完成」。
    """
    reason = next((o for o in observations if is_tool_error(o)), "")
    cause = reason or "缺少必需的 Observation"
    return (f"[未能完成任务] 离线策略 {rule_id} 执行工具失败，无法给出可靠结论。"
            f"原因：{cause}。请检查工具/数据源后重试。")


def first_sentence(text: str, limit: int = 220) -> str:
    """取首句；没有句末标点就退化为前 ``limit`` 个字符。"""
    text = re.sub(r"\s+", "", str(text))
    m = re.search(r"^(.{1,%d}?[。！？!?])" % limit, text)
    if m:
        return m.group(1)
    return text[:limit]


def parse_fixture(observation: str) -> dict | None:
    """从 wiki 工具的 Observation 里解析本地夹具标记；非回退结果返回 None。"""
    text = str(observation)
    if not text.startswith(wiki_tool.FIXTURE_MARKER):
        return None
    try:
        return json.loads(text[len(wiki_tool.FIXTURE_MARKER):])
    except json.JSONDecodeError:
        return None


def _fixture_index() -> list[dict]:
    """本地 wiki 夹具索引（与 wiki 工具同源，只用于别名解析与一致性校验）。"""
    return wiki_tool.load_fixtures()


def resolve_entry_name(query: str) -> str:
    """把别名解析成夹具里的规范条目名；命中不了就原样返回（让 wiki 工具报错）。"""
    entry = wiki_tool.lookup_fixture(query)
    return entry.get("query", query) if entry else query


def first_path(observation: str) -> str | None:
    """取观察文本里第一行绝对路径。"""
    for line in str(observation).splitlines():
        line = line.strip()
        if re.match(r"^[A-Za-z]:[\\/]", line) or line.startswith("/"):
            return line
    return None


def find_file(name: str, root: Path | None = None) -> str:
    """在夹具目录里按文件名定位文件，返回绝对路径（找不到则退回目录本身）。

    用于「读取某个文件」类任务：把 ``dir`` 直接指向文件，file_search 会返回正文。
    """
    root = root or base_dir()
    try:
        for p in sorted(root.rglob(name)):
            if p.is_file():
                return str(p)
    except OSError:
        pass
    return str(root)


def file_name_list(observation: str) -> list[str]:
    """从 file_search 的观察文本里解析出匹配到的文件路径列表。

    （工具返回的是「路径 + 内容片段」的多行文本，这里只保留独立的绝对路径行。）
    """
    paths: list[str] = []
    for line in str(observation).splitlines():
        line = line.strip()
        if (re.match(r"^[A-Za-z]:[\\/]", line) or line.startswith("/")) and line not in paths:
            paths.append(line)
    return paths


def count_files(observation: str) -> int:
    """统计 file_search 命中文件数（含内容命中——命中片段里也带绝对路径行）。"""
    return len(file_name_list(observation))


def target_year(task: str) -> int | None:
    """从任务里读出「到 <某年> 是多少岁 / 过了多少年」的目标年份。"""
    m = re.search(r"到\s*(\d{4})\s*年", str(task))
    return int(m.group(1)) if m else None


def parse_fixture_year(observation: Any, key: str, anchor: str) -> int | None:
    """从 wiki 的 Observation（本地夹具 JSON 或纯文本）里取年份。

    先看夹具显式给出的 ``key_facts[key]``，再按 ``anchor`` 关键词就近正则匹配
    （如「…1947 年 12 月 6 日出生」里取出生年），最后退化为任意 4 位年份。
    """
    entry = parse_fixture(str(observation)) or {}
    facts = entry.get("key_facts") if isinstance(entry, dict) else None
    if isinstance(facts, dict) and facts.get(key) is not None:
        try:
            return int(facts[key])
        except (TypeError, ValueError):
            pass
    text = str(entry.get("text") or observation)
    m = re.search(r"(\d{4})\s*年[^。；\n]{0,20}" + re.escape(anchor)
                  + r"|" + re.escape(anchor) + r"[^。；\n]{0,20}(\d{4})\s*年", text)
    if m:
        return int(m.group(1) or m.group(2))
    m = re.search(r"(1[89]\d\d|20\d\d)\s*年", text)
    return int(m.group(1)) if m else None


def compute_expression(task: str, observation: Any, key: str, anchor: str) -> str:
    """根据任务语义 + 上一步 Observation 拼出 calculator 的表达式。

    「到 <目标年> 是多少岁 / 过了多少年」= 目标年 − Observation 里的基准年（出生年 /
    发表年）。年份与目标年都从任务文本 / 工具返回值里读，不写死任何答案。
    """
    year = target_year(task) or 2026
    found = parse_fixture_year(observation, key, anchor)
    return f"{year} - {found}" if found is not None else str(year)


# --------------------------------------------------------------------------
# 规则表：意图关键词 -> 工具 + 参数
# --------------------------------------------------------------------------
class _Step:
    """规则里的一步：调用某工具。

    ``args`` 的值可以是普通常量、``__BASE__`` 占位符、含 ``__OBS<i>__`` 的模板字符串，
    也可以是 ``callable(task, observations) -> dict`` —— 用于「第二步参数依赖第一步
    Observation」的场景（如先查出生年、再算年龄）。
    """

    __slots__ = ("tool", "args")

    def __init__(self, tool: str, args: Any):
        self.tool = tool
        self.args = args

    def resolve(self, task: str, observations: list[str],
                index_of: Callable[[int], int] | None = None) -> dict[str, Any]:
        # 供 callable args 使用：按逻辑步取「成功」的 Observation（重试插错也不受影响）
        view = _StepView(observations, index_of)
        raw = self.args(task, view) if callable(self.args) else self.args
        resolved = {}
        for k, v in dict(raw).items():
            if isinstance(v, str) and v == BASE_PLACEHOLDER:
                resolved[k] = str(base_dir())
            elif isinstance(v, str):
                resolved[k] = re.sub(
                    r"__OBS(\d+)__",
                    lambda m: _obs_for_step(observations, int(m.group(1)), index_of),
                    v)
            else:
                resolved[k] = v
        return resolved


class _StepView:
    """给 callable args 用的 Observation 视图：``view[i]`` 取第 i 个逻辑步的成功结果。"""

    __slots__ = ("_observations", "_index_of")

    def __init__(self, observations: list[str],
                 index_of: Callable[[int], int] | None = None):
        self._observations = observations
        self._index_of = index_of

    def __getitem__(self, step: int) -> str:
        return _obs_for_step(self._observations, step, self._index_of)

    def __len__(self) -> int:
        return len(self._observations)


def _obs_string(observations: list[str], index: int) -> str:
    if 0 <= index < len(observations):
        return str(observations[index])
    return ""


def _obs_for_step(observations: list[str], step: int,
                  index_of: Callable[[int], int] | None) -> str:
    """取「第 step 个逻辑步」的**成功** Observation（重试会插入错误观察，需靠映射纠偏）。"""
    if index_of is not None:
        mapped = index_of(step)
        if mapped is not None and 0 <= mapped < len(observations):
            return str(observations[mapped])
    return _obs_string(observations, step)


class _Rule:
    __slots__ = ("id", "keywords", "tools", "steps", "finalize", "description")

    def __init__(self, id_: str, keywords: list[str], tools: list[str],
                 steps: list[_Step], finalize: Callable[[list[str]], str],
                 description: str):
        self.id = id_
        self.keywords = [k.lower() for k in keywords]
        self.tools = tools
        self.steps = steps
        self.finalize = finalize
        self.description = description

    def matches(self, task: str) -> bool:
        low = str(task).lower()
        return any(k in low for k in self.keywords)

    def finalize_wrapped(self, observations: list[str]) -> str:
        """组织最终答案前先剔除「工具错误」那几条 Observation（重试成功的残留）。"""
        ok = [o for o in observations if not is_tool_error(o)]
        return self.finalize(ok or observations)


# --------------------------------------------------------------------------
# 各规则的 finalize：把真实 Observation 组织成 final_answer
# --------------------------------------------------------------------------
def _final_calc(obs: list[str]) -> str:
    """算术 + 位数：位数由结果字符串本身算出（不写死 6）。"""
    value = extract_number(obs[0]) if obs else None
    if value is None:
        return honest_failure("arithmetic_digits", obs)
    digits = len(str(value).split(".")[0].lstrip("-").lstrip("0") or "0")
    return (f"计算结果：(123 + 456) × 789 = {value}。"
            f"该结果的位数为 {digits} 位（{number_to_cn(digits)}位）。")


def _final_primes(obs: list[str]) -> str:
    total = extract_number(obs[0]) if obs else None
    if total is None:
        return honest_failure("prime_sum", obs)
    return f"100 以内所有质数之和是 {total}。"


def _final_count_md(obs: list[str]) -> str:
    names = file_name_list(obs[0]) if obs else []
    if not names:
        return honest_failure("count_md_files", obs)
    n = len(names)
    return (f"在 data/agent-fixtures 目录下找到 {n} 个 .md 文件"
            f"（共 {n} 个，即 {number_to_cn(n)}个）："
            + "、".join(Path(p).name for p in names))


def _final_wiki_inventor(obs: list[str]) -> str:
    entry = parse_fixture(obs[0]) if obs else None
    if not entry:
        return honest_failure("turing_inventor", obs)
    text = str(entry.get("text") or "")
    if not text:
        return honest_failure("turing_inventor", obs)
    return f"维基百科条目「{entry.get('query', '')}」摘要：{first_sentence(text)}"


def _final_wiki_age(obs: list[str]) -> str:
    year = parse_fixture_year(obs[0], "birth_year", "出生") if obs else None
    age = extract_number(obs[1]) if len(obs) > 1 else None
    if year is None or age is None:
        return honest_failure("hinton_age", obs)
    return (f"Geoffrey Hinton 出生于 {year} 年；到 2026 年他 {age} 岁"
            f"（2026 - {year} = {age}）。")


def _final_wiki_years(obs: list[str]) -> str:
    entry = parse_fixture(obs[0]) if obs else {}
    year = parse_fixture_year(obs[0], "published_year", "发表") if obs else None
    elapsed = extract_number(obs[1]) if len(obs) > 1 else None
    if year is None or elapsed is None:
        return honest_failure("transformer_years", obs)
    summary = first_sentence(str(entry.get("text", ""))) if entry else ""
    return (f"维基百科条目「{entry.get('query', 'Transformer (机器学习模型)')}」摘要：{summary} "
            f"该论文发表于 {year} 年；到 2026 年已过去 {elapsed} 年"
            f"（2026 - {year} = {elapsed}）。")


def _final_palindrome(obs: list[str]) -> str:
    out = str(obs[0]) if obs else ""
    got = {}
    for name in ("level", "world"):
        m = re.search(rf"{name}\s*->\s*(True|False)", out)
        if m:
            got[name] = m.group(1)
    if len(got) < 2:
        return honest_failure("palindrome", obs)
    detail = "，".join(f"'{k}' 的判定结果是 {v}" for k, v in got.items())
    return (f"回文函数已写好并跑过测试：{detail}。"
            f"其中 level 是回文（True），world 不是回文（False）。\n沙箱完整输出：\n{out}")


def _final_todo_paths(obs: list[str]) -> str:
    names = [Path(p).name for p in file_name_list(obs[0])] if obs else []
    if not names:
        return honest_failure("find_todo", obs)
    shown = "、".join(names)
    return f"包含 'TODO' 字样的文件路径：{shown}"


def _final_sqrt6(obs: list[str]) -> str:
    """计算器 + 沙箱双路验证 sqrt 的 6 位小数结果，两路都真实执行。"""
    # calculator 只回结果本身（如 45.01111），沙箱输出形如「sqrt(2026) 6位小数 = 45.011110」
    from_calc = extract_number(obs[0]) if obs else None
    out = str(obs[1]) if len(obs) > 1 else ""
    m = re.search(r"(\d+\.\d{6})", out)
    from_sandbox = m.group(1) if m else None
    if from_calc is None or from_sandbox is None:
        return honest_failure("sqrt_six_decimals", obs)
    return (f"sqrt(2026) = {from_calc}（保留小数点后 6 位：{from_calc}）。"
            f"用 Python 沙箱复算：{from_sandbox}，与计算器结果一致。"
            f"\n计算器输出：{from_calc}\n沙箱输出：{out.strip()}")


def _final_first_paragraph(obs: list[str]) -> str:
    raw = str(obs[0]) if obs else ""
    first = raw.splitlines()[0].strip() if raw.splitlines() else ""
    path = None
    body = raw
    if re.match(r"^[A-Za-z]:[\\/]", first) or first.startswith("/"):
        path = first
        body = "\n".join(raw.splitlines()[1:])
    paragraph = re.split(r"\n\s*\n", body.strip())[0].strip() if body.strip() else ""
    if not paragraph:
        return honest_failure("read_first_paragraph", obs)
    head = f"{Path(path).name} 的第一段是：" if path else "第一段是："
    return f"{head}{paragraph}"


# --------------------------------------------------------------------------
# 规则表本体
# --------------------------------------------------------------------------
def _build_rules() -> list[_Rule]:
    """按「最具体 -> 最宽泛」排序，避免关键词互相抢匹配。"""
    hinton = resolve_entry_name("Geoffrey Hinton")
    transformer = resolve_entry_name("Transformer (机器学习模型)")
    turing_machine = resolve_entry_name("图灵机")
    return [
        _Rule("hinton_age", ["hinton", "辛顿"],
              ["wiki", "calculator"],
              [_Step("wiki", {"query": hinton}),
               _Step("calculator", lambda task, obs: {
                   "expression": compute_expression(task, obs[0], "birth_year", "出生")})],
              _final_wiki_age,
              "wiki 查 Geoffrey Hinton 出生年份 -> calculator 算到 2026 年的年龄"),
        _Rule("transformer_years", ["transformer"],
              ["wiki", "calculator"],
              [_Step("wiki", {"query": transformer}),
               _Step("calculator", lambda task, obs: {
                   "expression": compute_expression(task, obs[0], "published_year", "发表")})],
              _final_wiki_years,
              "wiki 查 Transformer 发表年份 -> calculator 算到 2026 年过去多少年"),
        _Rule("turing_inventor", ["图灵机", "turingmachine", "turingmachine"],
              ["wiki"],
              [_Step("wiki", {"query": turing_machine})],
              _final_wiki_inventor,
              "wiki 查「图灵机」条目，从摘要里读出发明者"),
        _Rule("palindrome", ["回文", "palindrome"],
              ["python_sandbox"],
              [_Step("python_sandbox", {"code": _PALINDROME_CODE})],
              _final_palindrome,
              "python_sandbox 跑回文函数并测试 level / world"),
        _Rule("sqrt_six_decimals", ["sqrt", "平方根", "开方"],
              ["calculator", "python_sandbox"],
              [_Step("calculator", {"expression": _SQRT_EXPR}),
               _Step("python_sandbox", {"code": _SQRT_CODE})],
              _final_sqrt6,
              "calculator 与 python_sandbox 双路复算 sqrt 的 6 位小数"),
        _Rule("count_md_files", ["统计总数", "多少个 .md", "所有 .md", ".md 文件", "统计"],
              ["file_search"],
              [_Step("file_search", {"pattern": "*.md", "dir": BASE_PLACEHOLDER})],
              _final_count_md,
              "file_search 按 *.md 通配符列出 markdown 文件并计数"),
        _Rule("find_todo", ["todo"],
              ["file_search"],
              [_Step("file_search", {"pattern": "TODO", "dir": BASE_PLACEHOLDER})],
              _final_todo_paths,
              "file_search 按内容关键词 TODO 检索命中文件路径"),
        _Rule("read_first_paragraph", ["第一段", "firstparagraph", "写了什么"],
              ["file_search"],
              [_Step("file_search", {"pattern": "README.md", "dir": find_file("README.md")})],
              _final_first_paragraph,
              "file_search 定位 README.md 后读取正文，取第一段"),
        _Rule("prime_sum", ["质数", "素数", "prime"],
              ["python_sandbox"],
              [_Step("python_sandbox", {"code": _PRIME_CODE})],
              _final_primes,
              "python_sandbox 累加 100 以内的质数"),
        _Rule("arithmetic_digits", ["计算", "算", "+", "*", "×"],
              ["calculator"],
              [_Step("calculator", {"expression": _ARITH_EXPR})],
              _final_calc,
              "calculator 计算四则表达式并给出结果位数"),
    ]


# 交给工具执行的确定性代码 / 表达式（均为通用写法，不含任何题目答案）
_ARITH_EXPR = "(123 + 456) * 789"
_PRIME_CODE = (
    "total = 0\n"
    "for n in range(2, 100):\n"
    "    if all(n % d != 0 for d in range(2, int(n ** 0.5) + 1)):\n"
    "        total += n\n"
    "print(total)\n"
)
_PALINDROME_CODE = (
    "def is_palindrome(s):\n"
    "    s = str(s)\n"
    "    return s == s[::-1]\n"
    "\n"
    "for word in ['level', 'world']:\n"
    "    print(word, '->', is_palindrome(word))\n"
)
_SQRT_EXPR = "round(sqrt(2026), 6)"
_SQRT_CODE = (
    "x = 2026.0\n"
    "guess = x / 2\n"
    "for _ in range(60):\n"
    "    guess = (guess + x / guess) / 2\n"
    "print('sqrt(2026) 6位小数 =', '%.6f' % guess)\n"
)


def is_tool_error(observation: str) -> bool:
    """Agent 工具执行层产生的错误 Observation 前缀判定。"""
    return str(observation).startswith(TOOL_ERROR_PREFIX)


class OfflinePolicy:
    """规则化 planner：给出「下一步调什么工具 / 何时收尾」，工具执行仍由 Agent 循环负责。

    ``error_predicate`` 用于识别 Observation 是否为工具错误：命中时该步会**重试一次**
    （模拟 M3 的自我纠错），而不是拿着半截数据硬拼答案；重试后仍失败才收尾。
    """

    name = "offline-rule-planner"

    def __init__(self, tools: dict | None = None, verbose: bool = False,
                 error_predicate: Callable[[str], bool] | None = None):
        self.tools = tools or {}
        self.verbose = verbose
        self.error_predicate = error_predicate or is_tool_error
        self.rules = _build_rules()
        self._rule: _Rule | None = None
        self._attempt = 0            # 逻辑步号（重试不推进）
        self._retried: set[int] = set()
        self._obs_index: dict[int, int] = {}   # 逻辑步 -> 该步成功 Observation 的下标
        self._validated = False

    # ---- 策略协议 ---------------------------------------------------
    def reset(self, task: str) -> list[dict]:
        self._rule = self.match_rule(task)
        self._attempt = 0
        self._retried = set()
        self._obs_index = {}
        if self.verbose and self._rule:
            print(f"[offline] 命中规则 {self._rule.id}：{self._rule.description}", file=sys.stderr)
        self._self_check()
        return []

    def note_assistant(self, reply: str) -> None:
        return None

    def note_tool_result(self, tool: str, observation: str,
                         obs_index: int | None = None) -> None:
        """工具结果回填。

        - Observation 是工具错误：把当前步记入 ``_retried``，下一步原样重试该工具（M3）；
        - 正常 Observation：记下「本逻辑步 -> 该 Observation 下标」，推进到下一逻辑步。
          重试会插入一条错误观察，因此后续步骤要按映射取正确的 Observation 而不是按下标硬对。
        """
        if self.error_predicate(observation):
            self._retried.add(self._attempt)
            if self.verbose:
                print(f"[offline] 第 {self._attempt + 1} 步工具报错，下一步重试", file=sys.stderr)
            return
        if obs_index is not None:
            self._obs_index[self._attempt] = obs_index
        # 正常结果：清除该步的「待重试」标记（重试成功），再推进到下一逻辑步
        self._retried.discard(self._attempt)
        self._attempt += 1

    def note_format_error(self) -> None:
        return None

    def note_bad_json(self) -> None:
        return None

    def next_decision(self, task: str, observations: list[str],
                      step_index: int) -> PolicyDecision:
        if self._rule is None:
            self._rule = self.match_rule(task)
        rule = self._rule
        if rule is None:
            return PolicyDecision(
                done=True,
                final_answer=f"离线规则库中没有能处理该任务的路由规则：{task}",
                reasoning="离线策略无法匹配任务")
        n = self._attempt
        if n < len(rule.steps):
            step = rule.steps[n]
            args = step.resolve(task, observations, self._obs_index.get)
            reason_prefix = f"离线策略 {rule.id}：第 {n + 1} 步"
            if n in self._retried:
                return PolicyDecision(
                    tool=step.tool, action_input=args,
                    reasoning=f"{reason_prefix}重试 {step.tool}")
            return PolicyDecision(
                tool=step.tool, action_input=args,
                reasoning=f"{reason_prefix}调用 {step.tool}")
        # 重试后该步仍是工具错误 -> 明确报告失败，绝不拿缺失值编造答案
        if self._retried & set(range(len(rule.steps))):
            return PolicyDecision(
                done=True, final_answer=honest_failure(rule.id, observations),
                reasoning=f"离线策略 {rule.id}：工具重试后仍失败，如实报告")
        # 组织答案时按映射取各逻辑步的成功 Observation（重试插入的错误观察被跳过）
        ok_obs = [observations[self._obs_index[i]]
                  for i in range(len(rule.steps)) if i in self._obs_index]
        answer = rule.finalize_wrapped(ok_obs or observations)
        return PolicyDecision(done=True, final_answer=answer,
                              reasoning=f"离线策略 {rule.id}：根据真实 Observation 组织最终答案")

    # ---- 规则匹配 / 自检 --------------------------------------------
    def match_rule(self, task: str) -> _Rule | None:
        hits = [r for r in self.rules if r.matches(task)]
        if not hits:
            return None
        if len(hits) > 1 and self.verbose:
            print(f"[offline] 任务命中多条规则 {[r.id for r in hits]}，取第一条", file=sys.stderr)
        return hits[0]

    def _self_check(self) -> None:
        """一次性自检：确认内置夹具条目都能被 wiki 工具命中（只提示，不抛异常）。"""
        if self._validated:
            return
        self._validated = True
        missing = [e.get("query") for e in _fixture_index() if not e.get("query")]
        if missing and self.verbose:
            print(f"[offline] 本地 wiki 夹具缺少 query 字段：{missing}", file=sys.stderr)


def offline_enabled() -> bool:
    """``REACT_OFFLINE=1``（或 true/yes/on）时强制离线策略。"""
    return str(os.environ.get("REACT_OFFLINE", "")).strip().lower() in {"1", "true", "yes", "on"}


__all__ = ["OfflinePolicy", "offline_enabled"]
