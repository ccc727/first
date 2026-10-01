"""维基百科 API 工具（支持中英文条目）。

联网时优先走维基百科 API；离线 / 被墙 / 条目缺失时回退到本地知识夹具
``data/wiki_fixtures.json``（见 ``load_fixtures``），保证无网环境下的教学自检仍可跑通。
回退结果以 ``__OFFLINE_FIXTURE__<json>`` 开头，Agent 的离线策略据此判断数据来源。
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "wiki",
        "description": "查询维基百科条目，返回摘要文本（支持中文/英文；离线时回退到本地知识夹具）。",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "要查询的条目名"}},
            "required": ["query"],
        },
    },
}

_LANGS = {"en": "en.wikipedia.org", "zh": "zh.wikipedia.org"}

# 离线夹具标记：前缀 + 一段 JSON（便于上层区分「真网络摘要」与「本地回退」）。
FIXTURE_MARKER = "__OFFLINE_FIXTURE__"
FIXTURE_PATH = Path(__file__).resolve().parents[2] / "data" / "wiki_fixtures.json"
# 单次请求超时：默认偏短，避免无外网环境下自检长时间挂起（可用 WIKI_TIMEOUT 覆盖）。
NETWORK_TIMEOUT = float(os.environ.get("WIKI_TIMEOUT", "4"))
# 连接级快速探测超时：DNS/连接阶段就失败时能很快判死，不必等满 NETWORK_TIMEOUT。
PROBE_TIMEOUT = float(os.environ.get("WIKI_PROBE_TIMEOUT", "2"))
# 一旦确认网络不可达 / 全部条目拿不到，就置位不再重复等超时（进程级缓存）。
_NETWORK_DOWN = False

_FIXTURES: dict | None = None


def load_fixtures() -> list[dict]:
    """读取本地知识夹具（进程内缓存）；文件缺失或损坏时返回空列表。"""
    global _FIXTURES
    if _FIXTURES is None:
        try:
            data = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
            entries = data.get("entries", []) if isinstance(data, dict) else []
            _FIXTURES = [e for e in entries if isinstance(e, dict) and e.get("text")]
        except Exception:      # noqa: BLE001 —— 夹具缺失不该让 wiki 工具直接崩
            _FIXTURES = []
    return _FIXTURES


def _norm(text: str) -> str:
    """归一化条目名：小写、去空白与常见标点/括号后缀。"""
    s = str(text).lower().replace(" ", "").replace("_", "")
    s = s.replace("（", "(").replace("）", ")")
    if "(" in s:
        s = s.split("(", 1)[0]        # 「Transformer (机器学习模型)」-> 「transformer」
    return s


def lookup_fixture(query: str) -> dict | None:
    """在本地夹具里按条目名 / 别名查找；命中返回条目 dict，否则 None。"""
    q = _norm(query)
    if not q:
        return None
    entries = load_fixtures()
    for e in entries:                       # 先精确匹配，再退化为包含匹配
        if _norm(e.get("query", "")) == q:
            return e
        if any(_norm(a) == q for a in e.get("aliases", []) or []):
            return e
    for e in entries:
        names = [e.get("query", "")] + list(e.get("aliases", []) or [])
        if any(q and (q in _norm(n) or _norm(n) in q) for n in names):
            return e
    return None


def _fixture_run(query: str) -> str:
    """离线回退：返回带标记的夹具 JSON（含 query / text），找不到条目则抛错。"""
    entry = lookup_fixture(query)
    if entry is None:
        raise RuntimeError(f"维基百科没有找到 {query!r} 的相关条目（离线夹具也未命中）")
    payload = {"query": entry.get("query", query), "text": entry["text"]}
    return FIXTURE_MARKER + json.dumps(payload, ensure_ascii=False)


def _summary(query: str, lang: str) -> str:
    api = (f"https://{_LANGS[lang]}/w/api.php?action=query&prop=extracts&exintro=1"
           f"&explaintext=1&format=json&titles={urllib.parse.quote(query)}")
    req = urllib.request.Request(api, headers={"User-Agent": "llm-beginner-agent/1.0"})
    with urllib.request.urlopen(req, timeout=NETWORK_TIMEOUT) as r:
        data = json.loads(r.read().decode("utf-8"))
    pages = data.get("query", {}).get("pages", {})
    for p in pages.values():
        if "extract" in p:
            return p["extract"]
    return ""


def network_reachable() -> bool:
    """轻量连通性探测（短超时，只建连不取数据）；失败后进程内不再重试。"""
    global _NETWORK_DOWN
    if _NETWORK_DOWN:
        return False
    try:
        req = urllib.request.Request(
            "https://en.wikipedia.org/w/api.php?action=query&format=json&meta=siteinfo",
            headers={"User-Agent": "llm-beginner-agent/1.0"})
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT):
            return True
    except Exception:       # noqa: BLE001 —— 离线 / 被墙 / DNS 失败都归为不可达
        _NETWORK_DOWN = True
        return False


def run(args: dict) -> str:
    query = str(args.get("query", "")).strip()
    if not query:
        raise ValueError("query 不能为空")
    network_error: Exception | None = None
    if network_reachable():
        for lang in ("zh", "en"):
            try:
                text = _summary(query, lang)
            except Exception as e:      # noqa: BLE001 —— 网络不可用时走本地夹具
                network_error = e
                continue
            if text:
                return text[:2000]
    # 网络拿不到内容：回退本地知识夹具（离线教学环境的合法数据源）。
    try:
        return _fixture_run(query)
    except RuntimeError:
        if network_error is not None:
            raise RuntimeError(
                f"维基百科不可用（{type(network_error).__name__}: {network_error}），"
                f"本地夹具也没有 {query!r} 的相关条目") from network_error
        raise RuntimeError(f"维基百科没有找到 {query!r} 的相关条目")
