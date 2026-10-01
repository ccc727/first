"""Skill 加载器（约 50 行的核心逻辑）：扫描 + 元数据解析 + 描述匹配 + 渐进式披露。

设计对齐 [Anthropic Skills](https://github.com/anthropics/skills) 约定：

1. **扫描**：读 `skills_dir/*/SKILL.md`，解析 YAML front-matter 的 `name` / `description`；
2. **按需加载（progressive disclosure）**：`list_skills()` 只回元数据（轻量，可全量塞进
   system prompt 给模型做路由）；正文**只有命中时才读**（`load()`），不会一上来就把
   所有 SKILL.md 正文都灌进上下文；
3. **description-based 路由**：`match(task)` 用「任务词 × skill 描述/关键词」打分，
   返回命中的 skill 名与正文，供 agent 注入到本次任务上下文。

front-matter 用 `pyyaml` 解析；缺 `name` / `description` 时回退到目录名 + 正文首行，
保证 `list_skills()` 的每一项至少含 name + description（自检 `skill_loader_metadata` 契约）。
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml


class SkillLoader:
    """扫描 skills 目录，提供元数据枚举与按 description 的按需加载。"""

    def __init__(self, skills_dir: str):
        self.skills_dir = Path(skills_dir)
        # 只缓存元数据；正文不缓存 —— 避免"全量读入 context"
        self._meta: dict[str, dict] = {}
        self._scan()

    # ------------------------------------------------------------------
    # 扫描与解析
    # ------------------------------------------------------------------
    def _scan(self) -> None:
        if not self.skills_dir.is_dir():
            return
        for path in sorted(self.skills_dir.glob("*/SKILL.md")):
            front, body = _split_front_matter(path.read_text(encoding="utf-8"))
            name = str(front.get("name") or path.parent.name)
            description = str(
                front.get("description") or _first_meaningful_line(body) or name)
            keywords = front.get("keywords") or []
            if isinstance(keywords, str):
                keywords = [k.strip() for k in re.split(r"[,，]", keywords) if k.strip()]
            self._meta[name] = {
                "name": name,
                "description": description,
                "keywords": [str(k) for k in keywords],
                "path": str(path),
                "dir": path.parent.name,
            }

    # ------------------------------------------------------------------
    # 对外接口
    # ------------------------------------------------------------------
    def list_skills(self) -> list[dict]:
        """返回每个 skill 的元数据（每项含 name + description）。"""
        return [
            {"name": m["name"], "description": m["description"],
             "keywords": m["keywords"], "path": m["path"]}
            for m in self._meta.values()
        ]

    def load(self, name: str) -> str:
        """命中后才读正文（渐进式披露的第二段）。"""
        meta = self._meta.get(name)
        if meta is None:
            raise KeyError(f"没有名为 {name!r} 的 skill；已有：{sorted(self._meta)}")
        _, body = _split_front_matter(Path(meta["path"]).read_text(encoding="utf-8"))
        return body.strip()

    def match(self, task: str, top_k: int = 1) -> list[dict]:
        """按 description / keywords 给 task 打分，返回命中的 skill（含正文）。

        打分规则：skill 名或关键词直接出现在 task 里 → 权重最高；
        否则按 description 里的词与 task 的字符级重合度打分。
        """
        task_l = (task or "").lower()
        scored: list[tuple[float, dict]] = []
        for meta in self._meta.values():
            score = 0.0
            if meta["name"].lower() in task_l or meta["dir"].lower() in task_l:
                score += 5.0
            for kw in meta["keywords"]:
                if kw and kw.lower() in task_l:
                    score += 3.0
            tokens = _tokens(meta["description"])
            if tokens:
                overlap = sum(1 for t in tokens if t in task_l) / len(tokens)
                score += 2.0 * overlap
            if score > 0:
                scored.append((score, meta))

        scored.sort(key=lambda kv: (-kv[0], kv[1]["name"]))
        hits = []
        for score, meta in scored[:top_k]:
            hits.append({"name": meta["name"], "description": meta["description"],
                         "score": round(score, 3), "body": self.load(meta["name"])})
        return hits


# ----------------------------------------------------------------------
# front-matter 解析
# ----------------------------------------------------------------------
_FRONT_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.S)


def _split_front_matter(text: str) -> tuple[dict, str]:
    """拆出 YAML front-matter 与正文；没有 front-matter 就返回 ({}, 全文)。"""
    m = _FRONT_RE.match(text.lstrip("\ufeff"))
    if not m:
        return {}, text
    try:
        data = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return data, m.group(2)


def _first_meaningful_line(body: str) -> str:
    for line in body.splitlines():
        line = line.strip().lstrip("#").strip()
        if line:
            return line[:200]
    return ""


def _tokens(text: str) -> list[str]:
    """把描述切成可用于匹配的词：英文单词 + 中文 2-gram。"""
    text = text.lower()
    words = re.findall(r"[a-z_]{3,}", text)
    cjk = re.findall(r"[\u4e00-\u9fff]{2,}", text)
    grams: list[str] = []
    for run in cjk:
        grams += [run[i:i + 2] for i in range(len(run) - 1)]
    return list(dict.fromkeys(words + grams))
