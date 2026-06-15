"""runs/*.json 文件命名与解析。"""

from __future__ import annotations

import re
from pathlib import Path

TO_SELECT_PATTERN = r"(?:best|mid|worst|generate)"
RUN_NAME_RE = re.compile(
    rf"^(?P<ac_mode>low|high)_(?P<agent>[^_]+)_top(?P<top_k>\d+)(?P<to_select>{TO_SELECT_PATTERN})_(?P<vlm_model>.+)$"
)
CPM_RUN_NAME_RE = re.compile(
    r"^(?P<ac_mode>low|high)_(?P<agent>CPM)_(?P<vlm_model>.+)$"
)


def run_filename(
    ac_mode: str,
    agent: str,
    top_k: int,
    vlm_model: str,
    *,
    to_select: str = "generate",
) -> str:
    """生成 runs 下的 JSON 文件名。"""
    ac_mode = ac_mode.lower()
    agent = agent.strip()
    to_select = to_select.lower()
    if agent.upper() == "CPM":
        return f"{ac_mode}_{agent}_{vlm_model}.json"
    return f"{ac_mode}_{agent}_top{top_k}{to_select}_{vlm_model}.json"


def parse_run_filename(stem: str) -> dict | None:
    """从文件名 stem（无 .json）解析 ac_mode / agent / top_k / to_select / vlm_model。"""
    m = RUN_NAME_RE.match(stem)
    if m:
        return {
            "ac_mode": m.group("ac_mode"),
            "agent": m.group("agent"),
            "top_k": int(m.group("top_k")),
            "to_select": m.group("to_select"),
            "vlm_model": m.group("vlm_model"),
        }
    m = CPM_RUN_NAME_RE.match(stem)
    if m:
        return {
            "ac_mode": m.group("ac_mode"),
            "agent": m.group("agent"),
            "top_k": None,
            "to_select": None,
            "vlm_model": m.group("vlm_model"),
        }
    return None


def run_label(meta: dict) -> str:
    """对比表用的简短标签。"""
    agent = str(meta.get("agent", ""))
    top_k = meta.get("top_k")
    if top_k is None:
        return agent
    to_select = meta.get("to_select") or "generate"
    return f"{agent}_top{top_k}{to_select}"
