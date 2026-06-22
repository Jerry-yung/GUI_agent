"""Infer action-type hint from step instruction keywords (pool routing only)."""
from __future__ import annotations

from typing import Literal

InstructionHit = Literal["click", "scroll", "input", "ambiguous"]

SCROLL_KEYWORDS = [
    "滑动",
    "滚动",
    "滑屏",
    "翻页",
    "划动",
    "划屏",
    "scroll",
    "swipe",
    "fling",
    "向下滚",
    "向上滚",
    "向左滚",
    "向右滚",
    "向下滑",
    "向上滑",
    "向左滑",
    "向右滑",
    "横向滚",
    "纵向滚",
    "继续滚",
]

CLICK_KEYWORDS = [
    "点击",
    "点按",
    "轻点",
    "click",
    "tap",
    "按一下",
]

INPUT_KEYWORDS = [
    "输入",
    "填写",
    "键入",
    "input",
    "填入",
]


def _matches_any(text: str, keywords: list[str]) -> bool:
    lowered = text.lower()
    return any(kw.lower() in lowered for kw in keywords)


def infer_instruction_hit(step_instruction: str) -> InstructionHit:
    """Return unambiguous hint when exactly one keyword group matches."""
    text = step_instruction or ""
    hits = {
        "scroll": _matches_any(text, SCROLL_KEYWORDS),
        "click": _matches_any(text, CLICK_KEYWORDS),
        "input": _matches_any(text, INPUT_KEYWORDS),
    }
    matched = [k for k, ok in hits.items() if ok]
    if len(matched) == 1:
        return matched[0]  # type: ignore[return-value]
    return "ambiguous"
