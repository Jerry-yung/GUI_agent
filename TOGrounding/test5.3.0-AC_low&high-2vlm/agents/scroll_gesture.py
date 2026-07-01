"""Scroll gesture direction helpers for m2 / m2p / TO agents."""

from __future__ import annotations

import re

from eval.scroll_direction import map_ac_content_to_gesture

_LIST_SCROLL_DOWN = re.compile(
    r"\b(?:scroll|swipe)\s+down\b(?:\s+to\s+(?:view|see|check|show)\b|\s+to\s+view\s+more\b|(?:\s+and\b))",
    re.I,
)
_DIAL_SLIDER = re.compile(
    r"\b(?:minute|hour|dial|slider|wheel|picker)\b",
    re.I,
)
_SCROLL_DOWN = re.compile(r"\b(?:scroll|swipe)\s+down\b", re.I)
_LIST_BROWSE = re.compile(
    r"\b(?:more|options|prices|cart|product|reviews)\b",
    re.I,
)


def normalize_scroll_gesture_from_instruction(
    direction: str,
    instruction: str,
) -> str:
    """
    当模型照抄 step instruction 中的 scroll/swipe down（列表浏览语义）时，
    将 content-down 翻转为手势 up。不处理 dial/slider 等物理控件。
    """
    direction = str(direction or "").strip().lower()
    instruction = instruction or ""
    if direction != "down":
        return direction
    if _DIAL_SLIDER.search(instruction):
        return direction
    if _LIST_SCROLL_DOWN.search(instruction):
        return map_ac_content_to_gesture("down") or direction
    if _SCROLL_DOWN.search(instruction) and _LIST_BROWSE.search(instruction):
        return map_ac_content_to_gesture("down") or direction
    return direction
