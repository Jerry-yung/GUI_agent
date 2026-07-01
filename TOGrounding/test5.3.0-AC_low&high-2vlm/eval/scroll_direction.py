"""Scroll direction mapping aligned with AgentCPM process_ac.py."""

from __future__ import annotations

# Android Control GT direction = content movement; model outputs gesture direction.
# Official: process_ac.py map_direction when converting AC GT to touch/lift points.
AC_CONTENT_TO_GESTURE: dict[str, str] = {
    "left": "right",
    "right": "left",
    "up": "down",
    "down": "up",
}


def map_ac_content_to_gesture(direction: str) -> str | None:
    """AC content-movement direction → swipe gesture direction."""
    key = str(direction or "").strip().lower()
    if not key:
        return None
    return AC_CONTENT_TO_GESTURE.get(key)


def scroll_gesture_matches_gt(pred_gesture: str, gt_ac_content: str) -> bool:
    """
  判定模型输出的滑动手势方向是否与 AC GT 一致。

  pred_gesture: VLM 输出的 direction（手势方向，对齐官方 POINT+to）
  gt_ac_content: GT 的 direction（内容移动方向）
    """
    pred = str(pred_gesture or "").strip().lower()
    expected = map_ac_content_to_gesture(gt_ac_content)
    if not pred or expected is None:
        return False
    return pred == expected
