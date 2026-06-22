"""Bounding-box helpers for SMAN candidate nodes."""
from __future__ import annotations

import re

TARGET_SCREEN_SIZE = (1080, 2400)


def scale_bounds(
    bounds: list[int] | tuple[int, ...],
    from_size: tuple[int, int],
    to_size: tuple[int, int] = TARGET_SCREEN_SIZE,
) -> list[int]:
    ow, oh = from_size
    tw, th = to_size
    if ow <= 0 or oh <= 0:
        return list(bounds[:4])
    sx, sy = tw / ow, th / oh
    left, top, right, bottom = bounds[:4]
    return [
        int(round(left * sx)),
        int(round(top * sy)),
        int(round(right * sx)),
        int(round(bottom * sy)),
    ]


def parse_bounds_from_action(action_str: str) -> tuple[int, int, int, int] | None:
    nums = re.findall(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", action_str)
    if not nums:
        return None
    x1, y1, x2, y2 = map(int, nums[0])
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    return left, top, right, bottom
