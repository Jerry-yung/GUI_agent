"""标注前节点 bounds 有效性检查。"""

from __future__ import annotations

FULL_SCREEN_RATIO = 0.95


def _visible_rect(
    bounds: list,
    screen_w: int,
    screen_h: int,
) -> tuple[int, int, int, int] | None:
    if len(bounds) != 4:
        return None
    x1, y1, x2, y2 = (int(bounds[0]), int(bounds[1]), int(bounds[2]), int(bounds[3]))
    if x2 <= x1 or y2 <= y1:
        return None

    vx1 = max(0, min(x1, screen_w))
    vy1 = max(0, min(y1, screen_h))
    vx2 = max(0, min(x2, screen_w))
    vy2 = max(0, min(y2, screen_h))
    if vx2 <= vx1 or vy2 <= vy1:
        return None
    return vx1, vy1, vx2, vy2


def is_valid_top1_bounds(
    bounds: list,
    screen_w: int,
    screen_h: int,
    *,
    full_screen_ratio: float = FULL_SCREEN_RATIO,
) -> bool:
    """
    top1 节点 bounds 是否可用于标注：
    - 与屏幕有正面积交集
    - 可见面积未覆盖整屏（默认 < 95%）
    """
    if screen_w <= 0 or screen_h <= 0:
        return False

    rect = _visible_rect(bounds, screen_w, screen_h)
    if rect is None:
        return False

    vx1, vy1, vx2, vy2 = rect
    visible_area = (vx2 - vx1) * (vy2 - vy1)
    screen_area = screen_w * screen_h
    if screen_area <= 0:
        return False
    return (visible_area / screen_area) < full_screen_ratio
