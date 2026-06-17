"""Reverse map global action_id -> SMAN area label (cN/sN).

Mirrors SMAN ``action_click`` / ``action_scroll`` forward mapping used in
``apply_action``: area index in page candidates <-> global action_id.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from utils.sman_setup import get_sman_utils

if TYPE_CHECKING:
    from utils.sman_bridge import RoundAssets

_SCROLL_DIRS = ("up", "down", "left", "right")


def _action_kind(action_str: str) -> str:
    return (action_str or "").split("(")[0].strip().lower()


def click_text_from_candidate(inner: str) -> str:
    inner = (inner or "").strip()
    m = re.match(r"click\((.+),\s*\[", inner)
    if m:
        return m.group(1).strip()
    m = re.search(r'description="([^"]+)"', inner)
    if m:
        return m.group(1)
    m = re.match(r"^([^,]+),", inner)
    if m:
        text = m.group(1).strip()
        if text.lower().startswith("click("):
            return click_text_from_candidate(text)
        return text
    return inner[:80]


def click_area_idx_from_action_id(
    action_id: int,
    click_actions: list[str],
    current_page_all_actions: dict,
    all_action_ids: dict[str, int],
) -> int | None:
    """Inverse of ``action_click``: global id -> 1-based click area index."""
    for i, candidate in enumerate(click_actions):
        info = current_page_all_actions.get(candidate)
        if info is None:
            continue
        if all_action_ids.get(f"click({info})") == action_id:
            return i + 1
    for i, inner in enumerate(click_actions):
        key = inner if inner.startswith("click(") else f"click({inner})"
        if all_action_ids.get(key) == action_id:
            return i + 1
    return None


def click_label_from_action_id(
    action_id: int,
    click_actions: list[str],
    current_page_all_actions: dict,
    all_action_ids: dict[str, int],
) -> str | None:
    area_idx = click_area_idx_from_action_id(
        action_id, click_actions, current_page_all_actions, all_action_ids
    )
    return f"c{area_idx}" if area_idx is not None else None


def _scroll_coords_from_action_str(action_str: str) -> tuple[int, int, int, int] | None:
    """Extract (sx, sy, ex, ey) from scroll(([sx, sy], [ex, ey]))."""
    import re as _re
    nums = _re.findall(r"\d+", action_str)
    if len(nums) >= 4:
        return int(nums[0]), int(nums[1]), int(nums[2]), int(nums[3])
    return None


def scroll_area_from_action_id(
    action_id: int,
    scroll_bounds: list[str],
    all_action_ids: dict[str, int],
    *,
    coord_tolerance: int = 20,
) -> tuple[int | None, str | None]:
    """Inverse of ``action_scroll``: global id -> (1-based scroll area, direction).

    First tries exact key match; if that fails, falls back to finding the
    scroll_bounds + direction whose generated coordinates are closest to the
    GT action's coordinates (within ``coord_tolerance`` pixels each axis).
    This handles the common case where the GT was recorded with a gesture
    using slightly different pixel coordinates than what SMAN re-generates
    from the XML (they share the same scroll region, same direction, same
    semantic intent, but differ by a few pixels).
    """
    sman = get_sman_utils()
    dict_scroll = sman.dict_scroll_parameters

    # Pass 1: exact match
    for i, bounds in enumerate(scroll_bounds):
        for direction in _SCROLL_DIRS:
            scroll_key = f"scroll({dict_scroll(bounds, direction)})"
            if all_action_ids.get(scroll_key) == action_id:
                return i + 1, direction

    # Pass 2: approximate coordinate match
    gt_action_str = None
    for k, v in all_action_ids.items():
        if v == action_id:
            gt_action_str = k
            break
    if gt_action_str is None:
        return None, None

    gt_coords = _scroll_coords_from_action_str(gt_action_str)
    if gt_coords is None:
        return None, None
    gsx, gsy, gex, gey = gt_coords

    # Determine intended direction from GT coords
    dx, dy = gex - gsx, gey - gsy
    if abs(dx) >= abs(dy):
        gt_dir = "right" if dx >= 0 else "left"
    else:
        gt_dir = "down" if dy >= 0 else "up"

    best_dist: float = float("inf")
    best_area: int | None = None

    for i, bounds in enumerate(scroll_bounds):
        sa = dict_scroll(bounds, gt_dir)
        csx, csy = sa[0][0], sa[0][1]
        cex, cey = sa[1][0], sa[1][1]
        dist = ((csx - gsx) ** 2 + (csy - gsy) ** 2 + (cex - gex) ** 2 + (cey - gey) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_area = i + 1

    if best_area is not None and best_dist <= coord_tolerance * 2:
        return best_area, gt_dir

    return None, None


def scroll_label_from_action_id(
    action_id: int,
    scroll_bounds: list[str],
    all_action_ids: dict[str, int],
) -> tuple[str | None, str | None]:
    area_idx, direction = scroll_area_from_action_id(
        action_id, scroll_bounds, all_action_ids
    )
    if area_idx is None:
        return None, direction
    return f"s{area_idx}", direction


def _scroll_direction_from_action_str(action_str: str) -> str | None:
    m = re.search(r",\s*(up|down|left|right)\)\s*$", action_str, re.IGNORECASE)
    return m.group(1).lower() if m else None


@dataclass
class ResolvedAreaLabel:
    kind: str
    label: str | None = None
    direction: str | None = None
    text: str | None = None


def resolve_area_label_from_action_id(
    action_id: int,
    action_str: str,
    *,
    assets: RoundAssets | None,
    all_action_ids: dict[str, int],
) -> ResolvedAreaLabel:
    """Resolve global action_id to SMAN cN/sN (+ text/direction) on current page."""
    kind = _action_kind(action_str)
    if assets is None:
        return ResolvedAreaLabel(kind=kind)

    if kind == "click":
        label = click_label_from_action_id(
            action_id,
            assets.click_actions,
            assets.current_page_all_actions,
            all_action_ids,
        )
        text = "?"
        if label:
            m = re.search(r"c(\d+)", label, re.IGNORECASE)
            if m:
                idx = int(m.group(1)) - 1
                if 0 <= idx < len(assets.click_actions):
                    text = click_text_from_candidate(assets.click_actions[idx])
        if text == "?":
            m = re.search(r'description="([^"]+)"', action_str)
            if m:
                text = m.group(1)
        return ResolvedAreaLabel(kind=kind, label=label, text=text)

    if kind == "scroll":
        label, direction = scroll_label_from_action_id(
            action_id, assets.scroll_action_bounds, all_action_ids
        )
        if not direction:
            direction = _scroll_direction_from_action_str(action_str)
        return ResolvedAreaLabel(kind=kind, label=label, direction=direction)

    if kind == "input":
        m = re.match(r"input\((.+),\s*\[", action_str)
        text = m.group(1).strip() if m else action_str
        return ResolvedAreaLabel(kind=kind, text=text)

    return ResolvedAreaLabel(kind=kind)


def format_labeled_action_from_action_id(
    action_id: int,
    action_str: str,
    *,
    assets: RoundAssets | None,
    all_action_ids: dict[str, int],
) -> str:
    """VLM-style display string from global action_id, e.g. ``click(c39, 歌曲队列)``."""
    resolved = resolve_area_label_from_action_id(
        action_id, action_str, assets=assets, all_action_ids=all_action_ids
    )
    if resolved.kind == "click":
        label = resolved.label or "?"
        text = resolved.text or "?"
        return f"click({label}, {text})"
    if resolved.kind == "scroll":
        label = resolved.label or "?"
        direction = resolved.direction or "?"
        return f"scroll({label}, {direction})"
    if resolved.kind == "input":
        return f"input({resolved.text or action_str})"
    return (action_str or "?")[:72]
