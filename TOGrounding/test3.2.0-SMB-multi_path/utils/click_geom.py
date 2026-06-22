"""Normalized click coordinates: bbox hit eval and SMAN area mapping."""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image

from annotate.bounds import TARGET_SCREEN_SIZE, parse_bounds_from_action, scale_bounds
from utils.action_id_resolve import click_area_idx_from_action_id

NORM_DISTANCE_THRESHOLD = 0.04


def _valid_norm_coord(x: float, y: float) -> bool:
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def parse_norm_coords(raw_x, raw_y) -> tuple[float, float] | None:
    try:
        x, y = float(raw_x), float(raw_y)
    except (TypeError, ValueError):
        return None
    if not _valid_norm_coord(x, y):
        return None
    return x, y


def screenshot_size(screenshot_path: str | Path) -> tuple[int, int]:
    with Image.open(screenshot_path) as img:
        return img.size


def denormalize_point(
    norm_x: float,
    norm_y: float,
    *,
    width: int = TARGET_SCREEN_SIZE[0],
    height: int = TARGET_SCREEN_SIZE[1],
) -> tuple[float, float]:
    return norm_x * width, norm_y * height


def point_in_bounds(px: float, py: float, bounds: list[int] | tuple[int, ...]) -> bool:
    if len(bounds) < 4:
        return False
    x1, y1, x2, y2 = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    return left <= px <= right and top <= py <= bottom


def _scaled_click_bounds(
    click_actions: list[str],
    screenshot_path: str | Path,
    *,
    to_size: tuple[int, int] = TARGET_SCREEN_SIZE,
) -> list[tuple[int, list[int]]]:
    orig = screenshot_size(screenshot_path)
    out: list[tuple[int, list[int]]] = []
    for i, act in enumerate(click_actions):
        raw = parse_bounds_from_action(act)
        if raw is None:
            continue
        out.append((i + 1, scale_bounds(raw, orig, to_size)))
    return out


def gt_click_bounds_scaled(
    gt_id: int,
    *,
    click_actions: list[str],
    current_page_all_actions: dict,
    all_action_ids: dict[str, int],
    screenshot_path: str | Path,
    to_size: tuple[int, int] = TARGET_SCREEN_SIZE,
) -> list[int] | None:
    area_idx = click_area_idx_from_action_id(
        gt_id, click_actions, current_page_all_actions, all_action_ids
    )
    if area_idx is None or area_idx < 1 or area_idx > len(click_actions):
        return None
    raw = parse_bounds_from_action(click_actions[area_idx - 1])
    if raw is None:
        return None
    return scale_bounds(raw, screenshot_size(screenshot_path), to_size)


def normalized_distance_to_bounds_center(
    norm_x: float,
    norm_y: float,
    bounds: list[int] | tuple[int, ...],
    *,
    width: int = TARGET_SCREEN_SIZE[0],
    height: int = TARGET_SCREEN_SIZE[1],
) -> float:
    px, py = denormalize_point(norm_x, norm_y, width=width, height=height)
    x1, y1, x2, y2 = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    nx_pred, ny_pred = px / width, py / height
    nx_gt, ny_gt = cx / width, cy / height
    return math.hypot(nx_pred - nx_gt, ny_pred - ny_gt)


def judge_click_norm_coords(
    norm_x: float,
    norm_y: float,
    gt_id: int,
    *,
    click_actions: list[str],
    current_page_all_actions: dict,
    all_action_ids: dict[str, int],
    screenshot_path: str | Path,
) -> dict:
    """Point-in-GT-bbox (+ optional center distance) for TOa coord clicks."""
    width, height = TARGET_SCREEN_SIZE
    px, py = denormalize_point(norm_x, norm_y, width=width, height=height)
    gt_bounds = gt_click_bounds_scaled(
        gt_id,
        click_actions=click_actions,
        current_page_all_actions=current_page_all_actions,
        all_action_ids=all_action_ids,
        screenshot_path=screenshot_path,
    )
    hit_by_bbox = bool(gt_bounds and point_in_bounds(px, py, gt_bounds))
    hit_by_distance = False
    norm_dist = None
    if gt_bounds is not None:
        norm_dist = normalized_distance_to_bounds_center(
            norm_x, norm_y, gt_bounds, width=width, height=height
        )
        hit_by_distance = norm_dist < NORM_DISTANCE_THRESHOLD
    return {
        "hit": hit_by_bbox or hit_by_distance,
        "hit_by_bbox": hit_by_bbox,
        "hit_by_distance": hit_by_distance,
        "norm_distance": round(norm_dist, 6) if norm_dist is not None else None,
        "norm_x": norm_x,
        "norm_y": norm_y,
        "pixel_x": px,
        "pixel_y": py,
    }


def click_area_idx_from_norm_coords(
    norm_x: float,
    norm_y: float,
    click_actions: list[str],
    screenshot_path: str | Path,
) -> int | None:
    """Map normalized tap to 1-based SMAN click area index (for apply_action)."""
    width, height = TARGET_SCREEN_SIZE
    px, py = denormalize_point(norm_x, norm_y, width=width, height=height)
    candidates = _scaled_click_bounds(click_actions, screenshot_path)
    containing: list[tuple[int, int]] = []
    for area_idx, bounds in candidates:
        if point_in_bounds(px, py, bounds):
            x1, y1, x2, y2 = bounds
            area = max(0, (x2 - x1) * (y2 - y1))
            containing.append((area, area_idx))
    if containing:
        containing.sort(key=lambda t: t[0])
        return containing[0][1]

    best_idx: int | None = None
    best_dist = float("inf")
    for area_idx, bounds in candidates:
        dist = normalized_distance_to_bounds_center(
            norm_x, norm_y, bounds, width=width, height=height
        )
        if dist < best_dist:
            best_dist = dist
            best_idx = area_idx
    return best_idx
