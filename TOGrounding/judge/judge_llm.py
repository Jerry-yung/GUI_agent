#!/usr/bin/env python3
"""
judge_llm.py

判定 LLM Agent 输出是否命中 GT。

- judge_baseline：归一化坐标 → 几何 EM（bbox 或距离 < 0.04）
- judge_m2：node_id → judge hit@1（pred node_id 在 nearest_5 中）
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "AC_data"
STEP_GT_DIR = BASE_DIR / "step_GT"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

DISTANCE_THRESHOLD = 0.04


def denormalize_coordinates(
    norm_x: float, norm_y: float, screenshot_path: Path
) -> tuple[float, float, int, int]:
    """归一化坐标 → 像素坐标，同时返回图片尺寸。"""
    with Image.open(screenshot_path) as img:
        width, height = img.size
    px = norm_x * width
    py = norm_y * height
    return px, py, width, height


def point_in_bounds(px: float, py: float, bounds: list) -> bool:
    """判断点是否在 bbox 内（含边界）。"""
    if len(bounds) != 4:
        return False
    x1, y1, x2, y2 = bounds
    return x1 <= px <= x2 and y1 <= py <= y2


def normalized_distance(
    px: float, py: float, gt_x: float, gt_y: float, width: int, height: int
) -> float:
    """归一化坐标下的欧氏距离。"""
    nx_pred, ny_pred = px / width, py / height
    nx_gt, ny_gt = gt_x / width, gt_y / height
    return float(np.linalg.norm(np.array([nx_pred, ny_pred]) - np.array([nx_gt, ny_gt])))


def judge_baseline(norm_x: float, norm_y: float, stem: str) -> dict:
    """
    对单个 stem 判定 Baseline Agent 预测是否命中。

    返回:
        {
            "hit": bool,
            "hit_by_bbox": bool,
            "hit_by_distance": bool,
            "norm_distance": float,
            "pixel_x": float,
            "pixel_y": float,
            "image_size": {"width": int, "height": int},
            "gt": {"x": float, "y": float},
        }
    """
    gt_path = STEP_GT_DIR / f"{stem}.json"
    screenshot_path = SCREENSHOTS_DIR / f"{stem}.png"

    if not gt_path.is_file():
        raise FileNotFoundError(f"缺少 step_GT: {gt_path}")
    if not screenshot_path.is_file():
        raise FileNotFoundError(f"缺少 screenshot: {screenshot_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_x = float(gt_data["x"])
    gt_y = float(gt_data["y"])
    nearest_5 = gt_data.get("nearest_5", [])

    px, py, width, height = denormalize_coordinates(norm_x, norm_y, screenshot_path)

    hit_by_bbox = any(
        point_in_bounds(px, py, item["bounds"])
        for item in nearest_5
        if "bounds" in item
    )
    norm_dist = normalized_distance(px, py, gt_x, gt_y, width, height)
    hit_by_distance = norm_dist < DISTANCE_THRESHOLD

    hit = hit_by_bbox or hit_by_distance

    return {
        "hit": hit,
        "hit_by_bbox": hit_by_bbox,
        "hit_by_distance": hit_by_distance,
        "norm_distance": round(norm_dist, 6),
        "pixel_x": px,
        "pixel_y": py,
        "image_size": {"width": width, "height": height},
        "gt": {"x": gt_x, "y": gt_y},
    }


def judge_m2(click_id: int, stem: str) -> dict:
    """
    判定 M2 Agent 预测的 node_id 是否命中（同 judge hit@1）。

    hit：click_id 属于 step_GT.nearest_5 的 node_id 集合。
    """
    gt_path = STEP_GT_DIR / f"{stem}.json"
    if not gt_path.is_file():
        raise FileNotFoundError(f"缺少 step_GT: {gt_path}")

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_node_ids = [int(item["node_id"]) for item in gt_data.get("nearest_5", [])]
    gt_set = set(gt_node_ids)
    pred_id = int(click_id)
    matched = sorted({pred_id} & gt_set)

    return {
        "hit": len(matched) > 0,
        "pred_node_id": pred_id,
        "gt_node_ids": gt_node_ids,
        "matched_node_ids": matched,
        "gt": {"x": float(gt_data["x"]), "y": float(gt_data["y"])},
    }
