#!/usr/bin/env python3
"""
judge_llm.py

判定 LLM Agent 输出是否命中 GT。

- judge_baseline：归一化坐标 → 几何 EM（bbox 或距离 < 0.04）
- judge_m2 / judge_top1_center：节点 bbox 中心 → `judge_baseline` 几何 EM（SoM / TO / TOa_top1）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from process.paths import step_paths

DISTANCE_THRESHOLD = 0.04


def load_gt(stem: str) -> dict:
    gt_path = step_paths(stem)["gt"]
    if not gt_path.is_file():
        raise FileNotFoundError(f"缺少 GT: {gt_path}")
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def screenshot_path_for(stem: str) -> Path:
    path = step_paths(stem)["screenshot"]
    if not path.is_file():
        raise FileNotFoundError(f"缺少 screenshot: {path}")
    return path


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
    gt_data = load_gt(stem)
    screenshot_path = screenshot_path_for(stem)

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


def _bounds_center(bounds: list) -> tuple[float, float] | None:
    if len(bounds) != 4:
        return None
    x1, y1, x2, y2 = (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _load_node_bounds(stem: str, node_id: int) -> list | None:
    nodes_path = step_paths(stem)["nodes"]
    if not nodes_path.is_file():
        return None
    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)
    for node in nodes:
        if int(node.get("node_id", -1)) == int(node_id):
            bounds = node.get("bounds")
            if isinstance(bounds, list) and len(bounds) == 4:
                return bounds
    return None


def _pointer_center_detail(
    stem: str,
    *,
    bounds: list | None,
    pred_node_id: int,
) -> dict:
    """节点 bbox 中心点 → judge_baseline 几何判定（与 CPM 同路径）。"""
    gt_data = load_gt(stem)
    nearest_5 = gt_data.get("nearest_5") or []
    gt_node_ids = [int(item["node_id"]) for item in nearest_5 if "node_id" in item]
    matched = sorted({int(pred_node_id)} & set(gt_node_ids))
    hit_by_node_id = len(matched) > 0

    if bounds is None or _bounds_center(bounds) is None:
        return {
            "hit": False,
            "hit_by_bbox": False,
            "hit_by_distance": False,
            "hit_by_center_distance": False,
            "hit_by_node_id": hit_by_node_id,
            "pred_node_id": int(pred_node_id),
            "gt_node_ids": gt_node_ids,
            "matched_node_ids": matched,
        }

    center_px, center_py = _bounds_center(bounds)
    screenshot_path = screenshot_path_for(stem)
    with Image.open(screenshot_path) as img:
        width, height = img.size
    norm_x = center_px / width
    norm_y = center_py / height

    baseline = judge_baseline(norm_x, norm_y, stem)
    return {
        **baseline,
        "hit_by_center_distance": baseline["hit_by_distance"],
        "hit_by_node_id": hit_by_node_id,
        "pred_node_id": int(pred_node_id),
        "gt_node_ids": gt_node_ids,
        "matched_node_ids": matched,
        "center_pixel": {"x": center_px, "y": center_py},
    }


def judge_top1_center(stem: str, top1_node: dict) -> dict:
    """
    TO / TOa_top1：取检索 top1 候选框中心点，走 judge_baseline 几何判定。

    优先使用 top1_node 内嵌 bounds（与标注/run 一致），否则回退 nodes.json。
    """
    node_id = int(top1_node["node_id"])
    bounds = top1_node.get("bounds")
    if not (isinstance(bounds, list) and len(bounds) == 4):
        bounds = _load_node_bounds(stem, node_id)
    return _pointer_center_detail(stem, bounds=bounds, pred_node_id=node_id)


def judge_m2(click_id: int, stem: str) -> dict:
    """
    判定 SoM Agent 预测的 node_id：取该节点 bbox 中心，几何判定同 judge_baseline。

    detail 中 hit_by_node_id 仅作分析，不参与 hit。
    """
    pred_id = int(click_id)
    bounds = _load_node_bounds(stem, pred_id)
    return _pointer_center_detail(stem, bounds=bounds, pred_node_id=pred_id)
