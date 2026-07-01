#!/usr/bin/env python3
"""
get_GT_nearest_K.py

读取 steps/{episode_id}/{step_idx}/{stem}_gt.json 中的 GT 坐标 (x, y)，
读取同目录 {stem}_nodes.json 中的候选框 bounds，
按中心点距离找最近的 5 个框，写回 {stem}_gt.json 的 nearest_5 字段。

仅处理含 x、y 坐标的 GT（如 click、long_press）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from process.paths import iter_stems, step_paths


def _center_distance(bounds: list, gt_x: float, gt_y: float) -> float:
    if len(bounds) != 4:
        return float("inf")
    x1, y1, x2, y2 = bounds
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return ((cx - gt_x) ** 2 + (cy - gt_y) ** 2) ** 0.5


def process_stem(stem: str) -> str:
    paths = step_paths(stem)
    gt_path = paths["gt"]
    nodes_path = paths["nodes"]

    if not gt_path.is_file():
        return "no_gt"
    if not nodes_path.is_file():
        return "no_nodes"

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    if "x" not in gt_data or "y" not in gt_data:
        return "no_xy"

    gt_x = float(gt_data["x"])
    gt_y = float(gt_data["y"])

    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    if not nodes:
        return "empty_nodes"

    scored = []
    for node in nodes:
        dist = _center_distance(node.get("bounds", []), gt_x, gt_y)
        scored.append((dist, node))

    scored.sort(key=lambda t: t[0])
    nearest_5 = [
        {"node_id": node["node_id"], "bounds": node["bounds"]}
        for _, node in scored[:5]
    ]

    gt_data["nearest_5"] = nearest_5
    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=2)
    return "ok"


def main() -> None:
    stems = iter_stems()
    print("=" * 60)
    print("get_GT_nearest_K.py — 为 GT 写入 nearest_5")
    print("=" * 60)
    print(f"steps 样本数: {len(stems)}")
    if stems:
        print(f"  示例: {stems[0]} ... {stems[-1]}")
    print("=" * 60)

    if not stems:
        print("没有样本，退出。")
        sys.exit(1)

    stats = {"ok": 0, "no_gt": 0, "no_nodes": 0, "no_xy": 0, "empty_nodes": 0}
    for i, stem in enumerate(stems, 1):
        result = process_stem(stem)
        stats[result] = stats.get(result, 0) + 1
        if i % 500 == 0 or i == len(stems):
            print(f"  已处理 {i}/{len(stems)}")

    print(f"\n{'=' * 60}")
    print("处理完成")
    print(f"  写入 nearest_5: {stats['ok']}")
    print(f"  无坐标跳过:     {stats['no_xy']}")
    print(f"  缺 nodes:       {stats['no_nodes']}")
    print(f"  空 nodes:       {stats['empty_nodes']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
