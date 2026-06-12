#!/usr/bin/env python3
"""
get_GT_nearest_K.py

读取 step_GT/*.json 中的 GT 坐标 (x, y)，
读取对应 nodes/*.json 中的候选框 bounds，
按中心点距离找最近的 5 个框，
写回 step_GT/*.json 的 nearest_5 字段。
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "AC_data"
STEP_GT_DIR = BASE_DIR / "step_GT"
NODES_DIR = BASE_DIR / "nodes"


def _center_distance(bounds: list, gt_x: float, gt_y: float) -> float:
    if len(bounds) != 4:
        return float("inf")
    x1, y1, x2, y2 = bounds
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return ((cx - gt_x) ** 2 + (cy - gt_y) ** 2) ** 0.5


def process_stem(stem: str) -> None:
    gt_path = STEP_GT_DIR / f"{stem}.json"
    nodes_path = NODES_DIR / f"{stem}.json"

    if not gt_path.is_file():
        print(f"  [{stem}] 缺少 step_GT，跳过")
        return
    if not nodes_path.is_file():
        print(f"  [{stem}] 缺少 nodes，跳过")
        return

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    gt_x = float(gt_data.get("x", -1))
    gt_y = float(gt_data.get("y", -1))

    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    # 计算每个节点到 GT 的距离
    scored = []
    for node in nodes:
        dist = _center_distance(node.get("bounds", []), gt_x, gt_y)
        scored.append((dist, node))

    # 按距离升序，取前 5
    scored.sort(key=lambda t: t[0])
    nearest_5 = [
        {
            "node_id": node["node_id"],
            "bounds": node["bounds"],
        }
        for _, node in scored[:5]
    ]

    gt_data["nearest_5"] = nearest_5

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=2)


def main() -> None:
    stems = sorted(p.stem for p in STEP_GT_DIR.glob("*.json"))
    print("=" * 60)
    print("get_GT_nearest_K.py — 为每个 GT 找最近的 5 个候选框")
    print("=" * 60)
    print(f"step_GT 总数: {len(stems)}")
    if stems:
        print(f"  示例: {stems[0]} ... {stems[-1]}")
    print("=" * 60)

    if not stems:
        print("没有样本，退出。")
        sys.exit(1)

    for i, stem in enumerate(stems, 1):
        process_stem(stem)
        if i % 500 == 0 or i == len(stems):
            print(f"  已处理 {i}/{len(stems)}")

    print(f"\n{'=' * 60}")
    print("处理完成")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
