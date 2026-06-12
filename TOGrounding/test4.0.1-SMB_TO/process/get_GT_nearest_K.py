#!/usr/bin/env python3
"""
get_GT_nearest_K.py

在 compress 之后，将 step_GT 中的 gt_action (+ gt_bounds) 映射到 nodes 中的 gt_node_id：
  1. 优先 resource-id / text / description 匹配，取面积最小 node
  2. 否则用 gt_bounds 与 nodes 做 IoU / 包含关系匹配

写回 step_GT：
  - gt_node_id
  - x, y（gt node bounds 中心，供 Baseline 距离判定）
  - nearest_5：仅含 gt_node_id 对应的一个框
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.data_paths import BASE_DIR
from utils.gt_node_match import (
    bbox_center,
    gt_action_from_step_gt,
    match_gt_node_in_nodes,
)

STEP_GT_DIR = BASE_DIR / "step_GT"
NODES_DIR = BASE_DIR / "nodes"


def process_stem(stem: str) -> str:
    """返回 ok | skip | fail"""
    gt_path = STEP_GT_DIR / f"{stem}.json"
    nodes_path = NODES_DIR / f"{stem}.json"

    if not gt_path.is_file():
        print(f"  [{stem}] 缺少 step_GT，跳过")
        return "skip"
    if not nodes_path.is_file():
        print(f"  [{stem}] 缺少 nodes，跳过")
        return "skip"

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    gt_action = gt_action_from_step_gt(gt_data)
    gt_bounds = gt_data.get("gt_bounds")
    if isinstance(gt_bounds, list) and len(gt_bounds) != 4:
        gt_bounds = None

    if not gt_action.startswith("click("):
        print(f"  [{stem}] 非 click GT，跳过")
        return "skip"

    matched = match_gt_node_in_nodes(nodes, gt_action, gt_bounds)
    if matched is None:
        print(f"  [{stem}] 无法匹配 gt_node_id")
        return "fail"

    node_id = int(matched["node_id"])
    bounds = matched["bounds"]
    cx, cy = bbox_center(bounds)

    gt_data["gt_action"] = gt_action
    gt_data["gt_node_id"] = node_id
    gt_data["gt_bounds"] = gt_bounds if gt_bounds else bounds
    gt_data["x"] = round(cx, 2)
    gt_data["y"] = round(cy, 2)
    gt_data["nearest_5"] = [{"node_id": node_id, "bounds": bounds}]

    with open(gt_path, "w", encoding="utf-8") as f:
        json.dump(gt_data, f, ensure_ascii=False, indent=2)

    return "ok"


def main() -> None:
    stems = sorted(p.stem for p in STEP_GT_DIR.glob("*.json"))
    print("=" * 60)
    print("get_GT_nearest_K.py — gt_action → gt_node_id + nearest_5")
    print("=" * 60)
    print(f"step_GT 总数: {len(stems)}")
    if stems:
        print(f"  示例: {stems[0]} ... {stems[-1]}")
    print("=" * 60)

    if not stems:
        print("没有样本，退出。")
        sys.exit(1)

    ok = fail = skip = 0
    for i, stem in enumerate(stems, 1):
        result = process_stem(stem)
        if result == "ok":
            ok += 1
        elif result == "fail":
            fail += 1
        else:
            skip += 1
        if i % 500 == 0 or i == len(stems):
            print(f"  已处理 {i}/{len(stems)}")

    print(f"\n{'=' * 60}")
    print(f"处理完成: 成功={ok}, 失败={fail}, 跳过={skip}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
