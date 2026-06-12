#!/usr/bin/env python3
"""
hit_1.py

top_1 评测（固定 top_k=1），对齐 AgentCPM-GUI evaluator 的 exact_match 思想：

1. 按 MODE（best / mid / worst）从 rank_to 选取 TO，取相似度最高的 1 个 node
2. 计算该 node bounds 的中心点 (cx, cy)
3. hit 条件（满足其一即可）：
   - 中心点落在 step_GT nearest_5 任一 bbox 内
   - 中心点与 GT (x,y) 的归一化欧氏距离 < 0.04
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_JUDGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _JUDGE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from judge.judge import _rank_top_k, _sims_for_mode
    from judge.rank_to import TO_MODES
except ImportError:
    import importlib.util

    _spec = importlib.util.spec_from_file_location("judge_core", _JUDGE_DIR / "judge.py")
    _judge_core = importlib.util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_judge_core)
    _rank_top_k = _judge_core._rank_top_k
    _sims_for_mode = _judge_core._sims_for_mode
    from rank_to import TO_MODES

PROJECT_ROOT = _PROJECT_ROOT
BASE_DIR = PROJECT_ROOT / "AC_data"
COS_SIM_DIR = BASE_DIR / "embeddings" / "cos_sim"
STEP_GT_DIR = BASE_DIR / "step_GT"
NODES_DIR = BASE_DIR / "nodes"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

DISTANCE_THRESHOLD = 0.04


def bounds_center(bounds: list) -> tuple[float, float]:
    """bounds: [x1, y1, x2, y2] → 中心点 (cx, cy)。"""
    if len(bounds) != 4:
        raise ValueError(f"invalid bounds: {bounds}")
    x1, y1, x2, y2 = bounds
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_bounds(x: float, y: float, bounds: list) -> bool:
    """判断点是否在 bbox 内（含边界）。"""
    if len(bounds) != 4:
        return False
    x1, y1, x2, y2 = bounds
    return x1 <= x <= x2 and y1 <= y <= y2


def normalized_distance(
    px: float,
    py: float,
    gt_x: float,
    gt_y: float,
    image_width: int,
    image_height: int,
) -> float:
    """归一化坐标下的欧氏距离，与 evaluator.py 一致。"""
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"invalid image size: {image_width}x{image_height}")
    nx_pred, ny_pred = px / image_width, py / image_height
    nx_gt, ny_gt = gt_x / image_width, gt_y / image_height
    return float(np.linalg.norm(np.array([nx_pred, ny_pred]) - np.array([nx_gt, ny_gt])))


def get_screen_size(stem: str, nodes: list[dict]) -> tuple[int, int]:
    """优先读 screenshot 尺寸，否则从 nodes bounds 推断。"""
    screenshot_path = SCREENSHOTS_DIR / f"{stem}.png"
    if screenshot_path.is_file():
        from PIL import Image

        with Image.open(screenshot_path) as img:
            return img.size

    max_x, max_y = 1, 1
    for node in nodes:
        bounds = node.get("bounds", [])
        if len(bounds) == 4:
            max_x = max(max_x, int(bounds[2]))
            max_y = max(max_y, int(bounds[3]))
    return max_x, max_y


def _load_nodes_by_id(stem: str) -> dict[int, dict]:
    nodes_path = NODES_DIR / f"{stem}.json"
    if not nodes_path.is_file():
        raise FileNotFoundError(f"缺少 nodes: {nodes_path}")
    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)
    return {int(n["node_id"]): n for n in nodes}


def hit_1(stem: str, mode: str) -> dict[str, Any]:
    """
    对单个 stem 做 top_1 hit 判定。

    参数:
        stem: 样本名
        mode: "best" | "mid" | "worst"
    """
    cos_sim_path = COS_SIM_DIR / f"{stem}.npz"
    gt_path = STEP_GT_DIR / f"{stem}.json"

    if not cos_sim_path.is_file():
        raise FileNotFoundError(f"缺少 cos_sim: {cos_sim_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"缺少 step_GT: {gt_path}")

    with np.load(cos_sim_path) as data:
        sim_matrix = data["sim_matrix"]
        node_ids = data["node_ids"]
        to_ids = data["to_ids"]

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    nodes_by_id = _load_nodes_by_id(stem)
    image_width, image_height = get_screen_size(stem, list(nodes_by_id.values()))

    final_sims, to_id = _sims_for_mode(stem, sim_matrix, to_ids, mode)
    pred_node_ids, pred_scores = _rank_top_k(final_sims, node_ids, 1)
    pred_node_id = pred_node_ids[0]
    pred_score = pred_scores[0]

    pred_node = nodes_by_id.get(pred_node_id)
    if pred_node is None:
        raise ValueError(f"[{stem}] top_1 node_id={pred_node_id} 不在 nodes 中")

    cx, cy = bounds_center(pred_node["bounds"])
    gt_x = float(gt_data["x"])
    gt_y = float(gt_data["y"])
    nearest_5 = gt_data.get("nearest_5", [])

    hit_by_bbox = any(
        point_in_bounds(cx, cy, item["bounds"]) for item in nearest_5 if "bounds" in item
    )
    norm_dist = normalized_distance(cx, cy, gt_x, gt_y, image_width, image_height)
    hit_by_distance = norm_dist < DISTANCE_THRESHOLD
    hit = hit_by_bbox or hit_by_distance

    return {
        "stem": stem,
        "hit": hit,
        "mode": mode.lower(),
        "to_id": to_id,
        "pred_node_id": pred_node_id,
        "pred_score": pred_score,
        "pred_center": {"x": cx, "y": cy},
        "gt_point": {"x": gt_x, "y": gt_y},
        "hit_by_bbox": hit_by_bbox,
        "hit_by_distance": hit_by_distance,
        "norm_distance": round(norm_dist, 6),
        "distance_threshold": DISTANCE_THRESHOLD,
        "image_size": {"width": image_width, "height": image_height},
    }


def hit_1_all(
    stems: list[str] | None = None,
    mode: str = "best",
) -> dict[str, Any]:
    """批量 top_1 评测。"""
    if stems is None:
        stems = sorted(p.stem for p in COS_SIM_DIR.glob("*.npz"))

    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for stem in stems:
        try:
            results.append(hit_1(stem, mode))
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{stem}: {exc}")

    hit_count = sum(1 for r in results if r["hit"])
    total = len(results)

    return {
        "mode": mode.lower(),
        "total": total,
        "hit_count": hit_count,
        "hit_rate": round(hit_count / total, 4) if total > 0 else 0.0,
        "results": results,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="hit_1.py — top_1 中心点 hit 评测")
    parser.add_argument(
        "--mode",
        type=str,
        default="best",
        choices=list(TO_MODES),
        help='TO 选取策略: "best" | "mid" | "worst"',
    )
    parser.add_argument("--stem", type=str, default=None, help="只评测单个 stem")
    args = parser.parse_args()

    print("=" * 60)
    print("hit_1.py — top_1 中心点 hit（bbox ∪ 距离<0.04）")
    print("=" * 60)
    print(f"mode={args.mode}")
    print("=" * 60)

    if args.stem:
        result = hit_1(args.stem, args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    summary = hit_1_all(mode=args.mode)
    print(f"样本数: {summary['total']}")
    print(f"命中数: {summary['hit_count']}")
    print(f"hit@1 ({args.mode}): {summary['hit_rate']}")
    if summary["errors"]:
        print(f"错误/跳过: {len(summary['errors'])}")
        for msg in summary["errors"][:5]:
            print(f"  {msg}")
        if len(summary["errors"]) > 5:
            print(f"  ... 共 {len(summary['errors'])} 条")
    print("=" * 60)


if __name__ == "__main__":
    main()
