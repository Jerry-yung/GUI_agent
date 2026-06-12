#!/usr/bin/env python3
"""
根据 cos_sim + MODE 取 top_k 节点，在截图上绘制框与 node_id，
输出到 agents/annotate/annotated_screenshots/top_{top_k}_{mode}/{stem}.png

MODE（best / mid / worst）：用 rank_to 排序后选取单个 TO 的相似度列取 top_k。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ANNOTATE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANNOTATE_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.annotate.annotate_utils import annotate_image, load_base_image
from config.data_paths import BASE_DIR
from judge.judge import _rank_top_k
from judge.rank_to import TO_MODES, pick_to_id, rank_to_for_stem
COS_SIM_DIR = BASE_DIR / "embeddings" / "cos_sim"
NODES_DIR = BASE_DIR / "nodes"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"
ANNOTATED_ROOT = ANNOTATE_DIR / "annotated_screenshots"


def annotated_dir_for(top_k: int, mode: str) -> Path:
    return ANNOTATED_ROOT / f"top_{top_k}_{mode.lower()}"


def _load_topk_nodes(stem: str, top_k: int, mode: str) -> list[dict]:
    cos_sim_path = COS_SIM_DIR / f"{stem}.npz"
    nodes_path = NODES_DIR / f"{stem}.json"

    if not cos_sim_path.is_file():
        raise FileNotFoundError(f"缺少 cos_sim: {cos_sim_path}")
    if not nodes_path.is_file():
        raise FileNotFoundError(f"缺少 nodes: {nodes_path}")

    mode = mode.lower()
    if mode not in TO_MODES:
        raise ValueError(f"Unknown mode: {mode!r}, expected one of {TO_MODES}")

    with np.load(cos_sim_path) as data:
        sim_matrix = data["sim_matrix"]
        node_ids = data["node_ids"]
        to_ids = data["to_ids"]

    with open(nodes_path, "r", encoding="utf-8") as f:
        all_nodes = json.load(f)

    nodes_by_id = {int(n["node_id"]): n for n in all_nodes}

    ranking = rank_to_for_stem(stem)["ranking"]
    to_id = pick_to_id(ranking, mode)
    col_idx = int(np.where(to_ids == to_id)[0][0])
    final_sims = sim_matrix[:, col_idx]

    top_ids, top_scores = _rank_top_k(final_sims, node_ids, top_k)

    selected: list[dict] = []
    for nid, score in zip(top_ids, top_scores):
        node = nodes_by_id.get(int(nid))
        if node is None:
            continue
        item = dict(node)
        item["final_sim"] = score
        selected.append(item)

    selected.sort(key=lambda n: int(n["node_id"]))
    return selected


def annotate_stem(stem: str, top_k: int, mode: str, force: bool = False) -> Path:
    """为单个 stem 生成标注截图，返回输出路径。"""
    out_dir = annotated_dir_for(top_k, mode)
    out_path = out_dir / f"{stem}.png"
    if out_path.is_file() and not force:
        return out_path

    selected_nodes = _load_topk_nodes(stem, top_k, mode)
    if not selected_nodes:
        raise ValueError(f"[{stem}] 无可用 top_{top_k} 节点")

    screenshot_path = SCREENSHOTS_DIR / f"{stem}.png"
    base_img = load_base_image(screenshot_path if screenshot_path.is_file() else None)
    annotate_image(base_img, selected_nodes, out_path)
    return out_path


def annotate_stems(
    stems: list[str],
    top_k: int,
    mode: str,
    force: bool = False,
) -> list[Path]:
    """批量生成标注截图。"""
    paths: list[Path] = []
    errors: list[str] = []
    for i, stem in enumerate(stems, 1):
        try:
            paths.append(annotate_stem(stem, top_k, mode, force=force))
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{stem}: {exc}")
        if i % 100 == 0 or i == len(stems):
            print(f"  标注进度 {i}/{len(stems)}")
    if errors:
        print(f"  标注跳过/失败: {len(errors)}")
        for msg in errors[:5]:
            print(f"    {msg}")
        if len(errors) > 5:
            print(f"    ... 共 {len(errors)} 条")
    return paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="生成 cos_sim top_k 标注截图")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", type=str, default="best", choices=list(TO_MODES))
    parser.add_argument("--force", action="store_true", help="覆盖已有标注图")
    args = parser.parse_args()

    stems = sorted(p.stem for p in COS_SIM_DIR.glob("*.npz"))
    print(f"候选 stem: {len(stems)} | top_k={args.top_k} mode={args.mode}")
    annotate_stems(stems, args.top_k, args.mode, force=args.force)
    print(f"输出目录: {annotated_dir_for(args.top_k, args.mode)}")


if __name__ == "__main__":
    main()
