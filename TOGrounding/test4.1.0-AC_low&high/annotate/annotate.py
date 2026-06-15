#!/usr/bin/env python3
"""
根据 top_k 节点在截图上绘制标注框，输出到：
    annotate/annotated_screenshots/top_{top_k}/{episode_id}/{step_idx:03d}.png
"""

from __future__ import annotations

import sys
from pathlib import Path

ANNOTATE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANNOTATE_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from annotate.annotate_utils import annotate_image, annotate_suggestion_image, load_base_image
from process.paths import parse_stem, step_paths

ANNOTATED_ROOT = ANNOTATE_DIR / "annotated_screenshots"


def annotated_path_for(stem: str, top_k: int) -> Path:
    episode_id, step_idx = parse_stem(stem)
    return ANNOTATED_ROOT / f"top_{top_k}" / episode_id / f"{step_idx:03d}.png"


def toa_annotated_path_for(stem: str, top_k: int) -> Path:
    episode_id, step_idx = parse_stem(stem)
    return ANNOTATED_ROOT / f"toa_top_{top_k}" / episode_id / f"{step_idx:03d}.png"


def save_original_annotated(stem: str, top_k: int, force: bool = False) -> Path:
    """nodes 为空时，将原截图复制到 annotated_screenshots 目录。"""
    out_path = annotated_path_for(stem, top_k)
    if out_path.is_file() and not force:
        return out_path

    paths = step_paths(stem)
    screenshot_path = paths["screenshot"]
    base_img = load_base_image(screenshot_path if screenshot_path.is_file() else None)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base_img.save(out_path, "PNG")
    return out_path


def annotate_step(
    stem: str,
    top_k: int,
    selected_nodes: list[dict],
    force: bool = False,
) -> Path:
    """为单个 stem 生成标注截图，返回输出路径。"""
    if not selected_nodes:
        return save_original_annotated(stem, top_k, force=force)

    out_path = annotated_path_for(stem, top_k)
    if out_path.is_file() and not force:
        return out_path

    paths = step_paths(stem)
    screenshot_path = paths["screenshot"]
    base_img = load_base_image(screenshot_path if screenshot_path.is_file() else None)
    annotate_image(base_img, selected_nodes, out_path)
    return out_path


def annotate_toa_step(
    stem: str,
    top_k: int,
    selected_nodes: list[dict],
    force: bool = False,
) -> Path:
    """TOa：无 #node_id 的建议框，输出到 toa_top_{k}/。"""
    if not selected_nodes:
        return save_original_annotated(stem, top_k, force=force)

    out_path = toa_annotated_path_for(stem, top_k)
    if out_path.is_file() and not force:
        return out_path

    paths = step_paths(stem)
    screenshot_path = paths["screenshot"]
    base_img = load_base_image(screenshot_path if screenshot_path.is_file() else None)
    annotate_suggestion_image(
        base_img,
        selected_nodes,
        out_path,
        draw_labels=False,
    )
    return out_path
