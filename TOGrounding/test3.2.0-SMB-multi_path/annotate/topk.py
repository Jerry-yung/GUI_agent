"""Semantic TopK over SMAN-aligned candidate nodes."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from annotate.annotate import annotate_nodes, default_label_dims, load_base_image
from annotate.bounds import TARGET_SCREEN_SIZE, parse_bounds_from_action, scale_bounds
from annotate.embedder import build_node_embeddings, embed_instruction
from annotate.node_dedupe import dedupe_nodes_by_bounds
from annotate.node_filter import filter_nodes_for_annotation
from annotate.similarity import cosine_similarity, label_ranks_by_score, topk_indices
from utils.paths import (
    cache_labeled_file,
    cache_nodes_file,
    cache_retrieval_file,
    nodes_emb_step_dir,
)
from utils.sman_bridge import render_scroll_only_labeled_image


def _label_from_action(action_str: str, area_idx: int, kind: str) -> str:
    del action_str, kind
    return str(area_idx)


def _text_from_click_action(action_str: str) -> str:
    m = re.match(r"click\((.+),\s*\[", action_str)
    if m:
        return m.group(1).strip()
    inner = action_str.split("(", 1)[-1].rsplit(")", 1)[0]
    return inner[:80]


def build_sman_click_nodes(click_actions: list[str]) -> list[dict]:
    nodes: list[dict] = []
    for i, act in enumerate(click_actions):
        bounds = parse_bounds_from_action(act)
        if not bounds:
            continue
        left, top, right, bottom = bounds
        label_text = _text_from_click_action(act)
        area_idx = i + 1
        nodes.append(
            {
                "label": _label_from_action(act, area_idx, "clickable"),
                "kind": "clickable",
                "sman_area_idx": area_idx,
                "sman_type": "click",
                "bounds": [left, top, right, bottom],
                "text": label_text,
                "ager_label": label_text,
                "content-desc": [label_text] if label_text else [None],
            }
        )
    return nodes


def build_sman_scroll_nodes(scroll_bounds: list[str]) -> list[dict]:
    nodes: list[dict] = []
    for i, bounds_str in enumerate(scroll_bounds):
        m = re.search(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_str)
        if not m:
            continue
        x1, y1, x2, y2 = map(int, m.groups())
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        area_idx = i + 1
        nodes.append(
            {
                "label": str(area_idx),
                "kind": "scroll",
                "sman_area_idx": area_idx,
                "sman_type": "scroll",
                "bounds": [left, top, right, bottom],
                "text": f"scroll_region_{area_idx}",
                "ager_label": f"scroll {area_idx}",
                "content-desc": [None],
            }
        )
    return nodes


def _deduped_scroll_nodes(scroll_bounds: list[str]) -> list[dict]:
    scroll_nodes = build_sman_scroll_nodes(scroll_bounds)
    return [n for n in dedupe_nodes_by_bounds(scroll_nodes) if n.get("kind") == "scroll"]


def _write_nodes_json(
    nodes_json: Path,
    selected: list[dict],
    orig_size: tuple[int, int],
) -> list[dict]:
    display_nodes = []
    for n in selected:
        display_nodes.append(
            {
                **n,
                "bounds": scale_bounds(n["bounds"], orig_size, TARGET_SCREEN_SIZE),
            }
        )

    payload = []
    for n in display_nodes:
        payload.append(
            {
                "id": n["sman_area_idx"],
                "label": n["label"],
                "sman_area_idx": n["sman_area_idx"],
                "sman_type": n["sman_type"],
                "bounds": n["bounds"],
                "type": n["kind"],
                "text": n.get("text", ""),
            }
        )
    nodes_json.parent.mkdir(parents=True, exist_ok=True)
    with open(nodes_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return display_nodes


def _pool_label_score(
    pool: list[dict],
    scores: np.ndarray,
    label: str,
) -> float | None:
    target = label.lower()
    for i, node in enumerate(pool):
        if str(node.get("label", "")).lower() == target:
            return float(scores[i])
    return None


def _write_retrieval_cache(
    cache_path: Path,
    *,
    target_object: str,
    top_k: int,
    action_type_hint: str | None,
    pool: list[dict],
    scores: np.ndarray,
    topk_idx: list[int],
    rank_by_label: dict[str, int],
    gt_label: str | None = None,
) -> None:
    topk_items: list[dict[str, Any]] = []
    for pool_idx in topk_idx:
        label = str(pool[pool_idx].get("label", "")).lower()
        if not label:
            continue
        topk_items.append(
            {
                "label": label,
                "score": round(float(scores[pool_idx]), 6),
                "rank": rank_by_label.get(label),
            }
        )

    payload: dict[str, Any] = {
        "target_object": target_object,
        "top_k": top_k,
        "action_type_hint": action_type_hint,
        "pool_size": len(pool),
        "topk": topk_items,
    }

    if gt_label:
        gl = gt_label.lower()
        gt_score = _pool_label_score(pool, scores, gl)
        payload["gt"] = {
            "label": gl,
            "score": round(gt_score, 6) if gt_score is not None else None,
            "rank": rank_by_label.get(gl),
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _topk_from_pool(
    pool: list[dict],
    *,
    task_name: str,
    page_name: str,
    screenshot: Image.Image,
    query_text: str,
    top_k: int,
    fresh_instruction_embed: bool,
    action_type_hint: str | None = None,
    gt_label: str | None = None,
) -> tuple[list[dict], dict[str, int]]:
    if not pool:
        raise ValueError(f"{page_name}: empty candidate pool")

    embed_dir = nodes_emb_step_dir(task_name, page_name)
    embed_dir.mkdir(parents=True, exist_ok=True)

    node_vecs = build_node_embeddings(pool, embed_dir, screenshot=screenshot)

    scores = None
    if query_text.strip():
        try:
            qvec = embed_instruction(
                query_text, embed_dir, reuse_cache=not fresh_instruction_embed
            )
            scores = cosine_similarity(qvec, node_vecs)
        except Exception:
            scores = None

    rank_by_label: dict[str, int] = {}
    if scores is not None and scores.size > 0:
        labels = [str(n.get("label", "")) for n in pool]
        rank_by_label = label_ranks_by_score(scores, labels)
        idx = topk_indices(scores, top_k).tolist()
        if query_text.strip():
            cache_path = cache_retrieval_file(
                task_name, page_name, top_k=top_k, query_text=query_text
            )
            _write_retrieval_cache(
                cache_path,
                target_object=query_text,
                top_k=top_k,
                action_type_hint=action_type_hint,
                pool=pool,
                scores=scores,
                topk_idx=idx,
                rank_by_label=rank_by_label,
                gt_label=gt_label,
            )
    else:
        idx = list(range(min(top_k, len(pool))))

    selected = [pool[i] for i in idx]
    selected.sort(key=lambda n: (n.get("kind", ""), n.get("sman_area_idx", 0)))

    lw, lh = default_label_dims()
    return filter_nodes_for_annotation(selected, label_w=lw, label_h=lh), rank_by_label


def _save_labeled_image(
    screenshot_path: str,
    selected: list[dict],
    labeled_png: Path,
) -> None:
    base = load_base_image(screenshot_path)
    labeled_png.parent.mkdir(parents=True, exist_ok=True)
    annotate_nodes(base, selected, labeled_png, resize_to=TARGET_SCREEN_SIZE, apply_node_filter=False)


def _save_raw_screenshot(screenshot_path: str, labeled_png: Path) -> None:
    base = load_base_image(screenshot_path)
    if base.size != TARGET_SCREEN_SIZE:
        base = base.resize(TARGET_SCREEN_SIZE, Image.Resampling.LANCZOS)
    labeled_png.parent.mkdir(parents=True, exist_ok=True)
    base.save(labeled_png)


def run_topk_pipeline(
    task_name: str,
    page_name: str,
    click_actions: list[str],
    scroll_bounds: list[str],
    screenshot_path: str,
    query_text: str,
    top_k: int,
    *,
    action_type_hint: str = "click",
    fresh_instruction_embed: bool = False,
    gt_label: str | None = None,
) -> tuple[Path, Path, list[dict], str, dict[str, int]]:
    effective_hint = action_type_hint
    rank_by_label: dict[str, int] = {}
    screenshot = Image.open(screenshot_path).convert("RGB")
    orig_size = screenshot.size

    labeled_png = cache_labeled_file(task_name, page_name, top_k=top_k)
    nodes_json = cache_nodes_file(task_name, page_name, top_k=top_k)

    click_nodes = build_sman_click_nodes(click_actions)
    scroll_nodes_deduped = _deduped_scroll_nodes(scroll_bounds)

    pool_hint = action_type_hint
    if pool_hint == "long_press":
        pool_hint = "click"

    if pool_hint == "scroll":
        if not scroll_nodes_deduped:
            raise ValueError(f"{page_name}: no scroll candidate nodes")
        selected, rank_by_label = _topk_from_pool(
            scroll_nodes_deduped,
            task_name=task_name,
            page_name=page_name,
            screenshot=screenshot,
            query_text=query_text,
            top_k=top_k,
            fresh_instruction_embed=fresh_instruction_embed,
            action_type_hint=action_type_hint,
            gt_label=gt_label,
        )
        display_nodes = _write_nodes_json(nodes_json, selected, orig_size)
        _save_labeled_image(screenshot_path, selected, labeled_png)
        return labeled_png, nodes_json, display_nodes, effective_hint, rank_by_label

    if pool_hint in ("input", "back"):
        _save_raw_screenshot(screenshot_path, labeled_png)
        nodes_json.parent.mkdir(parents=True, exist_ok=True)
        with open(nodes_json, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
        return labeled_png, nodes_json, [], effective_hint, rank_by_label

    if pool_hint == "click":
        pool = click_nodes
        if not pool:
            raise ValueError(f"{page_name}: no click candidate nodes")
        selected, rank_by_label = _topk_from_pool(
            pool,
            task_name=task_name,
            page_name=page_name,
            screenshot=screenshot,
            query_text=query_text,
            top_k=top_k,
            fresh_instruction_embed=fresh_instruction_embed,
            action_type_hint=action_type_hint,
            gt_label=gt_label,
        )
        display_nodes = _write_nodes_json(nodes_json, selected, orig_size)
        _save_labeled_image(screenshot_path, selected, labeled_png)
        return labeled_png, nodes_json, display_nodes, effective_hint, rank_by_label

    raise ValueError(f"{page_name}: unknown action_type_hint {action_type_hint!r}")
