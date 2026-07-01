#!/usr/bin/env python3
"""
compress.py

1. 读取 AC_data/steps/{episode_id}/{step_idx}/{stem}_a11y.json
2. 压缩并筛选可交互节点（AGER V0）
3. 保存压缩结果到同目录 {stem}_compressed_a11y.json
4. 保存节点映射到同目录 {stem}_nodes.json
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from process.paths import AC_DATA, iter_stems, step_paths


# ============================================================
# JSON a11y 解析
# ============================================================
def _flatten_json_node(node: dict, result: list | None = None) -> list[dict]:
    if result is None:
        result = []
    node_copy = {k: v for k, v in node.items() if k != "children"}
    result.append(node_copy)
    children = node.get("children", [])
    if isinstance(children, list):
        for child in children:
            if isinstance(child, dict):
                _flatten_json_node(child, result)
    return result


def parse_json_a11y(json_path: Path) -> list[dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    activity = data.get("activity", {})
    root = activity.get("root", {})
    if not root:
        return []
    return _flatten_json_node(root)


# ============================================================
# AGER V0 过滤
# ============================================================
def _semantic_priority(node: dict) -> int:
    desc = node.get("content-desc")
    if desc and ((isinstance(desc, list) and desc[0]) or (isinstance(desc, str) and desc.strip())):
        return 3
    if str(node.get("text", "")).strip():
        return 2
    if str(node.get("resource-id", "")).strip():
        return 1
    return 0


def _has_semantic_content(node: dict) -> bool:
    return _semantic_priority(node) > 0


def _compute_iou(box_a: list[int], box_b: list[int]) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def _is_interactive(node: dict) -> bool:
    return bool(
        node.get("clickable")
        or node.get("focusable")
        or node.get("checkable")
        or node.get("long-clickable")
        or node.get("scrollable-vertical")
        or node.get("scrollable-horizontal")
    )


def _is_visible_and_large(node: dict) -> bool:
    if not node.get("enabled"):
        return False
    if node.get("visible-to-user") is False:
        return False
    bounds = node.get("bounds", [0, 0, 0, 0])
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]
    return w > 10 and h > 10


def _get_semantic_label(node: dict) -> str:
    desc = node.get("content-desc")
    if desc and isinstance(desc, list) and desc[0]:
        return str(desc[0]).strip()
    if desc and isinstance(desc, str) and desc.strip():
        return desc.strip()
    text = node.get("text")
    if text and str(text).strip():
        return str(text).strip()
    rid = node.get("resource-id", "")
    if rid:
        return rid.split("/")[-1].strip()
    return ""


def filter_ager_nodes(nodes: list[dict]) -> list[dict]:
    """AGER V0 过滤：候选筛选 → IoU 去重 → 稳定编号。"""
    candidates = []
    for node in nodes:
        if not _is_visible_and_large(node):
            continue
        bounds = node.get("bounds", [0, 0, 0, 0])
        bottom_y = bounds[3] if len(bounds) >= 4 else 0
        if bottom_y <= 150 and not _is_interactive(node):
            continue
        if not (_is_interactive(node) or _has_semantic_content(node)):
            continue

        item = copy.deepcopy(node)
        item["_label"] = _get_semantic_label(node)
        item["_priority"] = _semantic_priority(node)
        candidates.append(item)

    candidates.sort(key=lambda n: n["_priority"], reverse=True)

    kept = []
    for node in candidates:
        box = node["bounds"]
        if any(_compute_iou(box, existing["bounds"]) > 0.9 for existing in kept):
            continue
        kept.append(node)

    def sort_key(node: dict) -> tuple[float, float]:
        box = node["bounds"]
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        return (cy, cx)

    kept.sort(key=sort_key)

    output = []
    for idx, node in enumerate(kept):
        out = {k: v for k, v in node.items() if not k.startswith("_")}
        out["ager_id"] = idx
        out["ager_label"] = node.get("_label", "")
        output.append(out)

    return output


def filter_interactive_nodes(nodes: list[dict]) -> list[dict]:
    return [n for n in nodes if _is_interactive(n)]


def _node_text_for_emb(node: dict) -> str:
    parts = []
    desc = node.get("content-desc")
    if isinstance(desc, list):
        for item in desc:
            if item:
                parts.append(str(item).strip())
                break
    elif isinstance(desc, str) and desc.strip():
        parts.append(desc.strip())
    text = (node.get("text") or "").strip()
    if text:
        parts.append(text)
    return " ".join(parts).strip()


def save_node_mappings(stem: str, nodes: list[dict]) -> None:
    mappings = []
    for i, node in enumerate(nodes):
        mappings.append({
            "node_id": i,
            "bounds": node.get("bounds", [0, 0, 0, 0]),
            "text_for_emb": _node_text_for_emb(node),
        })
    paths = step_paths(stem)
    paths["dir"].mkdir(parents=True, exist_ok=True)
    with open(paths["nodes"], "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)


def compress_stem(stem: str) -> str:
    """返回: ok | empty_nodes | skip"""
    paths = step_paths(stem)
    a11y_path = paths["a11y"]
    if not a11y_path.is_file():
        print(f"  [{stem}] 缺少 a11y，跳过")
        return "skip"

    flat_nodes = parse_json_a11y(a11y_path)
    compressed = filter_ager_nodes(flat_nodes)
    interactive_nodes = filter_interactive_nodes(compressed)

    paths["dir"].mkdir(parents=True, exist_ok=True)
    with open(paths["compressed_a11y"], "w", encoding="utf-8") as f:
        json.dump(compressed, f, ensure_ascii=False, indent=2)

    save_node_mappings(stem, interactive_nodes)
    if len(interactive_nodes) == 0:
        print(
            f"  [{stem}] 原始={len(flat_nodes)} → 压缩={len(compressed)} → 可交互=0"
        )
        return "empty_nodes"

    print(
        f"  [{stem}] 原始={len(flat_nodes)} → 压缩={len(compressed)} → 可交互={len(interactive_nodes)}"
    )
    return "ok"


def main() -> None:
    all_stems = iter_stems()

    print("=" * 60)
    print("compress.py — 压缩 a11y → steps/{episode}/{step}/{stem}_*")
    print("=" * 60)
    print(f"steps 样本数: {len(all_stems)}")
    if all_stems:
        print(f"  示例: {all_stems[0]} ... {all_stems[-1]}")
    print("=" * 60)

    if not all_stems:
        print("没有样本，退出。")
        sys.exit(1)

    ok = empty_nodes = skipped = 0
    for stem in all_stems:
        result = compress_stem(stem)
        if result == "ok":
            ok += 1
        elif result == "empty_nodes":
            empty_nodes += 1
        else:
            skipped += 1

    print(f"\n{'=' * 60}")
    print(f"处理完成: 成功={ok}, 可交互=0={empty_nodes}, 跳过={skipped}")
    print(f"  输出目录: {AC_DATA / 'steps'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
