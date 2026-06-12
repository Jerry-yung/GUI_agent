#!/usr/bin/env python3
"""
compress.py

1. 读取 AC_data/a11y_trees_L0/*.json
2. 压缩并筛选可交互节点（AGER V0）
3. 保存压缩结果到 AC_data/compressed_a11y/*.json
4. 为每个压缩后的节点分配 node_id，保存节点映射到 AC_data/nodes/*.json
   格式：[{"node_id": 0, "bounds": [x1,y1,x2,y2]}, ...]
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from process.stem_deleted import move_stem_to_deleted, purge_stems_with_empty_nodes

BASE_DIR = PROJECT_ROOT / "AC_data"
A11Y_DIR = BASE_DIR / "a11y_trees_L0"
COMPRESSED_DIR = BASE_DIR / "compressed_a11y"
NODES_DIR = BASE_DIR / "nodes"


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
        # 可见性 + 尺寸
        if not _is_visible_and_large(node):
            continue
        # 过滤状态栏：底部 y ≤ 150px 且非交互
        bounds = node.get("bounds", [0, 0, 0, 0])
        bottom_y = bounds[3] if len(bounds) >= 4 else 0
        if bottom_y <= 150 and not _is_interactive(node):
            continue
        # 交互 或 语义
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
    """从压缩节点中仅保留可交互节点。"""
    return [n for n in nodes if _is_interactive(n)]


# ============================================================
# 压缩 + 节点映射保存
# ============================================================
def compress_stem(stem: str) -> str:
    """
    返回: "ok" | "deleted" | "skip"
    """
    a11y_path = A11Y_DIR / f"{stem}.json"
    if not a11y_path.is_file():
        print(f"  [{stem}] 缺少 a11y_tree，跳过")
        return "skip"

    flat_nodes = parse_json_a11y(a11y_path)
    compressed = filter_ager_nodes(flat_nodes)
    interactive_nodes = filter_interactive_nodes(compressed)

    if len(interactive_nodes) == 0:
        COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)
        with open(COMPRESSED_DIR / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(compressed, f, ensure_ascii=False, indent=2)
        moved = move_stem_to_deleted(stem)
        print(
            f"  [{stem}] 原始={len(flat_nodes)} → 压缩={len(compressed)} → 可交互=0，"
            f"已移入 _deleted（{len(moved)} 个四件套文件）"
        )
        return "deleted"

    COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)
    with open(COMPRESSED_DIR / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(compressed, f, ensure_ascii=False, indent=2)

    save_node_mappings(stem, interactive_nodes)

    print(
        f"  [{stem}] 原始={len(flat_nodes)} → 压缩={len(compressed)} → 可交互={len(interactive_nodes)}"
    )
    return "ok"


def _node_text_for_emb(node: dict) -> str:
    """提取节点文本（content-desc + text），供 embedding 使用。"""
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
    """保存节点映射，node_id 即节点在可交互列表中的索引。"""
    mappings = []
    for i, node in enumerate(nodes):
        mappings.append({
            "node_id": i,
            "bounds": node.get("bounds", [0, 0, 0, 0]),
            "text_for_emb": _node_text_for_emb(node),
        })
    NODES_DIR.mkdir(parents=True, exist_ok=True)
    with open(NODES_DIR / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)


# ============================================================
# Main
# ============================================================
def main() -> None:
    all_stems = sorted(p.stem for p in A11Y_DIR.glob("*.json"))

    print("=" * 60)
    print("compress.py — 压缩 a11y_tree → 可交互节点 → 节点映射")
    print("=" * 60)
    print(f"a11y_trees 总数: {len(all_stems)}")
    if all_stems:
        print(f"  示例: {all_stems[0]} ... {all_stems[-1]}")
    print("=" * 60)

    if not all_stems:
        print("没有样本，退出。")
        sys.exit(1)

    existing_empty = purge_stems_with_empty_nodes()
    if existing_empty:
        print(f"已清理历史空 nodes 样本: {len(existing_empty)} 个")
        if len(existing_empty) <= 5:
            print(f"  {existing_empty}")
        else:
            print(f"  示例: {existing_empty[:3]} ...")
        print("-" * 60)

    ok = deleted = skipped = 0
    for stem in all_stems:
        result = compress_stem(stem)
        if result == "ok":
            ok += 1
        elif result == "deleted":
            deleted += 1
        else:
            skipped += 1

    print(f"\n{'=' * 60}")
    print(f"处理完成: 成功={ok}, 移入_deleted={deleted}, 跳过={skipped}")
    print(f"  compressed_a11y: {COMPRESSED_DIR}")
    print(f"  nodes mapping:   {NODES_DIR}")
    print(f"  _deleted:        {BASE_DIR / '_deleted'}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
