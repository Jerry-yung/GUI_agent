"""Match Mobile3M gt_action / gt_bounds to compress pipeline node_id."""
from __future__ import annotations

import re
from typing import Any


def parse_action_attrs(gt_action: str) -> dict[str, str]:
    inner = gt_action
    if gt_action.startswith("click(") and gt_action.endswith(")"):
        inner = gt_action[len("click(") : -1].strip()

    attrs: dict[str, str] = {}
    for key in ("id", "description", "class", "package"):
        m = re.search(rf'{key}="([^"]*)"', inner)
        if m:
            attrs[key] = m.group(1).strip()

    text_m = re.search(r">\s*([^<>]+?)\s*</", inner)
    if text_m:
        attrs["text"] = text_m.group(1).strip()

    return attrs


def bbox_area(bounds: list) -> float:
    if len(bounds) != 4:
        return float("inf")
    x1, y1, x2, y2 = bounds
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_center(bounds: list) -> tuple[float, float]:
    x1, y1, x2, y2 = bounds
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_bounds(bounds: list, x: float, y: float) -> bool:
    if len(bounds) != 4:
        return False
    x1, y1, x2, y2 = bounds
    return x1 <= x <= x2 and y1 <= y <= y2


def compute_iou(box_a: list, box_b: list) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    union = bbox_area(box_a) + bbox_area(box_b) - inter
    return inter / union if union > 0 else 0.0


def _xml_node_matches(node: dict, attrs: dict[str, str]) -> bool:
    rid = attrs.get("id", "")
    if rid and node.get("resource-id", "") != rid:
        return False

    if "text" in attrs:
        if str(node.get("text", "")).strip() != attrs["text"]:
            return False

    if "description" in attrs:
        desc = node.get("content-desc")
        node_desc = ""
        if isinstance(desc, list) and desc:
            node_desc = str(desc[0] or "").strip()
        elif isinstance(desc, str):
            node_desc = desc.strip()
        if node_desc != attrs["description"]:
            return False

    if "class" in attrs and node.get("class", "") != attrs["class"]:
        return False
    if "package" in attrs and node.get("package", "") != attrs["package"]:
        return False

    return bool(rid or "text" in attrs or "description" in attrs)


def _compress_node_matches(node: dict, attrs: dict[str, str]) -> bool:
    rid = attrs.get("id", "")
    if rid and node.get("resource-id", "") != rid:
        return False

    if "text" in attrs:
        t = attrs["text"]
        if node.get("text", "") != t and node.get("text_for_emb", "") != t:
            return False

    if "description" in attrs:
        if node.get("text_for_emb", "") != attrs["description"]:
            return False

    if "class" in attrs and node.get("class", "") != attrs.get("class", ""):
        return False

    return bool(rid or "text" in attrs or "description" in attrs)


def _pick_smallest(nodes: list[dict]) -> dict | None:
    valid = [n for n in nodes if len(n.get("bounds") or []) == 4]
    if not valid:
        return None
    return min(valid, key=lambda n: bbox_area(n["bounds"]))


def _is_interactive(node: dict) -> bool:
    """Align with process/compress.py interactive node definition."""
    return bool(
        node.get("clickable")
        or node.get("focusable")
        or node.get("checkable")
        or node.get("long-clickable")
        or node.get("scrollable-vertical")
        or node.get("scrollable-horizontal")
    )


def _nodes_matching_attrs(
    xml_nodes: list[dict],
    attrs: dict[str, str],
    *,
    require_clickable: bool = False,
) -> list[dict]:
    out: list[dict] = []
    for node in xml_nodes:
        if require_clickable and not node.get("clickable"):
            continue
        if _xml_node_matches(node, attrs):
            out.append(node)
    return out


def xml_bounds_from_action_with_tier(
    xml_nodes: list[dict],
    gt_action: str,
) -> tuple[list[int], str] | None:
    """
    Match gt_action on raw XML nodes; return (bounds, tier).

    Tiers (first hit wins):
      1. clickable=True + attribute match → smallest bounds
      2. attribute match (ignore clickable) → smallest label bounds
      3. smallest interactive node containing label center
      4. label bounds from tier-2 (non-clickable leaf, e.g. Tab TextView)
    """
    if not gt_action.startswith("click("):
        return None

    attrs = parse_action_attrs(gt_action)
    if not attrs.get("id") and not attrs.get("text") and not attrs.get("description"):
        return None

    clickable_matches = _nodes_matching_attrs(xml_nodes, attrs, require_clickable=True)
    picked = _pick_smallest(clickable_matches)
    if picked is not None:
        return list(picked["bounds"]), "clickable_attr"

    label_matches = _nodes_matching_attrs(xml_nodes, attrs)
    label = _pick_smallest(label_matches)
    if label is None:
        return None

    label_bounds = label["bounds"]
    cx, cy = bbox_center(label_bounds)
    interactive = [
        n
        for n in xml_nodes
        if _is_interactive(n) and point_in_bounds(n.get("bounds", []), cx, cy)
    ]
    container = _pick_smallest(interactive)
    if container is not None:
        return list(container["bounds"]), "interactive_container"

    return list(label_bounds), "label_bounds"


def xml_bounds_from_action(xml_nodes: list[dict], gt_action: str) -> list[int] | None:
    """Match gt_action on raw XML nodes; return bounds or None."""
    result = xml_bounds_from_action_with_tier(xml_nodes, gt_action)
    if result is None:
        return None
    return result[0]


def match_gt_node_in_nodes(
    nodes: list[dict],
    gt_action: str,
    gt_bounds: list | None = None,
) -> dict | None:
    """
    Map gt_action (+ optional gt_bounds from XML) to one compress node.
    Priority: resource-id/text match → IoU with gt_bounds → smallest containing gt center.
    """
    attrs = parse_action_attrs(gt_action) if gt_action.startswith("click(") else {}

    if attrs:
        by_attr = [n for n in nodes if _compress_node_matches(n, attrs)]
        picked = _pick_smallest(by_attr)
        if picked is not None:
            return picked

    if gt_bounds and len(gt_bounds) == 4:
        scored = [(n, compute_iou(n.get("bounds", []), gt_bounds)) for n in nodes]
        scored = [(n, s) for n, s in scored if s >= 0.5]
        if scored:
            return max(scored, key=lambda t: t[1])[0]

        cx, cy = bbox_center(gt_bounds)
        containing = [
            n for n in nodes if point_in_bounds(n.get("bounds", []), cx, cy)
        ]
        return _pick_smallest(containing)

    return None


def gt_action_from_step_gt(gt_data: dict) -> str:
    if gt_data.get("gt_action"):
        return str(gt_data["gt_action"])
    meta = gt_data.get("meta") or {}
    return str(meta.get("gt_action") or "")


def gt_node_ids_from_step_gt(gt_data: dict) -> list[int]:
    if gt_data.get("gt_node_id") is not None:
        return [int(gt_data["gt_node_id"])]
    return [int(item["node_id"]) for item in gt_data.get("nearest_5", [])]
