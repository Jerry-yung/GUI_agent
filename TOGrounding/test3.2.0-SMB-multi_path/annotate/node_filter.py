"""Pre-render geometric filtering for SoM annotation (P4)."""
from __future__ import annotations

NESTED_SCROLL_AREA_RATIO = 10.0


def _bounds_key(bounds: list[int] | tuple[int, ...]) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds[:4]
    return left, top, right, bottom


def _box_area(bounds: list[int] | tuple[int, ...]) -> int:
    left, top, right, bottom = _bounds_key(bounds)
    return max(0, right - left) * max(0, bottom - top)


def _contains(outer: list[int] | tuple[int, ...], inner: list[int] | tuple[int, ...]) -> bool:
    ox1, oy1, ox2, oy2 = _bounds_key(outer)
    ix1, iy1, ix2, iy2 = _bounds_key(inner)
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2


def _suppress_nested_scrolls(nodes: list[dict]) -> list[dict]:
    scrolls = [n for n in nodes if n.get("kind") == "scroll"]
    if len(scrolls) < 2:
        return nodes

    drop_ids: set[int] = set()
    for i, outer in enumerate(scrolls):
        if id(outer) in drop_ids:
            continue
        for j, inner in enumerate(scrolls):
            if i == j or id(inner) in drop_ids:
                continue
            area_outer = _box_area(outer["bounds"])
            area_inner = _box_area(inner["bounds"])
            if area_outer <= 0 or area_inner <= 0:
                continue
            smaller, larger = (
                (inner, outer) if area_inner < area_outer else (outer, inner)
            )
            if not _contains(larger["bounds"], smaller["bounds"]):
                continue
            ratio = _box_area(larger["bounds"]) / max(1, _box_area(smaller["bounds"]))
            if ratio >= NESTED_SCROLL_AREA_RATIO:
                drop_ids.add(id(larger))

    if not drop_ids:
        return nodes
    return [n for n in nodes if id(n) not in drop_ids]


def filter_nodes_for_annotation(
    nodes: list[dict],
    *,
    label_w: int = 0,
    label_h: int = 0,
) -> list[dict]:
    """Apply P4 filters before drawing; may set ``force_outside_label`` on tiny clicks."""
    if not nodes:
        return nodes

    working = [{**n} for n in nodes]
    filtered = _suppress_nested_scrolls(working)

    if label_w > 0 and label_h > 0:
        margin = 6
        for node in filtered:
            if node.get("kind") != "clickable":
                continue
            x1, y1, x2, y2 = node["bounds"]
            bw, bh = x2 - x1, y2 - y1
            if bw < label_w + margin or bh < label_h + margin:
                node["force_outside_label"] = True

    return filtered
