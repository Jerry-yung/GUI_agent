"""Deduplicate SMAN candidate nodes that share the same bounds."""
from __future__ import annotations


def _bounds_key(bounds: list[int] | tuple[int, ...]) -> tuple[int, int, int, int]:
    left, top, right, bottom = bounds[:4]
    return left, top, right, bottom


def dedupe_nodes_by_bounds(nodes: list[dict]) -> list[dict]:
    """Keep the first node per (kind, bounds); preserve original cN/sN / sman_area_idx."""
    kept: list[dict] = []
    seen: set[tuple[str, tuple[int, int, int, int]]] = set()
    for node in nodes:
        kind = str(node.get("kind") or "clickable")
        bounds = node.get("bounds") or []
        if len(bounds) < 4:
            continue
        key = (kind, _bounds_key(bounds))
        if key in seen:
            continue
        seen.add(key)
        kept.append(node)
    return kept
