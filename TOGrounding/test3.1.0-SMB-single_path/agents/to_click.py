"""TO click 定位：element 或 top1 注入。"""
from __future__ import annotations

from typing import Any

from agents.parser import (
    _extract_label,
    _normalize_type,
    _parse_action_dict,
    extract_json_object,
)


def _top1_label(top_k_nodes: list[dict]) -> str | None:
    if not top_k_nodes:
        return None
    label = str(top_k_nodes[0].get("label", "")).strip().lower()
    return label or None


def coerce_click_action_json(
    data: dict[str, Any] | None,
    top_k_nodes: list[dict],
    *,
    force_top1: bool,
) -> dict[str, Any] | None:
    """TO：click 一律补/改为 top1。"""
    if data is None:
        return None
    top1 = _top1_label(top_k_nodes)
    if not top1:
        return data

    action = data.get("action")
    if not isinstance(action, dict):
        return data
    if _normalize_type(str(action.get("type", ""))) not in ("", "click"):
        return data

    element = _extract_label(str(action.get("element", "")), kind="click")
    if force_top1 or not element:
        patched = dict(action)
        patched["element"] = top1
        return {**data, "action": patched}
    return data


def parse_to_response(
    rsp: str,
    top_k_nodes: list[dict],
    *,
    force_top1: bool,
) -> list[str] | None:
    data = extract_json_object(rsp)
    data = coerce_click_action_json(data, top_k_nodes, force_top1=force_top1)
    if data is None:
        return None
    parsed = _parse_action_dict(data)
    if parsed is None or parsed[0] != "click":
        return parsed

    top1 = _top1_label(top_k_nodes)
    if force_top1 and top1:
        thought = parsed[-1] if len(parsed) > 2 else ""
        return ["click", top1, thought]
    return parsed
