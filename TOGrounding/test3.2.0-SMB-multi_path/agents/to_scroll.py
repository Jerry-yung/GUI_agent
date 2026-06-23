"""TO scroll 定位：direction + top1 注入。"""
from __future__ import annotations

from typing import Any

from agents.parser import (
    _extract_label,
    _normalize_type,
    _parse_action_dict,
    extract_json_object,
)
from agents.to_click import _top1_label


def coerce_scroll_action_json(
    data: dict[str, Any] | None,
    top_k_nodes: list[dict],
    *,
    force_top1: bool,
) -> dict[str, Any] | None:
    """TO：scroll 无 element 时注入 top1 区域编号。"""
    if data is None:
        return None
    top1 = _top1_label(top_k_nodes)
    if not top1:
        return data

    action = data.get("action")
    if not isinstance(action, dict):
        action = {}
    act_type = _normalize_type(str(action.get("type", "")))
    if not act_type and action.get("direction"):
        act_type = "scroll"
    if act_type != "scroll":
        return data

    element = _extract_label(str(action.get("element", "")), kind="scroll")
    if force_top1 or not element:
        patched = dict(action)
        patched["type"] = "scroll"
        patched["element"] = top1
        return {**data, "action": patched}
    return data


def parse_to_scroll_response(
    rsp: str,
    top_k_nodes: list[dict],
    *,
    force_top1: bool = True,
) -> list[str] | None:
    data = extract_json_object(rsp)
    data = coerce_scroll_action_json(data, top_k_nodes, force_top1=force_top1)
    if data is None:
        return None
    parsed = _parse_action_dict(data)
    if parsed is None:
        return None
    if parsed.res[0] != "scroll":
        return parsed.res

    top1 = _top1_label(top_k_nodes)
    if force_top1 and top1:
        thought = parsed.res[-1] if len(parsed.res) > 3 else ""
        direction = parsed.res[2] if len(parsed.res) > 2 else ""
        return ["scroll", top1, direction, thought]
    return parsed.res


def coerce_fixed_scroll_response(rsp: str) -> dict[str, Any] | None:
    """TO 固定 scroll：允许仅输出 thought + direction（无 element、无 type）。"""
    data = extract_json_object(rsp)
    if data is None:
        return None
    action = data.get("action")
    if isinstance(action, dict) and _extract_label(str(action.get("element", "")), kind="scroll"):
        return data
    direction = None
    if isinstance(action, dict):
        direction = action.get("direction")
    if direction is None:
        direction = data.get("direction")
    if not direction:
        return data
    thought = str(data.get("thought", "")).strip()
    summary = str(data.get("summary", "")).strip()
    out: dict[str, Any] = {"thought": thought, "action": {"direction": str(direction).strip()}}
    if summary:
        out["summary"] = summary
    return out
