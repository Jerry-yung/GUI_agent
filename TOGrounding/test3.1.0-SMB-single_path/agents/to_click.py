"""TO / TOa click 定位：element、top1、归一化坐标。"""
from __future__ import annotations

from typing import Any

from agents.parser import (
    _extract_label,
    _extract_norm_coords,
    _normalize_type,
    _parse_action_dict,
    extract_json_object,
)
from utils.click_res import click_xy_from_res, format_click_xy_res, is_click_xy_res


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
    """TO：click 一律补/改为 top1；TOa：有 coords/element 时不改。"""
    if data is None:
        return None
    top1 = _top1_label(top_k_nodes)
    if not top1:
        return data

    action = data.get("action")
    if not isinstance(action, dict):
        return data
    if _normalize_type(str(action.get("type", ""))) != "click":
        return data

    if _extract_norm_coords(action) is not None:
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
    if isinstance(data, dict):
        action = data.get("action")
        if isinstance(action, dict) and _normalize_type(str(action.get("type", ""))) == "click":
            coords = _extract_norm_coords(action)
            if coords is not None and not force_top1:
                summary = str(
                    data.get("step_summary")
                    or data.get("summary")
                    or data.get("thought")
                    or ""
                )
                return format_click_xy_res(coords[0], coords[1], summary)

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


def parse_toa_response(
    rsp: str,
    top_k_nodes: list[dict],
) -> tuple[list[str] | None, str | None]:
    """TOa：返回 (res, locator_source)。source: coords | element | top1。"""
    data = extract_json_object(rsp)
    if isinstance(data, dict):
        action = data.get("action")
        if isinstance(action, dict) and _normalize_type(str(action.get("type", ""))) == "click":
            coords = _extract_norm_coords(action)
            if coords is not None:
                summary = str(
                    data.get("step_summary")
                    or data.get("summary")
                    or data.get("thought")
                    or ""
                )
                return format_click_xy_res(coords[0], coords[1], summary), "coords"

    res = parse_to_response(rsp, top_k_nodes, force_top1=False)
    if res is None:
        return None, None
    if res[0] != "click":
        return res, None
    source = click_locator_source(rsp, top_k_nodes, res)
    if source == "top1" and top_k_nodes:
        label = str(top_k_nodes[0].get("label", "")).lower()
        thought = res[-1] if len(res) > 2 else ""
        return ["click", label, thought], "top1"
    return res, source


def click_locator_source(
    rsp: str,
    top_k_nodes: list[dict],
    parsed: list[str] | None,
) -> str | None:
    """TOa：click 定位来源 coords / element / top1。"""
    if not parsed or parsed[0] != "click":
        return None
    if is_click_xy_res(parsed):
        return "coords"
    data = extract_json_object(rsp)
    if not isinstance(data, dict):
        return "top1" if top_k_nodes else None
    action = data.get("action")
    if not isinstance(action, dict):
        return "top1" if top_k_nodes else None
    if _extract_norm_coords(action) is not None:
        return "coords"
    element = _extract_label(str(action.get("element", "")), kind="click")
    if element:
        return "element"
    return "top1" if top_k_nodes else None


def toa_decide_meta(
    rsp: str,
    res: list[str] | None,
    locator_source: str | None,
) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if locator_source:
        meta["locator_source"] = locator_source
    coords = click_xy_from_res(res) if res else None
    if coords is not None:
        meta["norm_x"] = coords[0]
        meta["norm_y"] = coords[1]
    return meta
