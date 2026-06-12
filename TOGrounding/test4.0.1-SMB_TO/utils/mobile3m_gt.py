"""Mobile3M ground-truth helpers: page paths, instructions, XML bounds."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from utils.gt_node_match import bbox_center, xml_bounds_from_action_with_tier
from utils.xml_to_ager import parse_mobile3m_xml


def build_page_path(task_name: str) -> list[str]:
    """QQmusic0_34_347 -> [QQmusic0, QQmusic0_34, QQmusic0_34_347]."""
    parts = task_name.split("_")
    if not parts:
        return []
    pages = [parts[0]]
    for seg in parts[1:]:
        pages.append(f"{pages[-1]}_{seg}")
    return pages


def parse_step_instructions(task_text: str) -> tuple[str, list[str]]:
    lines = [ln.strip() for ln in task_text.splitlines() if ln.strip()]
    concise = lines[-1] if lines else ""
    steps = re.findall(r"^\d+\.\s(.*)", task_text, re.MULTILINE)
    return concise, steps


def gt_action_for_step(child_page_json: dict[str, Any]) -> str:
    history = child_page_json.get("history_actions") or []
    if not history:
        return ""
    return str(history[-1]).strip()


def click_bounds_from_action(
    xml_path: Path | str,
    gt_action: str,
) -> tuple[list[int], str] | None:
    """
    Match gt_action on parent-page XML.

    Returns (bounds, tier) where tier is one of:
      clickable_attr | interactive_container | label_bounds
    """
    if not gt_action.startswith("click("):
        return None
    nodes = parse_mobile3m_xml(xml_path)
    return xml_bounds_from_action_with_tier(nodes, gt_action)


def click_center_from_action(
    xml_path: Path | str,
    gt_action: str,
) -> tuple[float, float] | None:
    result = click_bounds_from_action(xml_path, gt_action)
    if result is None:
        return None
    bounds, _tier = result
    cx, cy = bbox_center(bounds)
    return cx, cy
