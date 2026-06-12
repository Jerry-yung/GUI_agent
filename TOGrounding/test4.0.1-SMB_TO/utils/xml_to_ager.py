"""Parse Mobile3M UI Automator XML into AGER-compatible node dicts."""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path


def _parse_bool(value: str | None) -> bool:
    return (value or "").lower() == "true"


def _parse_bounds(raw: str | None) -> list[int]:
    if not raw:
        return [0, 0, 0, 0]
    nums = [int(x) for x in re.findall(r"\d+", raw)]
    if len(nums) >= 4:
        return nums[:4]
    return [0, 0, 0, 0]


def element_to_node(elem: ET.Element) -> dict:
    attrs = elem.attrib
    desc = (attrs.get("content-desc") or "").strip()
    enabled = _parse_bool(attrs.get("enabled", "true"))
    scrollable = _parse_bool(attrs.get("scrollable"))
    return {
        "index": attrs.get("index", ""),
        "class": attrs.get("class", ""),
        "package": attrs.get("package", ""),
        "resource-id": attrs.get("resource-id", ""),
        "text": (attrs.get("text") or "").strip(),
        "content-desc": [desc] if desc else [None],
        "bounds": _parse_bounds(attrs.get("bounds")),
        "enabled": enabled,
        "visible-to-user": enabled,
        "scrollable-horizontal": False,
        "scrollable-vertical": scrollable,
        "clickable": _parse_bool(attrs.get("clickable")),
        "focusable": _parse_bool(attrs.get("focusable")),
        "checkable": _parse_bool(attrs.get("checkable")),
        "long-clickable": _parse_bool(attrs.get("long-clickable")),
        "selected": _parse_bool(attrs.get("selected")),
        "checked": _parse_bool(attrs.get("checked")),
    }


def parse_mobile3m_xml(xml_path: Path | str) -> list[dict]:
    """Flat list of all elements with bounds (Mobile3M uses android.widget.* tags)."""
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    nodes: list[dict] = []
    for elem in root.iter():
        if elem is root and elem.tag == "hierarchy":
            continue
        bounds = elem.attrib.get("bounds")
        if not bounds:
            continue
        b = _parse_bounds(bounds)
        if b[2] <= b[0] or b[3] <= b[1]:
            continue
        nodes.append(element_to_node(elem))
    return nodes


def get_screen_bounds_from_nodes(nodes: list[dict]) -> list[int]:
    for node in nodes:
        bounds = node.get("bounds") or []
        if len(bounds) == 4:
            w = bounds[2] - bounds[0]
            h = bounds[3] - bounds[1]
            if w >= 500 and h >= 800:
                return bounds
    return [0, 0, 1080, 2400]
