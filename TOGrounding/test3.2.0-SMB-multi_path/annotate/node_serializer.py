"""Node text for embedding."""
from __future__ import annotations

from typing import Any


def node_embedding_text(node: dict[str, Any]) -> str:
    parts: list[str] = []
    desc = node.get("content-desc")
    if isinstance(desc, list):
        for item in desc:
            if item:
                parts.append(str(item).strip())
    elif isinstance(desc, str) and desc.strip():
        parts.append(desc.strip())
    label = (node.get("ager_label") or node.get("label") or "").strip()
    if label and label not in parts:
        parts.append(label)
    text = (node.get("text") or "").strip()
    if text:
        parts.append(text)
    return " ".join(parts).strip()
