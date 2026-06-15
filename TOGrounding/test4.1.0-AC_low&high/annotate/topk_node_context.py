"""为 m12 agent 构建 top-k 候选节点的 VLM 文本表。"""

from __future__ import annotations

import json
from pathlib import Path

from process.paths import step_paths


def _load_text_for_emb_by_id(stem: str) -> dict[int, str]:
    nodes_path = step_paths(stem)["nodes"]
    if not nodes_path.is_file():
        return {}
    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)
    out: dict[int, str] = {}
    for node in nodes:
        if "node_id" not in node:
            continue
        out[int(node["node_id"])] = str(node.get("text_for_emb") or "").strip()
    return out


def _format_label(text: str) -> tuple[str, str]:
    text = (text or "").strip()
    if not text:
        return "⚠ (no label)", "—"
    return text, f'text="{text}"'


def build_topk_candidate_table(
    stem: str,
    top_k_nodes: list[dict],
    *,
    target_object: str = "",
) -> str:
    """
    将 top-k 检索候选序列化为 Markdown 表（按 final_sim 降序）。

    列：# | Label | Semantic | score
    """
    if not top_k_nodes:
        return ""

    text_by_id = _load_text_for_emb_by_id(stem)
    ranked = sorted(
        top_k_nodes,
        key=lambda n: float(n.get("final_sim", 0)),
        reverse=True,
    )

    lines = [
        "## Candidate Interactive Nodes (# = annotated boxes on screenshot)",
        f"Total candidates: {len(ranked)}",
    ]
    target_object = (target_object or "").strip()
    if target_object:
        lines.append(f'Retrieved target: "{target_object}"')
    lines.extend(
        [
            "Only these # labels are highlighted. For click/long_press, set node_id to one of them.",
            "",
            "| # | Label | Semantic | score |",
            "|-----|--------|----------|-------|",
        ]
    )

    for node in ranked:
        node_id = int(node["node_id"])
        sim = float(node.get("final_sim", 0))
        text = str(node.get("text_for_emb") or text_by_id.get(node_id, "")).strip()
        label, semantic = _format_label(text)
        lines.append(f"| #{node_id} | {label} | {semantic} | {sim:.3f} |")

    return "\n".join(lines) + "\n"
