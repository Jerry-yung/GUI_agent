#!/usr/bin/env python3
"""运行时 TO embedding 与 nodes_emb 余弦相似度，取 top_k 节点。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_set.llm import vlm_embedding
from process.paths import node_emb_path, step_paths

EMBEDDING_DIM = 2560


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _embed_to_text(to_text: str) -> np.ndarray:
    vec = np.asarray(vlm_embedding.embed_text(to_text), dtype=np.float32)
    if vec.shape != (EMBEDDING_DIM,):
        raise ValueError(f"TO embedding 维度异常: {vec.shape}")
    return vec


def rank_nodes_by_to(stem: str, to_text: str, top_k: int) -> list[dict]:
    """
    计算 TO 与当前 step 所有 node embedding 的余弦相似度，返回 top_k 节点。

    每个节点 dict 含 node_id, bounds, text_for_emb, final_sim。
    """
    if top_k <= 0:
        raise ValueError("top_k 必须 > 0")

    paths = step_paths(stem)
    nodes_path = paths["nodes"]
    if not nodes_path.is_file():
        raise FileNotFoundError(f"缺少 nodes: {nodes_path}")

    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    if not nodes:
        return []

    node_ids: list[int] = []
    node_vecs: list[np.ndarray] = []
    for node in nodes:
        node_id = int(node["node_id"])
        emb_path = node_emb_path(stem, node_id)
        if not emb_path.is_file():
            raise FileNotFoundError(f"缺少 node embedding: {emb_path}")
        vec = np.load(emb_path).astype(np.float32)
        if vec.shape != (EMBEDDING_DIM,):
            raise ValueError(f"node embedding 维度异常: {emb_path} {vec.shape}")
        node_ids.append(node_id)
        node_vecs.append(vec)

    to_vec = _embed_to_text(to_text)
    mat = _normalize_rows(np.stack(node_vecs, axis=0))
    to_norm = to_vec / (np.linalg.norm(to_vec) or 1.0)
    sims = mat @ to_norm

    k = min(top_k, len(node_ids))
    top_idx = np.argsort(-sims)[:k]

    nodes_by_id = {int(n["node_id"]): n for n in nodes}
    selected: list[dict] = []
    for idx in top_idx:
        nid = node_ids[int(idx)]
        node = dict(nodes_by_id[nid])
        node["final_sim"] = float(sims[int(idx)])
        selected.append(node)

    return selected
