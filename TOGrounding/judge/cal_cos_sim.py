#!/usr/bin/env python3
"""
cal_cos_sim.py

读取 AC_data/embeddings/nodes_emb 和 TO_emb，
对每个 stem 计算所有 nodes 与所有 TOs 的余弦相似度矩阵，
保存为 AC_data/embeddings/cos_sim/{stem}.npz

.npz 内部结构：
  - sim_matrix: float32, shape (num_nodes, num_TOs)
  - node_ids:   int,    shape (num_nodes,)
  - to_ids:     int,    shape (num_TOs,)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "AC_data"
NODES_DIR = BASE_DIR / "nodes"
NODES_EMB_DIR = BASE_DIR / "embeddings" / "nodes_emb"
TO_INDEX_DIR = PROJECT_ROOT / "target" / "TO_index"
TO_EMB_DIR = BASE_DIR / "embeddings" / "TO_emb"
COS_SIM_DIR = BASE_DIR / "embeddings" / "cos_sim"


def _normalize(vec: np.ndarray) -> np.ndarray:
    """L2 归一化（行向量）。"""
    norm = np.linalg.norm(vec, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return vec / norm


def process_stem(stem: str) -> bool:
    nodes_file = NODES_DIR / f"{stem}.json"
    to_index_file = TO_INDEX_DIR / f"{stem}.json"

    if not nodes_file.is_file():
        print(f"  [{stem}] 缺少 nodes，跳过")
        return False
    if not to_index_file.is_file():
        print(f"  [{stem}] 缺少 TO_index，跳过")
        return False

    # 读取 node_ids
    with open(nodes_file, "r", encoding="utf-8") as f:
        nodes = json.load(f)
    node_ids = [n["node_id"] for n in nodes]
    if not node_ids:
        print(f"  [{stem}] nodes 为空，跳过")
        return False

    # 读取 TO_ids
    with open(to_index_file, "r", encoding="utf-8") as f:
        to_index = json.load(f)
    to_ids = [t["TO_id"] for t in to_index]
    if not to_ids:
        print(f"  [{stem}] TO_index 为空，跳过")
        return False

    # 加载 node embeddings
    node_vecs = []
    for nid in node_ids:
        npy = NODES_EMB_DIR / stem / f"{stem}_{nid}.npy"
        if not npy.is_file():
            print(f"  [{stem}] 缺少 node embedding: {npy.name}")
            return False
        node_vecs.append(np.load(npy).astype(np.float32))

    # 加载 TO embeddings
    to_vecs = []
    for tid in to_ids:
        npy = TO_EMB_DIR / stem / f"{stem}_{tid}.npy"
        if not npy.is_file():
            print(f"  [{stem}] 缺少 TO embedding: {npy.name}")
            return False
        to_vecs.append(np.load(npy).astype(np.float32))

    # 构建矩阵
    node_matrix = np.stack(node_vecs, axis=0)  # (N, D)
    to_matrix = np.stack(to_vecs, axis=0)      # (M, D)

    # 余弦相似度 = 归一化后点积
    node_matrix = _normalize(node_matrix)
    to_matrix = _normalize(to_matrix)
    sim_matrix = node_matrix @ to_matrix.T     # (N, M)

    # 保存
    COS_SIM_DIR.mkdir(parents=True, exist_ok=True)
    out_path = COS_SIM_DIR / f"{stem}.npz"
    np.savez(
        out_path,
        sim_matrix=sim_matrix.astype(np.float32),
        node_ids=np.asarray(node_ids, dtype=np.int32),
        to_ids=np.asarray(to_ids, dtype=np.int32),
    )
    return True


def main() -> None:
    # 以 nodes 为基准，找同时有 TO_index 的 stem
    stems = sorted(p.stem for p in NODES_DIR.glob("*.json"))
    print("=" * 60)
    print("cal_cos_sim.py — 节点 embedding × TO embedding 余弦相似度")
    print("=" * 60)
    print(f"候选 stem 数: {len(stems)}")
    print("=" * 60)

    ok = 0
    skip = 0
    for i, stem in enumerate(stems, 1):
        if i % 500 == 0 or i == len(stems):
            print(f"  进度 {i}/{len(stems)} ...")
        if process_stem(stem):
            ok += 1
        else:
            skip += 1

    print(f"\n{'=' * 60}")
    print(f"处理完成: 成功={ok}, 跳过={skip}")
    print(f"输出目录: {COS_SIM_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
