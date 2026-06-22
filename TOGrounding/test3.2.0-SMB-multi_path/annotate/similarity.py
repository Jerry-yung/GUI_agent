"""Cosine similarity for TopK."""
from __future__ import annotations

import numpy as np


def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
    docs = np.asarray(doc_vecs, dtype=np.float32)
    if docs.size == 0:
        return np.array([], dtype=np.float32)
    q_norm = np.linalg.norm(q)
    doc_norms = np.linalg.norm(docs, axis=1)
    denom = q_norm * doc_norms
    denom = np.where(denom == 0, 1e-12, denom)
    return (docs @ q / denom).astype(np.float32)


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    if scores.size == 0 or k <= 0:
        return np.array([], dtype=np.int64)
    k = min(k, scores.size)
    return np.argsort(scores)[-k:][::-1]


def label_ranks_by_score(scores: np.ndarray, labels: list[str]) -> dict[str, int]:
    """Map node label (lowercase) to 1-based rank; rank 1 = highest similarity."""
    if scores.size == 0 or len(labels) != scores.size:
        return {}
    order = np.argsort(scores)[::-1]
    ranks: dict[str, int] = {}
    for rank, pool_idx in enumerate(order.tolist(), start=1):
        label = str(labels[pool_idx]).lower()
        ranks[label] = rank
    return ranks
