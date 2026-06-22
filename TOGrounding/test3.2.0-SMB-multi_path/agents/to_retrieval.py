"""从 cache/retrieval 读取 top1 分数与 margin。"""
from __future__ import annotations

import json

from utils.paths import cache_retrieval_file


def load_retrieval_scores(
    task_name: str,
    page_name: str,
    *,
    top_k: int,
    query_text: str,
) -> tuple[float | None, float | None]:
    if not query_text.strip():
        return None, None
    cache_path = cache_retrieval_file(
        task_name, page_name, top_k=top_k, query_text=query_text
    )
    if not cache_path.is_file():
        return None, None
    try:
        with open(cache_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None, None

    topk = data.get("topk")
    if not isinstance(topk, list) or not topk:
        return None, None

    first = topk[0]
    if not isinstance(first, dict):
        return None, None
    try:
        sim = float(first["score"])
    except (KeyError, TypeError, ValueError):
        sim = None

    margin: float | None = None
    if len(topk) >= 2 and isinstance(topk[1], dict):
        try:
            margin = float(first["score"]) - float(topk[1]["score"])
        except (KeyError, TypeError, ValueError):
            margin = None
    return sim, margin
