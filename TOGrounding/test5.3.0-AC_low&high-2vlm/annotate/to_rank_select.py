"""从 annotate/TO_rank 预存结果中选取 TO_string。"""

from __future__ import annotations

import json
from pathlib import Path

TO_RANK_DIR = Path(__file__).resolve().parent / "TO_rank"
TO_SELECT_CHOICES = frozenset({"best", "mid", "worst", "generate"})


def _rank_index(select: str, total: int) -> int:
    if total <= 0:
        raise ValueError("empty ranking")
    select = select.lower()
    if select == "best":
        return 0
    if select == "worst":
        return total - 1
    if select == "mid":
        return total // 2
    raise ValueError(f"unsupported select: {select!r}")


def load_to_rank_entry(stem: str) -> dict | None:
    path = TO_RANK_DIR / f"{stem}.json"
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_to_string_from_rank(stem: str, select: str) -> str | None:
    """
    从 TO_rank 中取 best / mid / worst 对应的 TO_string。

    Returns:
        TO_string，或文件/排序缺失时 None。
    """
    select = select.lower()
    if select not in ("best", "mid", "worst"):
        raise ValueError(f"pick_to_string_from_rank expects best|mid|worst, got {select!r}")

    payload = load_to_rank_entry(stem)
    if not payload:
        return None

    ranking = payload.get("ranking") or []
    if not ranking:
        return None

    idx = _rank_index(select, len(ranking))
    entry = ranking[idx]
    text = str(entry.get("TO_string", "")).strip()
    return text or None
