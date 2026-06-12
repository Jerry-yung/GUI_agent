#!/usr/bin/env python3
"""
rank_to.py

对每个 stem 的所有 TO 单独做 top-1 检索，按命中情况与分数排序。

每个 TO 使用 cos_sim 矩阵的一列检索 top-1 node；
hit 定义与 judge.py 一致：pred node_id ∈ step_GT.nearest_5。

排序键：
  - hit 组优先：(-pred_score, -margin)
  - miss 组：(-gt_max_sim, -pred_score)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from config.data_paths import BASE_DIR
from utils.gt_node_match import gt_node_ids_from_step_gt

COS_SIM_DIR = BASE_DIR / "embeddings" / "cos_sim"
STEP_GT_DIR = BASE_DIR / "step_GT"
TO_INDEX_DIR = PROJECT_ROOT / "target" / "TO_index"
TO_RANK_DIR = PROJECT_ROOT / "target" / "TO_rank"

TO_MODES = ("best", "mid", "worst")


def pick_to_id(ranking: list[dict[str, Any]], mode: str) -> int:
    """从 rank_to 排行榜选取 TO_id：best=第1名，mid=中位数，worst=倒数第1。"""
    mode = mode.lower()
    if mode not in TO_MODES:
        raise ValueError(f"Unknown TO mode: {mode!r}, expected one of {TO_MODES}")

    n = len(ranking)
    if n == 0:
        raise ValueError("ranking is empty")

    if mode == "best":
        return int(ranking[0]["TO_id"])
    if mode == "worst":
        return int(ranking[-1]["TO_id"])
    return int(ranking[(n - 1) // 2]["TO_id"])


def _top1_for_column(
    sims: np.ndarray,
    node_ids: np.ndarray,
) -> tuple[int, float, float]:
    """对单列相似度取 top-1，返回 (pred_node_id, pred_score, margin)。"""
    order = np.lexsort((node_ids, -sims))
    best_idx = int(order[0])
    pred_score = float(sims[best_idx])

    if len(order) > 1:
        second_idx = int(order[1])
        margin = pred_score - float(sims[second_idx])
    else:
        margin = pred_score

    return int(node_ids[best_idx]), pred_score, margin


def _gt_max_sim(
    sims: np.ndarray,
    node_ids: np.ndarray,
    gt_node_ids: set[int],
) -> float:
    """该 TO 在 nearest_5 节点上的最大相似度。"""
    if not gt_node_ids:
        return 0.0
    gt_sims = [float(sims[i]) for i, nid in enumerate(node_ids) if int(nid) in gt_node_ids]
    return max(gt_sims) if gt_sims else 0.0


def _sort_key(entry: dict[str, Any]) -> tuple:
    if entry["hit"]:
        return (0, -entry["pred_score"], -entry["margin"], entry["TO_id"])
    return (1, -entry["gt_max_sim"], -entry["pred_score"], entry["TO_id"])


def rank_to_for_stem(stem: str) -> dict[str, Any]:
    """对单个 stem 生成 TO 排行榜。"""
    cos_sim_path = COS_SIM_DIR / f"{stem}.npz"
    gt_path = STEP_GT_DIR / f"{stem}.json"
    to_index_path = TO_INDEX_DIR / f"{stem}.json"

    if not cos_sim_path.is_file():
        raise FileNotFoundError(f"缺少 cos_sim: {cos_sim_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"缺少 step_GT: {gt_path}")
    if not to_index_path.is_file():
        raise FileNotFoundError(f"缺少 TO_index: {to_index_path}")

    with np.load(cos_sim_path) as data:
        sim_matrix = data["sim_matrix"]
        node_ids = data["node_ids"]
        to_ids = data["to_ids"]

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    with open(to_index_path, "r", encoding="utf-8") as f:
        to_index = json.load(f)

    to_meta = {int(item["TO_id"]): item for item in to_index}
    gt_node_ids = gt_node_ids_from_step_gt(gt_data)
    gt_set = set(gt_node_ids)

    entries: list[dict[str, Any]] = []
    for j, tid in enumerate(to_ids):
        tid = int(tid)
        sims = sim_matrix[:, j]
        pred_node_id, pred_score, margin = _top1_for_column(sims, node_ids)
        hit = pred_node_id in gt_set
        gt_max = _gt_max_sim(sims, node_ids, gt_set)

        meta = to_meta.get(tid, {})
        entries.append(
            {
                "TO_id": tid,
                "TO_string": meta.get("TO_string", ""),
                "TO_LLM": meta.get("TO_LLM", []),
                "hit": hit,
                "pred_node_id": pred_node_id,
                "pred_score": round(pred_score, 6),
                "margin": round(margin, 6),
                "gt_max_sim": round(gt_max, 6),
            }
        )

    entries.sort(key=_sort_key)

    ranking: list[dict[str, Any]] = []
    first_hit_rank: int | None = None
    for rank, entry in enumerate(entries, start=1):
        item = {"rank": rank, **entry}
        ranking.append(item)
        if entry["hit"] and first_hit_rank is None:
            first_hit_rank = rank

    hit_tos = sum(1 for e in entries if e["hit"])
    oracle_hit = hit_tos > 0
    rank1_hit = ranking[0]["hit"] if ranking else False

    return {
        "stem": stem,
        "gt_node_ids": gt_node_ids,
        "total_tos": len(ranking),
        "hit_tos": hit_tos,
        "oracle_hit": oracle_hit,
        "rank1_hit": rank1_hit,
        "first_hit_rank": first_hit_rank,
        "ranking": ranking,
    }


def _list_stems(start: int | None = None, end: int | None = None) -> list[str]:
    stems = sorted(p.stem for p in COS_SIM_DIR.glob("*.npz"))
    if start is not None or end is not None:
        s = start if start is not None else 0
        e = end if end is not None else len(stems)
        stems = stems[s:e]
    return stems


def rank_to_all(
    stems: list[str] | None = None,
    *,
    save: bool = True,
) -> dict[str, Any]:
    """批量生成 TO 排行榜并汇总统计。"""
    if stems is None:
        stems = _list_stems()

    TO_RANK_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for stem in stems:
        try:
            result = rank_to_for_stem(stem)
            results.append(result)
            if save:
                out_path = TO_RANK_DIR / f"{stem}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{stem}: {exc}")

    total = len(results)
    oracle_count = sum(1 for r in results if r["oracle_hit"])
    rank1_count = sum(1 for r in results if r["rank1_hit"])
    mrr_sum = sum(1.0 / r["first_hit_rank"] for r in results if r["first_hit_rank"] is not None)
    avg_hit_tos = sum(r["hit_tos"] for r in results) / total if total > 0 else 0.0

    summary = {
        "total": total,
        "oracle_hit_count": oracle_count,
        "oracle_hit_rate": round(oracle_count / total, 4) if total > 0 else 0.0,
        "rank1_hit_count": rank1_count,
        "rank1_hit_rate": round(rank1_count / total, 4) if total > 0 else 0.0,
        "mrr": round(mrr_sum / total, 4) if total > 0 else 0.0,
        "avg_hit_tos": round(avg_hit_tos, 4),
        "errors": errors,
    }

    if save and results:
        summary_path = TO_RANK_DIR / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    return {**summary, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser(description="rank_to.py — 按 TO 独立检索排序")
    parser.add_argument("--stem", type=str, default=None, help="只处理单个 stem")
    parser.add_argument("--start", type=int, default=None, help="stem 列表切片起点")
    parser.add_argument("--end", type=int, default=None, help="stem 列表切片终点")
    parser.add_argument("--no-save", action="store_true", help="不写入 target/TO_rank/")
    args = parser.parse_args()

    print("=" * 60)
    print("rank_to.py — TO 独立检索排序")
    print("=" * 60)

    if args.stem:
        result = rank_to_for_stem(args.stem)
        if not args.no_save:
            TO_RANK_DIR.mkdir(parents=True, exist_ok=True)
            out_path = TO_RANK_DIR / f"{args.stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"已保存: {out_path}")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    stems = _list_stems(args.start, args.end)
    if args.start is not None or args.end is not None:
        print(f"范围: [{args.start or 0}, {args.end or 'end'}) -> {len(stems)} 个样本")
    print("=" * 60)

    summary = rank_to_all(stems, save=not args.no_save)

    print(f"样本数: {summary['total']}")
    print(f"oracle_hit_rate: {summary['oracle_hit_rate']} ({summary['oracle_hit_count']}/{summary['total']})")
    print(f"rank1_hit_rate:  {summary['rank1_hit_rate']} ({summary['rank1_hit_count']}/{summary['total']})")
    print(f"mrr:             {summary['mrr']}")
    print(f"avg_hit_tos:     {summary['avg_hit_tos']}")
    if summary["errors"]:
        print(f"错误/跳过: {len(summary['errors'])}")
        for msg in summary["errors"][:5]:
            print(f"  {msg}")
        if len(summary["errors"]) > 5:
            print(f"  ... 共 {len(summary['errors'])} 条")
    if not args.no_save and summary["total"] > 0:
        print(f"已保存: {TO_RANK_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
