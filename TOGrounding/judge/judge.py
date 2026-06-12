#!/usr/bin/env python3
"""
judge.py

根据 cos_sim/{stem}.npz 与 step_GT/{stem}.json 的 nearest_5，
判断 embedding 检索 top_k 是否命中 GT 可接受节点（hit@k）。

MODE（best / mid / worst）：用 rank_to 排序后选取单个 TO 的相似度列取 top_k。
hit 定义：pred top_k 的 node_id 与 nearest_5 的 node_id 存在交集。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_JUDGE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _JUDGE_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from judge.rank_to import TO_MODES, pick_to_id, rank_to_for_stem
except ImportError:
    from rank_to import TO_MODES, pick_to_id, rank_to_for_stem

PROJECT_ROOT = _PROJECT_ROOT
BASE_DIR = PROJECT_ROOT / "AC_data"
COS_SIM_DIR = BASE_DIR / "embeddings" / "cos_sim"
STEP_GT_DIR = BASE_DIR / "step_GT"


def _sims_for_mode(
    stem: str,
    sim_matrix: np.ndarray,
    to_ids: np.ndarray,
    mode: str,
) -> tuple[np.ndarray, int]:
    """按 MODE 从 rank_to 排行榜选取 TO，返回该 TO 的相似度列与 TO_id。"""
    ranking = rank_to_for_stem(stem)["ranking"]
    to_id = pick_to_id(ranking, mode)
    col_idx = int(np.where(to_ids == to_id)[0][0])
    return sim_matrix[:, col_idx], to_id


def _rank_top_k(
    final_sims: np.ndarray,
    node_ids: np.ndarray,
    top_k: int,
) -> tuple[list[int], list[float]]:
    """按 final_sim 降序取 top_k，分数相同则 node_id 升序。"""
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    n = len(node_ids)
    k = min(top_k, n)
    # lexsort 最后一键为主键；-final_sims 升序 = final_sims 降序
    order = np.lexsort((node_ids, -final_sims))
    top_indices = order[:k]

    pred_ids = [int(node_ids[i]) for i in top_indices]
    pred_scores = [float(final_sims[i]) for i in top_indices]
    return pred_ids, pred_scores


def judge_hit(stem: str, top_k: int, mode: str) -> dict[str, Any]:
    """
    对单个 stem 判断 hit@k。

    参数:
        stem:  样本名，如 "00000020_001"
        top_k: 取 final cos sim 最高的前 k 个 node
        mode:  "best" | "mid" | "worst"

    返回:
        {
            "stem": str,
            "hit": bool,
            "top_k": int,
            "mode": str,
            "to_id": int,
            "pred_node_ids": list[int],
            "pred_scores": list[float],
            "gt_node_ids": list[int],
            "matched_node_ids": list[int],
        }
    """
    cos_sim_path = COS_SIM_DIR / f"{stem}.npz"
    gt_path = STEP_GT_DIR / f"{stem}.json"

    if not cos_sim_path.is_file():
        raise FileNotFoundError(f"缺少 cos_sim: {cos_sim_path}")
    if not gt_path.is_file():
        raise FileNotFoundError(f"缺少 step_GT: {gt_path}")

    with np.load(cos_sim_path) as data:
        sim_matrix = data["sim_matrix"]
        node_ids = data["node_ids"]
        to_ids = data["to_ids"]

    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    final_sims, to_id = _sims_for_mode(stem, sim_matrix, to_ids, mode)
    pred_node_ids, pred_scores = _rank_top_k(final_sims, node_ids, top_k)

    gt_node_ids = [int(item["node_id"]) for item in gt_data.get("nearest_5", [])]
    gt_set = set(gt_node_ids)
    pred_set = set(pred_node_ids)
    matched = sorted(pred_set & gt_set)

    return {
        "stem": stem,
        "hit": len(matched) > 0,
        "top_k": top_k,
        "mode": mode.lower(),
        "to_id": to_id,
        "pred_node_ids": pred_node_ids,
        "pred_scores": pred_scores,
        "gt_node_ids": gt_node_ids,
        "matched_node_ids": matched,
    }


def judge_all(
    stems: list[str] | None = None,
    top_k: int = 1,
    mode: str = "best",
) -> dict[str, Any]:
    """批量评测，返回汇总统计。"""
    if stems is None:
        stems = sorted(p.stem for p in COS_SIM_DIR.glob("*.npz"))

    results: list[dict[str, Any]] = []
    errors: list[str] = []

    for stem in stems:
        try:
            results.append(judge_hit(stem, top_k, mode))
        except (FileNotFoundError, ValueError) as exc:
            errors.append(f"{stem}: {exc}")

    hit_count = sum(1 for r in results if r["hit"])
    total = len(results)

    return {
        "top_k": top_k,
        "mode": mode.lower(),
        "total": total,
        "hit_count": hit_count,
        "hit_rate": round(hit_count / total, 4) if total > 0 else 0.0,
        "results": results,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="judge.py — embedding 检索 hit@k 评测")
    parser.add_argument("--top-k", type=int, default=1, help="取相似度最高的前 k 个 node")
    parser.add_argument(
        "--mode",
        type=str,
        default="best",
        choices=list(TO_MODES),
        help='TO 选取策略: "best" | "mid" | "worst"',
    )
    parser.add_argument("--stem", type=str, default=None, help="只评测单个 stem")
    args = parser.parse_args()

    print("=" * 60)
    print("judge.py — embedding 检索 hit@k")
    print("=" * 60)
    print(f"top_k={args.top_k}, mode={args.mode}")
    print("=" * 60)

    if args.stem:
        result = judge_hit(args.stem, args.top_k, args.mode)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    summary = judge_all(top_k=args.top_k, mode=args.mode)
    print(f"样本数: {summary['total']}")
    print(f"命中数: {summary['hit_count']}")
    print(f"hit@{args.top_k} ({args.mode}): {summary['hit_rate']}")
    if summary["errors"]:
        print(f"错误/跳过: {len(summary['errors'])}")
        for msg in summary["errors"][:5]:
            print(f"  {msg}")
        if len(summary["errors"]) > 5:
            print(f"  ... 共 {len(summary['errors'])} 条")
    print("=" * 60)


if __name__ == "__main__":
    main()
