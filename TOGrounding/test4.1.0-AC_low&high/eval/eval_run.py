#!/usr/bin/env python3
"""
评估 runs/*.json：type_acc、step_acc 及按 action_type 分组统计。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.run_naming import parse_run_filename
from eval.step_judge import compute_retrieval_hit, is_evaluable_step, judge_step_match
from process.paths import step_paths

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"


def _load_gt_for_stem(stem: str, embedded_gt: dict | None) -> dict:
    if embedded_gt:
        return embedded_gt
    gt_path = step_paths(stem)["gt"]
    with open(gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_episodes_from_run(payload: object) -> list[dict]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        return payload["episodes"]
    return []


def _parse_run_filename(stem: str) -> dict:
    """从 runs 文件名解析元信息。"""
    return parse_run_filename(stem) or {}


def _iter_step_records(episode: dict) -> list[dict]:
    if episode.get("steps"):
        records = []
        for step in episode["steps"]:
            rec = dict(step)
            rec.setdefault("episode_id", episode.get("episode_id"))
            records.append(rec)
        return records

    if "stem" in episode:
        return [episode]
    return []


def _evaluate_step(
    step_rec: dict,
    *,
    agent: str = "m2",
    top_k: int | None = None,
    recompute: bool = False,
) -> dict:
    stem = step_rec.get("stem", "")
    gt = _load_gt_for_stem(stem, step_rec.get("gt"))
    pred = step_rec.get("pred_action")

    if (
        not recompute
        and "type_correct" in step_rec
        and "step_correct" in step_rec
    ):
        verdict = {
            "type_correct": bool(step_rec["type_correct"]),
            "step_correct": bool(step_rec["step_correct"]),
            "gt_action_type": gt.get("action_type"),
            "pred_action_type": (pred or {}).get("action_type"),
            "retrieval_hit": compute_retrieval_hit(
                gt,
                step_rec.get("top_k_nodes"),
                top_k=top_k,
            ),
            "detail": step_rec.get("eval_detail", {}),
        }
    else:
        verdict = judge_step_match(
            gt,
            pred,
            stem=stem,
            has_annotated_nodes=bool(step_rec.get("has_annotated_nodes", True)),
            agent=agent,
            top_k_nodes=step_rec.get("top_k_nodes"),
            top_k=top_k,
        )

    return {
        "episode_id": step_rec.get("episode_id"),
        "stem": stem,
        "step_idx": step_rec.get("step_idx"),
        "step_status": step_rec.get("status"),
        "has_annotated_nodes": step_rec.get("has_annotated_nodes"),
        "type_correct": verdict["type_correct"],
        "step_correct": verdict["step_correct"],
        "retrieval_hit": verdict.get("retrieval_hit"),
        "gt_action_type": verdict.get("gt_action_type") or gt.get("action_type"),
        "pred_action_type": verdict.get("pred_action_type"),
        "detail": verdict["detail"],
    }


def _aggregate(step_results: list[dict]) -> dict:
    total = len(step_results)
    type_correct = sum(1 for r in step_results if r["type_correct"])
    step_correct = sum(1 for r in step_results if r["step_correct"])
    retrieval_eligible = [r for r in step_results if r.get("retrieval_hit") is not None]
    retrieval_correct = sum(1 for r in retrieval_eligible if r["retrieval_hit"])

    by_type: dict[str, dict] = defaultdict(
        lambda: {"count": 0, "type_correct": 0, "step_correct": 0}
    )
    for r in step_results:
        gt_type = r["gt_action_type"] or "unknown"
        by_type[gt_type]["count"] += 1
        by_type[gt_type]["type_correct"] += int(r["type_correct"])
        by_type[gt_type]["step_correct"] += int(r["step_correct"])

    by_action_type = {}
    for action_type, stats in sorted(by_type.items()):
        count = stats["count"]
        by_action_type[action_type] = {
            "count": count,
            "type_acc": round(stats["type_correct"] / count, 6) if count else 0.0,
            "step_acc": round(stats["step_correct"] / count, 6) if count else 0.0,
        }

    retrieval_total = len(retrieval_eligible)
    return {
        "total_steps": total,
        "type_correct": type_correct,
        "step_correct": step_correct,
        "type_acc": round(type_correct / total, 6) if total else 0.0,
        "step_acc": round(step_correct / total, 6) if total else 0.0,
        "retrieval_eligible_steps": retrieval_total,
        "retrieval_correct": retrieval_correct,
        "retrieval_hit_rate": round(retrieval_correct / retrieval_total, 6)
        if retrieval_total
        else 0.0,
        "by_action_type": by_action_type,
    }


def evaluate_run(run_path: Path, *, recompute: bool = False) -> dict:
    with open(run_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    episodes = _load_episodes_from_run(payload)
    file_meta = _parse_run_filename(run_path.stem)
    if isinstance(payload, dict):
        for key in ("ac_mode", "agent", "top_k", "vlm_model"):
            file_meta.setdefault(key, payload.get(key))

    step_results: list[dict] = []
    for episode in episodes:
        for step_rec in _iter_step_records(episode):
            if not step_rec.get("stem"):
                continue
            if not is_evaluable_step(step_rec):
                continue
            step_results.append(
                _evaluate_step(
                    step_rec,
                    agent=str(file_meta.get("agent", "m2")),
                    top_k=file_meta.get("top_k"),
                    recompute=recompute,
                )
            )

    summary = _aggregate(step_results)

    return {
        "schema_version": "4.1.0-eval",
        "run_path": str(run_path.resolve()),
        "ac_mode": file_meta.get("ac_mode"),
        "agent": file_meta.get("agent"),
        "vlm_model": file_meta.get("vlm_model"),
        "top_k": file_meta.get("top_k"),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        **summary,
        "steps": step_results,
    }


def refresh_run_file(run_path: Path, *, recompute: bool = True) -> int:
    """用当前 judge 逻辑写回 runs/*.json 中每步的评测字段。返回更新的可评测步数。"""
    with open(run_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    episodes = _load_episodes_from_run(payload)
    file_meta = _parse_run_filename(run_path.stem)
    agent = str(file_meta.get("agent", "m2"))
    top_k = file_meta.get("top_k")
    updated = 0

    for episode in episodes:
        for step_rec in episode.get("steps", []):
            if not step_rec.get("stem") or not is_evaluable_step(step_rec):
                continue
            verdict = judge_step_match(
                _load_gt_for_stem(step_rec["stem"], step_rec.get("gt")),
                step_rec.get("pred_action"),
                stem=step_rec["stem"],
                has_annotated_nodes=bool(step_rec.get("has_annotated_nodes", True)),
                agent=agent,
                top_k_nodes=step_rec.get("top_k_nodes"),
                top_k=top_k,
            )
            step_rec["type_correct"] = verdict["type_correct"]
            step_rec["step_correct"] = verdict["step_correct"]
            if verdict.get("retrieval_hit") is not None:
                step_rec["retrieval_hit"] = verdict["retrieval_hit"]
            step_rec["eval_detail"] = verdict.get("detail", {})
            updated += 1

    with open(run_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description="评估 AC runs JSON")
    parser.add_argument(
        "run",
        nargs="?",
        help="runs JSON 路径（相对项目根或绝对路径）",
    )
    parser.add_argument(
        "--run",
        dest="run_flag",
        help="runs JSON 路径",
    )
    parser.add_argument(
        "--out",
        default=str(RESULTS_DIR),
        help="评估结果输出目录（默认 eval/results/）",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="忽略 runs 内已缓存的 step_correct，用当前 judge 重算",
    )
    parser.add_argument(
        "--update-run",
        action="store_true",
        help="将重算结果写回 runs JSON（常与 --recompute 联用）",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="处理 runs/ 下全部 *.json（忽略位置参数）",
    )
    args = parser.parse_args()

    if args.all:
        runs_dir = PROJECT_ROOT / "runs"
        run_paths = sorted(runs_dir.glob("*.json"))
        if not run_paths:
            raise SystemExit(f"runs 目录无 JSON: {runs_dir}")
        for run_path in run_paths:
            if args.update_run:
                n = refresh_run_file(run_path)
                print(f"[update-run] {run_path.name}: {n} steps")
            report = evaluate_run(run_path, recompute=args.recompute or args.update_run)
            out_dir = Path(args.out)
            if not out_dir.is_absolute():
                out_dir = PROJECT_ROOT / out_dir
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"eval_{run_path.stem}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(
                f"{run_path.name}: step_acc={report['step_acc']:.4f} "
                f"({report['step_correct']}/{report['total_steps']}) → {out_path.name}"
            )
        print(f"完成 {len(run_paths)} 个 run")
        return

    run_arg = args.run_flag or args.run
    if not run_arg:
        parser.error("请指定 runs JSON 路径")

    run_path = Path(run_arg)
    if not run_path.is_absolute():
        run_path = PROJECT_ROOT / run_path
    if not run_path.is_file():
        raise SystemExit(f"runs 文件不存在: {run_path}")

    if args.update_run:
        n = refresh_run_file(run_path)
        print(f"[update-run] 已写回 {n} 步 → {run_path.name}")

    report = evaluate_run(
        run_path,
        recompute=args.recompute or args.update_run,
    )

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"eval_{run_path.stem}.json"
    out_path = out_dir / out_name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"评估: {run_path.name}")
    print("=" * 60)
    print(f"total_steps:  {report['total_steps']}")
    print(f"type_acc:     {report['type_acc']:.4f} ({report['type_correct']}/{report['total_steps']})")
    print(f"step_acc:     {report['step_acc']:.4f} ({report['step_correct']}/{report['total_steps']})")
    print("\nby_action_type:")
    for action_type, stats in report["by_action_type"].items():
        print(
            f"  {action_type:16s} n={stats['count']:4d}  "
            f"type_acc={stats['type_acc']:.4f}  step_acc={stats['step_acc']:.4f}"
        )
    print("=" * 60)
    print(f"结果: {out_path}")


if __name__ == "__main__":
    main()
