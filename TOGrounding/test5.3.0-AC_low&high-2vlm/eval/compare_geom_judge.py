#!/usr/bin/env python3
"""
临时脚本：用 runs 内记录的 step_correct 对比「旧 id 规则」与当前正式 judge_m2（几何）。

当前正式 judge_m2（eval/judge_llm.py）：
  hit = (pred 节点中心 ∈ 任一 nearest_5 bbox) OR (norm_dist(center, GT) < 0.04)

本脚本中 old = runs 里已存 step_correct；new = 用正式评测逻辑重算。
旧 id 规则对照实现见 judge_m2_legacy_id()。

用法:
  python eval/compare_geom_judge.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.compare_runs import load_episodes, parse_run_meta
from eval.run_naming import run_label
from eval.step_judge import is_evaluable_step, judge_step_match


def _metrics_from_steps(
    episodes: list[dict],
    *,
    agent: str,
    top_k: int | None,
    use_geom: bool,
) -> dict:
    steps: list[dict] = []
    for ep in episodes:
        steps.extend(s for s in ep.get("steps", []) if is_evaluable_step(s))

    type_ok = step_ok = 0
    for s in steps:
        if use_geom:
            v = judge_step_match(
                s.get("gt", {}),
                s.get("pred_action"),
                stem=s.get("stem", ""),
                has_annotated_nodes=bool(s.get("has_annotated_nodes", True)),
                agent=agent,
                top_k_nodes=s.get("top_k_nodes"),
                top_k=top_k,
            )
            type_ok += int(v["type_correct"])
            step_ok += int(v["step_correct"])
        else:
            type_ok += int(s.get("type_correct", False))
            step_ok += int(s.get("step_correct", False))

    eval_eps = [ep for ep in episodes if any(is_evaluable_step(x) for x in ep.get("steps", []))]
    sr_ok = 0
    for ep in eval_eps:
        ep_steps = [s for s in ep.get("steps", []) if is_evaluable_step(s)]
        if not ep_steps:
            continue
        if use_geom:
            ok = all(
                judge_step_match(
                    s.get("gt", {}),
                    s.get("pred_action"),
                    stem=s.get("stem", ""),
                    has_annotated_nodes=bool(s.get("has_annotated_nodes", True)),
                    agent=agent,
                    top_k_nodes=s.get("top_k_nodes"),
                    top_k=top_k,
                )["step_correct"]
                for s in ep_steps
            )
        else:
            ok = all(s.get("step_correct", False) for s in ep_steps)
        if ok:
            sr_ok += 1

    n = len(steps)
    ne = len(eval_eps)
    return {
        "steps": n,
        "episodes": ne,
        "type_acc": type_ok / n if n else 0.0,
        "step_acc": step_ok / n if n else 0.0,
        "sr": sr_ok / ne if ne else 0.0,
        "type_correct": type_ok,
        "step_correct": step_ok,
        "sr_correct": sr_ok,
    }


def _fmt_delta(old: float, new: float) -> str:
    d = new - old
    sign = "+" if d >= 0 else ""
    return f"{sign}{d * 100:.2f}pp"


def _fmt_rate(x: float, num: int, den: int) -> str:
    return f"{x * 100:.2f}% ({num}/{den})"


def compare_run(path: Path) -> dict | None:
    meta = parse_run_meta(path)
    if meta is None:
        return None
    if str(meta.get("agent", "")).upper() == "CPM":
        return None

    episodes = load_episodes(path)
    agent = str(meta["agent"])
    top_k = meta.get("top_k")

    old = _metrics_from_steps(episodes, agent=agent, top_k=top_k, use_geom=False)
    new = _metrics_from_steps(episodes, agent=agent, top_k=top_k, use_geom=True)

    return {
        "run_file": path.name,
        "label": run_label(meta),
        "ac_mode": meta["ac_mode"],
        "agent": agent,
        "old": old,
        "new": new,
        "delta_type_pp": (new["type_acc"] - old["type_acc"]) * 100,
        "delta_step_pp": (new["step_acc"] - old["step_acc"]) * 100,
        "delta_sr_pp": (new["sr"] - old["sr"]) * 100,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="对比 judge_m2 几何统一前后指标")
    parser.add_argument(
        "runs_dir",
        nargs="?",
        default=str(PROJECT_ROOT / "runs"),
        help="runs 目录（默认 runs/）",
    )
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)

    rows: list[dict] = []
    for path in sorted(runs_dir.glob("*.json")):
        row = compare_run(path)
        if row:
            rows.append(row)

    if not rows:
        print("未找到非 CPM 的 run 文件。")
        return

    print("old = runs 内已存 step_correct（旧 id 规则）")
    print("new = 正式 judge_step_match 重算（几何 judge_m2）\n")
    header = (
        f"{'run':<46} {'type(old→new)':<22} {'step(old→new)':<22} "
        f"{'SR(old→new)':<22} {'Δstep':>8}"
    )
    print(header)
    print("-" * len(header))

    total_old_step = total_new_step = total_n = 0
    for r in rows:
        o, n = r["old"], r["new"]
        total_old_step += o["step_correct"]
        total_new_step += n["step_correct"]
        total_n += o["steps"]
        print(
            f"{r['ac_mode']}_{r['label']:<38} "
            f"{_fmt_rate(o['type_acc'], o['type_correct'], o['steps']):<11}→"
            f"{_fmt_rate(n['type_acc'], n['type_correct'], n['steps']):<10} "
            f"{_fmt_rate(o['step_acc'], o['step_correct'], o['steps']):<11}→"
            f"{_fmt_rate(n['step_acc'], n['step_correct'], n['steps']):<10} "
            f"{_fmt_rate(o['sr'], o['sr_correct'], o['episodes']):<11}→"
            f"{_fmt_rate(n['sr'], n['sr_correct'], n['episodes']):<10} "
            f"{_fmt_delta(o['step_acc'], n['step_acc']):>8}"
        )

    print("-" * len(header))
    if total_n:
        old_acc = total_old_step / total_n
        new_acc = total_new_step / total_n
        print(
            f"{'ALL (macro step)':<46} "
            f"{'—':<22} "
            f"{_fmt_rate(old_acc, total_old_step, total_n):<11}→"
            f"{_fmt_rate(new_acc, total_new_step, total_n):<10} "
            f"{'—':<22} "
            f"{_fmt_delta(old_acc, new_acc):>8}"
        )
        print(f"\n汇总: step_acc {old_acc*100:.2f}% → {new_acc*100:.2f}% "
              f"({_fmt_delta(old_acc, new_acc)}, +{total_new_step - total_old_step} steps)")


if __name__ == "__main__":
    main()
