#!/usr/bin/env python3
"""
Convert Mobile3M tasks into single-step four-piece samples for the TO pipeline.

Reads task JSON（默认见脚本顶部 TASK_FILE），expands each click step
parent -> child into Mobile3M_data/:
  step_GT, step_instructions, screenshots, a11y_trees_L0 (*.xml)

Run from GUI_agent/test4.0.1-SMB_query_emb/:
  python get_Mobile3M.py
  python get_Mobile3M.py --task-file simple_tasks_sample.json --limit 50
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.data_paths import MOBILE3M_SRC, PROJECT_ROOT as CFG_ROOT
from utils.mobile3m_gt import (
    build_page_path,
    click_bounds_from_action,
    gt_action_for_step,
    parse_step_instructions,
)
from utils.mobile3m_io import find_graph_dir, load_page_json, load_tasks, page_paths

# ============================================================
# 全局配置：任务 JSON（位于 MOBILE3M_SRC / datasets/Mobile3M/datasets/）
# ============================================================
# 可选值（四选一）：
#   "simple_tasks_sample.json"   — 简单任务抽样（约 600 条，推荐调试）
#   "simple_normal_tasks.json"   — 简单任务全量
#   "complex_tasks_sample.json"  — 复杂任务抽样
#   "complex_normal_tasks.json"  — 复杂任务全量
#
# CLI 仍可用 --task-file 覆盖本变量。
TASK_FILE = "simple_normal_tasks.json"

OUT_DIR = CFG_ROOT / "Mobile3M_data"


def _graph_ready(graph_dir: Path) -> bool:
    return (graph_dir / "all_action_id.json").is_file()


def _convert_task(
    task: dict[str, Any],
    data_dir: Path,
    report: dict[str, Any],
    *,
    step_gt_dir: Path,
    step_ins_dir: Path,
    screenshots_dir: Path,
    a11y_dir: Path,
    incremental: bool = False,
) -> None:
    task_name = str(task.get("name") or "").strip()
    if not task_name:
        report["skipped"].append({"task": task, "reason": "empty name"})
        return

    graph_dir = find_graph_dir(data_dir, task_name)
    if graph_dir is None:
        report["skipped"].append({"task_name": task_name, "reason": "graph dir not found"})
        return

    if not _graph_ready(graph_dir):
        report["skipped"].append(
            {
                "task_name": task_name,
                "reason": "missing all_action_id.json; run convert_mobile3m.py --mode explore",
            }
        )
        return

    pages = build_page_path(task_name)
    if len(pages) < 2:
        report["skipped"].append({"task_name": task_name, "reason": "path too short"})
        return

    task_text = str(task.get("task") or "").strip()
    concise, step_instructions = parse_step_instructions(task_text)

    for k in range(len(pages) - 1):
        parent_page = pages[k]
        child_page = pages[k + 1]
        stem = f"{task_name}_{k:03d}"

        if incremental and (step_gt_dir / f"{stem}.json").is_file():
            report["skipped_existing"].append(stem)
            continue

        try:
            child_json = load_page_json(graph_dir, child_page)
        except (OSError, json.JSONDecodeError) as exc:
            report["failed"].append(
                {"stem": stem, "reason": f"child page json error: {exc}"}
            )
            continue

        gt_action = gt_action_for_step(child_json)
        if not gt_action.startswith("click("):
            report["skipped_steps"].append(
                {
                    "stem": stem,
                    "reason": "non-click step",
                    "gt_action": gt_action[:120],
                }
            )
            continue

        paths = page_paths(graph_dir, parent_page)
        xml_src = paths["xml"]
        shot_src = paths["screenshot"]
        if not xml_src.is_file():
            report["failed"].append({"stem": stem, "reason": f"missing xml: {xml_src}"})
            continue
        if not shot_src.is_file():
            report["failed"].append({"stem": stem, "reason": f"missing screenshot: {shot_src}"})
            continue

        bounds_result = click_bounds_from_action(xml_src, gt_action)
        if bounds_result is None:
            report["failed"].append(
                {
                    "stem": stem,
                    "reason": "unmatched gt_action in xml",
                    "gt_action": gt_action[:200],
                }
            )
            continue

        gt_bounds, bounds_tier = bounds_result

        if k < len(step_instructions):
            instruction = step_instructions[k]
        elif concise:
            instruction = concise
        else:
            instruction = task_text

        gt_payload = {
            "action_type": "click",
            "gt_action": gt_action,
            "gt_bounds": gt_bounds,
            "meta": {
                "task_name": task_name,
                "parent_page": parent_page,
                "child_page": child_page,
                "bounds_match_tier": bounds_tier,
            },
        }

        with open(step_gt_dir / f"{stem}.json", "w", encoding="utf-8") as f:
            json.dump(gt_payload, f, ensure_ascii=False, indent=2)

        with open(step_ins_dir / f"{stem}.txt", "w", encoding="utf-8") as f:
            f.write(instruction)

        shutil.copy2(shot_src, screenshots_dir / f"{stem}.png")
        shutil.copy2(xml_src, a11y_dir / f"{stem}.xml")

        report["success"].append(stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Mobile3M tasks to four-piece samples.")
    parser.add_argument(
        "--task-file",
        default=TASK_FILE,
        help=f"Task JSON under --data-dir (default: TASK_FILE={TASK_FILE})",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=MOBILE3M_SRC,
        help="Mobile3M datasets root",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory (default: Mobile3M_data/)",
    )
    parser.add_argument(
        "--app-prefix",
        default="",
        help="Only process tasks whose name starts with this prefix (e.g. QQmusic)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Keep existing outputs; skip stems that already have step_GT (implies --no-clear)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process at most N tasks (0 = all)",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Do not clear existing outputs before conversion",
    )
    args = parser.parse_args()
    if args.incremental:
        args.no_clear = True

    out_dir = args.out_dir.resolve()
    step_gt_dir = out_dir / "step_GT"
    step_ins_dir = out_dir / "step_instructions"
    screenshots_dir = out_dir / "screenshots"
    a11y_dir = out_dir / "a11y_trees_L0"

    data_dir = args.data_dir.resolve()
    tasks = load_tasks(data_dir, args.task_file)
    if args.app_prefix:
        prefix = args.app_prefix.strip()
        tasks = [t for t in tasks if str(t.get("name") or "").startswith(prefix)]
    if args.limit > 0:
        tasks = tasks[: args.limit]

    for d in (step_gt_dir, step_ins_dir, screenshots_dir, a11y_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not args.no_clear:
        for d, ext in (
            (step_gt_dir, ".json"),
            (step_ins_dir, ".txt"),
            (screenshots_dir, ".png"),
            (a11y_dir, ".xml"),
        ):
            for f in d.glob(f"*{ext}"):
                f.unlink()

    report: dict[str, Any] = {
        "task_file": args.task_file,
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "tasks_total": len(tasks),
        "success": [],
        "skipped": [],
        "skipped_steps": [],
        "skipped_existing": [],
        "failed": [],
    }

    print("=" * 60)
    print("get_Mobile3M.py — Mobile3M → Mobile3M_data 四件套")
    print("=" * 60)
    print(f"  tasks: {len(tasks)} from {args.task_file}")
    if args.app_prefix:
        print(f"  app_prefix: {args.app_prefix}")
    if args.incremental:
        print("  mode: incremental (skip existing step_GT)")
    print(f"  data_dir: {data_dir}")
    print(f"  out_dir:  {out_dir}")
    print("=" * 60)

    for i, task in enumerate(tasks, 1):
        if i % 100 == 0 or i == len(tasks):
            print(f"  processing task {i}/{len(tasks)} ...")
        _convert_task(
            task,
            data_dir,
            report,
            step_gt_dir=step_gt_dir,
            step_ins_dir=step_ins_dir,
            screenshots_dir=screenshots_dir,
            a11y_dir=a11y_dir,
            incremental=args.incremental,
        )

    report_path = out_dir / "conversion_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    n_ok = len(report["success"])
    n_existing = len(report["skipped_existing"])
    print(f"\n{'=' * 60}")
    print(f"完成: 本次新增={n_ok}, 跳过任务={len(report['skipped'])}, "
          f"跳过步={len(report['skipped_steps'])}, "
          f"跳过已有={n_existing}, "
          f"失败步={len(report['failed'])}")
    print(f"  报告: {report_path}")

    counts = {
        "step_GT": len(list(step_gt_dir.glob("*.json"))),
        "step_instructions": len(list(step_ins_dir.glob("*.txt"))),
        "screenshots": len(list(screenshots_dir.glob("*.png"))),
        "a11y_trees_L0": len(list(a11y_dir.glob("*.xml"))),
    }
    print(f"  四件套: {counts}")
    vals = set(counts.values())
    if len(vals) == 1 and (not args.incremental) and counts["step_GT"] == n_ok:
        print(f"\n✅ 四件套数量一致，共 {counts['step_GT']} 个！")
    elif args.incremental and len(vals) == 1 and counts["step_GT"] == n_ok + n_existing:
        print(f"\n✅ 四件套数量一致，共 {counts['step_GT']} 个（已有 {n_existing} + 新增 {n_ok}）！")
    else:
        print("\n⚠️ 四件套数量不一致，请检查 conversion_report.json")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
