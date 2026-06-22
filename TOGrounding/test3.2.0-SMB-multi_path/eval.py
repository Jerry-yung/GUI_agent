#!/usr/bin/env python3
"""Evaluate SMAN multi-path results and save checkpoints."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.paths import CHECKPOINTS_DIR, RESULTS_DIR, resolve_data_dir, result_dir
from utils.result_io import list_result_files, load_result_payload
from utils.sman_setup import ensure_sman_path, get_sman_utils
from utils.task_filter import matches_app_names

# ── Global configuration (align with main.py) ───────────────────────

TASK_TYPE = "multi_simple"
# TASK_TYPE = "multi_complex"
MODEL = "qwen-vl-max"
DATA_DIR = "../../datasets/Mobile3M/datasets"
APP_NAMES = ["ximalaya"]
# simple: max 19 steps; complex: max 24 steps (official multi_path_test)
EVAL_TASK_TYPE = "simple"
# EVAL_TASK_TYPE = "complex"
RESULT_DIR: str | None = None

_MAX_ROUNDS_BY_EVAL_TYPE = {"simple": 19, "complex": 24}

# ─────────────────────────────────────────────────────────────────────


def _task_page_name_from_stem(stem: str) -> str:
    if "_" in stem and stem.split("_", 1)[0].isdigit():
        return stem.split("_", 1)[1]
    return stem


def _real_path_from_task_page(task_page_name: str) -> list[str]:
    return task_page_name.split("_")[1:]


def _parse_pred_trajectory(
    raw: str,
    *,
    final_page_name: str,
    eval_task_type: str,
) -> tuple[list[str], list[str]]:
    """Return (pred_lines, pred_path_suffix) from result_raw_txt."""
    max_rounds = _MAX_ROUNDS_BY_EVAL_TYPE.get(eval_task_type, 19)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if lines and lines[-1].startswith("Task "):
        lines = lines[:-1]

    pred_paths: list[str] = []
    for i, line in enumerate(lines):
        pred_paths.append(line)
        page_field = line.split(":")[-1].strip()
        if i == max_rounds - 1 or page_field == final_page_name:
            break

    if not pred_paths:
        return [], []

    last_page = pred_paths[-1].split(":")[-1].strip()
    pred_path = last_page.split("_")[1:] if "_" in last_page else []
    return pred_paths, pred_path


def multi_path_eval(
    result_path: Path,
    *,
    eval_task_type: str = "simple",
    app_names: list[str] | None = None,
) -> dict:
    apps = app_names if app_names is not None else APP_NAMES
    files = list_result_files(result_path)

    success_sum = 0
    gt_action_sum = 0
    all_true_action_sum = 0
    pred_action_sum = 0
    no_finish_list: list[str] = []
    evaluated = 0

    for filename in files:
        payload = load_result_payload(result_path, filename)
        raw = payload.get("result_raw_txt") or ""
        stem = filename.rsplit(".", 1)[0]
        task_page_name = _task_page_name_from_stem(stem)
        if not matches_app_names(task_page_name, apps):
            continue
        final_page_name = task_page_name
        real_path = _real_path_from_task_page(task_page_name)

        pred_paths, pred_path = _parse_pred_trajectory(
            raw,
            final_page_name=final_page_name,
            eval_task_type=eval_task_type,
        )
        if not pred_paths:
            no_finish_list.append(stem)
            continue

        evaluated += 1
        gt_action_sum += len(real_path)
        pred_action_sum += len(pred_paths)

        true_action_sum = 0
        if len(real_path) >= len(pred_path):
            for i, path in enumerate(pred_path):
                if path == real_path[i]:
                    true_action_sum += 1
            if true_action_sum == len(real_path):
                success_sum += 1
        else:
            for i, path in enumerate(real_path):
                if i < len(pred_path) and path == pred_path[i]:
                    true_action_sum += 1

        all_true_action_sum += true_action_sum

    task_num = evaluated or 1
    avg_pred_steps = pred_action_sum / task_num if evaluated else 0.0
    avg_gt_steps = gt_action_sum / task_num if evaluated else 0.0
    se = avg_pred_steps / avg_gt_steps if avg_gt_steps else 0.0
    return {
        "task_num": evaluated,
        "app_names": list(apps),
        "eval_task_type": eval_task_type,
        "success_sum": success_sum,
        "success_rate": success_sum / task_num if evaluated else 0.0,
        "action_acc": all_true_action_sum / gt_action_sum if gt_action_sum else 0.0,
        "SE": se,
        "avg_pred_steps": avg_pred_steps,
        "avg_gt_steps": avg_gt_steps,
        "gt_action_sum": gt_action_sum,
        "all_true_action_sum": all_true_action_sum,
        "pred_action_sum": pred_action_sum,
        "no_finish": no_finish_list,
        "no_finish_count": len(no_finish_list),
    }


def _result_dir_prefix(task_type: str, agent: str) -> str:
    return f"{task_type}_{agent}_"


def _result_dir_has_files(path: Path) -> bool:
    return bool(list_result_files(path))


def discover_agents_for_model(task_type: str, model_slug: str) -> list[str]:
    suffix = f"_{model_slug}"
    prefix = f"{task_type}_"
    agents: list[str] = []
    if not RESULTS_DIR.is_dir():
        return agents
    for path in sorted(RESULTS_DIR.iterdir()):
        if not path.is_dir():
            continue
        name = path.name
        if not name.startswith(prefix) or not name.endswith(suffix):
            continue
        agent = name[len(prefix) : -len(suffix)]
        if agent and _result_dir_has_files(path):
            agents.append(agent)
    return sorted(agents)


def parse_agent_from_result_path(path: Path, task_type: str, model_slug: str) -> str:
    candidates = [path]
    if path.parent != path and path.parent.is_dir():
        candidates.append(path.parent)
    suffix = f"_{model_slug}"
    prefix = f"{task_type}_"
    for candidate in candidates:
        name = candidate.name
        if name.startswith(prefix) and name.endswith(suffix):
            agent = name[len(prefix) : -len(suffix)]
            if agent:
                return agent
    return path.name


def _collect_agent_eval_entries(
    *,
    task_type: str,
    model_slug: str,
    result_dir_override: Path | None,
) -> list[tuple[str, Path]]:
    if result_dir_override is not None:
        agent = parse_agent_from_result_path(result_dir_override, task_type, model_slug)
        return [(agent, result_dir_override)]

    entries: list[tuple[str, Path]] = []
    for agent in discover_agents_for_model(task_type, model_slug):
        base = result_dir(task_type, agent, model_slug)
        if _result_dir_has_files(base):
            entries.append((agent, base))
    return entries


def format_eval_summary(report: dict) -> str:
    task_num = report.get("task_num", 0)
    success_sum = report.get("success_sum", 0)
    true_sum = report.get("all_true_action_sum")
    gt_sum = report.get("gt_action_sum")
    avg_pred = report.get("avg_pred_steps", 0)
    avg_gt = report.get("avg_gt_steps", 0)
    lines = [
        f"task_num: {task_num}",
        f"eval_task_type: {report.get('eval_task_type', 'simple')}",
        f"success_rate: {report.get('success_rate', 0):.4f} ( {success_sum} / {task_num} )",
        f"action_acc: {report.get('action_acc', 0):.4f} ( path segments {true_sum} / {gt_sum} )",
        (
            f"SE: {report.get('SE', 0):.4f} "
            f"( avg_pred_steps: {avg_pred:.4g} / gt_steps: {avg_gt:.4g} )"
        ),
        f"no_finish ({report.get('no_finish_count', 0)}): {report.get('no_finish', [])}",
    ]
    return "\n".join(lines)


_TABLE_COLUMNS: list[tuple[str, str, int]] = [
    ("agent", "agent", 24),
    ("task_num", "tasks", 6),
    ("success_rate", "success", 9),
    ("action_acc", "action", 9),
    ("SE", "SE", 9),
    ("no_finish_count", "no_fin", 6),
]


def format_results_table(reports: list[dict]) -> str:
    if not reports:
        return "(no results)"

    headers = [hdr for _, hdr, _ in _TABLE_COLUMNS]
    widths = [w for _, _, w in _TABLE_COLUMNS]

    def row_cells(report: dict) -> list[str]:
        cells: list[str] = []
        for key, _, width in _TABLE_COLUMNS:
            val = report.get(key, "")
            if key == "agent":
                val = report.get("agent") or val
            if isinstance(val, float):
                text = f"{val:.4f}"
            else:
                text = str(val)
            cells.append(text[:width].ljust(width) if key == "agent" else text.rjust(width))
        return cells

    sep = "-+-".join("-" * w for w in widths)
    header_line = " | ".join(h.rjust(w) if i else h.ljust(w) for i, (h, w) in enumerate(zip(headers, widths)))
    body_lines = [" | ".join(cells) for cells in (row_cells(r) for r in reports)]
    return "\n".join([header_line, sep, *body_lines])


def main() -> None:
    ensure_sman_path()
    sman = get_sman_utils()
    timestamp = datetime.now(timezone.utc).isoformat()

    result_override = Path(RESULT_DIR).resolve() if RESULT_DIR else None
    if result_override is not None and not result_override.is_dir():
        print(f"Result directory not found: {result_override}")
        raise SystemExit(1)

    agent_entries = _collect_agent_eval_entries(
        task_type=TASK_TYPE,
        model_slug=MODEL,
        result_dir_override=result_override,
    )
    if not agent_entries:
        print(f"No result directories found under {RESULTS_DIR}")
        print(
            f"  (expected: {TASK_TYPE}_<agent>_{MODEL}/ "
            f"for agents in {discover_agents_for_model(TASK_TYPE, MODEL)!r})"
        )
        raise SystemExit(1)

    sman.print_with_color(
        f"Evaluating model={MODEL} task_type={TASK_TYPE} "
        f"eval_task_type={EVAL_TASK_TYPE} app_names={APP_NAMES} "
        f"agents={len({a for a, _ in agent_entries})}",
        "cyan",
    )

    reports: list[dict] = []
    for agent, rdir in agent_entries:
        report = multi_path_eval(
            rdir,
            eval_task_type=EVAL_TASK_TYPE,
            app_names=APP_NAMES,
        )
        report.update(
            {
                "agent": agent,
                "model": MODEL,
                "result_dir": str(rdir),
            }
        )
        reports.append(report)

    print()
    print(format_results_table(reports))
    print()

    for report in reports:
        sman.print_with_color(f"--- {report['agent']} ({report['result_dir']}) ---", "yellow")
        print(format_eval_summary(report))
        print()

    summary = {
        "task_type": TASK_TYPE,
        "model": MODEL,
        "app_names": APP_NAMES,
        "eval_task_type": EVAL_TASK_TYPE,
        "data_dir": str(resolve_data_dir(DATA_DIR)),
        "timestamp": timestamp,
        "agent_count": len({r["agent"] for r in reports}),
        "agents": reports,
    }

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_name = f"eval_{TASK_TYPE}_{MODEL}_agents_{ts}.json"
    out_path = CHECKPOINTS_DIR / out_name
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    sman.print_with_color(f"Checkpoint 已保存: {out_path}", "green")


if __name__ == "__main__":
    main()
