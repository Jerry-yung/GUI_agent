#!/usr/bin/env python3
"""Evaluate SMAN single-path results and save checkpoints."""
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
from utils.step_log import (
    action_type,
    is_type_filter_all,
    iter_eval_step_indices,
    normalize_type_filter,
    step_at_round,
)
from utils.task_filter import matches_app_names

# ── Global configuration (align with main.py) ───────────────────────

TASK_TYPE = "single_simple"
MODEL = "qwen-vl-max"
DATA_DIR = "../../datasets/Mobile3M/datasets"
APP_NAMES = ["ximalaya"]
# 评估 GT 动作类型：具体类型列表，或含 ``all`` 合并各类型子目录评测整条任务
TYPE: list[str] = ["click"]
# TYPE: list[str] = ["scroll"]
# TYPE: list[str] = ["all"]
# TYPE: list[str] = ["click", "scroll"]  # 分目录各算一行（不合并）
# 可选：仅评测指定目录（覆盖 MODEL 多 agent 扫描）
RESULT_DIR: str | None = None

_EVALUABLE_GT_TYPES = frozenset({"click", "scroll", "input"})

# ─────────────────────────────────────────────────────────────────────


def _action_type(action_str: str) -> str:
    return action_type(action_str)


def _step_at_round(steps: list[dict], round_no: int) -> dict | None:
    return step_at_round(steps, round_no)


def _strict_action_hit(pred_action_num: str | None, gt_id: int) -> bool:
    return pred_action_num is not None and pred_action_num == str(gt_id)


def _pred_action_num(
    step_index: int,
    steps: list[dict],
    pred_lines: list[str],
) -> str | None:
    step = _step_at_round(steps, step_index + 1)
    if step is not None and step.get("pred_action_id") is not None:
        return str(step["pred_action_id"])
    if step_index < len(pred_lines):
        line = pred_lines[step_index].strip()
        if line and not line.startswith("Task "):
            return line.split(":")[0].strip()
    return None


def _score_task_eval_pairs(
    *,
    stem: str,
    eval_pairs: list[tuple[int, int]],
    steps: list[dict],
    pred_lines: list[str],
    id_to_action: dict[int, str],
    no_finish_list: list[str],
) -> tuple[int, int, int] | None:
    """返回 (true_action, type_true, retrieval_true)；未完成则 None。"""
    if any(_pred_action_num(i, steps, pred_lines) is None for i, _ in eval_pairs):
        no_finish_list.append(stem)
        return None

    true_action_sum = 0
    type_true_action_sum = 0
    retrieval_true_sum = 0

    for i, gt_id in eval_pairs:
        pred_action_num = _pred_action_num(i, steps, pred_lines)
        if pred_action_num is None:
            continue
        step = _step_at_round(steps, i + 1)
        try:
            pred_id = int(pred_action_num)
        except ValueError:
            pred_id = -999

        if pred_id >= 0 and pred_id in id_to_action and gt_id in id_to_action:
            if _action_type(id_to_action[pred_id]) == _action_type(id_to_action[gt_id]):
                type_true_action_sum += 1
        elif pred_id < 0:
            pred_type = ""
            if step and step.get("action"):
                pred_type = str(step["action"]).lower()
            if not pred_type and i < len(pred_lines):
                parts = pred_lines[i].split(":")
                pred_action_name = (
                    parts[1].split(" ")[1] if len(parts) > 1 and parts[1] else ""
                )
                pred_type = _action_type(pred_action_name)
            real_action = id_to_action.get(gt_id, "")
            if pred_type and _action_type(real_action) == pred_type:
                type_true_action_sum += 1

        if _strict_action_hit(pred_action_num, gt_id):
            true_action_sum += 1

        if step and step.get("topk_retrieval_match") is True:
            retrieval_true_sum += 1

    return (
        true_action_sum,
        type_true_action_sum,
        retrieval_true_sum,
    )


def single_path_eval(
    result_path: Path,
    source_data_dir: Path,
    *,
    type_filter: list[str] | None = None,
    app_names: list[str] | None = None,
) -> dict:
    apps = app_names if app_names is not None else APP_NAMES
    sman = get_sman_utils()
    files = list_result_files(result_path)

    success_sum = 0
    gt_action_sum = 0
    all_true_action_sum = 0
    no_finish_list: list[str] = []
    type_true_action_sum = 0
    retrieval_true_sum = 0
    evaluated = 0

    for filename in files:
        payload = load_result_payload(result_path, filename)
        raw = payload.get("result_raw_txt") or ""
        steps = payload.get("steps") or []
        pred_lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if pred_lines and pred_lines[-1].startswith("Task "):
            pred_lines = pred_lines[:-1]

        stem = filename.rsplit(".", 1)[0]
        if "_" in stem and stem.split("_", 1)[0].isdigit():
            task_page_name = stem.split("_", 1)[1]
        else:
            task_page_name = stem
        if not matches_app_names(task_page_name, apps):
            continue

        graph_name = sman.find_graph_dir_for_task(str(source_data_dir), task_page_name)
        if not graph_name:
            sman.print_with_color(f"Skip {filename}: no graph dir for {task_page_name}", "yellow")
            continue

        graph_dir = source_data_dir / graph_name
        try:
            all_action_ids = sman.load_all_action_ids(str(graph_dir))
            action_str_to_id, id_to_action = sman.build_action_id_maps(all_action_ids)
            gt_ids = sman.gt_action_ids_from_task_page(
                task_page_name, str(graph_dir), action_str_to_id
            )
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            sman.print_with_color(f"Skip {filename}: GT load failed ({exc})", "yellow")
            continue

        eval_pairs = iter_eval_step_indices(
            gt_ids, id_to_action, steps, type_filter
        )
        if not eval_pairs:
            continue

        scored = _score_task_eval_pairs(
            stem=stem,
            eval_pairs=eval_pairs,
            steps=steps,
            pred_lines=pred_lines,
            id_to_action=id_to_action,
            no_finish_list=no_finish_list,
        )
        if scored is None:
            continue

        evaluated += 1
        gt_action_sum += len(eval_pairs)
        true_action_sum, type_true, retrieval_true = scored
        all_true_action_sum += true_action_sum
        type_true_action_sum += type_true
        retrieval_true_sum += retrieval_true
        if true_action_sum == len(eval_pairs):
            success_sum += 1

    task_num = evaluated or 1
    allowed = normalize_type_filter(type_filter)
    return {
        "task_num": evaluated,
        "app_names": list(apps),
        "type_filter": sorted(allowed) if allowed is not None else ["all"],
        "success_sum": success_sum,
        "success_rate": success_sum / task_num if evaluated else 0.0,
        "action_acc": all_true_action_sum / gt_action_sum if gt_action_sum else 0.0,
        "type_acc": type_true_action_sum / gt_action_sum if gt_action_sum else 0.0,
        "gt_action_sum": gt_action_sum,
        "all_true_action_sum": all_true_action_sum,
        "type_true_action_sum": type_true_action_sum,
        "retrieval_true_sum": retrieval_true_sum,
        "mean_retrieval": retrieval_true_sum / gt_action_sum if gt_action_sum else 0.0,
        "no_finish": no_finish_list,
        "no_finish_count": len(no_finish_list),
        "incomplete_task_count": 0,
    }


def discover_typed_result_subdirs(agent_base: Path) -> list[str]:
    """``agent_base`` 下含结果文件的类型子目录名（click / scroll / …）。"""
    if not agent_base.is_dir():
        return []
    subs: list[str] = []
    for sub in sorted(agent_base.iterdir()):
        if sub.is_dir() and list_result_files(sub):
            subs.append(sub.name)
    return subs


def _task_stems_across_types(agent_base: Path, typed_subdirs: list[str]) -> list[str]:
    stems: set[str] = set()
    for sub in typed_subdirs:
        for filename in list_result_files(agent_base / sub):
            stems.add(filename.rsplit(".", 1)[0])
    return sorted(stems)


def _load_merged_task_steps(
    agent_base: Path,
    typed_subdirs: list[str],
    stem: str,
) -> tuple[list[dict], list[str]] | None:
    by_round: dict[int, dict] = {}
    for sub in typed_subdirs:
        subdir = agent_base / sub
        filename = None
        for name in list_result_files(subdir):
            if name.rsplit(".", 1)[0] == stem:
                filename = name
                break
        if filename is None:
            continue
        payload = load_result_payload(subdir, filename)
        for step in payload.get("steps") or []:
            round_no = int(step.get("round") or 0)
            if round_no > 0:
                by_round[round_no] = step
    if not by_round:
        return None
    steps = [by_round[r] for r in sorted(by_round)]
    pred_lines: list[str] = []
    for r in sorted(by_round):
        step = by_round[r]
        aid = step.get("pred_action_id")
        info = step.get("pred_action_info") or step.get("action_info") or ""
        page = step.get("page") or ""
        if aid is not None:
            pred_lines.append(f"{aid}: {info}: {page}")
    return steps, pred_lines


def _task_complete_for_all_eval(
    gt_ids: list[int],
    id_to_action: dict[int, str],
    steps_by_round: dict[int, dict],
) -> bool:
    """整条任务在 click/scroll/input 各步均有非 skip 记录，否则不计入分母。"""
    for i, gt_id in enumerate(gt_ids):
        gt_t = _action_type(id_to_action.get(gt_id, ""))
        if gt_t not in _EVALUABLE_GT_TYPES:
            continue
        step = steps_by_round.get(i + 1)
        if step is None or step.get("is_skip"):
            return False
    return True


def single_path_eval_merged(
    agent_base: Path,
    typed_subdirs: list[str],
    source_data_dir: Path,
    *,
    type_filter: list[str] | None = None,
    app_names: list[str] | None = None,
) -> dict:
    """合并各类型子目录，按整条任务算 SR / action_acc（缺步任务跳过）。"""
    apps = app_names if app_names is not None else APP_NAMES
    sman = get_sman_utils()
    success_sum = 0
    gt_action_sum = 0
    all_true_action_sum = 0
    no_finish_list: list[str] = []
    incomplete_list: list[str] = []
    type_true_action_sum = 0
    retrieval_true_sum = 0
    evaluated = 0

    for stem in _task_stems_across_types(agent_base, typed_subdirs):
        merged = _load_merged_task_steps(agent_base, typed_subdirs, stem)
        if merged is None:
            continue
        steps, pred_lines = merged
        steps_by_round = {
            int(s.get("round") or 0): s for s in steps if int(s.get("round") or 0) > 0
        }

        if "_" in stem and stem.split("_", 1)[0].isdigit():
            task_page_name = stem.split("_", 1)[1]
        else:
            task_page_name = stem
        if not matches_app_names(task_page_name, apps):
            continue

        graph_name = sman.find_graph_dir_for_task(str(source_data_dir), task_page_name)
        if not graph_name:
            sman.print_with_color(f"Skip {stem}: no graph dir for {task_page_name}", "yellow")
            continue

        graph_dir = source_data_dir / graph_name
        try:
            all_action_ids = sman.load_all_action_ids(str(graph_dir))
            action_str_to_id, id_to_action = sman.build_action_id_maps(all_action_ids)
            gt_ids = sman.gt_action_ids_from_task_page(
                task_page_name, str(graph_dir), action_str_to_id
            )
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as exc:
            sman.print_with_color(f"Skip {stem}: GT load failed ({exc})", "yellow")
            continue

        if not _task_complete_for_all_eval(gt_ids, id_to_action, steps_by_round):
            incomplete_list.append(stem)
            continue

        eval_pairs = iter_eval_step_indices(
            gt_ids, id_to_action, steps, type_filter
        )
        if not eval_pairs:
            continue

        scored = _score_task_eval_pairs(
            stem=stem,
            eval_pairs=eval_pairs,
            steps=steps,
            pred_lines=pred_lines,
            id_to_action=id_to_action,
            no_finish_list=no_finish_list,
        )
        if scored is None:
            continue

        evaluated += 1
        gt_action_sum += len(eval_pairs)
        true_action_sum, type_true, retrieval_true = scored
        all_true_action_sum += true_action_sum
        type_true_action_sum += type_true
        retrieval_true_sum += retrieval_true
        if true_action_sum == len(eval_pairs):
            success_sum += 1

    task_num = evaluated or 1
    return {
        "task_num": evaluated,
        "app_names": list(apps),
        "type_filter": ["all"],
        "success_sum": success_sum,
        "success_rate": success_sum / task_num if evaluated else 0.0,
        "action_acc": all_true_action_sum / gt_action_sum if gt_action_sum else 0.0,
        "type_acc": type_true_action_sum / gt_action_sum if gt_action_sum else 0.0,
        "gt_action_sum": gt_action_sum,
        "all_true_action_sum": all_true_action_sum,
        "type_true_action_sum": type_true_action_sum,
        "retrieval_true_sum": retrieval_true_sum,
        "mean_retrieval": retrieval_true_sum / gt_action_sum if gt_action_sum else 0.0,
        "no_finish": no_finish_list,
        "no_finish_count": len(no_finish_list),
        "incomplete_task_count": len(incomplete_list),
        "incomplete_tasks": incomplete_list,
    }


def _result_dir_prefix(task_type: str, agent: str) -> str:
    return f"{task_type}_{agent}_"


def _result_dir_has_files(path: Path) -> bool:
    if list_result_files(path):
        return True
    if not path.is_dir():
        return False
    return any(sub.is_dir() and list_result_files(sub) for sub in path.iterdir())


def discover_agents_for_model(task_type: str, model_slug: str) -> list[str]:
    """扫描 results/ 下存在结果的 agent：``{task_type}_{agent}_{model_slug}/``。"""
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
    """从结果目录名解析 agent（支持指向 ``.../click`` 子目录）。"""
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


def discover_result_dirs(task_type: str, agent: str) -> list[tuple[str, Path]]:
    """扫描 results/ 下所有 ``{task_type}_{agent}_{model_slug}`` 目录（含结果文件）。"""
    prefix = _result_dir_prefix(task_type, agent)
    found: list[tuple[str, Path]] = []
    if not RESULTS_DIR.is_dir():
        return found
    for path in sorted(RESULTS_DIR.iterdir()):
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        model_slug = path.name[len(prefix) :]
        if not model_slug:
            continue
        if list_result_files(path):
            found.append((model_slug, path))
            continue
        for sub in sorted(path.iterdir()):
            if sub.is_dir() and list_result_files(sub):
                found.append((f"{model_slug}/{sub.name}", sub))
    return found


def discover_eval_dirs(
    task_type: str,
    agent: str,
    model_slug: str,
    *,
    type_filter: list[str] | None = None,
) -> list[tuple[str, Path]]:
    """Resolve eval directories under ``results/{task_type}_{agent}_{model_slug}/``."""
    base = result_dir(task_type, agent, model_slug)
    if is_type_filter_all(type_filter):
        subs = discover_typed_result_subdirs(base)
        if subs:
            return [(model_slug, base)]
        return []

    allowed = normalize_type_filter(type_filter)
    entries: list[tuple[str, Path]] = []

    if allowed:
        for gt_type in sorted(allowed):
            sub = base / gt_type
            if sub.is_dir() and list_result_files(sub):
                entries.append((f"{model_slug}/{gt_type}", sub))
        if entries:
            return entries
        if base.is_dir() and list_result_files(base):
            return [(model_slug, base)]
        return entries

    if base.is_dir() and list_result_files(base):
        entries.append((model_slug, base))
    for sub in sorted(base.iterdir()) if base.is_dir() else []:
        if sub.is_dir() and list_result_files(sub):
            entries.append((f"{model_slug}/{sub.name}", sub))
    return entries


def discover_model_slugs(task_type: str, agent: str) -> list[str]:
    prefix = _result_dir_prefix(task_type, agent)
    slugs: list[str] = []
    if not RESULTS_DIR.is_dir():
        return slugs
    for path in sorted(RESULTS_DIR.iterdir()):
        if path.is_dir() and path.name.startswith(prefix):
            slug = path.name[len(prefix) :]
            if slug:
                slugs.append(slug)
    return slugs


def _model_slug_from_result_path(rdir: Path, task_type: str, agent: str) -> str:
    prefix = _result_dir_prefix(task_type, agent)
    name = rdir.name
    if name.startswith(prefix):
        return name[len(prefix) :]
    return name


def format_eval_summary(report: dict) -> str:
    task_num = report.get("task_num", 0)
    success_sum = report.get("success_sum", 0)
    true_sum = report.get("all_true_action_sum")
    gt_sum = report.get("gt_action_sum")
    type_true_sum = report.get("type_true_action_sum")
    retrieval_true = report.get("retrieval_true_sum")
    lines = [
        f"task_num: {task_num}",
        f"success_rate: {report.get('success_rate', 0):.4f} ( {success_sum} / {task_num} )",
        f"action_acc: {report.get('action_acc', 0):.4f} ( {true_sum} / {gt_sum} )",
        f"type_acc: {report.get('type_acc', 0):.4f} ( {type_true_sum} / {gt_sum} )",
    ]
    if report.get("agent") != "AppAgent":
        lines.append(
            f"Mean_Retrieval: {report.get('mean_retrieval', 0):.4f} ( {retrieval_true} / {gt_sum} )"
        )
    incomplete = report.get("incomplete_task_count", 0)
    if incomplete:
        lines.append(
            f"incomplete_tasks (skipped from SR/action/type denom): {incomplete}"
        )
    lines.append(f"no_finish ({report.get('no_finish_count', 0)}): {report.get('no_finish', [])}")
    return "\n".join(lines)


_TABLE_COLUMNS: list[tuple[str, str, int]] = [
    ("agent", "agent", 24),
    ("task_num", "tasks", 6),
    ("success_rate", "success", 9),
    ("action_acc", "action", 9),
    ("type_acc", "type", 9),
    ("mean_retrieval", "retrieval", 9),
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
            if key == "mean_retrieval" and report.get("agent") == "AppAgent":
                text = "n/a"
                cells.append(text.rjust(width))
                continue
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


def _collect_agent_eval_entries(
    *,
    task_type: str,
    model_slug: str,
    type_filter: list[str] | None,
    result_dir_override: Path | None,
) -> list[tuple[str, Path]]:
    """(agent, eval_dir) 列表。``all`` 时为 agent 根目录；否则为各 TYPE 子目录。"""
    if result_dir_override is not None:
        agent = parse_agent_from_result_path(result_dir_override, task_type, model_slug)
        if is_type_filter_all(type_filter):
            base = result_dir_override
            if not discover_typed_result_subdirs(base) and base.parent.is_dir():
                base = base.parent
            return [(agent, base)]
        return [(agent, result_dir_override)]

    entries: list[tuple[str, Path]] = []
    for agent in discover_agents_for_model(task_type, model_slug):
        if is_type_filter_all(type_filter):
            base = result_dir(task_type, agent, model_slug)
            if discover_typed_result_subdirs(base):
                entries.append((agent, base))
            continue
        dirs = discover_eval_dirs(task_type, agent, model_slug, type_filter=type_filter)
        if not dirs:
            continue
        for _label, rdir in dirs:
            entries.append((agent, rdir))
    return entries


def main() -> None:
    ensure_sman_path()
    sman = get_sman_utils()
    data_path = resolve_data_dir(DATA_DIR)
    timestamp = datetime.now(timezone.utc).isoformat()

    result_override = Path(RESULT_DIR).resolve() if RESULT_DIR else None
    if result_override is not None and not result_override.is_dir():
        print(f"Result directory not found: {result_override}")
        raise SystemExit(1)

    agent_entries = _collect_agent_eval_entries(
        task_type=TASK_TYPE,
        model_slug=MODEL,
        type_filter=TYPE,
        result_dir_override=result_override,
    )
    if not agent_entries:
        print(f"No result directories found under {RESULTS_DIR}")
        print(
            f"  (expected: {TASK_TYPE}_<agent>_{MODEL}/{{click,scroll,...}}/ "
            f"for agents in {discover_agents_for_model(TASK_TYPE, MODEL)!r})"
        )
        raise SystemExit(1)

    merge_all = is_type_filter_all(TYPE)
    type_label = "all" if merge_all else ",".join(TYPE) if TYPE else "all"
    sman.print_with_color(
        f"Evaluating model={MODEL} task_type={TASK_TYPE} "
        f"type={type_label} app_names={APP_NAMES} "
        f"agents={len({a for a, _ in agent_entries})} dir(s)={len(agent_entries)}",
        "cyan",
    )

    reports: list[dict] = []
    for agent, rdir in agent_entries:
        if merge_all:
            typed_subdirs = discover_typed_result_subdirs(rdir)
            report = single_path_eval_merged(
                rdir,
                typed_subdirs,
                data_path,
                type_filter=TYPE,
                app_names=APP_NAMES,
            )
            result_dir_label = f"{rdir}/{{{','.join(typed_subdirs)}}}"
        else:
            report = single_path_eval(rdir, data_path, type_filter=TYPE, app_names=APP_NAMES)
            result_dir_label = str(rdir)
        report.update(
            {
                "agent": agent,
                "model": MODEL,
                "result_dir": result_dir_label,
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
        "type_filter": TYPE,
        "data_dir": str(data_path),
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
