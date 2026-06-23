#!/usr/bin/env python3
"""Run SMAN-Bench multi-path inference with m2 / TO / AppAgent agents.

流水线（m2/TO）：
  llm_TO → action_type + target_object → 条件 Top-K 检索
  VLM → 固定 type，仅填 element/direction/text 等字段
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import load_agent
from llm_set.llm import get_vlm_model_name, slug_for_run_filename
from utils.mobile3m_io import load_tasks
from utils.paths import ensure_cache_dirs, resolve_data_dir, result_dir
from utils.result_io import write_result_json
from utils.sman_bridge import apply_action
from utils.sman_setup import ensure_sman_path, get_sman_utils
from utils.step_log import format_pred_vlm_display, print_step_multipath
from utils.task_context import get_multipath_step_instruction, load_task_context
from utils.task_filter import filter_tasks

# ── Global configuration ─────────────────────────────────────────────

AGENT = "m2"
# AGENT = "to"
# AGENT = "AppAgent"

TASK_JSON = "simple_normal_tasks.json"
# TASK_JSON = "complex_normal_tasks.json"
TASK_TYPE = "multi_simple"
# TASK_TYPE = "multi_complex"
TEST_START = 4
TEST_END = 5  # -1 = all after START
APP_NAMES = ["ximalaya"]
TOP_K = 5
DATA_DIR = "../../datasets/Mobile3M/datasets"
DRY_RUN = False
REQUEST_INTERVAL = 0.0
MAX_ROUNDS = 20 if TASK_TYPE == "multi_simple" else 25

# ─────────────────────────────────────────────────────────────────────


def _load_max_rounds() -> int:
    cfg_path = ROOT / "config.yaml"
    if not cfg_path.is_file():
        return MAX_ROUNDS
    try:
        with cfg_path.open(encoding="utf-8") as fp:
            cfg = yaml.safe_load(fp) or {}
        return int(cfg.get("max_rounds", MAX_ROUNDS))
    except (TypeError, ValueError, yaml.YAMLError):
        return MAX_ROUNDS


def _record_step(
    *,
    round_count: int,
    max_rounds: int,
    step_page: str,
    next_page: str,
    step_instruction: str,
    step_record: dict[str, Any],
    steps: list[dict[str, Any]],
    pred_id,
    pred_info: str,
    assets=None,
    pred_res: list[str] | None = None,
) -> None:
    pred_disp = format_pred_vlm_display(
        pred_info,
        pred_res,
        assets=assets,
        pred_id=pred_id,
        id_to_action=None,
        all_action_ids=None,
    )
    target_object = getattr(assets, "target_object", None) if assets is not None else None
    llm_action_type = (
        step_record.get("llm_action_type")
        or (getattr(assets, "llm_action_type", None) if assets is not None else None)
    )

    print_step_multipath(
        round_count,
        max_rounds,
        step_page,
        step_instruction,
        pred_disp,
        next_page,
        llm_action_type=llm_action_type,
        target_object=target_object,
    )

    step_update: dict[str, Any] = {
        "page": step_page,
        "next_page": next_page,
        "target_object": target_object,
        "llm_action_type": llm_action_type,
        "llm_to_raw": getattr(assets, "llm_to_raw", None) if assets else step_record.get("llm_to_raw"),
        "pred_action_id": pred_id,
        "pred_action_info": pred_disp,
    }
    step_record.update(step_update)
    steps.append(step_record)
    return pred_disp


def _record_agent_step_history(
    agent,
    *,
    step_page: str,
    next_page: str,
    step_instruction: str,
    pred_disp: str,
    assets=None,
) -> None:
    if not hasattr(agent, "record_step_outcome"):
        return
    agent.record_step_outcome(
        step_page=step_page,
        next_page=next_page,
        step_instruction=step_instruction,
        llm_action_type=(
            getattr(assets, "llm_action_type", None) if assets is not None else None
        )
        or "",
        target_object=(
            getattr(assets, "target_object", None) if assets is not None else None
        )
        or "",
        pred=pred_disp,
    )


def run_multi_task(
    agent,
    task: dict,
    task_index: int,
    data_dir: Path,
    out_dir: Path,
    *,
    task_progress: int,
    task_total: int,
    max_rounds: int,
    dry_run: bool = False,
) -> bool:
    sman = get_sman_utils()
    ctx = load_task_context(task, data_dir, task_index=task_index)
    if ctx is None:
        return False

    if hasattr(agent, "begin_task"):
        agent.begin_task(ctx)

    final_page_name = ctx.final_page_name
    current_page_name = final_page_name.split("_")[0]
    round_count = 0
    task_complete = False

    ans_action_id: list = []
    ans_action_info: list = []
    ans_history_pages: list = []
    steps: list[dict] = []
    top_k = getattr(agent, "top_k", TOP_K)

    task_title = (
        f"Task [{task_progress}/{task_total}] {task_index} {final_page_name} "
        f"(max {max_rounds} rounds)"
    )
    sman.print_with_color(f"\n====== {task_title} ======", "cyan")

    while round_count < max_rounds and not task_complete:
        round_count += 1
        step_instruction = get_multipath_step_instruction(ctx, current_page_name)
        step_page = current_page_name

        t0 = time.perf_counter()
        assets = agent.prepare_round(
            ctx,
            current_page_name,
            step_instruction,
            dry_run=dry_run,
        )
        if assets is None:
            sman.print_with_color(f"prepare_round failed: {current_page_name}", "red")
            step_record: dict[str, Any] = {
                "round": round_count,
                "step_instruction": step_instruction,
                "error": "prepare_error",
            }
            pred_disp = _record_step(
                round_count=round_count,
                max_rounds=max_rounds,
                step_page=step_page,
                next_page=current_page_name,
                step_instruction=step_instruction,
                step_record=step_record,
                steps=steps,
                pred_id=-2,
                pred_info="prepare_error",
            )
            _record_agent_step_history(
                agent,
                step_page=step_page,
                next_page=current_page_name,
                step_instruction=step_instruction,
                pred_disp=pred_disp,
            )
            ans_action_id.append(-2)
            ans_action_info.append("prepare_error")
            ans_history_pages.append(current_page_name)
            if REQUEST_INTERVAL:
                time.sleep(REQUEST_INTERVAL)
            continue

        ok, res, thought, vlm_stats = agent.decide(
            ctx,
            assets,
            step_instruction,
            dry_run=dry_run,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        step_record = {
            "round": round_count,
            "step_instruction": step_instruction,
            "llm_action_type": assets.llm_action_type,
            "llm_to_raw": assets.llm_to_raw,
            "llm_routing_fallback": assets.llm_routing_fallback,
            "thought": thought,
            "summary": getattr(agent, "last_vlm_summary", "") or "",
            "vlm_input_tokens": vlm_stats.input_tokens,
            "vlm_output_tokens": vlm_stats.output_tokens,
            "vlm_elapsed_ms": vlm_stats.vlm_elapsed_ms,
            "elapsed_ms": elapsed_ms,
        }

        if not ok:
            step_record["error"] = "vlm_error"
            pred_disp = _record_step(
                round_count=round_count,
                max_rounds=max_rounds,
                step_page=step_page,
                next_page=current_page_name,
                step_instruction=step_instruction,
                step_record=step_record,
                steps=steps,
                pred_id=-2,
                pred_info="vlm_error",
                assets=assets,
            )
            _record_agent_step_history(
                agent,
                step_page=step_page,
                next_page=current_page_name,
                step_instruction=step_instruction,
                pred_disp=pred_disp,
                assets=assets,
            )
            ans_action_id.append(-2)
            ans_action_info.append("vlm_error")
            ans_history_pages.append(current_page_name)
            if REQUEST_INTERVAL:
                time.sleep(REQUEST_INTERVAL)
            continue

        if res is None:
            step_record["error"] = "parse_error"
            pred_disp = _record_step(
                round_count=round_count,
                max_rounds=max_rounds,
                step_page=step_page,
                next_page=current_page_name,
                step_instruction=step_instruction,
                step_record=step_record,
                steps=steps,
                pred_id=-2,
                pred_info="parse_error",
                assets=assets,
            )
            _record_agent_step_history(
                agent,
                step_page=step_page,
                next_page=current_page_name,
                step_instruction=step_instruction,
                pred_disp=pred_disp,
                assets=assets,
            )
            ans_action_id.append(-2)
            ans_action_info.append("parse_error")
            ans_history_pages.append(current_page_name)
            if REQUEST_INTERVAL:
                time.sleep(REQUEST_INTERVAL)
            continue

        act_name = res[0]
        step_record["action"] = act_name

        new_page, action_id, action_info, aid_list, ainfo_list = apply_action(
            res,
            assets,
            ctx.all_action_ids,
            current_page_name,
            ctx.all_page_convert,
        )
        del aid_list, ainfo_list
        step_record["action_id"] = action_id
        step_record["action_info"] = action_info

        current_page_name = new_page
        pred_disp = _record_step(
            round_count=round_count,
            max_rounds=max_rounds,
            step_page=step_page,
            next_page=current_page_name,
            step_instruction=step_instruction,
            step_record=step_record,
            steps=steps,
            pred_id=action_id,
            pred_info=action_info,
            assets=assets,
            pred_res=res,
        )
        _record_agent_step_history(
            agent,
            step_page=step_page,
            next_page=current_page_name,
            step_instruction=step_instruction,
            pred_disp=pred_disp,
            assets=assets,
        )
        ans_action_id.append(action_id)
        ans_action_info.append(action_info)
        ans_history_pages.append(current_page_name)

        if current_page_name == final_page_name:
            task_complete = True

        if REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL)

    write_result_json(
        out_dir,
        final_page_name,
        task_index,
        ans_action_id=ans_action_id,
        ans_action_info=ans_action_info,
        ans_history_pages=ans_history_pages,
        task_complete=task_complete,
        round_count=round_count,
        max_rounds=max_rounds,
        steps=steps,
        task_description=task.get("task", ""),
    )
    return True


def main() -> None:
    ensure_sman_path()
    ensure_cache_dirs()
    data_path = resolve_data_dir(DATA_DIR)
    tasks = load_tasks(data_path, TASK_JSON)
    selected = filter_tasks(tasks, APP_NAMES, TEST_START, TEST_END)
    max_rounds = _load_max_rounds()

    model_slug = slug_for_run_filename(get_vlm_model_name())
    base_out_dir = result_dir(TASK_TYPE, AGENT, model_slug)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    agent = load_agent(AGENT, top_k=TOP_K)
    print(
        f"Agent={AGENT} model={model_slug} tasks={len(selected)} "
        f"task_type={TASK_TYPE} max_rounds={max_rounds} -> {base_out_dir}/"
    )

    task_total = len(selected)
    processed = 0
    for task_progress, (task_index, task) in enumerate(selected, start=1):
        if run_multi_task(
            agent,
            task,
            task_index,
            data_path,
            base_out_dir,
            task_progress=task_progress,
            task_total=task_total,
            max_rounds=max_rounds,
            dry_run=DRY_RUN,
        ):
            processed += 1

    print(f"Done. Processed {processed} tasks.")


if __name__ == "__main__":
    main()
