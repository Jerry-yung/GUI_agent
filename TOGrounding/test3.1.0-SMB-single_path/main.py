#!/usr/bin/env python3
"""Run SMAN-Bench single-path inference with m2 / TO / AppAgent agents.

流水线（m2/TO）：
  llm_TO → action_type + target_object → 条件 Top-K 检索
  VLM → 固定 type，仅填 element/direction/text 等字段
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents import load_agent
from llm_set.llm import get_vlm_model_name, slug_for_run_filename
from utils.mobile3m_io import load_tasks
from utils.paths import ensure_cache_dirs, resolve_data_dir, result_dir
from utils.result_io import write_typed_result_slices
from utils.sman_bridge import apply_action
from utils.sman_setup import ensure_sman_path, get_sman_utils
from utils.step_log import (
    action_match,
    action_type,
    build_step_displays,
    format_gt_vlm_display,
    gt_similarity_rank,
    gt_type_allowed,
    print_step_compare,
    print_step_skip,
    topk_retrieval_match,
    type_match,
)
from utils.task_context import get_step_instruction, load_task_context
from utils.task_filter import filter_tasks

# ── Global configuration ─────────────────────────────────────────────

AGENT = "m2"
# AGENT = "to"   # click/scroll 用 Top-1；type 由 llm_TO 固定
# AGENT = "AppAgent"  # 官方 AppAgent single-path（全页 SoM + 英文 prompt）

# TASK_JSON = "simple_tasks_sample.json"
TASK_JSON = "simple_normal_tasks.json"
TASK_TYPE = "single_simple"
TEST_START = 0
TEST_END = -1  # -1 = all after START
APP_NAMES = ["ximalaya"]
TOP_K = 5  # m2 全路径；TO click/scroll 为 Top-1
DATA_DIR = "../../datasets/Mobile3M/datasets"
DRY_RUN = False
REQUEST_INTERVAL = 0.0
# 仅推理 GT 动作类型在此列表中的步；其他类型跳过（终端 SKIP，结果写 is_skip）
TYPE: list[str] = ["click"]
# TYPE: list[str] = ["scroll"]

# ─────────────────────────────────────────────────────────────────────


def _resolve_write_types(
    type_filter: list[str] | None,
    steps: list[dict[str, Any]],
) -> list[str]:
    from utils.step_log import normalize_type_filter

    allowed = normalize_type_filter(type_filter)
    if allowed is not None:
        return sorted(allowed)
    seen: set[str] = set()
    for step in steps:
        gt_type = step.get("gt_action_type")
        if gt_type and not step.get("is_skip"):
            seen.add(str(gt_type))
    return sorted(seen)


def _finish_step(
    *,
    round_count: int,
    max_rounds: int,
    gt_ids: list[int],
    ctx,
    current_page_name: str,
    page_id: str,
    pred_id,
    pred_info: str,
    step_record: dict[str, Any],
    steps: list[dict[str, Any]],
    ans_action_id: list,
    ans_action_info: list,
    ans_history_pages: list,
    assets=None,
    pred_res: list[str] | None = None,
    top_k: int = 10,
) -> str:
    step_top_k = (
        assets.retrieval_top_k
        if assets is not None and assets.retrieval_top_k is not None
        else top_k
    )
    step_page = current_page_name
    next_page = f"{current_page_name}_{page_id}"
    gt_id = gt_ids[round_count - 1]
    gt_action = ctx.id_to_action.get(gt_id, "?")
    gt_type = action_type(gt_action)
    type_ok = type_match(pred_id, pred_info, gt_id, ctx.id_to_action, pred_res=pred_res)
    action_ok = action_match(pred_id, gt_id)
    topk_ok = topk_retrieval_match(
        gt_id,
        gt_action,
        assets=assets,
        all_action_ids=ctx.all_action_ids,
    )
    gt_rank = gt_similarity_rank(
        gt_id,
        gt_action,
        assets=assets,
        all_action_ids=ctx.all_action_ids,
    )
    gt_rank_display = gt_rank if topk_ok else None

    gt_disp, pred_disp = build_step_displays(
        gt_id,
        gt_action,
        pred_id,
        pred_info,
        assets=assets,
        all_action_ids=ctx.all_action_ids,
        id_to_action=ctx.id_to_action,
        pred_res=pred_res,
    )

    step_instruction = str(step_record.get("step_instruction") or "")
    target_object = getattr(assets, "target_object", None) if assets is not None else None
    llm_action_type = (
        step_record.get("llm_action_type")
        or (getattr(assets, "llm_action_type", None) if assets is not None else None)
    )
    scroll_node_cnt = (
        getattr(assets, "scroll_node_cnt", None) if assets is not None else None
    )

    print_step_compare(
        round_count,
        max_rounds,
        step_page,
        step_instruction,
        target_object,
        gt_disp,
        pred_disp,
        type_ok=type_ok,
        action_ok=action_ok,
        top_k=step_top_k,
        topk_ok=topk_ok,
        llm_action_type=llm_action_type,
        scroll_node_cnt=scroll_node_cnt,
        gt_similarity_rank=gt_rank_display,
    )

    step_update: dict[str, Any] = {
        "page": step_page,
        "target_object": target_object,
        "llm_action_type": llm_action_type,
        "llm_to_raw": getattr(assets, "llm_to_raw", None) if assets else step_record.get("llm_to_raw"),
        "gt_action_id": gt_id,
        "gt_action_type": gt_type,
        "gt_action_info": gt_disp,
        "pred_action_id": pred_id,
        "pred_action_info": pred_disp,
        "type_match": type_ok,
        "action_match": action_ok,
        "top_k": step_top_k,
        "topk_retrieval_match": topk_ok,
    }
    if gt_rank is not None:
        step_update["gt_similarity_rank"] = gt_rank
    if scroll_node_cnt is not None:
        step_update["scroll_node_cnt"] = scroll_node_cnt
    step_record.update(step_update)
    ans_action_id.append(pred_id)
    ans_action_info.append(pred_info)
    ans_history_pages.append(next_page)
    steps.append(step_record)
    return next_page


def _finish_skip_step(
    *,
    round_count: int,
    max_rounds: int,
    gt_ids: list[int],
    ctx,
    current_page_name: str,
    page_id: str,
    step_record: dict[str, Any],
    steps: list[dict[str, Any]],
    ans_action_id: list,
    ans_action_info: list,
    ans_history_pages: list,
) -> str:
    gt_id = gt_ids[round_count - 1]
    gt_action = ctx.id_to_action.get(gt_id, "?")
    gt_type = action_type(gt_action)
    gt_disp = format_gt_vlm_display(
        gt_id, gt_action, assets=None, all_action_ids=ctx.all_action_ids
    )
    step_instruction = str(step_record.get("step_instruction") or "")

    print_step_skip(round_count, max_rounds)

    next_page = f"{current_page_name}_{page_id}"
    step_record.update(
        {
            "page": current_page_name,
            "gt_action_id": gt_id,
            "gt_action_type": gt_type,
            "gt_action_info": gt_disp,
            "pred_action_id": -3,
            "pred_action_info": "skipped",
            "is_skip": True,
            "type_match": False,
            "action_match": False,
        }
    )
    ans_action_id.append(-3)
    ans_action_info.append("skipped")
    ans_history_pages.append(next_page)
    steps.append(step_record)
    return next_page


def run_single_task(
    agent,
    task: dict,
    task_index: int,
    data_dir: Path,
    out_dir: Path,
    *,
    task_progress: int,
    task_total: int,
    dry_run: bool = False,
    type_filter: list[str] | None = None,
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
    max_rounds = len(final_page_name.split("_")[1:])

    action_str_to_id, _ = sman.build_action_id_maps(ctx.all_action_ids)
    gt_ids = sman.gt_action_ids_from_task_page(
        final_page_name, ctx.task_dir, action_str_to_id
    )

    ans_action_id: list = []
    ans_action_info: list = []
    ans_history_pages: list = []
    steps: list[dict] = []
    top_k = getattr(agent, "top_k", TOP_K)

    task_title = (
        f"Task [{task_progress}/{task_total}] {task_index} {final_page_name} ({max_rounds} steps)"
    )
    sman.print_with_color(f"\n====== {task_title} ======", "cyan")

    for page_id in final_page_name.split("_")[1:]:
        round_count += 1
        step_instruction = get_step_instruction(ctx, round_count)
        gt_id = gt_ids[round_count - 1]

        if not gt_type_allowed(gt_id, ctx.id_to_action, type_filter):
            step_record = {
                "round": round_count,
                "step_instruction": step_instruction,
            }
            current_page_name = _finish_skip_step(
                round_count=round_count,
                max_rounds=max_rounds,
                gt_ids=gt_ids,
                ctx=ctx,
                current_page_name=current_page_name,
                page_id=page_id,
                step_record=step_record,
                steps=steps,
                ans_action_id=ans_action_id,
                ans_action_info=ans_action_info,
                ans_history_pages=ans_history_pages,
            )
            continue

        t0 = time.perf_counter()
        assets = agent.prepare_round(
            ctx,
            current_page_name,
            step_instruction,
            dry_run=dry_run,
            gt_id=gt_id,
        )
        if assets is None:
            sman.print_with_color(f"prepare_round failed: {current_page_name}", "red")
            step_record = {
                "round": round_count,
                "step_instruction": step_instruction,
                "error": "prepare_error",
            }
            current_page_name = _finish_step(
                round_count=round_count,
                max_rounds=max_rounds,
                gt_ids=gt_ids,
                ctx=ctx,
                current_page_name=current_page_name,
                page_id=page_id,
                pred_id=-2,
                pred_info="prepare_error",
                step_record=step_record,
                steps=steps,
                ans_action_id=ans_action_id,
                ans_action_info=ans_action_info,
                ans_history_pages=ans_history_pages,
                top_k=top_k,
            )
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

        step_record: dict[str, Any] = {
            "round": round_count,
            "step_instruction": step_instruction,
            "llm_action_type": assets.llm_action_type,
            "llm_to_raw": assets.llm_to_raw,
            "llm_routing_fallback": assets.llm_routing_fallback,
            "thought": thought,
            "vlm_input_tokens": vlm_stats.input_tokens,
            "vlm_output_tokens": vlm_stats.output_tokens,
            "vlm_elapsed_ms": vlm_stats.vlm_elapsed_ms,
            "elapsed_ms": elapsed_ms,
            "is_skip": False,
        }

        if not ok:
            step_record["error"] = "vlm_error"
            current_page_name = _finish_step(
                round_count=round_count,
                max_rounds=max_rounds,
                gt_ids=gt_ids,
                ctx=ctx,
                current_page_name=current_page_name,
                page_id=page_id,
                pred_id=-2,
                pred_info="vlm_error",
                step_record=step_record,
                steps=steps,
                ans_action_id=ans_action_id,
                ans_action_info=ans_action_info,
                ans_history_pages=ans_history_pages,
                assets=assets,
                top_k=top_k,
            )
            if REQUEST_INTERVAL:
                time.sleep(REQUEST_INTERVAL)
            continue

        if res is None:
            step_record["error"] = "parse_error"
            current_page_name = _finish_step(
                round_count=round_count,
                max_rounds=max_rounds,
                gt_ids=gt_ids,
                ctx=ctx,
                current_page_name=current_page_name,
                page_id=page_id,
                pred_id=-2,
                pred_info="parse_error",
                step_record=step_record,
                steps=steps,
                ans_action_id=ans_action_id,
                ans_action_info=ans_action_info,
                ans_history_pages=ans_history_pages,
                assets=assets,
                top_k=top_k,
            )
            if REQUEST_INTERVAL:
                time.sleep(REQUEST_INTERVAL)
            continue

        act_name = res[0]
        step_record["action"] = act_name

        _new_page, action_id, action_info, aid_list, ainfo_list = apply_action(
            res,
            assets,
            ctx.all_action_ids,
            current_page_name,
            ctx.all_page_convert,
        )
        del aid_list, ainfo_list
        step_record["action_id"] = action_id
        step_record["action_info"] = action_info
        current_page_name = _finish_step(
            round_count=round_count,
            max_rounds=max_rounds,
            gt_ids=gt_ids,
            ctx=ctx,
            current_page_name=current_page_name,
            page_id=page_id,
            pred_id=action_id,
            pred_info=action_info,
            step_record=step_record,
            steps=steps,
            ans_action_id=ans_action_id,
            ans_action_info=ans_action_info,
            ans_history_pages=ans_history_pages,
            assets=assets,
            pred_res=res,
            top_k=top_k,
        )

        if REQUEST_INTERVAL:
            time.sleep(REQUEST_INTERVAL)

    task_complete = len(ans_action_id) == max_rounds
    write_types = _resolve_write_types(type_filter, steps)
    written = write_typed_result_slices(
        out_dir,
        final_page_name,
        task_index,
        write_types=write_types,
        steps=steps,
        ans_action_id=ans_action_id,
        ans_action_info=ans_action_info,
        ans_history_pages=ans_history_pages,
        task_complete=task_complete,
        round_count=round_count,
        max_rounds=max_rounds,
        task_description=task.get("task", ""),
    )
    if not written:
        sman.print_with_color(
            f"No result written for task {task_index} (write_types={write_types})",
            "yellow",
        )
    return True


def main() -> None:
    ensure_sman_path()
    ensure_cache_dirs()
    data_path = resolve_data_dir(DATA_DIR)
    tasks = load_tasks(data_path, TASK_JSON)
    selected = filter_tasks(tasks, APP_NAMES, TEST_START, TEST_END)

    model_slug = slug_for_run_filename(get_vlm_model_name())
    base_out_dir = result_dir(TASK_TYPE, AGENT, model_slug)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    agent = load_agent(AGENT, top_k=TOP_K)
    type_label = ",".join(TYPE) if TYPE else "all"
    subdirs = ",".join(TYPE) if TYPE else "click,scroll,..."
    print(
        f"Agent={AGENT} model={model_slug} tasks={len(selected)} "
        f"type={type_label} -> {base_out_dir}/{{{subdirs}}}/"
    )

    task_total = len(selected)
    processed = 0
    for task_progress, (task_index, task) in enumerate(selected, start=1):
        if run_single_task(
            agent,
            task,
            task_index,
            data_path,
            base_out_dir,
            task_progress=task_progress,
            task_total=task_total,
            dry_run=DRY_RUN,
            type_filter=TYPE,
        ):
            processed += 1

    print(f"Done. Processed {processed} tasks.")


if __name__ == "__main__":
    main()
