"""M2 agent: llm_TO + TopK + labeled screenshot + VLM."""
from __future__ import annotations

import json
import logging

from colorama import Fore, Style

from agents.parser import parse_labeled_json_fields_fixed
from agents.prompts import VLM_ACTION_TYPES, build_m2_prompt_parts
from agents.vlm_client import call_vlm_parts
from annotate.llm_TO import (
    LlmToContext,
    MAX_ACTION_HISTORY,
    RETRIEVAL_ACTION_TYPES,
    format_action_history_line,
    generate_target_object,
)
from annotate.topk import run_topk_pipeline
from utils.sman_bridge import RoundAssets, prepare_round_assets
from utils.step_log import gt_area_label
from utils.task_context import TaskContext
from utils.vlm_stats import VlmCallStats

logger = logging.getLogger(__name__)

MAX_VLM_PARSE_RETRIES = 2

MOCK_RESPONSE = json.dumps(
    {
        "thought": "干跑：点击第一个可点击区域",
        "summary": "点击推荐入口；预计进入页面：下一级子页面",
        "action": {"element": "1"},
    },
    ensure_ascii=False,
)


def _print_parse_retry(retry_no: int, max_retries: int) -> None:
    color = getattr(Fore, "LIGHTMAGENTA_EX", Fore.MAGENTA)
    print(f"{color}VLM parse failed. Retry [{retry_no}/{max_retries}]{Style.RESET_ALL}")


class M2Agent:
    name = "m2"

    def __init__(self, top_k: int = 10) -> None:
        self.top_k = top_k
        self._task_desc = ""
        self._last_summary = ""
        self.last_vlm_summary = ""
        self._last_action = ""
        self._action_history: list[str] = []

    def begin_task(self, ctx: TaskContext) -> None:
        self._task_desc = ctx.task_desc
        self._last_summary = ""
        self.last_vlm_summary = ""
        self._last_action = ""
        self._action_history = []

    def _llm_to_context(self, current_page_name: str) -> LlmToContext:
        return LlmToContext(
            current_page=current_page_name,
            last_summary=self._last_summary,
            last_action=self._last_action,
            action_history=list(self._action_history),
        )

    def record_step_outcome(
        self,
        *,
        step_page: str,
        next_page: str,
        step_instruction: str,
        llm_action_type: str = "",
        target_object: str = "",
        pred: str = "",
    ) -> None:
        line = format_action_history_line(
            step_page=step_page,
            next_page=next_page,
            step_instruction=step_instruction,
            llm_action_type=llm_action_type,
            target_object=target_object,
            pred=pred,
        )
        self._action_history.append(line)
        if len(self._action_history) > MAX_ACTION_HISTORY:
            self._action_history = self._action_history[-MAX_ACTION_HISTORY:]
        if pred:
            if next_page and next_page != step_page:
                self._last_action = f"{pred} (@{step_page}→{next_page})"
            else:
                self._last_action = f"{pred} (@{step_page})"
        else:
            self._last_action = ""

    def retrieval_top_k_for_hint(self, hint: str) -> int:
        """m2 始终使用 ``self.top_k``；TO 对需检索的 type 覆盖为 1。"""
        del hint
        return self.top_k

    def prepare_round(
        self,
        ctx: TaskContext,
        current_page_name: str,
        step_instruction: str,
        *,
        dry_run: bool = False,
        gt_id: int | None = None,
    ) -> RoundAssets | None:
        assets = prepare_round_assets(
            ctx.task_dir,
            current_page_name,
            ctx.id_to_action,
            ctx.current_page_actions,
            use_official_labels=False,
        )
        if assets is None:
            return None

        llm_type = "click"
        target_object = ""
        llm_to_raw: str | None = None

        if dry_run:
            target_object = step_instruction[:40] or "目标"
        else:
            try:
                to_result = generate_target_object(
                    step_instruction,
                    ctx=self._llm_to_context(current_page_name),
                )
                llm_type = to_result["action_type"]
                target_object = to_result["target_object"]
                llm_to_raw = to_result.get("raw_response")
            except ValueError as exc:
                logger.warning("llm_TO failed: %s", exc)
                llm_type = "input"
                target_object = ""

        gt_label: str | None = None
        if gt_id is not None:
            gt_action = ctx.id_to_action.get(gt_id, "")
            gt_label = gt_area_label(
                gt_id,
                gt_action,
                assets=assets,
                all_action_ids=ctx.all_action_ids,
            )

        retrieval_k = self.retrieval_top_k_for_hint(llm_type)
        routing_fallback: str | None = None

        try:
            if llm_type in RETRIEVAL_ACTION_TYPES and not (target_object or "").strip():
                routing_fallback = "empty_to"
                labeled_png, _nodes_json, selected, effective_hint, rank_by_label = (
                    run_topk_pipeline(
                        ctx.final_page_name,
                        current_page_name,
                        assets.click_actions,
                        assets.scroll_action_bounds,
                        assets.screenshot_path,
                        "",
                        retrieval_k,
                        action_type_hint="input",
                        fresh_instruction_embed=True,
                        gt_label=gt_label,
                    )
                )
            elif llm_type in RETRIEVAL_ACTION_TYPES:
                labeled_png, _nodes_json, selected, effective_hint, rank_by_label = (
                    run_topk_pipeline(
                        ctx.final_page_name,
                        current_page_name,
                        assets.click_actions,
                        assets.scroll_action_bounds,
                        assets.screenshot_path,
                        target_object,
                        retrieval_k,
                        action_type_hint=llm_type,
                        fresh_instruction_embed=True,
                        gt_label=gt_label,
                    )
                )
            else:
                labeled_png, _nodes_json, selected, effective_hint, rank_by_label = (
                    run_topk_pipeline(
                        ctx.final_page_name,
                        current_page_name,
                        assets.click_actions,
                        assets.scroll_action_bounds,
                        assets.screenshot_path,
                        "",
                        retrieval_k,
                        action_type_hint=llm_type,
                        fresh_instruction_embed=True,
                        gt_label=gt_label,
                    )
                )
        except ValueError:
            return None

        assets.drawn_screenshot = str(labeled_png)
        assets.top_k_nodes = selected
        assets.target_object = target_object or None
        assets.llm_action_type = llm_type
        assets.llm_to_raw = llm_to_raw
        assets.llm_routing_fallback = routing_fallback
        assets.retrieval_top_k = retrieval_k
        assets.scroll_node_cnt = (
            len(selected) if effective_hint == "scroll" else None
        )
        assets.similarity_rank_by_label = rank_by_label
        assets.page_name = current_page_name
        return assets

    def decide(
        self,
        ctx: TaskContext,
        assets: RoundAssets,
        step_instruction: str,
        *,
        dry_run: bool = False,
    ) -> tuple[bool, list[str] | None, str, VlmCallStats]:
        del ctx
        fixed_type = assets.llm_action_type or "click"
        system_prompt, user_prompt = build_m2_prompt_parts(
            step_instruction,
            fixed_action_type=fixed_type,
            task_desc=self._task_desc,
            current_page_name=assets.page_name or "",
            last_summary=self._last_summary,
            top_k=assets.retrieval_top_k or self.top_k,
            target_object=assets.target_object or "",
        )
        stats = VlmCallStats()

        if dry_run:
            rsp = MOCK_RESPONSE
            if fixed_type == "scroll" and assets.top_k_nodes:
                label = str(assets.top_k_nodes[0].get("label", "1")).lower()
                rsp = json.dumps(
                    {
                        "thought": f"干跑：在 {label} 区域向下滑动",
                        "summary": f"滑动 {label}；预计进入页面：滑动后的列表页",
                        "action": {"element": label, "direction": "down"},
                    },
                    ensure_ascii=False,
                )
            elif fixed_type in ("click", "long_press") and assets.click_actions:
                rsp = json.dumps(
                    {
                        "thought": "干跑：点击 1",
                        "summary": "点击 1；预计进入页面：下一级子页面",
                        "action": {"element": "1"},
                    },
                    ensure_ascii=False,
                )
            elif fixed_type == "input":
                rsp = json.dumps(
                    {
                        "thought": "干跑：输入",
                        "summary": "输入文本；预计进入页面：当前页",
                        "action": {"text": "test"},
                    },
                    ensure_ascii=False,
                )
            elif fixed_type == "back":
                rsp = json.dumps(
                    {
                        "thought": "干跑：返回",
                        "summary": "返回上一页；预计进入页面：上一级页面",
                    },
                    ensure_ascii=False,
                )
            parsed = parse_labeled_json_fields_fixed(
                rsp, fixed_type, allow_click_xy=False
            )
            if parsed is None:
                return True, None, "", stats
            self._last_summary = parsed.summary
            self.last_vlm_summary = parsed.summary
            return True, parsed.res, parsed.thought, stats

        for attempt in range(MAX_VLM_PARSE_RETRIES + 1):
            status, rsp, call_stats = call_vlm_parts(
                system_prompt, user_prompt, assets.drawn_screenshot
            )
            stats.add(call_stats)

            if not status:
                logger.warning("VLM call failed: %s", str(rsp)[:200])
                return False, None, "", stats

            parsed = parse_labeled_json_fields_fixed(
                rsp, fixed_type, allow_click_xy=False
            )
            exec_type = parsed.res[0] if parsed and parsed.res else ""
            if parsed is not None and exec_type in VLM_ACTION_TYPES | {"back"}:
                self._last_summary = parsed.summary
                self.last_vlm_summary = parsed.summary
                return True, parsed.res, parsed.thought, stats

            if attempt < MAX_VLM_PARSE_RETRIES:
                _print_parse_retry(attempt + 1, MAX_VLM_PARSE_RETRIES)

        return True, None, "", stats

    def _commit_vlm_fields(self, rsp: str) -> str:
        parsed = parse_labeled_json_fields_fixed(
            rsp, "click", allow_click_xy=False
        )
        if parsed is not None:
            self._last_summary = parsed.summary
            self.last_vlm_summary = parsed.summary
            return parsed.thought
        return ""
