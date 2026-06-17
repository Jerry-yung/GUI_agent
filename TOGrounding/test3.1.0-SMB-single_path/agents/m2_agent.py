"""M2 agent: llm_TO + TopK + labeled screenshot + VLM."""
from __future__ import annotations

import json
import logging

from colorama import Fore, Style

from agents.parser import parse_labeled_json
from agents.prompts import VLM_ACTION_TYPES, build_m2_user_prompt
from agents.vlm_client import call_vlm
from annotate.instruction_hint import infer_instruction_hit
from annotate.llm_TO import generate_target_object
from annotate.topk import run_topk_pipeline
from utils.sman_bridge import RoundAssets, prepare_round_assets
from utils.step_log import gt_area_label
from utils.task_context import TaskContext
from utils.vlm_stats import VlmCallStats

logger = logging.getLogger(__name__)

MAX_VLM_PARSE_RETRIES = 2

MOCK_RESPONSE = json.dumps(
    {
        "thought": "干跑：点击 c1",
        "action": {"type": "click", "element": "c1"},
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

    def retrieval_top_k_for_hint(self, hint: str) -> int:
        """m2 始终使用 ``self.top_k``；TO/TOa 在非 ambiguous 时覆盖为 1。"""
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

        hint = infer_instruction_hit(step_instruction)
        target_object: str | None = None

        if hint in ("click", "ambiguous"):
            if dry_run:
                target_object = step_instruction[:40] or "目标"
            else:
                to_result = generate_target_object(step_instruction)
                target_object = to_result["target_object"]
        elif dry_run:
            target_object = None

        gt_label: str | None = None
        if gt_id is not None:
            gt_action = ctx.id_to_action.get(gt_id, "")
            gt_label = gt_area_label(
                gt_id,
                gt_action,
                assets=assets,
                all_action_ids=ctx.all_action_ids,
            )

        try:
            retrieval_k = self.retrieval_top_k_for_hint(hint)
            labeled_png, _nodes_json, selected, effective_hint, rank_by_label = run_topk_pipeline(
                ctx.final_page_name,
                current_page_name,
                assets.click_actions,
                assets.scroll_action_bounds,
                assets.screenshot_path,
                target_object or "",
                retrieval_k,
                instruction_hint=hint,
                fresh_instruction_embed=True,
                gt_label=gt_label,
            )
            ambiguous_k = self.retrieval_top_k_for_hint("ambiguous")
            if (
                effective_hint == "ambiguous"
                and hint != "ambiguous"
                and retrieval_k < ambiguous_k
            ):
                retrieval_k = ambiguous_k
                labeled_png, _nodes_json, selected, effective_hint, rank_by_label = run_topk_pipeline(
                    ctx.final_page_name,
                    current_page_name,
                    assets.click_actions,
                    assets.scroll_action_bounds,
                    assets.screenshot_path,
                    target_object or "",
                    retrieval_k,
                    instruction_hint=hint,
                    fresh_instruction_embed=True,
                    gt_label=gt_label,
                )
        except ValueError:
            return None

        assets.drawn_screenshot = str(labeled_png)
        assets.top_k_nodes = selected
        assets.target_object = target_object
        assets.instruction_hit = effective_hint
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
        prompt = build_m2_user_prompt(
            step_instruction,
            instruction_hit=assets.instruction_hit,
            top_k=assets.retrieval_top_k or self.top_k,
        )
        stats = VlmCallStats()

        if dry_run:
            rsp = MOCK_RESPONSE
            if assets.instruction_hit == "scroll" and assets.top_k_nodes:
                label = str(assets.top_k_nodes[0].get("label", "s1")).lower()
                rsp = json.dumps(
                    {
                        "thought": f"干跑：滑动 {label}",
                        "action": {"type": "scroll", "element": label, "direction": "down"},
                    },
                    ensure_ascii=False,
                )
            elif assets.click_actions:
                rsp = json.dumps(
                    {
                        "thought": "干跑：点击 c1",
                        "action": {"type": "click", "element": "c1"},
                    },
                    ensure_ascii=False,
                )
            res = parse_labeled_json(rsp, allow_click_xy=False)
            thought = res[-1] if res else ""
            return True, res, thought, stats

        for attempt in range(MAX_VLM_PARSE_RETRIES + 1):
            status, rsp, call_stats = call_vlm(prompt, assets.drawn_screenshot)
            stats.add(call_stats)

            if not status:
                logger.warning("VLM call failed: %s", str(rsp)[:200])
                return False, None, "", stats

            res = parse_labeled_json(rsp, allow_click_xy=False)
            if res is not None and res[0] in VLM_ACTION_TYPES:
                thought = res[-1] if res else ""
                return True, res, thought, stats

            if attempt < MAX_VLM_PARSE_RETRIES:
                _print_parse_retry(attempt + 1, MAX_VLM_PARSE_RETRIES)

        return True, None, "", stats
