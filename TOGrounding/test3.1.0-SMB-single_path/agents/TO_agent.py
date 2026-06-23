"""TO agent: Top-1 检索 + VLM 填字段；type 由 llm_TO 固定。"""
from __future__ import annotations

import json
import logging

from agents.m2_agent import M2Agent, MAX_VLM_PARSE_RETRIES, MOCK_RESPONSE, _print_parse_retry
from agents.parser import coerce_fixed_pointer_response, parse_labeled_json_fixed
from agents.prompts import VLM_ACTION_TYPES, build_to_prompt_parts
from agents.to_click import parse_to_response
from agents.to_scroll import coerce_fixed_scroll_response, parse_to_scroll_response
from agents.vlm_client import call_vlm_parts
from utils.sman_bridge import RoundAssets
from utils.task_context import TaskContext
from utils.vlm_stats import VlmCallStats

logger = logging.getLogger(__name__)


class TOAgent(M2Agent):
    """需检索的 type 用 Top-1；VLM 不输出 action.type。"""

    name = "to"

    def retrieval_top_k_for_hint(self, hint: str) -> int:
        from annotate.llm_TO import RETRIEVAL_ACTION_TYPES

        if hint in RETRIEVAL_ACTION_TYPES:
            return 1
        return self.top_k

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
        system_prompt, user_prompt = build_to_prompt_parts(
            step_instruction,
            fixed_action_type=fixed_type,
            target_object=assets.target_object or "",
        )
        stats = VlmCallStats()
        top_k_nodes = assets.top_k_nodes or []

        if dry_run:
            rsp = MOCK_RESPONSE
            if fixed_type == "scroll" and top_k_nodes:
                rsp = json.dumps(
                    {
                        "thought": "干跑：滑动",
                        "action": {"direction": "down"},
                    },
                    ensure_ascii=False,
                )
                res = parse_to_scroll_response(rsp, top_k_nodes, force_top1=True)
            elif fixed_type in ("click", "long_press") and top_k_nodes:
                rsp = json.dumps({"thought": "干跑：点击"}, ensure_ascii=False)
                res = parse_to_response(rsp, top_k_nodes, force_top1=True)
            else:
                res = parse_labeled_json_fixed(rsp, fixed_type)
            thought = res[-1] if res else ""
            return True, res, thought, stats

        for attempt in range(MAX_VLM_PARSE_RETRIES + 1):
            status, rsp, call_stats = call_vlm_parts(
                system_prompt, user_prompt, assets.drawn_screenshot
            )
            stats.add(call_stats)

            if not status:
                logger.warning("VLM call failed: %s", str(rsp)[:200])
                return False, None, "", stats

            if fixed_type in ("click", "long_press"):
                coerced = coerce_fixed_pointer_response(rsp, fixed_type)
                res = parse_to_response(
                    json.dumps(coerced, ensure_ascii=False) if coerced else rsp,
                    top_k_nodes,
                    force_top1=True,
                )
                if res is None:
                    res = parse_labeled_json_fixed(rsp, fixed_type)
            elif fixed_type == "scroll":
                coerced = coerce_fixed_scroll_response(rsp)
                res = parse_to_scroll_response(
                    json.dumps(coerced, ensure_ascii=False) if coerced else rsp,
                    top_k_nodes,
                    force_top1=True,
                )
                if res is None:
                    res = parse_labeled_json_fixed(rsp, fixed_type)
            else:
                res = parse_labeled_json_fixed(rsp, fixed_type)

            exec_type = res[0] if res else ""
            if res is not None and exec_type in VLM_ACTION_TYPES | {"back"}:
                thought = res[-1] if res else ""
                return True, res, thought, stats

            if attempt < MAX_VLM_PARSE_RETRIES:
                _print_parse_retry(attempt + 1, MAX_VLM_PARSE_RETRIES)

        return True, None, "", stats
