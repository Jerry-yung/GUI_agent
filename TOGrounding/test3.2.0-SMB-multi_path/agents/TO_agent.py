"""TO agent: Top-1 检索 + VLM 仅判 type；click 后处理注入 top1 label。"""
from __future__ import annotations

import json
import logging

from agents.m2_agent import M2Agent, MAX_VLM_PARSE_RETRIES, MOCK_RESPONSE, _print_parse_retry
from agents.parser import parse_labeled_json
from agents.prompts import VLM_ACTION_TYPES, build_to_user_prompt
from agents.to_click import parse_to_response
from agents.vlm_client import call_vlm
from utils.sman_bridge import RoundAssets
from utils.task_context import TaskContext
from utils.vlm_stats import VlmCallStats

logger = logging.getLogger(__name__)


class TOAgent(M2Agent):
    """click 路由用 Top-1；instruction_hit=ambiguous 时回退 m2（TOP_K + 动作空间）。"""

    name = "to"

    def retrieval_top_k_for_hint(self, hint: str) -> int:
        if hint == "ambiguous":
            return self.top_k
        return 1

    def decide(
        self,
        ctx: TaskContext,
        assets: RoundAssets,
        step_instruction: str,
        *,
        dry_run: bool = False,
    ) -> tuple[bool, list[str] | None, str, VlmCallStats]:
        if assets.instruction_hit == "ambiguous":
            return super().decide(ctx, assets, step_instruction, dry_run=dry_run)

        del ctx
        prompt = build_to_user_prompt(
            step_instruction,
            task_desc=self._task_desc,
            current_page_name=assets.page_name or "",
            last_summary=self._last_summary,
            target_object=assets.target_object or "",
            instruction_hit=assets.instruction_hit,
        )
        stats = VlmCallStats()
        top_k_nodes = assets.top_k_nodes or []

        if dry_run:
            rsp = MOCK_RESPONSE
            if assets.instruction_hit == "scroll" and top_k_nodes:
                label = str(top_k_nodes[0].get("label", "s1")).lower()
                rsp = json.dumps(
                    {
                        "thought": f"干跑：滑动 {label}",
                        "summary": f"滑动 {label}；预计进入页面：滑动后的列表页",
                        "action": {"type": "scroll", "element": label, "direction": "down"},
                    },
                    ensure_ascii=False,
                )
                res = parse_labeled_json(rsp)
            elif top_k_nodes:
                label = str(top_k_nodes[0].get("label", "c1")).lower()
                rsp = json.dumps(
                    {
                        "thought": f"干跑：点击 {label}",
                        "summary": "点击检索目标；预计进入页面：下一级子页面",
                        "action": {"type": "click"},
                    },
                    ensure_ascii=False,
                )
                res = parse_to_response(rsp, top_k_nodes, force_top1=True)
            else:
                res = parse_labeled_json(rsp)
            if res is None:
                return True, None, "", stats
            thought = self._commit_vlm_fields(rsp)
            return True, res, thought, stats

        force_top1 = assets.instruction_hit != "scroll"
        for attempt in range(MAX_VLM_PARSE_RETRIES + 1):
            status, rsp, call_stats = call_vlm(prompt, assets.drawn_screenshot)
            stats.add(call_stats)

            if not status:
                logger.warning("VLM call failed: %s", str(rsp)[:200])
                return False, None, "", stats

            if force_top1:
                res = parse_to_response(rsp, top_k_nodes, force_top1=True)
                if res is None:
                    res = parse_labeled_json(rsp)
                    if res and res[0] == "click" and top_k_nodes:
                        label = str(top_k_nodes[0].get("label", "")).lower()
                        trailing = res[-1] if len(res) > 2 else ""
                        res = ["click", label, trailing]
            else:
                res = parse_labeled_json(rsp)

            if res is not None and res[0] in VLM_ACTION_TYPES:
                thought = self._commit_vlm_fields(rsp)
                return True, res, thought, stats

            if attempt < MAX_VLM_PARSE_RETRIES:
                _print_parse_retry(attempt + 1, MAX_VLM_PARSE_RETRIES)

        return True, None, "", stats
