"""TOa agent: Top-1 建议框 + element 或归一化坐标。"""
from __future__ import annotations

import json
import logging

from agents.m2_agent import M2Agent, MAX_VLM_PARSE_RETRIES, MOCK_RESPONSE, _print_parse_retry
from utils.click_res import is_click_xy_res
from agents.parser import parse_labeled_json
from agents.prompts import VLM_ACTION_TYPES, build_toa_user_prompt
from agents.to_click import parse_toa_response, toa_decide_meta
from agents.to_retrieval import load_retrieval_scores
from agents.vlm_client import call_vlm
from utils.sman_bridge import RoundAssets
from utils.task_context import TaskContext
from utils.vlm_stats import VlmCallStats

logger = logging.getLogger(__name__)


class TOaAgent(M2Agent):
    """click 路由用 Top-1；ambiguous 时回退 m2（TOP_K + 动作空间）。"""

    name = "toa"

    def __init__(self, top_k: int = 10) -> None:
        super().__init__(top_k=top_k)
        self.last_decide_meta: dict = {}

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
            self.last_decide_meta = {}
            return super().decide(ctx, assets, step_instruction, dry_run=dry_run)

        top_k_nodes = assets.top_k_nodes or []
        query = assets.target_object or ""
        page_name = assets.page_name or ""
        retrieval_sim: float | None = None
        retrieval_margin: float | None = None
        if assets.instruction_hit not in ("scroll", "input"):
            retrieval_sim, retrieval_margin = load_retrieval_scores(
                ctx.final_page_name,
                page_name,
                top_k=assets.retrieval_top_k or 1,
                query_text=query,
            )
        prompt = build_toa_user_prompt(
            step_instruction,
            task_desc=self._task_desc,
            current_page_name=page_name,
            last_summary=self._last_summary,
            target_object=query,
            retrieval_final_sim=retrieval_sim,
            retrieval_margin=retrieval_margin,
            instruction_hit=assets.instruction_hit,
        )
        stats = VlmCallStats()
        self.last_decide_meta = {}

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
                locator_source = "element" if res else None
            elif top_k_nodes:
                label = str(top_k_nodes[0].get("label", "c1")).lower()
                rsp = json.dumps(
                    {
                        "thought": f"干跑：点击 {label}",
                        "summary": "点击检索目标；预计进入页面：下一级子页面",
                        "action": {"type": "click", "element": label},
                    },
                    ensure_ascii=False,
                )
                res, locator_source = parse_toa_response(rsp, top_k_nodes)
            else:
                res, locator_source = parse_toa_response(rsp, top_k_nodes)
            if res is None:
                return True, None, "", stats
            self.last_decide_meta = toa_decide_meta(rsp, res, locator_source)
            thought = self._commit_vlm_fields(rsp)
            return True, res, thought, stats

        use_toa_click = assets.instruction_hit not in ("scroll", "input")
        last_rsp = ""
        for attempt in range(MAX_VLM_PARSE_RETRIES + 1):
            status, rsp, call_stats = call_vlm(prompt, assets.drawn_screenshot)
            stats.add(call_stats)
            last_rsp = rsp

            if not status:
                logger.warning("VLM call failed: %s", str(rsp)[:200])
                return False, None, "", stats

            if use_toa_click:
                res, locator_source = parse_toa_response(rsp, top_k_nodes)
                if res is None:
                    res = parse_labeled_json(rsp)
                    if res and res[0] == "click" and not is_click_xy_res(res):
                        res, locator_source = parse_toa_response(rsp, top_k_nodes)
            else:
                res = parse_labeled_json(rsp)
                locator_source = None

            if res is not None and res[0] in VLM_ACTION_TYPES:
                self.last_decide_meta = toa_decide_meta(rsp, res, locator_source)
                thought = self._commit_vlm_fields(rsp)
                return True, res, thought, stats

            if attempt < MAX_VLM_PARSE_RETRIES:
                _print_parse_retry(attempt + 1, MAX_VLM_PARSE_RETRIES)

        self.last_decide_meta = toa_decide_meta(last_rsp, None, None)
        return True, None, "", stats
