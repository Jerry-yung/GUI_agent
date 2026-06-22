"""Official AppAgent multi-path adapter (SMAN-Bench baseline)."""
from __future__ import annotations

import importlib.util
import logging
import re
import sys
from pathlib import Path

from agents.appagent_parse import parse_explore_rsp
from agents.m2_agent import MAX_VLM_PARSE_RETRIES, _print_parse_retry
from agents.vlm_client import call_vlm
from utils.paths import SMAN_BENCH_DIR
from utils.sman_bridge import RoundAssets, prepare_round_assets
from utils.task_context import TaskContext, get_multipath_step_instruction
from utils.vlm_stats import VlmCallStats

logger = logging.getLogger(__name__)

VLM_ACTION_TYPES = frozenset({"click", "scroll", "input", "back"})

MOCK_EXPLORE_RSP = (
    "Observation: Dry run screenshot with labeled c1.\n"
    "Thought: Click the first clickable element.\n"
    "Action: click(c1)\n"
    "Summary: Clicked c1 in dry run."
)


def _load_multipath_template() -> str:
    prompts_path = SMAN_BENCH_DIR / "appagent" / "prompts.py"
    if not prompts_path.is_file():
        raise FileNotFoundError(f"AppAgent prompts not found: {prompts_path}")
    appagent_dir = str(SMAN_BENCH_DIR / "appagent")
    if appagent_dir not in sys.path:
        sys.path.insert(0, appagent_dir)
    spec = importlib.util.spec_from_file_location("appagent_prompts", prompts_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load AppAgent prompts from {prompts_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return str(mod.multipath_task_template)


class AppAgentAgent:
    """Official AppAgent multi-path: full-page SoM + English multipath prompt."""

    name = "AppAgent"

    def __init__(self) -> None:
        self._multipath_template = _load_multipath_template()
        self._task_desc = ""
        self._multi_task_desc: list[str] = []
        self._last_act = "None"
        self._current_page_name = ""

    def begin_task(self, ctx: TaskContext) -> None:
        self._task_desc = ctx.task_desc
        self._multi_task_desc = list(ctx.multi_task_desc)
        self._last_act = "None"
        self._current_page_name = ctx.final_page_name.split("_")[0]

    def prepare_round(
        self,
        ctx: TaskContext,
        current_page_name: str,
        step_instruction: str,
        *,
        dry_run: bool = False,
        gt_id: int | None = None,
    ) -> RoundAssets | None:
        del step_instruction, dry_run, gt_id
        self._current_page_name = current_page_name
        labeled_output = (
            Path(ctx.task_dir) / current_page_name / f"{current_page_name}_labeled_multi.png"
        )
        return prepare_round_assets(
            ctx.task_dir,
            current_page_name,
            ctx.id_to_action,
            ctx.current_page_actions,
            labeled_output=labeled_output,
            use_official_labels=True,
        )

    def _build_prompt(self, ctx: TaskContext, step_instruction: str) -> str:
        current_desc = get_multipath_step_instruction(ctx, self._current_page_name)

        prompt = re.sub(r"<ui_document>", "", self._multipath_template)
        prompt = re.sub(r"<task_description>", self._task_desc, prompt)
        prompt = re.sub(r"<current_task_desc>", current_desc, prompt)
        prompt = re.sub(r"<last_act>", self._last_act, prompt)
        del step_instruction
        return prompt

    def decide(
        self,
        ctx: TaskContext,
        assets: RoundAssets,
        step_instruction: str,
        *,
        dry_run: bool = False,
    ) -> tuple[bool, list[str] | None, str, VlmCallStats]:
        prompt = self._build_prompt(ctx, step_instruction)
        stats = VlmCallStats()

        if dry_run:
            rsp = MOCK_EXPLORE_RSP
            parsed = parse_explore_rsp(rsp)
            res = self._parsed_to_action_res(parsed)
            if res is None:
                return True, None, "", stats
            summary = res[-1] if res else ""
            self._last_act = str(summary) if summary else self._last_act
            return True, res, str(summary), stats

        last_rsp = ""
        for attempt in range(MAX_VLM_PARSE_RETRIES + 1):
            status, rsp, call_stats = call_vlm(prompt, assets.drawn_screenshot)
            stats.add(call_stats)
            last_rsp = rsp

            if not status:
                logger.warning("VLM call failed: %s", str(rsp)[:200])
                return False, None, "", stats

            parsed = parse_explore_rsp(rsp)
            res = self._parsed_to_action_res(parsed)
            if res is not None and res[0] in VLM_ACTION_TYPES:
                summary = res[-1] if res else ""
                self._last_act = str(summary) if summary else self._last_act
                return True, res, str(summary), stats

            if attempt < MAX_VLM_PARSE_RETRIES:
                _print_parse_retry(attempt + 1, MAX_VLM_PARSE_RETRIES)

        del last_rsp
        return True, None, "", stats

    @staticmethod
    def _parsed_to_action_res(parsed: list[str]) -> list[str] | None:
        if not parsed or parsed[0] == "ERROR":
            return None
        act_name = parsed[0]
        if act_name == "click" and len(parsed) >= 3:
            return [act_name, str(parsed[1]), str(parsed[2])]
        if act_name == "input" and len(parsed) >= 3:
            return [act_name, str(parsed[1]), str(parsed[2])]
        if act_name == "scroll" and len(parsed) >= 4:
            return [act_name, str(parsed[1]), str(parsed[2]), str(parsed[3])]
        if act_name == "back" and len(parsed) >= 2:
            return [act_name, str(parsed[1])]
        return None
