#!/usr/bin/env python3
"""
M2P Agent (AC-high only): vlm_TO planner on raw screenshot + vlm_action executor on TopK or raw.

Low mode is not supported — main.py rejects AGENT=m2p with AC_MODE=low.
Non-pointer steps (scroll, input_text, wait, navigate_*) skip vlm_action; planner outputs full action.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from agents.action_validate import validate_ac_action, validate_vlm_fields
from agents.m2_agent import (
    VLM_TEMPERATURE,
    VLM_TOP_P,
    _encode_image,
    _normalize_action,
    merge_low_action,
)
from agents.m2p_prompts import (
    M2P_PLANNER_SYSTEM_PROMPT,
    StepHistoryEntry,
    build_m2p_planner_prompt,
)
from agents.parse_utils import parse_vlm_response
from agents.prompts import AC_ACTION_TYPES, SCROLL_DIRECTIONS, build_m2_prompt_parts
from agents.scroll_gesture import normalize_scroll_gesture_from_instruction
from agents.vlm_tokens import invoke_vlm
from llm_set.llm import LLM_QWEN_VL_MAX

M2P_EXECUTOR_TYPES = frozenset({"click", "long_press"})
M2P_PLANNER_ONLY_TYPES = frozenset(
    {"scroll", "input_text", "wait", "navigate_back", "navigate_home"}
)

__all__ = [
    "M2PAgent",
    "StepHistoryEntry",
    "M2P_EXECUTOR_TYPES",
    "M2P_PLANNER_ONLY_TYPES",
    "build_action_from_planner",
]


def _normalize_action_type(action_type: str) -> str:
    action_type = str(action_type or "").strip()
    if action_type == "long_click":
        return "long_press"
    return action_type


def _normalize_planner(data: dict) -> dict:
    """Parse and normalize planner JSON from vlm_TO."""
    planned_action_type = _normalize_action_type(data.get("planned_action_type", ""))
    if planned_action_type not in AC_ACTION_TYPES:
        planned_action_type = ""

    direction = str(data.get("direction", "")).strip().lower()
    if direction not in SCROLL_DIRECTIONS:
        direction = ""

    return {
        "thought": str(data.get("thought", "")).strip(),
        "planned_action_type": planned_action_type,
        "step_instruction": str(data.get("step_instruction", "")).strip(),
        "target_object": str(data.get("target_object", "")).strip(),
        "direction": direction,
        "text": str(data.get("text", "")).strip(),
        "step_summary": str(data.get("step_summary", "")).strip(),
    }


def build_action_from_planner(
    planner: dict,
    *,
    step_instruction: str = "",
) -> dict | None:
    """Build pred_action from planner-only step (no vlm_action)."""
    action_type = _normalize_action_type(planner.get("planned_action_type", ""))
    if action_type not in M2P_PLANNER_ONLY_TYPES:
        return None

    thought = str(planner.get("thought", "")).strip()
    result: dict = {"thought": thought, "action_type": action_type}

    if action_type == "scroll":
        direction = str(planner.get("direction", "")).strip().lower()
        if direction not in SCROLL_DIRECTIONS:
            return None
        scroll_text = (step_instruction or planner.get("step_instruction") or "").strip()
        direction = normalize_scroll_gesture_from_instruction(direction, scroll_text)
        result["direction"] = direction
    elif action_type == "input_text":
        text = str(planner.get("text", "")).strip()
        if not text:
            return None
        result["text"] = text

    return result


class M2PAgent:
    """AC-high dual-VLM agent: plan(raw) → execute(annotated or raw) for pointer steps only."""

    def __init__(self):
        self.vlm = LLM_QWEN_VL_MAX(temperature=VLM_TEMPERATURE, top_p=VLM_TOP_P)

    def plan(
        self,
        raw_screenshot_path: str | Path,
        *,
        goal: str,
        step_num: int,
        max_steps: int,
        history: list[StepHistoryEntry],
    ) -> dict:
        raw_screenshot_path = Path(raw_screenshot_path)
        if not raw_screenshot_path.is_file():
            raise FileNotFoundError(f"原图不存在: {raw_screenshot_path}")

        user_prompt = build_m2p_planner_prompt(goal, step_num, max_steps, history)
        b64_image = _encode_image(raw_screenshot_path)

        messages = [
            SystemMessage(content=M2P_PLANNER_SYSTEM_PROMPT),
            HumanMessage(
                content=[
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ]
            ),
        ]

        raw, vlm_tokens = invoke_vlm(self.vlm.model, messages)
        parsed = parse_vlm_response(raw)
        if parsed is None:
            return {
                "planner": None,
                "raw_response": raw,
                "vlm_tokens": vlm_tokens,
                "error": "无法解析 vlm_TO JSON 输出",
            }

        return {
            "planner": _normalize_planner(parsed),
            "raw_response": raw,
            "vlm_tokens": vlm_tokens,
        }

    def build_action_from_planner(
        self,
        planner: dict,
        *,
        step_instruction: str = "",
    ) -> dict | None:
        return build_action_from_planner(planner, step_instruction=step_instruction)

    def execute(
        self,
        screenshot_path: str | Path,
        *,
        goal: str,
        step_instruction: str,
        has_annotated_nodes: bool = True,
        fixed_action_type: str | None = None,
    ) -> dict:
        screenshot_path = Path(screenshot_path)
        if not screenshot_path.is_file():
            raise FileNotFoundError(f"截图不存在: {screenshot_path}")

        instruction = (step_instruction or goal or "").strip()
        if not instruction:
            instruction = goal.strip()

        fixed = _normalize_action_type(fixed_action_type or "")
        if fixed and fixed not in AC_ACTION_TYPES:
            fixed = ""

        use_fixed = bool(fixed)
        system_prompt, user_prompt = build_m2_prompt_parts(
            "low",
            instruction=instruction,
            goal=goal,
            prev_step_instruction="",
            has_annotated_nodes=has_annotated_nodes,
            fixed_action_type=fixed if use_fixed else None,
        )
        b64_image = _encode_image(screenshot_path)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=[
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                    },
                ]
            ),
        ]

        raw, vlm_tokens = invoke_vlm(self.vlm.model, messages)
        parsed = parse_vlm_response(raw)
        if parsed is None:
            return {
                "action": None,
                "raw_response": raw,
                "vlm_tokens": vlm_tokens,
                "error": "无法解析 vlm_action JSON 输出",
            }

        norm_kwargs = dict(
            has_annotated_nodes=has_annotated_nodes,
            instruction=instruction,
        )
        if use_fixed:
            schema_ok, schema_error = validate_vlm_fields(
                parsed,
                fixed_action_type=fixed,
                has_annotated_nodes=has_annotated_nodes,
                mode="low",
                agent="m2p",
            )
            action = merge_low_action(fixed, parsed, "low", **norm_kwargs)
        else:
            schema_ok, schema_error = validate_ac_action(
                parsed,
                has_annotated_nodes=has_annotated_nodes,
                mode="low",
                agent="m2p",
            )
            action = _normalize_action(parsed, "low", **norm_kwargs)

        result: dict = {
            "action": action,
            "raw_response": raw,
            "vlm_tokens": vlm_tokens,
        }
        if not schema_ok:
            result["schema_error"] = schema_error
        return result
