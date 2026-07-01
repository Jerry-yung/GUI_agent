#!/usr/bin/env python3
"""
TO Agent (AC-high): single vlm_TO planner + top-1 retrieval for pointer steps.
AC-low: top-1 annotated screenshot + predict() (unchanged).
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from agents.m2_agent import VLM_TEMPERATURE, VLM_TOP_P, _encode_image, _normalize_action, merge_low_action
from agents.m2p_agent import (
    M2P_EXECUTOR_TYPES,
    _normalize_planner,
    build_action_from_planner,
)
from agents.m2p_prompts import M2P_PLANNER_SYSTEM_PROMPT, StepHistoryEntry, build_m2p_planner_prompt
from agents.action_validate import validate_ac_action, validate_vlm_fields
from agents.parse_utils import parse_json_from_text, parse_vlm_response
from agents.prompts import build_to_vlm_prompt_parts
from agents.vlm_tokens import invoke_vlm
from llm_set.llm import LLM_QWEN_VL_MAX

__all__ = [
    "TOAgent",
    "StepHistoryEntry",
    "M2P_EXECUTOR_TYPES",
    "build_action_from_planner",
    "build_pointer_pred",
]


def _normalize_action_type(action_type: str) -> str:
    action_type = str(action_type or "").strip()
    if action_type == "long_click":
        return "long_press"
    return action_type


def build_pointer_pred(
    planner: dict,
    top_k_nodes: list | None,
    *,
    has_annotated_nodes: bool = True,
) -> dict | None:
    """Build pred_action for click/long_press from planner + retrieval top-1 (no vlm_action)."""
    action_type = _normalize_action_type(planner.get("planned_action_type", ""))
    if action_type not in M2P_EXECUTOR_TYPES:
        return None
    if not has_annotated_nodes or not top_k_nodes:
        return None

    thought = str(planner.get("thought", "")).strip()
    return {
        "thought": thought,
        "action_type": action_type,
        "node_id": int(top_k_nodes[0]["node_id"]),
    }


class TOAgent:
    """AC-high: vlm_TO planner + top-1 pred. AC-low: predict() on annotated screenshot."""

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

    def build_pointer_pred(
        self,
        planner: dict,
        top_k_nodes: list | None,
        *,
        has_annotated_nodes: bool = True,
    ) -> dict | None:
        return build_pointer_pred(
            planner,
            top_k_nodes,
            has_annotated_nodes=has_annotated_nodes,
        )

    def predict(
        self,
        annotated_screenshot_path: str | Path,
        mode: str,
        *,
        instruction: str = "",
        goal: str = "",
        target_object: str = "",
        current_step_instruction: str = "",
        prev_step_instruction: str = "",
        has_annotated_nodes: bool = True,
        top_k_nodes: list | None = None,
        fixed_action_type: str | None = None,
    ) -> dict:
        """AC-low TO: VLM on annotated screenshot (legacy path)."""
        annotated_screenshot_path = Path(annotated_screenshot_path)
        if not annotated_screenshot_path.is_file():
            raise FileNotFoundError(f"标注截图不存在: {annotated_screenshot_path}")

        use_fixed_fields = mode.lower() == "low" and fixed_action_type

        system_prompt, user_prompt = build_to_vlm_prompt_parts(
            mode,
            instruction=instruction,
            goal=goal,
            target_object=target_object,
            current_step_instruction=current_step_instruction,
            prev_step_instruction=prev_step_instruction,
            has_annotated_nodes=has_annotated_nodes,
            fixed_action_type=fixed_action_type if use_fixed_fields else None,
        )
        b64_image = _encode_image(annotated_screenshot_path)

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
        if parsed is None and not has_annotated_nodes:
            try:
                coord = parse_json_from_text(raw)
                if "x" in coord and "y" in coord:
                    parsed = {
                        "thought": "",
                        "action_type": "click",
                        "x": coord["x"],
                        "y": coord["y"],
                    }
            except ValueError:
                pass

        if parsed is None:
            return {
                "action": None,
                "raw_response": raw,
                "vlm_tokens": vlm_tokens,
                "error": "无法解析 VLM JSON 输出",
            }

        schema_ok, schema_error = (
            validate_vlm_fields(
                parsed,
                fixed_action_type=fixed_action_type,
                has_annotated_nodes=has_annotated_nodes,
                mode=mode,
                agent="TO",
            )
            if use_fixed_fields
            else validate_ac_action(
                parsed,
                has_annotated_nodes=has_annotated_nodes,
                mode=mode,
                agent="TO",
            )
        )

        norm_kwargs = dict(
            has_annotated_nodes=has_annotated_nodes,
            skip_pointer_node_id=has_annotated_nodes,
            skip_input_node_id=True,
            instruction=instruction,
            current_step_instruction=current_step_instruction,
            include_target_object=False,
            include_next_action_type=False,
        )
        action = (
            merge_low_action(fixed_action_type, parsed, mode, **norm_kwargs)
            if use_fixed_fields
            else _normalize_action(parsed, mode, **norm_kwargs)
        )

        pointer_types = ("click", "long_press")
        effective_type = action.get("action_type")
        if (
            has_annotated_nodes
            and top_k_nodes
            and effective_type in pointer_types
        ):
            action["node_id"] = int(top_k_nodes[0]["node_id"])
            action.pop("x", None)
            action.pop("y", None)

        result: dict = {"action": action, "raw_response": raw, "vlm_tokens": vlm_tokens}
        if not schema_ok:
            result["schema_error"] = schema_error
        return result
