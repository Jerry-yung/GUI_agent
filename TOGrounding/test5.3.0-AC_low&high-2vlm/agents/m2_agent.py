#!/usr/bin/env python3
"""
M2 Agent：标注截图 + instruction/goal → Android Control 格式 action。
"""

from __future__ import annotations

import base64
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from agents.action_validate import validate_ac_action, validate_vlm_fields
from agents.parse_utils import parse_vlm_response
from agents.prompts import AC_ACTION_TYPES, SCROLL_DIRECTIONS, build_m2_prompt_parts
from agents.scroll_gesture import normalize_scroll_gesture_from_instruction
from agents.vlm_tokens import invoke_vlm
from llm_set.llm import LLM_QWEN_VL_MAX

VLM_TEMPERATURE = 0.1
VLM_TOP_P = 0.3


def _encode_image(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _parse_norm_coord(data: dict, key: str) -> float | None:
    if key not in data:
        return None
    try:
        value = float(data[key])
    except (TypeError, ValueError):
        return None
    if not (0.0 <= value <= 1.0):
        return None
    return _clamp01(value)


def _normalize_action(
    data: dict,
    mode: str = "low",
    *,
    has_annotated_nodes: bool = True,
    skip_pointer_node_id: bool = False,
    keep_pointer_coords: bool = False,
    skip_input_node_id: bool = False,
    include_target_object: bool = False,
    include_next_action_type: bool = False,
    instruction: str = "",
    current_step_instruction: str = "",
) -> dict:
    action_type = str(data.get("action_type", "")).strip()
    if action_type == "long_click":
        action_type = "long_press"

    result = {
        "thought": str(data.get("thought", "")).strip(),
        "action_type": action_type,
    }

    if action_type in ("click", "long_press"):
        if has_annotated_nodes and not skip_pointer_node_id:
            if "node_id" in data:
                result["node_id"] = int(data["node_id"])
            elif "click_id" in data:
                result["node_id"] = int(data["click_id"])
        elif not has_annotated_nodes:
            x = _parse_norm_coord(data, "x")
            y = _parse_norm_coord(data, "y")
            if x is not None and y is not None:
                result["x"] = x
                result["y"] = y
        elif keep_pointer_coords:
            x = _parse_norm_coord(data, "x")
            y = _parse_norm_coord(data, "y")
            if x is not None and y is not None:
                result["x"] = x
                result["y"] = y
    elif action_type == "scroll":
        direction = str(data.get("direction", "")).strip().lower()
        if direction in SCROLL_DIRECTIONS:
            scroll_text = (instruction or "").strip()
            if mode.lower() == "high" and not scroll_text:
                scroll_text = (current_step_instruction or "").strip()
            direction = normalize_scroll_gesture_from_instruction(
                direction,
                scroll_text,
            )
            result["direction"] = direction
    elif action_type == "input_text":
        result["text"] = str(data.get("text", ""))
        if not skip_input_node_id and "node_id" in data:
            result["node_id"] = int(data["node_id"])

    if action_type not in AC_ACTION_TYPES:
        result["action_type"] = action_type

    if mode.lower() == "high":
        next_instruction = str(
            data.get("next_instruction", data.get("next_instrcution", ""))
        ).strip()
        if next_instruction:
            result["next_instruction"] = next_instruction

    if include_target_object:
        target_object = str(data.get("target_object", "")).strip()
        if target_object:
            result["target_object"] = target_object

    if include_next_action_type:
        next_action_type = str(data.get("next_action_type", "")).strip()
        if next_action_type == "long_click":
            next_action_type = "long_press"
        if next_action_type:
            result["next_action_type"] = next_action_type

    return result


def merge_low_action(
    fixed_action_type: str,
    vlm_parsed: dict,
    mode: str,
    **normalize_kwargs,
) -> dict:
    """AC-low / AC-high：将固定的 action_type 与 VLM 字段合并。"""
    merged = {"action_type": fixed_action_type, **vlm_parsed}
    mode_l = mode.lower()
    if mode_l == "high":
        normalize_kwargs.setdefault("include_target_object", True)
        normalize_kwargs.setdefault("include_next_action_type", True)
    return _normalize_action(merged, mode, **normalize_kwargs)


class M2Agent:
    """标注截图 + 用户文本 → AC action JSON。"""

    def __init__(self):
        self.vlm = LLM_QWEN_VL_MAX(temperature=VLM_TEMPERATURE, top_p=VLM_TOP_P)

    def predict(
        self,
        annotated_screenshot_path: str | Path,
        mode: str,
        *,
        instruction: str = "",
        goal: str = "",
        current_step_instruction: str = "",
        prev_step_instruction: str = "",
        has_annotated_nodes: bool = True,
        top_k_nodes: list | None = None,
        fixed_action_type: str | None = None,
    ) -> dict:
        _ = top_k_nodes
        annotated_screenshot_path = Path(annotated_screenshot_path)
        if not annotated_screenshot_path.is_file():
            raise FileNotFoundError(f"标注截图不存在: {annotated_screenshot_path}")

        use_fixed_fields = mode.lower() in ("low", "high") and fixed_action_type

        system_prompt, user_prompt = build_m2_prompt_parts(
            mode,
            instruction=instruction,
            goal=goal,
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
                agent="m2",
            )
            if use_fixed_fields
            else validate_ac_action(
                parsed,
                has_annotated_nodes=has_annotated_nodes,
                mode=mode,
                agent="m2",
            )
        )

        is_high = mode.lower() == "high"
        norm_kwargs = dict(
            has_annotated_nodes=has_annotated_nodes,
            instruction=instruction,
            current_step_instruction=current_step_instruction,
            include_target_object=is_high,
            include_next_action_type=is_high,
        )
        action = (
            merge_low_action(fixed_action_type, parsed, mode, **norm_kwargs)
            if use_fixed_fields
            else _normalize_action(parsed, mode, **norm_kwargs)
        )

        result: dict = {
            "action": action,
            "raw_response": raw,
            "vlm_tokens": vlm_tokens,
        }
        if not schema_ok:
            result["schema_error"] = schema_error
        return result
