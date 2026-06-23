#!/usr/bin/env python3
"""
M2V Agent：与 M2 相同接口；high 模式下 VLM 输出下一步规划三字段。
step0 的 action_type + target_object 由 main 通过 llm_TO 生成。
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
from agents.parse_utils import parse_vlm_response
from agents.prompts import build_m2v_prompt_parts
from agents.vlm_tokens import invoke_vlm
from llm_set.llm import LLM_QWEN_VL_MAX


class M2VAgent:
    """标注截图 + 用户文本 → AC action JSON（high 含下一步规划三字段）。"""

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

        system_prompt, user_prompt = build_m2v_prompt_parts(
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
                agent="m2v",
            )
            if use_fixed_fields
            else validate_ac_action(
                parsed,
                has_annotated_nodes=has_annotated_nodes,
                mode=mode,
                agent="m2v",
            )
        )

        is_high = mode.lower() == "high"
        norm_kwargs = dict(
            has_annotated_nodes=has_annotated_nodes,
            include_target_object=is_high,
            include_next_action_type=is_high,
            instruction=instruction,
            current_step_instruction=current_step_instruction,
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
