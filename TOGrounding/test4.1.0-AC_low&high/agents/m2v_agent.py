#!/usr/bin/env python3
"""
M2V Agent：与 M2 相同接口；high 模式下 VLM 同时输出 next_instruction + target_object。
step0 的 target_object 仍由 main 通过 llm_TO 生成。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from agents.action_validate import validate_ac_action
from agents.m2_agent import (
    VLM_TEMPERATURE,
    VLM_TOP_P,
    _encode_image,
    _normalize_action,
)
from agents.parse_utils import parse_vlm_response
from agents.prompts import build_m2v_prompt_parts
from agents.vlm_tokens import invoke_vlm
from llm_set.llm import LLM_QWEN_VL_MAX


class M2VAgent:
    """标注截图 + 用户文本 → AC action JSON（含 next_instruction 与 target_object）。"""

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
    ) -> dict:
        _ = top_k_nodes
        annotated_screenshot_path = Path(annotated_screenshot_path)
        if not annotated_screenshot_path.is_file():
            raise FileNotFoundError(f"标注截图不存在: {annotated_screenshot_path}")

        system_prompt, user_prompt = build_m2v_prompt_parts(
            mode,
            instruction=instruction,
            goal=goal,
            current_step_instruction=current_step_instruction,
            prev_step_instruction=prev_step_instruction,
            has_annotated_nodes=has_annotated_nodes,
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

        schema_ok, schema_error = validate_ac_action(
            parsed,
            has_annotated_nodes=has_annotated_nodes,
            mode=mode,
            agent="m2v",
        )

        result: dict = {
            "action": _normalize_action(
                parsed,
                mode,
                has_annotated_nodes=has_annotated_nodes,
                include_target_object=True,
                instruction=instruction,
                current_step_instruction=current_step_instruction,
            ),
            "raw_response": raw,
            "vlm_tokens": vlm_tokens,
        }
        if not schema_ok:
            result["schema_error"] = schema_error
        return result
