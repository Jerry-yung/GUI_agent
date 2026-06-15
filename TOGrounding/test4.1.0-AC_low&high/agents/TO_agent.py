#!/usr/bin/env python3
"""
TO Agent：top-1 标注截图 + instruction/goal → Android Control 格式 action。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from agents.m2_agent import VLM_TEMPERATURE, VLM_TOP_P, _encode_image, _normalize_action
from agents.action_validate import validate_ac_action
from agents.parse_utils import parse_json_from_text, parse_vlm_response
from agents.prompts import build_to_vlm_prompt_parts
from agents.vlm_tokens import invoke_vlm
from llm_set.llm import LLM_QWEN_VL_MAX


class TOAgent:
    """top-1 标注截图 + 用户文本 → AC action JSON。"""

    def __init__(self):
        self.vlm = LLM_QWEN_VL_MAX(temperature=VLM_TEMPERATURE, top_p=VLM_TOP_P)

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
    ) -> dict:
        annotated_screenshot_path = Path(annotated_screenshot_path)
        if not annotated_screenshot_path.is_file():
            raise FileNotFoundError(f"标注截图不存在: {annotated_screenshot_path}")

        system_prompt, user_prompt = build_to_vlm_prompt_parts(
            mode,
            instruction=instruction,
            goal=goal,
            target_object=target_object,
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

        schema_ok, schema_error = validate_ac_action(
            parsed,
            has_annotated_nodes=has_annotated_nodes,
            mode=mode,
            agent="TO",
        )

        action = _normalize_action(
            parsed,
            mode,
            has_annotated_nodes=has_annotated_nodes,
            skip_pointer_node_id=has_annotated_nodes,
            skip_input_node_id=True,
            instruction=instruction,
            current_step_instruction=current_step_instruction,
        )

        if (
            has_annotated_nodes
            and top_k_nodes
            and action.get("action_type") in ("click", "long_press")
        ):
            action["node_id"] = int(top_k_nodes[0]["node_id"])
            action.pop("x", None)
            action.pop("y", None)

        result: dict = {"action": action, "raw_response": raw, "vlm_tokens": vlm_tokens}
        if not schema_ok:
            result["schema_error"] = schema_error
        return result
