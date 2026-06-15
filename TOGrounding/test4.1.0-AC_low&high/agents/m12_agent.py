#!/usr/bin/env python3
"""
M12 Agent：与 M2 相同流水线；额外将 top-k 候选节点的文本描述注入 VLM prompt。
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
from agents.prompts import build_m12_prompt_parts
from agents.vlm_tokens import invoke_vlm
from annotate.topk_node_context import build_topk_candidate_table
from llm_set.llm import LLM_QWEN_VL_MAX


class M12Agent:
    """标注截图 + top-k 候选语义表 + instruction/goal → AC action JSON。"""

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
        target_object: str = "",
        stem: str = "",
        has_annotated_nodes: bool = True,
        top_k_nodes: list | None = None,
    ) -> dict:
        annotated_screenshot_path = Path(annotated_screenshot_path)
        if not annotated_screenshot_path.is_file():
            raise FileNotFoundError(f"标注截图不存在: {annotated_screenshot_path}")

        candidate_table = ""
        if has_annotated_nodes and top_k_nodes and stem:
            candidate_table = build_topk_candidate_table(
                stem,
                top_k_nodes,
                target_object=target_object,
            )

        system_prompt, user_prompt = build_m12_prompt_parts(
            mode,
            instruction=instruction,
            goal=goal,
            current_step_instruction=current_step_instruction,
            prev_step_instruction=prev_step_instruction,
            candidate_nodes_table=candidate_table,
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
            agent="m12",
        )

        result: dict = {
            "action": _normalize_action(
                parsed,
                mode,
                has_annotated_nodes=has_annotated_nodes,
                instruction=instruction,
                current_step_instruction=current_step_instruction,
            ),
            "raw_response": raw,
            "vlm_tokens": vlm_tokens,
            "candidate_nodes_table": candidate_table or None,
        }
        if not schema_ok:
            result["schema_error"] = schema_error
        return result
