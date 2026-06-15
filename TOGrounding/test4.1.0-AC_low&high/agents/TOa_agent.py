#!/usr/bin/env python3
"""
TOa Agent：top-1 建议框（无 #）+ instruction → AC action；
click/long_press 可输出归一化坐标，否则回退 top1 node。
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from agents.m2_agent import VLM_TEMPERATURE, VLM_TOP_P, _encode_image, _normalize_action
from agents.action_validate import validate_ac_action
from agents.parse_utils import parse_json_from_text, parse_vlm_response
from agents.prompts import TAU_MARGIN_STRONG, TAU_SIM_STRONG, build_toa_prompt_parts
from agents.vlm_tokens import invoke_vlm
from llm_set.llm import LLM_QWEN_VL_MAX


def _has_valid_norm_coords(action: dict) -> bool:
    try:
        x = float(action["x"])
        y = float(action["y"])
    except (KeyError, TypeError, ValueError):
        return False
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def _apply_pointer_locator(
    action: dict,
    *,
    has_annotated_nodes: bool,
    top_k_nodes: list | None,
) -> tuple[dict, str | None]:
    """click/long_press 定位仲裁：coords 优先，否则 top1。"""
    action_type = action.get("action_type")
    if action_type not in ("click", "long_press"):
        return action, None

    if _has_valid_norm_coords(action):
        action = dict(action)
        action.pop("node_id", None)
        action.pop("click_id", None)
        return action, "coords"

    if has_annotated_nodes and top_k_nodes:
        action = dict(action)
        action["node_id"] = int(top_k_nodes[0]["node_id"])
        action.pop("x", None)
        action.pop("y", None)
        return action, "top1"

    if _has_valid_norm_coords(action):
        return action, "coords"

    return action, None


class TOaAgent:
    """top-1 建议框 + 用户文本 → AC action JSON（可选坐标）。"""

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
        retrieval_final_sim: float = 0.0,
        retrieval_margin: float | None = None,
    ) -> dict:
        annotated_screenshot_path = Path(annotated_screenshot_path)
        if not annotated_screenshot_path.is_file():
            raise FileNotFoundError(f"标注截图不存在: {annotated_screenshot_path}")

        system_prompt, user_prompt = build_toa_prompt_parts(
            mode,
            instruction=instruction,
            goal=goal,
            target_object=target_object,
            current_step_instruction=current_step_instruction,
            prev_step_instruction=prev_step_instruction,
            has_annotated_nodes=has_annotated_nodes,
            retrieval_final_sim=retrieval_final_sim,
            retrieval_margin=retrieval_margin,
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
            agent="TOa",
        )

        action = _normalize_action(
            parsed,
            mode,
            has_annotated_nodes=has_annotated_nodes,
            skip_pointer_node_id=True,
            keep_pointer_coords=True,
            skip_input_node_id=True,
            instruction=instruction,
            current_step_instruction=current_step_instruction,
        )

        locator_source: str | None = None
        if action.get("action_type") in ("click", "long_press"):
            action, locator_source = _apply_pointer_locator(
                action,
                has_annotated_nodes=has_annotated_nodes,
                top_k_nodes=top_k_nodes,
            )
            if locator_source is None:
                return {
                    "action": None,
                    "raw_response": raw,
                    "vlm_tokens": vlm_tokens,
                    "error": "click/long_press 缺少有效 x,y 或 top1 候选",
                }

        result: dict = {
            "action": action,
            "raw_response": raw,
            "vlm_tokens": vlm_tokens,
            "retrieval_final_sim": retrieval_final_sim,
            "retrieval_margin": retrieval_margin,
        }
        if locator_source:
            result["locator_source"] = locator_source
        if not schema_ok:
            result["schema_error"] = schema_error
        return result


__all__ = [
    "TOaAgent",
    "TAU_SIM_STRONG",
    "TAU_MARGIN_STRONG",
]
