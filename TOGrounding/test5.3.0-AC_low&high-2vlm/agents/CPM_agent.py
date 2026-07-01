#!/usr/bin/env python3
"""
CPM Agent：原图 + AC 风格 JSON（归一化 x,y）→ pred_action。

与 m2 无标注回退一致；内部忽略标注图，使用 step 原图（长边 ≤1120 resize）。
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image

from agents.cpm_convert import parse_and_convert_cpm
from agents.cpm_prompts import build_cpm_prompt_parts
from agents.m2_agent import VLM_TEMPERATURE, VLM_TOP_P
from agents.vlm_tokens import invoke_vlm
from llm_set.llm import LLM_QWEN_VL_MAX

MAX_IMAGE_LINE = 1120


def _resize_image_bytes(image_path: str | Path, *, max_line: int = MAX_IMAGE_LINE) -> str:
    with Image.open(image_path) as origin_img:
        img = origin_img.convert("RGB")
        w, h = img.size
        if h > max_line:
            w = int(w * max_line / h)
            h = max_line
        if w > max_line:
            h = int(h * max_line / w)
            w = max_line
        img = img.resize((w, h), resample=Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


class CPMAgent:
    """原图 + AC 风格 JSON → pred_action（click/long_press 使用归一化 x,y）。"""

    def __init__(self):
        self.vlm = LLM_QWEN_VL_MAX(temperature=VLM_TEMPERATURE, top_p=VLM_TOP_P)

    def predict(
        self,
        annotated_screenshot_path: str | Path,
        mode: str,
        *,
        instruction: str = "",
        goal: str = "",
        has_annotated_nodes: bool = True,
        top_k_nodes: list | None = None,
        **kwargs,
    ) -> dict:
        _ = has_annotated_nodes, top_k_nodes, kwargs
        screenshot_path = Path(annotated_screenshot_path)
        if not screenshot_path.is_file():
            raise FileNotFoundError(f"截图不存在: {screenshot_path}")

        system_prompt, user_prompt = build_cpm_prompt_parts(
            mode,
            instruction=instruction,
            goal=goal,
        )
        b64_image = _resize_image_bytes(screenshot_path)

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

        cpm_action, ac_action, schema_error = parse_and_convert_cpm(raw, mode=mode)
        if cpm_action is None:
            return {
                "action": None,
                "cpm_action": None,
                "raw_response": raw,
                "vlm_tokens": vlm_tokens,
                "error": "无法解析 VLM JSON 输出",
            }
        if ac_action is None:
            return {
                "action": None,
                "cpm_action": cpm_action,
                "raw_response": raw,
                "vlm_tokens": vlm_tokens,
                "error": schema_error or "无法将 CPM action 转为 AC 格式",
            }

        result: dict = {
            "action": ac_action,
            "cpm_action": cpm_action,
            "raw_response": raw,
            "vlm_tokens": vlm_tokens,
        }
        if schema_error:
            result["schema_error"] = schema_error
        return result
