"""
M2 Agent：使用 cos_sim top_k 标注截图 + 单步指令进行决策。

标注截图由 annotate.py 生成：MODE=best/mid/worst 时，先用 rank_to 选取对应 TO，
再用该 TO 与所有节点的相似度取 top_k 个候选框绘制到截图上。

输入：
    - agents/annotate/annotated_screenshots/top_{TOP_K}_{MODE}/*.png
    - Mobile3M_data/step_instructions/*.txt

输出：{"click_id": int}  （node_id）
"""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

from langchain_core.messages import HumanMessage

from agents.prompts import build_m2_prompt
from llm_set.llm import vlm


def _encode_image(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_click_id(text: str) -> int:
    text = text.strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict) and "click_id" in data:
            return int(data["click_id"])
    except (json.JSONDecodeError, ValueError):
        pass

    code_block_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")
    for match in code_block_pattern.finditer(text):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "click_id" in data:
                return int(data["click_id"])
        except (json.JSONDecodeError, ValueError):
            continue

    brace_pattern = re.compile(r"(\{[\s\S]*\})")
    for match in brace_pattern.finditer(text):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "click_id" in data:
                return int(data["click_id"])
        except (json.JSONDecodeError, ValueError):
            continue

    digit_pattern = re.compile(r"click_id\s*[:=]\s*(\d+)")
    match = digit_pattern.search(text)
    if match:
        return int(match.group(1))

    raise ValueError(f"无法从 VLM 输出中解析 click_id: {text[:200]}")


class M2Agent:
    """标注截图 + 指令 → 点击 node_id。"""

    def __init__(self):
        self.vlm = vlm

    def predict(self, annotated_screenshot_path: str | Path, instruction: str) -> dict:
        annotated_screenshot_path = Path(annotated_screenshot_path)
        if not annotated_screenshot_path.exists():
            raise FileNotFoundError(f"标注截图不存在: {annotated_screenshot_path}")

        b64_image = _encode_image(annotated_screenshot_path)
        prompt = build_m2_prompt(instruction)

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                },
            ]
        )

        response = self.vlm.model.invoke([message])
        click_id = _parse_click_id(response.content)
        return {"click_id": click_id}

    def predict_batch(
        self,
        annotated_screenshot_paths: list[str | Path],
        instructions: list[str],
    ) -> list[dict]:
        if len(annotated_screenshot_paths) != len(instructions):
            raise ValueError("annotated_screenshot_paths 与 instructions 长度不一致")

        results = []
        for asp, ins in zip(annotated_screenshot_paths, instructions):
            try:
                result = self.predict(asp, ins)
            except Exception as e:
                result = {"click_id": -1, "error": str(e)}
            results.append(result)
        return results
