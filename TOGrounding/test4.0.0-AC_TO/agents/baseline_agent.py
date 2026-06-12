"""
Baseline Agent：纯 VLM 端到端 GUI Agent。

输入：截图 + 单步自然语言指令
输出：归一化点击坐标 JSON {"x": 0.0~1.0, "y": 0.0~1.0}

调用 llm_set.llm 中的 vlm（多模态大模型）。
"""

import json
import re
import base64
from pathlib import Path

from langchain_core.messages import HumanMessage

# 从项目 llm 配置统一入口导入 vlm
from llm_set.llm import vlm
from agents.prompts import build_baseline_prompt


def _encode_image(image_path: str | Path) -> str:
    """将图片文件转为 base64 字符串。"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_json_from_text(text: str) -> dict:
    """
    从 VLM 返回的文本中提取 JSON 坐标对象。
    多层容错：直接解析 → 代码块 → 花括号 → 引号修复 → 数组fallback
    """
    text = text.strip()

    # 1. 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 尝试匹配 markdown 代码块中的 JSON
    code_block_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")
    for match in code_block_pattern.finditer(text):
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            continue

    # 3. 尝试匹配最外层花括号包裹的 JSON
    brace_pattern = re.compile(r"(\{[\s\S]*\})")
    for match in brace_pattern.finditer(text):
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            continue

    # 4. 尝试修复常见引号错误后解析
    # 如: {"x": 905, y": 171} -> {"x": 905, "y": 171}
    #     {x: 0.5, y: 0.3}    -> {"x": 0.5, "y": 0.3}
    fixed = re.sub(r'(?<=[{,])\s*"?([a-zA-Z_][a-zA-Z0-9_]*)"?\s*:', r'"\1":', text)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # 5. 尝试从数组格式 [x, y] 中提取
    array_pattern = re.compile(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]")
    match = array_pattern.search(text)
    if match:
        return {"x": float(match.group(1)), "y": float(match.group(2))}

    # 6. 尝试从 "x": num ... num 模式提取两个数字（允许中间有任意字符）
    num_pair_pattern = re.compile(r'["\']?x["\']?\s*[:=]\s*([0-9.]+)[^0-9.]*([0-9.]+)')
    match = num_pair_pattern.search(text)
    if match:
        return {"x": float(match.group(1)), "y": float(match.group(2))}

    raise ValueError(f"无法从 VLM 输出中解析 JSON: {text[:200]}")


class BaselineAgent:
    """
    Baseline GUI Agent。

    直接调用 VLM，输入 screenshot + instruction，
    输出归一化点击坐标 {"x": float, "y": float}。
    """

    def __init__(self):
        self.vlm = vlm

    def predict(
        self,
        screenshot_path: str | Path,
        instruction: str,
    ) -> dict:
        """
        预测单次点击的归一化坐标。

        Args:
            screenshot_path: 截图 PNG 文件路径
            instruction: 单步自然语言指令（如 "Click the Search button"）

        Returns:
            {"x": 0.52, "y": 0.31} 归一化坐标，范围 [0, 1]

        Raises:
            FileNotFoundError: screenshot_path 不存在
            ValueError: VLM 输出无法解析为有效 JSON 坐标
        """
        screenshot_path = Path(screenshot_path)
        if not screenshot_path.exists():
            raise FileNotFoundError(f"截图不存在: {screenshot_path}")

        # 编码图片
        b64_image = _encode_image(screenshot_path)

        # 构建多模态消息
        prompt = build_baseline_prompt(instruction)
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64_image}"},
                },
            ]
        )

        # 调用 VLM（vlm 是 LLM_* 包装类，真正的模型在 .model 中）
        response = self.vlm.model.invoke([message])
        raw_text = response.content

        # 解析 JSON
        result = _parse_json_from_text(raw_text)

        # 校验坐标格式
        if not isinstance(result, dict):
            raise ValueError(f"VLM 输出不是 JSON 对象: {raw_text[:200]}")
        if "x" not in result or "y" not in result:
            raise ValueError(f"VLM 输出缺少 x/y 字段: {raw_text[:200]}")

        norm_x = float(result["x"])
        norm_y = float(result["y"])

        # 简单 clamp 到 [0, 1]
        norm_x = max(0.0, min(1.0, norm_x))
        norm_y = max(0.0, min(1.0, norm_y))

        return {"x": norm_x, "y": norm_y}

    def predict_batch(
        self,
        screenshot_paths: list[str | Path],
        instructions: list[str],
    ) -> list[dict]:
        """
        批量预测。

        Args:
            screenshot_paths: 截图路径列表
            instructions: 指令列表（与截图一一对应）

        Returns:
            归一化坐标列表
        """
        if len(screenshot_paths) != len(instructions):
            raise ValueError("screenshot_paths 与 instructions 长度不一致")

        results = []
        for sp, ins in zip(screenshot_paths, instructions):
            try:
                result = self.predict(sp, ins)
            except Exception as e:
                result = {"x": 0.0, "y": 0.0, "error": str(e)}
            results.append(result)
        return results
