"""根据 GT step_instruction 生成中文 target_object。"""
from __future__ import annotations

import json
import re

from llm_set.llm import llm

MAX_TOKENS = 512


def _to_prompt(prompt_text: str) -> str:
    return (
        "你是 GUI 自动化助手。根据用户当前步骤指令，预测用户下一步应交互的界面元素名称。\n\n"
        "规则：\n"
        "1. 只输出元素本身的可见名称/标签，不要输出其他内容。\n"
        "2. 不要添加位置或布局描述（如顶部、底部、左侧、屏幕上）。\n"
        "3. 不要添加通用 UI 类型词（如按钮、图标、链接），除非它们是可见标签的一部分。\n"
        "4. 优先输出最短且忠实的中文标签，例如「播放」「设置」「搜索」。\n"
        "5. 只输出中文。\n"
        "6. 只输出一个合法 JSON 对象，且仅包含 target_object 字段（字符串）。\n"
        "7. 不要输出 markdown 代码块或解释文字。\n\n"
        "示例：\n"
        '- 差：「顶部的搜索栏」 → 好：「搜索」\n'
        '- 差：「QQ音乐应用图标」 → 好：「QQ音乐」\n'
        '- 差：「角落的关闭按钮」 → 好：「关闭」\n\n'
        f'输入："{prompt_text}"\n\n'
        "输出格式：\n"
        '{"target_object": "播放"}\n'
    )


def _parse_target_object(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    candidates: list[dict] = []
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            candidates.append(data)
    except (json.JSONDecodeError, ValueError):
        pass

    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                candidates.append(data)
        except (json.JSONDecodeError, ValueError):
            continue

    for match in re.finditer(r"(\{[\s\S]*\})", text):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                candidates.append(data)
        except (json.JSONDecodeError, ValueError):
            continue

    for data in candidates:
        if "target_object" in data and isinstance(data["target_object"], str):
            val = data["target_object"].strip()
            if val:
                return val
        if "target_objects" in data and isinstance(data["target_objects"], list):
            for item in data["target_objects"]:
                if isinstance(item, str) and item.strip():
                    return item.strip()

    return ""


def _call_llm(prompt_text: str) -> str:
    prompt = _to_prompt(prompt_text)
    response = llm.model.invoke(prompt, max_tokens=MAX_TOKENS)
    return response.content if hasattr(response, "content") else str(response)


def generate_target_object(prompt_text: str) -> dict:
    """生成单条中文 Target Object。"""
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        raise ValueError("prompt_text 为空")

    raw = _call_llm(prompt_text)
    target = _parse_target_object(raw)
    if not target:
        raise ValueError(f"无法解析 target_object: {raw[:300]}")

    return {"target_object": target}
