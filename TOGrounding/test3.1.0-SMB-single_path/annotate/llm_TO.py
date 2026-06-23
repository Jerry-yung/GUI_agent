"""根据 step instruction 生成 action_type + 中文 target_object。"""
from __future__ import annotations

import json
import re

from llm_set.llm import llm

MAX_TOKENS = 512

TO_ACTION_TYPES = frozenset({"click", "scroll", "input", "long_press"})
RETRIEVAL_ACTION_TYPES = frozenset({"click", "scroll", "long_press"})


def _normalize_action_type(action_type: str) -> str:
    action_type = str(action_type or "").strip()
    if action_type in ("long_click", "长按"):
        return "long_press"
    return action_type


def _to_prompt(prompt_text: str) -> str:
    return (
        "你是 GUI 自动化助手。根据用户当前步骤指令，预测下一步动作类型，"
        "并在需要时给出用于语义检索的目标描述。\n\n"
        "规则：\n"
        "1. action_type 必须是以下之一：click、scroll、input、long_press。\n"
        "2. 指令含点击/点按/轻点/tap/click 等 → click；含滑动/滚动/swipe/scroll → scroll；"
        "含输入/填写/input → input。\n"
        "3. long_press 仅当指令明确含「长按」「long press」「long_press」等长按时才使用；"
        "普通点击一律用 click，不要用 long_press。\n"
        "4. click / long_press：target_object 为控件可见中文标签（如「播放」「搜索」）。\n"
        "5. scroll：target_object 为应滑动的内容区域/列表简短中文描述（如「推荐内容」），"
        "不要写滑动方向。\n"
        "6. input：target_object 输出空字符串 \"\"。\n"
        "7. target_object 不要加位置词、不要加通用 UI 类型词（按钮、图标等），只输出中文。\n"
        "8. 只输出一个合法 JSON，字段 action_type（字符串）与 target_object（字符串），无 markdown。\n\n"
        "示例：\n"
        '- 「点击播放」 → {"action_type":"click","target_object":"播放"}\n'
        '- 「向下滑动推荐列表」 → {"action_type":"scroll","target_object":"推荐列表"}\n'
        '- 「输入你好」 → {"action_type":"input","target_object":""}\n'
        '- 「长按设置图标」 → {"action_type":"long_press","target_object":"设置"}\n\n'
        f'输入："{prompt_text}"\n\n'
        "输出格式：\n"
        '{"action_type": "click", "target_object": "播放"}\n'
    )


def _extract_json_candidates(text: str) -> list[dict]:
    text = (text or "").strip()
    if not text:
        return []

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

    return candidates


def _parse_to_result(text: str) -> dict | None:
    for data in _extract_json_candidates(text):
        action_type = _normalize_action_type(data.get("action_type", ""))
        if action_type not in TO_ACTION_TYPES:
            continue

        target_object = ""
        if data.get("target_object") is not None:
            target_object = str(data["target_object"]).strip()
        elif isinstance(data.get("target_objects"), list):
            for item in data["target_objects"]:
                if isinstance(item, str) and item.strip():
                    target_object = item.strip()
                    break

        if action_type in RETRIEVAL_ACTION_TYPES and not target_object:
            continue

        return {"action_type": action_type, "target_object": target_object}

    for data in _extract_json_candidates(text):
        if "target_object" in data and isinstance(data["target_object"], str):
            val = data["target_object"].strip()
            if val and "action_type" not in data:
                return {"action_type": "click", "target_object": val}

    return None


def _call_llm(prompt_text: str) -> str:
    response = llm.model.invoke(_to_prompt(prompt_text), max_tokens=MAX_TOKENS)
    return response.content if hasattr(response, "content") else str(response)


def generate_target_object(prompt_text: str) -> dict:
    """
    生成 action_type + target_object。

    Returns:
        {"action_type": str, "target_object": str, "raw_response": str}
    """
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        raise ValueError("prompt_text 为空")

    raw = _call_llm(prompt_text)
    parsed = _parse_to_result(raw)
    if parsed is None:
        raise ValueError(f"无法解析 llm_TO 输出: {raw[:300]}")

    return {
        "action_type": parsed["action_type"],
        "target_object": parsed["target_object"],
        "raw_response": raw,
    }
