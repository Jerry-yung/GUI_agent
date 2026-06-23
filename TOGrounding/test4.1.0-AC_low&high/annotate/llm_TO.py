#!/usr/bin/env python3
"""
根据 step instruction（AC-low）或 episode goal（AC-high step0）生成 action plan。

API:
    generate_target_object(prompt_text) -> {
        "action_type": str,
        "target_object": str,
        "raw_response": str,  # optional, for logging
    }

AC-low: 每步 llm_TO 负责 action_type + target_object（pointer 步）。
AC-high: 仅 step0 调用 generate_target_object(goal)；step1 起由上步 VLM 的
next_instruction + target_object + next_action_type 驱动检索。
generate_retrieval_target 保留兼容，high 主流程不再调用。
不落盘 TO embedding；不写入 TO_index / TO_emb。
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_set.llm import llm_target

MAX_TOKENS = 512

TO_ACTION_TYPES = frozenset({
    "click",
    "long_press",
    "scroll",
    "input_text",
    "wait",
    "navigate_back",
    "navigate_home",
})

POINTER_ACTION_TYPES = frozenset({"click", "long_press"})


def _normalize_action_type(action_type: str) -> str:
    action_type = str(action_type or "").strip()
    if action_type == "long_click":
        return "long_press"
    return action_type


def _to_prompt(prompt_text: str) -> str:
    return (
        "You are a GUI automation assistant. Given a user instruction or task goal, "
        "predict the NEXT action type and, when tapping is required, the on-screen "
        "text label of the UI element to interact with.\n\n"
        "Rules:\n"
        "1. action_type must be one of: click, long_press, scroll, input_text, wait, "
        "navigate_back, navigate_home.\n"
        "2. Use scroll when the step asks to scroll/swipe; input_text when typing or "
        "searching text; wait for loading/pause; navigate_back for system back; "
        "navigate_home for home screen.\n"
        "3. target_object: ONLY for click or long_press — the element's own visible "
        "label/name in English. For all other action types output \"\".\n"
        "4. Do NOT add position words (at the top, bottom, left, on the screen) to "
        "target_object.\n"
        "5. Do NOT add generic UI-type words (icon, button, link, bar, field, tab) "
        "unless part of the actual visible label.\n"
        "6. Prefer the shortest faithful label: e.g. \"search\", \"Yahoo\", \"Past\".\n"
        "7. Do NOT output Chinese or other non-English text in target_object.\n"
        "8. Output ONLY a valid JSON object with fields action_type (string) and "
        "target_object (string). No markdown.\n\n"
        "Examples:\n"
        '- "Go to the Past section" → {"action_type":"click","target_object":"Past"}\n'
        '- "Swipe up to view reviews" → {"action_type":"scroll","target_object":""}\n'
        '- "Type hello in the search bar" → {"action_type":"input_text","target_object":""}\n'
        '- "Wait for the page to load" → {"action_type":"wait","target_object":""}\n\n'
        f'Input: "{prompt_text}"\n\n'
        "Output format:\n"
        '{"action_type": "click", "target_object": "Past"}\n'
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
        if "target_object" in data and data["target_object"] is not None:
            target_object = str(data["target_object"]).strip()
        elif action_type in POINTER_ACTION_TYPES:
            # legacy single-field TO output
            legacy = data.get("target_object")
            if isinstance(legacy, str) and legacy.strip():
                target_object = legacy.strip()
            elif isinstance(data.get("target_objects"), list):
                for item in data["target_objects"]:
                    if isinstance(item, str) and item.strip():
                        target_object = item.strip()
                        break

        if action_type in POINTER_ACTION_TYPES and not target_object:
            # allow parsing if only target_object field exists (old format)
            if "target_object" not in data and "action_type" not in data:
                continue
            if "target_object" in data or "target_objects" in data:
                continue
            # old format: {"target_object": "..."} only — infer click
            for legacy_data in _extract_json_candidates(text):
                if "target_object" in legacy_data:
                    t = str(legacy_data.get("target_object", "")).strip()
                    if t:
                        return {"action_type": "click", "target_object": t}
            continue

        return {"action_type": action_type, "target_object": target_object}

    # pure legacy: {"target_object": "..."} without action_type
    for data in _extract_json_candidates(text):
        if "target_object" in data and isinstance(data["target_object"], str):
            val = data["target_object"].strip()
            if val and "action_type" not in data:
                return {"action_type": "click", "target_object": val}

    return None


def _call_llm(prompt_text: str) -> str:
    prompt = _to_prompt(prompt_text)
    response = llm_target.model.invoke(prompt, max_tokens=MAX_TOKENS)
    return response.content if hasattr(response, "content") else str(response)


def _to_prompt_retrieval_only(prompt_text: str) -> str:
    """AC-high 检索用：仅预测 target_object（保持旧行为）。"""
    return (
        "You are a GUI automation assistant. Given a user instruction or task goal, "
        "predict the on-screen text label or name of the UI element the user should interact with next.\n\n"
        "Rules:\n"
        "1. Output ONLY the element's own label/name — nothing else.\n"
        "2. Do NOT add position or layout words (e.g. at the top, bottom, left, on the screen).\n"
        "3. Do NOT add generic UI-type words (e.g. icon, button, link, bar, field, tab) "
        "unless they are part of the actual visible label.\n"
        "4. Prefer the shortest faithful label: e.g. \"search\", \"Yahoo\", \"Yahoo Mail\".\n"
        "5. Do NOT output Chinese or other non-English text.\n"
        "6. Output ONLY a valid JSON object with exactly one field: target_object (string).\n"
        "7. Do NOT include markdown code blocks or explanations.\n\n"
        f'Input: "{prompt_text}"\n\n'
        "Output format:\n"
        '{"target_object": "search"}\n'
    )


def _parse_target_object_only(text: str) -> str:
    for data in _extract_json_candidates(text):
        if "target_object" in data and isinstance(data["target_object"], str):
            val = data["target_object"].strip()
            if val:
                return val
        if "target_objects" in data and isinstance(data["target_objects"], list):
            for item in data["target_objects"]:
                if isinstance(item, str) and item.strip():
                    return item.strip()
    return ""


def generate_retrieval_target(prompt_text: str) -> dict:
    """
    AC-high 检索专用：仅生成 target_object。

    Returns:
        {"target_object": str, "raw_response": str}
    """
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        raise ValueError("prompt_text 为空")

    raw = _call_llm_retrieval_only(prompt_text)
    target = _parse_target_object_only(raw)
    if not target:
        raise ValueError(f"无法解析 target_object: {raw[:300]}")

    return {"target_object": target, "raw_response": raw}


def _call_llm_retrieval_only(prompt_text: str) -> str:
    prompt = _to_prompt_retrieval_only(prompt_text)
    response = llm_target.model.invoke(prompt, max_tokens=MAX_TOKENS)
    return response.content if hasattr(response, "content") else str(response)


def generate_target_object(prompt_text: str) -> dict:
    """
    生成 action plan（AC-low 主路径）。

    Returns:
        {"action_type": str, "target_object": str, "raw_response": str}
    """
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        raise ValueError("prompt_text 为空")

    raw = _call_llm(prompt_text)
    parsed = _parse_to_result(raw)
    if not parsed:
        raise ValueError(f"无法解析 llm_TO 输出: {raw[:300]}")

    action_type = parsed["action_type"]
    target_object = parsed.get("target_object", "")
    if action_type in POINTER_ACTION_TYPES and not target_object:
        raise ValueError(
            f"click/long_press 需要非空 target_object: {raw[:300]}"
        )

    return {
        "action_type": action_type,
        "target_object": target_object,
        "raw_response": raw,
    }


def generate_to_plan(prompt_text: str) -> dict:
    """别名，与 generate_target_object 相同。"""
    return generate_target_object(prompt_text)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="单条 TO plan 生成测试")
    parser.add_argument("text", help="instruction 或 goal")
    args = parser.parse_args()
    result = generate_target_object(args.text)
    out = {k: v for k, v in result.items() if k != "raw_response"}
    print(json.dumps(out, ensure_ascii=False, indent=2))
