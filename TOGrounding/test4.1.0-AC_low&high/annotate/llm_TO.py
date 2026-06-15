#!/usr/bin/env python3
"""
根据 step instruction（AC-low）或 episode goal（AC-high）生成单条 Target Object。

API:
    generate_target_object(prompt_text) -> {"target_object": "<English string>"}

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


def _to_prompt(prompt_text: str) -> str:
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
        "Examples:\n"
        '- Bad: "the search bar at the top" → Good: "search"\n'
        '- Bad: "Yahoo Mail app icon" → Good: "Yahoo Mail"\n'
        '- Bad: "the close button in the corner" → Good: "Close"\n\n'
        f'Input: "{prompt_text}"\n\n'
        "Output format:\n"
        '{"target_object": "search"}\n'
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
    response = llm_target.model.invoke(prompt, max_tokens=MAX_TOKENS)
    return response.content if hasattr(response, "content") else str(response)


def generate_target_object(prompt_text: str) -> dict:
    """
    生成单条 Target Object。

    Returns:
        {"target_object": "<English string>"}
    """
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        raise ValueError("prompt_text 为空")

    raw = _call_llm(prompt_text)
    target = _parse_target_object(raw)
    if not target:
        raise ValueError(f"无法解析 target_object: {raw[:300]}")

    return {"target_object": target}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="单条 TO 生成测试")
    parser.add_argument("text", help="instruction 或 goal")
    args = parser.parse_args()
    result = generate_target_object(args.text)
    print(json.dumps(result, ensure_ascii=False, indent=2))
