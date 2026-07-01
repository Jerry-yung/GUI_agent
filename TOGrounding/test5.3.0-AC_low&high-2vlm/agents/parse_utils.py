"""VLM 输出 JSON 解析（含 baseline 级容错）。"""

from __future__ import annotations

import json
import re


def parse_vlm_response(text: str) -> dict | None:
    """标准 AC VLM JSON 解析。"""
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else None
            except json.JSONDecodeError:
                pass
    return None


def parse_json_from_text(text: str) -> dict:
    """Baseline 级多层容错解析（用于 TO 无标注坐标回退）。无法解析时 raise ValueError。"""
    text = (text or "").strip()

    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    code_block_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```")
    for match in code_block_pattern.finditer(text):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue

    brace_pattern = re.compile(r"(\{[\s\S]*\})")
    for match in brace_pattern.finditer(text):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue

    fixed = re.sub(r'(?<=[{,])\s*"?([a-zA-Z_][a-zA-Z0-9_]*)"?\s*:', r'"\1":', text)
    try:
        data = json.loads(fixed)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass

    array_pattern = re.compile(r"\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]")
    match = array_pattern.search(text)
    if match:
        return {"x": float(match.group(1)), "y": float(match.group(2))}

    num_pair_pattern = re.compile(r'["\']?x["\']?\s*[:=]\s*([0-9.]+)[^0-9.]*([0-9.]+)')
    match = num_pair_pattern.search(text)
    if match:
        return {"x": float(match.group(1)), "y": float(match.group(2))}

    raise ValueError(f"无法从 VLM 输出中解析 JSON: {text[:200]}")
