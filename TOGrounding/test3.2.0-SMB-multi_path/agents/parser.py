"""JSON parser for N / legacy cN/sN SMAN actions."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from utils.click_res import format_click_xy_res

_TYPE_ALIASES = {
    "click": "click",
    "tap": "click",
    "press": "click",
    "点击": "click",
    "点按": "click",
    "long_press": "long_press",
    "长按": "long_press",
    "long press": "long_press",
    "scroll": "scroll",
    "swipe": "scroll",
    "滑动": "scroll",
    "滑": "scroll",
    "input": "input",
    "type": "input",
    "输入": "input",
    "back": "back",
    "返回": "back",
}

VALID_SCROLL = frozenset({"up", "down", "left", "right"})

_DIR_ALIASES = {
    "up": "up",
    "down": "down",
    "left": "left",
    "right": "right",
    "向上": "up",
    "上": "up",
    "向下": "down",
    "下": "down",
    "向左": "left",
    "左": "left",
    "向右": "right",
    "右": "right",
}

_ELEMENT_RE = re.compile(r"\b(?:([cs]\d+)|(\d+))\b", re.IGNORECASE)
_ACTION_CALL_RE = re.compile(
    r"(?P<kind>click|scroll|input|back)\s*\((?P<args>.*)\)\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _normalize_json_text(text: str) -> str:
    # Do not map curly/smart quotes to ASCII " inside JSON values — VLM often writes
    # thought like: "在"听书"页面…" with U+201C/U+201D, which is valid JSON until
    # normalized; converting them to " breaks string boundaries and causes parse_error.
    return text


def _fix_trailing_commas(text: str) -> str:
    prev = None
    cur = text
    while prev != cur:
        prev = cur
        cur = re.sub(r",(\s*[}\]])", r"\1", cur)
    return cur


def _try_load_json_dict(text: str) -> dict[str, Any] | None:
    for candidate in (text, _fix_trailing_commas(text)):
        try:
            data = json.loads(candidate)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            continue
    return None


def _find_matching_brace(text: str, start: int) -> int:
    depth = 0
    in_str = False
    escape = False
    quote = ""
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == quote:
                in_str = False
            continue
        if ch in ('"', "'"):
            in_str = True
            quote = ch
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _json_candidates(text: str) -> list[str]:
    text = _normalize_json_text(text.strip())
    if not text:
        return []

    chunks: list[str] = []
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
    if fence:
        chunks.append(fence.group(1).strip())

    chunks.append(text)

    out: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk and chunk not in seen:
            out.append(chunk)
            seen.add(chunk)
        pos = 0
        while pos < len(chunk):
            start = chunk.find("{", pos)
            if start < 0:
                break
            end = _find_matching_brace(chunk, start)
            if end > start:
                snippet = chunk[start : end + 1].strip()
                if snippet not in seen:
                    out.append(snippet)
                    seen.add(snippet)
                pos = end + 1
            else:
                pos = start + 1
    return out


def extract_json_object(text: str) -> dict[str, Any] | None:
    for candidate in _json_candidates(text):
        data = _try_load_json_dict(candidate)
        if data is not None:
            return data
    return None


def _parse_action_call_string(action_str: str) -> dict[str, Any] | None:
    text = (action_str or "").strip()
    if not text:
        return None
    m = _ACTION_CALL_RE.match(text)
    if not m:
        if _ELEMENT_RE.fullmatch(text):
            return {"type": "click", "element": text.lower()}
        return None

    kind = _normalize_type(m.group("kind"))
    args = (m.group("args") or "").strip()
    if kind == "back":
        return {"type": "back"}
    if kind == "input":
        text_val = args.strip().strip('"').strip("'")
        return {"type": "input", "text": text_val} if text_val else None
    if kind == "click":
        label = _extract_label(args, kind="click")
        return {"type": "click", "element": label} if label else None
    if kind == "scroll":
        parts = [p.strip() for p in args.split(",", 1)]
        if len(parts) != 2:
            return None
        label = _extract_label(parts[0], kind="scroll")
        direction = _normalize_direction(parts[1])
        if label and direction:
            return {"type": "scroll", "element": label, "direction": direction}
    return None


def _coerce_action_dict(data: dict[str, Any]) -> dict[str, Any] | None:
    action = data.get("action")
    if isinstance(action, dict):
        return data
    if isinstance(action, str):
        parsed = _parse_action_call_string(action)
        if parsed:
            return {**data, "action": parsed}
    if isinstance(data.get("type"), str):
        action_dict = {
            "type": data.get("type"),
            "element": data.get("element", data.get("label", data.get("target", ""))),
            "direction": data.get("direction"),
            "text": data.get("text"),
        }
        return {**data, "action": action_dict}
    return None


def _normalize_type(raw: str) -> str:
    key = (raw or "").strip().lower()
    return _TYPE_ALIASES.get(key, key)


def _normalize_direction(raw: str) -> str | None:
    key = (raw or "").strip().lower().strip("\"'")
    mapped = _DIR_ALIASES.get(key, key)
    return mapped if mapped in VALID_SCROLL else None


def _extract_norm_coords(action: dict[str, Any]) -> tuple[float, float] | None:
    for xk, yk in (("x", "y"), ("norm_x", "norm_y")):
        if xk not in action or yk not in action:
            continue
        try:
            x = float(action[xk])
            y = float(action[yk])
        except (TypeError, ValueError):
            continue
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return x, y
    return None


def _extract_label(raw: str, *, kind: str) -> str | None:
    text = (raw or "").strip().strip('"').strip("'")
    if not text:
        return None
    if text.startswith("#"):
        text = text[1:]
    m = _ELEMENT_RE.search(text)
    if m:
        legacy = m.group(1)
        if legacy:
            return legacy.lower()
        digit = m.group(2)
        if digit:
            return digit
    text_l = text.lower()
    if re.fullmatch(r"\d+", text_l):
        return text_l
    prefix = "c" if kind == "click" else "s"
    if re.fullmatch(rf"{prefix}\d+", text_l):
        return text_l
    return None


def _label_from_element_match(m: re.Match[str]) -> str:
    legacy = m.group(1)
    if legacy:
        return legacy.lower()
    digit = m.group(2)
    return digit if digit else ""


def _find_element_labels(text: str) -> list[str]:
    labels: list[str] = []
    for m in _ELEMENT_RE.finditer(text):
        label = _label_from_element_match(m)
        if label:
            labels.append(label)
    return labels


def _infer_from_summary(summary: str) -> tuple[str, str] | None:
    if not summary:
        return None
    clicks = _find_element_labels(summary)
    if not clicks:
        return None
    label = clicks[-1].lower()
    if label.startswith("s"):
        kind = "scroll"
    elif label.startswith("c"):
        kind = "click"
    else:
        kind = "click"
    return kind, label


@dataclass(frozen=True)
class LabeledParseResult:
    res: list[str]
    thought: str
    summary: str


def _parse_action_dict(
    data: dict[str, Any], *, allow_click_xy: bool = True
) -> LabeledParseResult | None:
    coerced = _coerce_action_dict(data)
    if coerced is None:
        return None

    thought = str(coerced.get("thought") or "")
    summary = str(coerced.get("step_summary") or coerced.get("summary") or "")
    trailing = summary or thought
    action = coerced.get("action")
    if not isinstance(action, dict):
        return None

    act_type = _normalize_type(str(action.get("type", "")))
    if act_type in ("terminate", "finish", "done"):
        return None
    element_raw = action.get("element", action.get("label", action.get("target", "")))

    if act_type == "back":
        return LabeledParseResult(["back", trailing], thought, summary)
    if act_type == "input":
        text = action.get("text")
        if text is None:
            return None
        return LabeledParseResult(["input", str(text), trailing], thought, summary)

    if act_type == "click" or (not act_type and _extract_label(str(element_raw), kind="click")):
        if allow_click_xy:
            coords = _extract_norm_coords(action)
            if coords is not None and (
                act_type == "click" or not _extract_label(str(element_raw), kind="click")
            ):
                xy_res = format_click_xy_res(coords[0], coords[1], trailing)
                return LabeledParseResult(xy_res, thought, summary)
        element = _extract_label(str(element_raw), kind="click")
        if element is None:
            node_index = action.get("node_index")
            if node_index is not None:
                try:
                    element = str(int(node_index))
                except (TypeError, ValueError):
                    element = None
        if element is None:
            inferred = _infer_from_summary(trailing)
            if inferred and inferred[0] == "click":
                element = inferred[1]
        if element is None:
            return None
        return LabeledParseResult(["click", element, trailing], thought, summary)

    if act_type == "scroll" or (not act_type and _extract_label(str(element_raw), kind="scroll")):
        element = _extract_label(str(element_raw), kind="scroll")
        if element is None:
            return None
        direction = _normalize_direction(str(action.get("direction", "")))
        if direction is None:
            direction = _normalize_direction(str(coerced.get("direction", "")))
        if direction is None:
            return None
        return LabeledParseResult(["scroll", element, direction, trailing], thought, summary)

    if not act_type:
        inferred = _infer_from_summary(trailing)
        if inferred:
            kind, label = inferred
            if kind == "click":
                return LabeledParseResult(["click", label, trailing], thought, summary)
            direction = _normalize_direction(str(action.get("direction", "")))
            if direction:
                return LabeledParseResult(["scroll", label, direction, trailing], thought, summary)

    return None


def parse_labeled_json_fields(
    rsp: str, *, allow_click_xy: bool = True
) -> LabeledParseResult | None:
    data = extract_json_object(rsp)
    if data is not None:
        parsed = _parse_action_dict(data, allow_click_xy=allow_click_xy)
        if parsed is not None:
            return parsed

    # Fallback: function-call style without JSON wrapper.
    text = _normalize_json_text((rsp or "").strip())
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("{") or line.startswith("```"):
            continue
        for prefix in ("Action:", "action:", "动作:", "操作:"):
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()
        action_dict = _parse_action_call_string(line)
        if action_dict:
            wrapped = {"thought": "", "action": action_dict}
            return _parse_action_dict(wrapped, allow_click_xy=allow_click_xy)

    call = _parse_action_call_string(text)
    if call:
        return _parse_action_dict(
            {"thought": "", "action": call}, allow_click_xy=allow_click_xy
        )

    return None


def parse_labeled_json(rsp: str) -> list[str] | None:
    parsed = parse_labeled_json_fields(rsp)
    return parsed.res if parsed else None


def extract_thought_summary(rsp: str) -> tuple[str, str]:
    data = extract_json_object(rsp)
    if not isinstance(data, dict):
        return "", ""
    thought = str(data.get("thought") or "")
    summary = str(data.get("step_summary") or data.get("summary") or "")
    return thought, summary


def _exec_action_type(fixed_action_type: str) -> str:
    act_type = _normalize_type(fixed_action_type)
    if act_type == "long_press":
        return "click"
    return act_type


def parse_labeled_json_fixed(
    rsp: str,
    fixed_action_type: str,
    *,
    allow_click_xy: bool = True,
) -> list[str] | None:
    """解析 VLM 输出并注入 llm_TO 固定的 action_type。"""
    exec_type = _exec_action_type(fixed_action_type)
    data = extract_json_object(rsp)
    if data is not None:
        coerced = _coerce_action_dict(data)
        if coerced is not None:
            action = coerced.get("action")
            if not isinstance(action, dict):
                action = {}
            if exec_type == "back":
                patched = {"type": "back"}
            elif exec_type == "input":
                patched = {**action, "type": "input"}
            else:
                patched = {**action, "type": exec_type}
            parsed = _parse_action_dict(
                {**coerced, "action": patched},
                allow_click_xy=allow_click_xy,
            )
            if parsed is not None:
                return parsed.res
        if exec_type == "back" and data.get("thought") is not None:
            parsed = _parse_action_dict(
                {"thought": data.get("thought", ""), "action": {"type": "back"}},
                allow_click_xy=allow_click_xy,
            )
            if parsed is not None:
                return parsed.res

    return parse_labeled_json(rsp)


def coerce_fixed_pointer_response(
    rsp: str,
    fixed_action_type: str,
) -> dict[str, Any] | None:
    """TO 固定 click/long_press：允许仅输出 thought。"""
    if _normalize_type(fixed_action_type) not in ("click", "long_press"):
        return extract_json_object(rsp)
    data = extract_json_object(rsp)
    if data is None:
        return None
    action = data.get("action")
    if isinstance(action, dict) and action:
        return data
    thought = str(data.get("thought", "")).strip()
    if thought:
        return {"thought": thought, "action": {"type": "click"}}
    return data


def parse_labeled_json_fields_fixed(
    rsp: str,
    fixed_action_type: str,
    *,
    allow_click_xy: bool = True,
) -> LabeledParseResult | None:
    res = parse_labeled_json_fixed(
        rsp, fixed_action_type, allow_click_xy=allow_click_xy
    )
    if res is None:
        return None
    thought, summary = extract_thought_summary(rsp)
    if not summary:
        summary = res[-1] if res else ""
    if not thought:
        thought = summary
    return LabeledParseResult(res=res, thought=thought, summary=summary)
