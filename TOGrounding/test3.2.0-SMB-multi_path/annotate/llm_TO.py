"""根据 step instruction 与 multipath 上下文生成 action_type + 中文 target_object。"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from llm_set.llm import llm

MAX_TOKENS = 512
MAX_ACTION_HISTORY = 3

TO_ACTION_TYPES = frozenset({"click", "scroll", "input", "long_press", "back"})
RETRIEVAL_ACTION_TYPES = frozenset({"click", "scroll", "long_press"})


@dataclass
class LlmToContext:
    current_page: str = ""
    last_summary: str = ""
    last_action: str = ""
    action_history: list[str] = field(default_factory=list)


def format_action_history_line(
    *,
    step_page: str,
    next_page: str,
    step_instruction: str,
    llm_action_type: str = "",
    target_object: str = "",
    pred: str = "",
) -> str:
    parts = [f"@{step_page}"]
    if llm_action_type:
        parts.append(f"llm={llm_action_type}")
    if target_object:
        parts.append(f'TO="{target_object}"')
    if pred:
        parts.append(f"pred={pred}")
    if next_page and next_page != step_page:
        parts.append(f"→{next_page}")
    instr = (step_instruction or "").strip()
    if instr:
        short = instr if len(instr) <= 48 else instr[:45] + "..."
        parts.append(f'instr="{short}"')
    return " ".join(parts)


def _context_block(ctx: LlmToContext | None) -> str:
    if ctx is None:
        return ""
    lines: list[str] = ["【上下文】"]
    if ctx.current_page.strip():
        lines.append(f"当前页面：{ctx.current_page.strip()}")
    if ctx.last_summary.strip():
        lines.append(f"上一步 summary：{ctx.last_summary.strip()}")
    if ctx.last_action.strip():
        lines.append(f"上一步执行：{ctx.last_action.strip()}")
    history = [line.strip() for line in ctx.action_history if line.strip()]
    if history:
        lines.append("最近动作历史（最多 3 步）：")
        for idx, line in enumerate(history, start=1):
            lines.append(f"  Step{idx}: {line}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines) + "\n\n"


def _normalize_action_type(action_type: str) -> str:
    action_type = str(action_type or "").strip()
    if action_type in ("long_click", "长按"):
        return "long_press"
    return action_type


def _to_prompt(prompt_text: str, *, ctx: LlmToContext | None = None) -> str:
    context_text = _context_block(ctx)
    return (
        "你是 GUI 自动化助手。根据用户当前步骤指令与 multipath 上下文，"
        "预测下一步动作类型，并在需要时给出用于语义检索的目标描述。\n\n"
        "规则：\n"
        "1. action_type 必须是以下之一：click、scroll、input、long_press、back。\n"
        "2. 指令含点击/点按 → click；滑动/滚动 → scroll；输入/填写 → input；"
        "返回/后退/back → back。\n"
        "3. long_press 仅当指令明确含「长按」「long press」「long_press」等长按时才使用。\n"
        "4. click / long_press：target_object 为控件可见中文标签。\n"
        "5. scroll：target_object 为可滑动内容区域/列表的简短中文描述，不要写方向。\n"
        "6. input / back：target_object 输出空字符串 \"\"。\n"
        "7. target_object 只输出中文，不加位置词与通用 UI 类型词。\n"
        "8. 只输出一个合法 JSON：action_type + target_object，无 markdown。\n"
        "9. 【Multi-path / back】若当前页面与上一步 summary 中「预计进入的页面」明显不符，"
        "且当前 instruction 无法在本页完成 → action_type=back，target_object=\"\"。\n"
        "10. 【Multi-path / back — 死循环】若最近历史显示：①多步操作后【当前页面】仍与历史中的 "
        "@页面 相同（页面未前进/未变化）；②且出现重复操作（相同或相近 TO、相同 pred、"
        "parse_error/vlm_error/prepare_error、或连续多步 stuck 在同一页）——说明当前路径无效，"
        "不要继续 click/scroll/input，应 action_type=back，target_object=\"\"。\n"
        "11. 【Multi-path / back】若上一步执行结果与 instruction 意图明显背离"
        "（例如要点某入口却进入其他 Tab/页面）→ action_type=back。\n"
        "12. 判断 back 时优先看【最近动作历史】：若 Step1/2/3 的 @页面 相同且无有效 →新页面，"
        "即视为页面未更新，应 back。\n"
        "13. 仅当确信当前页可完成 instruction 且未陷入上述死循环时，"
        "才输出 click/scroll/input/long_press。\n\n"
        "示例：\n"
        '- 「点击播放」 → {"action_type":"click","target_object":"播放"}\n'
        '- 「向下滑动推荐列表」 → {"action_type":"scroll","target_object":"推荐列表"}\n'
        '- 「输入你好」 → {"action_type":"input","target_object":""}\n'
        '- 「返回上一页」 → {"action_type":"back","target_object":""}\n'
        '- 错页/死循环（同页重复操作且页面未更新） → {"action_type":"back","target_object":""}\n\n'
        f"{context_text}"
        f'【当前步骤指令】\n"{prompt_text}"\n\n'
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


def _call_llm(prompt_text: str, *, ctx: LlmToContext | None = None) -> str:
    response = llm.model.invoke(_to_prompt(prompt_text, ctx=ctx), max_tokens=MAX_TOKENS)
    return response.content if hasattr(response, "content") else str(response)


def generate_target_object(
    prompt_text: str,
    *,
    ctx: LlmToContext | None = None,
) -> dict:
    """
    生成 action_type + target_object。

    Args:
        prompt_text: 当前子任务 instruction。
        ctx: multipath 上下文（当前页、上一步 summary/动作、最近 3 步历史）。

    Returns:
        {"action_type": str, "target_object": str, "raw_response": str}
    """
    prompt_text = (prompt_text or "").strip()
    if not prompt_text:
        raise ValueError("prompt_text 为空")

    raw = _call_llm(prompt_text, ctx=ctx)
    parsed = _parse_to_result(raw)
    if parsed is None:
        raise ValueError(f"无法解析 llm_TO 输出: {raw[:300]}")

    return {
        "action_type": parsed["action_type"],
        "target_object": parsed["target_object"],
        "raw_response": raw,
    }
