"""CPM Agent prompt：中文任务描述 + AC 归一化坐标 action schema。"""

from __future__ import annotations

import json
from pathlib import Path

from agents.prompts import (
    _build_instruction_hints,
    _wait_and_navigate_rules,
    build_baseline_coord_rules,
)

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "cpm_action.schema.json"
_CPM_SCHEMA: dict | None = None


def _load_cpm_schema() -> dict:
    global _CPM_SCHEMA
    if _CPM_SCHEMA is None:
        with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        items = list(schema.items())
        items.insert(3, ("required", ["thought", "action_type"]))
        _CPM_SCHEMA = dict(items)
    return _CPM_SCHEMA


def _cpm_rules_block() -> str:
    return (
        f"{build_baseline_coord_rules()}"
        "- input_text: set text to the string to type (text field only).\n"
        f"{_wait_and_navigate_rules(has_annotated_nodes=False)}"
        "- Use fields that match the chosen action_type; omit unused fields.\n"
        "- Do NOT use POINT arrays or 0–1000 integer coordinates.\n"
    )


def build_cpm_prompt_parts(
    mode: str,
    *,
    instruction: str = "",
    goal: str = "",
) -> tuple[str, str]:
    """
    返回 (system_prompt, user_prompt)。

    low: <Question>{instruction}</Question>
    high: <Question>{goal}</Question>
    """
    mode = mode.lower()
    schema = _load_cpm_schema()
    system_prompt = (
        "# Role\n"
        "你是一名熟悉安卓系统触屏 GUI 操作的智能体，将根据用户的问题，"
        "分析当前界面的 GUI 元素和布局，生成相应的操作。\n\n"
        "# Task\n"
        "针对用户问题，根据输入的当前屏幕截图，输出下一步的操作。\n\n"
        "# Rule\n"
        "- 以紧凑 JSON 格式输出，不要 markdown 或代码块\n"
        "- 输出操作必须遵循 Schema 与 Rules\n\n"
        f"# Schema\n{json.dumps(schema, indent=None, ensure_ascii=False, separators=(',', ':'))}\n\n"
        f"# Rules\n{_cpm_rules_block()}"
    )

    if mode == "low":
        query = (instruction or "").strip()
        if not query:
            raise ValueError("low mode requires instruction")
    elif mode == "high":
        query = (goal or "").strip()
        if not query:
            raise ValueError("high mode requires goal")
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    hints = _build_instruction_hints(instruction, goal, mode=mode)
    user_prompt = f"{hints}<Question>{query}</Question>\n当前屏幕截图："
    return system_prompt, user_prompt
