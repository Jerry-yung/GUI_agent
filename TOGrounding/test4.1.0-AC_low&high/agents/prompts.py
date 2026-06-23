"""AC-low / AC-high VLM prompt 构建。"""

from __future__ import annotations

import re

AC_ACTION_TYPES = (
    "click",
    "long_press",
    "scroll",
    "input_text",
    "wait",
    "navigate_back",
    "navigate_home",
)

SCROLL_DIRECTIONS = ("up", "down", "left", "right")

_ANNOTATED_DESC = (
    "The screenshot shows interactive UI elements annotated with numbered tags "
    "(#0, #1, #2, etc.) in colored semi-transparent boxes. "
    "Each tag corresponds to a candidate node_id.\n"
)

_TO_ANNOTATED_DESC = (
    "The screenshot shows exactly ONE interactive UI element highlighted with a numbered tag "
    "(e.g. #3) in a colored semi-transparent box. "
    "This is the top-1 candidate retrieved for the current target object.\n"
)

_COORD_FALLBACK_DESC = (
    "The screenshot has NO usable node annotation. "
    "For click/long_press you must predict normalized tap coordinates on the raw screenshot.\n"
)


def _build_task_block(
    mode: str,
    *,
    instruction: str = "",
    goal: str = "",
    current_step_instruction: str = "",
) -> str:
    mode = mode.lower()
    if mode == "low":
        instruction = instruction.strip()
        goal = goal.strip()
        if not instruction:
            raise ValueError("low mode requires instruction")
        if not goal:
            raise ValueError("low mode requires goal")
        return (
            f"Task goal: {goal}\n"
            f"Step instruction (current step): {instruction}\n"
            "Use the step instruction to decide the immediate action; "
            "use the task goal for overall context."
        )
    if mode == "high":
        goal = goal.strip()
        if not goal:
            raise ValueError("high mode requires goal")
        return f"Task goal: {goal}"
    raise ValueError(f"Unknown mode: {mode!r}")


def _build_prev_step_context_block(prev_step_instruction: str) -> str:
    text = (prev_step_instruction or "").strip()
    if not text:
        return ""
    return f'Previous step instruction: "{text}"\n'


def _wait_repeat_instruction_rules() -> str:
    lines = [
        "REPEATED INSTRUCTION / LOADING:",
        "- When previous step instruction is provided below, compare it with the CURRENT "
        "step instruction.",
        "- If they are the SAME or very SIMILAR, the UI is often still loading or the "
        "previous action has not finished → prefer action_type=wait.",
        "- This applies even when SoM boxes are visible; do NOT click just because the "
        "text mentions Click/Open.",
        "- EXCEPTIONS: do NOT prefer wait when the CURRENT step instruction clearly asks for:",
        "  • scroll / swipe (e.g. scroll up, swipe down) → use scroll.",
        "  • input_text (e.g. type, enter, search <text>) → use input_text.",
        "- For those steps, repeated text means continue scrolling or typing, not waiting.",
        "- If no previous step instruction is provided (e.g. step 0), skip this heuristic.",
    ]
    return "\n".join(lines) + "\n"


def _scroll_gesture_direction_rules() -> str:
  """scroll direction = 手指滑动手势，非屏幕内容移动方向（对齐 AC / AgentCPM 评测）。"""
  return (
      "SCROLL direction (finger swipe / gesture — NOT content movement):\n"
      '- For scroll action_type, "direction" is where your FINGER moves on the screen.\n'
      "- This is the OPPOSITE of how on-screen content moves.\n"
      "- List / feed / page browsing:\n"
      '  • Step says "scroll down" / "swipe down" to see MORE content BELOW '
      '→ output direction="up" (finger swipes up).\n'
      '  • Step says "scroll up" / "swipe up" to see content ABOVE or pull content down '
      '→ output direction="down" (finger swipes down).\n'
      '  • Step says "swipe up to view reviews" → output direction="up".\n'
      "- Physical controls (time picker dial, slider, wheel): follow the literal swipe "
      "direction on that control (scroll down on dial → direction=\"down\").\n"
      "- When unsure for a list, look at the screenshot: swipe toward hidden content.\n"
  )


def _wait_and_navigate_rules(*, has_annotated_nodes: bool) -> str:
    """wait / navigate_back / navigate_home 判定规则（有标注时抑制误 click）。"""
    lines = [
        _wait_repeat_instruction_rules().rstrip(),
        "WAIT / NAVIGATION (no tap on highlighted candidate):",
        '- wait: use when the step asks to wait, pause, or let the screen/app load '
        '(e.g. "wait", "loading", "let it load"). Output {"action_type":"wait"} only.',
        '- navigate_back: use when the step asks to go back, return to the previous page, '
        'or use the system Back key — NOT click on a "Back" label in the UI.',
        '  Examples: "Go back", "Go to the previous page", "navigate back" → navigate_back.',
        '- navigate_home: use when the step asks to return to the home screen / launcher.',
    ]
    if has_annotated_nodes:
        lines.append(
            "- CRITICAL: For wait / navigate_back / navigate_home, do NOT output click or "
            "long_press just because a highlighted box is visible. The highlight is only "
            "for target-object retrieval; match the step instruction action_type first."
        )
    return "\n".join(lines) + "\n"


def _build_instruction_hints(
    instruction: str,
    goal: str,
    *,
    mode: str = "low",
) -> str:
    """根据 step instruction（low）或 goal（high）追加简短提示。"""
    instruction = (instruction or "").strip()
    goal = (goal or "").strip()
    mode = mode.lower()

    if mode == "low":
        if not instruction:
            return ""
        text = instruction
    else:
        if not goal:
            return ""
        text = goal

    low = text.lower()
    hints: list[str] = []

    if any(k in low for k in ("wait", "loading", "load ", "let it load", "pause")):
        hints.append(
            "Step hint: this step is about waiting/loading → prefer action_type=wait, not click."
        )

    if any(
        p in low
        for p in (
            "go back",
            "go to the previous",
            "previous page",
            "navigate back",
            "back to the",
        )
    ):
        hints.append(
            "Step hint: this step is about going back → use navigate_back, not click on Back UI."
        )

    is_dial = any(
        k in low for k in ("minute", "hour", "dial", "slider", "wheel", "picker")
    )
    if not is_dial and re.search(r"\b(?:scroll|swipe)\s+down\b", low):
        hints.append(
            'Step hint: "scroll/swipe down" on a list usually means reveal content below '
            '→ set direction="up" (finger swipes up), not "down".'
        )
    elif not is_dial and re.search(r"\b(?:scroll|swipe)\s+up\b", low):
        hints.append(
            'Step hint: for list scrolling, direction is the FINGER swipe. '
            '"swipe up" in the step often means direction="up".'
        )

    if not hints:
        return ""
    return "\n".join(hints) + "\n"


def build_baseline_coord_rules() -> str:
    """强约束归一化坐标规则（TO 无标注回退 / baseline 复用）。"""
    return (
        "COORDINATE RULES (click / long_press only):\n"
        "- Output x and y as FLOAT numbers in [0.0, 1.0] (NOT pixel coordinates).\n"
        "- x = horizontal ratio (0.0 = left edge, 1.0 = right edge).\n"
        "- y = vertical ratio (0.0 = top edge, 1.0 = bottom edge).\n"
        "- Both x and y keys must use double quotes.\n"
        "- Do NOT output arrays like [0.5, 0.3].\n"
        "- Do NOT output pixel coordinates.\n"
        "- Do NOT use node_id.\n"
    )


def _target_object_rules() -> str:
    """target_object 生成规则（与 annotate/llm_TO.py 一致）。"""
    return (
        "TARGET_OBJECT rules (for next-step UI retrieval):\n"
        "1. Output ONLY the on-screen text label or name of the UI element the user should "
        "interact with on the NEXT step.\n"
        "2. Do NOT add position or layout words (e.g. at the top, bottom, left, on the screen).\n"
        "3. Do NOT add generic UI-type words (e.g. icon, button, link, bar, field, tab) "
        "unless they are part of the actual visible label.\n"
        "4. Prefer the shortest faithful label: e.g. \"search\", \"Yahoo\", \"Yahoo Mail\".\n"
        "5. Do NOT output Chinese or other non-English text.\n"
        "Examples:\n"
        '- Bad: "the search bar at the top" → Good: "search"\n'
        '- Bad: "Yahoo Mail app icon" → Good: "Yahoo Mail"\n'
        '- Bad: "the close button in the corner" → Good: "Close"\n'
    )


POINTER_ACTION_TYPES = frozenset({"click", "long_press"})


def _high_planning_schema_lines() -> tuple[list[str], list[str]]:
    return (
        [
            '  "next_instruction": "one concise English sentence for the NEXT step after the current action",',
            '  "target_object": "English UI label for the element to interact with on the NEXT step",',
            '  "next_action_type": "click"|"long_press"|"scroll"|"input_text"|"wait"|"navigate_back"|"navigate_home",',
        ],
        [
            "- next_instruction: one short English imperative for the step AFTER the current action.",
            "- target_object: English on-screen label for the UI element the NEXT step should "
            "interact with (for retrieval; do not call a separate TO model).",
            "- next_action_type: action type for the NEXT step (used to prepare retrieval and "
            "fix the following step's action_type).",
            "- On the last step of a task, omit or leave planning fields empty.",
        ],
    )


def _high_planning_duties_block() -> str:
    return (
        "# High-mode planning duties\n"
        "- Execute the current step action (fixed action_type when provided).\n"
        "- Also plan the step AFTER the current action: next_instruction, target_object, "
        "next_action_type.\n"
    )


def _append_high_planning(
    schema_lines: list[str],
    extra_rules: list[str],
) -> tuple[list[str], list[str]]:
    plan_lines, plan_rules = _high_planning_schema_lines()
    return schema_lines + plan_lines, extra_rules + plan_rules


def _fixed_action_type_block(fixed_action_type: str) -> str:
    return (
        f"FIXED ACTION TYPE: The action_type for this step is already determined: "
        f"{fixed_action_type!r}. Do NOT output action_type.\n"
    )


def _schema_lines_low_fields(
    fixed_action_type: str,
    *,
    has_annotated_nodes: bool,
    agent: str,
    include_pointer_coords: bool = False,
) -> tuple[list[str], list[str]]:
    """AC-low：VLM 不输出 action_type，仅输出该 type 所需字段。"""
    action_type = fixed_action_type.strip()
    if action_type == "long_click":
        action_type = "long_press"

    schema_lines = ['  "thought": "1-3 concise sentences",']
    extra_rules: list[str] = []
    agent_upper = agent.upper()

    if action_type in POINTER_ACTION_TYPES:
        if has_annotated_nodes:
            if agent_upper in ("M2", "M2V", "M12"):
                schema_lines.append('  "node_id": 0,')
            # TO: thought only — locator injected by system
        elif include_pointer_coords or not has_annotated_nodes:
            schema_lines.extend(['  "x": 0.52,', '  "y": 0.31,'])
    elif action_type == "scroll":
        schema_lines.append('  "direction": "up"|"down"|"left"|"right",')
    elif action_type == "input_text":
        schema_lines.append('  "text": ""')

    return schema_lines, extra_rules


def _rules_block_low(
    fixed_action_type: str,
    *,
    agent: str,
    has_annotated_nodes: bool,
    extra_rules: list[str],
) -> str:
    action_type = fixed_action_type.strip()
    if action_type == "long_click":
        action_type = "long_press"
    agent_upper = agent.upper()

    if action_type in POINTER_ACTION_TYPES:
        if has_annotated_nodes:
            if agent_upper in ("M2", "M2V", "M12"):
                click_rules = (
                    "- Set node_id to an integer from annotated # labels on screen.\n"
                    "- node_id must be one of the annotated # numbers.\n"
                )
            elif agent_upper == "TO":
                click_rules = (
                    "- The system will tap the highlighted retrieval candidate automatically.\n"
                    "- Do NOT output node_id, x, or y.\n"
                )
            else:
                click_rules = build_baseline_coord_rules()
        else:
            click_rules = build_baseline_coord_rules()
        rules = click_rules
    elif action_type == "scroll":
        rules = _scroll_gesture_direction_rules()
    elif action_type == "input_text":
        rules = "- Set text to the string to type.\n"
    else:
        rules = "- No extra fields beyond thought are required.\n"

    if agent_upper == "M12" and action_type in POINTER_ACTION_TYPES and has_annotated_nodes:
        rules = f"{rules}{_m12_candidate_node_rules()}"

    rules = (
        f"{rules}"
        "- Use only fields relevant to the fixed action_type; omit unused fields.\n"
        + "\n".join(extra_rules)
        + ("\n" if extra_rules else "")
    )
    return rules


def _schema_lines(
    *,
    has_annotated_nodes: bool,
    mode: str,
    agent: str,
    include_pointer_coords: bool = False,
) -> tuple[list[str], list[str]]:
    """返回 (schema_lines, extra_rules)。"""
    schema_lines = [
        '  "thought": "1-3 concise sentences",',
        '  "action_type": "click"|"long_press"|"scroll"|"input_text"|"wait"|"navigate_back"|"navigate_home",',
    ]
    extra_rules: list[str] = []

    agent_upper = agent.upper()
    if has_annotated_nodes:
        if agent_upper in ("M2", "M2V", "M12"):
            schema_lines.append('  "node_id": 0,')
    elif include_pointer_coords or not has_annotated_nodes:
        schema_lines.extend(['  "x": 0.52,', '  "y": 0.31,'])

    schema_lines.extend(
        [
            '  "direction": "up"|"down"|"left"|"right",',
            '  "text": ""',
        ]
    )

    if mode == "high":
        schema_lines, extra_rules = _append_high_planning(schema_lines, extra_rules)

    return schema_lines, extra_rules


def _ac_rules_block(
    *,
    has_annotated_nodes: bool,
    extra_rules: list[str],
) -> str:
    if has_annotated_nodes:
        click_rules = (
            "- click / long_press: set node_id to an integer from annotated # labels on screen.\n"
            "- node_id must be one of the annotated # numbers.\n"
        )
    else:
        click_rules = build_baseline_coord_rules()

    rules = (
        f"{click_rules}"
        "- input_text: set text to the string to type (text field only).\n"
        f"{_scroll_gesture_direction_rules()}"
        f"{_wait_and_navigate_rules(has_annotated_nodes=has_annotated_nodes)}"
        "- Use fields that match the chosen action_type; omit unused fields.\n"
        + "\n".join(extra_rules)
        + ("\n" if extra_rules else "")
    )
    return rules


def _to_rules_block(
    *,
    has_annotated_nodes: bool,
    extra_rules: list[str],
) -> str:
    if has_annotated_nodes:
        click_rules = (
            "- click / long_press: output action_type only if the highlighted candidate matches "
            "the current step; the system taps that box automatically.\n"
            "- Do NOT output node_id, x, or y for click/long_press.\n"
            "- If the step needs scroll, input_text, wait, or navigation, "
            "output that action_type instead — do not click just because a box is shown.\n"
        )
    else:
        click_rules = build_baseline_coord_rules()

    rules = (
        f"{click_rules}"
        "- input_text: set text to the string to type (text field only).\n"
        f"{_scroll_gesture_direction_rules()}"
        f"{_wait_and_navigate_rules(has_annotated_nodes=has_annotated_nodes)}"
        "- Use fields that match the chosen action_type; omit unused fields.\n"
        + "\n".join(extra_rules)
        + ("\n" if extra_rules else "")
    )
    return rules


def _m12_candidate_node_rules() -> str:
    return (
        "CANDIDATE NODES (m12):\n"
        "- For click/long_press, set node_id to an integer from the Candidate Interactive "
        "Nodes table in the user message.\n"
        "- node_id must match a # listed in that table; do not invent indices.\n"
        "- Trust the table (Label / Semantic) over visual guess when boxes look similar.\n"
        "- Prefer rows with clear labels over ⚠ (no label) rows.\n"
        "- Match the step instruction to Label/Semantic; use score only as a tie-breaker.\n"
        "- For scroll, input_text, wait, or navigation, follow action_type rules; "
        "do not click just because a candidate row exists.\n"
    )


def _m12_ac_rules_block(
    *,
    has_annotated_nodes: bool,
    extra_rules: list[str],
) -> str:
    if has_annotated_nodes:
        click_rules = (
            "- click / long_press: set node_id from the Candidate Interactive Nodes table.\n"
            "- node_id must be one of the annotated # numbers listed in the table.\n"
        )
    else:
        click_rules = build_baseline_coord_rules()

    rules = (
        f"{click_rules}"
        f"{_m12_candidate_node_rules()}"
        "- input_text: set text to the string to type (text field only).\n"
        f"{_scroll_gesture_direction_rules()}"
        f"{_wait_and_navigate_rules(has_annotated_nodes=has_annotated_nodes)}"
        "- Use fields that match the chosen action_type; omit unused fields.\n"
        + "\n".join(extra_rules)
        + ("\n" if extra_rules else "")
    )
    return rules


def build_m12_prompt_parts(
    mode: str,
    *,
    instruction: str = "",
    goal: str = "",
    current_step_instruction: str = "",
    prev_step_instruction: str = "",
    candidate_nodes_table: str = "",
    has_annotated_nodes: bool = True,
    fixed_action_type: str | None = None,
) -> tuple[str, str]:
    """返回 (system_prompt, user_prompt)。m2 + top-k 候选节点语义表。"""
    mode = mode.lower()
    use_low_fixed = mode == "low" and fixed_action_type
    use_high_fixed = mode == "high" and fixed_action_type
    if use_low_fixed or use_high_fixed:
        schema_lines, extra_rules = _schema_lines_low_fields(
            fixed_action_type,
            has_annotated_nodes=has_annotated_nodes,
            agent="m12",
        )
        if use_high_fixed:
            schema_lines, extra_rules = _append_high_planning(schema_lines, extra_rules)
        rules_block = _rules_block_low(
            fixed_action_type,
            agent="m12",
            has_annotated_nodes=has_annotated_nodes,
            extra_rules=extra_rules,
        )
        fixed_block = _fixed_action_type_block(fixed_action_type)
        high_duties = _high_planning_duties_block() if use_high_fixed else ""
    else:
        schema_lines, extra_rules = _schema_lines(
            has_annotated_nodes=has_annotated_nodes,
            mode=mode,
            agent="m12",
        )
        rules_block = _m12_ac_rules_block(
            has_annotated_nodes=has_annotated_nodes,
            extra_rules=extra_rules,
        )
        fixed_block = ""
        high_duties = _high_planning_duties_block() if mode == "high" else ""
    schema = "{\n" + ",\n".join(schema_lines) + "\n}"
    screen_desc = _ANNOTATED_DESC if has_annotated_nodes else _COORD_FALLBACK_DESC

    system_prompt = (
        "# Role\n"
        "You are an Android GUI automation agent (M12). "
        "Given a mobile screenshot, candidate node descriptions, and a task description, "
        "predict the NEXT action on the current screen.\n\n"
        "# Task\n"
        "Output exactly ONE JSON object for the immediate next action.\n\n"
        f"{high_duties}"
        f"{fixed_block}"
        f"# Screen\n{screen_desc}\n"
        "# Rule\n"
        "- Output compact raw JSON only. No markdown or code fences.\n"
        "- Follow the schema and rules below.\n\n"
        f"# Schema\n{schema}\n\n"
        f"# Rules\n{rules_block}"
    )

    task_block = _build_task_block(
        mode,
        instruction=instruction,
        goal=goal,
        current_step_instruction=current_step_instruction if mode != "high" else "",
    )
    prev_block = (
        "" if mode == "high" else _build_prev_step_context_block(prev_step_instruction)
    )
    hints = _build_instruction_hints(instruction, goal, mode=mode)
    candidate_block = (candidate_nodes_table or "").strip()
    if candidate_block:
        candidate_block = candidate_block + "\n\n"
    user_prompt = (
        f"{prev_block}{candidate_block}{hints}{task_block}\n\nCurrent screen screenshot:"
    )
    return system_prompt, user_prompt


def build_m2_prompt_parts(
    mode: str,
    *,
    instruction: str = "",
    goal: str = "",
    current_step_instruction: str = "",
    prev_step_instruction: str = "",
    has_annotated_nodes: bool = True,
    fixed_action_type: str | None = None,
) -> tuple[str, str]:
    """返回 (system_prompt, user_prompt)。"""
    mode = mode.lower()
    use_low_fixed = mode == "low" and fixed_action_type
    use_high_fixed = mode == "high" and fixed_action_type
    if use_low_fixed or use_high_fixed:
        schema_lines, extra_rules = _schema_lines_low_fields(
            fixed_action_type,
            has_annotated_nodes=has_annotated_nodes,
            agent="m2",
        )
        if use_high_fixed:
            schema_lines, extra_rules = _append_high_planning(schema_lines, extra_rules)
        rules_block = _rules_block_low(
            fixed_action_type,
            agent="m2",
            has_annotated_nodes=has_annotated_nodes,
            extra_rules=extra_rules,
        )
        fixed_block = _fixed_action_type_block(fixed_action_type)
        high_duties = _high_planning_duties_block() if use_high_fixed else ""
    else:
        schema_lines, extra_rules = _schema_lines(
            has_annotated_nodes=has_annotated_nodes,
            mode=mode,
            agent="m2",
        )
        rules_block = _ac_rules_block(
            has_annotated_nodes=has_annotated_nodes,
            extra_rules=extra_rules,
        )
        fixed_block = ""
        high_duties = _high_planning_duties_block() if mode == "high" else ""
    schema = "{\n" + ",\n".join(schema_lines) + "\n}"
    screen_desc = _ANNOTATED_DESC if has_annotated_nodes else _COORD_FALLBACK_DESC

    system_prompt = (
        "# Role\n"
        "You are an Android GUI automation agent. "
        "Given a mobile screenshot and a task description, "
        "predict the NEXT action on the current screen.\n\n"
        "# Task\n"
        "Output exactly ONE JSON object for the immediate next action.\n\n"
        f"{high_duties}"
        f"{fixed_block}"
        f"# Screen\n{screen_desc}\n"
        "# Rule\n"
        "- Output compact raw JSON only. No markdown or code fences.\n"
        "- Follow the schema and rules below.\n\n"
        f"# Schema\n{schema}\n\n"
        f"# Rules\n{rules_block}"
    )

    task_block = _build_task_block(
        mode,
        instruction=instruction,
        goal=goal,
        current_step_instruction=current_step_instruction if mode != "high" else "",
    )
    prev_block = (
        "" if mode == "high" else _build_prev_step_context_block(prev_step_instruction)
    )
    hints = _build_instruction_hints(instruction, goal, mode=mode)
    user_prompt = f"{prev_block}{hints}{task_block}\n\nCurrent screen screenshot:"
    return system_prompt, user_prompt


def build_m2v_prompt_parts(
    mode: str,
    *,
    instruction: str = "",
    goal: str = "",
    current_step_instruction: str = "",
    prev_step_instruction: str = "",
    has_annotated_nodes: bool = True,
    fixed_action_type: str | None = None,
) -> tuple[str, str]:
    """返回 (system_prompt, user_prompt)。high 模式额外要求下一步规划三字段。"""
    mode = mode.lower()
    use_low_fixed = mode == "low" and fixed_action_type
    use_high_fixed = mode == "high" and fixed_action_type
    if use_low_fixed or use_high_fixed:
        schema_lines, extra_rules = _schema_lines_low_fields(
            fixed_action_type,
            has_annotated_nodes=has_annotated_nodes,
            agent="m2v",
        )
        if use_high_fixed:
            schema_lines, extra_rules = _append_high_planning(schema_lines, extra_rules)
        rules_block = _rules_block_low(
            fixed_action_type,
            agent="m2v",
            has_annotated_nodes=has_annotated_nodes,
            extra_rules=extra_rules,
        )
        fixed_block = _fixed_action_type_block(fixed_action_type)
        m2v_duties = _high_planning_duties_block() if use_high_fixed else ""
    else:
        schema_lines, extra_rules = _schema_lines(
            has_annotated_nodes=has_annotated_nodes,
            mode=mode,
            agent="m2v",
        )
        rules_block = _ac_rules_block(
            has_annotated_nodes=has_annotated_nodes,
            extra_rules=extra_rules,
        )
        fixed_block = ""
        m2v_duties = _high_planning_duties_block() if mode == "high" else ""
    schema = "{\n" + ",\n".join(schema_lines) + "\n}"
    screen_desc = _ANNOTATED_DESC if has_annotated_nodes else _COORD_FALLBACK_DESC

    system_prompt = (
        "# Role\n"
        "You are an Android GUI automation agent with integrated target-object planning (M2V). "
        "Given a mobile screenshot and a task description, "
        "predict the NEXT action on the current screen.\n\n"
        "# Task\n"
        "Output exactly ONE JSON object for the immediate next action.\n\n"
        f"{m2v_duties}"
        f"{fixed_block}"
        f"# Screen\n{screen_desc}\n"
        "# Rule\n"
        "- Output compact raw JSON only. No markdown or code fences.\n"
        "- Follow the schema and rules below.\n\n"
        f"# Schema\n{schema}\n\n"
        f"# Rules\n{rules_block}"
    )

    task_block = _build_task_block(
        mode,
        instruction=instruction,
        goal=goal,
        current_step_instruction=current_step_instruction if mode != "high" else "",
    )
    prev_block = (
        "" if mode == "high" else _build_prev_step_context_block(prev_step_instruction)
    )
    hints = _build_instruction_hints(instruction, goal, mode=mode)
    user_prompt = f"{prev_block}{hints}{task_block}\n\nCurrent screen screenshot:"
    return system_prompt, user_prompt


def build_ac_vlm_prompt_parts(
    mode: str,
    *,
    instruction: str = "",
    goal: str = "",
    current_step_instruction: str = "",
    prev_step_instruction: str = "",
    has_annotated_nodes: bool = True,
) -> tuple[str, str]:
    """兼容旧接口。"""
    return build_m2_prompt_parts(
        mode,
        instruction=instruction,
        goal=goal,
        current_step_instruction=current_step_instruction,
        prev_step_instruction=prev_step_instruction,
        has_annotated_nodes=has_annotated_nodes,
    )


def build_ac_vlm_prompt(
    mode: str,
    *,
    instruction: str = "",
    goal: str = "",
    current_step_instruction: str = "",
    prev_step_instruction: str = "",
    has_annotated_nodes: bool = True,
) -> str:
    """兼容旧接口：system + user 合并为单条文本。"""
    system_prompt, user_prompt = build_m2_prompt_parts(
        mode,
        instruction=instruction,
        goal=goal,
        current_step_instruction=current_step_instruction,
        prev_step_instruction=prev_step_instruction,
        has_annotated_nodes=has_annotated_nodes,
    )
    return f"{system_prompt}\n\n{user_prompt}"


def build_to_vlm_prompt_parts(
    mode: str,
    *,
    instruction: str = "",
    goal: str = "",
    target_object: str = "",
    current_step_instruction: str = "",
    prev_step_instruction: str = "",
    has_annotated_nodes: bool = True,
    fixed_action_type: str | None = None,
) -> tuple[str, str]:
    """返回 (system_prompt, user_prompt)。"""
    mode = mode.lower()
    use_low_fixed = mode == "low" and fixed_action_type
    use_high_fixed = mode == "high" and fixed_action_type
    if use_low_fixed or use_high_fixed:
        schema_lines, extra_rules = _schema_lines_low_fields(
            fixed_action_type,
            has_annotated_nodes=has_annotated_nodes,
            agent="TO",
            include_pointer_coords=not has_annotated_nodes,
        )
        if use_high_fixed:
            schema_lines, extra_rules = _append_high_planning(schema_lines, extra_rules)
        rules_block = _rules_block_low(
            fixed_action_type,
            agent="TO",
            has_annotated_nodes=has_annotated_nodes,
            extra_rules=extra_rules,
        )
        fixed_block = _fixed_action_type_block(fixed_action_type)
        to_duties = (
            "# TO agent duties\n"
            f"- The action_type is fixed to {fixed_action_type!r}.\n"
            "- Fill in only the fields required for that action_type.\n"
            "- For click/long_press with a highlighted candidate, the system taps it automatically.\n"
        )
        if use_high_fixed:
            to_duties = f"{to_duties}\n{_high_planning_duties_block()}"
    else:
        schema_lines, extra_rules = _schema_lines(
            has_annotated_nodes=has_annotated_nodes,
            mode=mode,
            agent="TO",
            include_pointer_coords=not has_annotated_nodes,
        )
        rules_block = _to_rules_block(
            has_annotated_nodes=has_annotated_nodes,
            extra_rules=extra_rules,
        )
        fixed_block = ""
        to_duties = (
            "# TO agent duties\n"
            "- Decide the NEXT action_type for the current step (follow step instruction in low mode).\n"
            "- For click/long_press ONLY when the step requires tapping the retrieved target: "
            "output action_type only; the system uses the highlighted candidate.\n"
            "- For wait, navigate_back, navigate_home, scroll, input_text: "
            "output that action_type — never default to click because a highlight exists.\n"
        )
        if mode == "high":
            to_duties = f"{to_duties}\n{_high_planning_duties_block()}"
    schema = "{\n" + ",\n".join(schema_lines) + "\n}"
    screen_desc = _TO_ANNOTATED_DESC if has_annotated_nodes else _COORD_FALLBACK_DESC

    system_prompt = (
        "# Role\n"
        "You are an Android GUI automation agent using target-object retrieval (TO). "
        "A retrieval pipeline has proposed ONE on-screen candidate for the current target.\n\n"
        "# Task\n"
        "Output exactly ONE JSON object for the immediate next action.\n\n"
        f"{to_duties}\n"
        f"{fixed_block}"
        f"# Screen\n{screen_desc}\n"
        "# Rule\n"
        "- Output compact raw JSON only. No markdown or code fences.\n"
        "- Follow the schema and rules below.\n\n"
        f"# Schema\n{schema}\n\n"
        f"# Rules\n{rules_block}"
    )

    task_block = _build_task_block(
        mode,
        instruction=instruction,
        goal=goal,
        current_step_instruction=current_step_instruction if mode != "high" else "",
    )
    prev_block = (
        "" if mode == "high" else _build_prev_step_context_block(prev_step_instruction)
    )
    target_line = ""
    if target_object.strip():
        target_line = f'Retrieved target: "{target_object.strip()}"\n'
    hints = _build_instruction_hints(instruction, goal, mode=mode)
    user_prompt = (
        f"{prev_block}{target_line}{hints}{task_block}\n\nCurrent screen screenshot:"
    )
    return system_prompt, user_prompt


def build_to_vlm_prompt(
    mode: str,
    *,
    instruction: str = "",
    goal: str = "",
    target_object: str = "",
    current_step_instruction: str = "",
    prev_step_instruction: str = "",
    has_annotated_nodes: bool = True,
) -> str:
    system_prompt, user_prompt = build_to_vlm_prompt_parts(
        mode,
        instruction=instruction,
        goal=goal,
        target_object=target_object,
        current_step_instruction=current_step_instruction,
        prev_step_instruction=prev_step_instruction,
        has_annotated_nodes=has_annotated_nodes,
    )
    return f"{system_prompt}\n\n{user_prompt}"
