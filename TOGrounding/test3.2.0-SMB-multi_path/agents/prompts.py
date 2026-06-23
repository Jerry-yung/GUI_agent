"""VLM prompts for m2 (English) / TO (Chinese) agents (SMAN action space, multi-path)."""
from __future__ import annotations

import re

VLM_ACTION_TYPES = frozenset({"click", "scroll", "input", "back", "long_press"})

_TO_SUMMARY_FIELD_SPEC = (
    "  summary：用一两句话总结：①本步执行的操作；②执行后预计进入的页面"
    "（写出页面标识如 QQmusic0_10，或清晰的界面描述）。不要在 summary 中出现区域编号标签。\n"
)

_M2_SUMMARY_FIELD_SPEC = (
    '"summary": Briefly describe ① the action taken in this step and ② the page you '
    "expect to reach next (use page identifiers like QQmusic0_10 or a clear interface "
    "description). Do NOT include element tag numbers in the summary.\n"
)

_M2_OUTPUT_INTRO = (
    "  根据截图与任务信息思考并输出本步动作。每次只能执行一个动作。\n"
    "  仅输出一个 JSON 对象，包含 thought、summary、action 三个字段：\n"
    "  thought：描述你在截图中的观察，以及为完成当前子任务下一步应做什么。\n"
    f"{_TO_SUMMARY_FIELD_SPEC}"
)

_JSON_SCHEMA = (
    '  {"thought":"...","summary":"本步操作；预计进入页面：...","action":{"type":"click|scroll|input|back",'
    '"element":"N","x":0.0,"y":0.0,"direction":"up|down|left|right","text":"..."}}\n'
)

_M2_JSON_SCHEMA = (
    '  {"thought":"...","summary":"本步操作；预计进入页面：...","action":{"type":"click|scroll|input|back",'
    '"element":"N","direction":"up|down|left|right","text":"..."}}\n'
)

_SCROLL_JSON_SCHEMA = (
    '  {"thought":"...","summary":"本步操作；预计进入页面：...","action":{"type":"scroll","element":"N",'
    '"direction":"up|down|left|right"}}\n'
)

_INPUT_JSON_SCHEMA = (
    '  {"thought":"...","summary":"本步操作；预计进入页面：...","action":{"type":"input","text":"..."}}\n'
)

_BACK_JSON_SCHEMA = (
    '  {"thought":"...","summary":"本步操作；预计进入页面：...","action":{"type":"back"}}\n'
)

_CLICK_ONLY_SCHEMA = '  {"thought":"...","summary":"...","action":{"type":"click"}}\n'


def _m2_action_catalog() -> str:
    return (
        "你可以调用以下函数控制手机（action 字段中 type 与之对应）：\n\n"
        "1. click(element)\n"
        "   点击截图上带数字标签的可点击 UI 元素。示例：click(5)。\n\n"
        "2. input(text)\n"
        "   在输入框中输入文本，text 为要输入的字符串。示例：input(\"你好\")。\n\n"
        "3. scroll(element, direction)\n"
        "   在带数字标签的可滚动区域内滑动。direction 为 up / down / left / right 之一。\n"
        '   示例：scroll(2, "down")。\n\n'
        "4. back()\n"
        "   返回上一页。示例：back()。\n"
    )


def _m2_en_action_catalog() -> str:
    return (
        "Available actions (action.type when not fixed):\n\n"
        "1. click(element) — tap a clickable UI element tagged #N on the screenshot.\n"
        "2. input(text) — type text into an input field.\n"
        "3. scroll(element, direction) — swipe inside scrollable region #N; "
        "direction is up / down / left / right.\n"
        "4. back() — return to the previous page (system back).\n"
    )


def _m2_task_context_block(
    *,
    task_desc: str,
    step_instruction: str,
    current_page_name: str,
    last_summary: str,
) -> str:
    lines = [
        "User:",
        "  你是一个在智能手机上执行多步 GUI 任务的操作智能体。",
        "  截图中的检索候选以纯数字 1、2、3… 标注（本步仅 click 或 scroll 一种类型），标签位于元素中心。",
        "",
        _m2_action_catalog(),
        f"  整体任务：{task_desc}",
        f"  为完成整体任务，当前需要执行的子任务：{step_instruction}",
        f"  当前所在页面标识：{current_page_name}",
    ]
    if last_summary.strip():
        lines.extend(
            [
                "",
                "  上一步操作摘要及对下一步页面的预期（由上一步模型输出）：",
                f"  {last_summary.strip()}",
                "",
                "  【页面一致性】若当前页面与上一步 summary 中「预计进入的下一步页面」明显不符，",
                "  或当前页面无法继续完成当前子任务，应输出 back() 返回上一页后重新规划。",
            ]
        )
    return "\n".join(lines) + "\n"


def _m2_en_task_context_block(
    *,
    task_desc: str,
    step_instruction: str,
    current_page_name: str,
    last_summary: str,
) -> str:
    lines = [
        "# Role",
        "You are an Android GUI automation agent executing multi-step tasks on a smartphone.",
        "Candidate elements are annotated with tags (#1, #2, #3, …); only one action type "
        "(click or scroll) is shown per step.",
        "",
        _m2_en_action_catalog(),
        f"Task goal: {task_desc}",
        f"Current sub-task: {step_instruction}",
        f"Current page: {current_page_name}",
    ]
    if last_summary.strip():
        lines.extend(
            [
                "",
                f'Previous step summary (from the prior model output): "{last_summary.strip()}"',
                "",
                "Page consistency: if the current page clearly does not match the page "
                "expected in the previous summary, or the current page cannot advance "
                "the sub-task, output back() to return and replan.",
            ]
        )
    return "\n".join(lines) + "\n\n"


def _m2_en_user_context_block(
    *,
    task_desc: str,
    step_instruction: str,
    current_page_name: str,
    last_summary: str,
) -> str:
    lines = [
        f"Task goal: {task_desc}",
        f"Current sub-task: {step_instruction}",
        f"Current page: {current_page_name}",
    ]
    if last_summary.strip():
        lines.extend(
            [
                "",
                f'Previous step summary (from the prior model output): "{last_summary.strip()}"',
                "",
                "Page consistency: if the current page clearly does not match the page "
                "expected in the previous summary, or the current page cannot advance "
                "the sub-task, output back() to return and replan.",
            ]
        )
    return "\n".join(lines) + "\n\n"


def _m2_to_user_context_block(
    *,
    task_desc: str,
    step_instruction: str,
    current_page_name: str,
    last_summary: str,
) -> str:
    lines = [
        f"整体任务：{task_desc}",
        f"当前子任务：{step_instruction}",
        f"当前所在页面标识：{current_page_name}",
    ]
    if last_summary.strip():
        lines.extend(
            [
                "",
                "上一步操作摘要及对下一步页面的预期（由上一步模型输出）：",
                f"{last_summary.strip()}",
                "",
                "【页面一致性】若当前页面与上一步 summary 中「预计进入的下一步页面」明显不符，",
                "或当前页面无法继续完成当前子任务，应输出 back() 返回上一页后重新规划。",
            ]
        )
    return "\n".join(lines) + "\n"


def _back_action_spec() -> str:
    return (
        "  【back 动作】back()：返回上一页（系统级返回，无 element 参数）。\n"
        '    示例：{"type":"back"} 表示从当前子页面退回上一级页面。\n'
    )


def _m2_en_back_action_spec() -> str:
    return (
        "back(): Returns to the previous page (system-level back, no element argument).\n"
    )


def _append_back_sections(step_instruction: str, *, include_mismatch_hint: bool) -> str:
    parts = [f"\n{_back_action_spec()}"]
    if include_mismatch_hint:
        parts.append(
            "  【Multi-path 决策】若当前页面无法完成本步指令、截图无可操作标注，"
            "或页面状态与上一步 summary 中的页面预期不符，应输出 back。\n"
        )
    else:
        parts.append(
            "  【Multi-path 决策】若当前页面无法完成本步指令、截图无可操作标注，"
            "或页面状态与预期不符，可考虑输出 back 返回上一页再重新规划。\n"
        )
    parts.append(f"  当前子任务：{step_instruction}\n")
    return "".join(parts)


def _m2_en_append_back_sections(step_instruction: str, *, include_mismatch_hint: bool) -> str:
    parts = ["\n# Multi-path / back option\n", _m2_en_back_action_spec()]
    if include_mismatch_hint:
        parts.append(
            "If the current page cannot complete the sub-task, has no operable annotations, "
            "or the page state does not match the page expected in the previous summary, "
            "output back.\n"
        )
    else:
        parts.append(
            "If the current page cannot complete the sub-task, has no operable annotations, "
            "or the page state does not match expectations, consider outputting back.\n"
        )
    parts.append(f"Current sub-task: {step_instruction}\n")
    return "".join(parts)


def _scroll_direction_rules() -> str:
    return (
        "  【方向约定】action.direction 表示手指在 element 所指区域内的滑动方向：\n"
        "    up    = 手指从下向上拖\n"
        "    down  = 手指从上向下拖\n"
        "    left  = 手指从右向左拖\n"
        "    right = 手指从左向右拖\n"
        "  请根据步骤指令语义选择区域编号与方向；不要按「页面内容往哪移动」理解，"
        "而应转换为手指在区域内的拖拽方向。\n"
        '  示例：scroll(2, "down") 表示在 2 号区域内手指从上向下拖。\n'
    )


def _scroll_official_annotation_legend() -> str:
    return (
        "  【截图标注】本步为滑动操作，截图采用官方风格标注：\n"
        "    · 所有可滑动区域均已标为 s1、s2、…（并非只标一个区域）\n"
        "    · 每个标签位于对应可滑动区域的中心，黑底白字\n"
        "    · 同一区域只有一个 sN，不区分方向；方向由你根据指令填写 direction\n"
    )


def _topk_annotation_legend(top_k: int) -> str:
    return (
        f"  【截图标注】本步展示语义检索 Top-{top_k} 候选子集（非全页所有控件）：\n"
        "    · 可点击：红色实线框 + 浅红填充，标签为纯数字 1、2、3…\n"
        "    · 可滑动：蓝色虚线框（无填充），标签为纯数字 1、2、3…\n"
        "  本步仅一种候选类型（click 或 scroll），标签为纯数字；若应 input，按 input 规则输出（可能为无标注原图）。\n"
        "  勿因图中 click 框更多就默认 click；请结合步骤指令判断动作类型。\n"
    )


def _scroll_action_spec() -> str:
    return (
        "  【scroll 动作】scroll(element, direction)：\n"
        "    element = 图中数字标签（如 1、2）；direction = up | down | left | right 之一。\n"
    )


def _scroll_to_line(target_object: str) -> str:
    if target_object.strip():
        return f'  检索目标 TO="{target_object.strip()}"\n'
    return ""


def _scroll_topk_annotation_legend(top_k: int) -> str:
    return (
        f"  【截图标注】本步为滑动操作，截图展示语义检索 Top-{top_k} 个可滑动区域候选（非全页所有区域）：\n"
        "    · 可滑动区域：蓝色虚线框（无填充），标签为纯数字 1、2、3…\n"
        "  必须从图中可见的编号中选择 element，并给出 direction；不要假设未标注区域。\n"
    )


def _scroll_top1_annotation_legend() -> str:
    return (
        "  【截图标注】本步为滑动检索，截图已用 Top-1 高亮单个可滑动区域候选。\n"
        "    · 蓝色虚线框（无填充），标签为纯数字（可能在框内或框旁）\n"
    )


def _fixed_type_block(fixed_action_type: str) -> str:
    return (
        f"  【固定动作】本步 action_type 已由 llm_TO 确定为 {fixed_action_type}；"
        "不要输出 action.type。\n"
    )


def _m2_fixed_type_block(fixed_action_type: str) -> str:
    return (
        f"FIXED ACTION TYPE: The action_type for this step is already determined: "
        f"{fixed_action_type!r}. Do NOT output action_type.\n"
    )


def _m2_retrieved_target_line(target_object: str) -> str:
    if target_object.strip():
        return f'Retrieved target: "{target_object.strip()}"\n'
    return ""


def _m2_topk_annotation_legend(top_k: int) -> str:
    return (
        f"# Screen\n"
        f"Top-{top_k} semantically retrieved candidate UI elements are annotated with "
        "numbered tags (#1, #2, #3, …) in colored semi-transparent boxes "
        "(red solid border + light-red fill for clickable elements). "
        "Tags are at element centers.\n"
        "Only this retrieved subset is shown; do not assume unlabeled elements.\n"
    )


def _m2_scroll_topk_annotation_legend(top_k: int) -> str:
    return (
        f"# Screen\n"
        f"Top-{top_k} scrollable regions are annotated with tags (#1, #2, #3, …) "
        "in blue dashed boxes (no fill).\n"
        "Only this retrieved subset is shown; do not assume unlabeled regions.\n"
    )


def _m2_scroll_gesture_direction_rules() -> str:
    return (
        "# Scroll Direction Rules\n"
        'SCROLL direction (finger swipe / gesture — NOT content movement):\n'
        '- For scroll, "direction" is where your FINGER moves on the screen.\n'
        "- This is the OPPOSITE of how on-screen content moves.\n"
        "- List / feed / page browsing:\n"
        '  • Step says "scroll down" / "swipe down" to see MORE content BELOW '
        '→ output direction="up" (finger swipes up).\n'
        '  • Step says "scroll up" / "swipe up" to see content ABOVE or pull content down '
        '→ output direction="down" (finger swipes down).\n'
        '  • Step says "swipe up to view reviews" → output direction="up".\n'
        "- Physical controls (time picker dial, slider, wheel): follow the literal swipe "
        'direction on that control (scroll down on dial → direction="down").\n'
        "- When unsure for a list, look at the screenshot: swipe toward hidden content.\n"
        "- Select the scrollable region tag (#N) from the screenshot.\n"
    )


def _instruction_hints_3x(step_instruction: str) -> str:
    text = (step_instruction or "").strip()
    if not text:
        return ""
    low = text.lower()
    hints: list[str] = []
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


_M2_FIXED_OUTPUT_INTRO = (
    "# Output\n"
    "Think based on the screenshot and task context. Execute exactly ONE action.\n"
    "Output exactly one JSON object with thought, summary, and action (do NOT output action.type):\n"
    "thought: describe what you observe and what to do next for the current sub-task.\n"
    f"{_M2_SUMMARY_FIELD_SPEC}"
)

_TO_FIXED_OUTPUT_INTRO = (
    "  根据截图与任务信息思考并输出本步动作。每次只能执行一个动作。\n"
    "  仅输出一个 JSON 对象，包含 thought、summary、action 三个字段（不要 action.type）：\n"
    "  thought：描述你在截图中的观察，以及为完成当前子任务下一步应做什么。\n"
    f"{_TO_SUMMARY_FIELD_SPEC}"
)

_M2_CLICK_ELEMENT_SCHEMA = (
    '  {"thought": "...", "summary": "action done; expected next page: ...", '
    '"action": {"element": "#N"}}\n'
)

_M2_SCROLL_ELEMENT_SCHEMA = (
    '  {"thought": "...", "summary": "action done; expected next page: ...", '
    '"action": {"element": "#N", "direction": "up|down|left|right"}}\n'
)

_M2_INPUT_TEXT_SCHEMA = (
    '  {"thought": "...", "summary": "action done; expected next page: ...", '
    '"action": {"text": "..."}}\n'
)

_M2_BACK_THOUGHT_SCHEMA = (
    '  {"thought": "...", "summary": "back to previous page; expected next page: ..."}\n'
)

_CLICK_ELEMENT_SCHEMA = (
    '  {"thought":"...","summary":"本步操作；预计进入页面：...","action":{"element":"N"}}\n'
)

_TO_THOUGHT_ONLY_SCHEMA = (
    '  {"thought":"...","summary":"本步操作；预计进入页面：..."}\n'
)

_SCROLL_ELEMENT_SCHEMA = (
    '  {"thought":"...","summary":"本步操作；预计进入页面：...",'
    '"action":{"element":"N","direction":"up|down|left|right"}}\n'
)

_TO_SCROLL_DIRECTION_SCHEMA = (
    '  {"thought":"...","summary":"本步操作；预计进入页面：...",'
    '"action":{"direction":"up|down|left|right"}}\n'
)

_INPUT_TEXT_SCHEMA = (
    '  {"thought":"...","summary":"本步操作；预计进入页面：...","action":{"text":"..."}}\n'
)

_BACK_THOUGHT_SCHEMA = (
    '  {"thought":"...","summary":"本步操作；预计进入页面：..."}\n'
)

_M2_ROLE_TASK = (
    "# Role\n"
    "You are an Android GUI automation agent executing multi-step tasks on a smartphone.\n"
    "Candidate elements are annotated with tags (#1, #2, #3, …); only one action type "
    "(click or scroll) is shown per step.\n\n"
    f"{_m2_en_action_catalog()}\n"
    "# Task\n"
    "Output exactly ONE JSON object for the immediate next action.\n\n"
)

_M2_COMMON_RULE_HEADER = (
    "# Rule\n"
    "- Output compact raw JSON only. No markdown, no code fences.\n"
    "- Only complete the current sub-task; do not judge whether the entire task is finished.\n"
    "- Do NOT output Chinese.\n\n"
)

_SCREENSHOT_SUFFIX = "Current screen screenshot:"


def _m2_user_dynamic_block(
    step_instruction: str,
    *,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
    target_object: str = "",
    include_back: bool = False,
) -> str:
    has_history = bool(last_summary.strip())
    parts = [
        _m2_en_user_context_block(
            task_desc=task_desc,
            step_instruction=step_instruction,
            current_page_name=current_page_name,
            last_summary=last_summary,
        )
    ]
    hints = _instruction_hints_3x(step_instruction)
    if hints:
        parts.append(hints)
    target = _m2_retrieved_target_line(target_object)
    if target:
        parts.append(target)
    if include_back:
        parts.append(
            _m2_en_append_back_sections(
                step_instruction, include_mismatch_hint=has_history
            )
        )
    parts.append(f"{_SCREENSHOT_SUFFIX}")
    return "".join(parts)


def _m2_system_click(*, fixed_action_type: str, top_k: int) -> str:
    return (
        f"{_M2_ROLE_TASK}"
        f"{_m2_fixed_type_block(fixed_action_type)}\n"
        f"{_m2_topk_annotation_legend(top_k)}"
        f"{_M2_COMMON_RULE_HEADER}"
        f"{_M2_FIXED_OUTPUT_INTRO}"
        "# Schema\n"
        f"{_M2_CLICK_ELEMENT_SCHEMA}\n"
        "# Rules\n"
        '- Output "element" = the #-tag visible in the screenshot (e.g. "#3" or 3).\n'
        "  Do NOT output x, y coordinates.\n"
    )


def _m2_system_scroll(*, top_k: int) -> str:
    return (
        f"{_M2_ROLE_TASK}"
        f"{_m2_fixed_type_block('scroll')}\n"
        f"{_m2_scroll_topk_annotation_legend(top_k)}"
        f"{_M2_COMMON_RULE_HEADER}"
        f"{_m2_scroll_gesture_direction_rules()}\n"
        f"{_M2_FIXED_OUTPUT_INTRO}"
        "# Schema\n"
        f"{_M2_SCROLL_ELEMENT_SCHEMA}\n"
        "# Rules\n"
        '- Select "element" = the scrollable-region tag (#N) most relevant to the step.\n'
        "  Do NOT default to the largest box; pick the region that matches the step description.\n"
        '- "direction" = finger swipe direction (see rules above).\n'
        "  Do NOT output x, y coordinates.\n"
    )


def _m2_system_input() -> str:
    return (
        f"{_M2_ROLE_TASK}"
        f"{_m2_fixed_type_block('input')}\n"
        "# Screen\n"
        "No annotated elements. Plain screenshot for context.\n\n"
        f"{_M2_COMMON_RULE_HEADER}"
        f"{_M2_FIXED_OUTPUT_INTRO}"
        "# Schema\n"
        f"{_M2_INPUT_TEXT_SCHEMA}\n"
        "# Rules\n"
        "- Output the text to type; do NOT output element or coordinates.\n"
    )


def _m2_system_back() -> str:
    return (
        f"{_M2_ROLE_TASK}"
        f"{_m2_fixed_type_block('back')}\n"
        "# Screen\n"
        "No annotated elements.\n\n"
        "# Back Rule\n"
        "back() returns to the previous page. No element or coordinates needed.\n\n"
        f"{_M2_COMMON_RULE_HEADER}"
        f"{_M2_FIXED_OUTPUT_INTRO}"
        "# Schema\n"
        f"{_M2_BACK_THOUGHT_SCHEMA}\n"
    )


def build_m2_prompt_parts(
    step_instruction: str,
    *,
    fixed_action_type: str,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
    target_object: str = "",
    top_k: int = 10,
) -> tuple[str, str]:
    """返回 (system_prompt, user_prompt)。"""
    common_user = {
        "task_desc": task_desc,
        "current_page_name": current_page_name,
        "last_summary": last_summary,
        "target_object": target_object,
        "include_back": True,
    }
    ft = (fixed_action_type or "click").strip()
    if ft in ("click", "long_press"):
        return (
            _m2_system_click(fixed_action_type=ft, top_k=top_k),
            _m2_user_dynamic_block(step_instruction, **common_user),
        )
    if ft == "scroll":
        return (
            _m2_system_scroll(top_k=top_k),
            _m2_user_dynamic_block(step_instruction, **common_user),
        )
    if ft == "back":
        return (
            _m2_system_back(),
            _m2_user_dynamic_block(step_instruction, **common_user),
        )
    return (
        _m2_system_input(),
        _m2_user_dynamic_block(step_instruction, **common_user),
    )


_TO_ROLE_TASK = (
    "# 角色\n"
    "你是一个在智能手机上执行多步 GUI 任务的操作智能体。\n"
    "截图中的检索候选以纯数字 1、2、3… 标注（本步仅 click 或 scroll 一种类型），标签位于元素中心。\n\n"
    f"{_m2_action_catalog()}\n"
    "# 任务\n"
    "仅输出一个 JSON 对象，完成当前子任务动作。\n\n"
)

_TO_COMMON_RULE = (
    "# 规则\n"
    "- 仅输出紧凑的原始 JSON，不要 markdown 或代码块。\n"
    "- 只需完成本步子任务，无需判断整体任务是否结束。\n\n"
)

_TO_SCREENSHOT_SUFFIX = "当前屏幕截图："


def _to_user_dynamic_block(
    step_instruction: str,
    *,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
    target_object: str = "",
    include_back: bool = False,
) -> str:
    has_history = bool(last_summary.strip())
    parts = [
        _m2_to_user_context_block(
            task_desc=task_desc,
            step_instruction=step_instruction,
            current_page_name=current_page_name,
            last_summary=last_summary,
        )
        + "\n"
    ]
    to_line = _scroll_to_line(target_object)
    if to_line:
        parts.append(to_line)
    if include_back:
        parts.append(
            _append_back_sections(step_instruction, include_mismatch_hint=has_history)
        )
    parts.append(f"\n{_TO_SCREENSHOT_SUFFIX}")
    return "".join(parts)


def _to_system_click(*, fixed_action_type: str) -> str:
    return (
        f"{_TO_ROLE_TASK}"
        f"{_fixed_type_block(fixed_action_type)}"
        "  【截图标注】本步为点击检索，截图已用 Top-1 高亮单个候选区域（数字标签）。\n\n"
        f"{_TO_COMMON_RULE}"
        f"{_TO_FIXED_OUTPUT_INTRO}"
        "# Schema\n"
        f"{_TO_THOUGHT_ONLY_SCHEMA}\n"
        "# 规则\n"
        "  仅输出 thought 与 summary；系统将使用检索 Top-1 区域执行点击。\n"
    )


def _to_system_scroll() -> str:
    return (
        f"{_TO_ROLE_TASK}"
        f"{_fixed_type_block('scroll')}"
        f"{_scroll_top1_annotation_legend()}\n"
        f"{_TO_COMMON_RULE}"
        f"{_TO_FIXED_OUTPUT_INTRO}"
        "# Schema\n"
        f"{_TO_SCROLL_DIRECTION_SCHEMA}\n"
        "# 规则\n"
        "  仅输出 direction；系统将使用检索 Top-1 区域执行 scroll。不要输出 element。\n"
        f"{_scroll_direction_rules()}"
    )


def _to_system_input() -> str:
    return (
        f"{_TO_ROLE_TASK}"
        f"{_fixed_type_block('input')}"
        "  【截图标注】本步为输入操作，截图为无标注原图。\n\n"
        f"{_TO_COMMON_RULE}"
        f"{_TO_FIXED_OUTPUT_INTRO}"
        "# Schema\n"
        f"{_INPUT_TEXT_SCHEMA}\n"
        "# 规则\n"
        "  请输出要输入的文本。\n"
    )


def _to_system_back() -> str:
    return (
        f"{_TO_ROLE_TASK}"
        f"{_fixed_type_block('back')}"
        f"{_back_action_spec()}\n"
        f"{_TO_COMMON_RULE}"
        f"{_TO_FIXED_OUTPUT_INTRO}"
        "# Schema\n"
        f"{_BACK_THOUGHT_SCHEMA}\n"
    )


def build_to_prompt_parts(
    step_instruction: str,
    *,
    fixed_action_type: str,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
    target_object: str = "",
    top_k: int = 10,
) -> tuple[str, str]:
    """返回 (system_prompt, user_prompt)。"""
    del top_k
    common_user = {
        "task_desc": task_desc,
        "current_page_name": current_page_name,
        "last_summary": last_summary,
        "target_object": target_object,
        "include_back": True,
    }
    ft = (fixed_action_type or "click").strip()
    if ft in ("click", "long_press"):
        return (
            _to_system_click(fixed_action_type=ft),
            _to_user_dynamic_block(step_instruction, **common_user),
        )
    if ft == "scroll":
        return (
            _to_system_scroll(),
            _to_user_dynamic_block(step_instruction, **common_user),
        )
    if ft == "back":
        return (
            _to_system_back(),
            _to_user_dynamic_block(step_instruction, **common_user),
        )
    return (
        _to_system_input(),
        _to_user_dynamic_block(step_instruction, **common_user),
    )


def _back_multipath_hint(step_instruction: str) -> str:
    return (
        "  【Multi-path 决策】若当前页面无法完成本步指令、截图无可操作标注、"
        "或页面状态与预期不符，可考虑输出 back 返回上一页再重新规划。\n"
        f"  当前步骤指令：{step_instruction}\n"
    )


def _append_back_sections_to(step_instruction: str) -> str:
    return f"\n{_back_action_spec()}\n{_back_multipath_hint(step_instruction)}"


def build_m2_scroll_prompt(
    step_instruction: str,
    *,
    task_desc: str,
    current_page_name: str,
    last_summary: str = "",
) -> str:
    has_history = bool(last_summary.strip())
    hints = _instruction_hints_3x(step_instruction)
    return (
        f"{_m2_en_task_context_block(task_desc=task_desc, step_instruction=step_instruction, current_page_name=current_page_name, last_summary=last_summary)}"
        f"{_m2_scroll_topk_annotation_legend(10)}"
        f"{hints}"
        "# Rule\n"
        "This step should be scroll. Only complete the current sub-task.\n"
        f"{_m2_scroll_gesture_direction_rules()}\n"
        f"{_m2_en_append_back_sections(step_instruction, include_mismatch_hint=has_history)}\n"
        f"{_M2_FIXED_OUTPUT_INTRO}"
        "# Schema\n"
        f"{_M2_SCROLL_ELEMENT_SCHEMA}"
        "Current screen screenshot:"
    )


def build_m2_topk_prompt(
    step_instruction: str,
    *,
    top_k: int = 10,
    task_desc: str,
    current_page_name: str,
    last_summary: str = "",
) -> str:
    has_history = bool(last_summary.strip())
    hints = _instruction_hints_3x(step_instruction)
    return (
        f"{_m2_en_task_context_block(task_desc=task_desc, step_instruction=step_instruction, current_page_name=current_page_name, last_summary=last_summary)}"
        f"{_m2_topk_annotation_legend(top_k)}"
        f"{hints}"
        "# Rule\n"
        "- Decide click, scroll, input, or back from the sub-task.\n"
        '- For click: output element = visible #-tag; do NOT output x, y.\n'
        f"{_m2_scroll_gesture_direction_rules()}"
        f"{_m2_en_append_back_sections(step_instruction, include_mismatch_hint=has_history)}\n"
        f"{_M2_FIXED_OUTPUT_INTRO}"
        "# Schema\n"
        f"{_M2_JSON_SCHEMA}"
        "Current screen screenshot:"
    )


def build_m2_input_prompt(
    step_instruction: str,
    *,
    task_desc: str,
    current_page_name: str,
    last_summary: str = "",
) -> str:
    return build_m2_input_fixed_prompt(
        step_instruction,
        task_desc=task_desc,
        current_page_name=current_page_name,
        last_summary=last_summary,
    )


def build_m2_click_fixed_prompt(
    step_instruction: str,
    *,
    top_k: int = 10,
    target_object: str = "",
    fixed_action_type: str = "click",
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
) -> str:
    system, user = build_m2_prompt_parts(
        step_instruction,
        fixed_action_type=fixed_action_type,
        top_k=top_k,
        target_object=target_object,
        task_desc=task_desc,
        current_page_name=current_page_name,
        last_summary=last_summary,
    )
    return f"{system}\n{user}"


def build_m2_scroll_fixed_prompt(
    step_instruction: str,
    *,
    top_k: int = 10,
    target_object: str = "",
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
) -> str:
    system, user = build_m2_prompt_parts(
        step_instruction,
        fixed_action_type="scroll",
        top_k=top_k,
        target_object=target_object,
        task_desc=task_desc,
        current_page_name=current_page_name,
        last_summary=last_summary,
    )
    return f"{system}\n{user}"


def build_m2_input_fixed_prompt(
    step_instruction: str,
    *,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
) -> str:
    system, user = build_m2_prompt_parts(
        step_instruction,
        fixed_action_type="input",
        task_desc=task_desc,
        current_page_name=current_page_name,
        last_summary=last_summary,
    )
    return f"{system}\n{user}"


def build_m2_back_fixed_prompt(
    step_instruction: str,
    *,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
) -> str:
    system, user = build_m2_prompt_parts(
        step_instruction,
        fixed_action_type="back",
        task_desc=task_desc,
        current_page_name=current_page_name,
        last_summary=last_summary,
    )
    return f"{system}\n{user}"


def build_m2_user_prompt(
    step_instruction: str,
    *,
    fixed_action_type: str,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
    target_object: str = "",
    top_k: int = 10,
) -> str:
    system, user = build_m2_prompt_parts(
        step_instruction,
        fixed_action_type=fixed_action_type,
        task_desc=task_desc,
        current_page_name=current_page_name,
        last_summary=last_summary,
        target_object=target_object,
        top_k=top_k,
    )
    return f"{system}\n{user}"


def build_to_scroll_prompt(
    step_instruction: str,
    *,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
) -> str:
    has_history = bool(last_summary.strip())
    system = (
        f"{_TO_ROLE_TASK}"
        f"{_scroll_official_annotation_legend()}\n"
        "  本步应执行 scroll：输出 element=数字编号 与 direction，不要输出 click。\n"
        f"{_scroll_action_spec()}\n"
        f"{_scroll_direction_rules()}\n"
        f"{_TO_COMMON_RULE}"
        f"{_M2_OUTPUT_INTRO}"
        "# Schema\n"
        f"{_SCROLL_JSON_SCHEMA}"
        f"{_BACK_JSON_SCHEMA}"
    )
    user = (
        _m2_to_user_context_block(
            task_desc=task_desc,
            step_instruction=step_instruction,
            current_page_name=current_page_name,
            last_summary=last_summary,
        )
        + "\n"
        + _append_back_sections(step_instruction, include_mismatch_hint=has_history)
        + f"\n{_TO_SCREENSHOT_SUFFIX}"
    )
    return f"{system}\n{user}"


def build_to_click_fixed_prompt(
    step_instruction: str,
    *,
    target_object: str = "",
    fixed_action_type: str = "click",
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
) -> str:
    system, user = build_to_prompt_parts(
        step_instruction,
        fixed_action_type=fixed_action_type,
        target_object=target_object,
        task_desc=task_desc,
        current_page_name=current_page_name,
        last_summary=last_summary,
    )
    return f"{system}\n{user}"


def build_to_input_prompt(
    step_instruction: str,
    *,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
) -> str:
    system, user = build_to_prompt_parts(
        step_instruction,
        fixed_action_type="input",
        task_desc=task_desc,
        current_page_name=current_page_name,
        last_summary=last_summary,
    )
    return f"{system}\n{user}"


def build_to_click_top1_prompt(
    step_instruction: str,
    *,
    target_object: str = "",
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
) -> str:
    has_history = bool(last_summary.strip())
    to_line = ""
    if target_object.strip():
        to_line = f'检索目标 TO="{target_object.strip()}"\n'
    system = (
        f"{_TO_ROLE_TASK}"
        "  【截图标注】本步为点击检索，截图已用 Top-1 高亮单个候选区域（数字标签）。\n\n"
        f"{_TO_COMMON_RULE}"
        f"{_M2_OUTPUT_INTRO}"
        "# Schema\n"
        f"{_CLICK_ONLY_SCHEMA}"
        "  （scroll/input/back 步用完整 schema）\n"
        f"{_JSON_SCHEMA}\n"
        "# 规则\n"
        "  若本步应为 click：仅输出 action.type=click，不要输出 element；系统将使用检索 Top-1 区域。\n"
        "  若本步应为 scroll、input 或 back（与截图标注不符时以指令为准），按对应类型完整输出。\n"
    )
    user = (
        _m2_to_user_context_block(
            task_desc=task_desc,
            step_instruction=step_instruction,
            current_page_name=current_page_name,
            last_summary=last_summary,
        )
        + "\n"
        + to_line
        + _append_back_sections(step_instruction, include_mismatch_hint=has_history)
        + f"\n{_TO_SCREENSHOT_SUFFIX}"
    )
    return f"{system}\n{user}"


def build_to_scroll_fixed_prompt(
    step_instruction: str,
    *,
    target_object: str = "",
    top_k: int = 10,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
) -> str:
    system, user = build_to_prompt_parts(
        step_instruction,
        fixed_action_type="scroll",
        target_object=target_object,
        top_k=top_k,
        task_desc=task_desc,
        current_page_name=current_page_name,
        last_summary=last_summary,
    )
    return f"{system}\n{user}"


def build_to_input_fixed_prompt(
    step_instruction: str,
    *,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
) -> str:
    system, user = build_to_prompt_parts(
        step_instruction,
        fixed_action_type="input",
        task_desc=task_desc,
        current_page_name=current_page_name,
        last_summary=last_summary,
    )
    return f"{system}\n{user}"


def build_to_back_fixed_prompt(
    step_instruction: str,
    *,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
) -> str:
    system, user = build_to_prompt_parts(
        step_instruction,
        fixed_action_type="back",
        task_desc=task_desc,
        current_page_name=current_page_name,
        last_summary=last_summary,
    )
    return f"{system}\n{user}"


def build_to_user_prompt(
    step_instruction: str,
    *,
    fixed_action_type: str,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
    target_object: str = "",
    top_k: int = 10,
) -> str:
    system, user = build_to_prompt_parts(
        step_instruction,
        fixed_action_type=fixed_action_type,
        task_desc=task_desc,
        current_page_name=current_page_name,
        last_summary=last_summary,
        target_object=target_object,
        top_k=top_k,
    )
    return f"{system}\n{user}"
