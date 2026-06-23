"""VLM prompts for m2 (English) / TO (Chinese) agents (SMAN action space)."""
from __future__ import annotations

import re

VLM_ACTION_TYPES = frozenset({"click", "scroll", "input", "long_press"})

_JSON_SCHEMA = (
    '  {"thought":"...","action":{"type":"click|scroll|input",'
    '"element":"N","x":0.0,"y":0.0,"direction":"up|down|left|right","text":"..."}}\n'
)

_M2_JSON_SCHEMA = (
    '  {"thought":"...","action":{"type":"click|scroll|input",'
    '"element":"N","direction":"up|down|left|right","text":"..."}}\n'
)

_SCROLL_JSON_SCHEMA = (
    '  {"thought":"...","action":{"type":"scroll","element":"N",'
    '"direction":"up|down|left|right"}}\n'
)

_INPUT_JSON_SCHEMA = (
    '  {"thought":"...","action":{"type":"input","text":"..."}}\n'
)

_CLICK_ONLY_SCHEMA = '  {"thought":"...","action":{"type":"click"}}\n'

_SCROLL_DIRECTION_ONLY_SCHEMA = (
    '  {"thought":"...","action":{"direction":"up|down|left|right"}}\n'
)


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


def _scroll_top1_annotation_legend() -> str:
    return (
        "  【截图标注】本步为滑动检索，截图已用 Top-1 高亮单个可滑动区域候选。\n"
        "    · 可滑动区域：蓝色虚线框（无填充），标签为纯数字 1、2、3…（可能在框内或框旁）\n"
    )


def _scroll_topk_annotation_legend(top_k: int) -> str:
    return (
        f"  【截图标注】本步为滑动操作，截图展示语义检索 Top-{top_k} 个可滑动区域候选（非全页所有区域）：\n"
        "    · 可滑动区域：蓝色虚线框（无填充），标签为纯数字 1、2、3…\n"
        "  必须从图中可见的编号中选择 element，并给出 direction；不要假设未标注区域。\n"
        "  若存在嵌套可滑动区域，选择与步骤指令描述的内容范围最匹配者，勿默认最大框。\n"
    )


def _scroll_to_line(target_object: str) -> str:
    if target_object.strip():
        return f'  检索目标 TO="{target_object.strip()}"\n'
    return ""


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


_CLICK_ELEMENT_SCHEMA = (
    '  {"thought":"...","action":{"element":"#N"}}\n'
)

_TO_THOUGHT_ONLY_SCHEMA = '  {"thought":"..."}\n'


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


_M2_CLICK_ELEMENT_SCHEMA = (
    '  {"thought": "1-3 concise sentences", "action": {"element": "#N"}}\n'
)

_M2_SCROLL_ELEMENT_SCHEMA = (
    '  {"thought": "...", "action": {"element": "#N", '
    '"direction": "up|down|left|right"}}\n'
)

_M2_INPUT_TEXT_SCHEMA = '  {"thought": "...", "action": {"text": "..."}}\n'

_M2_ROLE_TASK = (
    "# Role\n"
    "You are an Android GUI automation agent executing multi-step tasks on a smartphone.\n\n"
    "# Task\n"
    "Output exactly ONE JSON object for the immediate next action.\n\n"
)

_M2_COMMON_RULE_HEADER = (
    "# Rule\n"
    "- Output compact raw JSON only. No markdown, no code fences.\n"
    "- Only complete the current step; do not judge whether the entire task is finished.\n"
    "- Do NOT output Chinese.\n\n"
)

_SCREENSHOT_SUFFIX = "Current screen screenshot:"


def _m2_user_dynamic_block(
    step_instruction: str,
    *,
    target_object: str = "",
) -> str:
    parts = [f"Step instruction (current step): {step_instruction}\n"]
    hints = _instruction_hints_3x(step_instruction)
    if hints:
        parts.append(hints)
    target = _m2_retrieved_target_line(target_object)
    if target:
        parts.append(target)
    parts.append(f"\n{_SCREENSHOT_SUFFIX}")
    return "".join(parts)


def _m2_system_click(*, fixed_action_type: str, top_k: int) -> str:
    return (
        f"{_M2_ROLE_TASK}"
        f"{_m2_fixed_type_block(fixed_action_type)}\n"
        f"{_m2_topk_annotation_legend(top_k)}"
        f"{_M2_COMMON_RULE_HEADER}"
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
        "# Schema\n"
        f"{_M2_SCROLL_ELEMENT_SCHEMA}\n"
        "# Rules\n"
        f"{_m2_scroll_gesture_direction_rules()}\n"
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
        "# Schema\n"
        f"{_M2_INPUT_TEXT_SCHEMA}\n"
        "# Rules\n"
        "- Output the text to type; do NOT output element or coordinates.\n"
    )


def build_m2_prompt_parts(
    step_instruction: str,
    *,
    fixed_action_type: str,
    top_k: int = 10,
    target_object: str = "",
) -> tuple[str, str]:
    """返回 (system_prompt, user_prompt)。"""
    ft = (fixed_action_type or "click").strip()
    user = _m2_user_dynamic_block(step_instruction, target_object=target_object)
    if ft in ("click", "long_press"):
        return _m2_system_click(fixed_action_type=ft, top_k=top_k), user
    if ft == "scroll":
        return _m2_system_scroll(top_k=top_k), user
    return _m2_system_input(), user


def build_m2_click_fixed_prompt(
    step_instruction: str,
    *,
    top_k: int = 10,
    target_object: str = "",
    fixed_action_type: str = "click",
) -> str:
    system, user = build_m2_prompt_parts(
        step_instruction,
        fixed_action_type=fixed_action_type,
        top_k=top_k,
        target_object=target_object,
    )
    return f"{system}\n{user}"


def build_m2_scroll_fixed_prompt(
    step_instruction: str,
    *,
    top_k: int = 10,
    target_object: str = "",
) -> str:
    system, user = build_m2_prompt_parts(
        step_instruction,
        fixed_action_type="scroll",
        top_k=top_k,
        target_object=target_object,
    )
    return f"{system}\n{user}"


def build_m2_input_fixed_prompt(step_instruction: str) -> str:
    system, user = build_m2_prompt_parts(
        step_instruction,
        fixed_action_type="input",
    )
    return f"{system}\n{user}"


def build_m2_topk_prompt(step_instruction: str, *, top_k: int = 10) -> str:
    hints = _instruction_hints_3x(step_instruction)
    system = (
        f"{_M2_ROLE_TASK}"
        f"{_m2_topk_annotation_legend(top_k)}"
        f"{_M2_COMMON_RULE_HEADER}"
        "# Schema\n"
        f"{_M2_JSON_SCHEMA}"
        "# Rules\n"
        "- Decide whether this step is click, scroll, or input from the instruction.\n"
        '- For click: output "element" = visible #-tag; do NOT output x, y.\n'
        f"{_m2_scroll_gesture_direction_rules()}"
    )
    user = (
        f"Step instruction (current step): {step_instruction}\n"
        f"{hints}\n"
        f"{_SCREENSHOT_SUFFIX}"
    )
    return f"{system}\n{user}"


def build_m2_input_prompt(step_instruction: str) -> str:
    return build_m2_input_fixed_prompt(step_instruction)


def build_m2_user_prompt(
    step_instruction: str,
    *,
    fixed_action_type: str,
    top_k: int = 10,
    target_object: str = "",
) -> str:
    system, user = build_m2_prompt_parts(
        step_instruction,
        fixed_action_type=fixed_action_type,
        top_k=top_k,
        target_object=target_object,
    )
    return f"{system}\n{user}"


_TO_ROLE_TASK = (
    "# 角色\n"
    "你是一个在智能手机上执行 GUI 自动化任务的操作智能体。\n\n"
    "# 任务\n"
    "仅输出一个 JSON 对象，完成当前步骤动作。\n\n"
)

_TO_COMMON_RULE = (
    "# 规则\n"
    "- 仅输出紧凑的原始 JSON，不要 markdown 或代码块。\n"
    "- 只需完成本步操作，无需判断任务是否结束。\n\n"
)

_TO_SCREENSHOT_SUFFIX = "当前屏幕截图："


def _to_user_step_block(step_instruction: str, *, target_object: str = "") -> str:
    parts = [f"当前步骤指令：{step_instruction}\n"]
    if target_object.strip():
        parts.append(f'检索目标 TO="{target_object.strip()}"\n')
    parts.append(f"\n{_TO_SCREENSHOT_SUFFIX}")
    return "".join(parts)


def _to_system_click(*, fixed_action_type: str) -> str:
    return (
        f"{_TO_ROLE_TASK}"
        f"{_fixed_type_block(fixed_action_type)}"
        "  【截图标注】本步为点击检索，截图已用 Top-1 高亮单个候选区域（数字标签）。\n\n"
        f"{_TO_COMMON_RULE}"
        "# Schema\n"
        f"{_TO_THOUGHT_ONLY_SCHEMA}\n"
        "# 规则\n"
        "  仅输出 thought；系统将使用检索 Top-1 区域执行点击。\n"
    )


def _to_system_scroll() -> str:
    return (
        f"{_TO_ROLE_TASK}"
        f"{_fixed_type_block('scroll')}"
        f"{_scroll_top1_annotation_legend()}\n"
        f"{_TO_COMMON_RULE}"
        "# Schema\n"
        f"{_SCROLL_DIRECTION_ONLY_SCHEMA}\n"
        "# 规则\n"
        "  仅输出 direction；系统将使用检索 Top-1 区域执行 scroll。不要输出 element。\n"
        f"{_scroll_direction_rules()}"
    )


def _to_system_input() -> str:
    return (
        f"{_TO_ROLE_TASK}"
        "  【截图标注】本步为输入操作，截图为无标注原图。\n\n"
        f"{_TO_COMMON_RULE}"
        "# Schema\n"
        f"{_INPUT_JSON_SCHEMA}\n"
        "# 规则\n"
        "  请输出 input 动作及要输入的文本。\n"
    )


def build_to_prompt_parts(
    step_instruction: str,
    *,
    fixed_action_type: str,
    target_object: str = "",
    top_k: int = 10,
) -> tuple[str, str]:
    """返回 (system_prompt, user_prompt)。"""
    del top_k
    ft = (fixed_action_type or "click").strip()
    user = _to_user_step_block(step_instruction, target_object=target_object)
    if ft in ("click", "long_press"):
        return _to_system_click(fixed_action_type=ft), user
    if ft == "scroll":
        return _to_system_scroll(), user
    return _to_system_input(), user


def build_to_click_fixed_prompt(step_instruction: str, *, target_object: str = "") -> str:
    system, user = build_to_prompt_parts(
        step_instruction,
        fixed_action_type="click",
        target_object=target_object,
    )
    return f"{system}\n{user}"


def build_to_scroll_fixed_prompt(step_instruction: str, *, target_object: str = "", top_k: int = 10) -> str:
    system, user = build_to_prompt_parts(
        step_instruction,
        fixed_action_type="scroll",
        target_object=target_object,
        top_k=top_k,
    )
    return f"{system}\n{user}"


def build_to_input_prompt(step_instruction: str) -> str:
    system, user = build_to_prompt_parts(
        step_instruction,
        fixed_action_type="input",
    )
    return f"{system}\n{user}"


def build_to_click_top1_prompt(step_instruction: str, *, target_object: str = "") -> str:
    to_line = ""
    if target_object.strip():
        to_line = f'检索目标 TO="{target_object.strip()}"\n'
    system = (
        f"{_TO_ROLE_TASK}"
        "  【截图标注】本步为点击检索，截图已用 Top-1 高亮单个候选区域。\n\n"
        f"{_TO_COMMON_RULE}"
        "# Schema\n"
        f"{_CLICK_ONLY_SCHEMA}"
        "  （scroll/input 步用完整 schema）\n"
        f"{_JSON_SCHEMA}\n"
        "# 规则\n"
        "  若本步应为 click：仅输出 action.type=click，不要输出 element；系统将使用检索 Top-1 区域。\n"
        "  若本步应为 scroll 或 input（与截图标注不符时以指令为准），按对应类型完整输出。\n"
    )
    user = (
        f"当前步骤指令：{step_instruction}\n"
        f"{to_line}\n"
        f"{_TO_SCREENSHOT_SUFFIX}"
    )
    return f"{system}\n{user}"


def build_to_user_prompt(
    step_instruction: str,
    *,
    fixed_action_type: str,
    target_object: str = "",
    top_k: int = 10,
) -> str:
    system, user = build_to_prompt_parts(
        step_instruction,
        fixed_action_type=fixed_action_type,
        target_object=target_object,
        top_k=top_k,
    )
    return f"{system}\n{user}"
