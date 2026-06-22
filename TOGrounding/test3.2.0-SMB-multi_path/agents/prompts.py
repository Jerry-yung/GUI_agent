"""Chinese VLM prompts for m2 / TO / TOa agents (SMAN action space, multi-path)."""
from __future__ import annotations

VLM_ACTION_TYPES = frozenset({"click", "scroll", "input", "back"})

# TOa 低置信阈值（待定，实验后再启用 _toa_low_confidence_hint）
TAU_SIM_STRONG: float | None = 0.50
TAU_MARGIN_STRONG: float | None = 0.04

_SUMMARY_FIELD_SPEC = (
    "  summary：用一两句话总结：①本步执行的操作；②执行后预计进入的页面"
    "（写出页面标识如 QQmusic0_10，或清晰的界面描述）。不要在 summary 中出现 cN/sN 标签。\n"
)

_M2_OUTPUT_INTRO = (
    "  根据截图与任务信息思考并输出本步动作。每次只能执行一个动作。\n"
    "  仅输出一个 JSON 对象，包含 thought、summary、action 三个字段：\n"
    "  thought：描述你在截图中的观察，以及为完成当前子任务下一步应做什么。\n"
    f"{_SUMMARY_FIELD_SPEC}"
)

_JSON_SCHEMA = (
    '  {"thought":"...","summary":"本步操作；预计进入页面：...","action":{"type":"click|scroll|input|back",'
    '"element":"cN 或 sN","x":0.0,"y":0.0,"direction":"up|down|left|right","text":"..."}}\n'
)

_M2_JSON_SCHEMA = (
    '  {"thought":"...","summary":"本步操作；预计进入页面：...","action":{"type":"click|scroll|input|back",'
    '"element":"cN 或 sN","direction":"up|down|left|right","text":"..."}}\n'
)

_SCROLL_JSON_SCHEMA = (
    '  {"thought":"...","summary":"本步操作；预计进入页面：...","action":{"type":"scroll","element":"sN",'
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
        "   点击截图上带 cN 标签的可点击 UI 元素。示例：click(c5)。\n\n"
        "2. input(text)\n"
        "   在输入框中输入文本，text 为要输入的字符串。示例：input(\"你好\")。\n\n"
        "3. scroll(element, direction)\n"
        "   在带 sN 标签的可滚动区域内滑动。direction 为 up / down / left / right 之一。\n"
        '   示例：scroll(s2, "down")。\n\n'
        "4. back()\n"
        "   返回上一页。示例：back()。\n"
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
        "  截图中的可点击元素以 c1、c2、… 标注，可滚动元素以 s1、s2、… 标注，标签位于元素中心。",
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


def _scroll_direction_rules() -> str:
    return (
        "  【方向约定】action.direction 表示手指在 element 所指 sN 区域内的滑动方向：\n"
        "    up    = 手指从下向上拖\n"
        "    down  = 手指从上向下拖\n"
        "    left  = 手指从右向左拖\n"
        "    right = 手指从左向右拖\n"
        "  请根据步骤指令语义选择区域 sN 与方向；不要按「页面内容往哪移动」理解，"
        "而应转换为手指在区域内的拖拽方向。\n"
        '  示例：scroll(s2, "down") 表示在 s2 区域内手指从上向下拖。\n'
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
        "    · 可点击 c1、c2、…：红色实线框 + 浅红填充，标签可能在框内或框旁\n"
        "    · 可滑动 s1、s2、…：蓝色虚线框（无填充），标签可能在框内或框旁\n"
        "  若本步应 scroll，必须从图中可见的 sN 中选择并给出 direction；"
        "若应 click，从可见 cN 中选择；若应 input，按 input 规则输出（可能为无标注原图）。\n"
        "  勿因图中 click 框更多就默认 click；请结合步骤指令判断动作类型。\n"
    )


def _scroll_action_spec() -> str:
    return (
        "  【scroll 动作】scroll(element, direction)：\n"
        "    element = 图中标签 sN（如 s1）；direction = up | down | left | right 之一。\n"
    )


def _back_action_spec() -> str:
    return (
        "  【back 动作】back()：返回上一页（系统级返回，无 element 参数）。\n"
        '    示例：{"type":"back"} 表示从当前子页面退回上一级页面。\n'
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
    return (
        f"{_m2_task_context_block(task_desc=task_desc, step_instruction=step_instruction, current_page_name=current_page_name, last_summary=last_summary)}\n"
        f"{_scroll_official_annotation_legend()}\n"
        "  本步应执行 scroll。只需完成当前子任务，无需判断整体任务是否结束。\n"
        f"{_scroll_action_spec()}\n"
        f"{_scroll_direction_rules()}\n"
        f"{_append_back_sections(step_instruction, include_mismatch_hint=has_history)}\n"
        f"{_M2_OUTPUT_INTRO}"
        f"{_SCROLL_JSON_SCHEMA}"
        f"{_BACK_JSON_SCHEMA}"
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
    return (
        f"{_m2_task_context_block(task_desc=task_desc, step_instruction=step_instruction, current_page_name=current_page_name, last_summary=last_summary)}\n"
        f"{_topk_annotation_legend(top_k)}\n"
        "  只需完成当前子任务，无需判断整体任务是否结束。\n"
        "  根据子任务自行判断本步应执行 click、scroll、input 或 back。\n"
        "  click 时必须输出 element=cN（图中可见红色标签）；不要输出 x,y 坐标。\n"
        f"{_scroll_action_spec()}\n"
        f"{_scroll_direction_rules()}\n"
        f"{_append_back_sections(step_instruction, include_mismatch_hint=has_history)}\n"
        f"{_M2_OUTPUT_INTRO}"
        f"{_M2_JSON_SCHEMA}"
        f"{_BACK_JSON_SCHEMA}"
    )


def build_m2_input_prompt(
    step_instruction: str,
    *,
    task_desc: str,
    current_page_name: str,
    last_summary: str = "",
) -> str:
    has_history = bool(last_summary.strip())
    return (
        f"{_m2_task_context_block(task_desc=task_desc, step_instruction=step_instruction, current_page_name=current_page_name, last_summary=last_summary)}\n"
        "  【截图标注】本步为输入操作，截图为无标注原图。\n"
        "  只需完成当前子任务，无需判断整体任务是否结束。\n"
        "  请输出 input 动作及要输入的文本。\n"
        f"{_append_back_sections(step_instruction, include_mismatch_hint=has_history)}\n"
        f"{_M2_OUTPUT_INTRO}"
        f"{_INPUT_JSON_SCHEMA}"
        f"{_BACK_JSON_SCHEMA}"
    )


def build_m2_user_prompt(
    step_instruction: str,
    *,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
    instruction_hit: str | None = None,
    top_k: int = 10,
) -> str:
    common = {
        "task_desc": task_desc,
        "current_page_name": current_page_name,
        "last_summary": last_summary,
    }
    if instruction_hit == "scroll":
        return build_m2_scroll_prompt(step_instruction, **common)
    if instruction_hit == "input":
        return build_m2_input_prompt(step_instruction, **common)
    return build_m2_topk_prompt(step_instruction, top_k=top_k, **common)


def build_to_scroll_prompt(
    step_instruction: str,
    *,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
) -> str:
    has_history = bool(last_summary.strip())
    return (
        f"{_m2_task_context_block(task_desc=task_desc, step_instruction=step_instruction, current_page_name=current_page_name, last_summary=last_summary)}\n"
        f"{_scroll_official_annotation_legend()}\n"
        "  只需完成本步操作，无需判断任务是否结束。\n"
        "  本步应执行 scroll：输出 element=sN 与 direction，不要输出 click。\n"
        f"{_scroll_action_spec()}\n"
        f"{_scroll_direction_rules()}\n"
        f"{_append_back_sections(step_instruction, include_mismatch_hint=has_history)}\n"
        f"{_M2_OUTPUT_INTRO}"
        f"{_SCROLL_JSON_SCHEMA}"
        f"{_BACK_JSON_SCHEMA}"
    )


def build_to_input_prompt(
    step_instruction: str,
    *,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
) -> str:
    has_history = bool(last_summary.strip())
    return (
        f"{_m2_task_context_block(task_desc=task_desc, step_instruction=step_instruction, current_page_name=current_page_name, last_summary=last_summary)}\n"
        "  【截图标注】本步为输入操作，截图为无标注原图。\n"
        "  只需完成本步操作，无需判断任务是否结束。\n"
        "  请输出 input 动作及要输入的文本。\n"
        f"{_append_back_sections(step_instruction, include_mismatch_hint=has_history)}\n"
        f"{_M2_OUTPUT_INTRO}"
        f"{_INPUT_JSON_SCHEMA}"
        f"{_BACK_JSON_SCHEMA}"
    )


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
        to_line = f'  检索目标 TO="{target_object.strip()}"\n'
    return (
        f"{_m2_task_context_block(task_desc=task_desc, step_instruction=step_instruction, current_page_name=current_page_name, last_summary=last_summary)}\n"
        f"{to_line}"
        "  【截图标注】本步为点击检索，截图已用 Top-1 高亮单个候选区域（cN）。\n"
        "  只需完成本步操作，无需判断任务是否结束。\n"
        "  若本步应为 click：仅输出 action.type=click，不要输出 element；系统将使用检索 Top-1 区域。\n"
        "  若本步应为 scroll、input 或 back（与截图标注不符时以指令为准），按对应类型完整输出。\n"
        f"{_append_back_sections(step_instruction, include_mismatch_hint=has_history)}\n"
        f"{_M2_OUTPUT_INTRO}"
        f"{_CLICK_ONLY_SCHEMA}"
        "  （scroll/input/back 步用完整 schema）\n"
        f"{_JSON_SCHEMA}"
    )


def build_to_user_prompt(
    step_instruction: str,
    *,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
    target_object: str = "",
    instruction_hit: str | None = None,
) -> str:
    common = {
        "task_desc": task_desc,
        "current_page_name": current_page_name,
        "last_summary": last_summary,
    }
    if instruction_hit == "scroll":
        return build_to_scroll_prompt(step_instruction, **common)
    if instruction_hit == "input":
        return build_to_input_prompt(step_instruction, **common)
    return build_to_click_top1_prompt(
        step_instruction,
        target_object=target_object,
        **common,
    )


def _toa_low_confidence_hint(
    *,
    retrieval_final_sim: float | None,
    retrieval_margin: float | None,
) -> str:
    if TAU_SIM_STRONG is None and TAU_MARGIN_STRONG is None:
        return ""
    sim_low = (
        retrieval_final_sim is not None
        and TAU_SIM_STRONG is not None
        and retrieval_final_sim < TAU_SIM_STRONG
    )
    margin_low = (
        retrieval_margin is not None
        and TAU_MARGIN_STRONG is not None
        and retrieval_margin < TAU_MARGIN_STRONG
    )
    if not sim_low and not margin_low:
        return ""
    return (
        "  检索置信度较低：高亮框可能不准；click 时可输出归一化坐标 x,y（0~1，相对截图宽高），"
        "或输出 element 指定 cN。\n"
    )


def build_toa_scroll_prompt(
    step_instruction: str,
    *,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
) -> str:
    has_history = bool(last_summary.strip())
    return (
        f"{_m2_task_context_block(task_desc=task_desc, step_instruction=step_instruction, current_page_name=current_page_name, last_summary=last_summary)}\n"
        f"{_scroll_official_annotation_legend()}\n"
        "  只需完成本步操作，无需判断任务是否结束。\n"
        "  本步应执行 scroll：输出 element=sN 与 direction；不要输出 click 或坐标 x,y。\n"
        f"{_scroll_action_spec()}\n"
        f"{_scroll_direction_rules()}\n"
        f"{_append_back_sections(step_instruction, include_mismatch_hint=has_history)}\n"
        f"{_M2_OUTPUT_INTRO}"
        f"{_SCROLL_JSON_SCHEMA}"
        f"{_BACK_JSON_SCHEMA}"
    )


def build_toa_input_prompt(
    step_instruction: str,
    *,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
) -> str:
    return build_to_input_prompt(
        step_instruction,
        task_desc=task_desc,
        current_page_name=current_page_name,
        last_summary=last_summary,
    )


def build_toa_click_top1_prompt(
    step_instruction: str,
    *,
    target_object: str = "",
    retrieval_final_sim: float | None = None,
    retrieval_margin: float | None = None,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
) -> str:
    has_history = bool(last_summary.strip())
    to_line = ""
    if target_object.strip():
        to_line = f'  检索目标 TO="{target_object.strip()}"\n'
    score_line = ""
    if retrieval_final_sim is not None:
        margin_s = f"{retrieval_margin:.3f}" if retrieval_margin is not None else "n/a"
        score_line = (
            f"  检索分数 top1={retrieval_final_sim:.3f}；margin(top1-top2)={margin_s}\n"
        )
    low_conf = _toa_low_confidence_hint(
        retrieval_final_sim=retrieval_final_sim,
        retrieval_margin=retrieval_margin,
    )
    return (
        f"{_m2_task_context_block(task_desc=task_desc, step_instruction=step_instruction, current_page_name=current_page_name, last_summary=last_summary)}\n"
        f"{to_line}{score_line}{low_conf}"
        "  【截图标注】本步为点击检索，截图为 Top-1 建议区域（单个 cN 参考框，可能错误）。\n"
        "  只需完成本步操作，无需判断任务是否结束。\n"
        "  click 时：可信任建议框（省略 element/x,y，由系统使用 Top-1）；"
        "或输出 element=cN；低置信时可输出 x,y（0~1 归一化坐标）指定点击位置。\n"
        "  若本步应为 scroll、input 或 back，按对应类型完整输出，勿默认 click。\n"
        f"{_append_back_sections(step_instruction, include_mismatch_hint=has_history)}\n"
        f"{_M2_OUTPUT_INTRO}"
        f"{_JSON_SCHEMA}"
    )


def build_toa_user_prompt(
    step_instruction: str,
    *,
    task_desc: str = "",
    current_page_name: str = "",
    last_summary: str = "",
    target_object: str = "",
    retrieval_final_sim: float | None = None,
    retrieval_margin: float | None = None,
    instruction_hit: str | None = None,
) -> str:
    common = {
        "task_desc": task_desc,
        "current_page_name": current_page_name,
        "last_summary": last_summary,
    }
    if instruction_hit == "scroll":
        return build_toa_scroll_prompt(step_instruction, **common)
    if instruction_hit == "input":
        return build_toa_input_prompt(step_instruction, **common)
    return build_toa_click_top1_prompt(
        step_instruction,
        target_object=target_object,
        retrieval_final_sim=retrieval_final_sim,
        retrieval_margin=retrieval_margin,
        **common,
    )
