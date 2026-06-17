"""Chinese VLM prompts for m2 / TO / TOa agents (SMAN action space)."""
from __future__ import annotations

VLM_ACTION_TYPES = frozenset({"click", "scroll", "input"})

# TOa 低置信阈值（待定，实验后再启用 _toa_low_confidence_hint）
TAU_SIM_STRONG: float | None = 0.50
TAU_MARGIN_STRONG: float | None = 0.04

_JSON_SCHEMA = (
    '  {"thought":"...","action":{"type":"click|scroll|input",'
    '"element":"cN 或 sN","x":0.0,"y":0.0,"direction":"up|down|left|right","text":"..."}}\n'
)

_M2_JSON_SCHEMA = (
    '  {"thought":"...","action":{"type":"click|scroll|input",'
    '"element":"cN 或 sN","direction":"up|down|left|right","text":"..."}}\n'
)

_SCROLL_JSON_SCHEMA = (
    '  {"thought":"...","action":{"type":"scroll","element":"sN",'
    '"direction":"up|down|left|right"}}\n'
)

_INPUT_JSON_SCHEMA = (
    '  {"thought":"...","action":{"type":"input","text":"..."}}\n'
)

_CLICK_ONLY_SCHEMA = '  {"thought":"...","action":{"type":"click"}}\n'


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


def build_m2_scroll_prompt(step_instruction: str) -> str:
    return (
        "User:\n"
        f"  当前步骤指令：{step_instruction}\n\n"
        f"{_scroll_official_annotation_legend()}\n"
        "  本步应执行 scroll。只需完成本步操作，无需判断任务是否结束。\n"
        f"{_scroll_action_spec()}\n"
        f"{_scroll_direction_rules()}\n"
        "  仅输出一个 JSON 对象：\n"
        f"{_SCROLL_JSON_SCHEMA}"
    )


def build_m2_topk_prompt(step_instruction: str, *, top_k: int = 10) -> str:
    return (
        "User:\n"
        f"  当前步骤指令：{step_instruction}\n\n"
        f"{_topk_annotation_legend(top_k)}\n"
        "  只需完成本步操作，无需判断任务是否结束。\n"
        "  根据指令自行判断本步应执行 click、scroll 或 input。\n"
        "  click 时必须输出 element=cN（图中可见红色标签）；不要输出 x,y 坐标。\n"
        f"{_scroll_action_spec()}\n"
        f"{_scroll_direction_rules()}\n"
        "  仅输出一个 JSON 对象：\n"
        f"{_M2_JSON_SCHEMA}"
    )


def build_m2_input_prompt(step_instruction: str) -> str:
    return (
        "User:\n"
        f"  当前步骤指令：{step_instruction}\n\n"
        "  【截图标注】本步为输入操作，截图为无标注原图。\n"
        "  只需完成本步操作，无需判断任务是否结束。\n"
        "  请输出 input 动作及要输入的文本。\n\n"
        "  仅输出一个 JSON 对象：\n"
        f"{_INPUT_JSON_SCHEMA}"
    )


def build_m2_user_prompt(
    step_instruction: str,
    *,
    instruction_hit: str | None = None,
    top_k: int = 10,
) -> str:
    if instruction_hit == "scroll":
        return build_m2_scroll_prompt(step_instruction)
    if instruction_hit == "input":
        return build_m2_input_prompt(step_instruction)
    return build_m2_topk_prompt(step_instruction, top_k=top_k)


def build_to_scroll_prompt(step_instruction: str) -> str:
    return (
        "User:\n"
        f"  当前步骤指令：{step_instruction}\n\n"
        f"{_scroll_official_annotation_legend()}\n"
        "  只需完成本步操作，无需判断任务是否结束。\n"
        "  本步应执行 scroll：输出 element=sN 与 direction，不要输出 click。\n"
        f"{_scroll_action_spec()}\n"
        f"{_scroll_direction_rules()}\n"
        "  仅输出一个 JSON 对象：\n"
        f"{_SCROLL_JSON_SCHEMA}"
    )


def build_to_input_prompt(step_instruction: str) -> str:
    return (
        "User:\n"
        f"  当前步骤指令：{step_instruction}\n\n"
        "  【截图标注】本步为输入操作，截图为无标注原图。\n"
        "  只需完成本步操作，无需判断任务是否结束。\n"
        "  请输出 input 动作及要输入的文本。\n\n"
        "  仅输出一个 JSON 对象：\n"
        f"{_INPUT_JSON_SCHEMA}"
    )


def build_to_click_top1_prompt(step_instruction: str, *, target_object: str = "") -> str:
    to_line = ""
    if target_object.strip():
        to_line = f'  检索目标 TO="{target_object.strip()}"\n'
    return (
        "User:\n"
        f"  当前步骤指令：{step_instruction}\n"
        f"{to_line}"
        "  【截图标注】本步为点击检索，截图已用 Top-1 高亮单个候选区域（cN）。\n"
        "  只需完成本步操作，无需判断任务是否结束。\n"
        "  若本步应为 click：仅输出 action.type=click，不要输出 element；系统将使用检索 Top-1 区域。\n"
        "  若本步应为 scroll 或 input（与截图标注不符时以指令为准），按对应类型完整输出。\n\n"
        "  仅输出一个 JSON 对象：\n"
        f"{_CLICK_ONLY_SCHEMA}"
        "  （scroll/input 步用完整 schema）\n"
        f"{_JSON_SCHEMA}"
    )


def build_to_user_prompt(
    step_instruction: str,
    *,
    target_object: str = "",
    instruction_hit: str | None = None,
) -> str:
    if instruction_hit == "scroll":
        return build_to_scroll_prompt(step_instruction)
    if instruction_hit == "input":
        return build_to_input_prompt(step_instruction)
    return build_to_click_top1_prompt(step_instruction, target_object=target_object)


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


def build_toa_scroll_prompt(step_instruction: str) -> str:
    return (
        "User:\n"
        f"  当前步骤指令：{step_instruction}\n\n"
        f"{_scroll_official_annotation_legend()}\n"
        "  只需完成本步操作，无需判断任务是否结束。\n"
        "  本步应执行 scroll：输出 element=sN 与 direction；不要输出 click 或坐标 x,y。\n"
        f"{_scroll_action_spec()}\n"
        f"{_scroll_direction_rules()}\n"
        "  仅输出一个 JSON 对象：\n"
        f"{_SCROLL_JSON_SCHEMA}"
    )


def build_toa_input_prompt(step_instruction: str) -> str:
    return build_to_input_prompt(step_instruction)


def build_toa_click_top1_prompt(
    step_instruction: str,
    *,
    target_object: str = "",
    retrieval_final_sim: float | None = None,
    retrieval_margin: float | None = None,
) -> str:
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
        "User:\n"
        f"  当前步骤指令：{step_instruction}\n"
        f"{to_line}{score_line}{low_conf}"
        "  【截图标注】本步为点击检索，截图为 Top-1 建议区域（单个 cN 参考框，可能错误）。\n"
        "  只需完成本步操作，无需判断任务是否结束。\n"
        "  click 时：可信任建议框（省略 element/x,y，由系统使用 Top-1）；"
        "或输出 element=cN；低置信时可输出 x,y（0~1 归一化坐标）指定点击位置。\n"
        "  若本步应为 scroll 或 input，按对应类型完整输出，勿默认 click。\n\n"
        "  仅输出一个 JSON 对象：\n"
        f"{_JSON_SCHEMA}"
    )


def build_toa_user_prompt(
    step_instruction: str,
    *,
    target_object: str = "",
    retrieval_final_sim: float | None = None,
    retrieval_margin: float | None = None,
    instruction_hit: str | None = None,
) -> str:
    if instruction_hit == "scroll":
        return build_toa_scroll_prompt(step_instruction)
    if instruction_hit == "input":
        return build_toa_input_prompt(step_instruction)
    return build_toa_click_top1_prompt(
        step_instruction,
        target_object=target_object,
        retrieval_final_sim=retrieval_final_sim,
        retrieval_margin=retrieval_margin,
    )
