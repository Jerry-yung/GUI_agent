"""四个 Agent 统一的 prompt 构建方法。"""


def _critical_rules(rules: list[str]) -> str:
    lines = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, 1))
    return f"关键规则：\n{lines}"


def build_baseline_prompt(instruction: str) -> str:
    """Baseline：原始截图 + 单步指令 → 归一化坐标。"""
    rules = _critical_rules([
        "输出必须是合法的 JSON 对象，且仅包含 x、y 两个字段",
        "x 和 y 必须是 0.0 到 1.0 之间的浮点数（归一化坐标，不是像素坐标）",
        "x 表示水平位置（0.0 = 左边缘，1.0 = 右边缘）",
        "y 表示垂直位置（0.0 = 上边缘，1.0 = 下边缘）",
        "x 和 y 的键名必须使用双引号",
        "不要输出数组格式，如 [0.5, 0.3]",
        "不要输出像素坐标，如 {\"x\": 540, \"y\": 390}",
        "只输出 JSON 对象，不要解释，不要使用 markdown",
    ])
    return (
        "你是一个 GUI 自动化智能体。请根据给定的手机截图和用户指令，"
        "预测应点击位置的归一化坐标。\n\n"
        f"任务：{instruction}\n\n"
        f"{rules}\n\n"
        "输出格式（严格 JSON，每行一个字段）：\n"
        "{\n"
        '    \"x\": 0.52,\n'
        '    \"y\": 0.31\n'
        "}\n"
    )


def build_m1_prompt(instruction: str, context_text: str) -> str:
    """M1：语义文本 + 单步指令 → click_id。"""
    rules = _critical_rules([
        "输出必须是合法的 JSON 对象，且仅包含 click_id 一个字段",
        "click_id 必须是整数（0, 1, 2, ...）",
        "click_id 的键名必须使用双引号",
        "不要输出数组或额外字段",
        "不要用 markdown 代码块包裹 JSON",
        "只输出 JSON 对象，不要解释，不要使用 markdown",
    ])
    return (
        "你是一个 GUI 自动化智能体。根据可交互 UI 元素的语义描述列表和用户指令，"
        "判断应点击哪个元素。\n\n"
        f"{context_text}\n\n"
        f"任务：{instruction}\n\n"
        f"{rules}\n\n"
        "输出格式（严格 JSON，每行一个字段）：\n"
        "{\n"
        '    \"click_id\": 1\n'
        "}\n"
    )


_ANNOTATED_SCREENSHOT_DESC = (
    "截图中的可交互 UI 元素已用带编号的标签（#0、#1、#2 等）标注在彩色半透明框中，"
    "每个标签位于对应元素的左上角。\n\n"
)


def build_m2_prompt(instruction: str) -> str:
    """M2：标注截图 + 单步指令 → click_id。"""
    rules = _critical_rules([
        "输出必须是合法的 JSON 对象，且仅包含 click_id 一个字段",
        "click_id 必须是整数（0, 1, 2, ...），与截图中标签编号一致",
        "click_id 的键名必须使用双引号",
        "不要输出数组或额外字段",
        "不要用 markdown 代码块包裹 JSON",
        "只输出 JSON 对象，不要解释，不要使用 markdown",
    ])
    return (
        "你是一个 GUI 自动化智能体。根据带编号标注的手机截图和用户指令，"
        "判断应点击哪个编号对应的元素。\n\n"
        f"{_ANNOTATED_SCREENSHOT_DESC}"
        f"任务：{instruction}\n\n"
        f"{rules}\n\n"
        "输出格式（严格 JSON，每行一个字段）：\n"
        "{\n"
        '    \"click_id\": 3\n'
        "}\n"
    )
