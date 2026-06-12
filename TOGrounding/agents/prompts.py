"""四个 Agent 统一的 prompt 构建方法。"""


def _critical_rules(rules: list[str]) -> str:
    lines = "\n".join(f"{i}. {rule}" for i, rule in enumerate(rules, 1))
    return f"CRITICAL RULES:\n{lines}"


def build_baseline_prompt(instruction: str) -> str:
    """Baseline：原始截图 + 单步指令 → 归一化坐标。"""
    rules = _critical_rules([
        "Output must be a valid JSON object with EXACTLY two fields: x and y",
        "x and y must be FLOAT numbers between 0.0 and 1.0 (NOT pixel coordinates)",
        "x = horizontal ratio (0.0 = left edge, 1.0 = right edge)",
        "y = vertical ratio (0.0 = top edge, 1.0 = bottom edge)",
        "Both x and y keys must have double quotes",
        "Do NOT output arrays like [0.5, 0.3]",
        "Do NOT output pixel coordinates like {\"x\": 540, \"y\": 390}",
        "Only output the JSON object, no explanation, no markdown",
    ])
    return (
        "You are a GUI automation agent. Your task is to predict the click location "
        "on the given mobile screenshot to accomplish the user's instruction.\n\n"
        f"Task: {instruction}\n\n"
        f"{rules}\n\n"
        "Output format (STRICT JSON, one field per line):\n"
        "{\n"
        '    \"x\": 0.52,\n'
        '    \"y\": 0.31\n'
        "}\n"
    )


def build_m1_prompt(instruction: str, context_text: str) -> str:
    """M1：语义文本 + 单步指令 → click_id。"""
    rules = _critical_rules([
        "Output must be a valid JSON object with EXACTLY one field: click_id",
        "click_id must be an INTEGER (0, 1, 2, ...)",
        "The click_id key must use double quotes",
        "Do NOT output arrays or extra fields",
        "Do NOT wrap the JSON in markdown code blocks",
        "Only output the JSON object, no explanation, no markdown",
    ])
    return (
        "You are a GUI automation agent. Given a list of interactive UI elements "
        "with their semantic descriptions and a task instruction, "
        "determine which element to click.\n\n"
        f"{context_text}\n\n"
        f"Task: {instruction}\n\n"
        f"{rules}\n\n"
        "Output format (STRICT JSON, one field per line):\n"
        "{\n"
        '    \"click_id\": 1\n'
        "}\n"
    )


_ANNOTATED_SCREENSHOT_DESC = (
    "The screenshot shows interactive UI elements annotated with numbered tags "
    "(#0, #1, #2, etc.) in colored semi-transparent boxes. "
    "Each tag is placed at the top-left corner of its corresponding element.\n\n"
)


def build_m2_prompt(instruction: str) -> str:
    """M2：标注截图 + 单步指令 → click_id。"""
    rules = _critical_rules([
        "Output must be a valid JSON object with EXACTLY one field: click_id",
        "click_id must be an INTEGER (0, 1, 2, ...) matching the tag number in the screenshot",
        "The click_id key must use double quotes",
        "Do NOT output arrays or extra fields",
        "Do NOT wrap the JSON in markdown code blocks",
        "Only output the JSON object, no explanation, no markdown",
    ])
    return (
        "You are a GUI automation agent. Given an annotated mobile screenshot and a task instruction, "
        "determine which numbered element to click.\n\n"
        f"{_ANNOTATED_SCREENSHOT_DESC}"
        f"Task: {instruction}\n\n"
        f"{rules}\n\n"
        "Output format (STRICT JSON, one field per line):\n"
        "{\n"
        '    \"click_id\": 3\n'
        "}\n"
    )
