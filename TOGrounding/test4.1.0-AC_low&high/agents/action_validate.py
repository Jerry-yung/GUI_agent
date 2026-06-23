"""AC VLM action JSON 校验。"""

from __future__ import annotations

import json
from pathlib import Path

from agents.prompts import AC_ACTION_TYPES, SCROLL_DIRECTIONS

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "ac_vlm_action.schema.json"
_BASE_SCHEMA: dict | None = None

POINTER_TYPES = frozenset({"click", "long_press", "long_click"})


def _load_schema() -> dict:
    global _BASE_SCHEMA
    if _BASE_SCHEMA is None:
        with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
            _BASE_SCHEMA = json.load(f)
    return _BASE_SCHEMA


def _normalize_action_type(action_type: str) -> str:
    action_type = str(action_type or "").strip()
    if action_type == "long_click":
        return "long_press"
    return action_type


def validate_ac_action(
    data: dict,
    *,
    has_annotated_nodes: bool = True,
    mode: str = "low",
    agent: str = "m2",
) -> tuple[bool, str | None]:
    """
    校验 VLM 解析后的 action dict。

    Returns:
        (ok, error_message)
    """
    if not isinstance(data, dict):
        return False, "output is not a JSON object"

    try:
        import jsonschema

        jsonschema.validate(data, _load_schema())
    except ImportError:
        pass
    except Exception as exc:
        return False, f"schema: {exc}"

    action_type = _normalize_action_type(data.get("action_type", ""))
    if action_type not in AC_ACTION_TYPES:
        return False, f"unknown action_type: {action_type!r}"

    agent_upper = agent.upper()

    if action_type in POINTER_TYPES:
        if has_annotated_nodes:
            if agent_upper == "TO":
                if any(k in data for k in ("node_id", "click_id", "x", "y")):
                    return False, "TO annotated mode must not output locator fields for pointer"
            elif agent_upper in ("M2", "M2V", "M12") and not (
                "node_id" in data or "click_id" in data
            ):
                return False, "m2/m12 annotated mode requires node_id for pointer"
        else:
            if agent_upper == "CPM" and any(k in data for k in ("node_id", "click_id", "POINT")):
                return False, "CPM must use normalized x,y, not node_id or POINT"
            if "x" not in data or "y" not in data:
                return False, "fallback mode requires x,y for pointer"
            try:
                x, y = float(data["x"]), float(data["y"])
            except (TypeError, ValueError):
                return False, "x,y must be floats"
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                return False, "x,y must be in [0,1]"

    if action_type == "scroll":
        direction = str(data.get("direction", "")).strip().lower()
        if direction not in SCROLL_DIRECTIONS:
            return False, f"scroll requires direction in {SCROLL_DIRECTIONS}"

    if action_type == "input_text":
        if "text" not in data:
            return False, "input_text requires text"
        if agent_upper == "TO" and "node_id" in data:
            return False, "TO input_text must not include node_id"

    if mode.lower() == "high":
        plan_ok, plan_err = _validate_high_planning_fields(data)
        if not plan_ok:
            return False, plan_err

    return True, None


def _validate_high_planning_fields(data: dict) -> tuple[bool, str | None]:
    next_action_type = data.get("next_action_type")
    if next_action_type is None or str(next_action_type).strip() == "":
        return True, None
    normalized = _normalize_action_type(str(next_action_type))
    if normalized not in AC_ACTION_TYPES:
        return False, f"unknown next_action_type: {normalized!r}"
    return True, None


def validate_vlm_fields(
    data: dict,
    *,
    fixed_action_type: str,
    has_annotated_nodes: bool = True,
    mode: str = "low",
    agent: str = "m2",
) -> tuple[bool, str | None]:
    """
    AC-low：VLM 不输出 action_type，由 llm_TO 固定。
    仅校验该 type 下 VLM 应提供的字段。
    """
    if not isinstance(data, dict):
        return False, "output is not a JSON object"

    action_type = _normalize_action_type(fixed_action_type)
    if action_type not in AC_ACTION_TYPES:
        return False, f"unknown fixed_action_type: {action_type!r}"

    merged = dict(data)
    merged["action_type"] = action_type
    ok, err = validate_ac_action(
        merged,
        has_annotated_nodes=has_annotated_nodes,
        mode=mode,
        agent=agent,
    )
    if not ok:
        return ok, err
    if mode.lower() == "high":
        return _validate_high_planning_fields(data)
    return True, None
