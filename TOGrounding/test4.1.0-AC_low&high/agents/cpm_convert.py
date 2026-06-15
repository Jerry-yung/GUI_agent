"""CPM VLM JSON 解析与 AC action 转换（归一化 x,y，对齐 m2 baseline）。"""

from __future__ import annotations

import json
from pathlib import Path

from agents.action_validate import validate_ac_action
from agents.m2_agent import _normalize_action
from agents.parse_utils import parse_vlm_response

_SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "cpm_action.schema.json"
_BASE_SCHEMA: dict | None = None


def _load_schema() -> dict:
    global _BASE_SCHEMA
    if _BASE_SCHEMA is None:
        with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
            _BASE_SCHEMA = json.load(f)
    return _BASE_SCHEMA


def validate_cpm_action(data: dict) -> tuple[bool, str | None]:
    if not isinstance(data, dict):
        return False, "output is not a JSON object"
    try:
        import jsonschema

        jsonschema.validate(data, _load_schema())
    except ImportError:
        pass
    except Exception as exc:
        return False, f"schema: {exc}"
    return True, None


def cpm_to_ac(data: dict, *, mode: str = "low") -> dict | None:
    """将 CPM VLM 输出的 AC 风格 JSON 转为 pred_action。"""
    if not isinstance(data, dict):
        return None
    action_type = str(data.get("action_type", "")).strip()
    if not action_type:
        return None
    return _normalize_action(data, mode, has_annotated_nodes=False)


def parse_and_convert_cpm(
    raw: str,
    *,
    mode: str = "low",
) -> tuple[dict | None, dict | None, str | None]:
    """
    解析 VLM 原文并转为 AC action。

    Returns:
        (cpm_action, ac_action, schema_error)
    """
    parsed = parse_vlm_response(raw)
    if parsed is None:
        return None, None, None

    schema_ok, schema_error = validate_cpm_action(parsed)
    ac_ok, ac_error = validate_ac_action(
        parsed,
        has_annotated_nodes=False,
        mode=mode,
        agent="CPM",
    )
    ac_action = cpm_to_ac(parsed, mode=mode)
    if ac_action is None:
        return parsed, None, schema_error or ac_error or "unable to map CPM action to AC"

    if not schema_ok:
        return parsed, ac_action, schema_error
    if not ac_ok:
        return parsed, ac_action, ac_error
    return parsed, ac_action, None
