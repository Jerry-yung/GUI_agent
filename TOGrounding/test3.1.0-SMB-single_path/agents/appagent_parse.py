"""AppAgent response parser (extracted from SMAN-Bench appagent/model.py)."""
from __future__ import annotations

import re

from utils.sman_setup import get_sman_utils


def _print_with_color(text: str, color: str) -> None:
    get_sman_utils().print_with_color(text, color)


def _extract_field(rsp: str, field_name: str) -> str | None:
    """Extract Observation / Thought / Action / Summary (multiline-safe)."""
    pattern = rf"{field_name}:\s*(.*?)(?=\n(?:Observation|Thought|Action|Summary):|\Z)"
    matches = re.findall(pattern, rsp, re.DOTALL | re.IGNORECASE)
    return matches[0].strip() if matches else None


def parse_explore_rsp(rsp: str) -> list[str]:
    try:
        observation = _extract_field(rsp, "Observation")
        think = _extract_field(rsp, "Thought")
        act = _extract_field(rsp, "Action")
        last_act = _extract_field(rsp, "Summary") or ""

        if not act:
            raise ValueError("missing Action field in model response")

        if observation:
            _print_with_color("Observation:", "yellow")
            _print_with_color(observation, "magenta")
        if think:
            _print_with_color("Thought:", "yellow")
            _print_with_color(think, "magenta")
        _print_with_color("Action:", "yellow")
        _print_with_color(act, "magenta")
        if last_act:
            _print_with_color("Summary:", "yellow")
            _print_with_color(last_act, "magenta")

        act = act.strip()
        act_name = act.split("(")[0].strip()
        if act_name == "click":
            click_match = re.findall(r"click\((.*?)\)", act, re.DOTALL)
            if not click_match:
                raise ValueError(f"cannot parse click action: {act}")
            area = str(click_match[0]).strip()
            return [act_name, area, last_act]
        if act_name == "input":
            input_match = re.findall(r"input\((.*?)\)", act, re.DOTALL)
            if not input_match:
                raise ValueError(f"cannot parse input action: {act}")
            input_str = input_match[0].strip().strip('"').strip("'")
            return [act_name, input_str, last_act]
        if act_name == "scroll":
            scroll_match = re.findall(r"scroll\((.*?)\)", act, re.DOTALL)
            if not scroll_match:
                raise ValueError(f"cannot parse scroll action: {act}")
            params = scroll_match[0]
            if "," not in params:
                raise ValueError(f"cannot parse scroll params: {act}")
            area, scroll_direction = params.split(",", 1)
            area = str(area.strip())
            scroll_direction = scroll_direction.strip().strip('"').strip("'")
            return [act_name, area, scroll_direction, last_act]
        _print_with_color(f"ERROR: Undefined act {act_name}!", "red")
        return ["ERROR"]
    except Exception as exc:
        _print_with_color(
            f"ERROR: an exception occurs while parsing the model response: {exc}",
            "red",
        )
        _print_with_color(rsp, "red")
        return ["ERROR"]
