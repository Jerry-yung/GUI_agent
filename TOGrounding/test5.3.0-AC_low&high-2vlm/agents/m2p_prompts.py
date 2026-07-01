"""AC-high M2P planner prompts (vlm_TO on raw screenshot + history)."""

from __future__ import annotations

from dataclasses import dataclass

from agents.prompts import (
    AC_ACTION_TYPES,
    _scroll_gesture_direction_rules,
    _target_object_rules,
)

_PLANNER_SUMMARY_HINT = (
    "one short sentence: current visible UI state and what should happen this step"
)

_STUCK_PAGE_RECOVERY = """
SAME-PAGE STALL (compare step history with the CURRENT screenshot):
- "Same page" = 2+ recent summaries describe the same screen/dialog with no progress toward the goal.
- Stall signals: repeated action type or target, similar step_summary wording, or taps with no UI change.
- Do NOT repeat the action that just failed.

Recovery ladder:
1) First stall on this page: plan a clearly DIFFERENT visible control or scroll direction.
2) Still stuck on the same page for 2+ steps: set planned_action_type=navigate_back.
- Unexpected wrong page (permission sheet, search overlay): prefer planned_action_type=navigate_back immediately.
"""

_WAIT_NAV_PLANNER_RULES = """
WAIT / NAVIGATION (planner chooses planned_action_type):
- wait: screen/app still loading, or the user must pause before the next interaction.
- navigate_back: go back / return to previous page via system Back — NOT tap a "Back" label.
- navigate_home: return to the home screen / launcher.
- Do NOT plan click when the screen is still loading — use wait instead.
- Do NOT plan click to reach off-screen list items — plan scroll first, then click on a later step.
- For wait / navigate_back / navigate_home: no extra fields beyond thought are required.
"""

_NON_POINTER_ACTION_RULES = """
NON-POINTER ACTION FIELDS (planner outputs the full action — no separate executor VLM):
- scroll: MUST output direction ("up"|"down"|"left"|"right"). Omit when not scroll.
- input_text: MUST output text (string to type). Omit when not input_text.
- Set text to the string to type.
- wait / navigate_back / navigate_home: only thought + planned_action_type; omit direction and text.
"""

_PLANNER_RULES = f"""
PLANNER RULES (AC-high step replay — plan THIS step only; you do NOT output node_id or coordinates):
- AC-high is a step-by-step replay benchmark: every step MUST plan one concrete action.
  Do NOT output task completion, confirmation-only, or "no further action" semantics.
- planned_action_type: the AC action type for THIS step (one of: {", ".join(AC_ACTION_TYPES)}).
  Must match step_instruction semantics.
- click / long_press: MUST set target_object to a short English UI label for Top-K retrieval.
- scroll, input_text, wait, navigate_back, navigate_home: target_object must be "".
- step_instruction: one short imperative sentence describing what happens THIS step.

{_WAIT_NAV_PLANNER_RULES}
{_NON_POINTER_ACTION_RULES}
{_scroll_gesture_direction_rules()}
{_STUCK_PAGE_RECOVERY}
- When stuck on the same page for 2+ steps, strongly prefer planned_action_type=navigate_back
  over repeating the same tap.

{_target_object_rules()}
"""

_M2P_PLANNER_SCHEMA = """
Schema:
{
  "thought": "1-3 concise sentences",
  "planned_action_type": "click"|"long_press"|"scroll"|"input_text"|"wait"|"navigate_back"|"navigate_home",
  "step_instruction": "short imperative for this step",
  "target_object": "English UI label for click/long_press retrieval, or empty",
  "direction": "up"|"down"|"left"|"right",
  "text": "",
  "step_summary": "{hint}"
}
""".replace("{hint}", _PLANNER_SUMMARY_HINT)

M2P_PLANNER_SYSTEM_PROMPT = f"""
Android GUI task planner (M2P / AC-high). You see the RAW (unannotated) screenshot.

You receive:
1. Task goal
2. Step history (prior summaries, actions, step instructions)
3. Current raw screenshot

You plan ONE step at a time. You do NOT terminate the episode or skip remaining steps.
For click/long_press you output target_object for retrieval (no node_id or coordinates).
For scroll, input_text, wait, navigate_back, navigate_home you output the COMPLETE action
fields for this step (direction, text, or type-only as applicable).

OUTPUT: Exactly ONE raw JSON object. No markdown, code fences, or text outside JSON.

{_M2P_PLANNER_SCHEMA}
{_PLANNER_RULES}
"""


@dataclass
class StepHistoryEntry:
    step_num: int
    step_summary: str = ""
    action_summary: str = ""
    step_instruction: str = ""


def _stuck_page_user_hint(history_len: int) -> str:
    if history_len < 2:
        return ""
    if history_len >= 3:
        return (
            "STALL CHECK: Compare earlier summaries with the current screenshot. "
            "If 3+ steps stayed on the same page with no goal progress, set "
            "planned_action_type=navigate_back or plan a different route — "
            "do not repeat the failed action."
        )
    return (
        "If the last 2 steps show the same page with no progress, plan a different route "
        "(alternate control, scroll, or planned_action_type=navigate_back)."
    )


def _format_history_lines(history: list[StepHistoryEntry]) -> list[str]:
    lines: list[str] = []
    if history:
        lines.append("## Earlier steps")
        for entry in history[-8:]:
            parts = [f"Step {entry.step_num}:"]
            if entry.step_summary:
                parts.append(f"summary: {entry.step_summary}")
            if entry.action_summary:
                parts.append(f"action: {entry.action_summary}")
            instr = (entry.step_instruction or "").strip()
            if instr:
                parts.append(f"step_instruction: {instr}")
            lines.append("  " + " · ".join(parts))
        lines.append("")
    else:
        lines.extend(["## Earlier steps", "  (none)", ""])
    return lines


def build_m2p_planner_prompt(
    goal: str,
    step_num: int,
    max_steps: int,
    history: list[StepHistoryEntry],
) -> str:
    lines = [
        f"Task Goal: {goal}",
        f"Current Step: {step_num} / {max_steps}",
        "",
    ]
    lines.extend(_format_history_lines(history))

    if history:
        last = history[-1]
        lines.append(f"## Last step (step {last.step_num})")
        if last.step_summary:
            lines.append(f"summary: {last.step_summary}")
        if last.action_summary:
            lines.append(f"action: {last.action_summary}")
        instr = (last.step_instruction or "").strip()
        if instr:
            lines.append(f"step_instruction: {instr}")
        lines.append("")

    lines.append(f"## Current screenshot (step {step_num}) — raw, unannotated")
    hint = _stuck_page_user_hint(len(history))
    if hint:
        lines.append(hint)
    lines.append("Plan this step: output the planner JSON.")
    return "\n".join(lines)
