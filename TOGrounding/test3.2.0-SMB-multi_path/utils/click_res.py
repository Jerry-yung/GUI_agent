"""Click action list helpers (no agents import — safe for step_log / eval)."""
from __future__ import annotations


def format_click_xy_res(norm_x: float, norm_y: float, summary: str) -> list[str]:
    return ["click", "xy", f"{norm_x:.6f}", f"{norm_y:.6f}", summary]


def is_click_xy_res(res: list[str] | None) -> bool:
    return bool(
        res
        and len(res) >= 5
        and res[0] == "click"
        and str(res[1]).lower() == "xy"
    )


def click_xy_from_res(res: list[str]) -> tuple[float, float] | None:
    if not is_click_xy_res(res):
        return None
    try:
        return float(res[2]), float(res[3])
    except (TypeError, ValueError, IndexError):
        return None
