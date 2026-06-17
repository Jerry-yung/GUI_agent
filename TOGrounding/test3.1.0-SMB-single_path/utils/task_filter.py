"""Filter Mobile3M task list by APP name and index range."""
from __future__ import annotations

from typing import Any


def app_prefix(task_name: str) -> str:
    """QQmusic0_34_347 -> QQmusic"""
    if "0" in task_name:
        return task_name.split("0")[0]
    return task_name.split("_")[0]


def filter_tasks(
    tasks: list[dict[str, Any]],
    app_names: list[str],
    start: int = 0,
    end: int = -1,
) -> list[tuple[int, dict[str, Any]]]:
    """Return (original_index, task) pairs after APP filter and slice."""
    allowed = set(app_names)
    filtered: list[tuple[int, dict[str, Any]]] = []
    for i, task in enumerate(tasks):
        name = task.get("name", "")
        if app_prefix(name) not in allowed:
            continue
        filtered.append((i, task))

    sliced = filtered[start:] if end < 0 else filtered[start:end]
    return sliced


def matches_app_names(task_name: str, app_names: list[str]) -> bool:
    """Whether ``task_name`` belongs to one of ``app_names`` (same rule as ``filter_tasks``)."""
    return app_prefix(task_name) in set(app_names)
