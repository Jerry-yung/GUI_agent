"""Shared task / SMAN context for multi-path runner."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from utils.mobile3m_io import find_graph_dir
from utils.paths import resolve_data_dir
from utils.sman_bridge import load_graph_indices
from utils.sman_setup import get_sman_utils


@dataclass
class TaskContext:
    final_page_name: str
    task_desc: str
    multi_task_desc: list[str]
    task_dir: str
    all_action_ids: dict[str, int]
    id_to_action: dict[int, str]
    current_page_actions: dict[str, list[int]]
    all_page_convert: dict[str, str]
    task_index: int = 0


def parse_task_descriptions(task_info: str) -> tuple[str, list[str]]:
    task_desc = task_info.split("\n")[-1]
    multi_task_desc = re.findall(r"^\d+\.\s(.*)", task_info, re.MULTILINE)
    return task_desc, multi_task_desc


def get_step_instruction(ctx: TaskContext, round_count: int) -> str:
    """round_count starts at 1; maps to multi_task_desc[round_count - 1]."""
    if ctx.multi_task_desc:
        idx = round_count - 1
        if idx < len(ctx.multi_task_desc):
            return ctx.multi_task_desc[idx]
        return ctx.multi_task_desc[-1]
    return ctx.task_desc


def get_multipath_step_instruction(ctx: TaskContext, current_page_name: str) -> str:
    """Select step instruction by current page depth (SMAN multipath convention).

    ``QQmusic0`` → index 0; ``QQmusic0_10`` → index 1; after ``back`` to ``QQmusic0``
    the instruction returns to index 0 automatically.
    """
    if not ctx.multi_task_desc:
        return ctx.task_desc
    task_count = current_page_name.count("_")
    if task_count < len(ctx.multi_task_desc) - 1:
        return ctx.multi_task_desc[task_count]
    return ctx.multi_task_desc[-1]


def gt_path_suffix(task_name: str) -> list[str]:
    return task_name.split("_")[1:]


def missing_page_in_chain(task_dir: str, task_page_name: str) -> str | None:
    """Return first page name along the task path with no {page}/{page}.json, or None."""
    sman = get_sman_utils()
    for page in sman.page_chain_from_task_name(task_page_name):
        node_path = os.path.join(task_dir, page, page + ".json")
        if not os.path.isfile(node_path):
            return page
    return None


def load_task_context(
    task: dict[str, Any],
    data_dir: str | os.PathLike,
    task_index: int = 0,
) -> TaskContext | None:
    sman = get_sman_utils()

    final_page_name = task["name"]
    task_text = str(task.get("task") or "").strip()
    task_desc, multi_task_desc = parse_task_descriptions(task_text)

    data_path = resolve_data_dir(data_dir)
    graph = find_graph_dir(data_path, final_page_name)
    if graph is None:
        sman.print_with_color(
            f"Skip task {final_page_name}: no graph dir under {data_path}",
            "yellow",
        )
        return None

    task_dir = str(graph)
    missing_page = missing_page_in_chain(task_dir, final_page_name)
    if missing_page is not None:
        sman.print_with_color(
            f"Skip task {final_page_name}: missing page {missing_page} in graph",
            "yellow",
        )
        return None

    all_action_ids, id_to_action, current_page_actions, all_page_convert = load_graph_indices(task_dir)

    return TaskContext(
        final_page_name=final_page_name,
        task_desc=task_desc,
        multi_task_desc=multi_task_desc,
        task_dir=task_dir,
        all_action_ids=all_action_ids,
        id_to_action=id_to_action,
        current_page_actions=current_page_actions,
        all_page_convert=all_page_convert,
        task_index=task_index,
    )
