"""Read-only Mobile3M data access."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def resolve_data_dir(data_dir: Path | str | None = None) -> Path:
    if data_dir is None:
        from config.data_paths import MOBILE3M_SRC

        return MOBILE3M_SRC
    p = Path(data_dir)
    return p.resolve() if p.is_absolute() else p.resolve()


def load_tasks(data_dir: Path | str, task_file: str) -> list[dict[str, Any]]:
    path = resolve_data_dir(data_dir) / task_file
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_graph_dir(data_dir: Path | str, task_name: str) -> Path | None:
    """Find first *_graph_* subdirectory matching the task app prefix."""
    root = resolve_data_dir(data_dir)
    if "0" in task_name:
        app_prefix = task_name.split("0")[0] + "_"
    else:
        app_prefix = task_name.split("_")[0] + "_"

    for name in sorted(os.listdir(root)):
        if not name.startswith(app_prefix):
            continue
        path = root / name
        if path.is_dir() and "_graph_" in name:
            return path
    return None


def page_paths(graph_dir: Path, page_name: str) -> dict[str, Path]:
    page_dir = graph_dir / page_name
    return {
        "dir": page_dir,
        "screenshot": page_dir / f"{page_name}-screen.png",
        "html": page_dir / f"{page_name}-html.txt",
        "xml": page_dir / f"{page_name}-xml.txt",
        "json": page_dir / f"{page_name}.json",
    }


def load_page_json(graph_dir: Path, page_name: str) -> dict[str, Any]:
    path = page_paths(graph_dir, page_name)["json"]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
