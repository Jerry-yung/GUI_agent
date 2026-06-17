"""Read-only Mobile3M data access."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from utils.paths import page_paths, resolve_data_dir
from utils.sman_setup import get_sman_utils


def load_tasks(data_dir: Path | str, task_file: str) -> list[dict[str, Any]]:
    path = resolve_data_dir(data_dir) / task_file
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_graph_dir(data_dir: Path, task_name: str) -> Path | None:
    sman = get_sman_utils()

    app_prefix = task_name.split("0")[0] + "_" if "0" in task_name else task_name.split("_")[0] + "_"
    name = sman.find_dir_with_prefix(str(data_dir), app_prefix)
    if not name:
        return None
    graph = data_dir / name
    return graph if graph.is_dir() else None


def read_page_files(graph_dir: Path, page_name: str) -> dict[str, str | Path]:
    paths = page_paths(graph_dir, page_name)
    out: dict[str, str | Path] = {"paths": paths}
    for key in ("screenshot", "html", "xml"):
        out[key] = paths[key]
    if paths["html"].is_file():
        out["html_content"] = paths["html"].read_text(encoding="utf-8")
    else:
        out["html_content"] = ""
    if paths["xml"].is_file():
        out["xml_content"] = paths["xml"].read_text(encoding="utf-8")
    else:
        out["xml_content"] = ""
    return out
