#!/usr/bin/env python3
"""
压缩后可交互节点为 0 的样本：将四件套移入 Mobile3M_data/_deleted/，并清理衍生产物。
"""

from __future__ import annotations

import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from config.data_paths import BASE_DIR
DELETED_DIR = BASE_DIR / "_deleted"

# (相对 BASE_DIR 的子目录, 扩展名)
FOUR_PIECE_FILES = [
    (BASE_DIR / "compressed_a11y", ".json"),
    (BASE_DIR / "a11y_trees_L0", ".json"),
    (BASE_DIR / "a11y_trees_L0", ".xml"),
    (BASE_DIR / "screenshots", ".png"),
    (BASE_DIR / "step_GT", ".json"),
    (BASE_DIR / "step_instructions", ".txt"),
]

DERIVED_FILES = [
    BASE_DIR / "nodes",
    BASE_DIR / "embeddings" / "cos_sim",
    PROJECT_ROOT / "target" / "target_object",
    PROJECT_ROOT / "target" / "TO_index",
]

DERIVED_DIRS = [
    BASE_DIR / "embeddings" / "nodes_emb",
    BASE_DIR / "embeddings" / "TO_emb",
]


def _move_file(src: Path, dst_dir: Path) -> bool:
    if not src.is_file():
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.is_file():
        dst.unlink()
    shutil.move(str(src), str(dst))
    return True


def remove_derived_artifacts(stem: str) -> list[str]:
    """删除 nodes / target_object / TO_index / embeddings 等衍生文件。"""
    removed: list[str] = []
    for parent in DERIVED_FILES:
        if parent.name == "cos_sim":
            path = parent / f"{stem}.npz"
        else:
            path = parent / f"{stem}.json"
        if path.is_file():
            path.unlink()
            removed.append(str(path.relative_to(PROJECT_ROOT)))

    for parent in DERIVED_DIRS:
        path = parent / stem
        if path.is_dir():
            shutil.rmtree(path)
            removed.append(str(path.relative_to(PROJECT_ROOT)))

    return removed


def move_stem_to_deleted(stem: str) -> list[str]:
    """
    将四件套移入 _deleted/{subdir}/，并清理衍生产物。
    返回已移动的文件路径（相对项目根）。
    """
    moved: list[str] = []
    for src_dir, ext in FOUR_PIECE_FILES:
        src = src_dir / f"{stem}{ext}"
        dst_dir = DELETED_DIR / src_dir.name
        if _move_file(src, dst_dir):
            moved.append(str((dst_dir / src.name).relative_to(PROJECT_ROOT)))

    removed = remove_derived_artifacts(stem)
    return moved


def purge_stems_with_empty_nodes(nodes_dir: Path | None = None) -> list[str]:
    """扫描 nodes/*.json 为空列表的 stem，执行移入 _deleted。"""
    import json

    nodes_dir = nodes_dir or (BASE_DIR / "nodes")
    purged: list[str] = []
    if not nodes_dir.is_dir():
        return purged

    for path in sorted(nodes_dir.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            nodes = json.load(f)
        if isinstance(nodes, list) and len(nodes) == 0:
            move_stem_to_deleted(path.stem)
            purged.append(path.stem)
    return purged
