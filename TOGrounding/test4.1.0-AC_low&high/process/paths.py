"""4.1.0 AC_data 路径约定：steps/{episode_id}/{step_idx:03d}/{stem}_*"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
AC_DATA = PROJECT_ROOT / "AC_data"
STEPS_DIR = AC_DATA / "steps"
EPISODES_DIR = AC_DATA / "episodes"
MANIFEST_PATH = AC_DATA / "manifest.json"


def stem_name(episode_id: str, step_idx: int) -> str:
    return f"{episode_id}_{step_idx:03d}"


def parse_stem(stem: str) -> tuple[str, int]:
    episode_id, step_str = stem.rsplit("_", 1)
    return episode_id, int(step_str)


def step_dir(episode_id: str, step_idx: int) -> Path:
    return STEPS_DIR / episode_id / f"{step_idx:03d}"


def step_dir_from_stem(stem: str) -> Path:
    episode_id, step_idx = parse_stem(stem)
    return step_dir(episode_id, step_idx)


def step_paths(stem: str) -> dict[str, Path]:
    d = step_dir_from_stem(stem)
    return {
        "dir": d,
        "gt": d / f"{stem}_gt.json",
        "instruction": d / f"{stem}_instruction.txt",
        "screenshot": d / f"{stem}_screenshot.png",
        "a11y": d / f"{stem}_a11y.json",
        "meta": d / f"{stem}_meta.json",
        "compressed_a11y": d / f"{stem}_compressed_a11y.json",
        "nodes": d / f"{stem}_nodes.json",
    }


NODES_EMB_DIR = AC_DATA / "embeddings" / "nodes_emb"


def node_emb_dir(stem: str) -> Path:
    episode_id, step_idx = parse_stem(stem)
    return NODES_EMB_DIR / episode_id / f"{step_idx:03d}"


def node_emb_legacy_dir(stem: str) -> Path:
    return NODES_EMB_DIR / stem


def node_emb_path(stem: str, node_id: int) -> Path:
    return node_emb_dir(stem) / f"{stem}_{node_id}.npy"


def node_emb_legacy_path(stem: str, node_id: int) -> Path:
    return node_emb_legacy_dir(stem) / f"{stem}_{node_id}.npy"


def iter_episode_ids() -> list[str]:
    if EPISODES_DIR.is_dir():
        episodes = sorted(p.stem for p in EPISODES_DIR.glob("*.json"))
        if episodes:
            return episodes
    return sorted({parse_stem(stem)[0] for stem in iter_stems()})


def stems_in_episode(episode_id: str) -> list[str]:
    episode_dir = STEPS_DIR / episode_id
    if not episode_dir.is_dir():
        return []
    stems: list[str] = []
    for step_path in sorted(episode_dir.iterdir()):
        if step_path.is_dir():
            stems.append(stem_name(episode_id, int(step_path.name)))
    return stems


def iter_stems() -> list[str]:
    if MANIFEST_PATH.is_file():
        with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        stems = data.get("stems", [])
        if stems:
            return stems

    stems: list[str] = []
    if not STEPS_DIR.is_dir():
        return stems

    for episode_dir in sorted(STEPS_DIR.iterdir()):
        if not episode_dir.is_dir():
            continue
        for step_path in sorted(episode_dir.iterdir()):
            if not step_path.is_dir():
                continue
            stems.append(stem_name(episode_dir.name, int(step_path.name)))
    return stems
