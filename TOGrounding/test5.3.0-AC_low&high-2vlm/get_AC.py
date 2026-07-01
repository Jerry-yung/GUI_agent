#!/usr/bin/env python3
"""
将 Android Control episode 级数据拆分为单步样本，输出 4.1.0 目录结构：

    AC_data/steps/{episode_id}/{step_idx:03d}/{stem}_gt.json
    AC_data/steps/{episode_id}/{step_idx:03d}/{stem}_instruction.txt
    AC_data/steps/{episode_id}/{step_idx:03d}/{stem}_screenshot.png
    AC_data/steps/{episode_id}/{step_idx:03d}/{stem}_a11y.json
    AC_data/steps/{episode_id}/{step_idx:03d}/{stem}_meta.json

其中 stem = {episode_id}_{step_idx:03d}，保留全部 action_type，不做筛选。

数据来源：datasets/AndroidControl/data/（GT、instructions、screenshots、a11y_trees_L0）
输出目录：GUI_agent/test4.1.0-AC_high/AC_data/
"""

from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

SCHEMA_VERSION = "4.1.0"

PROJECT_ROOT = Path(__file__).resolve().parent
GUI_ROOT = PROJECT_ROOT.parent.parent  # .../GUI

SOURCE_DIR = GUI_ROOT / "datasets" / "AndroidControl" / "data"
GT_DIR = SOURCE_DIR / "GT"
INS_DIR = SOURCE_DIR / "instructions"
SCREENSHOTS_DIR = SOURCE_DIR / "screenshots"
A11Y_DIR = SOURCE_DIR / "a11y_trees_L0"

OUTPUT_DIR = PROJECT_ROOT / "AC_data"
STEPS_DIR = OUTPUT_DIR / "steps"
EPISODES_DIR = OUTPUT_DIR / "episodes"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


def stem_name(episode_id: str, step_idx: int) -> str:
    return f"{episode_id}_{step_idx:03d}"


def step_dir(episode_id: str, step_idx: int) -> Path:
    return STEPS_DIR / episode_id / f"{step_idx:03d}"


def step_paths(episode_id: str, step_idx: int) -> dict[str, Path]:
    stem = stem_name(episode_id, step_idx)
    d = step_dir(episode_id, step_idx)
    return {
        "dir": d,
        "gt": d / f"{stem}_gt.json",
        "instruction": d / f"{stem}_instruction.txt",
        "screenshot": d / f"{stem}_screenshot.png",
        "a11y": d / f"{stem}_a11y.json",
        "meta": d / f"{stem}_meta.json",
    }


def clear_output_dirs() -> None:
    if STEPS_DIR.exists():
        shutil.rmtree(STEPS_DIR)
    if EPISODES_DIR.exists():
        shutil.rmtree(EPISODES_DIR)
    STEPS_DIR.mkdir(parents=True, exist_ok=True)
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)


def screenshot_size(gt_data: dict, step_idx: int) -> dict[str, int] | None:
    widths = gt_data.get("screenshot_widths") or []
    heights = gt_data.get("screenshot_heights") or []
    if step_idx < len(widths) and step_idx < len(heights):
        return {"width": widths[step_idx], "height": heights[step_idx]}
    screenshots = gt_data.get("screenshots") or []
    if step_idx < len(screenshots):
        item = screenshots[step_idx]
        if "width" in item and "height" in item:
            return {"width": item["width"], "height": item["height"]}
    return None


def copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def process_episode(gt_file: Path, ins_file: Path | None) -> dict | None:
    episode_id = gt_file.stem

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    ins_data: dict = {}
    if ins_file and ins_file.exists():
        with open(ins_file, "r", encoding="utf-8") as f:
            ins_data = json.load(f)

    actions = gt_data.get("actions", [])
    step_instructions = ins_data.get("step_instructions", [])
    goal = ins_data.get("goal") or gt_data.get("goal", "")
    num_steps = len(actions)

    if step_instructions and len(step_instructions) != num_steps:
        print(
            f"  ⚠️ {episode_id}: actions({num_steps}) 与 "
            f"step_instructions({len(step_instructions)}) 数量不一致"
        )

    episode_record = {
        "episode_id": episode_id,
        "goal": goal,
        "num_steps": num_steps,
        "stems": [],
        "action_types": [],
    }

    missing_assets: Counter[str] = Counter()

    for step_idx, action in enumerate(actions):
        paths = step_paths(episode_id, step_idx)
        stem = stem_name(episode_id, step_idx)
        paths["dir"].mkdir(parents=True, exist_ok=True)

        with open(paths["gt"], "w", encoding="utf-8") as f:
            json.dump(action, f, ensure_ascii=False, indent=2)

        instruction = ""
        if step_idx < len(step_instructions):
            instruction = step_instructions[step_idx]
        with open(paths["instruction"], "w", encoding="utf-8") as f:
            f.write(instruction)

        if not copy_if_exists(SCREENSHOTS_DIR / f"{stem}.png", paths["screenshot"]):
            missing_assets["screenshot"] += 1
        if not copy_if_exists(A11Y_DIR / f"{stem}.json", paths["a11y"]):
            missing_assets["a11y"] += 1

        meta = {
            "schema_version": SCHEMA_VERSION,
            "stem": stem,
            "episode_id": episode_id,
            "step_idx": step_idx,
            "action_type": action.get("action_type"),
            "episode_goal": goal,
            "screenshot_size": screenshot_size(gt_data, step_idx),
            "num_steps_in_episode": num_steps,
        }
        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        episode_record["stems"].append(stem)
        episode_record["action_types"].append(action.get("action_type"))

    episode_path = EPISODES_DIR / f"{episode_id}.json"
    with open(episode_path, "w", encoding="utf-8") as f:
        json.dump(episode_record, f, ensure_ascii=False, indent=2)

    if missing_assets:
        parts = ", ".join(f"{k}={v}" for k, v in missing_assets.items())
        print(f"  ⚠️ {episode_id}: 缺失资源 ({parts})")

    return episode_record


def main() -> None:
    print("=" * 50)
    print("4.1.0 get_AC: episode → steps/{episode_id}/{step_idx}/")
    print("=" * 50)

    if not SOURCE_DIR.is_dir():
        raise SystemExit(f"数据源目录不存在: {SOURCE_DIR}")

    gt_files = sorted(GT_DIR.glob("*.json"))
    if not gt_files:
        raise SystemExit(f"未找到 GT 文件: {GT_DIR}")

    print(f"数据源:   {SOURCE_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"发现 {len(gt_files)} 个 episode")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clear_output_dirs()

    all_stems: list[str] = []
    action_type_counts: Counter[str] = Counter()
    total_steps = 0

    for i, gt_file in enumerate(gt_files, 1):
        ins_file = INS_DIR / f"{gt_file.stem}.json"
        record = process_episode(gt_file, ins_file)
        if record is None:
            continue
        total_steps += record["num_steps"]
        all_stems.extend(record["stems"])
        action_type_counts.update(record["action_types"])

        if i % 100 == 0 or i == len(gt_files):
            print(f"  已处理 {i}/{len(gt_files)} episodes, {total_steps} steps")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "layout": "steps/{episode_id}/{step_idx:03d}/{stem}_*",
        "total_episodes": len(gt_files),
        "total_steps": total_steps,
        "action_type_counts": dict(sorted(action_type_counts.items())),
        "stems": all_stems,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 50)
    print("完成")
    print("=" * 50)
    print(f"  episodes: {len(gt_files)}")
    print(f"  steps:    {total_steps}")
    print(f"  输出:     {STEPS_DIR}/")
    print(f"  索引:     {MANIFEST_PATH}")
    print("\naction_type 分布:")
    for action_type, count in action_type_counts.most_common():
        print(f"  {action_type}: {count}")


if __name__ == "__main__":
    main()
