"""Read/write single-path result JSON (with SMAN-compatible result_raw_txt)."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def build_status_tail(task_complete: bool, round_count: int, max_rounds: int) -> str:
    if task_complete:
        return "Task completed successfully"
    if round_count >= max_rounds:
        return "Task finished due to reaching max rounds"
    return "Task failed"


def build_result_raw_txt(
    ans_action_id: list,
    ans_action_info: list,
    ans_history_pages: list,
    task_complete: bool,
    round_count: int,
    max_rounds: int,
) -> str:
    lines: list[str] = []
    for action_id, action_info, history_page in zip(
        ans_action_id, ans_action_info, ans_history_pages
    ):
        lines.append(f"{action_id}: {action_info}: {history_page}")
    lines.append(build_status_tail(task_complete, round_count, max_rounds))
    return "\n".join(lines) + "\n"


def write_result_json(
    result_dir: Path | str,
    final_page_name: str,
    task_index: int,
    *,
    ans_action_id: list,
    ans_action_info: list,
    ans_history_pages: list,
    task_complete: bool,
    round_count: int,
    max_rounds: int,
    steps: list[dict[str, Any]],
    task_description: str = "",
    gt_type_bucket: str | None = None,
) -> Path:
    result_path = Path(result_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    out_file = result_path / f"{task_index}_{final_page_name}.json"

    result_raw_txt = build_result_raw_txt(
        ans_action_id,
        ans_action_info,
        ans_history_pages,
        task_complete,
        round_count,
        max_rounds,
    )
    total_vlm_in = sum(int(s.get("vlm_input_tokens") or 0) for s in steps)
    total_vlm_out = sum(int(s.get("vlm_output_tokens") or 0) for s in steps)
    total_elapsed = round(sum(float(s.get("elapsed_ms") or 0) for s in steps), 2)
    total_vlm_elapsed = round(sum(float(s.get("vlm_elapsed_ms") or 0) for s in steps), 2)

    payload: dict[str, Any] = {
        "task_name": final_page_name,
        "task_description": task_description,
        "task_index": task_index,
        "result_raw_txt": result_raw_txt,
        "task_complete": task_complete,
        "round_count": round_count,
        "max_rounds": max_rounds,
        "status_tail": build_status_tail(task_complete, round_count, max_rounds),
        "steps": steps,
        "totals": {
            "step_count": len(steps),
            "elapsed_ms": total_elapsed,
            "vlm_elapsed_ms": total_vlm_elapsed,
            "vlm_input_tokens": total_vlm_in,
            "vlm_output_tokens": total_vlm_out,
        },
    }
    if gt_type_bucket:
        payload["gt_type_bucket"] = gt_type_bucket
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_file


def write_typed_result_slices(
    base_result_dir: Path | str,
    final_page_name: str,
    task_index: int,
    *,
    write_types: list[str],
    steps: list[dict[str, Any]],
    ans_action_id: list,
    ans_action_info: list,
    ans_history_pages: list,
    task_complete: bool,
    round_count: int,
    max_rounds: int,
    task_description: str = "",
) -> list[Path]:
    """Write one JSON per GT action type under ``base_result_dir/{type}/``."""
    base = Path(base_result_dir)
    written: list[Path] = []
    for gt_type in write_types:
        typed_steps: list[dict[str, Any]] = []
        typed_aids: list = []
        typed_ainfos: list = []
        typed_ahists: list = []
        for i, step in enumerate(steps):
            if str(step.get("gt_action_type") or "") != gt_type:
                continue
            typed_steps.append(step)
            typed_aids.append(ans_action_id[i])
            typed_ainfos.append(ans_action_info[i])
            typed_ahists.append(ans_history_pages[i])

        if not typed_steps:
            continue

        typed_complete = all(
            bool(s.get("action_match"))
            for s in typed_steps
            if not s.get("is_skip")
        )
        out = write_result_json(
            base / gt_type,
            final_page_name,
            task_index,
            ans_action_id=typed_aids,
            ans_action_info=typed_ainfos,
            ans_history_pages=typed_ahists,
            task_complete=typed_complete,
            round_count=round_count,
            max_rounds=max_rounds,
            steps=typed_steps,
            task_description=task_description,
            gt_type_bucket=gt_type,
        )
        written.append(out)
    return written


def list_result_files(result_dir: str | Path) -> list[str]:
    p = Path(result_dir)
    stems: dict[str, str] = {}
    for name in os.listdir(p):
        if name.endswith(".json"):
            stems[name[: -len(".json")]] = name
        elif name.endswith(".txt") and name[: -len(".txt")] not in stems:
            stems[name[: -len(".txt")]] = name
    return sorted(stems.values())


def load_result_payload(result_dir: str | Path, filename: str) -> dict[str, Any]:
    path = Path(result_dir) / filename
    if filename.endswith(".json"):
        return json.loads(path.read_text(encoding="utf-8"))
    raw = path.read_text(encoding="utf-8")
    return {"result_raw_txt": raw, "file": filename}


def load_result_raw_lines(result_dir: str | Path, filename: str) -> list[str]:
    payload = load_result_payload(result_dir, filename)
    raw = payload.get("result_raw_txt") or ""
    return raw.splitlines(keepends=True)
