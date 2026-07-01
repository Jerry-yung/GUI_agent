"""扫描 runs/ 并汇总 type_acc、step_acc、SR，供对比.ipynb 使用。"""

from __future__ import annotations

import json
from pathlib import Path

from eval.run_naming import parse_run_filename, run_label
from eval.step_judge import compute_retrieval_hit, is_evaluable_step, judge_step_match

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"


def load_episodes(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        return payload["episodes"]
    return []


def _step_correct(step: dict, *, agent: str = "m2", top_k: int | None = None) -> bool:
    if "step_correct" in step:
        return bool(step["step_correct"])
    verdict = judge_step_match(
        step.get("gt", {}),
        step.get("pred_action"),
        stem=step.get("stem", ""),
        has_annotated_nodes=bool(step.get("has_annotated_nodes", True)),
        agent=agent,
        top_k_nodes=step.get("top_k_nodes"),
        top_k=top_k,
    )
    return verdict["step_correct"]


def _step_retrieval_hit(step: dict, *, top_k: int | None = None) -> bool | None:
    if "retrieval_hit" in step:
        hit = step.get("retrieval_hit")
        return None if hit is None else bool(hit)
    return compute_retrieval_hit(
        step.get("gt", {}),
        step.get("top_k_nodes"),
        top_k=top_k,
    )


def _step_type_correct(step: dict, *, agent: str = "m2", top_k: int | None = None) -> bool:
    if "type_correct" in step:
        return bool(step["type_correct"])
    verdict = judge_step_match(
        step.get("gt", {}),
        step.get("pred_action"),
        stem=step.get("stem", ""),
        has_annotated_nodes=bool(step.get("has_annotated_nodes", True)),
        agent=agent,
        top_k_nodes=step.get("top_k_nodes"),
        top_k=top_k,
    )
    return verdict["type_correct"]


def episode_success(episode: dict, *, agent: str = "m2", top_k: int | None = None) -> bool:
    steps = [s for s in episode.get("steps", []) if is_evaluable_step(s)]
    if not steps:
        return False
    return all(_step_correct(s, agent=agent, top_k=top_k) for s in steps)


def summarize_run(path: Path) -> dict:
    episodes = load_episodes(path)
    meta = parse_run_meta(path)
    agent = str(meta["agent"]) if meta else "m2"
    top_k = meta.get("top_k") if meta else None
    steps: list[dict] = []
    for ep in episodes:
        steps.extend(s for s in ep.get("steps", []) if is_evaluable_step(s))

    total_steps = len(steps)
    type_ok = sum(_step_type_correct(s, agent=agent, top_k=top_k) for s in steps)
    step_ok = sum(_step_correct(s, agent=agent, top_k=top_k) for s in steps)
    retrieval_eligible = [
        hit
        for s in steps
        if (hit := _step_retrieval_hit(s, top_k=top_k)) is not None
    ]
    retrieval_total = len(retrieval_eligible)
    retrieval_ok = sum(1 for hit in retrieval_eligible if hit)
    eval_episodes = [
        ep
        for ep in episodes
        if any(is_evaluable_step(s) for s in ep.get("steps", []))
    ]
    total_eps = len(eval_episodes)
    sr_ok = sum(episode_success(ep, agent=agent, top_k=top_k) for ep in eval_episodes)

    return {
        "run_file": path.name,
        "episodes": total_eps,
        "steps": total_steps,
        "type_acc": type_ok / total_steps if total_steps else 0.0,
        "step_acc": step_ok / total_steps if total_steps else 0.0,
        "sr": sr_ok / total_eps if total_eps else 0.0,
        "type_correct": type_ok,
        "step_correct": step_ok,
        "sr_correct": sr_ok,
        "retrieval_eligible": retrieval_total,
        "retrieval_correct": retrieval_ok,
        "topk_retrieval": retrieval_ok / retrieval_total if retrieval_total else None,
    }


def parse_run_meta(path: Path) -> dict | None:
    return parse_run_filename(path.stem)


def _matches_to_select(meta: dict, to_select: str | list[str] | None) -> bool:
    if to_select is None:
        return True
    if str(meta.get("agent", "")).upper() == "CPM":
        return True
    value = meta.get("to_select")
    if isinstance(to_select, str):
        return value == to_select
    return value in to_select


def collect_comparison(
    ac_mode: str,
    model: str,
    runs_dir: Path | None = None,
    *,
    to_select: str | list[str] | None = None,
) -> list[dict]:
    runs_dir = runs_dir or RUNS_DIR
    rows: list[dict] = []
    for path in sorted(runs_dir.glob("*.json")):
        meta = parse_run_meta(path)
        if meta is None:
            continue
        if meta["ac_mode"] != ac_mode or meta["vlm_model"] != model:
            continue
        if not _matches_to_select(meta, to_select):
            continue
        summary = summarize_run(path)
        label = run_label(meta)
        rows.append(
            {
                "label": label,
                "agent": meta["agent"],
                "top_k": meta.get("top_k"),
                "to_select": meta.get("to_select"),
                **summary,
            }
        )
    rows.sort(key=lambda r: (r["agent"], r.get("top_k") or 0, r.get("to_select") or ""))
    return rows
