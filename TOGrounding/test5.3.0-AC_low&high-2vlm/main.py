#!/usr/bin/env python3
"""
AC-low / AC-high 实验入口（全步轨迹，按 episode 并行）。

每步流程（AC-low）：
  1. llm_TO → action_type + target_object
  2. click/long_press：target_object → top_k 标注；其他类型：原图
  3. VLM 填充 action 字段（node_id / direction / text 等）；type 由 llm_TO 固定

每步流程（AC-high TO）：
  每步 vlm_TO(原图 + history) → click/long_press：target_object → top-1 pred；其余 type：planner 直出

每步流程（AC-high m2 / 旧 TO 已废弃）：
  step0：llm_TO(goal) → action_type + target_object → 条件检索
  step n+1：上步 VLM 的 next_action_type + target_object + next_instruction → 条件检索
  VLM：固定本步 type（llm_TO 或上步 next_action_type）+ 填字段 + 输出下一步规划三字段
"""

from __future__ import annotations

import json
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.TO_agent import TOAgent
from agents.CPM_agent import CPMAgent
from agents.m2_agent import M2Agent
from agents.m2p_agent import (
    M2PAgent,
    M2P_EXECUTOR_TYPES,
    M2P_PLANNER_ONLY_TYPES,
    StepHistoryEntry,
)
from agents.prompts import AC_ACTION_TYPES
from annotate.annotate import annotate_step, save_original_annotated
from annotate.cos_sim_topk import rank_nodes_by_to
from annotate.node_filter import is_valid_top1_bounds
from annotate.llm_TO import POINTER_ACTION_TYPES, generate_target_object
from annotate.to_rank_select import pick_to_string_from_rank, TO_SELECT_CHOICES
from eval.run_naming import run_filename
from eval.step_judge import is_evaluable_step, is_skipped_gt_action, judge_step_match
from llm_set.llm import get_vlm_model_name
from process.paths import (
    EPISODES_DIR,
    stem_name,
    step_paths,
    iter_episode_ids,
    stems_in_episode,
)

# ============= 实验配置 =============

# AC_MODE = "low"
AC_MODE = "high"  # "low" | "high"

# AGENT = "CPM" # 不看 TOP_K, TO_SELECT 限制
# AGENT = "m2"
# AGENT = "m2p"  # AC-high only: vlm_TO + vlm_action
AGENT = "TO"

TOP_K = 1 # TO 时必须为 1
TO_SELECT = "generate"  # generate

TEST_START = 50
TEST_END = 100  # episode end, None 表示到最后一个 episode

# TEST_LIST: list[str] = [ "00000020", "00000040", "00000220", "00000240", "00000421",
#     "00000542", "00000682", "00000743", "00000763", "00000843",
#     "00000863", "00000883", "00001023", "00001084", "00001104"]  # scroll
TEST_LIST: list[str] = []  # 非空即替代 TEST_START/TEST_END 切片
# TEST_LIST = ["00001424", "00002284", "00002324"]


MAX_STEPS: int | None = None  # 每个 episode 最多跑几步，None 表示全部
PARALLEL_WORKERS = 25  # 最大并发 episode 数

_PRINT_LOCK = threading.Lock()
_RUNS_LOCK = threading.Lock()

# ===================================

_BOX_WIDTH = 72
_ANSI_GREEN = "\033[32m"
_ANSI_RED = "\033[31m"
_ANSI_YELLOW = "\033[33m"
_ANSI_RESET = "\033[0m"


def _use_color() -> bool:
    return sys.stdout.isatty()


def _clip(text: str, max_len: int = 96) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _load_episode_goal(episode_id: str) -> str:
    path = EPISODES_DIR / f"{episode_id}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return str(data.get("goal", "")).strip()


def _load_instruction(stem: str) -> str:
    path = step_paths(stem)["instruction"]
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _load_gt(stem: str) -> dict:
    path = step_paths(stem)["gt"]
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _select_episodes() -> list[str]:
    all_eps = iter_episode_ids()
    if TEST_LIST:
        return list(TEST_LIST)
    start = TEST_START if TEST_START is not None else 0
    end = TEST_END if TEST_END is not None else len(all_eps)
    return all_eps[start:end]


def _to_prompt_step0(episode_id: str, stem: str) -> str:
    """step0 标注前 llm_TO 输入。"""
    if AC_MODE == "low":
        text = _load_instruction(stem)
        if not text:
            raise ValueError(f"[{stem}] 缺少 step instruction")
        return text
    if AC_MODE == "high":
        text = _load_episode_goal(episode_id)
        if not text:
            raise ValueError(f"[{episode_id}] 缺少 episode goal")
        return text
    raise ValueError(f"未知 AC_MODE: {AC_MODE!r}")


def _parse_next_instruction(pred_action: dict | None) -> str | None:
    text = str((pred_action or {}).get("next_instruction", "")).strip()
    return text or None


def _parse_next_action_type(pred_action: dict | None) -> str:
    action_type = str((pred_action or {}).get("next_action_type", "")).strip()
    if action_type == "long_click":
        return "long_press"
    return action_type


def _to_prompt_after_step(
    episode_id: str,
    step_idx: int,
    pred_action: dict | None,
) -> str:
    """step n VLM 之后，为 step n+1 准备标注的 llm_TO 输入。"""
    if AC_MODE == "high":
        next_instr = _parse_next_instruction(pred_action)
        if next_instr:
            return next_instr
        return _load_episode_goal(episode_id)
    if AC_MODE == "low":
        next_stem = stem_name(episode_id, step_idx + 1)
        text = _load_instruction(next_stem)
        if not text:
            raise ValueError(f"[{next_stem}] 缺少 step instruction")
        return text
    raise ValueError(f"未知 AC_MODE: {AC_MODE!r}")


def _prev_step_instruction_for_vlm(
    stems: list[str],
    step_idx: int,
    *,
    prev_pred_action: dict | None,
) -> str | None:
    """low: 上一步 GT instruction；high: 上上步 VLM 的 next_instruction（即上一步所执行的指令）。"""
    if step_idx <= 0:
        return None
    if AC_MODE == "low":
        return _load_instruction(stems[step_idx - 1]) or None
    if AC_MODE == "high":
        if step_idx < 2 or not prev_pred_action:
            return None
        return _parse_next_instruction(prev_pred_action)
    raise ValueError(f"未知 AC_MODE: {AC_MODE!r}")


def _current_step_instruction_for_vlm(
    step_idx: int,
    *,
    prev_pred_action: dict | None,
) -> str | None:
    """high 模式：本步指令来自上一步 VLM 的 next_instruction。"""
    if AC_MODE != "high" or step_idx <= 0:
        return None
    return _parse_next_instruction(prev_pred_action)


def _vlm_context(episode_id: str, stem: str) -> dict[str, str]:
    goal = _load_episode_goal(episode_id)
    if not goal:
        raise ValueError(f"[{episode_id}] 缺少 episode goal")

    if AC_MODE == "low":
        instruction = _load_instruction(stem)
        if not instruction:
            raise ValueError(f"[{stem}] 缺少 step instruction")
        return {"instruction": instruction, "goal": goal}

    if AC_MODE == "high":
        return {"goal": goal}

    raise ValueError(f"未知 AC_MODE: {AC_MODE!r}")


def _step_has_nodes(stem: str) -> bool:
    nodes_path = step_paths(stem)["nodes"]
    if not nodes_path.is_file():
        return False
    with open(nodes_path, "r", encoding="utf-8") as f:
        nodes = json.load(f)
    return bool(nodes)


def _screenshot_size(stem: str) -> tuple[int, int]:
    meta_path = step_paths(stem)["meta"]
    if meta_path.is_file():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        size = meta.get("screenshot_size") or {}
        w, h = size.get("width"), size.get("height")
        if w and h:
            return int(w), int(h)

    screenshot_path = step_paths(stem)["screenshot"]
    if screenshot_path.is_file():
        from PIL import Image

        with Image.open(screenshot_path) as img:
            return img.size
    return 1080, 2400


def _fallback_annotation(
    stem: str,
    *,
    to_prompt_text: str,
    target_object: str,
    reason: str,
) -> dict:
    ann_path = save_original_annotated(stem, TOP_K)
    return {
        "status": "ok",
        "to_prompt_text": to_prompt_text,
        "target_object": target_object,
        "has_annotated_nodes": False,
        "top_k_nodes": [],
        "annotation_fallback_reason": reason,
        "annotated_screenshot": str(ann_path.relative_to(PROJECT_ROOT)),
        "ann_path": ann_path,
    }


def _is_to_family() -> bool:
    return AGENT.upper() == "TO"


def _compute_retrieval_margin(top_nodes: list[dict]) -> float | None:
    if len(top_nodes) < 2:
        return None
    return float(top_nodes[0].get("final_sim", 0)) - float(top_nodes[1].get("final_sim", 0))


def _serialize_top_nodes(top_nodes: list[dict]) -> list[dict]:
    return [
        {
            "node_id": int(n["node_id"]),
            "bounds": n.get("bounds"),
            "final_sim": round(float(n.get("final_sim", 0)), 4),
        }
        for n in top_nodes
    ]


def _annotate_from_target_object(
    stem: str,
    target_object: str,
    *,
    to_prompt_text: str = "",
) -> dict:
    """已知 target_object → top_k 检索 + 标注（跳过 llm_TO）。"""
    target_object = (target_object or "").strip()
    if not target_object:
        raise ValueError(f"[{stem}] target_object 为空")

    to_text = (to_prompt_text or "").strip() or target_object
    has_nodes = _step_has_nodes(stem)

    if not has_nodes:
        return _fallback_annotation(
            stem,
            to_prompt_text=to_text,
            target_object=target_object,
            reason="no_nodes",
        )

    top_nodes_ranked = rank_nodes_by_to(stem, target_object, max(TOP_K, 2))
    if not top_nodes_ranked:
        return _fallback_annotation(
            stem,
            to_prompt_text=to_text,
            target_object=target_object,
            reason="empty_rank",
        )

    top_nodes = top_nodes_ranked[:TOP_K]
    retrieval_margin = _compute_retrieval_margin(top_nodes_ranked)
    retrieval_final_sim = float(top_nodes[0].get("final_sim", 0))

    screen_w, screen_h = _screenshot_size(stem)
    top1_bounds = top_nodes[0].get("bounds")
    if not is_valid_top1_bounds(top1_bounds, screen_w, screen_h):
        return _fallback_annotation(
            stem,
            to_prompt_text=to_text,
            target_object=target_object,
            reason="invalid_top1_bounds",
        )

    top_k_nodes = _serialize_top_nodes(top_nodes)
    ann_path = annotate_step(stem, TOP_K, top_nodes)
    return {
        "status": "ok",
        "to_prompt_text": to_text,
        "target_object": target_object,
        "has_annotated_nodes": True,
        "top_k_nodes": top_k_nodes,
        "retrieval_final_sim": round(retrieval_final_sim, 4),
        "retrieval_margin": (
            round(retrieval_margin, 4) if retrieval_margin is not None else None
        ),
        "annotated_screenshot": str(ann_path.relative_to(PROJECT_ROOT)),
        "ann_path": ann_path,
    }


def _is_pointer_gt_stem(stem: str) -> bool:
    gt_type = str(_load_gt(stem).get("action_type", "")).strip()
    return gt_type in ("click", "long_press", "long_click")


def _resolve_to_plan(stem: str, to_text: str) -> tuple[str, str, str, str | None]:
    """
    解析本步 llm_TO 结果（AC-low 每步；AC-high 仅 step0）。

    Returns:
        (action_type, target_object, source, raw_response)
        source 为 generate | best | mid | worst
    """
    to_result = generate_target_object(to_text)
    action_type = to_result["action_type"]
    target_object = to_result["target_object"]
    raw = to_result.get("raw_response")
    source = "generate"

    mode = TO_SELECT.lower()
    if mode in ("best", "mid", "worst") and _is_pointer_gt_stem(stem):
        ranked_to = pick_to_string_from_rank(stem, mode)
        if ranked_to:
            return action_type, ranked_to, mode, raw

    return action_type, target_object, source, raw


def _annotate_from_llm_plan(stem: str, to_text: str) -> dict:
    """llm_TO + 条件检索/标注（AC-low 每步；AC-high step0）。"""
    action_type, target_object, to_source, raw = _resolve_to_plan(stem, to_text)

    if action_type not in POINTER_ACTION_TYPES:
        ann = _fallback_annotation(
            stem,
            to_prompt_text=to_text,
            target_object=target_object,
            reason="llm_non_pointer",
        )
    elif not (target_object or "").strip():
        ann = _fallback_annotation(
            stem,
            to_prompt_text=to_text,
            target_object="",
            reason="empty_to",
        )
    else:
        ann = _annotate_from_target_object(
            stem,
            target_object,
            to_prompt_text=to_text,
        )

    ann["planned_action_type"] = action_type
    ann["llm_action_type"] = action_type
    ann["to_select"] = to_source
    if raw:
        ann["llm_to_raw"] = raw
    return ann


def _annotate_from_to(stem: str, to_text: str) -> dict:
    """兼容入口：llm_TO 标注路径。"""
    return _annotate_from_llm_plan(stem, to_text)


def _annotate_from_vlm_plan(
    episode_id: str,
    stem: str,
    pred_action: dict | None,
) -> dict:
    """AC-high step n+1：由上一步 VLM 规划驱动检索/标注。"""
    pred_action = pred_action or {}
    action_type = _parse_next_action_type(pred_action)
    target_object = str(pred_action.get("target_object", "")).strip()
    next_instr = _parse_next_instruction(pred_action)
    goal = _load_episode_goal(episode_id)
    to_text = next_instr or goal or ""

    if not action_type:
        with _PRINT_LOCK:
            print(f"  [high] 上步无 next_action_type，回退原图")
        ann = _fallback_annotation(
            stem,
            to_prompt_text=to_text,
            target_object=target_object,
            reason="missing_next_action_type",
        )
    elif action_type not in POINTER_ACTION_TYPES:
        ann = _fallback_annotation(
            stem,
            to_prompt_text=to_text,
            target_object=target_object,
            reason="vlm_non_pointer",
        )
    elif not target_object:
        ann = _fallback_annotation(
            stem,
            to_prompt_text=to_text,
            target_object="",
            reason="empty_to",
        )
    else:
        ann = _annotate_from_target_object(
            stem,
            target_object,
            to_prompt_text=to_text,
        )

    ann["planned_action_type"] = action_type or None
    ann["vlm_planned_action_type"] = action_type or None
    return ann


def _color_ok_fail(ok: bool) -> str:
    text = "OK" if ok else "FAIL"
    if not _use_color():
        return text
    color = _ANSI_GREEN if ok else _ANSI_RED
    return f"{color}{text}{_ANSI_RESET}"


def _color_skip(text: str) -> str:
    if not _use_color():
        return text
    return f"{_ANSI_YELLOW}{text}{_ANSI_RESET}"


def _print_skip_open_app(step_idx: int, stem: str) -> None:
    with _PRINT_LOCK:
        print(f"  {_color_skip('SKIP open_app')}  step {step_idx} | {stem}")


def _format_action_detail(action: dict | None, *, include_next_instruction: bool = False) -> str:
    if not action:
        return "(empty)"

    parts = [f"type={action.get('action_type', '-')}"]
    action_type = str(action.get("action_type", ""))

    if action_type in ("click", "long_press"):
        if "node_id" in action:
            parts.append(f"node_id={action['node_id']}")
        if "x" in action and "y" in action:
            parts.append(f"x={float(action['x']):.4f}")
            parts.append(f"y={float(action['y']):.4f}")
    elif action_type == "scroll":
        parts.append(f"direction={action.get('direction', '-')}")
    elif action_type == "input_text":
        parts.append(f"text={json.dumps(action.get('text', ''), ensure_ascii=False)}")

    if include_next_instruction and action.get("next_instruction"):
        parts.append(
            f"next_instruction={json.dumps(action.get('next_instruction', ''), ensure_ascii=False)}"
        )

    return " | ".join(parts)


def _step_display_instruction(
    episode_id: str,
    step_idx: int,
    stem: str,
    prev_pred_action: dict | None,
) -> str | None:
    if AC_MODE == "low":
        return _load_instruction(stem)
    if step_idx == 0:
        return f"(step0 / episode goal) {_load_episode_goal(episode_id)}"
    return _parse_next_instruction(prev_pred_action)


def _print_episode_header(
    episode_id: str,
    episode_idx: int,
    episode_total: int,
    goal: str,
) -> None:
    with _PRINT_LOCK:
        print("═" * _BOX_WIDTH)
        print(f" Episode {episode_id}  [{episode_idx}/{episode_total}]")
        print(f" goal: {_clip(goal, _BOX_WIDTH - 6)}")
        print("═" * _BOX_WIDTH)


def _apply_step_eval(step_rec: dict) -> None:
    verdict = judge_step_match(
        step_rec["gt"],
        step_rec.get("pred_action"),
        stem=step_rec["stem"],
        has_annotated_nodes=bool(step_rec.get("has_annotated_nodes", True)),
        agent=AGENT,
        top_k_nodes=step_rec.get("top_k_nodes"),
        top_k=TOP_K,
    )
    step_rec["type_correct"] = verdict["type_correct"]
    step_rec["step_correct"] = verdict["step_correct"]
    step_rec["eval_detail"] = verdict["detail"]
    if "retrieval_hit" in verdict:
        step_rec["retrieval_hit"] = verdict["retrieval_hit"]


def _print_step_result(step_rec: dict) -> None:
    step_idx = step_rec.get("step_idx", "-")
    stem = step_rec.get("stem", "-")
    node_count = len(step_rec.get("top_k_nodes", []))
    if step_rec.get("has_annotated_nodes"):
        nodes_line = f"{node_count} annotated"
        if node_count:
            sims = ", ".join(f"#{n['node_id']}:{n['final_sim']:.3f}" for n in step_rec["top_k_nodes"])
            nodes_line += f"  [{sims}]"
    else:
        nodes_line = "0 (fallback to original screenshot)"

    with _PRINT_LOCK:
        print(f"  ── step {step_idx} | {stem} ──")
        instr = step_rec.get("display_instruction")
        instr_text = "None" if instr is None else _clip(instr)
        print(f"  instruction : {instr_text}")
        if step_rec.get("llm_action_type"):
            print(f"  llm_type    : {step_rec.get('llm_action_type')}")
        print(f"  target_obj  : {_clip(step_rec.get('target_object', '-'))}")
        print(f"  nodes       : {nodes_line}")
        print(f"  GT          : {_format_action_detail(step_rec.get('gt'))}")
        print(
            f"  Pred        : {_format_action_detail(step_rec.get('pred_action'), include_next_instruction=True)}"
        )
        retrieval = step_rec.get("retrieval_hit")
        if retrieval is None:
            retrieval_line = "N/A"
        else:
            retrieval_line = _color_ok_fail(bool(retrieval))
        fallback = step_rec.get("annotation_fallback_reason")
        print(
            f"  Eval        : Type={_color_ok_fail(step_rec['type_correct'])}   "
            f"Step={_color_ok_fail(step_rec['step_correct'])}   "
            f"Top{TOP_K}Retrieval={retrieval_line}"
        )
        if fallback:
            print(f"  fallback    : {fallback}")
        if step_rec.get("status") not in (None, "ok"):
            print(f"  status      : {step_rec.get('status')} | {step_rec.get('error', '')}")
        print()


def _episode_eval_summary(episode_record: dict) -> dict:
    steps = episode_record.get("steps", [])
    eval_steps = [s for s in steps if is_evaluable_step(s)]
    skipped_open_app = sum(
        1
        for s in steps
        if s.get("status") == "skipped" or is_skipped_gt_action(s.get("gt") or {})
    )
    total = len(eval_steps)
    type_correct = sum(1 for s in eval_steps if s.get("type_correct"))
    step_correct = sum(1 for s in eval_steps if s.get("step_correct"))
    sr = bool(eval_steps) and all(s.get("step_correct") for s in eval_steps)
    return {
        "eval_steps": total,
        "skipped_open_app": skipped_open_app,
        "type_correct": type_correct,
        "step_correct": step_correct,
        "type_acc": round(type_correct / total, 4) if total else 0.0,
        "step_acc": round(step_correct / total, 4) if total else 0.0,
        "sr": sr,
    }


def _annotate_for_next_step(
    episode_id: str,
    step_idx: int,
    next_stem: str,
    pred_action: dict | None,
) -> dict:
    """为 step n+1 准备标注上下文。"""
    if AC_MODE == "high":
        return _annotate_from_vlm_plan(episode_id, next_stem, pred_action)

    to_text_next = _to_prompt_after_step(episode_id, step_idx, pred_action)
    return _annotate_from_to(next_stem, to_text_next)


def _is_cpm_agent() -> bool:
    return AGENT.upper() == "CPM"


def _cpm_step_context(stem: str) -> dict:
    """CPM：无 TO 标注，使用原图路径。"""
    screenshot_path = step_paths(stem)["screenshot"]
    if not screenshot_path.is_file():
        raise FileNotFoundError(f"[{stem}] 缺少 screenshot")
    rel = str(screenshot_path.relative_to(PROJECT_ROOT))
    return {
        "status": "ok",
        "to_prompt_text": None,
        "target_object": None,
        "has_annotated_nodes": False,
        "top_k_nodes": [],
        "annotated_screenshot": rel,
        "ann_path": screenshot_path,
        "annotation_skipped": "cpm_agent",
    }


def _is_m2p_agent() -> bool:
    return AGENT.upper() == "M2P"


def _sum_vlm_tokens(plan_out: dict | None, exec_out: dict | None) -> dict | None:
    to_t = (plan_out or {}).get("vlm_tokens") or {}
    act_t = (exec_out or {}).get("vlm_tokens") or {}
    if not to_t and not act_t:
        return None
    return {
        "input": int(to_t.get("input", 0) or 0) + int(act_t.get("input", 0) or 0),
        "output": int(to_t.get("output", 0) or 0) + int(act_t.get("output", 0) or 0),
    }


def _action_summary(action: dict | None) -> str:
    if not action:
        return ""
    action_type = str(action.get("action_type", "")).strip()
    if action_type in ("click", "long_press") and "node_id" in action:
        return f"{action_type} node_id={action['node_id']}"
    if action_type == "scroll":
        return f"scroll {action.get('direction', '-')}"
    if action_type == "input_text":
        text = str(action.get("text", ""))
        return f"input_text {text[:40]}"
    return action_type or "(empty)"


def _parse_planned_action_type(planner: dict | None) -> str:
    """M2P planner → executor fixed_action_type or planner-only action type."""
    if not planner:
        return ""
    action_type = str(planner.get("planned_action_type", "")).strip()
    if action_type == "long_click":
        action_type = "long_press"
    if action_type in AC_ACTION_TYPES:
        return action_type
    return ""


def _create_agent() -> M2Agent | TOAgent | CPMAgent | M2PAgent:
    agent_upper = AGENT.upper()
    if agent_upper == "TO":
        return TOAgent()
    if agent_upper == "M2P":
        return M2PAgent()
    if agent_upper == "CPM":
        return CPMAgent()
    return M2Agent()


def _run_episode(
    episode_id: str,
    agent: M2Agent | TOAgent | CPMAgent,
    *,
    episode_idx: int = 1,
    episode_total: int = 1,
) -> dict:
    stems = stems_in_episode(episode_id)
    if not stems:
        return {
            "episode_id": episode_id,
            "ac_mode": AC_MODE,
            "status": "error",
            "error": "无 step 数据",
            "steps": [],
        }

    if MAX_STEPS is not None:
        stems = stems[:MAX_STEPS]

    goal = _load_episode_goal(episode_id)
    episode_record: dict = {
        "episode_id": episode_id,
        "ac_mode": AC_MODE,
        "goal": goal,
        "num_steps": len(stems),
        "status": "ok",
        "steps": [],
    }

    _print_episode_header(episode_id, episode_idx, episode_total, goal)

    try:
        is_cpm = _is_cpm_agent()
        ann: dict | None = None
        if not is_cpm and not is_skipped_gt_action(_load_gt(stems[0])):
            ann = _annotate_from_to(stems[0], _to_prompt_step0(episode_id, stems[0]))
        prev_pred_action: dict | None = None
        prev_prev_pred_action: dict | None = None

        for step_idx, stem in enumerate(stems):
            gt = _load_gt(stem)

            if is_skipped_gt_action(gt):
                _print_skip_open_app(step_idx, stem)
                step_rec = {
                    "step_idx": step_idx,
                    "stem": stem,
                    "status": "skipped",
                    "skip_reason": "open_app_gt",
                    "gt": gt,
                    "gt_action_type": gt.get("action_type"),
                }
                episode_record["steps"].append(step_rec)
                if not is_cpm and step_idx + 1 < len(stems):
                    next_stem = stems[step_idx + 1]
                    ann = _annotate_for_next_step(
                        episode_id,
                        step_idx,
                        next_stem,
                        prev_pred_action,
                    )
                continue

            if is_cpm:
                ann = _cpm_step_context(stem)
            elif ann is None:
                ann = _annotate_from_to(stem, _to_prompt_step0(episode_id, stem))

            display_instruction = _step_display_instruction(
                episode_id, step_idx, stem, prev_pred_action
            )
            step_rec: dict = {
                "step_idx": step_idx,
                "stem": stem,
                "status": "ok",
                "display_instruction": display_instruction,
                "next_instruction_from_prev": (
                    _parse_next_instruction(prev_pred_action)
                    if AC_MODE == "high" and step_idx > 0
                    else None
                ),
                "to_prompt_text": ann["to_prompt_text"],
                "target_object": ann["target_object"],
                "has_annotated_nodes": ann["has_annotated_nodes"],
                "top_k_nodes": ann["top_k_nodes"],
                "annotated_screenshot": ann["annotated_screenshot"],
                "annotation_fallback_reason": ann.get("annotation_fallback_reason"),
                "annotation_skipped": ann.get("annotation_skipped"),
                "to_select": ann.get("to_select"),
            }
            if ann.get("llm_action_type"):
                step_rec["llm_action_type"] = ann["llm_action_type"]
            if ann.get("vlm_planned_action_type"):
                step_rec["vlm_planned_action_type"] = ann["vlm_planned_action_type"]
            if AC_MODE == "high" and step_idx > 0:
                prev_next_type = _parse_next_action_type(prev_pred_action)
                if prev_next_type:
                    step_rec["next_action_type_from_prev"] = prev_next_type
            if ann.get("llm_to_raw"):
                step_rec["llm_to_raw"] = ann.get("llm_to_raw")
            if ann.get("retrieval_final_sim") is not None:
                step_rec["retrieval_final_sim"] = ann.get("retrieval_final_sim")
            if "retrieval_margin" in ann:
                step_rec["retrieval_margin"] = ann.get("retrieval_margin")

            vlm_ctx = _vlm_context(episode_id, stem)
            prev_step_instruction = _prev_step_instruction_for_vlm(
                stems,
                step_idx,
                prev_pred_action=prev_prev_pred_action,
            )
            current_step_instruction = _current_step_instruction_for_vlm(
                step_idx,
                prev_pred_action=prev_pred_action,
            )
            step_rec.update(vlm_ctx)
            if prev_step_instruction:
                step_rec["prev_step_instruction"] = prev_step_instruction
            if current_step_instruction:
                step_rec["current_step_instruction"] = current_step_instruction

            step_rec["gt"] = gt
            step_rec["gt_action_type"] = gt.get("action_type")

            predict_kwargs = {
                "has_annotated_nodes": ann["has_annotated_nodes"],
                "top_k_nodes": ann["top_k_nodes"],
                **vlm_ctx,
            }
            if not is_cpm and AC_MODE == "low":
                predict_kwargs["prev_step_instruction"] = prev_step_instruction or ""
                predict_kwargs["current_step_instruction"] = current_step_instruction or ""
            if AGENT.upper() == "TO":
                predict_kwargs["target_object"] = ann.get("target_object", "")
            if not is_cpm and ann.get("planned_action_type"):
                predict_kwargs["fixed_action_type"] = ann["planned_action_type"]
            vlm_out = agent.predict(ann["ann_path"], AC_MODE, **predict_kwargs)
            step_rec["vlm_raw_response"] = vlm_out.get("raw_response")
            step_rec["vlm_tokens"] = vlm_out.get("vlm_tokens")
            step_rec["pred_action"] = vlm_out.get("action")
            if vlm_out.get("locator_source"):
                step_rec["locator_source"] = vlm_out.get("locator_source")
            if vlm_out.get("retrieval_final_sim") is not None:
                step_rec["retrieval_final_sim"] = vlm_out.get("retrieval_final_sim")
            if "retrieval_margin" in vlm_out:
                step_rec["retrieval_margin"] = vlm_out.get("retrieval_margin")
            if vlm_out.get("cpm_action") is not None:
                step_rec["cpm_action"] = vlm_out.get("cpm_action")
            if vlm_out.get("schema_error"):
                step_rec["schema_error"] = vlm_out["schema_error"]
            if vlm_out.get("error"):
                step_rec["status"] = "vlm_parse_error"
                step_rec["error"] = vlm_out["error"]
                episode_record["status"] = "partial"

            _apply_step_eval(step_rec)
            _print_step_result(step_rec)
            episode_record["steps"].append(step_rec)
            prev_prev_pred_action = prev_pred_action
            prev_pred_action = step_rec.get("pred_action")

            if vlm_out.get("error"):
                break

            if is_cpm or step_idx + 1 >= len(stems):
                continue

            next_stem = stems[step_idx + 1]
            ann = _annotate_for_next_step(
                episode_id,
                step_idx,
                next_stem,
                step_rec.get("pred_action"),
            )

    except Exception as exc:
        episode_record["status"] = "error"
        episode_record["error"] = str(exc)
        episode_record["traceback"] = traceback.format_exc()
        with _PRINT_LOCK:
            print(f"  {episode_id} | ERROR: {exc}")

    episode_record["eval"] = _episode_eval_summary(episode_record)
    ev = episode_record["eval"]
    skip_note = ""
    if ev.get("skipped_open_app"):
        skip_note = f" | skipped_open_app={ev['skipped_open_app']}"
    with _PRINT_LOCK:
        print(
            f">> episode {episode_id} | steps={ev['eval_steps']}{skip_note} | "
            f"Type {ev['type_correct']}/{ev['eval_steps']} ({ev['type_acc']:.2%}) | "
            f"Step {ev['step_correct']}/{ev['eval_steps']} ({ev['step_acc']:.2%}) | "
            f"SR={'OK' if ev.get('sr') else 'FAIL'}"
        )
        print("═" * _BOX_WIDTH)
        print()

    return episode_record


def _run_episode_m2p(
    episode_id: str,
    agent: M2PAgent,
    *,
    episode_idx: int = 1,
    episode_total: int = 1,
) -> dict:
    """AC-high M2P：每步 vlm_TO(原图)；click/long_press → 检索/标注 → vlm_action；其余 type 由 planner 直接出 action。"""
    stems = stems_in_episode(episode_id)
    if not stems:
        return {
            "episode_id": episode_id,
            "ac_mode": AC_MODE,
            "status": "error",
            "error": "无 step 数据",
            "steps": [],
        }

    if MAX_STEPS is not None:
        stems = stems[:MAX_STEPS]

    goal = _load_episode_goal(episode_id)
    episode_record: dict = {
        "episode_id": episode_id,
        "ac_mode": AC_MODE,
        "goal": goal,
        "num_steps": len(stems),
        "status": "ok",
        "steps": [],
    }

    _print_episode_header(episode_id, episode_idx, episode_total, goal)
    history: list[StepHistoryEntry] = []

    try:
        for step_idx, stem in enumerate(stems):
            gt = _load_gt(stem)

            if is_skipped_gt_action(gt):
                _print_skip_open_app(step_idx, stem)
                step_rec = {
                    "step_idx": step_idx,
                    "stem": stem,
                    "status": "skipped",
                    "skip_reason": "open_app_gt",
                    "gt": gt,
                    "gt_action_type": gt.get("action_type"),
                }
                episode_record["steps"].append(step_rec)
                continue

            raw_png = step_paths(stem)["screenshot"]
            if not raw_png.is_file():
                raise FileNotFoundError(f"[{stem}] 缺少 screenshot")

            plan_out = agent.plan(
                raw_png,
                goal=goal,
                step_num=step_idx + 1,
                max_steps=len(stems),
                history=history,
            )
            planner = plan_out.get("planner")

            if planner is None:
                step_rec = {
                    "step_idx": step_idx,
                    "stem": stem,
                    "status": "vlm_parse_error",
                    "error": plan_out.get("error", "vlm_TO parse error"),
                    "goal": goal,
                    "display_instruction": None,
                    "planner_raw": plan_out.get("raw_response"),
                    "vlm_to_tokens": plan_out.get("vlm_tokens"),
                    "gt": gt,
                    "gt_action_type": gt.get("action_type"),
                    "pred_action": None,
                }
                _apply_step_eval(step_rec)
                _print_step_result(step_rec)
                episode_record["steps"].append(step_rec)
                episode_record["status"] = "partial"
                continue

            step_instruction = (planner.get("step_instruction") or "").strip()
            target_object = (planner.get("target_object") or "").strip()
            planned_type = _parse_planned_action_type(planner)
            exec_out: dict | None = None
            action: dict | None = None
            ann: dict

            if planned_type in M2P_PLANNER_ONLY_TYPES:
                ann = _fallback_annotation(
                    stem,
                    to_prompt_text=step_instruction,
                    target_object=target_object,
                    reason="non_pointer",
                )
                action = agent.build_action_from_planner(
                    planner,
                    step_instruction=step_instruction,
                )
                if action is None:
                    step_rec = {
                        "step_idx": step_idx,
                        "stem": stem,
                        "status": "vlm_parse_error",
                        "error": "planner 缺少该 action type 所需字段 (direction/text)",
                        "goal": goal,
                        "display_instruction": step_instruction or None,
                        "planned_action_type": planned_type or None,
                        "to_prompt_text": ann["to_prompt_text"],
                        "target_object": target_object,
                        "has_annotated_nodes": ann["has_annotated_nodes"],
                        "top_k_nodes": ann["top_k_nodes"],
                        "annotated_screenshot": ann["annotated_screenshot"],
                        "annotation_fallback_reason": ann.get(
                            "annotation_fallback_reason"
                        ),
                        "planner_decision": planner,
                        "planner_raw": plan_out.get("raw_response"),
                        "vlm_to_tokens": plan_out.get("vlm_tokens"),
                        "vlm_tokens": plan_out.get("vlm_tokens"),
                        "gt": gt,
                        "gt_action_type": gt.get("action_type"),
                        "pred_action": None,
                    }
                    _apply_step_eval(step_rec)
                    _print_step_result(step_rec)
                    episode_record["steps"].append(step_rec)
                    episode_record["status"] = "partial"
                    continue
            elif planned_type in M2P_EXECUTOR_TYPES:
                if not target_object:
                    ann = _fallback_annotation(
                        stem,
                        to_prompt_text=step_instruction,
                        target_object="",
                        reason="empty_target_object",
                    )
                    step_rec = {
                        "step_idx": step_idx,
                        "stem": stem,
                        "status": "vlm_parse_error",
                        "error": "click/long_press 缺少 target_object",
                        "goal": goal,
                        "display_instruction": step_instruction or None,
                        "planned_action_type": planned_type or None,
                        "to_prompt_text": ann["to_prompt_text"],
                        "target_object": target_object,
                        "has_annotated_nodes": ann["has_annotated_nodes"],
                        "top_k_nodes": ann["top_k_nodes"],
                        "annotated_screenshot": ann["annotated_screenshot"],
                        "annotation_fallback_reason": ann.get(
                            "annotation_fallback_reason"
                        ),
                        "planner_decision": planner,
                        "planner_raw": plan_out.get("raw_response"),
                        "vlm_to_tokens": plan_out.get("vlm_tokens"),
                        "vlm_tokens": plan_out.get("vlm_tokens"),
                        "gt": gt,
                        "gt_action_type": gt.get("action_type"),
                        "pred_action": None,
                    }
                    _apply_step_eval(step_rec)
                    _print_step_result(step_rec)
                    episode_record["steps"].append(step_rec)
                    episode_record["status"] = "partial"
                    continue
                ann = _annotate_from_target_object(
                    stem,
                    target_object,
                    to_prompt_text=step_instruction,
                )
                exec_out = agent.execute(
                    ann["ann_path"],
                    goal=goal,
                    step_instruction=step_instruction,
                    has_annotated_nodes=ann["has_annotated_nodes"],
                    fixed_action_type=planned_type or None,
                )
                action = exec_out.get("action")
                if exec_out.get("error"):
                    step_rec = {
                        "step_idx": step_idx,
                        "stem": stem,
                        "status": "vlm_parse_error",
                        "error": exec_out.get("error"),
                        "goal": goal,
                        "display_instruction": step_instruction or None,
                        "planned_action_type": planned_type or None,
                        "to_prompt_text": ann["to_prompt_text"],
                        "target_object": target_object,
                        "has_annotated_nodes": ann["has_annotated_nodes"],
                        "top_k_nodes": ann["top_k_nodes"],
                        "annotated_screenshot": ann["annotated_screenshot"],
                        "annotation_fallback_reason": ann.get(
                            "annotation_fallback_reason"
                        ),
                        "planner_decision": planner,
                        "planner_raw": plan_out.get("raw_response"),
                        "vlm_to_tokens": plan_out.get("vlm_tokens"),
                        "vlm_action_raw": exec_out.get("raw_response"),
                        "vlm_action_tokens": exec_out.get("vlm_tokens"),
                        "vlm_tokens": _sum_vlm_tokens(plan_out, exec_out),
                        "gt": gt,
                        "gt_action_type": gt.get("action_type"),
                        "pred_action": action,
                    }
                    _apply_step_eval(step_rec)
                    _print_step_result(step_rec)
                    episode_record["steps"].append(step_rec)
                    episode_record["status"] = "partial"
                    if action:
                        history.append(
                            StepHistoryEntry(
                                step_num=step_idx + 1,
                                step_summary=planner.get("step_summary", "")
                                or _action_summary(action),
                                action_summary=_action_summary(action),
                                step_instruction=step_instruction,
                            )
                        )
                    continue
            else:
                ann = _fallback_annotation(
                    stem,
                    to_prompt_text=step_instruction,
                    target_object=target_object,
                    reason="invalid_planned_type",
                )
                step_rec = {
                    "step_idx": step_idx,
                    "stem": stem,
                    "status": "vlm_parse_error",
                    "error": f"无效 planned_action_type: {planned_type!r}",
                    "goal": goal,
                    "display_instruction": step_instruction or None,
                    "planned_action_type": planned_type or None,
                    "planner_decision": planner,
                    "planner_raw": plan_out.get("raw_response"),
                    "vlm_to_tokens": plan_out.get("vlm_tokens"),
                    "vlm_tokens": plan_out.get("vlm_tokens"),
                    "gt": gt,
                    "gt_action_type": gt.get("action_type"),
                    "pred_action": None,
                }
                _apply_step_eval(step_rec)
                _print_step_result(step_rec)
                episode_record["steps"].append(step_rec)
                episode_record["status"] = "partial"
                continue

            step_rec = {
                "step_idx": step_idx,
                "stem": stem,
                "status": "ok",
                "goal": goal,
                "display_instruction": step_instruction or None,
                "planned_action_type": planned_type or None,
                "to_prompt_text": ann["to_prompt_text"],
                "target_object": target_object,
                "has_annotated_nodes": ann["has_annotated_nodes"],
                "top_k_nodes": ann["top_k_nodes"],
                "annotated_screenshot": ann["annotated_screenshot"],
                "annotation_fallback_reason": ann.get("annotation_fallback_reason"),
                "planner_decision": planner,
                "planner_raw": plan_out.get("raw_response"),
                "vlm_to_tokens": plan_out.get("vlm_tokens"),
                "vlm_action_tokens": (
                    exec_out.get("vlm_tokens") if exec_out else None
                ),
                "vlm_tokens": _sum_vlm_tokens(plan_out, exec_out),
                "gt": gt,
                "gt_action_type": gt.get("action_type"),
                "pred_action": action,
            }
            if exec_out:
                step_rec["vlm_action_raw"] = exec_out.get("raw_response")
                if exec_out.get("schema_error"):
                    step_rec["schema_error"] = exec_out["schema_error"]
            if ann.get("retrieval_final_sim") is not None:
                step_rec["retrieval_final_sim"] = ann.get("retrieval_final_sim")
            if "retrieval_margin" in ann:
                step_rec["retrieval_margin"] = ann.get("retrieval_margin")

            _apply_step_eval(step_rec)
            _print_step_result(step_rec)
            episode_record["steps"].append(step_rec)

            history.append(
                StepHistoryEntry(
                    step_num=step_idx + 1,
                    step_summary=planner.get("step_summary", "")
                    or _action_summary(action),
                    action_summary=_action_summary(action),
                    step_instruction=step_instruction,
                )
            )

    except Exception as exc:
        episode_record["status"] = "error"
        episode_record["error"] = str(exc)
        episode_record["traceback"] = traceback.format_exc()
        with _PRINT_LOCK:
            print(f"  {episode_id} | ERROR: {exc}")

    episode_record["eval"] = _episode_eval_summary(episode_record)
    ev = episode_record["eval"]
    skip_note = ""
    if ev.get("skipped_open_app"):
        skip_note = f" | skipped_open_app={ev['skipped_open_app']}"
    with _PRINT_LOCK:
        print(
            f">> episode {episode_id} | steps={ev['eval_steps']}{skip_note} | "
            f"Type {ev['type_correct']}/{ev['eval_steps']} ({ev['type_acc']:.2%}) | "
            f"Step {ev['step_correct']}/{ev['eval_steps']} ({ev['step_acc']:.2%}) | "
            f"SR={'OK' if ev.get('sr') else 'FAIL'}"
        )
        print("═" * _BOX_WIDTH)
        print()

    return episode_record


def _run_episode_to(
    episode_id: str,
    agent: TOAgent,
    *,
    episode_idx: int = 1,
    episode_total: int = 1,
) -> dict:
    """AC-high TO：每步 vlm_TO(原图)；click/long_press → top-1 检索直接 pred；其余 type 由 planner 直出。"""
    stems = stems_in_episode(episode_id)
    if not stems:
        return {
            "episode_id": episode_id,
            "ac_mode": AC_MODE,
            "status": "error",
            "error": "无 step 数据",
            "steps": [],
        }

    if MAX_STEPS is not None:
        stems = stems[:MAX_STEPS]

    goal = _load_episode_goal(episode_id)
    episode_record: dict = {
        "episode_id": episode_id,
        "ac_mode": AC_MODE,
        "goal": goal,
        "num_steps": len(stems),
        "status": "ok",
        "steps": [],
    }

    _print_episode_header(episode_id, episode_idx, episode_total, goal)
    history: list[StepHistoryEntry] = []

    try:
        for step_idx, stem in enumerate(stems):
            gt = _load_gt(stem)

            if is_skipped_gt_action(gt):
                _print_skip_open_app(step_idx, stem)
                step_rec = {
                    "step_idx": step_idx,
                    "stem": stem,
                    "status": "skipped",
                    "skip_reason": "open_app_gt",
                    "gt": gt,
                    "gt_action_type": gt.get("action_type"),
                }
                episode_record["steps"].append(step_rec)
                continue

            raw_png = step_paths(stem)["screenshot"]
            if not raw_png.is_file():
                raise FileNotFoundError(f"[{stem}] 缺少 screenshot")

            plan_out = agent.plan(
                raw_png,
                goal=goal,
                step_num=step_idx + 1,
                max_steps=len(stems),
                history=history,
            )
            planner = plan_out.get("planner")

            if planner is None:
                step_rec = {
                    "step_idx": step_idx,
                    "stem": stem,
                    "status": "vlm_parse_error",
                    "error": plan_out.get("error", "vlm_TO parse error"),
                    "goal": goal,
                    "display_instruction": None,
                    "planner_raw": plan_out.get("raw_response"),
                    "vlm_to_tokens": plan_out.get("vlm_tokens"),
                    "vlm_tokens": plan_out.get("vlm_tokens"),
                    "gt": gt,
                    "gt_action_type": gt.get("action_type"),
                    "pred_action": None,
                }
                _apply_step_eval(step_rec)
                _print_step_result(step_rec)
                episode_record["steps"].append(step_rec)
                episode_record["status"] = "partial"
                continue

            step_instruction = (planner.get("step_instruction") or "").strip()
            target_object = (planner.get("target_object") or "").strip()
            planned_type = _parse_planned_action_type(planner)
            action: dict | None = None
            ann: dict

            if planned_type in M2P_PLANNER_ONLY_TYPES:
                ann = _fallback_annotation(
                    stem,
                    to_prompt_text=step_instruction,
                    target_object=target_object,
                    reason="non_pointer",
                )
                action = agent.build_action_from_planner(
                    planner,
                    step_instruction=step_instruction,
                )
                if action is None:
                    step_rec = {
                        "step_idx": step_idx,
                        "stem": stem,
                        "status": "vlm_parse_error",
                        "error": "planner 缺少该 action type 所需字段 (direction/text)",
                        "goal": goal,
                        "display_instruction": step_instruction or None,
                        "planned_action_type": planned_type or None,
                        "to_prompt_text": ann["to_prompt_text"],
                        "target_object": target_object,
                        "has_annotated_nodes": ann["has_annotated_nodes"],
                        "top_k_nodes": ann["top_k_nodes"],
                        "annotated_screenshot": ann["annotated_screenshot"],
                        "annotation_fallback_reason": ann.get(
                            "annotation_fallback_reason"
                        ),
                        "planner_decision": planner,
                        "planner_raw": plan_out.get("raw_response"),
                        "vlm_to_tokens": plan_out.get("vlm_tokens"),
                        "vlm_tokens": plan_out.get("vlm_tokens"),
                        "gt": gt,
                        "gt_action_type": gt.get("action_type"),
                        "pred_action": None,
                    }
                    _apply_step_eval(step_rec)
                    _print_step_result(step_rec)
                    episode_record["steps"].append(step_rec)
                    episode_record["status"] = "partial"
                    continue
            elif planned_type in M2P_EXECUTOR_TYPES:
                if not target_object:
                    ann = _fallback_annotation(
                        stem,
                        to_prompt_text=step_instruction,
                        target_object="",
                        reason="empty_target_object",
                    )
                    step_rec = {
                        "step_idx": step_idx,
                        "stem": stem,
                        "status": "vlm_parse_error",
                        "error": "click/long_press 缺少 target_object",
                        "goal": goal,
                        "display_instruction": step_instruction or None,
                        "planned_action_type": planned_type or None,
                        "to_prompt_text": ann["to_prompt_text"],
                        "target_object": target_object,
                        "has_annotated_nodes": ann["has_annotated_nodes"],
                        "top_k_nodes": ann["top_k_nodes"],
                        "annotated_screenshot": ann["annotated_screenshot"],
                        "annotation_fallback_reason": ann.get(
                            "annotation_fallback_reason"
                        ),
                        "planner_decision": planner,
                        "planner_raw": plan_out.get("raw_response"),
                        "vlm_to_tokens": plan_out.get("vlm_tokens"),
                        "vlm_tokens": plan_out.get("vlm_tokens"),
                        "gt": gt,
                        "gt_action_type": gt.get("action_type"),
                        "pred_action": None,
                    }
                    _apply_step_eval(step_rec)
                    _print_step_result(step_rec)
                    episode_record["steps"].append(step_rec)
                    episode_record["status"] = "partial"
                    continue
                ann = _annotate_from_target_object(
                    stem,
                    target_object,
                    to_prompt_text=step_instruction,
                )
                action = agent.build_pointer_pred(
                    planner,
                    ann["top_k_nodes"],
                    has_annotated_nodes=ann["has_annotated_nodes"],
                )
                if action is None:
                    step_rec = {
                        "step_idx": step_idx,
                        "stem": stem,
                        "status": "vlm_parse_error",
                        "error": "top-1 检索失败或缺少有效标注节点",
                        "goal": goal,
                        "display_instruction": step_instruction or None,
                        "planned_action_type": planned_type or None,
                        "to_prompt_text": ann["to_prompt_text"],
                        "target_object": target_object,
                        "has_annotated_nodes": ann["has_annotated_nodes"],
                        "top_k_nodes": ann["top_k_nodes"],
                        "annotated_screenshot": ann["annotated_screenshot"],
                        "annotation_fallback_reason": ann.get(
                            "annotation_fallback_reason"
                        ),
                        "planner_decision": planner,
                        "planner_raw": plan_out.get("raw_response"),
                        "vlm_to_tokens": plan_out.get("vlm_tokens"),
                        "vlm_tokens": plan_out.get("vlm_tokens"),
                        "gt": gt,
                        "gt_action_type": gt.get("action_type"),
                        "pred_action": None,
                    }
                    _apply_step_eval(step_rec)
                    _print_step_result(step_rec)
                    episode_record["steps"].append(step_rec)
                    episode_record["status"] = "partial"
                    continue
            else:
                ann = _fallback_annotation(
                    stem,
                    to_prompt_text=step_instruction,
                    target_object=target_object,
                    reason="invalid_planned_type",
                )
                step_rec = {
                    "step_idx": step_idx,
                    "stem": stem,
                    "status": "vlm_parse_error",
                    "error": f"无效 planned_action_type: {planned_type!r}",
                    "goal": goal,
                    "display_instruction": step_instruction or None,
                    "planned_action_type": planned_type or None,
                    "planner_decision": planner,
                    "planner_raw": plan_out.get("raw_response"),
                    "vlm_to_tokens": plan_out.get("vlm_tokens"),
                    "vlm_tokens": plan_out.get("vlm_tokens"),
                    "gt": gt,
                    "gt_action_type": gt.get("action_type"),
                    "pred_action": None,
                }
                _apply_step_eval(step_rec)
                _print_step_result(step_rec)
                episode_record["steps"].append(step_rec)
                episode_record["status"] = "partial"
                continue

            step_rec = {
                "step_idx": step_idx,
                "stem": stem,
                "status": "ok",
                "goal": goal,
                "display_instruction": step_instruction or None,
                "planned_action_type": planned_type or None,
                "to_prompt_text": ann["to_prompt_text"],
                "target_object": target_object,
                "has_annotated_nodes": ann["has_annotated_nodes"],
                "top_k_nodes": ann["top_k_nodes"],
                "annotated_screenshot": ann["annotated_screenshot"],
                "annotation_fallback_reason": ann.get("annotation_fallback_reason"),
                "planner_decision": planner,
                "planner_raw": plan_out.get("raw_response"),
                "vlm_to_tokens": plan_out.get("vlm_tokens"),
                "vlm_tokens": plan_out.get("vlm_tokens"),
                "gt": gt,
                "gt_action_type": gt.get("action_type"),
                "pred_action": action,
            }
            if ann.get("retrieval_final_sim") is not None:
                step_rec["retrieval_final_sim"] = ann.get("retrieval_final_sim")
            if "retrieval_margin" in ann:
                step_rec["retrieval_margin"] = ann.get("retrieval_margin")

            _apply_step_eval(step_rec)
            _print_step_result(step_rec)
            episode_record["steps"].append(step_rec)

            history.append(
                StepHistoryEntry(
                    step_num=step_idx + 1,
                    step_summary=planner.get("step_summary", "")
                    or _action_summary(action),
                    action_summary=_action_summary(action),
                    step_instruction=step_instruction,
                )
            )

    except Exception as exc:
        episode_record["status"] = "error"
        episode_record["error"] = str(exc)
        episode_record["traceback"] = traceback.format_exc()
        with _PRINT_LOCK:
            print(f"  {episode_id} | ERROR: {exc}")

    episode_record["eval"] = _episode_eval_summary(episode_record)
    ev = episode_record["eval"]
    skip_note = ""
    if ev.get("skipped_open_app"):
        skip_note = f" | skipped_open_app={ev['skipped_open_app']}"
    with _PRINT_LOCK:
        print(
            f">> episode {episode_id} | steps={ev['eval_steps']}{skip_note} | "
            f"Type {ev['type_correct']}/{ev['eval_steps']} ({ev['type_acc']:.2%}) | "
            f"Step {ev['step_correct']}/{ev['eval_steps']} ({ev['step_acc']:.2%}) | "
            f"SR={'OK' if ev.get('sr') else 'FAIL'}"
        )
        print("═" * _BOX_WIDTH)
        print()

    return episode_record


def _run_one_episode(
    *,
    episode_id: str,
    episode_idx: int,
    episode_total: int,
) -> dict:
    agent = _create_agent()
    if _is_m2p_agent():
        assert isinstance(agent, M2PAgent)
        return _run_episode_m2p(
            episode_id,
            agent,
            episode_idx=episode_idx,
            episode_total=episode_total,
        )
    if _is_to_family() and AC_MODE == "high":
        assert isinstance(agent, TOAgent)
        return _run_episode_to(
            episode_id,
            agent,
            episode_idx=episode_idx,
            episode_total=episode_total,
        )
    return _run_episode(
        episode_id,
        agent,
        episode_idx=episode_idx,
        episode_total=episode_total,
    )


def _runs_path(ac_mode: str, agent: str, top_k: int, vlm_model: str) -> Path:
    return PROJECT_ROOT / "runs" / run_filename(
        ac_mode,
        agent,
        top_k,
        vlm_model,
        to_select=TO_SELECT,
    )


def _load_runs_episodes(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("episodes"), list):
        return data["episodes"]
    return []


def _upsert_episode(episodes: list[dict], episode: dict) -> list[dict]:
    episode_id = episode.get("episode_id")
    kept = [e for e in episodes if e.get("episode_id") != episode_id]
    kept.append(episode)
    kept.sort(key=lambda e: str(e.get("episode_id", "")))
    return kept


def _save_runs_episodes(path: Path, episodes: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)


def main() -> None:
    if AC_MODE not in ("low", "high"):
        raise SystemExit(f"AC_MODE 必须为 low 或 high，当前: {AC_MODE!r}")
    if AGENT.upper() == "M2P" and AC_MODE != "high":
        raise SystemExit("m2p 仅支持 AC-high 模式")
    if _is_to_family() and TOP_K != 1:
        raise SystemExit("AGENT=TO 时 TOP_K 必须为 1")
    if TO_SELECT not in TO_SELECT_CHOICES:
        raise SystemExit(
            f"TO_SELECT 必须为 {sorted(TO_SELECT_CHOICES)} 之一，当前: {TO_SELECT!r}"
        )

    episodes = _select_episodes()
    vlm_model = get_vlm_model_name()

    workers = min(PARALLEL_WORKERS, len(episodes)) if episodes else 1
    if TEST_LIST:
        range_label = f"TEST_LIST={TEST_LIST}"
    else:
        range_label = f"TEST_START={TEST_START} TEST_END={TEST_END}"

    print("=" * 60)
    print(
        f"AC-{AC_MODE} 全步实验 | AGENT={AGENT} | TOP_K={TOP_K} | "
        f"TO_SELECT={TO_SELECT} | VLM={vlm_model}"
    )
    if _is_to_family() and AC_MODE == "high":
        print("TO high: single vlm_TO (m2p planner) + top-1 pred for pointer steps")
    if MAX_STEPS is not None:
        print(f"MAX_STEPS={MAX_STEPS}")
    print("=" * 60)
    print(
        f"episode 数: {len(episodes)} | {range_label} | parallel={workers}"
    )
    if episodes:
        print(f"  范围: {episodes[0]} ... {episodes[-1]}")
    print("=" * 60)

    if not episodes:
        print("没有 episode，退出。")
        sys.exit(1)

    out_path = _runs_path(AC_MODE, AGENT, TOP_K, vlm_model)
    stored_episodes = _load_runs_episodes(out_path)
    episode_total = len(episodes)
    processed = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _run_one_episode,
                episode_id=episode_id,
                episode_idx=episode_idx,
                episode_total=episode_total,
            ): (episode_idx, episode_id)
            for episode_idx, episode_id in enumerate(episodes, start=1)
        }
        for future in as_completed(futures):
            episode_idx, episode_id = futures[future]
            try:
                record = future.result()
                with _RUNS_LOCK:
                    stored_episodes = _upsert_episode(stored_episodes, record)
                    _save_runs_episodes(out_path, stored_episodes)
                processed += 1
            except Exception as exc:
                with _PRINT_LOCK:
                    print(
                        f"Episode [{episode_idx}/{episode_total}] {episode_id} failed: {exc}",
                        file=sys.stderr,
                    )

    ok = sum(1 for r in stored_episodes if r.get("status") == "ok")
    partial = sum(1 for r in stored_episodes if r.get("status") == "partial")
    skipped = sum(1 for r in stored_episodes if r.get("status") == "skipped")
    errors = len(stored_episodes) - ok - partial - skipped

    batch_ids = set(episodes)
    batch_steps = 0
    batch_type_ok = 0
    batch_step_ok = 0
    batch_sr_eps = 0
    batch_sr_ok = 0
    batch_retrieval_eligible = 0
    batch_retrieval_ok = 0
    for record in stored_episodes:
        if record.get("episode_id") not in batch_ids:
            continue
        ep_eval_steps = [
            s for s in record.get("steps", []) if is_evaluable_step(s)
        ]
        if ep_eval_steps:
            batch_sr_eps += 1
            if all(s.get("step_correct") for s in ep_eval_steps):
                batch_sr_ok += 1
        for step in record.get("steps", []):
            if not is_evaluable_step(step):
                continue
            batch_steps += 1
            batch_type_ok += int(step.get("type_correct", False))
            batch_step_ok += int(step.get("step_correct", False))
            if step.get("retrieval_hit") is not None:
                batch_retrieval_eligible += 1
                batch_retrieval_ok += int(step.get("retrieval_hit", False))

    print(f"\n{'=' * 60}")
    print(
        f"本批 episode: {len(episodes)} | 完成: {processed} | "
        f"文件中 episode 总数: {len(stored_episodes)}"
    )
    print(f"文件内状态: ok={ok} partial={partial} skipped={skipped} error={errors}")
    if batch_steps:
        retrieval_line = ""
        if batch_retrieval_eligible:
            retrieval_line = (
                f" | Top{TOP_K}Retrieval {batch_retrieval_ok}/{batch_retrieval_eligible} "
                f"({batch_retrieval_ok/batch_retrieval_eligible:.2%})"
            )
        if batch_sr_eps:
            sr_line = f" | SR {batch_sr_ok}/{batch_sr_eps} ({batch_sr_ok/batch_sr_eps:.2%})"
        else:
            sr_line = ""
        print(
            f"本批评估: steps={batch_steps} | "
            f"Type {batch_type_ok}/{batch_steps} ({batch_type_ok/batch_steps:.2%}) | "
            f"Step {batch_step_ok}/{batch_steps} ({batch_step_ok/batch_steps:.2%})"
            f"{sr_line}"
            f"{retrieval_line}"
        )
    print(f"结果: {out_path}")
    print("=" * 60)
    print(f"Done. Processed {processed} episodes.")


if __name__ == "__main__":
    main()
