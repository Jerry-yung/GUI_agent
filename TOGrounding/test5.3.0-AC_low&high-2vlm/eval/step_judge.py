#!/usr/bin/env python3
"""单步 type_acc / step_acc 判定。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.judge_llm import judge_baseline, judge_m2, judge_top1_center
from eval.scroll_direction import map_ac_content_to_gesture, scroll_gesture_matches_gt

TYPE_ONLY_ACTIONS = frozenset({"wait", "navigate_back", "navigate_home"})
POINTER_ACTIONS = frozenset({"click", "long_press"})
SKIP_GT_ACTION_TYPES = frozenset({"open_app"})


def normalize_action_type(action_type: str) -> str:
    action_type = str(action_type or "").strip()
    if action_type == "long_click":
        return "long_press"
    return action_type


def is_skipped_gt_action(gt: dict | None) -> bool:
    """GT 为 open_app 等不参与评测的动作时返回 True（对齐官方 AgentCPM 剔除 open_app）。"""
    if not isinstance(gt, dict):
        return False
    return normalize_action_type(gt.get("action_type", "")) in SKIP_GT_ACTION_TYPES


def is_evaluable_step(step: dict) -> bool:
    """跳过的步或 GT 为 open_app 的步不计入 type_acc / step_acc / SR。"""
    if step.get("status") == "skipped":
        return False
    gt = step.get("gt")
    if isinstance(gt, dict) and is_skipped_gt_action(gt):
        return False
    gt_type = step.get("gt_action_type")
    if gt_type and normalize_action_type(str(gt_type)) in SKIP_GT_ACTION_TYPES:
        return False
    return True


def judge_type_match(gt: dict, pred: dict | None) -> bool:
    if not pred:
        return False
    return normalize_action_type(pred.get("action_type", "")) == normalize_action_type(
        gt.get("action_type", "")
    )


def _input_text_match(pred_text: str, gt_text: str) -> bool:
    """与 AgentCPM 官方 eval 对齐：lower + strip 后互为子串。"""
    pred_norm = str(pred_text or "").lower().strip()
    gt_norm = str(gt_text or "").lower().strip()
    if not pred_norm or not gt_norm:
        return False
    return pred_norm in gt_norm or gt_norm in pred_norm


def compute_retrieval_hit(
    gt: dict,
    top_k_nodes: list | None,
    *,
    top_k: int | None = None,
) -> bool | None:
    """
    Top-K Retrieval Hit：仅对 click/long_press 且存在 top_k_nodes 时计算。
    检索 top_k 个候选中任一 node_id 落在 GT nearest_5 即为 hit。
    """
    gt_type = normalize_action_type(gt.get("action_type", ""))
    if gt_type not in POINTER_ACTIONS:
        return None
    if not top_k_nodes:
        return None

    nearest = gt.get("nearest_5") or []
    gt_ids = {int(item["node_id"]) for item in nearest if "node_id" in item}
    if not gt_ids:
        return None

    k = top_k if top_k is not None else len(top_k_nodes)
    k = min(max(int(k), 0), len(top_k_nodes))
    if k == 0:
        return None

    candidate_ids = {int(top_k_nodes[i]["node_id"]) for i in range(k)}
    return bool(candidate_ids & gt_ids)


def _has_norm_coords(pred: dict) -> bool:
    try:
        x = float(pred["x"])
        y = float(pred["y"])
    except (KeyError, TypeError, ValueError):
        return False
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def _judge_pointer_action(
    gt: dict,
    pred: dict,
    *,
    stem: str,
    agent: str = "m2",
    top_k_nodes: list | None = None,
) -> tuple[bool, dict]:
    if agent.upper() == "TO" and top_k_nodes:
        detail = judge_top1_center(stem, top_k_nodes[0])
        detail["mode"] = "TO_top1"
        return detail["hit"], detail

    if "node_id" in pred:
        detail = judge_m2(int(pred["node_id"]), stem)
        detail["mode"] = "m2"
        return detail["hit"], detail

    if _has_norm_coords(pred):
        detail = judge_baseline(float(pred["x"]), float(pred["y"]), stem)
        detail["mode"] = "baseline"
        return detail["hit"], detail

    return False, {"mode": "missing_locator", "reason": "no node_id or x,y"}


def judge_step_match(
    gt: dict,
    pred: dict | None,
    *,
    stem: str,
    has_annotated_nodes: bool = True,
    agent: str = "m2",
    top_k_nodes: list | None = None,
    top_k: int | None = None,
) -> dict:
    """
    判定单步 type_acc 与 step_acc。

    Returns:
        {
            "type_correct": bool,
            "step_correct": bool,
            "gt_action_type": str,
            "pred_action_type": str | None,
            "detail": dict,
        }
    """
    gt_type = normalize_action_type(gt.get("action_type", ""))
    pred_type = (
        normalize_action_type(pred.get("action_type", "")) if pred else None
    )

    result: dict = {
        "type_correct": judge_type_match(gt, pred),
        "step_correct": False,
        "gt_action_type": gt_type,
        "pred_action_type": pred_type,
        "retrieval_hit": compute_retrieval_hit(gt, top_k_nodes, top_k=top_k),
        "detail": {},
    }

    if not result["type_correct"]:
        result["detail"] = {"reason": "type_mismatch"}
        return result

    if gt_type in TYPE_ONLY_ACTIONS:
        result["step_correct"] = True
        result["detail"] = {"reason": "type_only_action"}
        return result

    if gt_type in ("click", "long_press"):
        hit, detail = _judge_pointer_action(
            gt,
            pred or {},
            stem=stem,
            agent=agent,
            top_k_nodes=top_k_nodes if has_annotated_nodes else None,
        )
        result["step_correct"] = hit
        result["detail"] = detail
        return result

    if gt_type == "scroll":
        pred_dir = str((pred or {}).get("direction", "")).strip().lower()
        gt_dir = str(gt.get("direction", "")).strip().lower()
        expected_gesture = map_ac_content_to_gesture(gt_dir)
        correct = scroll_gesture_matches_gt(pred_dir, gt_dir)
        result["step_correct"] = correct
        result["detail"] = {
            "pred_direction": pred_dir,
            "gt_direction": gt_dir,
            "expected_gesture_direction": expected_gesture,
            "mode": "gesture_vs_ac_gt",
        }
        return result

    if gt_type == "input_text":
        pred_text = str((pred or {}).get("text", ""))
        gt_text = str(gt.get("text", ""))
        correct = _input_text_match(pred_text, gt_text)
        result["step_correct"] = correct
        result["detail"] = {
            "pred_text": pred_text,
            "gt_text": gt_text,
            "pred_text_norm": pred_text.lower().strip(),
            "gt_text_norm": gt_text.lower().strip(),
        }
        return result

    result["detail"] = {"reason": f"unsupported_gt_type:{gt_type}"}
    return result
