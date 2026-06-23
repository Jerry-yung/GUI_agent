"""Step-level terminal logging: GT vs pred with colored Type/Action flags."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from colorama import Fore, Style

from utils.click_res import is_click_xy_res
from utils.action_id_resolve import (
    click_label_from_action_id,
    click_text_from_candidate,
    format_labeled_action_from_action_id,
    scroll_label_from_action_id,
)

if TYPE_CHECKING:
    from utils.sman_bridge import RoundAssets


def action_type(action_str: str) -> str:
    return (action_str or "").split("(")[0].strip().lower()


def is_type_filter_all(types: list[str] | None) -> bool:
    """TYPE 列表含 ``all`` 时合并各类型子目录做全任务评测。"""
    if not types:
        return False
    return any(str(t).strip().lower() == "all" for t in types)


def normalize_type_filter(types: list[str] | None) -> set[str] | None:
    """Return lowercase type set; None means no GT-type filter (all action types)."""
    if not types:
        return None
    if is_type_filter_all(types):
        return None
    lowered = {
        t.strip().lower()
        for t in types
        if t and t.strip() and t.strip().lower() != "all"
    }
    return lowered or None


def step_at_round(steps: list[dict], round_no: int) -> dict | None:
    for step in steps:
        if int(step.get("round") or 0) == round_no:
            return step
    return None


def gt_type_allowed(
    gt_id: int,
    id_to_action: dict[int, str],
    type_filter: list[str] | None,
) -> bool:
    allowed = normalize_type_filter(type_filter)
    if allowed is None:
        return True
    return action_type(id_to_action.get(gt_id, "")) in allowed


def iter_eval_step_indices(
    gt_ids: list[int],
    id_to_action: dict[int, str],
    steps: list[dict],
    type_filter: list[str] | None,
) -> list[tuple[int, int]]:
    """(step_index, gt_id) pairs that should be included in metrics."""
    pairs: list[tuple[int, int]] = []
    for i, gt_id in enumerate(gt_ids):
        if not gt_type_allowed(gt_id, id_to_action, type_filter):
            continue
        step = step_at_round(steps, i + 1)
        if step and step.get("is_skip"):
            continue
        pairs.append((i, gt_id))
    return pairs


def format_gt_vlm_display(
    gt_id: int,
    gt_action: str,
    *,
    assets: RoundAssets | None,
    all_action_ids: dict[str, int],
) -> str:
    return format_labeled_action_from_action_id(
        gt_id, gt_action, assets=assets, all_action_ids=all_action_ids
    )


def format_pred_vlm_display(
    pred_info: str,
    pred_res: list[str] | None,
    *,
    assets: RoundAssets | None,
    pred_id=None,
    id_to_action: dict[int, str] | None = None,
    all_action_ids: dict[str, int] | None = None,
) -> str:
    if pred_res:
        act = pred_res[0]
        if act == "click" and is_click_xy_res(pred_res):
            return f"click(xy={pred_res[2]},{pred_res[3]})"
        if act == "click":
            label = str(pred_res[1]).lower()
            text = "?"
            if assets is not None:
                m = re.search(r"(?:c)?(\d+)", label, re.IGNORECASE)
                if m:
                    idx = int(m.group(1)) - 1
                    if 0 <= idx < len(assets.click_actions):
                        text = click_text_from_candidate(assets.click_actions[idx])
            return f"click({label}, {text})"
        if act == "scroll":
            return f"scroll({str(pred_res[1]).lower()}, {str(pred_res[2]).lower()})"
        if act == "input":
            return f"input({pred_res[1]})"
        if act == "back":
            return "back"

    if (
        pred_id is not None
        and int(pred_id) >= 0
        and id_to_action
        and all_action_ids is not None
        and int(pred_id) in id_to_action
    ):
        return format_labeled_action_from_action_id(
            int(pred_id),
            id_to_action[int(pred_id)],
            assets=assets,
            all_action_ids=all_action_ids,
        )

    return (pred_info or "?")[:72]


def gt_area_label(
    gt_id: int,
    gt_action: str,
    *,
    assets: RoundAssets | None,
    all_action_ids: dict[str, int],
) -> str | None:
    if assets is None:
        return None
    kind = action_type(gt_action)
    if kind == "click":
        return click_label_from_action_id(
            gt_id,
            assets.click_actions,
            assets.current_page_all_actions,
            all_action_ids,
        )
    if kind == "scroll":
        label, _ = scroll_label_from_action_id(
            gt_id, assets.scroll_action_bounds, all_action_ids
        )
        return label
    return None


def topk_retrieval_match(
    gt_id: int,
    gt_action: str,
    *,
    assets: RoundAssets | None,
    all_action_ids: dict[str, int],
) -> bool:
    gt_label = gt_area_label(
        gt_id, gt_action, assets=assets, all_action_ids=all_action_ids
    )
    if not gt_label or assets is None or not assets.top_k_nodes:
        return False
    selected = {str(n.get("label", "")).lower() for n in assets.top_k_nodes}
    return gt_label.lower() in selected


def gt_similarity_rank(
    gt_id: int,
    gt_action: str,
    *,
    assets: RoundAssets | None,
    all_action_ids: dict[str, int],
) -> int | None:
    gt_label = gt_area_label(
        gt_id, gt_action, assets=assets, all_action_ids=all_action_ids
    )
    if not gt_label or assets is None:
        return None
    return assets.similarity_rank_by_label.get(gt_label.lower())


def type_match(
    pred_id,
    pred_info: str,
    gt_id: int,
    id_to_action: dict[int, str],
    pred_res: list[str] | None = None,
) -> bool:
    gt_type = action_type(id_to_action.get(gt_id, ""))
    if pred_id is not None and int(pred_id) >= 0:
        pred_type = action_type(id_to_action.get(int(pred_id), pred_info))
        return pred_type == gt_type
    if pred_res:
        pred_type = action_type(str(pred_res[0]))
        if pred_type:
            return pred_type == gt_type
    info = (pred_info or "").lower()
    if "scroll" in info:
        return gt_type == "scroll"
    if "click" in info:
        return gt_type == "click"
    if "input" in info:
        return gt_type == "input"
    return False


def action_match(pred_id, gt_id: int) -> bool:
    return str(pred_id) == str(gt_id)


def _color_bool(ok: bool) -> str:
    color = Fore.GREEN if ok else Fore.RED
    return f"{color}{str(ok).lower()}{Style.RESET_ALL}"


def _color_topk_retrieval(ok: bool, gt_rank: int | None = None) -> str:
    text = str(ok).lower()
    if ok and gt_rank is not None:
        text += f" ( No {gt_rank} )"
    color = Fore.GREEN if ok else Fore.RED
    return f"{color}{text}{Style.RESET_ALL}"


def print_step_multipath(
    round_count: int,
    max_rounds: int,
    step_page: str,
    step_instruction: str,
    pred_disp: str,
    next_page: str,
    *,
    llm_action_type: str | None = None,
    target_object: str | None = None,
) -> None:
    print(f'Step {round_count}/{max_rounds} | {step_page} | "{step_instruction}"')
    if llm_action_type:
        print(f"  {Fore.YELLOW}llm_action_type={llm_action_type}{Style.RESET_ALL}")
    if target_object and llm_action_type not in ("scroll", "input", "back"):
        to_text = target_object.replace('"', "'")
        print(f'  TO="{to_text}"')
    print(f"  pred={pred_disp} | next_page={next_page}\n")


def print_step_skip(round_count: int, max_rounds: int) -> None:
    color = getattr(Fore, "LIGHTMAGENTA_EX", Fore.MAGENTA)
    print(f"Step {round_count}/{max_rounds}  {color}SKIP{Style.RESET_ALL}\n")


def print_step_compare(
    round_count: int,
    max_rounds: int,
    step_page: str,
    step_instruction: str,
    target_object: str | None,
    gt_disp: str,
    pred_disp: str,
    *,
    type_ok: bool,
    action_ok: bool,
    top_k: int,
    topk_ok: bool,
    llm_action_type: str | None = None,
    scroll_node_cnt: int | None = None,
    gt_similarity_rank: int | None = None,
) -> None:
    print(f'Step {round_count}/{max_rounds} | {step_page} | "{step_instruction}"')
    if llm_action_type:
        print(f"  {Fore.YELLOW}llm_action_type={llm_action_type}{Style.RESET_ALL}")

    if llm_action_type == "scroll":
        cnt = scroll_node_cnt if scroll_node_cnt is not None else 0
        print(f"  Scroll_node_cnt={cnt} | Retrieval={_color_bool(topk_ok)}")
    elif llm_action_type not in ("input", "back"):
        to_text = (target_object or "?").replace('"', "'")
        print(f'  TO="{to_text}" | Top_{top_k}_Retrieval={_color_topk_retrieval(topk_ok, gt_similarity_rank)}')

    print(
        f"  GT={gt_disp} | pred={pred_disp} | "
        f"Type={_color_bool(type_ok)} | Action={_color_bool(action_ok)}\n"
    )


def build_step_displays(
    gt_id: int,
    gt_action: str,
    pred_id,
    pred_info: str,
    *,
    assets: RoundAssets | None,
    all_action_ids: dict[str, int],
    id_to_action: dict[int, str] | None = None,
    pred_res: list[str] | None = None,
) -> tuple[str, str]:
    gt_disp = format_gt_vlm_display(
        gt_id, gt_action, assets=assets, all_action_ids=all_action_ids
    )
    pred_disp = format_pred_vlm_display(
        pred_info,
        pred_res,
        assets=assets,
        pred_id=pred_id,
        id_to_action=id_to_action,
        all_action_ids=all_action_ids,
    )
    return gt_disp, pred_disp
