"""SMAN-Bench action mapping and round asset preparation."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from utils.click_geom import click_area_idx_from_norm_coords, parse_norm_coords
from utils.sman_setup import get_sman_utils


@dataclass
class RoundAssets:
    click_actions: list
    input_actions: list
    scroll_actions: list
    scroll_action_bounds: list
    current_page_all_actions: dict
    drawn_screenshot: str
    valid_action_id_count: int
    html_content: str
    xml_content: str
    screenshot_path: str
    top_k_nodes: list[dict] = field(default_factory=list)
    target_object: str | None = None
    instruction_hit: str | None = None
    scroll_node_cnt: int | None = None
    similarity_rank_by_label: dict[str, int] = field(default_factory=dict)
    page_name: str | None = None
    retrieval_top_k: int | None = None
    topk_retrieval: bool | None = None
    gt_action_id: int | None = None
    gt_click_area_idx: int | None = None
    gt_scroll_area_idx: int | None = None
    gt_action_type: str | None = None


def _sman():
    sman = get_sman_utils()
    return {
        "action_click": sman.action_click,
        "action_input": sman.action_input,
        "action_scroll": sman.action_scroll,
        "actions_generate": sman.actions_generate,
        "build_action_id_maps": sman.build_action_id_maps,
        "current_actions_generate": sman.current_actions_generate,
        "draw_bbox_multi": sman.draw_bbox_multi,
        "find_dir_with_prefix": sman.find_dir_with_prefix,
        "load_all_action_ids": sman.load_all_action_ids,
        "print_with_color": sman.print_with_color,
    }


def load_graph_indices(task_dir: str) -> tuple[dict, dict[int, str], dict[str, list[int]], dict[str, str]]:
    sman = _sman()
    all_action_ids = sman["load_all_action_ids"](task_dir)

    with open(os.path.join(task_dir, "all_page_actions.json"), encoding="utf-8") as fp:
        all_page_actions = json.load(fp)

    with open(os.path.join(task_dir, "all_triple.json"), encoding="utf-8") as fp:
        all_page_triples = json.load(fp)["data"]

    _, id_to_action = sman["build_action_id_maps"](all_action_ids)

    all_page_convert: dict[str, str] = {}
    for triple in all_page_triples:
        all_page_convert[triple[0] + "act" + str(triple[1])] = triple[2]

    current_page_actions: dict[str, list[int]] = {}
    for page_data in all_page_actions["data"]:
        current_page_actions[page_data["name"]] = page_data["action_valid"]

    return all_action_ids, id_to_action, current_page_actions, all_page_convert


def prepare_sman_candidates(
    html_content: str,
    xml_content: str,
    current_page_name: str,
    id_to_action: dict[int, str],
    current_page_actions: dict[str, list[int]],
) -> tuple[list, list, list, list, dict, int]:
    sman = _sman()
    current_page_all_action_ids = current_page_actions.get(current_page_name, [])
    current_action_infos = [id_to_action[int(aid)] for aid in current_page_all_action_ids]

    click_actions, input_actions_pre, scroll_actions, current_page_all_actions = sman["actions_generate"](
        html_content, xml_content
    )
    click_actions, input_actions, scroll_actions, scroll_action_bounds = sman["current_actions_generate"](
        current_action_infos,
        click_actions,
        input_actions_pre,
        scroll_actions,
        current_page_all_actions,
    )
    return (
        click_actions,
        input_actions,
        scroll_actions,
        scroll_action_bounds,
        current_page_all_actions,
        len(current_page_all_action_ids),
    )


def render_labeled_image(
    screenshot_path: str,
    click_actions: list,
    scroll_action_bounds: list,
    output_path: Path,
    *,
    resize: tuple[int, int] = (1080, 2400),
) -> str:
    sman = _sman()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sman["draw_bbox_multi"](screenshot_path, str(output_path), click_actions)
    sman["draw_bbox_multi"](str(output_path), str(output_path), scroll_action_bounds)
    imgcv = cv2.imread(str(output_path))
    if imgcv is not None and resize:
        imgcv = cv2.resize(imgcv, resize)
        cv2.imwrite(str(output_path), imgcv)
    return str(output_path)


def render_scroll_only_labeled_image(
    screenshot_path: str,
    scroll_nodes: list[dict],
    output_path: Path,
    *,
    resize: tuple[int, int] = (1080, 2400),
) -> str:
    """Official SMAN scroll labeling: center sN text via put_btext (no dashed boxes)."""
    get_sman_utils()
    from put_btext import put_btext

    output_path.parent.mkdir(parents=True, exist_ok=True)
    imgcv = cv2.imread(screenshot_path)
    if imgcv is None:
        raise FileNotFoundError(f"Cannot read screenshot: {screenshot_path}")

    text_color = (255, 250, 250)
    bg_color = (10, 10, 10)
    for node in scroll_nodes:
        bounds = node.get("bounds") or []
        if len(bounds) < 4:
            continue
        left, top, right, bottom = bounds[:4]
        label = str(node.get("label") or "")
        if not label:
            continue
        imgcv = put_btext(
            imgcv,
            label,
            text_offset_x=(left + right) // 2 + 10,
            text_offset_y=(top + bottom) // 2 + 10,
            vspace=10,
            hspace=10,
            font_scale=1,
            thickness=2,
            background_RGB=bg_color,
            text_RGB=text_color,
            alpha=0.5,
        )

    if resize:
        imgcv = cv2.resize(imgcv, resize)
    cv2.imwrite(str(output_path), imgcv)
    return str(output_path)


def prepare_round_assets(
    task_dir: str,
    current_page_name: str,
    id_to_action: dict[int, str],
    current_page_actions: dict[str, list[int]],
    *,
    labeled_output: Path | None = None,
    use_official_labels: bool = True,
) -> RoundAssets | None:
    sman = _sman()
    screenshot_path = os.path.join(task_dir, current_page_name, current_page_name + "-screen.png")
    html_file = os.path.join(task_dir, current_page_name, current_page_name + "-html.txt")
    xml_file = os.path.join(task_dir, current_page_name, current_page_name + "-xml.txt")

    if not os.path.isfile(screenshot_path):
        sman["print_with_color"](f"Missing screenshot: {screenshot_path}", "red")
        return None

    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()
    with open(xml_file, "r", encoding="utf-8") as f:
        xml_content = f.read()

    (
        click_actions,
        input_actions,
        scroll_actions,
        scroll_action_bounds,
        current_page_all_actions,
        valid_count,
    ) = prepare_sman_candidates(
        html_content, xml_content, current_page_name, id_to_action, current_page_actions
    )

    if use_official_labels:
        if labeled_output is None:
            raise ValueError("labeled_output is required when use_official_labels=True")
        drawn = render_labeled_image(
            screenshot_path, click_actions, scroll_action_bounds, labeled_output
        )
    else:
        # Agent writes cache/labeled/{task_name}/*.png; do not touch labeled/ root.
        drawn = str(screenshot_path)

    return RoundAssets(
        click_actions=click_actions,
        input_actions=input_actions,
        scroll_actions=scroll_actions,
        scroll_action_bounds=scroll_action_bounds,
        current_page_all_actions=current_page_all_actions,
        drawn_screenshot=drawn,
        valid_action_id_count=valid_count,
        html_content=html_content,
        xml_content=xml_content,
        screenshot_path=screenshot_path,
    )


def apply_action(
    res: list[str],
    assets: RoundAssets,
    all_action_ids: dict,
    current_page_name: str,
    all_page_convert: dict,
) -> tuple[str, str | int, str, list, list]:
    sman = _sman()
    ans_action_id: list = []
    ans_action_info: list = []
    act_name = res[0]
    params = res[:-1]

    if act_name == "click":
        if len(res) >= 5 and str(res[1]).lower() == "xy":
            coords = parse_norm_coords(res[2], res[3])
            if coords is None:
                ans_action_id.append(-3)
                ans_action_info.append("click xy parse error")
                return current_page_name, -3, "click xy parse error", ans_action_id, ans_action_info
            area_idx = click_area_idx_from_norm_coords(
                coords[0],
                coords[1],
                assets.click_actions,
                assets.screenshot_path,
            )
            if area_idx is None:
                ans_action_id.append(-3)
                ans_action_info.append("click xy area error")
                return current_page_name, -3, "click xy area error", ans_action_id, ans_action_info
        else:
            _, area = params
            area_idx = int(re.findall(r"(\d+)", str(area))[0])
        new_page_name, action_info, action_id = sman["action_click"](
            assets.click_actions,
            area_idx,
            assets.current_page_all_actions,
            all_action_ids,
            current_page_name,
            all_page_convert,
        )
        if action_info == "ERROR":
            ans_action_info.append(str(area) + "click error")
            ans_action_id.append(action_id)
        else:
            ans_action_info.append(action_info)
            ans_action_id.append(action_id)
        return new_page_name, action_id, action_info, ans_action_id, ans_action_info

    if act_name == "scroll":
        _, area, direction = params
        area_idx = int(re.findall(r"(\d+)", str(area))[0])
        new_page_name, action_info, action_id = sman["action_scroll"](
            assets.scroll_action_bounds,
            area_idx,
            direction,
            all_action_ids,
            current_page_name,
            all_page_convert,
        )
        if action_info == "ERROR":
            ans_action_info.append(str(area) + str(direction) + "scroll error")
            ans_action_id.append(action_id)
        else:
            ans_action_info.append(action_info)
            ans_action_id.append(action_id)
        return new_page_name, action_id, action_info, ans_action_id, ans_action_info

    if act_name == "input":
        _, text = params
        new_page_name, action_info, action_id = sman["action_input"](
            text,
            assets.input_actions,
            assets.current_page_all_actions,
            all_action_ids,
            current_page_name,
            all_page_convert,
        )
        if action_info == "ERROR":
            ans_action_info.append("input" + text + "error")
            ans_action_id.append(action_id)
        else:
            ans_action_info.append(action_info)
            ans_action_id.append(action_id)
        return new_page_name, action_id, action_info, ans_action_id, ans_action_info

    if act_name == "back":
        new_page_name = current_page_name.rsplit("_", 1)[0]
        ans_action_id.append(-1)
        ans_action_info.append("back")
        return new_page_name, -1, "back", ans_action_id, ans_action_info

    ans_action_id.append(-3)
    ans_action_info.append("no such action error")
    return current_page_name, -3, "no such action error", ans_action_id, ans_action_info
