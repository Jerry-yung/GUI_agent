#!/usr/bin/env python3
"""
主入口：GUI Agent 评测（baseline / m2）

修改全局变量：
    AGENT = "baseline" | "m2"
    TOP_K = 10          # m2 标注候选数
    MODE = "best"       # m2 标注 TO 选取：best | mid | worst
    TEST_START = 0
    TEST_END = 10
    TEST_LIST = []

流程：
1. 读取 step_instructions 确定样本列表
2. m2：按 cos_sim+MODE 生成 top_k 标注截图 → M2Agent → judge_m2
   baseline：原始截图 → BaselineAgent → judge_baseline
3. 保存结果到 runs/{agent}_{model}.json（m2 为 m2_top{TOP_K}_{MODE}_{model}.json）
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from llm_set.llm import get_vlm_model_name, slug_for_run_filename

# ============================================================
# 全局配置
# ============================================================
# AGENT = "baseline"
AGENT = "m2"
TOP_K = 3 # 仅 m2 有效
MODE = "worst" # 仅 m2 有效

TEST_START = 0
TEST_END = 300

TEST_LIST = [] # 非空即替代 TEST_START/TEST_END 切片

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

SCREENSHOTS_DIR = BASE_DIR / "AC_data" / "screenshots"
INSTRUCTIONS_DIR = BASE_DIR / "AC_data" / "step_instructions"
GT_DIR = BASE_DIR / "AC_data" / "step_GT"
RUNS_DIR = BASE_DIR / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 样本索引
# ============================================================
def list_all_step_ins_stems() -> list[str]:
    """以 step_instructions/*.txt 为基准，返回所有 stem 排序列表。"""
    return sorted(p.stem for p in INSTRUCTIONS_DIR.glob("*.txt"))


def get_step_ins_stems(start: int | None, end: int | None) -> list[str]:
    """按切片取 stem 列表。"""
    all_stems = list_all_step_ins_stems()
    s = start if start is not None else 0
    e = end if end is not None else len(all_stems)
    return all_stems[s:e]


# ============================================================
# runs 结果文件命名与合并
# ============================================================
def get_run_output_path(
    agent: str,
    model_slug: str,
    *,
    top_k: int | None = None,
    mode: str | None = None,
) -> Path:
    """结果文件路径：baseline → runs/{agent}_{model}.json；m2 → runs/m2_top{k}_{mode}_{model}.json"""
    if agent == "m2":
        if top_k is None or mode is None:
            raise ValueError("m2 结果命名需要 top_k 与 mode")
        return RUNS_DIR / f"m2_top{top_k}_{mode.lower()}_{model_slug}.json"
    return RUNS_DIR / f"{agent}_{model_slug}.json"


def merge_run_records(existing: list[dict], new_records: list[dict]) -> list[dict]:
    """按 sample_id 合并；新结果覆盖同 id 的旧记录，其余保留。"""
    by_id: dict[str, dict] = {}
    for r in existing:
        sid = r.get("sample_id")
        if sid:
            by_id[sid] = r
    for r in new_records:
        sid = r.get("sample_id")
        if sid:
            by_id[sid] = r
    return [by_id[k] for k in sorted(by_id.keys())]


def save_run_results(
    agent: str,
    model_slug: str,
    new_records: list[dict],
    *,
    top_k: int | None = None,
    mode: str | None = None,
) -> Path:
    """写入 runs；同 agent/model（及 m2 的 top_k/mode）时合并并覆盖相同 sample_id。"""
    output_path = get_run_output_path(agent, model_slug, top_k=top_k, mode=mode)
    merged = list(new_records)
    if output_path.is_file():
        with open(output_path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        if isinstance(existing, list):
            merged = merge_run_records(existing, new_records)
            overwritten = sum(
                1
                for r in new_records
                if r.get("sample_id") in {e.get("sample_id") for e in existing}
            )
            print(
                f"  合并已有结果: {len(existing)} 条 + 本次 {len(new_records)} 条 -> {len(merged)} 条"
            )
            if overwritten:
                print(f"  覆盖 sample_id: {overwritten} 个")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    return output_path


# ============================================================
# Agent 评测
# ============================================================
def _run_baseline(samples: list[str]) -> list[dict]:
    """运行 BaselineAgent：原始截图 → 归一化坐标 → judge_llm 判定。"""
    from agents.baseline_agent import BaselineAgent
    from judge.judge_llm import judge_baseline

    agent = BaselineAgent()
    records = []

    for stem in samples:
        screenshot_path = SCREENSHOTS_DIR / f"{stem}.png"
        instruction_path = INSTRUCTIONS_DIR / f"{stem}.txt"
        gt_path = GT_DIR / f"{stem}.json"

        with open(instruction_path, "r", encoding="utf-8") as f:
            instruction = f.read().strip()

        try:
            pred = agent.predict(str(screenshot_path), instruction)
            norm_x, norm_y = pred["x"], pred["y"]

            judge_result = judge_baseline(norm_x, norm_y, stem)

            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)

            record = {
                "sample_id": stem,
                "agent": "baseline",
                "instruction": instruction,
                "prediction": {
                    "norm_x": norm_x,
                    "norm_y": norm_y,
                    "pixel_x": judge_result["pixel_x"],
                    "pixel_y": judge_result["pixel_y"],
                },
                "gt": {
                    "x": gt_data["x"],
                    "y": gt_data["y"],
                    "nearest_5": gt_data.get("nearest_5", []),
                },
                "hit": judge_result["hit"],
                "hit_detail": {
                    "hit_by_bbox": judge_result["hit_by_bbox"],
                    "hit_by_distance": judge_result["hit_by_distance"],
                    "norm_distance": judge_result["norm_distance"],
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"    ⚠️  ERROR: {type(e).__name__}: {e}")
            record = {
                "sample_id": stem,
                "agent": "baseline",
                "instruction": instruction if "instruction" in dir() else "",
                "prediction": {"error": str(e)},
                "gt": None,
                "hit": False,
                "timestamp": datetime.now().isoformat(),
            }

        records.append(record)
        print(f"  [{len(records)}/{len(samples)}] {stem}: {'✅' if record.get('hit') else '❌'}")

    return records


def _run_m2(samples: list[str], top_k: int, mode: str) -> list[dict]:
    """运行 M2Agent：top_k 标注截图 + 指令 → click_id → judge_m2 判定。"""
    from agents.annotate.annotate import annotated_dir_for
    from agents.m2_agent import M2Agent
    from judge.judge_llm import judge_m2

    annotated_dir = annotated_dir_for(top_k, mode)
    agent = M2Agent()
    records = []

    for stem in samples:
        annotated_path = annotated_dir / f"{stem}.png"
        instruction_path = INSTRUCTIONS_DIR / f"{stem}.txt"
        gt_path = GT_DIR / f"{stem}.json"

        with open(instruction_path, "r", encoding="utf-8") as f:
            instruction = f.read().strip()

        try:
            pred = agent.predict(str(annotated_path), instruction)
            click_id = pred["click_id"]

            judge_result = judge_m2(click_id, stem)

            with open(gt_path, "r", encoding="utf-8") as f:
                gt_data = json.load(f)

            record = {
                "sample_id": stem,
                "agent": "m2",
                "config": {"top_k": top_k, "mode": mode},
                "instruction": instruction,
                "prediction": {"click_id": click_id, "pred_node_id": click_id},
                "gt": {
                    "x": gt_data["x"],
                    "y": gt_data["y"],
                    "nearest_5": gt_data.get("nearest_5", []),
                },
                "hit": judge_result["hit"],
                "hit_detail": {
                    "gt_node_ids": judge_result["gt_node_ids"],
                    "matched_node_ids": judge_result["matched_node_ids"],
                },
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"    ⚠️  ERROR: {type(e).__name__}: {e}")
            record = {
                "sample_id": stem,
                "agent": "m2",
                "config": {"top_k": top_k, "mode": mode},
                "instruction": instruction if "instruction" in dir() else "",
                "prediction": {"error": str(e)},
                "gt": None,
                "hit": False,
                "timestamp": datetime.now().isoformat(),
            }

        records.append(record)
        print(f"  [{len(records)}/{len(samples)}] {stem}: {'✅' if record.get('hit') else '❌'}")

    return records


# ============================================================
# 主入口
# ============================================================
def main():
    all_stems = list_all_step_ins_stems()

    if TEST_LIST:
        samples = [all_stems[i] for i in TEST_LIST if 0 <= i < len(all_stems)]
        start, end = None, None
    else:
        samples = get_step_ins_stems(TEST_START, TEST_END)
        start = TEST_START if TEST_START is not None else 0
        end = TEST_END if TEST_END is not None else len(all_stems)

    print("=" * 50)
    print("GUI Agent 评测")
    print("=" * 50)
    print(f"Agent: {AGENT}")
    print(f"TOP_K: {TOP_K}")
    print(f"MODE: {MODE}")
    print(f"step_ins 总数: {len(all_stems)}")
    if TEST_LIST:
        print(f"本次范围: TEST_LIST={TEST_LIST} -> {len(samples)} 个样本")
    else:
        print(f"本次范围: [{start}, {end}) -> {len(samples)} 个样本")
    if samples:
        print(f"  示例: {samples[0]}" + (f" ... {samples[-1]}" if len(samples) > 1 else ""))
    print("=" * 50)

    if not samples:
        print("没有找到样本，请检查 step_ins 目录与 TEST_START/TEST_END。")
        print(f"  指令目录: {INSTRUCTIONS_DIR}")
        sys.exit(1)

    print(f"\nStep: Agent 评测 ({AGENT})\n")

    if AGENT == "baseline":
        records = _run_baseline(samples)
    elif AGENT == "m2":
        from agents.annotate.annotate import annotate_stems

        print(f"Step: 生成 top_{TOP_K} ({MODE}) 标注截图\n")
        annotate_stems(samples, TOP_K, MODE)
        records = _run_m2(samples, TOP_K, MODE)
    else:
        print(f"未知 AGENT: {AGENT!r}，支持 baseline / m2")
        sys.exit(1)

    # 统计
    total = len(records)
    hit_count = sum(1 for r in records if r.get("hit"))
    accuracy = hit_count / total * 100 if total > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"评测完成: {hit_count}/{total} 命中 (命中率: {accuracy:.2f}%)")
    print(f"{'=' * 50}")

    # 保存结果（固定文件名；同 sample_id 覆盖）
    model_name = slug_for_run_filename(get_vlm_model_name())
    save_kwargs = (
        {"top_k": TOP_K, "mode": MODE.lower()} if AGENT == "m2" else {}
    )
    output_path = save_run_results(AGENT, model_name, records, **save_kwargs)
    print(f"结果已保存: {output_path}")


if __name__ == "__main__":
    main()
