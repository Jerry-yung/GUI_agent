#!/usr/bin/env python3
"""
并行调用 3 个 target LLM 生成 target_object，每个 LLM 调用 3 轮。

输入：AC_data/step_instructions/*.txt
输出：
  1. target/target_object/{stem}.json      — 原始 LLM 输出（含多轮结果）
  2. target/TO_index/{stem}.json           — 去重后的 TO 索引映射
  3. AC_data/embeddings/TO_emb/{stem}/{stem}_{TO_id}.npy — TO text embedding

处理流程：
  1. 读取 instruction → 3 轮 × 3 LLM 并行生成 target_object
  2. 汇总保存 target_object
  3. 对所有 TO 去重（大小写敏感），分配 TO_id
  4. 保存 TO_index（TO_id / TO_string / TO_LLM 映射）
  5. 调用 vlm_embedding 为每个唯一 TO 生成 embedding
"""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# ============================================================
# 路径与全局配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm_set.llm import llm_target1, llm_target2, llm_target3, vlm_embedding

BASE_DIR = PROJECT_ROOT / "AC_data"
STEP_INS_DIR = BASE_DIR / "step_instructions"
OUTPUT_DIR = PROJECT_ROOT / "target" / "target_object"
TO_INDEX_DIR = PROJECT_ROOT / "target" / "TO_index"
TO_EMB_DIR = BASE_DIR / "embeddings" / "TO_emb"

EMBEDDING_DIM = 2560

TEST_START = 300
TEST_END = 400

MAX_TOKENS = 512
NUM_ROUNDS = 3

LLM_CONFIGS = [
    llm_target1,
    llm_target2,
    llm_target3,
]


def _get_model_name(inst) -> str:
    """从 LLM 实例中提取模型名称。"""
    m = inst.model
    if isinstance(m, str):
        return m
    # ChatOpenAI 实例
    return getattr(m, "model", getattr(m, "model_name", type(inst).__name__))


# ============================================================
# Prompt & 解析
# ============================================================
def _desc_prompt(instruction: str) -> str:
    return (
        "You are a GUI automation assistant. Given a user's step instruction, "
        "predict the text label or description of the next UI element the user should interact with.\n\n"
        "Rules:\n"
        "1. Generate exactly 3 different English words, short phrases, or concise sentences that describe the target UI element.\n"
        "2. Do NOT output Chinese or other non-English text.\n"
        "3. Output ONLY a valid JSON object with exactly one field: target_objects.\n"
        "4. Do NOT include markdown code blocks or explanations.\n\n"
        f'Instruction: "{instruction}"\n\n'
        "Output format:\n"
        '{"target_objects": ["word1", "word2", "word3"]}\n'
    )


def _parse_target_objects(text: str) -> list[str]:
    """从 LLM 响应中解析 target_objects list，失败返回空列表。"""
    text = text.strip()
    parsed: list[str] = []

    # 1. 直接 JSON
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "target_objects" in data:
            val = data["target_objects"]
            if isinstance(val, list):
                parsed = [str(v).strip() for v in val if str(v).strip()]
    except (json.JSONDecodeError, ValueError):
        pass

    if parsed:
        return parsed

    # 2. markdown 代码块
    for match in re.finditer(r"```(?:json)?\s*([\s\S]*?)\s*```", text):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "target_objects" in data:
                val = data["target_objects"]
                if isinstance(val, list):
                    parsed = [str(v).strip() for v in val if str(v).strip()]
                    if parsed:
                        return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    # 3. 最外层花括号
    for match in re.finditer(r"(\{[\s\S]*\})", text):
        try:
            data = json.loads(match.group(1))
            if isinstance(data, dict) and "target_objects" in data:
                val = data["target_objects"]
                if isinstance(val, list):
                    parsed = [str(v).strip() for v in val if str(v).strip()]
                    if parsed:
                        return parsed
        except (json.JSONDecodeError, ValueError):
            continue

    return parsed


# ============================================================
# LLM 调用封装
# ============================================================
def _is_openrouter(inst) -> bool:
    return type(inst).__name__ == "LLM_openrouter"


def _call_llm(name: str, llm_instance, instruction: str, round_idx: int) -> list[str]:
    """调用单个 LLM，返回 target_objects 字符串列表。"""
    print(f"  [第{round_idx}轮] 正在请求 {name} ...")
    prompt = _desc_prompt(instruction)

    try:
        if _is_openrouter(llm_instance):
            # LLM_openrouter.chat 返回 dict
            resp = llm_instance.chat(content=instruction, max_tokens=MAX_TOKENS)
            raw_text = resp.get("content", "")
        else:
            # ChatOpenAI invoke
            response = llm_instance.model.invoke(prompt, max_tokens=MAX_TOKENS)
            raw_text = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        print(f"    [{name}] 调用失败: {exc}")
        return []

    targets = _parse_target_objects(raw_text)
    print(f"  [第{round_idx}轮] 已拿到 {name} 的答案：{targets}")
    return targets


def _run_single_round(instruction: str, round_idx: int) -> dict[str, list[str]]:
    """并行调用 3 个 LLM 一次，返回 {llm_name: [target_objects]}。"""
    results: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(_call_llm, _get_model_name(inst), inst, instruction, round_idx): _get_model_name(inst)
            for inst in LLM_CONFIGS
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                targets = future.result(timeout=60)
                results[name] = targets
            except Exception as exc:
                print(f"    [{name}] 第 {round_idx} 轮超时/异常: {exc}")
                results[name] = []
    return results


# ============================================================
# 主流程
# ============================================================
def process_stem(stem: str, index: int, total: int) -> None:
    print(f"\n===================[{index}/{total}]======================")

    ins_path = STEP_INS_DIR / f"{stem}.txt"
    if not ins_path.is_file():
        print(f"[{index}/{total}] {stem}: 缺少 instruction，跳过")
        return

    instruction = ins_path.read_text(encoding="utf-8").strip()
    if not instruction:
        print(f"[{index}/{total}] {stem}: instruction 为空，跳过")
        return

    print(f"instruction: {instruction}")

    # 3 轮调用，每轮并行 3 个 LLM
    round_results: list[dict[str, list[str]]] = []
    for r in range(1, NUM_ROUNDS + 1):
        round_res = _run_single_round(instruction, r)
        round_results.append(round_res)

    # 汇总：每个 LLM 的 3 轮结果，每轮是一个包含 3 个词的 list
    output = []
    for inst in LLM_CONFIGS:
        name = _get_model_name(inst)
        targets = [rnd.get(name, []) for rnd in round_results]
        output.append({
            "llm": name,
            "results": targets,
        })

    # 增量更新：新模型 append，相同模型以列表为单位 extend results
    out_path = OUTPUT_DIR / f"{stem}.json"
    existing: list[dict] = []
    if out_path.is_file():
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, KeyError):
            existing = []

    # 用 dict 加速查找
    existing_map = {item["llm"]: item for item in existing if "llm" in item}
    for item in output:
        llm_name = item["llm"]
        if llm_name in existing_map:
            existing_map[llm_name]["results"] = item["results"]
        else:
            existing.append(item)
            existing_map[llm_name] = item

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    # 打印摘要
    summary = " | ".join(
        f"{item['llm']}: {item['results']}" for item in existing
    )
    print(f"[{index}/{total}] {stem}: {summary}")

    # ============================================================
    # 去重 → 分配 TO_id → 保存 TO_index → 生成 embedding
    # ============================================================
    _build_to_index_and_embed(stem, existing)


def _build_to_index_and_embed(stem: str, target_data: list[dict]) -> None:
    """对 target_object 去重，保存索引，并为每个唯一 TO 生成 embedding。"""
    to_records: dict[str, dict] = {}

    for item in target_data:
        llm_name = item.get("llm", "unknown")
        results = item.get("results", [])
        for round_list in results:
            if not isinstance(round_list, list):
                continue
            for to_str in round_list:
                if not isinstance(to_str, str):
                    continue
                to_str = to_str.strip()
                if not to_str:
                    continue
                if to_str not in to_records:
                    to_records[to_str] = {
                        "TO_id": len(to_records),
                        "TO_string": to_str,
                        "TO_LLM": set(),
                    }
                to_records[to_str]["TO_LLM"].add(llm_name)

    if not to_records:
        print(f"  [{stem}] 无有效 TO，跳过索引/embedding")
        return

    index_list = sorted(to_records.values(), key=lambda x: x["TO_id"])
    for item in index_list:
        item["TO_LLM"] = sorted(item["TO_LLM"])

    # 保存 TO_index
    TO_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(TO_INDEX_DIR / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(index_list, f, ensure_ascii=False, indent=2)

    # 生成 embedding
    success = 0
    zero = 0
    skip = 0
    for item in index_list:
        to_id = item["TO_id"]
        to_string = item["TO_string"]
        emb_path = TO_EMB_DIR / stem / f"{stem}_{to_id}.npy"

        if emb_path.is_file():
            cached = np.load(emb_path).astype(np.float32)
            if cached.shape == (EMBEDDING_DIM,):
                skip += 1
                continue
            emb_path.unlink(missing_ok=True)

        TO_EMB_DIR.mkdir(parents=True, exist_ok=True)
        try:
            vec = np.asarray(vlm_embedding.embed_text(to_string), dtype=np.float32)
        except Exception as exc:
            print(f"    [{stem}_{to_id}] embedding 失败 ({to_string!r}): {exc}")
            vec = np.zeros(EMBEDDING_DIM, dtype=np.float32)

        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, vec.astype(np.float32))

        if np.allclose(vec, 0):
            zero += 1
        else:
            success += 1

    print(
        f"\033[92m  [{stem}] TO_index={len(index_list)} | "
        f"emb成功={success} 零向量={zero} 跳过={skip}\033[0m"
    )


def main() -> None:
    all_stems = sorted(p.stem for p in STEP_INS_DIR.glob("*.txt"))
    start = TEST_START if TEST_START is not None else 0
    end = TEST_END if TEST_END is not None else len(all_stems)
    stems = all_stems[start:end]

    print("=" * 60)
    print("llm_target.py — 并行调用 3 LLM × 3 轮生成 target_object + TO_index + TO_emb")
    print("=" * 60)
    print(f"instruction 总数: {len(all_stems)}")
    print(f"本次范围: [{start}, {end}) → {len(stems)} 个")
    if stems:
        print(f"  示例: {stems[0]} ... {stems[-1]}")
    print(f"  MAX_TOKENS={MAX_TOKENS}, NUM_ROUNDS={NUM_ROUNDS}")
    print("=" * 60)

    if not stems:
        print("没有样本，退出。")
        sys.exit(1)

    for i, stem in enumerate(stems, 1):
        process_stem(stem, i, len(stems))

    print(f"\n{'=' * 60}")
    print("处理完成")
    print(f"  target_object: {OUTPUT_DIR}")
    print(f"  TO_index:      {TO_INDEX_DIR}")
    print(f"  TO_emb:        {TO_EMB_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
