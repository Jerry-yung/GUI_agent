#!/usr/bin/env python3
"""
处理 AC_data 数据：
1. 拆分 instructions/*.json → step_instructions/*.txt
2. 拆分 GT/*.json → step_GT/*.json（单步格式，参考 test1.6.2）
3. 筛选 action_type 为 click 或 long_press 的 4 件套，其余删除

运行前确保当前目录为 GUI_agent/test4.0.0-DPO/
"""

import json
import os
from pathlib import Path

# ============ 路径配置 ============
BASE_DIR = Path("AC_data")

GT_DIR = BASE_DIR / "GT"
STEP_GT_DIR = BASE_DIR / "step_GT"

INS_DIR = BASE_DIR / "instructions"
STEP_INS_DIR = BASE_DIR / "step_instructions"

SCREENSHOTS_DIR = BASE_DIR / "screenshots"
A11Y_DIR = BASE_DIR / "a11y_trees_L0"

# 确保输出目录存在
STEP_GT_DIR.mkdir(parents=True, exist_ok=True)
STEP_INS_DIR.mkdir(parents=True, exist_ok=True)

# ============ Step 1: 拆分 GT → step_GT ============
print("=" * 50)
print("Step 1: 拆分 GT/*.json → step_GT/*.json")
print("=" * 50)

# 清空旧 step_GT
for old in STEP_GT_DIR.glob("*.json"):
    old.unlink()

gt_files = sorted(GT_DIR.glob("*.json"))
print(f"发现 {len(gt_files)} 个 GT 文件")

for gt_file in gt_files:
    with open(gt_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    actions = data.get("actions", [])
    base_name = gt_file.stem

    for idx, action in enumerate(actions):
        file_name = f"{base_name}_{idx:03d}.json"
        output_path = STEP_GT_DIR / file_name
        with open(output_path, "w", encoding="utf-8") as out_f:
            json.dump(action, out_f, ensure_ascii=False, indent=2)

print(f"生成 {len(list(STEP_GT_DIR.glob('*.json')))} 个 step_GT 文件\n")

# ============ Step 2: 拆分 instructions → step_instructions ============
print("=" * 50)
print("Step 2: 拆分 instructions/*.json → step_instructions/*.txt")
print("=" * 50)

# 清空旧 step_instructions
for old in STEP_INS_DIR.glob("*.txt"):
    old.unlink()

ins_files = sorted(INS_DIR.glob("*.json"))
print(f"发现 {len(ins_files)} 个 instructions 文件")

for ins_file in ins_files:
    with open(ins_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    step_instructions = data.get("step_instructions", [])
    base_name = ins_file.stem

    for idx, instruction in enumerate(step_instructions):
        file_name = f"{base_name}_{idx:03d}.txt"
        output_path = STEP_INS_DIR / file_name
        with open(output_path, "w", encoding="utf-8") as out_f:
            out_f.write(instruction)

print(f"生成 {len(list(STEP_INS_DIR.glob('*.txt')))} 个 step_instructions 文件\n")

# ============ Step 3: 筛选 click/long_press 的 4 件套 ============
print("=" * 50)
print("Step 3: 筛选 action_type ∈ {click, long_press}")
print("=" * 50)

# 收集保留的 stem
keep_stems = set()
for step_gt_file in STEP_GT_DIR.glob("*.json"):
    with open(step_gt_file, "r", encoding="utf-8") as f:
        action = json.load(f)
    if action.get("action_type") in ("click", "long_press"):
        keep_stems.add(step_gt_file.stem)

print(f"保留的 step 数量: {len(keep_stems)}\n")

# 定义需要清理的目录和扩展名
dirs_to_clean = [
    (STEP_GT_DIR, ".json"),
    (STEP_INS_DIR, ".txt"),
    (SCREENSHOTS_DIR, ".png"),
    (A11Y_DIR, ".json"),
]

for dir_path, ext in dirs_to_clean:
    files = sorted(dir_path.glob(f"*{ext}"))
    deleted = 0
    kept = 0
    for f in files:
        if f.stem not in keep_stems:
            f.unlink()
            deleted += 1
        else:
            kept += 1
    print(f"  {dir_path.name}: 删除 {deleted} 个, 保留 {kept} 个")

# ============ 最终验证 ============
print("\n" + "=" * 50)
print("最终验证")
print("=" * 50)

counts = {}
for dir_path, ext in dirs_to_clean:
    cnt = len(list(dir_path.glob(f"*{ext}")))
    counts[dir_path.name] = cnt
    print(f"  {dir_path.name}: {cnt} 个")

# 检查一致性
vals = list(counts.values())
if len(set(vals)) == 1:
    print(f"\n✅ 四件套数量一致，共 {vals[0]} 个！")
else:
    print(f"\n⚠️ 数量不一致，请检查！")

print("\n处理完成！")
