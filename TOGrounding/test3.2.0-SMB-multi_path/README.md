# test3.2.0-SMB-multi_path 工程化

SMAN-Bench **Multi-path** 离线推理与评测。Agent 在真实轨迹上动态翻页（`apply_action` + `all_triple` / `back`），直到到达任务终点页或达到 `MAX_ROUNDS`。评测按 **页面路径后缀** 与 GT 对齐（官方 `multi_path_test`）。

数据只读 [Mobile3M](../../datasets/Mobile3M/datasets/)，产物写入本目录 `cache/`、`results/`、`checkpoints/`。官方 SMAN 工具在 `SMAN-Bench/`（只读引用）。

---

## 目录结构

```
test3.2.0-SMB-multi_path/
├── main.py                 # Multi-path 推理入口
├── eval.py                 # Multi-path 评测（路径后缀对齐）
├── config.yaml             # max_rounds 等默认值
├── agents/
│   ├── m2_agent.py         # M2：llm_TO + TopK + VLM（含 back / summary）
│   ├── TO_agent.py         # TO：Top-1 + VLM 填字段
│   ├── AppAgent.py         # 官方 multipath baseline
│   ├── parser.py           # VLM JSON → click(N) / scroll(N, dir) / back
│   └── prompts.py          # 英文 VLM 提示词（m2）；中文（TO）
├── annotate/               # TopK 标注（图上 #N 标签）
├── utils/                  # sman_bridge、task_context、result_io
├── results/{task_type}_{agent}_{model}/
│   └── {task_index}_{final_page_name}.json
├── checkpoints/
└── SMAN-Bench/
```

---

## 环境

```bash
cd GUI_agent/test3.2.0-SMB-multi_path
pip install opencv-python pillow langchain-openai python-dotenv httpx numpy pyyaml colorama
```

在 `.env` 或上级 `GUI_agent/.env` 配置 API Key（见 `llm_set/llm.py`）。

---

## Mobile3M 前置

```bash
cd datasets/Mobile3M
python3 convert_mobile3m.py datasets/QQmusic_graph_8152 --mode explore
```

Multi-path **必须**使用 `explore` 模式生成 `all_triple.json`。

---

## Multi-path 与 back

### 动态翻页

- 初始页：`final_page_name.split("_")[0]`（如 `QQmusic0`）
- 每步 `apply_action` 经 `all_triple` 跳转到下一页；**点错则进入错误页面**
- 任务完成：`current_page_name == final_page_name`

### back 动作（全 Agent 统一）

| 项 | 说明 |
|----|------|
| 动作空间 | `click` / `scroll` / `input` / **`back`** |
| 翻页 | `QQmusic0_10` + `back` → `QQmusic0`（`rsplit("_", 1)[0]`） |
| action_id | `-1` |
| m2 | 英文 JSON prompt 含 back / multipath 决策；VLM 输出 `summary` |
| to | 中文 JSON prompt 含 back 说明 |
| AppAgent | 英文 `multipath_task_template`（含 `back("back")`） |

### 分步 instruction（官方 multipath 规则）

由 `get_multipath_step_instruction(ctx, current_page_name)` 按**当前页深度**选取：

```python
task_count = current_page_name.count("_")
# QQmusic0 → multi_task_desc[0]
# QQmusic0_10 → multi_task_desc[1]
# back 回 QQmusic0 后自动回到 multi_task_desc[0]
```

与 single-path 的 `round_count` 索引不同。

---

## Agent 概览

| Agent | 说明 |
|-------|------|
| `m2` | llm_TO + Top-K **#N** 标注 + **英文** VLM（含 back、`summary`、`last_summary`） |
| `to` | llm_TO + Top-1 + **中文** VLM 填字段（含 back） |
| `AppAgent` | 官方 multipath：全页 SoM + `multipath_task_template` + `_labeled_multi.png` |

### m2 特有机制（3.2.0）

| 机制 | 说明 |
|------|------|
| `task_desc` / `current_page_name` | 写入 VLM prompt 的任务与页面上下文 |
| `last_summary` | 上一步 VLM 输出的 `summary`，用于页面一致性判断 |
| `summary` | 每步 VLM 必填：本步操作 + 预计下一页 |
| multipath back | 页面与预期不符、无法完成子任务、或无标注时可输出 `back` |

`M2Agent.begin_task()` 初始化任务描述；每步 `decide()` 后更新 `_last_summary`。

---

## llm_TO 流水线

与 3.1.0 相同：`llm_TO` 输出 `action_type` + `target_object`；节点 `label` 为数字 **N**，图上显示 **#N**；步骤日志使用 `llm_action_type`（无 `instruction_hit` 关键词路由）。

| 变量 | 说明 |
|------|------|
| `AGENT` | `m2` / `to` / `AppAgent` |
| `TOP_K` | m2 全路径；TO 对 click/scroll 为 1 |

### VLM 提示词（m2 / TO）

VLM 调用采用 **SystemMessage + HumanMessage** 拆分（对齐 4.1.0）：System 含角色、任务、固定 type、标注图例、Schema、Rules；User 含任务上下文（goal / page / 上步 summary）、当前子任务、检索目标、multi-path back 提示与截图。JSON 输出 schema 不变。

按 **llm_action_type** 路由；**m2 英文** / **TO 中文**；scroll 仍要求 **element（#N）+ direction**：

| 场景 | 入口函数 | 要点 |
|------|----------|------|
| click / long_press | `build_m2_prompt_parts` / `build_to_prompt_parts` | 固定 type + `summary`（3.2.0） |
| scroll | 同上 | element + direction + finger-gesture 规则 |
| input | 同上 | text + `summary` |
| back | 同上 | 无 action 字段，仅 thought + summary |

`AppAgent` 仍使用单条 `call_vlm(prompt, image)`，未拆分。

---

## 推理配置（`main.py` 顶部）

| 变量 | 说明 | 默认 |
|------|------|------|
| `AGENT` | `m2` / `to` / `AppAgent` | `AppAgent` |
| `TASK_JSON` | `simple_normal_tasks.json` 或 `complex_normal_tasks.json` | simple |
| `TASK_TYPE` | `multi_simple` 或 `multi_complex` | `multi_simple` |
| `TOP_K` | m2 全路径；TO click/scroll 为 1 | `5` |
| `MAX_ROUNDS` | 单任务最大轮数（可被 `config.yaml` 覆盖） | `20` |
| `DRY_RUN` | 不调 VLM | `False` |

结果目录：`results/{TASK_TYPE}_{AGENT}_{model_slug}/{task_index}_{final_page_name}.json`

```bash
python main.py
```

---

## 评测配置（`eval.py` 顶部）

| 变量 | 说明 |
|------|------|
| `TASK_TYPE` | 与 `main.py` 一致 |
| `MODEL` | VLM model_slug |
| `EVAL_TASK_TYPE` | `simple`（≤19 步）或 `complex`（≤24 步） |

```bash
python eval.py
```

### Multi-path 指标（路径后缀对齐，非 action_id）

| 指标 | 含义 |
|------|------|
| `success_rate` | 预测路径后缀与 GT **逐步全对** 的任务占比 |
| `action_acc` | 路径段对齐数 / GT 路径段总数 |
| `SE` | 步数效率：`avg_pred_steps / gt_steps`（平均每任务预测步数 ÷ 平均 GT 路径步数，越大越绕路） |

任务 `QQmusic0_34_347` 的 GT 路径后缀为 `["34", "347"]`。

---

## 单步流水线

```
get_multipath_step_instruction(current_page)
  → llm_TO → prepare_round（当前页截图/标注，#N 标签）
  → decide（VLM：click/scroll/input/back；m2 另输出 summary）
  → apply_action → 更新 current_page_name
  → 记录 step；若 current_page == final_page 则 task_complete
```

终端日志示例：

```
Step 2/20 | QQmusic0_10 | "点击搜索栏…"
  llm_action_type=click
  TO="搜索"
  pred=back | next_page=QQmusic0
```

---

## 注意事项

- 在工程根目录运行 `main.py` / `eval.py`。
- 不向 Mobile3M 写入推理产物。
- 修改 `TOP_K`、标注或 prompt 后，清理对应 `cache/` 再重跑。
- 官方 Multi-path 细节见 `SMAN-Bench/运行说明.md`。
