# Agents

本目录实现 VLM Agent：**m2**、**m2p**（AC-high）、**TO**，以及无 SoM 的 **CPM**。实验入口在 `main.py`，通过 `AGENT` 切换；m2 / TO（AC-low）使用 `predict()`；m2p / TO（AC-high）使用 `plan()` 主循环。

评测规则见 [eval/README.md](../eval/README.md)。

---

## 快速对比

| | **m2** | **m2p** | **TO** | **CPM** |
|---|--------|---------|--------|---------|
| 模块 | `m2_agent.py` | `m2p_agent.py` | `TO_agent.py` | `CPM_agent.py` |
| 模式 | low / high | **仅 high** | low / high | low / high |
| VLM 数（high） | 1 | 2（planner + action） | **1（planner only）** | 1 |
| 输入图 | 标注图 top_k | planner 原图 + executor 标注图 | planner 原图；pointer 步 top-1 标注 | **原图** |
| click 定位 | VLM `node_id` | executor `node_id` | **检索 top-1**（无 vlm_action） | 归一化 `x,y` |
| step 评测 | `judge_m2` | 同 m2 | `TO_top1` | `judge_baseline` |

---

## m2（`M2Agent`）

**定位**：标准「llm_TO + SoM 标注 + VLM 选 `node_id`」。

- **AC-low**：每步 `llm_TO` 定 type + `target_object` → 条件检索 → VLM 填字段
- **AC-high**：step0 `llm_TO(goal)`；step n+1 由上步 VLM `next_action_type` 等三字段驱动
- **导出**：`_encode_image`、`_normalize_action` 供 TO / m2p 复用

---

## m2p（`M2PAgent`）

**定位**：AC-high 双 VLM — planner（`vlm_TO` 原图 + history）+ executor（pointer 步 `vlm_action` 标注图）。

| planned_action_type | 行为 |
|---------------------|------|
| click / long_press | planner 输出 `target_object` → TopK + SoM → **vlm_action** |
| scroll / input_text / wait / navigate_* | planner **直接输出完整 action**（含 direction/text），**不调 vlm_action** |

详见 `m2p_prompts.py`、`m2p_agent.py` 与 `main._run_episode_m2p()`。

---

## TO（`TOAgent`）

**定位**：与 m2p 相同的每步 `vlm_TO` planner + step history；pointer 步用检索 **top-1** 直接作为 pred，**不调用 vlm_action**。

| 模式 | 流程 |
|------|------|
| **AC-high** | 每步 `plan(原图)` → click/long_press：`target_object` → top-1 → `build_pointer_pred`；其余 type：`build_action_from_planner`（同 m2p） |
| **AC-low** | 每步 `llm_TO` + `predict(标注图)`（不变） |

- **配置**：`AGENT="TO"` 且 `TOP_K=1`
- **主循环**：high → `main._run_episode_to()`；low → `main._run_episode()`
- **评测**：`judge_top1_center`；另统计 Retrieval Hit

---

## CPM（`CPMAgent`）

**定位**：原图坐标 baseline（无 TO / 无 SoM），对齐 AgentCPM 协议。

- **main**：跳过 llm_TO、标注、TO_SELECT
- **评测**：`judge_baseline`；无 `retrieval_hit`

---

## 目录结构

| 文件 | 作用 |
|------|------|
| `m2_agent.py` | M2Agent |
| `m2p_agent.py` / `m2p_prompts.py` | M2P planner + executor |
| `TO_agent.py` | TOAgent |
| `CPM_agent.py` | CPMAgent |
| `prompts.py` | m2 / TO prompt 构建 |
| `scroll_gesture.py` | scroll 方向保守后处理 |
| `action_validate.py` | jsonschema + agent 规则 |
| `schema/` | AC / CPM JSON schema |

---

## 切换方式

`main.py` 顶部：

```python
AC_MODE = "low"       # 或 "high"
AGENT = "m2"          # m2 | m2p | TO | CPM
TOP_K = 5             # TO 时必须为 1
TO_SELECT = "generate"  # generate | best | mid | worst（CPM / m2p 无效）
```

**结果文件**：

- m2 / TO：`runs/{AC_MODE}_{agent}_top{TOP_K}{to_select}_{vlm_model}.json`
- m2p：`runs/high_m2p_top{TOP_K}{to_select}_{vlm_model}.json`
- CPM：`runs/{AC_MODE}_CPM_{vlm_model}.json`

**离线评估**：

```bash
python eval/eval_run.py runs/low_m2_top5generate_qwen-vl-max.json
```
