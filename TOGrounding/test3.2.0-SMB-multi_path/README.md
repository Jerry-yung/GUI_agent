# test3.2.0-SMB-multi 工程化

SMAN-Bench **Multi-path** 离线推理与评测。Agent 在真实轨迹上动态翻页（`apply_action` + `all_triple` / `back`），直到到达任务终点页或达到 `MAX_ROUNDS`。评测按 **页面路径后缀** 与 GT 对齐（官方 `multi_path_test`）。

数据只读 [Mobile3M](../../datasets/Mobile3M/datasets/)，产物写入本目录 `cache/`、`results/`、`checkpoints/`。官方 SMAN 工具在 `SMAN-Bench/`（只读引用）。

---

## 目录结构

```
test3.2.0-SMB-multi-工程化/
├── main.py                 # Multi-path 推理入口
├── eval.py                 # Multi-path 评测（路径后缀对齐）
├── config.yaml             # max_rounds 等默认值
├── agents/                 # m2 / to / toa / AppAgent（均支持 back）
├── annotate/               # TopK 标注管线
├── utils/                  # sman_bridge、task_context、result_io
├── results/{task_type}_{agent}_{model}/
│   └── {task_index}_{final_page_name}.json
├── checkpoints/
└── SMAN-Bench/
```

---

## 环境

```bash
cd GUI_agent/test3.2.0-SMB-multi-工程化
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
| m2/to/toa | 中文 JSON prompt 含 back 说明（借鉴官方 multipath） |
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
| `m2` | llm_TO + Top-K 标注 + 中文 VLM（含 back） |
| `to` | 非 ambiguous 步 Top-1 + VLM 判 type（含 back） |
| `toa` | Top-1 + element/坐标（含 back） |
| `AppAgent` | 官方 multipath：全页 SoM + `multipath_task_template` + `_labeled_multi.png` |

---

## `instruction_hit` 与任务分布

与 3.1.0 相同：`annotate/instruction_hint.py` 的 `infer_instruction_hit` 对**分步描述**做关键词匹配，路由标注与 prompt；**不是 GT 动作类型**。

#### 如何判定 `click` / `scroll` / `input` / `ambiguous`

| 结果 | 规则 |
|------|------|
| `click` | 文案中**仅**命中点击类词（「点击」「点按」「tap」等） |
| `scroll` | 文案中**仅**命中滑动类词（「滑动」「滚动」「scroll」「swipe」等） |
| `input` | 文案中**仅**命中输入类词（「输入」「填写」「input」等） |
| `ambiguous` | 同时命中两类及以上，或三类都未命中 |

详见 `annotate/instruction_hint.py`（`len(matched)==1` 才为明确类，否则 `ambiguous`）。

#### Mobile3M 任务 JSON：逐步分布（参考）

四文件任务名去重：12654 任务、51605 步、49 App。

**全局（逐步）**

| instruction_hit | 步数 | 占比 |
|-----------------|------|------|
| click | 30362 | 58.8% |
| scroll | 9306 | 18.0% |
| ambiguous | 9546 | 18.5% |
| input | 2391 | 4.6% |

**各 App（单位：步；按任务数降序）**

| App | 任务 | click | scroll | input | ambiguous |
|-----|------|-------|--------|-------|-----------|
| Qunar | 2000 | 5149 | 1805 | 423 | 1401 |
| QQmusic | 2000 | 2669 | 4385 | 2 | 494 |
| zhuishushenqi | 1999 | 5771 | 231 | 550 | 3078 |
| duapp | 1399 | 3983 | 610 | 56 | 1231 |
| baicizhan | 927 | 2549 | 204 | 184 | 533 |
| QQmail | 771 | 1527 | 102 | 371 | 244 |
| pdfreader | 535 | 865 | 9 | 118 | 864 |
| kugou | 116 | 222 | 201 | 0 | 40 |
| baiduBrowser | 102 | 304 | 58 | 26 | 33 |
| anjuke | 100 | 205 | 73 | 48 | 51 |
| wuba | 99 | 217 | 72 | 32 | 72 |
| taptap | 99 | 232 | 116 | 2 | 34 |
| youdao | 98 | 233 | 102 | 32 | 37 |
| tonghuashun | 98 | 269 | 99 | 4 | 28 |
| BaiduMap | 97 | 178 | 22 | 83 | 80 |
| vipshop | 96 | 207 | 95 | 2 | 71 |
| medicinehelper | 95 | 248 | 21 | 29 | 73 |
| gaodeMap | 94 | 220 | 15 | 50 | 76 |
| QQBrowser | 94 | 276 | 37 | 3 | 36 |
| smarthome | 93 | 286 | 54 | 28 | 65 |
| xiaoyuan | 93 | 253 | 51 | 32 | 64 |
| qqdownloader | 91 | 267 | 58 | 4 | 35 |
| ximalaya | 91 | 251 | 35 | 1 | 44 |
| bili | 91 | 228 | 65 | 7 | 41 |
| keep | 88 | 256 | 46 | 10 | 59 |
| ludashi | 87 | 290 | 31 | 1 | 15 |
| wpsOffice | 87 | 229 | 55 | 26 | 68 |
| didi | 85 | 216 | 29 | 26 | 87 |
| ctrip | 83 | 189 | 57 | 16 | 51 |
| xiaomiShop | 81 | 204 | 92 | 10 | 62 |
| tencentnews | 80 | 197 | 19 | 49 | 30 |
| seekbooks | 80 | 284 | 68 | 4 | 67 |
| zuoyebang | 74 | 282 | 14 | 2 | 35 |
| QQ | 68 | 213 | 22 | 3 | 30 |
| mtxx | 68 | 179 | 42 | 6 | 58 |
| calculator | 64 | 145 | 15 | 32 | 56 |
| kuaishou | 60 | 142 | 42 | 9 | 32 |
| netmail | 59 | 233 | 9 | 50 | 32 |
| xiaohongshu | 58 | 123 | 65 | 4 | 5 |
| cainiao | 48 | 94 | 18 | 24 | 21 |
| qqlive | 47 | 102 | 35 | 3 | 11 |
| zhihu | 34 | 50 | 70 | 2 | 24 |
| qqpimsecure | 32 | 122 | 18 | 17 | 23 |
| UCMobile | 27 | 59 | 14 | 2 | 11 |
| androidesk | 25 | 69 | 10 | 3 | 10 |
| supercaculator | 20 | 42 | 0 | 5 | 22 |
| QQReader | 9 | 19 | 5 | 0 | 7 |
| cloudweather | 7 | 5 | 9 | 0 | 2 |
| pureweather | 5 | 9 | 1 | 0 | 3 |

---

## 推理配置（`main.py` 顶部）

| 变量 | 说明 | 默认 |
|------|------|------|
| `AGENT` | `m2` / `to` / `toa` / `AppAgent` | `AppAgent` |
| `TASK_JSON` | `simple_normal_tasks.json` 或 `complex_normal_tasks.json` | simple |
| `TASK_TYPE` | `multi_simple` 或 `multi_complex` | `multi_simple` |
| `TOP_K` | m2 Top-K；to/toa ambiguous 步 | `5` |
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
  → infer_instruction_hit → prepare_round（当前页截图/标注）
  → decide（VLM：click/scroll/input/back）
  → apply_action → 更新 current_page_name
  → 记录 step；若 current_page == final_page 则 task_complete
```

终端日志示例：

```
Step 2/20 | QQmusic0_10 | "点击搜索栏…"
  instruction_hit=click
  TO="搜索"
  pred=back | next_page=QQmusic0
```

---

## 注意事项

- 在工程根目录运行 `main.py` / `eval.py`。
- 不向 Mobile3M 写入推理产物。
- 修改 `TOP_K` 或标注后，清理对应 `cache/` 再重跑。
- 官方 Multi-path 细节见 `SMAN-Bench/运行说明.md`。
