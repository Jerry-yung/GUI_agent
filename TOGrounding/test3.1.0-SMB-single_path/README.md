# test3.1.0-SMB 工程化

SMAN-Bench **Single-path** 离线推理与评测。每步沿 GT 路径推进，对比 VLM 预测与 GT 的 **Type / Action / TopK Retrieval**（TOa 坐标点击另计 **action_acc_geom**）。

数据只读 [Mobile3M](../../datasets/Mobile3M/datasets/)，产物写入本目录 `cache/`、`results/`、`checkpoints/`。官方 SMAN 工具在 `SMAN-Bench/`（只读引用）。

- 数据来源：https://huggingface.co/datasets/xwk123/Mobile3M/tree/main

---

## 目录结构

```
test3.1.0-SMB工程化/
├── main.py                 # 推理入口
├── eval.py                 # 评测入口（多 agent 对比表）
├── config.yaml
├── llm_set/llm.py          # LLM / VLM / embedding 配置
├── agents/
│   ├── m2_agent.py         # M2：llm_TO → TopK → VLM（全标注动作空间）
│   ├── TO_agent.py         # TO：非 ambiguous 步 Top-1 + VLM 仅判 type
│   ├── TOa_agent.py        # TOa：Top-1 建议框 + element / 归一化坐标
│   ├── AppAgent.py         # 官方 AppAgent single-path 适配
│   ├── appagent_parse.py   # AppAgent Observation/Action 解析
│   ├── to_click.py         # TO / TOa VLM 解析与 click 后处理
│   ├── to_retrieval.py     # TOa 检索分数读取
│   ├── parser.py           # VLM JSON → click(cN) / scroll(sN, dir) / xy
│   ├── prompts.py          # 中文 VLM 提示词（m2 / TO / TOa）
│   └── vlm_client.py
├── annotate/
│   ├── instruction_hint.py # 分步 instruction → click/scroll/input/ambiguous
│   ├── llm_TO.py           # step_instruction → target_object（中文）
│   ├── topk.py             # embedding TopK + 标注管线
│   ├── embedder.py         # qwen3-vl-embedding
│   ├── annotate.py         # 画框与 cN/sN 标签（P1–P3）
│   ├── node_dedupe.py      # TopK 前：同 bounds 去重
│   └── node_filter.py      # TopK 后：嵌套 scroll 抑制等
├── utils/
│   ├── sman_bridge.py      # prepare_round_assets / apply_action
│   ├── click_geom.py       # 归一化坐标 → bbox 命中判定
│   ├── click_res.py        # click xy 动作格式工具
│   ├── action_id_resolve.py
│   ├── step_log.py         # 终端 Step 日志、TYPE 过滤、评测对齐
│   ├── task_context.py
│   └── result_io.py        # 按 GT 类型分目录写结果
├── cache/
│   ├── labeled/top_{K}/{task_name}/{page}.png
│   ├── nodes/top_{K}/{task_name}/{page}.json
│   ├── retrieval/top_{K}/{task_name}/{page}/{to_digest}.json
│   └── embeddings/{task_name}/{page}/
├── results/{task_type}_{agent}_{model}/
│   ├── click/{task_index}_{final_page_name}.json
│   ├── scroll/...
│   └── input/...
├── checkpoints/            # eval 指标 JSON
└── SMAN-Bench/             # 官方 utils（action_click / gt_action_ids 等）
```

---

## 环境

```bash
cd GUI_agent/test3.1.0-SMB工程化
pip install opencv-python pillow langchain-openai python-dotenv httpx numpy pyyaml colorama
```

在 `.env` 或上级 `GUI_agent/.env` 配置 API Key（见 `llm_set/llm.py`）：

| 用途 | 变量 / 模型 |
|------|-------------|
| VLM | `QWEN_API_KEY`，默认 `qwen-vl-max` |
| llm_TO | `llm`（默认 DeepSeek 文本模型） |
| TopK embedding | `vlm_embedding`（`qwen3-vl-embedding`） |

切换 VLM：修改 `llm_set/llm.py` 中 `vlm = ...`。

---

## Mobile3M 前置

图目录需已生成 SMAN 索引（`all_action_id.json`、`all_page_actions.json` 等）：

```bash
cd datasets/Mobile3M
python3 convert_mobile3m.py datasets/QQmusic_graph_8152 --mode explore
```

---

## Agent 与 TOP_K

| Agent | 说明 |
|-------|------|
| `m2` | 全页候选 → embedding **Top-K** 标注 → VLM 在 **cN/sN 动作空间** 内选 click/scroll/input |
| `to` | 与 m2 共用 `prepare_round`；**非 ambiguous** 步固定 **Top-1** 标注，VLM 主要判 **type**，click 由系统注入 top1 label |
| `toa` | 同 TO 的 Top-1 路由；click 可输出 **element** 覆盖 top1，或 **归一化坐标 (x,y)**；低置信时 prompt 提示可用坐标 |
| `AppAgent` | 官方 **single-path** baseline：全页 SoM（`draw_bbox_multi`）+ 英文 `singlepath_task_template`；VLM 经 `llm.py`（Qwen-VL-Max） |

### `instruction_hit` 与检索 K

`annotate/instruction_hint.py` 的 `infer_instruction_hit` 对**分步描述**（任务 JSON 里 `1. …` `2. …` 各行）做关键词子串匹配，用于候选池 / 标注 / prompt 路由。**不是图语料里的 GT 动作类型**，但与 `main.py` 实际行为一致。

#### 如何判定 `click` / `scroll` / `input` / `ambiguous`

| 结果 | 规则 |
|------|------|
| `click` | 文案中**仅**命中点击类词（如「点击」「点按」「轻点」「tap」等） |
| `scroll` | 文案中**仅**命中滑动类词（如「滑动」「滚动」「向下滑」「scroll」「swipe」等） |
| `input` | 文案中**仅**命中输入类词（如「输入」「填写」「键入」「input」等） |
| `ambiguous` | **同时**命中两类及以上，或**三类都未**命中 |

实现见 `annotate/instruction_hint.py`：`len(matched)==1` 时返回该类，否则为 `ambiguous`。to/toa 在 `scroll` 步若无 scroll 节点，会降为 `ambiguous` 并重跑 Top-K（见下表）。

#### Mobile3M 任务 JSON：`instruction_hit` 逐步分布（参考）

四个任务文件按任务名**去重**后：12654 任务、51605 步、49 个 App。统计脚本：对每条分步 instruction 调用 `infer_instruction_hit`。

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

#### Agent 与检索 K

| Agent | `instruction_hit` | 检索 K | 标注 | `decide` |
|-------|-------------------|--------|------|----------|
| **m2** | `scroll` | —（全标 scroll） | 官方 sN | scroll 专用 prompt |
| **m2** | `click` / `ambiguous` / `input` | `TOP_K`（input 无标） | Top-K 或原图 | Top-K / input prompt |
| **to / toa** | `click` | **1** | Top-1 cN | Top-1 click prompt + force_top1 |
| **to / toa** | `scroll` | —（全标 scroll） | 官方 sN | scroll 专用 prompt |
| **to / toa** | `input` | — | 原图 | input prompt |
| **to / toa** | `ambiguous` | **`TOP_K`** | Top-K 混合 | **回退 m2** Top-K prompt |

说明：

- `TOP_K` 在 `main.py` 传入 agent；**m2 始终使用**（click/ambiguous 路径）；**to/toa 仅在 ambiguous 步使用**。
- **`scroll` 步不做语义 TopK**：全页所有 scroll 区域均标注为 s1、s2、…（官方 `put_btext` 中心黑底白字）；`to/toa` 的 `retrieval_top_k=1` 仅影响 click 路径缓存键，**不改变 scroll 全标逻辑**。
- `scroll` 步若无可用 scroll 节点，管线会将 `instruction_hit` 降为 `ambiguous` 并用 `TOP_K` **重新检索**（to/toa 专用逻辑，不影响 m2）。
- 每步实际检索 K 写入 `assets.retrieval_top_k` 与结果 JSON 的 `top_k` 字段。

### VLM 提示词（`agents/prompts.py`）

按 **`instruction_hit` + 标注方式** 选用两套中文 prompt（均无 `swipe` 字样）：

| 场景 | 标注图 | Prompt 函数 | 要点 |
|------|--------|-------------|------|
| `scroll` | 官方 `put_btext`：全页 sN、中心黑底白字 | `build_*_scroll_prompt` | 说明 `scroll(sN, direction)`；**方向 = 手指在 sN 区域内的拖拽方向**（up/down/left/right） |
| `click` / `ambiguous`（m2 Top-K） | Top-K 子集：cN 红实线+浅填色，sN 蓝虚线 | `build_m2_topk_prompt` | 说明仅展示 Top-K 候选、勿因 click 框多默认 click |
| `input` | 无标注原图 | `build_*_input_prompt` | 仅 output `input` + `text` |
| TO/TOa `click` | Top-1 单框 cN | `build_to_click_top1_prompt` / `build_toa_click_top1_prompt` | click 可不输出 element；TOa 可输出坐标 |

`decide` 时由 `assets.instruction_hit` 路由；**TO/TOa 的 scroll 步不走 click 的 `force_top1` 后处理**；ambiguous 步 to/toa **回退 m2** 的 Top-K prompt。

---

## 推理配置（`main.py` 顶部）

| 变量 | 说明 | 示例 |
|------|------|------|
| `AGENT` | `m2` / `to` / `toa` / `AppAgent` | `to` |
| `TASK_JSON` | 任务列表 | `simple_tasks_sample.json` |
| `TASK_TYPE` | 结果目录前缀 | `single_simple` |
| `TEST_START` / `TEST_END` | 任务切片 `[START, END)`，`END=-1` 到末尾 | `0` / `100` |
| `APP_NAMES` | 按任务名前缀过滤 | `["QQmusic"]` |
| `TOP_K` | m2 语义 Top-K；to/toa 仅 **ambiguous** 步使用 | `5` |
| `DATA_DIR` | Mobile3M datasets 路径 | `../../datasets/Mobile3M/datasets` |
| `DRY_RUN` | `True` 不调 VLM，用 mock 动作 | `False` |
| `REQUEST_INTERVAL` | 步间休眠秒数 | `0` |
| `TYPE` | 仅推理 GT 类型在此列表中的步；其余步 **SKIP**（`is_skip=true`，仍沿 GT 推进） | `["click"]` / `["scroll"]` / `["click","scroll"]` |

结果根目录：`results/{TASK_TYPE}_{AGENT}_{model_slug}/`  
按本任务实际出现的 GT 类型写入子目录：`click/`、`scroll/`、`input/` 等。  
单任务文件：`{task_index}_{final_page_name}.json`

---

## 运行推理

```bash
python main.py
```

试跑（无 API）：`main.py` 中设 `DRY_RUN = True`、`TEST_END = 1`。

---

## 单步流水线

### 标注路由（`annotate/topk.py`）

| `instruction_hit` | 候选 / 检索 | 标注方式 |
|-------------------|-------------|----------|
| `scroll` | 全页 scroll 区域（不跑 embedding TopK） | `render_scroll_only_labeled_image`（官方 put_btext） |
| `click` | click 池 → embedding Top-K | `annotate.py`（红实线 cN） |
| `ambiguous` | click+scroll 混合池 → Top-K | `annotate.py`（cN 红实线 + sN 蓝虚线） |
| `input` | 无候选 | 原图，不画标 |

### M2（及 to/toa 的 ambiguous 步）

```
分步 instruction
    → infer_instruction_hit（候选池路由）
    → llm_TO：click/ambiguous 步生成中文 target_object
    → SMAN 候选（按 hint 选 click 池 / scroll 全标 / 全池 Top-K）
    → qwen3-vl-embedding Top-K 选节点（scroll 步跳过）
    → 画标注图（scroll：官方 sN；Top-K：cN 红实线 + sN 蓝虚线）
    → VLM（Top-K prompt 或 scroll 专用 prompt）→ parse_labeled_json
    → apply_action（cN/sN → 全局 action_id）
    → 与 GT 对比；页面沿 GT 路径强制推进
```

### TO（非 ambiguous 步）

```
prepare_round（click：Top-1 标注；scroll：全页官方 sN；input：原图）
    → VLM（按 instruction_hit 选 click / scroll / input prompt）
    → click：强制 top_k_nodes[0].label，不要求 VLM 输出 element
    → scroll：输出 sN + direction（不走 force_top1）
    → input：输出 text
```

### AppAgent（官方 single-path baseline）

```
分步 instruction + 整体 task_desc
    → 全页官方 SoM（draw_bbox_multi：cN + sN，1080×2400）
    → 英文 singlepath_task_template（含 last_act Summary 多轮上下文）
    → VLM（llm.py Qwen-VL-Max）→ parse_explore_rsp
    → apply_action → 与 GT 对比
```

要点：

- 入口：`AGENT = "AppAgent"`；每任务调用 `begin_task(ctx)` 重置 `last_act`。
- 不走 TopK / llm_TO / 中文 JSON prompt。
- `eval.py` 自动扫描 `results/{TASK_TYPE}_AppAgent_{model}/`；对比表中 **Mean_Retrieval** 对 AppAgent 显示 `n/a`。

### TOa（非 ambiguous 步）

```
同 TO 的 prepare_round
    → click 步：读取检索 top1 分数与 margin，写入 Top-1 prompt
    → scroll/input 步：专用 prompt，不加载检索分数
    → click：信任 top1 / element / 归一化 (x,y)
    → apply_action 将坐标映射到最近 SMAN click 区域
    → 额外记录 locator_source、action_match_geom（仅 click 坐标路径）
```

要点：

- **页面推进**：无论预测对错，每步结束后跳到 GT 路径的下一页，保证多步任务能跑完。
- **VLM 动作空间**：`click` / `scroll` / `input`；scroll 输出 `sN` + `direction`（方向为手指拖拽方向，见 prompt）。
- **parse 重试**：解析失败最多重试 2 次；仍失败记 `pred_id=-2`、`parse_error`。
- **GT 显示**：`utils/action_id_resolve.py` 将 `action_id` 反解为 `click(cN, 文本)` / `scroll(sN, dir)`。

---

## 终端 Step 日志

```
Step 2/4 | QQmusic0_3 | "在听书页面，向下滚动…"
  TO="听书" | Top_5_Retrieval=true
  GT=scroll(s13, up) | pred=scroll(s25, down) | Type=true | Action=false
```

| 字段 | 含义 |
|------|------|
| `TO` | llm_TO 生成的 `target_object`（click/ambiguous 步） |
| `Top_K_Retrieval` | GT 的 cN/sN 是否在当步 TopK 标注子集中（K 为当步实际 `retrieval_top_k`） |
| `Type` | 预测动作类型与 GT 一致 |
| `Action` | `pred_action_id == gt_action_id` |

**click 补充**：同一页上不同 **cN** 可能对应同一 **action_id**（候选池里重复/多条目指向同一控件）。`action_acc` 按 id 比对；`Mean_Retrieval` 按 GT 反查的 cN 是否在 Top-K 标注子集里比对，故偶发 **action 对、retrieval 错**（TO 强制 top1 时更明显）。

---

## 标注与节点过滤

### 两套标注视觉（与 prompt 对应）

| 样式 | 用于 | 实现 |
|------|------|------|
| **官方 scroll** | `instruction_hit=scroll` | `utils/sman_bridge.render_scroll_only_labeled_image`：中心黑底白字 sN，无彩色边框 |
| **工程化 Top-K** | click / ambiguous 的 Top-K 子集 | `annotate/annotate.py`：cN 红实线+浅填色，sN 蓝虚线无填色，标签可偏移 |

### 过滤规则

| 阶段 | 规则 |
|------|------|
| **TopK 前** `dedupe_nodes_by_bounds` | 相同 `(kind, bounds)` 只保留第一个 |
| **TopK 后** `filter_nodes_for_annotation` | 嵌套 scroll：大框包含小框且面积比 ≥10 时删大框；极小 click 仅改标签位置 |

标注图与 `cache/nodes/` 使用**过滤后**的展示子集；GT 反查使用**全页** `scroll_action_bounds`（二者可能不一致，导致 `Top_K_Retrieval=false` 但 GT 仍能显示 `sN`）。**scroll 专用路径全标所有 scroll 区域，不受 TopK 裁剪。**

---

## 评测配置（`eval.py` 顶部）

| 变量 | 说明 |
|------|------|
| `TASK_TYPE` | 与 `main.py` 一致 |
| `MODEL` | VLM 模型名（`model_slug`，用于扫描 `results/{TASK_TYPE}_*_{MODEL}/`） |
| `DATA_DIR` | Mobile3M 路径 |
| `TYPE` | 评测范围（见下表） |
| `RESULT_DIR` | 可选，直接指定某一 agent 的结果目录 |

`eval.py` **不再绑定单个 AGENT**：对同一 `MODEL` 下已推理的 `m2` / `to` / `toa` 结果目录自动扫描，终端打印**对比表**。

### `TYPE` 评测模式

| `TYPE` | 行为 |
|--------|------|
| `["click"]` | 只评 `results/.../click/` 下任务；计分步仅含 GT 为 click 的步 |
| `["scroll"]` | 只评 `scroll/` 子目录 |
| `["click", "scroll"]` | **分目录各算一行**（不合并 SR） |
| `["all"]` | 合并各类型子目录，按**整条任务**算统一 SR；缺步任务跳过且不计分母（`incomplete_task_count`） |

须与推理时的 `TASK_TYPE`、`MODEL`（及 `TYPE` 子目录布局）一致。

---

## 运行评测

```bash
python eval.py
```

指标（逐步对齐 GT `action_id`）：

| 指标 | 含义 |
|------|------|
| `success_rate` | 任务内参与计分步全部 `pred_id == gt_id` 的任务占比 |
| `action_acc` | 全局步级 action_id 准确率 |
| `type_acc` | 全局步级动作类型准确率 |
| `Mean_Retrieval` | GT 的 cN/sN 落在当步 TopK 标注子集的比例 |
| `action_acc_geom` | 仅 TOa 坐标 click 步：归一化点落入 GT bbox（或中心距离阈值内）的比例 |

终端对比表与 `checkpoints/eval_{TASK_TYPE}_{MODEL}_agents_{timestamp}.json` 均带 `( 命中 / 总数 )`。

---

## 原始 GT 动作格式（Mobile3M）

数据集**不存** `cN/sN`，存全局动作字符串：

| 类型 | 格式示例 |
|------|----------|
| click | `click(<p id="..." clickable="true"> 听书 </p>)` |
| scroll | `scroll(([360, 817], [360, 429]))` |
| input | `input(<input ... EditText>, '文本')` |

`cN/sN` 由当页 HTML/XML 生成的候选列表 + `action_id` 反查得到。详见 `SMAN-Bench/接口.md`。

---

## 注意事项

- 在工程根目录运行 `main.py` / `eval.py`。
- 不向 Mobile3M 写入推理产物。
- 修改 `TOP_K`、标注或 filter 后，删除对应 `cache/labeled/`、`cache/nodes/`、`cache/retrieval/` 再重跑。
- 本工程为 **single-path**，支持 **m2 / to / toa / AppAgent**；Multi-path、reflection、旧版 m123/baseline 已移除。官方 Multi-path 说明见 `SMAN-Bench/运行说明.md`。
