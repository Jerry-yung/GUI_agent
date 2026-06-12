# Mobile3M_data

Mobile3M 单步样本数据目录，与 `target/`、`judge/`、`agents/` 中的脚本配合使用。

## 命名约定

每个样本以 **stem** 标识：`{task_name}_{step_id:03d}`

示例：`QQmusic0_34_347_000` 表示任务 `QQmusic0_34_347` 的第 0 步（在 `QQmusic0` 页面上点击）。

同一 stem 对应四件套：

| 文件 / 目录 | 扩展名 |
|-------------|--------|
| `step_GT/{stem}.json` | `.json` |
| `step_instructions/{stem}.txt` | `.txt` |
| `screenshots/{stem}.png` | `.png` |
| `a11y_trees_L0/{stem}.xml` | `.xml` |

## 生成方式

```bash
# 前置：对每个 App graph 生成索引（只需一次）
cd datasets/Mobile3M
python3 convert_mobile3m.py datasets/QQmusic_graph_8152 --mode explore

# 转换四件套
cd GUI_agent/test4.0.1-SMB_query_emb
python get_Mobile3M.py
python get_Mobile3M.py --app-prefix QQmusic --incremental   # 仅补跑新样本，保留已有四件套
```

`gt_bounds` 匹配分级（`meta.bounds_match_tier`）：`clickable_attr` → `interactive_container` → `label_bounds`（Tab/底栏等 clickable=false 控件）。

转换报告：`conversion_report.json`（成功 / 跳过 / 失败明细）。

## GT 生成原理

### 1. 单步拆分逻辑

任务名如 `QQmusic0_3_98_1071` 会被拆成页面轨迹：

```
["QQmusic0", "QQmusic0_3", "QQmusic0_3_98", "QQmusic0_3_98_1071"]
```

对每一对 **父页 → 子页**（例如 `QQmusic0_3 → QQmusic0_3_98`），生成一个单步样本（stem）。

### 2. GT Action 的来源

加载**子页**的 JSON，取其 `history_actions` 的**最后一个元素**：

```python
gt_action = child_json["history_actions"][-1]
```

**为什么取最后一个？**

`history_actions` 记录的是从根页面到达当前页面的**完整动作序列**。例如 `QQmusic0_3_98`：

| 页面 | `history_actions` |
|------|------------------|
| `QQmusic0` | `[]` |
| `QQmusic0_3` | `[action_3]` |
| `QQmusic0_3_98` | `[action_3, action_98]` |
| `QQmusic0_3_98_1071` | `[action_3, action_98, action_1071]` |

`history_actions[-1]` 永远是**刚刚执行的那一步**——也就是从父页转移到当前子页的那个动作。前面的元素是更早的祖先转移动作。

> 如果 `gt_action` 不是 `click(...)`（如 `scroll(...)`），该步骤会被跳过，不生成四件套。

### 3. GT Bounds 的来源

`gt_bounds` 不是从 JSON 中直接读取的，而是通过在**父页 XML** 中匹配 `gt_action` 的属性动态计算得出。

匹配流程（三级 fallback）：

```
Tier 1: clickable_attr
  └─ 找 clickable=True 且 id/text/description/class/package 完全匹配的节点
     └─ 命中 → 返回该节点 bounds

Tier 2: interactive_container
  └─ 先找属性匹配的标签节点（不管 clickable 值）
  └─ 取该标签的中心点 (cx, cy)
  └─ 找包含 (cx, cy) 的最小「interactive 祖先」
     （interactive = clickable / focusable / checkable / long-clickable / scrollable）
  └─ 命中 → 返回该祖先的 bounds

Tier 3: label_bounds
  └─ 如果连 interactive 祖先也没有
  └─ 直接返回标签节点自身的 bounds
```

**为什么需要 Tier 2/3？**

Mobile3M 数据集中大量导航控件（顶部 Tab `id=ihw`、底部导航 `id=e60`）在 XML 中自身是 `clickable="false"`，但其父容器是 `clickable="true"` 或 `focusable="true"`。旧版本只匹配 Tier 1 时，这些合法点击全部失败；新版本通过 Tier 2 的**中心点容器推断**解决了这个问题。

最终 `step_GT/{stem}.json` 中会写入 `meta.bounds_match_tier`，标记实际命中了哪一级。

### 4. Instruction 的来源

从任务 JSON 的 `task` 字段中提取：

1. 用正则 `^\d+\.\s(.*)` 提取「分步描述」中的编号步骤列表
2. 第 `k` 步（第 `k` 次页面转移）对应 `step_instructions[k]`
3. 如果编号步骤不足，回退到「简洁任务」或原始 `task` 文本

### 5. 输入状态的选择

每个单步样本的**输入状态**（screenshot + XML）取自**父页**，而不是子页。因为模型应当根据「动作前的页面」预测点击位置。

## 使用流水线

```bash
python process/compress.py
python process/get_emb.py
python target/llm_target.py
python process/get_GT_nearest_K.py
python judge/cal_cos_sim.py
```

## step_GT 格式

```json
{
  "action_type": "click",
  "gt_action": "click(<p id=\"...\"> 歌手 </p>)",
  "gt_bounds": [40, 716, 152, 855],
  "gt_node_id": 13,
  "x": 96.0,
  "y": 785.5,
  "nearest_5": [{"node_id": 13, "bounds": [...]}]
}
```

- `get_Mobile3M.py` 写入 `gt_action` + `gt_bounds`（XML 最小匹配框）
- `compress` 之后运行 `get_GT_nearest_K.py` 写入 `gt_node_id`、`x/y`、`nearest_5`（单框）

## `_deleted/` 归档规则

由 `process/stem_deleted.py` 在 `compress.py` 中调用。

**触发条件**：`compress.py` 压缩后 **可交互节点数 = 0**（或历史上 `nodes/{stem}.json` 为空列表 `[]`）。

**操作**：

1. 将以下四件套移入 `Mobile3M_data/_deleted/{子目录}/`（保持原子目录名）：
   - `compressed_a11y/{stem}.json`
   - `screenshots/{stem}.png`
   - `step_GT/{stem}.json`
   - `step_instructions/{stem}.txt`
2. **不生成 / 删除** 若已存在的衍生产物：
   - `nodes/`、`target/target_object/`、`target/TO_index/`
   - `embeddings/nodes_emb/{stem}/`、`embeddings/TO_emb/{stem}/`、`embeddings/cos_sim/{stem}.npz`

`a11y_trees_L0` 仍保留在原位，便于排查；该 stem 不再参与后续 embedding、检索与 Agent 评测。

手动清理历史空 nodes 样本：

```python
from process.stem_deleted import purge_stems_with_empty_nodes
purge_stems_with_empty_nodes()
```

## 数据流（本目录视角）

```
a11y_trees_L0  ──compress.py──►  compressed_a11y
                                      │
                         可交互节点>0 ▼
                                 nodes/*.json
                                      │
              screenshots ──get_emb.py──► embeddings/nodes_emb/

step_instructions ──llm_target.py──► embeddings/TO_emb/
                                      (TO_index 在 target/)

nodes_emb + TO_emb ──cal_cos_sim.py──► embeddings/cos_sim/

step_GT + nodes ──get_GT_nearest_K.py──► step_GT.gt_node_id

cos_sim + nodes + screenshots ──annotate.py──► agents/annotate/annotated_screenshots/
```
