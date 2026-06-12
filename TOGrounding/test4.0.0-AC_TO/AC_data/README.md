# AC_data

Android Control 单步样本数据目录。与 `target/`、`judge/`、`agents/` 中的脚本配合使用。

## 命名约定

每个样本以 **stem** 标识：`{episode_id}_{step_id:03d}`

示例：`00000020_001` 表示 episode `00000020` 的第 1 步（从 0 起编）。

同一 stem 在下列目录中一一对应（四件套 + 衍生数据）：

| 文件 / 目录 | 扩展名 |
|-------------|--------|
| `step_GT/{stem}.json` | `.json` |
| `step_instructions/{stem}.txt` | `.txt` |
| `screenshots/{stem}.png` | `.png` |
| `a11y_trees_L0/{stem}.json` | `.json` |

## 目录结构

```
AC_data/
├── step_GT/                  # 单步 GT 动作（含 nearest_5）
├── step_instructions/        # 单步自然语言指令
├── screenshots/              # 当前步截图
├── a11y_trees_L0/            # 原始无障碍树（L0）
├── compressed_a11y/          # 压缩后的 a11y（中间产物）
├── nodes/                    # 可交互 UI 节点列表
├── embeddings/
│   ├── nodes_emb/            # 节点 embedding
│   ├── TO_emb/               # Target Object 文本 embedding
│   └── cos_sim/              # 节点×TO 余弦相似度矩阵
└── _deleted/                 # 无效样本归档（见下文）
    ├── compressed_a11y/
    ├── screenshots/
    ├── step_GT/
    └── step_instructions/
```

## 各目录说明

### `step_GT/{stem}.json`

单步 ground truth，由 `get_AC.py` 从原始 `GT/*.json` 拆分。

```json
{
  "action_type": "click",
  "x": 110,
  "y": 506,
  "nearest_5": [
    {"node_id": 2, "bounds": [10, 438, 210, 588]},
    ...
  ]
}
```

- `x`, `y`：GT 点击坐标（像素）
- `nearest_5`：由 `process/get_GT_nearest_K.py` 写入，为距 GT 点中心最近的 5 个 node 及其 bbox

### `step_instructions/{stem}.txt`

该步的用户指令文本，供 `target/llm_target.py` 生成 Target Object，以及 `main.py` 中 Baseline / M2 Agent 评测。

### `screenshots/{stem}.png`

当前步界面截图：

- `process/get_emb.py`：crop 节点 patch
- `agents/baseline_agent.py`：Baseline 输入
- `agents/annotate/annotate.py`：生成 M2 标注图底图

### `a11y_trees_L0/{stem}.json`

原始 Android 无障碍树（JSON 格式），`process/compress.py` 的输入。

### `compressed_a11y/{stem}.json`

`process/compress.py` 输出：AGER V0 压缩后的节点列表（含可交互与语义节点）。

### `nodes/{stem}.json`

**可交互** UI 节点列表（`compress.py` 在压缩结果上再筛 `clickable` 等），评测与 embedding 的节点空间。

```json
[
  {
    "node_id": 0,
    "bounds": [956, 157, 1072, 272],
    "text_for_emb": "scan it"
  }
]
```

- `bounds`：`[x1, y1, x2, y2]` 像素坐标
- `text_for_emb`：用于 multimodal embedding 的文本

若压缩后**可交互节点数为 0**，则**不保留**本文件，四件套移入 `_deleted/`（见下文）。

### `embeddings/nodes_emb/{stem}/{stem}_{node_id}.npy`

节点 embedding 向量（dim=2560），由 `process/get_emb.py` 生成。

策略：multimodal(text+patch) → patch only → text only → zero fallback。  
`nodes` 为空时跳过，不生成 embedding。

### `embeddings/TO_emb/{stem}/{stem}_{TO_id}.npy`

Target Object 文本 embedding，由 `target/llm_target.py` 生成。  
`TO_id` 定义见 `target/TO_index/{stem}.json`。已存在的 `.npy` 会跳过（增量缓存）。

### `embeddings/cos_sim/{stem}.npz`

由 `judge/cal_cos_sim.py` 生成：

| 字段 | shape | 说明 |
|------|-------|------|
| `sim_matrix` | `(num_nodes, num_TOs)` | 余弦相似度 |
| `node_ids` | `(num_nodes,)` | 行对应的 node_id |
| `to_ids` | `(num_TOs,)` | 列对应的 TO_id |

`nodes` 或 `TO_index` 为空时无法生成，对应 stem 会被 `cal_cos_sim.py` 跳过。

## `_deleted/` 归档规则

由 `process/stem_deleted.py` 在 `compress.py` 中调用。

**触发条件**：`compress.py` 压缩后 **可交互节点数 = 0**（或历史上 `nodes/{stem}.json` 为空列表 `[]`）。

**操作**：

1. 将以下四件套移入 `AC_data/_deleted/{子目录}/`（保持原子目录名）：
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

step_GT + nodes ──get_GT_nearest_K.py──► step_GT.nearest_5

cos_sim + nodes + screenshots ──annotate.py──► agents/annotate/annotated_screenshots/
                                              (供 M2 Agent，路径在项目 agents/ 下)
```

## 预处理说明

`get_AC.py`（项目根目录）会：

1. 将 `GT/*.json` → `step_GT/*.json`
2. 将 `instructions/*.json` → `step_instructions/*.txt`
3. 仅保留 `action_type ∈ {click, long_press}` 的四件套，删除其余 stem

`compress.py` 启动时会自动扫描并清理历史上 `nodes/*.json` 为空的 stem。

## 与评测的关系

| 数据 | 用途 |
|------|------|
| `cos_sim/*.npz` | 检索 top 节点；M2 标注 top_k 候选 |
| `step_GT.nearest_5` | 检索 judge hit@k；M2 Agent `judge_m2` |
| `step_GT.x/y` + `nodes.bounds` | `hit_1` 几何 EM；Baseline `judge_baseline` |
| `screenshots` | Baseline 输入；标注图底图；归一化距离计算 |
| `step_instructions` | LLM 生成 TO；Agent 指令输入 |

| 评测类型 | 位置 | 说明 |
|----------|------|------|
| 检索 hit@k | `judge/judge.py` | cos_sim 检索 node_id |
| 检索 top_1 EM | `judge/hit_1.py` | 检索节点中心几何 |
| Baseline Agent | `main.py` + `judge/judge_llm.py` | VLM 坐标几何 EM |
| M2 Agent | `main.py` + `judge/judge_llm.py` | VLM node_id hit@1 |

详细 Agent 流程见项目根 [README.md](../README.md)。
