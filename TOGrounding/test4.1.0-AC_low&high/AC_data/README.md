# AC_data

`test4.1.0-AC_high` 项目的 Android Control（AC）单步样本数据目录。每个 step 的核心文件集中在同一目录下，便于按轨迹（episode）组织与扩展。

## 数据来源

原始数据来自仓库内 Android Control 数据集：

```
GUI/datasets/AndroidControl/data/
├── GT/                 # episode 级 ground truth（actions 列表）
├── instructions/       # episode 级逐步自然语言指令
├── screenshots/        # 单步截图，{stem}.png
└── a11y_trees_L0/      # 单步无障碍树，{stem}.json
```

由项目根目录的 `get_AC.py` 读取上述原始数据，拆分并写入本目录 `AC_data/`。**不修改、不依赖** `datasets/` 以外的副本；`get_AC.py` 通过相对路径自动定位 `datasets/AndroidControl/data/`。

与 4.0.0 的主要区别：

| 项目 | 4.0.0 | 4.1.0（本目录） |
|------|-------|----------------|
| 动作筛选 | 仅 `click` / `long_press` | **保留全部** action_type |
| 目录结构 | 扁平四目录（`step_GT/`、`screenshots/` 等） | `steps/{episode_id}/{step_idx}/` |
| 样本规模 | ~2510 step | **784 episode，4273 step** |

## 命名约定

- **episode_id**：8 位数字，如 `00000020`
- **step_idx**：从 0 起的 3 位步序号，如 `001`
- **stem**：`{episode_id}_{step_idx:03d}`，如 `00000020_001`

路径示例：

```
steps/00000020/001/00000020_001_gt.json
```

## 目录结构

```
AC_data/
├── manifest.json           # 全局索引（step 列表、action 统计）
├── episodes/               # episode 级元数据
│   └── {episode_id}.json
├── steps/                  # 单步样本（核心）
│   └── {episode_id}/
│       └── {step_idx:03d}/
│           ├── {stem}_gt.json
│           ├── {stem}_instruction.txt
│           ├── {stem}_screenshot.png
│           ├── {stem}_a11y.json
│           ├── {stem}_meta.json
│           ├── {stem}_compressed_a11y.json   # process/compress.py
│           └── {stem}_nodes.json             # process/compress.py
└── embeddings/
    └── nodes_emb/            # process/get_emb.py
        └── {episode_id}/
            └── {step_idx:03d}/
                └── {stem}_{node_id}.npy
```

## 各文件说明

### `manifest.json`

`get_AC.py` 生成。记录 schema 版本、episode/step 总数、各 `action_type` 数量及完整 `stems` 列表。下游脚本优先用此文件枚举样本。

### `episodes/{episode_id}.json`

单个 episode 的轨迹摘要：

```json
{
  "episode_id": "00000020",
  "goal": "Go to the mobile category ...",
  "num_steps": 4,
  "stems": ["00000020_000", "00000020_001", "..."],
  "action_types": ["navigate_back", "click", "click", "scroll"]
}
```

### `steps/.../{stem}_gt.json`

该步 ground truth 动作（与原始 AC `actions[i]` 一致）。`click` / `long_press` 等含坐标动作，经 `process/get_GT_nearest_K.py` 可追加 `nearest_5`：

```json
{
  "action_type": "click",
  "x": 110,
  "y": 506,
  "nearest_5": [
    {"node_id": 2, "bounds": [10, 438, 210, 588]}
  ]
}
```

### `steps/.../{stem}_instruction.txt`

该步自然语言指令（纯文本，一行）。

### `steps/.../{stem}_screenshot.png`

执行该步动作**之前**的界面截图（与原始 `screenshots/{stem}.png` 相同内容）。

### `steps/.../{stem}_a11y.json`

原始 Android 无障碍树（L0 JSON），`process/compress.py` 的输入。

### `steps/.../{stem}_meta.json`

步级元数据，便于不打开 GT 即可获知 `action_type`、截图尺寸等：

```json
{
  "schema_version": "4.1.0",
  "stem": "00000020_001",
  "episode_id": "00000020",
  "step_idx": 1,
  "action_type": "click",
  "episode_goal": "...",
  "screenshot_size": {"width": 1080, "height": 2400},
  "num_steps_in_episode": 4
}
```

### `steps/.../{stem}_compressed_a11y.json`

`process/compress.py` 输出：AGER V0 压缩后的节点列表。可交互节点为 0 时仍写入（可能为空列表 `[]`），**不删除样本**。

### `steps/.../{stem}_nodes.json`

`process/compress.py` 输出：可交互 UI 节点列表，供 embedding 与 grounding 使用：

```json
[
  {
    "node_id": 0,
    "bounds": [956, 157, 1072, 272],
    "text_for_emb": "scan it"
  }
]
```

`node_id` 为可交互节点在该列表中的索引。列表可为空 `[]`。

### `embeddings/nodes_emb/{episode_id}/{step_idx}/{stem}_{node_id}.npy`

`process/get_emb.py` 生成的节点 embedding 向量（dim=2560）。已存在的 `.npy` 会跳过（支持增量与路径迁移后的复用）。

## action_type 分布

全量 4273 step（见 `manifest.json`）：

| action_type | 数量 | GT 示例 |
|-------------|------|---------|
| click | 2603 | `{"action_type": "click", "x": 110, "y": 506}` |
| scroll | 595 | `{"action_type": "scroll", "direction": "down"}` |
| input_text | 312 | `{"action_type": "input_text", "text": "..."}` |
| open_app | 302 | `{"action_type": "open_app", "app_name": "Papago"}` |
| wait | 280 | `{"action_type": "wait"}` |
| navigate_back | 174 | `{"action_type": "navigate_back"}` |
| long_press | 6 | `{"action_type": "long_press", "x": 571, "y": 1714}` |
| navigate_home | 1 | `{"action_type": "navigate_home"}` |

## 数据处理流水线

在项目根目录 `test4.1.0-AC_high/` 下推荐顺序：

```bash
# 1. 从 datasets/AndroidControl/data 拆分单步样本
python get_AC.py

# 2. 压缩 a11y → compressed_a11y + nodes（写入各 step 目录）
python process/compress.py

# 3. 为含坐标的 GT 写入 nearest_5
python process/get_GT_nearest_K.py

# 4. 节点 multimodal embedding（按 episode 切片，见 get_emb.py 内 TEST_START/END）
python process/get_emb.py
```

路径解析统一见 `process/paths.py`（`step_paths(stem)`、`node_emb_path(stem, node_id)` 等）。

## 实验流水线（AC-low / AC-high）

在数据预处理（`get_AC` → `compress` → `get_emb`）完成后，运行 `main.py` 进行 step0 实验：

| 模式 | `main.AC_MODE` | Agent 可见文本 | TO 生成输入 |
|------|----------------|---------------|------------|
| AC-low | `"low"` | 当前 step instruction | 同上 |
| AC-high | `"high"` | episode `goal` | 同上 |

**step0 流程（每个 episode 的第一步）：**

1. `annotate/llm_TO.py` — 生成单条 `target_object`（英文，不落盘 embedding）
2. `annotate/cos_sim_topk.py` — TO 与 `nodes_emb` 余弦相似度，取 `TOP_K`
3. `annotate/annotate.py` — 标注截图 → `annotate/annotated_screenshots/top_{K}/{episode_id}/{step_idx}.png`
4. `agents/m2_agent.py` / `agents/m2v_agent.py` — VLM 读标注图 + 用户文本，输出 AC 格式 action

**配置**（`main.py` 顶部）：`AC_MODE`、`TOP_K`、`TEST_START`/`TEST_END`（episode 粒度）

**结果**：`runs/ac_{mode}_step0_{vlm_model}.json`

## 路径速查

| 用途 | 路径 |
|------|------|
| 枚举全部 stem | `manifest.json` → `stems`，或 `process/paths.iter_stems()` |
| 枚举 episode | `episodes/*.json`，或 `process/paths.iter_episode_ids()` |
| 单步 GT | `steps/{episode_id}/{step_idx}/{stem}_gt.json` |
| 单步截图 | `steps/{episode_id}/{step_idx}/{stem}_screenshot.png` |
| 可交互节点 | `steps/{episode_id}/{step_idx}/{stem}_nodes.json` |
| 节点 embedding | `embeddings/nodes_emb/{episode_id}/{step_idx}/{stem}_{node_id}.npy` |
