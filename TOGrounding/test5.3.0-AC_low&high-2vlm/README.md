# test5.3.0-AC_low&high_mem

Android Control（AC）全步轨迹实验：Target Object 检索 + 节点标注 + VLM 预测，支持 **AC-low** / **AC-high**。

- 数据说明：[AC_data/README.md](AC_data/README.md)
- Agent 说明（m2 / m2p / TO / CPM）：[agents/README.md](agents/README.md)
- **评测机制**：[eval/README.md](eval/README.md)

---

## 目录概览

```
test5.3.0-AC_low&high_mem/
├── get_AC.py
├── process/                  # 压缩 a11y、nearest_5、节点 embedding
├── annotate/                 # llm_TO、top_k 检索、画标注图
├── agents/                   # VLM Agent
├── llm_set/llm.py
├── main.py                   # 实验入口
├── eval/
├── runs/
├── 对比.ipynb
└── AC_data/
```

---

## 在线推理（`main.py`）

每步：`llm_TO（或 m2p planner）→ top_k 检索 → 标注图 → VLM → 评测`。节点 embedding 读离线文件；TO embedding 在线计算不落盘。

---

## AC-low vs AC-high

| | AC-low | AC-high |
|---|--------|---------|
| VLM 文本 | `goal` + step `instruction` | 仅 `goal`（m2）；m2p 用 planner + history |
| 本步 type | 每步 **llm_TO** | m2：step0 llm_TO，之后上步 VLM 三字段；**m2p：每步 planner** |
| VLM 额外输出 | 无 | m2 high：`next_instruction` 等三字段 |

---

## 运行

```bash
python get_AC.py
python process/compress.py
python process/get_GT_nearest_K.py
python process/get_emb.py

# 编辑 main.py：AC_MODE、AGENT、TOP_K、TEST_START/END
python main.py
```

结果：

- m2 / TO：`runs/{AC_MODE}_{agent}_top{TOP_K}{to_select}_{vlm_model}.json`
- m2p：`runs/high_m2p_top{TOP_K}{to_select}_{vlm_model}.json`
- CPM：`runs/{AC_MODE}_CPM_{vlm_model}.json`

```bash
python eval/eval_run.py runs/low_m2_top5generate_qwen-vl-max.json
```

---

## 评估指标

| 指标 | 简述 |
|------|------|
| **Type Acc** | 动作类型是否与 GT 一致 |
| **Step Acc** | 类型正确且该类型字段判定通过 |
| **SR** | episode 内全部可评测步均正确 |
| **Top-K Retrieval Hit** | click/long_press：top_k 是否命中 `nearest_5` |

**不计入评测**：GT 为 `open_app` 的步。

详见 [eval/README.md](eval/README.md)。
