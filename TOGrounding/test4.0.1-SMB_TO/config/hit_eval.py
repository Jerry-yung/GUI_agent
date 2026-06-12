"""Hit 评估配置（默认宽松模式，对齐 Mobile3M / AgentCPM-GUI 默认 evaluator）。

严格模式（参考，本仓库未启用）：
  - 候选：距 GT 中心最近的 5 框
  - 距离阈值：0.04

当前默认（宽松）：
  - GT：gt_node_id（nearest_5 仅含 1 框）
  - 几何兜底：与 GT 中心归一化距离 < 0.14
"""

DISTANCE_THRESHOLD = 0.14
