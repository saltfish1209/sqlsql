# CrossEncoder 实验与出图数据（CSV）

所有结果在 `figure/crossencoder/results/`，**一类实验一个 CSV**。大模型权重在 `checkpoints/`（不提交 git）。

## 三阶段训练流程

```text
Phase 1 — 剥离推理参数
  训练评估使用 NDCG@6 作为北极星指标，topk 仅在推理/部署阶段使用

Phase 2 — 探索损失空间（margin × lambda 网格搜索）
  python figure/crossencoder/run_grid_sweep.py
  默认 3×3 正交网格: margin=[0.05, 0.15, 0.3] × lambda=[0.3, 0.7, 1.0]
  固定 epoch=5，固定 LR，用 NDCG@6 选出最优 (margin, lambda)

Phase 3 — 动态确定训练时长（Early Stopping）
  python figure/crossencoder/run_train_track.py
  自动读取 grid_sweep.csv 的最优参数，max_epochs=10，
  NDCG@6 连续 2 个 epoch 不提升则停止，回滚到最佳权重
```

## 目录

```
figure/crossencoder/
  README.md
  paths.py                 # CSV 路径常量 + 网格默认值
  k_selection.py           # 推理 Top-K 选取规则
  run_gold_stats.py        # gold 列数统计 → gold_column_stats.csv
  run_grid_sweep.py        # Phase 2: margin × lambda 网格 → grid_sweep.csv
  run_train_track.py       # Phase 3: Early Stopping 训练 → train_loss.csv + epoch_val.csv
  run_topk_curve.py        # 推理: 多档 K → topk_curve.csv
  run_pick_topk.py         # 推理: 从曲线选 K → topk_choice.csv
  run_baseline_compare.py  # 基座 vs 微调 → baseline_vs_finetuned.csv
  results/
  checkpoints/
```

## 推荐流程

```text
1. python figure/crossencoder/run_gold_stats.py            # 统计 gold 列分布
2. python figure/crossencoder/run_grid_sweep.py             # Phase 2: 3×3 网格搜索
3. python figure/crossencoder/run_train_track.py            # Phase 3: 最终训练 + 早停
4. python figure/crossencoder/run_baseline_compare.py       # 可选: 对比基座 vs 微调
5. python figure/crossencoder/run_topk_curve.py             # 推理: 多档 K 曲线
6. python figure/crossencoder/run_pick_topk.py              # 推理: 选部署 K
```

画图：用 `results/train_loss.csv`（横轴 `global_step`）+ `epoch_val.csv`（每 epoch 一点 NDCG/MRR/Recall）。

## 指标说明

| 指标 | 阶段 | 含义 |
|------|------|------|
| **NDCG@6** | 训练北极星 | 归一化折损累积增益，衡量 gold 列排序质量 |
| **MRR** | 训练参考 | 第一个 gold 列的倒数排名 |
| **Recall@K** | 推理/部署 | gold 列全部落在 Top-K 内的比例 |

### 训练 vs 推理参数分离

| 参数 | 阶段 | 说明 |
|------|------|------|
| `margin` | 训练 | 损失函数 pair margin，Phase 2 网格搜索 |
| `lambda` | 训练 | pair loss 权重，Phase 2 网格搜索 |
| `epoch` | 训练 | 由 Early Stopping 动态终止，不作为扫参变量 |
| `top_k` | 推理 | 业务/推理策略参数，不参与训练扫参 |

### `epoch_val.csv` 字段说明

| 列名 | 含义 |
|------|------|
| `ndcg` | **NDCG@K**：gold 列排序质量（训练北极星） |
| `mrr` | **MRR**：第一个 gold 列的倒数排名 |
| `recall_at_k` | **Recall@K**：gold 列是否全部落在 Top-K 内（推理参考） |
| `is_best` | 本 epoch 的 NDCG 是否为目前最高（1/0） |
| `patience_counter` | 自上次创新高以来，连续多少个 epoch 未创新高 |
| `stopped_early` | 本 epoch 评估后是否触发早停（1/0） |

### chosen_k_deploy 规则（`run_pick_topk.py`）

在 `K >= k_primary` 的档位中，取满足
`Full Recall@K >= 0.95 × max_recall(曲线)` 的 **最小 K**。
若无满足则退回 recall 最高的 K。

## CSV 一览

| 文件 | 内容 |
|------|------|
| `train_loss.csv` | 每 20 step：loss / point / pair |
| `epoch_val.csv` | 每 epoch 结束：NDCG / MRR / Recall@K |
| `gold_column_stats.csv` | gold 列数分布 + k_primary / k_operational |
| `grid_sweep.csv` | Phase 2: margin × lambda 网格搜索结果 |
| `topk_curve.csv` | 多档 K 的 Recall / NDCG 曲线 |
| `topk_choice.csv` | 推荐的 chosen_k_deploy |
| `baseline_vs_finetuned.csv` | 基座 vs 微调 NDCG / MRR / Recall |
