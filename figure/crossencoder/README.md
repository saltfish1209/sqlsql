# CrossEncoder 实验与出图数据（CSV）

所有结果在 `figure/crossencoder/results/`，**一类实验一个 CSV**。大模型权重在 `checkpoints/`（不提交 git）。

## 目录

```
figure/crossencoder/
  README.md
  paths.py                 # CSV 路径常量
  k_selection.py           # Top-K 选取规则
  run_gold_stats.py        # gold 列数统计 → gold_column_stats.csv
  run_train_track.py       # 默认训练 + train_loss.csv + epoch_val.csv
  run_topk_curve.py        # 多档 K → topk_curve.csv
  run_pick_topk.py         # 从曲线选 K → topk_choice.csv
  run_baseline_compare.py  # 基座 vs 微调 → baseline_vs_finetuned.csv + topk_curve.csv
  run_sweep_lambda.py      # → lambda_sweep.csv
  run_sweep_margin.py      # → margin_sweep.csv
  run_sweep_epochs.py      # → epoch_sweep.csv
  results/
  checkpoints/
```

## 推荐流程

```text
1. python figure/crossencoder/run_gold_stats.py
2. python figure/crossencoder/run_train_track.py          # 或 sweep 脚本
3. python figure/crossencoder/run_baseline_compare.py     # 可选
4. python figure/crossencoder/run_topk_curve.py
5. python figure/crossencoder/run_pick_topk.py
```

画图：用 `results/train_loss.csv`（横轴 `global_step`）+ `epoch_val.csv`（每 epoch 一点 val recall）。

## Top-K 怎么选（重要）

有两类 K，**不要混在一次扫参里**：

| 名称 | 来源 | 用途 |
|------|------|------|
| **VAL_TOP_K** | `paths.py` → **6**（`NL2SQL_CE_VAL_TOP_K` 可覆盖） | **epoch val / λ / margin / epoch 扫参**固定对比 K |
| **k_primary** | `gold_column_stats.csv` → `max(6, gold_p95)` | 仅统计参考，与 VAL_TOP_K 独立 |
| **k_operational** | `settings.candidate_top_k`（默认 20） | 对齐主系统 SchemaLinker Top20 |
| **chosen_k_deploy** | `topk_choice.csv`（曲线 elbow） | 报告/分析「最小够用 K」 |

### 原则

1. **扫 epoch、λ、margin 时**：全程用同一个 **`VAL_TOP_K`（默认 6）**，改 `paths.py` 或环境变量。
2. **Top-K 本身不是和 λ 一起扫的**：用 `run_topk_curve.py` 单独出曲线，再用 `run_pick_topk.py` 选部署 K。
3. **选模型 / 早停**：默认 patience 见 `paths.py` 中 **`EARLY_STOP_PATIENCE`**（可用 `NL2SQL_CE_EARLY_STOP_PATIENCE` 或 `--early-stop-patience` 覆盖）：连续 N 个 epoch val `recall_at_k` 未创新高则停止，并将 **best epoch** 复制到 `save_path`。可用 `--no-early-stop` 跑满 epoch。

### `epoch_val.csv` 字段说明

| 列名 | 含义 |
|------|------|
| `recall_at_k` | **Top-K 召回率**：gold 列（问题+回答模版字段）是否全部落在预测 Top-K 内；比例 = 命中题数/总题数 |
| `is_best` | 本 epoch 的 `recall_at_k` 是否为目前最高（1/0） |
| `patience_counter` | 自上次创新高以来，连续多少个 epoch 未创新高（早停计数） |
| `stopped_early` | 本 epoch 评估后是否触发早停（1/0） |
| `top_k` | 评估使用的 K（默认 `VAL_TOP_K=6`） |

### chosen_k_deploy 规则（`run_pick_topk.py`）

在 `K >= k_primary` 的档位中，取满足  
`Full Recall@K >= 0.95 × max_recall(曲线)` 的 **最小 K**。  
若无满足则退回 recall 最高的 K。

## CSV 一览

| 文件 | 内容 |
|------|------|
| `train_loss.csv` | 每 20 step：loss / point / pair |
| `epoch_val.csv` | 每 epoch 结束：`recall_at_k`（仅 Top-K，无 Top-1） |
| `gold_column_stats.csv` | gold 列数分布 + k_primary / k_operational |
| `topk_curve.csv` | 多档 K 的 recall 曲线 |
| `topk_choice.csv` | 推荐的 chosen_k_deploy |
| `lambda_sweep.csv` | 每个 λ 一行（含 best_epoch） |
| `margin_sweep.csv` | 每个 margin 一行 |
| `epoch_sweep.csv` | 每个总 epoch 数一行 |
| `baseline_vs_finetuned.csv` | 基座 vs 微调 @k_primary |
