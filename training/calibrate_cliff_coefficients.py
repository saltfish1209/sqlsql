"""
按 run_grid_sweep 的输入拼接方式，用默认最优权重在验证集做 TopK 排序，
逐题估计“要保住正确列”所需的比例断崖参数，并汇总平均值。

说明：
- 输入拼接与 run_grid_sweep 评估一致： [question, build_column_passage(...)]。
- 默认模型路径: settings.cross_encoder_model（即最终权重默认位置）。
- 每题都会打印一次参数估计；若 gold 列未完整落入 TopK，会显式标记。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import CrossEncoder

sys.path.insert(0, str(os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))))
from config.settings import settings
from pipeline.cross_encoder_passage import build_column_passage_map
from training.dataset_io import read_jsonl
from training.cliff_param_selection import select_dual_thresholds
from training.cross_encoder_eval import load_eval_records


def _calc_stats(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p30": 0.0,
            "p50": 0.0,
            "p70": 0.0,
            "p75": 0.0,
        }
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.percentile(arr, 50)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p25": float(np.percentile(arr, 25)),
        "p30": float(np.percentile(arr, 30)),
        "p50": float(np.percentile(arr, 50)),
        "p70": float(np.percentile(arr, 70)),
        "p75": float(np.percentile(arr, 75)),
    }


def _filter_pure_max_cliff_columns(
    ranked_cols: list[str],
    ranked_scores: list[float],
    *,
    top_k: int,
) -> list[str]:
    """TopK 内纯最大相对跌幅断崖截断（与 evaluate_cliff_recall max_cliff_only 一致）。"""
    topk_cols = ranked_cols[:top_k]
    topk_scores = [float(s) for s in ranked_scores[:top_k]]
    if len(topk_scores) < 2:
        return list(topk_cols)
    best = topk_scores[0]
    if best <= 0:
        return list(topk_cols)

    cliff_idx = 0
    max_decay = -1.0
    for i in range(len(topk_scores) - 1):
        cur = topk_scores[i]
        nxt = topk_scores[i + 1]
        if cur <= 0:
            continue
        decay = (cur - nxt) / cur
        if decay > max_decay:
            max_decay = decay
            cliff_idx = i
    return topk_cols[: cliff_idx + 1]


def _recall_ratio(gold: set[str], cols: list[str]) -> float:
    if not gold:
        return 0.0
    selected = {str(c).strip() for c in cols if str(c).strip()}
    return len(gold & selected) / len(gold)


def _sort_pure_cliff_fail_entries(
    entries: list[dict],
) -> list[dict]:
    """按 protect_ratio（r_min）从大到小排序。"""
    return sorted(
        entries,
        key=lambda row: (-float(row["protect_ratio"]), int(row["case_idx"])),
    )


def _build_ratio_points_rows(
    *,
    min_values: list[float],
    protect_values: list[float],
    pure_cliff_fail_entries: list[dict] | None = None,
) -> list[dict]:
    min_sorted = sorted(float(x) for x in (min_values or [0.0]))
    protect_sorted = sorted((float(x) for x in (protect_values or [0.0])), reverse=True)
    fail_sorted = _sort_pure_cliff_fail_entries(list(pure_cliff_fail_entries or []))
    fail_protect = [float(x["protect_ratio"]) for x in fail_sorted]
    fail_case_idx = [int(x["case_idx"]) for x in fail_sorted]

    count = max(len(min_sorted), len(protect_sorted), len(fail_protect))
    if len(min_sorted) < count:
        min_sorted.extend([min_sorted[-1]] * (count - len(min_sorted)))
    if len(protect_sorted) < count:
        protect_sorted.extend([protect_sorted[-1]] * (count - len(protect_sorted)))

    rows: list[dict] = []
    for idx in range(count):
        percent = (idx * 100.0 / (count - 1)) if count > 1 else 0.0
        row = {
            "rank": idx + 1,
            "rank_percent": round(float(percent), 6),
            "min_ratio_sorted_asc": round(float(min_sorted[idx]), 6),
            "protect_ratio_sorted_desc": round(float(protect_sorted[idx]), 6),
        }
        if idx < len(fail_protect):
            row["protect_ratio_pure_cliff_fail_sorted_desc"] = round(fail_protect[idx], 6)
            row["pure_cliff_fail_case_idx"] = fail_case_idx[idx]
        else:
            row["protect_ratio_pure_cliff_fail_sorted_desc"] = ""
            row["pure_cliff_fail_case_idx"] = ""
        rows.append(row)
    return rows


def _write_ratio_points_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "rank_percent",
        "min_ratio_sorted_asc",
        "protect_ratio_sorted_desc",
        "protect_ratio_pure_cliff_fail_sorted_desc",
        "pure_cliff_fail_case_idx",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _detect_cliff_candidates(
    ranked_cols: list[str],
    ranked_scores: list[float],
    *,
    top_k: int,
    last_gold_pos: int | None = None,
    min_decay: float = 0.15,
) -> list[dict]:
    """
    在拉长后的 TopK 序列里寻找相邻分数断崖。

    cut_position 使用 1-based 位置，表示若在该断崖切分，应保留前 cut_position 列；
    noise_start_position 是断崖右侧第一列，也就是候选噪声起点。
    """
    cols = ranked_cols[:top_k]
    scores = ranked_scores[:top_k]
    best = float(scores[0]) if scores else 0.0
    if best <= 0:
        return []

    candidates: list[dict] = []
    for idx in range(len(scores) - 1):
        cur = float(scores[idx])
        nxt = float(scores[idx + 1])
        if cur <= 0:
            continue
        decay = (cur - nxt) / cur
        if decay < float(min_decay):
            continue
        cut_position = idx + 1
        entry = {
            "cut_position": cut_position,
            "noise_start_position": idx + 2,
            "kept_column": cols[idx],
            "noise_start_column": cols[idx + 1],
            "kept_score": round(cur, 6),
            "noise_start_score": round(nxt, 6),
            "kept_ratio": round(cur / best, 6),
            "noise_start_ratio": round(nxt / best, 6),
            "decay_rate": round(decay, 6),
        }
        if last_gold_pos is not None:
            entry["distance_after_last_gold"] = cut_position - int(last_gold_pos)
        candidates.append(entry)
    return candidates


def _choose_cliff_roles(
    cliff_candidates: list[dict],
    *,
    last_gold_pos: int,
    cliff_search_count: int,
) -> tuple[dict | None, dict | None]:
    """
    将 last gold 后的候选断崖拆成两个角色。

    - expected_cliff：离 last gold 最近，切除字段更多，是比例断崖法期望捕获的边界。
    - noise_floor_cliff：在前几个 post-gold cliff 中稍远，切除字段更少，用于估计
      min_ratio 的真正噪声地板。
    """
    post_gold = [
        c for c in cliff_candidates
        if int(c.get("cut_position", 0)) >= int(last_gold_pos)
    ]
    if not post_gold:
        return None, None

    nearest = sorted(
        post_gold,
        key=lambda c: (int(c.get("cut_position", 0)) - int(last_gold_pos), -float(c.get("decay_rate", 0.0))),
    )[: max(1, int(cliff_search_count))]
    expected_cliff = nearest[0]
    noise_floor_cliff = nearest[1] if len(nearest) >= 2 else None
    return expected_cliff, noise_floor_cliff


def _estimate_params_for_case(
    ranked_cols: list[str],
    ranked_scores: list[float],
    gold_set: set[str],
    *,
    top_k: int,
    analysis_top_k: int | None = None,
    cliff_search_count: int = 2,
    noise_decay_threshold: float = 0.15,
    min_ratio_eps: float = 1e-4,
) -> dict:
    top_cols = ranked_cols[:top_k]
    top_scores = ranked_scores[:top_k]
    best = top_scores[0] if top_scores else 0.0
    analysis_k = max(int(top_k), int(analysis_top_k or top_k))
    if best <= 0:
        return {
            "covered": False,
            "missing_gold": sorted(gold_set),
            "required_min_ratio": None,
            "required_protect_ratio": None,
            "gold_positions": [],
            "gold_scores": [],
            "analysis_top_k": analysis_k,
            "cliff_candidates": [],
            "expected_cliff": None,
            "noise_floor_cliff": None,
            "selected_noise_cliff": None,
            "noise_start_column": None,
            "min_ratio_reason": "non_positive_best_score",
        }

    pos_map = {c: i for i, c in enumerate(top_cols)}
    covered_gold = [g for g in gold_set if g in pos_map]
    missing = sorted([g for g in gold_set if g not in pos_map])
    covered = len(missing) == 0

    gold_positions = sorted([pos_map[g] + 1 for g in covered_gold])
    gold_scores = [top_scores[pos_map[g]] for g in covered_gold]
    if not covered:
        return {
            "covered": False,
            "missing_gold": missing,
            "required_min_ratio": None,
            "required_protect_ratio": None,
            "gold_positions": gold_positions,
            "gold_scores": [round(float(x), 6) for x in gold_scores],
            "analysis_top_k": analysis_k,
            "cliff_candidates": [],
            "expected_cliff": None,
            "noise_floor_cliff": None,
            "selected_noise_cliff": None,
            "noise_start_column": None,
            "min_ratio_reason": "gold_missing_from_topk",
        }

    min_gold_score = min(gold_scores)
    last_gold_pos = max(gold_positions)
    min_gold_ratio = max(0.0, min(1.0, min_gold_score / best))

    cliff_candidates = _detect_cliff_candidates(
        ranked_cols,
        ranked_scores,
        top_k=analysis_k,
        last_gold_pos=last_gold_pos,
        min_decay=noise_decay_threshold,
    )
    expected_cliff, noise_floor_cliff = _choose_cliff_roles(
        cliff_candidates,
        last_gold_pos=last_gold_pos,
        cliff_search_count=cliff_search_count,
    )

    eps = float(min_ratio_eps)
    # 保护区只覆盖 gold 分数下界（r_min），不与 expected_cliff 联动抬高，
    # 避免把断崖切点包进保护前缀、使断崖/Otsu 搜索失效。
    protect_ratio_needed = min_gold_ratio
    min_ratio_reason = "no_post_gold_cliff"
    noise_start_column = None
    if expected_cliff is None:
        min_ratio_needed = 0.0
    else:
        if noise_floor_cliff is None:
            min_ratio_needed = 0.0
            min_ratio_reason = "single_expected_cliff_no_noise_floor"
        else:
            noise_ratio = float(noise_floor_cliff["noise_start_ratio"])
            min_ratio_needed = max(0.0, min(1.0, noise_ratio + eps))
            min_ratio_reason = "noise_floor_cliff"
            noise_start_column = noise_floor_cliff["noise_start_column"]

    return {
        "covered": True,
        "missing_gold": [],
        "required_min_ratio": min_ratio_needed,
        "required_protect_ratio": protect_ratio_needed,
        "gold_positions": gold_positions,
        "gold_scores": [round(float(x), 6) for x in gold_scores],
        "last_gold_position": last_gold_pos,
        "last_gold_score": round(float(top_scores[last_gold_pos - 1]), 6),
        "min_gold_ratio": round(float(min_gold_ratio), 6),
        "analysis_top_k": analysis_k,
        "cliff_candidates": cliff_candidates,
        "expected_cliff": expected_cliff,
        "noise_floor_cliff": noise_floor_cliff,
        "selected_noise_cliff": noise_floor_cliff,
        "noise_start_column": noise_start_column,
        "min_ratio_reason": min_ratio_reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="逐题估计比例断崖参数（默认验证集 + 默认最优权重）")
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--top-k", type=int, default=10,help="候选截断 K，默认 10")
    parser.add_argument(
        "--analysis-top-k",
        type=int,
        default=50,
        help="用于寻找真实噪声断崖的拉长候选 K，默认 50；小于 --top-k 时自动取 --top-k",
    )
    parser.add_argument(
        "--cliff-search-count",
        type=int,
        default=2,
        help="在 last gold 后按位置取前 N 个候选断崖：近者为期望断崖，稍远者为噪声地板，默认 2",
    )
    parser.add_argument(
        "--noise-decay-threshold",
        type=float,
        default=0.15,
        help="判定候选噪声断崖的最小相邻衰减率，默认 0.15",
    )
    parser.add_argument(
        "--min-ratio-eps",
        type=float,
        default=1e-4,
        help="min_ratio 设置在噪声起点分数比例之上的安全余量，默认 1e-4",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-path", type=str, default=str(settings.cross_encoder_model))
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--ratio-points-csv-out", type=str, default="")
    parser.add_argument(
        "--robust-min-percentile",
        type=float,
        default=30.0,
        help="稳健推荐中 min_ratio 使用的百分位数（默认 30）",
    )
    parser.add_argument(
        "--robust-protect-percentile",
        type=float,
        default=50.0,
        help="稳健推荐中 protect_ratio 使用的百分位数（默认 50）",
    )
    parser.add_argument(
        "--target-coverage",
        type=float,
        default=0.95,
        help="双阈值法 protect_ratio 的目标覆盖率 tau（默认 0.95）",
    )
    parser.add_argument(
        "--min-ratio-margin-eps",
        type=float,
        default=1e-4,
        help="双阈值 min 聚合时加在断崖点 2 noise_start_ratio 上的余量（默认 1e-4）",
    )
    parser.add_argument(
        "--dual-min-percentile",
        type=float,
        default=50.0,
        help="双阈值 min 对远端断崖 noise_start_ratio 使用的百分位数（默认 50）",
    )
    args = parser.parse_args()

    model_path = str(args.model_path).strip()
    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"模型目录不存在: {model_path}")

    csv_path = str(settings.csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV 数据文件不存在: {csv_path}")

    if args.top_k <= 0:
        raise ValueError("--top-k 必须 > 0")
    if args.analysis_top_k <= 0:
        raise ValueError("--analysis-top-k 必须 > 0")
    if args.cliff_search_count <= 0:
        raise ValueError("--cliff-search-count 必须 > 0")
    if not (0.0 <= args.noise_decay_threshold <= 1.0):
        raise ValueError("--noise-decay-threshold 必须在 [0,1]")
    if args.min_ratio_eps < 0:
        raise ValueError("--min-ratio-eps 必须 >= 0")
    if not (0.0 < args.target_coverage <= 1.0):
        raise ValueError("--target-coverage 必须在 (0, 1]")
    if args.min_ratio_margin_eps < 0:
        raise ValueError("--min-ratio-margin-eps 必须 >= 0")

    analysis_top_k = max(args.top_k, args.analysis_top_k)
    print(
        f"[INFO] split={args.split} top_k={args.top_k} analysis_top_k={analysis_top_k} "
        f"noise_decay={args.noise_decay_threshold} model={model_path}"
    )
    model = CrossEncoder(model_path, trust_remote_code=True)
    passage_map = build_column_passage_map(str(settings.schema_path), csv_path)
    all_cols = list(passage_map.keys())
    if args.split == "train":
        train_eval_path = Path(__file__).parent / "cross_encoder_train_eval_data.jsonl"
        if not train_eval_path.is_file():
            raise FileNotFoundError(
                f"未找到训练评估集: {train_eval_path}。"
                "请先运行 training/prepare_data.py 生成 cross_encoder_train_eval_data.jsonl。"
            )
        records = list(read_jsonl(str(train_eval_path)))
    else:
        records = load_eval_records(args.split)

    per_case: list[dict] = []
    valid_ratios: list[dict[str, float]] = []
    pure_cliff_fail_entries: list[dict] = []
    r_min_values: list[float] = []
    min_hat_values: list[float] = []
    cliff1_kept_ratios: list[float | None] = []
    cliff2_noise_start_ratios: list[float | None] = []
    covered_count = 0
    total = 0

    for idx, item in enumerate(records, start=1):
        question = str(item.get("question", "")).strip()
        gold = set(item.get("gold_columns") or [])
        if not question or not gold:
            continue
        total += 1
        inputs = [[question, passage_map[col]] for col in all_cols]
        scores = model.predict(inputs, batch_size=args.batch_size)
        sorted_idx = np.argsort(scores)[::-1]
        ranked_cols = [all_cols[j] for j in sorted_idx]
        ranked_scores = [float(scores[j]) for j in sorted_idx]

        est = _estimate_params_for_case(
            ranked_cols,
            ranked_scores,
            gold,
            top_k=args.top_k,
            analysis_top_k=analysis_top_k,
            cliff_search_count=args.cliff_search_count,
            noise_decay_threshold=args.noise_decay_threshold,
            min_ratio_eps=args.min_ratio_eps,
        )
        row = {
            "idx": idx,
            "question": question,
            "gold_columns": sorted(gold),
            "topk_columns": ranked_cols[:args.top_k],
            "topk_scores": [round(float(x), 6) for x in ranked_scores[:args.top_k]],
            "analysis_topk_columns": ranked_cols[:analysis_top_k],
            "analysis_topk_scores": [round(float(x), 6) for x in ranked_scores[:analysis_top_k]],
            **est,
        }
        per_case.append(row)

        if est["covered"]:
            covered_count += 1
            r_min_values.append(float(est["min_gold_ratio"]))
            min_hat_values.append(float(est["required_min_ratio"]))
            expected = est.get("expected_cliff")
            noise_floor = est.get("noise_floor_cliff")
            cliff1_kept_ratios.append(
                float(expected["kept_ratio"]) if expected else None
            )
            cliff2_noise_start_ratios.append(
                float(noise_floor["noise_start_ratio"]) if noise_floor else None
            )
            valid_ratios.append(
                {
                    "min_ratio": float(est["required_min_ratio"]),
                    "protect_ratio": float(est["required_protect_ratio"]),
                    "r_min": float(est["min_gold_ratio"]),
                }
            )
            pure_cols = _filter_pure_max_cliff_columns(
                ranked_cols,
                ranked_scores,
                top_k=args.top_k,
            )
            pure_recall = _recall_ratio(gold, pure_cols)
            if pure_recall + 1e-12 < 1.0:
                pure_cliff_fail_entries.append(
                    {
                        "case_idx": idx,
                        "protect_ratio": float(est["min_gold_ratio"]),
                        "pure_cliff_recall": round(float(pure_recall), 6),
                    }
                )
            print(
                f"[Case {idx}] covered=YES "
                f"pos={est['gold_positions']} "
                f"min_ratio={est['required_min_ratio']:.4f} "
                f"protect_ratio={est['required_protect_ratio']:.4f} "
                f"reason={est['min_ratio_reason']}"
            )
        else:
            print(
                f"[Case {idx}] covered=NO "
                f"missing={est['missing_gold']} "
                f"gold_in_topk_pos={est['gold_positions']}"
            )

    min_values = [x["min_ratio"] for x in valid_ratios]
    protect_values = [x["protect_ratio"] for x in valid_ratios]
    min_stats = _calc_stats(min_values)
    protect_stats = _calc_stats(protect_values)

    robust_min = float(np.percentile(min_values, args.robust_min_percentile)) if min_values else 0.0
    robust_protect = float(np.percentile(protect_values, args.robust_protect_percentile)) if protect_values else 0.0

    dual_thresholds = select_dual_thresholds(
        r_min_values=r_min_values,
        remote_cliff_noise_start_ratios=cliff2_noise_start_ratios,
        target_coverage=args.target_coverage,
        margin_eps=args.min_ratio_margin_eps,
        aggregate_percentile=args.dual_min_percentile,
        protect_min_coverage=0.80,
        protect_selection_method="elbow",
        cliff1_kept_ratios=cliff1_kept_ratios,
    )

    ratio_points_rows = _build_ratio_points_rows(
        min_values=min_hat_values,
        protect_values=r_min_values,
        pure_cliff_fail_entries=pure_cliff_fail_entries,
    )

    default_artifact_name = (
        f"cliff_coefficients_{args.split}_top{args.top_k}_analysis{analysis_top_k}"
    )
    ratio_points_csv_path = (
        Path(args.ratio_points_csv_out)
        if args.ratio_points_csv_out
        else Path(__file__).parent / f"{default_artifact_name}_ratio_points.csv"
    )
    _write_ratio_points_csv(ratio_points_csv_path, ratio_points_rows)

    summary = {
        "split": args.split,
        "model_path": model_path,
        "top_k": args.top_k,
        "analysis_top_k": analysis_top_k,
        "cliff_search_count": args.cliff_search_count,
        "noise_decay_threshold": args.noise_decay_threshold,
        "min_ratio_eps": args.min_ratio_eps,
        "evaluated_samples": total,
        "covered_samples": covered_count,
        "covered_ratio": covered_count / total if total else 0.0,
        "pure_cliff_fail_sample_count": len(pure_cliff_fail_entries),
        "pure_cliff_fail_ratio": len(pure_cliff_fail_entries) / covered_count
        if covered_count
        else 0.0,
        "artifacts": {
            "ratio_points_csv": str(ratio_points_csv_path),
        },
        "stats_on_covered_cases": {
            "candidate_cliff_min_ratio": {k: round(v, 6) if isinstance(v, float) else v for k, v in min_stats.items()},
            "candidate_cliff_protect_ratio": {k: round(v, 6) if isinstance(v, float) else v for k, v in protect_stats.items()},
        },
        "recommended_params_mean": {
            "candidate_cliff_min_ratio": round(min_stats["mean"], 6),
            "candidate_cliff_protect_ratio": round(protect_stats["mean"], 6),
        },
        "recommended_params_robust": {
            "percentiles": {
                "min_ratio": args.robust_min_percentile,
                "protect_ratio": args.robust_protect_percentile,
            },
            "candidate_cliff_min_ratio": round(robust_min, 6),
            "candidate_cliff_protect_ratio": round(robust_protect, 6),
        },
        "recommended_params_dual_threshold": dual_thresholds,
        "notes": (
            "post-gold 候选断崖会拆成两个角色：expected_cliff 是离 last gold 最近、切除字段更多的比例断崖期望位置；"
            "noise_floor_cliff 是稍远、切除字段更少的噪声地板位置，min_ratio 只由它右侧分数比例估计；"
            "运行时断崖截断仅由 protect_ratio 保护区边界决定，不再使用 decay_threshold；"
            "若只有 expected_cliff 或没有 post-gold cliff，则该样本推荐 min_ratio=0，避免分数地板抢先截断。"
            "若 covered_ratio 过低，建议先提升 reranker 质量或增大 top_k。"
        ),
    }

    out_path = (
        Path(args.out)
        if args.out
        else Path(__file__).parent / f"{default_artifact_name}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "per_case": per_case}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"输出: {out_path}")
    print(f"ratio points CSV: {ratio_points_csv_path}")
    print("建议环境变量（稳健推荐）:")
    print(f"  export CANDIDATE_CLIFF_MIN_RATIO={summary['recommended_params_robust']['candidate_cliff_min_ratio']}")
    print(f"  export CANDIDATE_CLIFF_PROTECT_RATIO={summary['recommended_params_robust']['candidate_cliff_protect_ratio']}")
    dual = summary["recommended_params_dual_threshold"]
    print("双阈值推荐（覆盖率约束 + 受限肘部）:")
    print(f"  export CANDIDATE_CLIFF_PROTECT_RATIO={dual['candidate_cliff_protect_ratio']}")
    print(f"  export CANDIDATE_CLIFF_MIN_RATIO={dual['candidate_cliff_min_ratio']}")
    print(json.dumps(dual, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
