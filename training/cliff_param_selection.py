"""
双阈值动态截断参数选择（Dual-Threshold Dynamic Truncation）。

- protect_ratio: 默认用全体 covered 样本 r_min 降序曲线拐点（elbow / 最大垂距法）；
  可选回退到覆盖率约束法。
- min_ratio: 仅由远端断崖点（noise_floor_cliff）的 noise_start_ratio 聚合；
  不加 0.15 等人为下界，也不与 protect_ratio 或 expected_cliff 联动截断。
"""
from __future__ import annotations

from typing import Any

import numpy as np


def _round6(value: float) -> float:
    return round(float(value), 6)


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": _round6(float(arr.mean())),
        "median": _round6(float(np.percentile(arr, 50))),
        "min": _round6(float(arr.min())),
        "max": _round6(float(arr.max())),
    }


def coverage_at_alpha(r_min_values: list[float], alpha: float) -> float:
    """
    Coverage(alpha) = (1/N) * sum_i I(r_min^(i) >= alpha)

    r_min^(i): 样本 i 中最弱 gold 列分数 / top1 分数。
    当 protect_ratio = alpha 时，样本 i 的全部 gold 都在保护区内，当且仅当 alpha <= r_min^(i)，
    等价于 r_min^(i) >= alpha。
    """
    values = [float(x) for x in r_min_values if x is not None]
    if not values:
        return 0.0
    hit = sum(1 for v in values if v >= float(alpha))
    return hit / len(values)


def _perpendicular_distance_to_chord(
    x: float,
    y: float,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float:
    """点 (x,y) 到线段 (x0,y0)-(x1,y1) 的垂直距离（用于拐点检测）。"""
    dx = x1 - x0
    dy = y1 - y0
    denom = (dx * dx + dy * dy) ** 0.5
    if denom <= 0.0:
        return 0.0
    return abs(dy * x - dx * y + x1 * y0 - y1 * x0) / denom


def select_protect_ratio_by_elbow(
    r_min_values: list[float],
    *,
    min_coverage: float = 0.80,
) -> dict[str, Any]:
    """
    在 Protect Ratio 曲线（r_min 降序）上找拐点（elbow）。

    曲线横轴为 rank percent，纵轴为 protect_ratio_sorted_desc。用首尾连线上的
    最大垂距点作为「平台区结束、困难样本开始」的转折，得到全局 protect_ratio。

    相比覆盖率法（取满足 tau 的最大 alpha，往往偏低如 0.3），拐点法在曲线
    平台与陡降之间取阈值，protect_ratio 更高 → 保护区更小 → 断崖/Otsu 搜索段更长。

    若拐点覆盖率低于 min_coverage，回退到覆盖率约束法（满足 min_coverage 的最大 alpha）。
    """
    if not (0.0 < float(min_coverage) <= 1.0):
        raise ValueError("min_coverage 必须在 (0, 1]")
    values = sorted(float(x) for x in r_min_values)
    n = len(values)
    if n == 0:
        return {
            "tau_protect": 0.0,
            "method": "elbow",
            "achieved_coverage": 0.0,
            "elbow_rank_percent": None,
            "elbow_index": None,
            "fallback_used": False,
            "total_samples": 0,
        }
    if n == 1:
        tau = values[0]
        return {
            "tau_protect": _round6(tau),
            "method": "elbow",
            "achieved_coverage": 1.0,
            "elbow_rank_percent": 0.0,
            "elbow_index": 0,
            "elbow_raw_ratio": _round6(tau),
            "fallback_used": False,
            "total_samples": 1,
        }

    ys = sorted(values, reverse=True)
    xs = [i * 100.0 / (n - 1) for i in range(n)]
    x0, y0 = xs[0], ys[0]
    x1, y1 = xs[-1], ys[-1]

    best_idx = 0
    best_dist = -1.0
    for i in range(n):
        dist = _perpendicular_distance_to_chord(xs[i], ys[i], x0, y0, x1, y1)
        if dist > best_dist:
            best_dist = dist
            best_idx = i

    tau = float(ys[best_idx])
    achieved = coverage_at_alpha(values, tau)
    fallback_used = False
    if achieved + 1e-12 < float(min_coverage):
        fallback = select_protect_ratio_by_coverage(
            values,
            target_coverage=float(min_coverage),
        )
        tau = float(fallback["tau_protect"])
        achieved = float(fallback["achieved_coverage"])
        fallback_used = True

    return {
        "tau_protect": _round6(tau),
        "method": "elbow",
        "achieved_coverage": _round6(achieved),
        "elbow_rank_percent": _round6(xs[best_idx]),
        "elbow_index": int(best_idx),
        "elbow_raw_ratio": _round6(float(ys[best_idx])),
        "min_coverage": float(min_coverage),
        "fallback_used": fallback_used,
        "total_samples": n,
    }


def select_protect_ratio_by_coverage(
    r_min_values: list[float],
    *,
    target_coverage: float = 0.95,
) -> dict[str, Any]:
    """
    tau_protect = max { alpha | Coverage(alpha) >= tau }

    在 Protect Ratio 曲线上，这是满足系统召回覆盖率要求的最激进（最大）保护阈值。
    """
    if not (0.0 < float(target_coverage) <= 1.0):
        raise ValueError("target_coverage 必须在 (0, 1]")
    values = sorted(float(x) for x in r_min_values)
    n = len(values)
    if n == 0:
        return {
            "tau_protect": 0.0,
            "target_coverage": float(target_coverage),
            "achieved_coverage": 0.0,
            "covered_sample_count": 0,
            "total_samples": 0,
            "selection_index": None,
        }

    # 从小到大扫描 alpha，取仍满足 Coverage >= tau 的最大 alpha。
    tau_protect = 0.0
    achieved = 0.0
    for alpha in sorted(set(values)):
        cov = coverage_at_alpha(values, alpha)
        if cov + 1e-12 >= float(target_coverage):
            tau_protect = float(alpha)
            achieved = float(cov)

    # 等价分位数：约 (1-tau) 分位点
    quantile_pct = max(0.0, min(100.0, (1.0 - float(target_coverage)) * 100.0))
    quantile_value = float(np.percentile(values, quantile_pct))

    return {
        "tau_protect": _round6(tau_protect),
        "target_coverage": float(target_coverage),
        "achieved_coverage": _round6(achieved),
        "covered_sample_count": int(round(achieved * n)),
        "total_samples": n,
        "quantile_percent": _round6(quantile_pct),
        "quantile_r_min": _round6(quantile_value),
    }


def _remote_cliff_min_ratios(
    remote_cliff_noise_start_ratios: list[float | None],
    *,
    margin_eps: float,
) -> list[float]:
    """远端断崖点 noise_start_ratio / top1，加 margin_eps 后裁剪到 [0, 1]。"""
    return [
        max(0.0, min(1.0, float(r) + float(margin_eps)))
        for r in remote_cliff_noise_start_ratios
        if r is not None
    ]


def select_min_ratio_from_remote_cliff(
    remote_cliff_noise_start_ratios: list[float | None],
    *,
    aggregate_percentile: float = 50.0,
    margin_eps: float = 1e-4,
) -> dict[str, Any]:
    """
    从远端断崖点（noise_floor_cliff）聚合 min_ratio。

    每题贡献 noise_start_ratio / top1（可选加 margin_eps）；无远端断崖的样本跳过。
    全体缺失时 tau_min=0。不设 0.15 等下界。
    """
    per_sample = _remote_cliff_min_ratios(
        remote_cliff_noise_start_ratios,
        margin_eps=margin_eps,
    )

    if per_sample:
        tau_min = float(np.percentile(per_sample, float(aggregate_percentile)))
    else:
        tau_min = 0.0

    return {
        "tau_min": _round6(max(0.0, tau_min)),
        "aggregate_percentile": float(aggregate_percentile),
        "margin_eps": float(margin_eps),
        "remote_cliff_noise_start_ratio": _stats(per_sample),
        "remote_cliff_sample_count": len(per_sample),
        "missing_remote_cliff_count": sum(
            1 for r in remote_cliff_noise_start_ratios if r is None
        ),
    }


# 兼容旧调用名
select_min_ratio_from_cliff_points = select_min_ratio_from_remote_cliff


def local_adaptive_split(
    scores: list[float],
    *,
    eps: float = 1e-12,
) -> dict[str, Any]:
    """
    局部自适应聚类切分（Local Adaptive Split）。

    对单题的候选分数集合 S = {s_1, ..., s_k}，先 min-max 归一化到 [0,1]，
    再用最大类间方差（Otsu 原理，等价于 1D K-Means K=2）找到局部阈值 T_local，
    把 S 分为保留簇 C_keep（高分）与噪声簇 C_drop（低分）：

        T_local = argmax_t [ w_keep(t) * w_drop(t) * (mu_keep(t) - mu_drop(t))^2 ]

    其中 w 是两簇的点数占比，mu 是两簇的归一化均分。

    约定：传入的 scores 视为按分数从高到低排序（评估流程中即为 TopK 降序），
    返回 keep_count 表示高分簇的列数，可直接作为前缀截断长度。

    退化兜底（均返回 keep_count = k，全保留，召回安全且对照公平）：
    - k < 2；
    - 全部分数相等（max == min，无法归一化）；
    - 最优类间方差 ≈ 0（分数近似均匀，无显著双峰）。
    """
    vals = [float(s) for s in scores]
    k = len(vals)
    base = {
        "keep_count": k,
        "drop_count": 0,
        "threshold": None,
        "between_class_variance": 0.0,
        "degenerate": True,
        "reason": "fewer_than_two_candidates",
    }
    if k < 2:
        return base

    s_min = min(vals)
    s_max = max(vals)
    span = s_max - s_min
    if span <= eps:
        base["reason"] = "uniform_scores"
        return base

    # min-max 归一化；保持与传入顺序一致（评估流程为降序）。
    norm = [(v - s_min) / span for v in vals]

    prefix = [0.0] * (k + 1)
    for i in range(k):
        prefix[i + 1] = prefix[i] + norm[i]
    total = prefix[k]

    best_cut = 0
    best_var = -1.0
    for cut in range(1, k):
        w_keep = cut / k
        w_drop = (k - cut) / k
        mu_keep = prefix[cut] / cut
        mu_drop = (total - prefix[cut]) / (k - cut)
        between = w_keep * w_drop * (mu_keep - mu_drop) ** 2
        if between > best_var:
            best_var = between
            best_cut = cut

    if best_cut <= 0 or best_var <= eps:
        base["reason"] = "no_separating_threshold"
        base["between_class_variance"] = _round6(max(0.0, best_var))
        return base

    # 阈值取断崖两侧归一化分数的中点（仅诊断用）。
    threshold_norm = (norm[best_cut - 1] + norm[best_cut]) / 2.0
    return {
        "keep_count": int(best_cut),
        "drop_count": int(k - best_cut),
        "threshold": _round6(threshold_norm),
        "threshold_score": _round6(s_min + threshold_norm * span),
        "between_class_variance": _round6(best_var),
        "degenerate": False,
        "reason": "otsu_max_between_class_variance",
    }


def select_dual_thresholds(
    *,
    r_min_values: list[float],
    remote_cliff_noise_start_ratios: list[float | None],
    target_coverage: float = 0.95,
    margin_eps: float = 1e-4,
    aggregate_percentile: float = 50.0,
    protect_min_coverage: float = 0.80,
    protect_selection_method: str = "elbow",
    # 兼容旧参数名
    cliff2_noise_start_ratios: list[float | None] | None = None,
    min_percentile: float | None = None,
    cliff1_kept_ratios: list[float | None] | None = None,
) -> dict[str, Any]:
    ratios = (
        remote_cliff_noise_start_ratios
        if cliff2_noise_start_ratios is None
        else cliff2_noise_start_ratios
    )
    pct = float(min_percentile) if min_percentile is not None else float(aggregate_percentile)

    if protect_selection_method == "coverage":
        protect_diag = select_protect_ratio_by_coverage(
            r_min_values,
            target_coverage=target_coverage,
        )
        protect_diag["elbow_value_source"] = "all_covered_r_min"
    elif protect_selection_method == "elbow":
        protect_diag = select_protect_ratio_by_elbow(
            r_min_values,
            min_coverage=float(protect_min_coverage),
        )
        protect_diag["elbow_value_source"] = "all_covered_r_min"
    else:
        raise ValueError("protect_selection_method 必须是 'elbow' 或 'coverage'")
    min_diag = select_min_ratio_from_remote_cliff(
        ratios,
        aggregate_percentile=pct,
        margin_eps=margin_eps,
    )
    result: dict[str, Any] = {
        "candidate_cliff_protect_ratio": protect_diag["tau_protect"],
        "candidate_cliff_min_ratio": min_diag["tau_min"],
        "target_coverage": float(target_coverage),
        "protect_min_coverage": float(protect_min_coverage),
        "protect_selection_method": protect_selection_method,
        "margin_eps": float(margin_eps),
        "aggregate_percentile": pct,
        "protect_selection": protect_diag,
        "min_selection": min_diag,
    }
    if cliff1_kept_ratios is not None:
        cliff1 = [float(x) for x in cliff1_kept_ratios if x is not None]
        result["expected_cliff_kept_ratio"] = _stats(cliff1)
        result["expected_cliff_sample_count"] = len(cliff1)
    return result
