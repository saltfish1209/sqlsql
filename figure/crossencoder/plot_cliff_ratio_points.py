from __future__ import annotations

import argparse
import csv
from pathlib import Path


def _default_training_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "training"


def _resolve_ratio_points_csv(csv_arg: str | None) -> Path:
    if csv_arg:
        csv_path = Path(csv_arg)
        if "ratio_grid" in csv_path.name:
            raise ValueError(
                f"检测到旧版分位点 CSV: {csv_path}\n"
                "请重新运行 training/calibrate_cliff_coefficients.py，"
                "并使用新生成的 *_ratio_points.csv（每个 covered 样本一行）。"
            )
        return csv_path

    training_dir = _default_training_dir()
    candidates = sorted(
        training_dir.glob("*_ratio_points.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"未在 {training_dir} 找到 *_ratio_points.csv。\n"
            "请先运行: python training/calibrate_cliff_coefficients.py --split val ..."
        )
    return candidates[0]


def _read_ratio_points(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV 为空: {csv_path}")
    keys = set(rows[0].keys())
    if "grid_percent" in keys or "candidate_cliff_min_ratio" in keys:
        raise ValueError(
            f"CSV 为旧版 ratio_grid 格式（约 11 行分位点）: {csv_path}\n"
            "请重新运行 calibrate_cliff_coefficients.py 生成 *_ratio_points.csv。"
        )
    required = {
        "rank",
        "rank_percent",
        "min_ratio_sorted_asc",
        "protect_ratio_sorted_desc",
    }
    if not required.issubset(keys):
        raise ValueError(
            f"CSV 缺少字段，期望包含: {sorted(required)}，实际: {list(rows[0].keys())}"
        )
    return rows


def _series_from_column(rows: list[dict], column: str) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        raw = str(row.get(column, "")).strip()
        if not raw:
            continue
        xs.append(float(row["rank_percent"]))
        ys.append(float(raw))
    return xs, ys


def _plot_curve(
    *,
    series: list[tuple[list[float], list[float], str, str]],
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    grid_step: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for x, y, label, style in series:
        ax.plot(x, y, linewidth=1.6, label=label, linestyle=style)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(np.arange(0, 101, grid_step))
    y_max = max((max(y) for _, y, _, _ in series if y), default=0.0)
    upper = min(1.0, max(y_max, 0.1))
    ax.set_yticks(np.arange(0.0, upper + 1e-9, grid_step / 100.0))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
    if len(series) > 1:
        ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="按 ratio points CSV 绘制 min/protect 折线图")
    parser.add_argument(
        "--csv",
        default="",
        help="ratio_points.csv 路径；省略则自动使用 training/ 下最新的 *_ratio_points.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent),
        help="输出图片目录，默认 figure/crossencoder",
    )
    parser.add_argument(
        "--grid-step",
        type=int,
        default=10,
        help="背景网格线间隔百分比，默认 10",
    )
    args = parser.parse_args()

    if args.grid_step <= 0 or 100 % int(args.grid_step) != 0:
        raise ValueError("--grid-step 必须是 100 的正因子，例如 10")

    csv_path = _resolve_ratio_points_csv(str(args.csv).strip() or None)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV 不存在: {csv_path}")
    out_dir = Path(args.out_dir)
    rows = _read_ratio_points(csv_path)

    x = [float(r["rank_percent"]) for r in rows]
    min_y = [float(r["min_ratio_sorted_asc"]) for r in rows]
    protect_y = [float(r["protect_ratio_sorted_desc"]) for r in rows]
    fail_x, fail_y = _series_from_column(
        rows, "protect_ratio_pure_cliff_fail_sorted_desc"
    )

    stem = csv_path.stem.replace("_ratio_points", "")
    min_out = out_dir / f"{stem}_min_ratio_curve.png"
    protect_out = out_dir / f"{stem}_protect_ratio_curve.png"

    _plot_curve(
        series=[(x, min_y, "min_ratio (all covered)", "solid")],
        xlabel="Rank Percent (%)",
        ylabel="min_ratio_sorted_asc",
        title="Min Ratio Curve (All Points)",
        out_path=min_out,
        grid_step=int(args.grid_step),
    )

    protect_series: list[tuple[list[float], list[float], str, str]] = [
        (x, protect_y, "r_min all covered (desc)", "solid"),
    ]
    if fail_x:
        protect_series.append(
            (
                fail_x,
                fail_y,
                "r_min pure-cliff fail (desc)",
                "dashed",
            )
        )
    _plot_curve(
        series=protect_series,
        xlabel="Rank Percent (%)",
        ylabel="protect_ratio / r_min",
        title="Protect Ratio Curves (Elbow uses all covered r_min)",
        out_path=protect_out,
        grid_step=int(args.grid_step),
    )

    print(f"输入 CSV: {csv_path}（共 {len(rows)} 行）")
    print(f"纯 cliff 失败曲线点数: {len(fail_x)}")
    print(f"输出图像: {min_out}")
    print(f"输出图像: {protect_out}")


if __name__ == "__main__":
    main()
