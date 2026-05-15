import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams
from matplotlib.lines import Line2D


def _setup_chinese_font() -> None:
    """与 plot_topk_pruned_combo 一致：设置 sans-serif 才能显示中文。"""
    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Microsoft JhengHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "WenQuanYi Zen Hei",
        "Droid Sans Fallback",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in preferred_fonts:
        if name in installed:
            rcParams["font.sans-serif"] = [name]
            rcParams["axes.unicode_minus"] = False
            return
    rcParams["axes.unicode_minus"] = False

df = pd.read_csv("topk_pruned_metrics1.csv", encoding="utf-8-sig")
correct_col = "correct" if "correct" in df.columns else "correct248"

plt.style.use("seaborn-v0_8-white")
_setup_chinese_font()

fig, ax1 = plt.subplots(figsize=(15, 6))

color_line = "#D62728"
full_schema_correct = 142
full_schema_avg_time = 7.55
ax1.plot(
    df["k"],
    df["avg_cost_time_seconds"],
    color=color_line,
    marker="o",
    linewidth=2,
    label="平均消耗时间",
)
ax1.set_ylabel("平均消耗时间", color=color_line, fontsize=12, fontweight="bold")
ax1.set_ylim(2, 8)
ax1.tick_params(axis="y", labelcolor=color_line)

ax2 = ax1.twinx()
color_bar = "#4682B4"
ax2.bar(df["k"], df[correct_col], color=color_bar, alpha=0.6, label="正确数量 (Correct)")
ax2.set_ylabel("Top-K 正确数", color=color_bar, fontsize=12, fontweight="bold")
ax2.set_ylim(90, 248)
ax2.tick_params(axis="y", labelcolor=color_bar)

# 基准线（不在 artist 上写 label，用 proxy 保证图例一定出现）
ax2.axhline(y=full_schema_correct, color="green", linestyle="--", alpha=0.5)
ax1.axhline(y=full_schema_avg_time, color="orange", linestyle=":", linewidth=2)
proxy_correct = Line2D(
    [0], [0], color="green", linestyle="--", alpha=0.5,
    label=f"全量Schema正确数量 {full_schema_correct}"
)
proxy_time = Line2D(
    [0], [0], color="orange", linestyle=":", linewidth=2,
    label=f"全量Schema平均消耗时间 {full_schema_avg_time:.2f}s"
)

ax1.set_xlim(df["k"].min() - 2, df["k"].max() + 5)
ax2.text(
    df["k"].min(),
    full_schema_correct + 1.0,
    f"全量Schema正确数量: {full_schema_correct}",
    color="green",
    fontweight="bold",
    va="bottom",
    ha="left",
)
ax1.text(
    df["k"].min(),
    full_schema_avg_time + 0.05,
    f"全量Schema平均消耗时间: {full_schema_avg_time:.2f}s",
    color="orange",
    fontweight="bold",
    va="bottom",
    ha="left",
)

ax1.set_xlabel("Top-K 参数值", fontsize=12, fontweight="bold")
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(
    h1 + h2 + [proxy_time, proxy_correct],
    l1 + l2 + [proxy_time.get_label(), proxy_correct.get_label()],
    loc="upper left",
    frameon=True,
    shadow=True,
)

ax1.grid(axis="y", linestyle="--", alpha=0.4)
ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.set_title("模型剪枝性能分析：消耗时间 vs 正确数", fontsize=15, pad=15)

for _, row in df.iterrows():
    ax1.text(
        row["k"],
        row["avg_cost_time_seconds"] + 0.015,
        f"{row['avg_cost_time_seconds']:.4f}",
        ha="center",
        va="bottom",
        fontsize=8,
        color=color_line,
    )
    ax2.text(
        row["k"],
        row[correct_col] + 2,
        f"{row[correct_col]:.0f}",
        ha="center",
        va="bottom",
        fontsize=8,
        color=color_bar,
    )

plt.tight_layout()
plt.show()
