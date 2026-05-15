import csv
import matplotlib.pyplot as plt
from matplotlib import font_manager, rcParams


def _setup_chinese_font():
    """优先使用 Windows / Linux 常见中文字体，避免中文乱码。"""
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
    # 回退：即使没有中文字体也保证负号正常
    rcParams["axes.unicode_minus"] = False


def main():
    input_csv = r"/home/xs-a100/yj/nl2sql/baseline/topk_pruned_metrics.csv"
    output_png = r"/home/xs-a100/yj/nl2sql/baseline/topk_pruned_combo_chart.png"

    _setup_chinese_font()

    # 按用户要求：使用前 3 列（k / accuracy / correct）
    x_k = []
    y_acc = []
    y_correct = []
    with open(input_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # 跳过表头
        for row in reader:
            if not row:
                continue
            x_k.append(float(row[0]))
            y_acc.append(float(row[1]) * 100.0)
            y_correct.append(float(row[2]))

    full_schema_acc = 57.26
    full_schema_correct = 142

    fig, ax1 = plt.subplots(figsize=(20, 10))
    ax2 = ax1.twinx()

    bar_color = "#7EA1FF"
    line_color = "#1E3A8A"
    full_acc_color = "#D62728"
    full_correct_color = "#FF7F0E"

    bars = ax2.bar(
        x_k,
        y_correct,
        color=bar_color,
        alpha=0.55,
        width=1.5,
        label="TopK精简Schema正确数量",
        zorder=1,
    )

    line, = ax1.plot(
        x_k,
        y_acc,
        color=line_color,
        marker="o",
        linewidth=2.5,
        markersize=5,
        label="TopK精简Schema准确率",
        zorder=3,
    )

    h_acc = ax1.axhline(
        y=full_schema_acc,
        color=full_acc_color,
        linestyle="-",
        linewidth=2.2,
        label=f"全量Schema准确率 {full_schema_acc:.2f}%",
        zorder=2,
    )
    h_correct = ax2.axhline(
        y=full_schema_correct,
        color=full_correct_color,
        linestyle="--",
        linewidth=2.2,
        label=f"全量Schema正确数量 {full_schema_correct:.2f}",
        zorder=2,
    )

    # 左轴：准确率
    ax1.set_xlabel("TopK", fontsize=12)
    ax1.set_ylabel("准确率(%)", fontsize=12, color=line_color)
    ax1.set_ylim(30, 80)
    ax1.tick_params(axis="y", labelcolor=line_color)

    # 右轴：正确数量
    ax2.set_ylabel("正确数量", fontsize=12, color="#334155")
    ax2.set_ylim(90, 248)
    ax2.tick_params(axis="y", labelcolor="#334155")

    # 右侧坐标轴位置红点：正确数量轴一半位置
    y_half = 248 / 2.0
    ax2.plot(
        1.0,
        y_half,
        "o",
        color="red",
        markersize=8,
        transform=ax2.get_yaxis_transform(),
        clip_on=False,
        zorder=5,
    )
    ax2.text(
        1.01,
        y_half,
        f"{y_half:.2f}",
        color="red",
        fontsize=10,
        va="center",
        transform=ax2.get_yaxis_transform(),
    )

    # 折线数据标注（2位小数）
    for x, y in zip(x_k, y_acc):
        ax1.annotate(
            f"{y:.2f}%",
            xy=(x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color=line_color,
        )

    # 柱状数据标注（2位小数）
    for b in bars:
        h = b.get_height()
        ax2.annotate(
            f"{h:.2f}",
            xy=(b.get_x() + b.get_width() / 2.0, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1F2937",
        )

    # 全量schema基线标注
    ax1.text(
        min(x_k),
        full_schema_acc + 0.5,
        f"全量Schema准确率: {full_schema_acc:.2f}%",
        color=full_acc_color,
        fontsize=10,
        va="bottom",
    )
    ax2.text(
        min(x_k),
        full_schema_correct + 1.0,
        f"全量Schema正确数量: {full_schema_correct:.2f}",
        color=full_correct_color,
        fontsize=10,
        va="bottom",
    )

    # 网格与图例（合并双轴的 handles 确保所有 label 都显示）
    ax1.grid(axis="y", linestyle="--", alpha=0.3)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", fontsize=10)

    ax1.set_title("TopK精简Schema组合图（准确率与正确数量）", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    print(output_png)


if __name__ == "__main__":
    main()
