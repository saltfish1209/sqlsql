import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rcParams


def _setup_chinese_font() -> None:
    """检测已安装字体后再设 sans-serif（含常见 Linux 中文字体）。"""
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


_setup_chinese_font()

# 1. 加载数据
df = pd.read_csv('bar.csv')
models = df['模型'].tolist()
correct_counts = df['正确数量'].tolist()
avg_times = df['平均耗时'].tolist()

# 2. 设置分组位置 (0,1 为第一组 | 3,4 为第二组 | 6 为第三组)
x_bases = np.array([0, 1, 3, 4, 6])
width = 0.35  # 单个柱子的宽度

# 3. 优化横坐标 Label：将长名称中的下划线替换为换行符，以便水平显示
wrapped_labels = [label.replace('_', '\n').replace(' ', '\n') for label in models]

# 4. 创建画布与双 Y 轴
fig, ax1 = plt.subplots(figsize=(14, 8))
ax2 = ax1.twinx()

# 5. 绘制并列柱状图
# 正确数量（左轴 - 蓝色系列）
bar1 = ax1.bar(x_bases - width/2, correct_counts, width=width, label='正确数量', color='#5B8FF9', alpha=0.85)
# 平均耗时（右轴 - 红色系列）
bar2 = ax2.bar(x_bases + width/2, avg_times, width=width, label='平均耗时', color='#E86452', alpha=0.85)

# 6. 设置轴标签与颜色区分
ax1.set_ylabel('正确数量 (个)', fontsize=12, color='#5B8FF9', fontweight='bold')
ax2.set_ylabel('平均耗时 (s)', fontsize=12, color='#E86452', fontweight='bold')
ax1.tick_params(axis='y', labelcolor='#5B8FF9')
ax2.tick_params(axis='y', labelcolor='#E86452')
ax1.set_ylim(90, 260)
# 7. 设置横坐标（平行显示且分组）
ax1.set_xticks(x_bases)
ax1.set_xticklabels(wrapped_labels, rotation=0, fontsize=10)
ax1.set_xlabel('模型版本对比', fontsize=12, labelpad=15)

# 8. 添加数值标注
def add_labels(rects, ax, color, unit=""):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}{unit}' if height % 1 != 0 else f'{int(height)}{unit}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, color=color, fontweight='bold')

add_labels(bar1, ax1, '#2E5BCC')
add_labels(bar2, ax2, '#B0392B', "s")


# 10. 完善细节
plt.title('模型性能对比分析 (优化版柱状图)', fontsize=16, pad=30)
ax1.grid(axis='y', linestyle='--', alpha=0.3)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True)

plt.tight_layout()
plt.show()