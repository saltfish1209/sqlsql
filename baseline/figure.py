import pandas as pd
import matplotlib.pyplot as plt

# 【解决中文问题】
preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Microsoft JhengHei",
        "Noto Sans CJK SC",
    ]
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('topk_pruned_metrics1.csv')
plt.style.use('seaborn-v0_8-white')

fig ,ax1 = plt.subplots(figsize=(15,6))

# --- ax1: 准确率 ---
color_line = "#D62728"
ax1.plot(df["k"], df["accuracy"], color=color_line, marker="o", linewidth=2, label="准确率 (Accuracy)")
ax1.set_ylabel("Top-K 准确率", color=color_line, fontsize=12, fontweight='bold')
ax1.set_ylim(0.3, 0.8)
ax1.tick_params(axis='y', labelcolor=color_line)

# --- ax2: 正确数量 ---
ax2 = ax1.twinx()
color_bar = "#4682B4"
ax2.bar(df["k"], df["correct248"], color=color_bar, alpha=0.6, label="正确数量 (Correct)")
ax2.set_ylabel("Top-K 正确数", color=color_bar, fontsize=12, fontweight='bold')
ax2.set_ylim(90, 260)
ax2.tick_params(axis='y', labelcolor=color_bar)

# --- 添加参考横线 ---
# 注意：一定要在 legend 之前定义 label
ax2.axhline(y=142, color='green', linestyle='--', alpha=0.5, label='全量Schema正确数量')
ax1.axhline(y=0.5726, color='orange', linestyle=':', linewidth=2, label='全量Schema准确率')

# --- 【解决数字太近问题】通过在 y 上加减 offset 来调整位置 ---
# va='bottom' 表示文字底部对齐坐标点，即文字在线上方
ax2.text(df['k'].max() + 0.5, 142 + 3, '142', color='green', fontweight='bold', va='bottom', ha='left')
ax1.text(df['k'].min() - 1, 0.5726 + 0.01, '0.5726', color='orange', fontweight='bold', va='bottom', ha='right')

# --- 【解决图例与标签问题】 ---
ax1.set_xlabel("Top-K 参数值", fontsize=12, fontweight='bold') # 显式设置 X 轴
h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper left', frameon=True, shadow=True)

# 基础美化
ax1.grid(axis='y', linestyle='--', alpha=0.4)
ax1.spines["top"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax1.set_title('模型剪枝性能分析：准确率 vs 正确数', fontsize=15, pad=15)

# 数据点标注 (微调了 offset 让它更美观)
for i, row in df.iterrows():
    ax1.text(row['k'], row['accuracy'] + 0.015, f"{row['accuracy']:.2f}",
             ha='center', va='bottom', fontsize=8, color=color_line)
    ax2.text(row['k'], row['correct248'] + 2, f"{row['correct248']:.0f}",
             ha='center', va='bottom', fontsize=8, color=color_bar)

plt.tight_layout()
plt.show()