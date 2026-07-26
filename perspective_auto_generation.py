import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import numpy as np

data = {
    "Condition": ["Dense", "MPP Sparse", "MPP Hybrid", "MPP Hybrid (human-made)"],
    "Flickr30K t2i": [60.5, 54.5, 64.1, 65.5],
    "Flickr30K i2t": [72.2, 63.4, 74.9, 75.4],
    "RSTPReid": [9.3, 11.7, 13.9, 14.8]
}
df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(11, 4.2))

# 图表设置
n_groups = len(df)  # 条件的数量
n_datasets = 3  # 数据集的数量
bar_width = 0.05  # 条形宽度
index = np.arange(n_datasets) * (n_groups * bar_width + bar_width)  # 数据集索引，确保有间隙
colors = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeeb", " #f2a7da"]
# 为每个条件绘制条形图
for i, condition in enumerate(df["Condition"]):
    offsets = index + (i+0.5) * bar_width
    bars = ax.bar(offsets, df.iloc[i, 1:], bar_width, label=condition, color=colors[i],hatch='',edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{str(height)[:]}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=18)

# 设置图表属性
# ax.set_xlabel('Datasets', fontsize=20)
ax.set_ylabel('r@1', fontsize=25)
ax.set_ylim(0, df.iloc[:, 1:].to_numpy().max() * 1.20)
# ax.set_title('Hits@1 by Condition and Dataset')
ax.set_xticks(index + n_groups * bar_width / 2)
ax.set_xticklabels(['Flickr30K t2i', 'Flickr30k i2t', 'RSTPReid'],fontsize=25)
ax.tick_params(axis='y', labelsize=20)
# ax.legend( loc='upper center',ncol=3, fontsize=20, bbox_to_anchor=(0.5, 1.3),handlelength=8.5, handletextpad=1)
fig.legend(['Dense', 'MPP Sparse', 'MPP Hybrid', 'MPP Hybrid (manual)'],
           loc='upper center', bbox_to_anchor=(0.5, 0.98), ncol=2,
           fontsize=22)
plt.tight_layout(rect=[0, 0, 1, 0.72])
plt.savefig('perspective_auto_generation.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)

