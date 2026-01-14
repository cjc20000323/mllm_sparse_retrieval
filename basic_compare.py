import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import numpy as np
# Define the data
data = {
    "Condition": ["Sparse","Hybrid"],
    "SSE(T) PTT(I) r@1": [25.0, 68.4],
    "PTT(T) PTT(I) r@1": [54.0, 66.0],
    "MPP(SSE(T) PTT(I)) r@1": [31.2, 69.4],
    "MPP(PTT(T) PTT(I)) r@1": [62.7, 70.4],
}
df = pd.DataFrame(data)

# 图表设置
n_groups = len(df)  # 条件的数量
n_datasets = 4  # 数据集的数量
bar_width = 0.05  # 条形宽度
index = np.arange(n_datasets) * (n_groups * bar_width + bar_width)  # 数据集索引，确保有间隙

# 创建图表
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.axhline(y=66.2, color='b', linestyle='--', label='Dense')
plt.text(-0.02, 66.2, f'{66.2}', color='black',
         va='bottom', ha='left', fontsize=18)
colors = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeeb", " #f2a7da"]
# 为每个条件绘制条形图
for i, condition in enumerate(df["Condition"]):
    offsets = index + (i+0.5) * bar_width
    bars = ax.bar(offsets, df.iloc[i, 1:], bar_width, label=condition, color=colors[i],hatch='',edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{str(height)[:]}',
                    xy=(bar.get_x() + bar.get_width() / 2, height-1.5),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=18)
# 设置图表属性
# ax.set_xlabel('Datasets', fontsize=20)
ax.set_ylabel('r@1', fontsize=25)
# ax.set_ylim([0.7, 0.88])
# ax.set_title('Hits@1 by Condition and Dataset')
ax.set_xticks(index + n_groups * bar_width / 2)
ax.set_xticklabels(['SSR(T) PTT(I)', 'PTT(T) PTT(I)', 'MPP(SSR(T) PTT(I))', 'MPP(PTT(T) PTT(I))'],fontsize=17)
ax.tick_params(axis='y', labelsize=20)
# ax.legend( loc='upper center',ncol=3, fontsize=20, bbox_to_anchor=(0.5, 1.3),handlelength=8.5, handletextpad=1)
ax.legend( loc='upper center',ncol=3, fontsize=20, bbox_to_anchor=(0.5, 1.15),handlelength=2, handletextpad=1)
plt.tight_layout()
plt.savefig('compare_flickr.pdf', format='pdf',  bbox_inches='tight', pad_inches=0.05)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = {
    "Condition": ["Sparse","Hybrid"],
    "SSE(T) PTT(I) r@1": [15.3, 73.7],
    "PTT(T) PTT(I) r@1": [59.4, 71.9],
    "MPP(SSE(T) PTT(I)) r@1": [28.2, 73.6],
    "MPP(PTT(T) PTT(I)) r@1": [68.4, 75.4],
}
df = pd.DataFrame(data)

# 图表设置
n_groups = len(df)  # 条件的数量
n_datasets = 4  # 数据集的数量
bar_width = 0.05  # 条形宽度
index = np.arange(n_datasets) * (n_groups * bar_width + bar_width)  # 数据集索引，确保有间隙

# 创建图表
fig, ax = plt.subplots(figsize=(20, 8))
colors = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeeb", " #f2a7da"]
ax.axhline(y=72.3, color='b', linestyle='--', label='Dense')
plt.text(-0.02, 72.3, f'{72.3}', color='black',
         va='bottom', ha='left', fontsize=18)
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
# ax.set_ylim([0.7, 0.88])
# ax.set_title('Hits@1 by Condition and Dataset')
ax.set_xticks(index + n_groups * bar_width / 2)
ax.set_xticklabels(['SSR(T) PTT(I)', 'PTT(T) PTT(I)', 'MPP(SSR(T) PTT(I))', 'MPP(PTT(T) PTT(I))'],fontsize=17)
ax.tick_params(axis='y', labelsize=20)
# ax.legend( loc='upper center',ncol=3, fontsize=20, bbox_to_anchor=(0.5, 1.3),handlelength=8.5, handletextpad=1)
ax.legend(loc='upper center',ncol=3, fontsize=20, bbox_to_anchor=(0.5, 1.2),handlelength=2, handletextpad=1)
plt.tight_layout()
plt.savefig('compare_i2t_flickr.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)