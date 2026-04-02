import matplotlib.pyplot as plt
import numpy as np

x = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 67, 68, 70, 71, 72, 73, 74, 77, 79, 85])  # 生成从0到10的100个等间距点

y = np.array([3.9724862575531006, 3.9724862575531006, 3.9724862575531006, 3.6767919063568115, 3.505838632583618, 2.989149808883667, 2.83364200592041, 2.827512502670288, 2.588071584701538, 2.5301289558410645, 2.3861355781555176, 2.477884292602539, 2.367269277572632, 2.261726140975952, 2.2691142559051514, 2.284722089767456, 2.3927650451660156, 2.332524061203003, 2.297410011291504, 2.2876994609832764, 2.237255096435547, 2.32297945022583, 2.2496631145477295, 2.2665224075317383, 2.275254249572754, 2.2914764881134033, 2.31320858001709, 2.323814630508423, 2.2712512016296387, 2.368576765060425, 2.424960136413574, 2.3999671936035156, 2.356038808822632, 2.4250874519348145, 2.2944157123565674, 2.3815205097198486, 2.4473068714141846] + [2.410219669342041] * 2 + [2.3822968006134033] * 3 + [2.4836671352386475] * 4 + [2.493242025375366] * 27)

# 设置大字体
plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(1, 2, figsize=(14, 6))
colors = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeeb", " #f2a7da"]

ax[1].plot(x, -y, marker='o', color=colors[0], linestyle='-', linewidth=3, markersize=8)

ax[1].set_xlabel('Tokens Sum', fontsize=22)
ax[1].set_ylabel('average log-likelihood', fontsize=22)
# ax[0].set_xticks(x_axis)
ax[1].tick_params(axis='both', which='major', labelsize=25)
ax[1].grid(True)

# ax.legend(['average log-likelihood'], loc='lower right', ncol=1, fontsize=25)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整整体布局以防止重叠
plt.savefig('flickr_bias.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import numpy as np

data = {
    "Condition": ["t2i", "i2t"],
    "RGP r@1": [73.9, 79.2],
    "CGP r@1": [79.1, 75.0],
    "modified CGP r@1": [79.1, 81.5]
}
df = pd.DataFrame(data)

# 图表设置
n_groups = len(df)  # 条件的数量
n_datasets = 3  # 数据集的数量
bar_width = 0.05  # 条形宽度
index = np.arange(n_datasets) * (n_groups * bar_width + bar_width)  # 数据集索引，确保有间隙
colors = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeeb", " #f2a7da"]

# 为每个条件绘制条形图
for i, condition in enumerate(df["Condition"]):
    offsets = index + (i+0.5) * bar_width
    bars = ax[0].bar(offsets, df.iloc[i, 1:], bar_width, label=condition, color=colors[i],hatch='',edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        ax[0].annotate(f'{str(height)[:]}',
                    xy=(bar.get_x() + bar.get_width() / 2, height-0.8),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=20)
# 设置图表属性
# ax.set_xlabel('Datasets', fontsize=20)
ax[0].set_ylabel('r@1', fontsize=25)
# ax.set_ylim([0.7, 0.88])
# ax.set_title('Hits@1 by Condition and Dataset')
ax[0].set_xticks(index + n_groups * bar_width / 2)
ax[0].set_xticklabels(['RGP', 'CGP', 'Modified CGP'],fontsize=25)
ax[0].tick_params(axis='y', labelsize=20)
# ax.legend( loc='upper center',ncol=3, fontsize=20, bbox_to_anchor=(0.5, 1.3),handlelength=8.5, handletextpad=1)
ax[0].axhline(y=74.1, color="blue", linestyle='--', label='i2t hybrid')
ax[0].text(-0.065, 74.1, f'{74.1}', color='black',
         va='bottom', ha='left', fontsize=18)
ax[0].axhline(y=64.4, color="green", linestyle='--', label='t2i hybrid')
ax[0].text(-0.065, 64.4, f'{64.4}', color='black',
         va='bottom', ha='left', fontsize=18)
ax[0].legend(['i2t hybrid', 't2i hybrid', 't2i', 'i2t'], ncol=1, fontsize=25)
plt.tight_layout()
plt.savefig('flickr_bias.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)