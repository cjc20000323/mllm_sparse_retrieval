import matplotlib.pyplot as plt
import numpy as np

modalities = [
    ("structure_weight_l", "structure_weight_r"),
    ("relation_weight_l", "relation_weight_r"),
    ("attribute_weight_l", "attribute_weight_r"),
    ("image_weight_l", "image_weight_r")
]
ordered_modalities = [
    ("name_weight_l", "name_weight_r", "Name"),
    ("image_weight_l", "image_weight_r", "Image"),
    ("structure_weight_l", "structure_weight_r", "Structure"),
    ("relation_weight_l", "relation_weight_r", "Relation"),
    ("attribute_weight_l", "attribute_weight_r", "Attribute"),
]
low_saturation_colors = [
    "#F4C2C2",  # Light Red
    "#FADFAD",  # Light Yellow
    "#ADD8E6",  # Light Blue
    "#90EE90",  # Light Green
    "#D8BFD8"   # Light Purple
]

x = np.linspace(0, 200000000000)
y1 = 768 * 2 * x  # 绘制二次函数 y = x^2
y2 = 150 * 2 * x + 4096 * 2 * 200
# y3 = 150 * 2 * x + 4096 * 2 * 200 + 5 * (2 * 8000000000 * 2000)
y3 = 150 * 2 * x + 4096 * 2 * 200 + 5 * (32 * (8 * 4096 * 4096 + 6 * 4096 * 14336) + 2 * 4096 * 128256) * 2000

# 设置大字体
plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(1, 1, figsize=(13, 5))

colors = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeeb", " #f2a7da"]

ax.plot(x, y1, label='CLIP', color=colors[5], linestyle='-', linewidth=3, markersize=8)
ax.plot(x, y3, label='three pipelines', color=colors[0], linestyle='-', linewidth=3, markersize=8)
ax.set_xlabel('Scale', fontsize=25)
ax.set_ylabel('FLOP', fontsize=25)
# ax[0].set_xticks(x_axis)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.grid(True)

ax.legend(['CLIP', 'Re-M (Three-stage)'], loc='lower right', fontsize=25)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整整体布局以防止重叠
plt.text(0.3, 0.3,  # 坐标位置 (图形坐标，0-1之间)
         r'$f_1(x)=768 \times 2s$',  # LaTeX公式
         transform=plt.gca().transAxes,  # 使用坐标轴坐标
         fontsize=20,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.text(0, 0.65,  # 坐标位置 (图形坐标，0-1之间)
         r'$f_2(x)=150\times2s+4096\times 2\times200+5\times2\times8B\times2000$',  # LaTeX公式
         transform=plt.gca().transAxes,  # 使用坐标轴坐标
         fontsize=18,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
plt.savefig('flop_rerank.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)

x = np.linspace(0, 9000)
y1 = 1436 * x  # 绘制二次函数 y = x^2
y2 = 300 * x + 4096 * 2 * 200

fig, ax = plt.subplots(1, 1, figsize=(15, 10))

colors = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeeb", " #f2a7da"]

ax.plot(x, y1, label='CLIP', color=colors[1], linestyle='-', linewidth=3, markersize=8)
ax.plot(x, y2, label='sparse+hybrid', color=colors[4], linestyle='-', linewidth=3, markersize=8)

ax.set_xlabel('Scale', fontsize=25)
ax.set_ylabel('FLOP', fontsize=25)
# ax[0].set_xticks(x_axis)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.grid(True)

ax.legend(['CLIP', 'sparse+hybrid'], loc='lower right', fontsize=25)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整整体布局以防止重叠
plt.savefig('flop_retriever.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)