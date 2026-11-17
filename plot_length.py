from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent

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

length = [10, 20, 30, 40, 50]
# R@200 proportions from the supplied table, converted to percentages.
hybrid_recall_t2i = [value * 100 for value in [0.9738, 0.9806, 0.9826, 0.9848, 0.9890]]
hybrid_recall_i2t = [value * 100 for value in [0.990, 0.994, 0.994, 0.995, 0.993]]
hybrid_recall_mean = [(t2i + i2t) / 2 for t2i, i2t in zip(hybrid_recall_t2i, hybrid_recall_i2t)]

# 设置大字体
plt.rcParams.update({'font.size': 12, 'pdf.fonttype': 42, 'ps.fonttype': 42})

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

colors = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeeb", " #f2a7da"]

ax.plot(length, hybrid_recall_t2i, marker='o', label='t2i', color=colors[0], linestyle='-', linewidth=3, markersize=8)
ax.plot(length, hybrid_recall_i2t, marker='o', label='i2t', color=colors[5], linestyle='-', linewidth=3, markersize=8)
ax.plot(length, hybrid_recall_mean, marker='o', label='mean', color=colors[1], linestyle='-', linewidth=3, markersize=8)

ax.set_xlabel('Sparse Representation Length k', fontsize=25)
ax.set_ylabel('R@200', fontsize=25)
ax.set_ylim(96, 100)
plt.xticks([10, 20, 30, 40, 50])
# ax[0].set_xticks(x_axis)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.grid(True)

ax.legend(['t2i', 'i2t', 'mean'], loc='lower right', ncol=1, fontsize=25)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整整体布局以防止重叠
fig.tight_layout()
for extension in ('pdf', 'png'):
    fig.savefig(OUTPUT_DIR / f'flickr_length.{extension}', format=extension,
                dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig)

rstpreid_recall = [value * 100 for value in [0.9130, 0.9155, 0.9285, 0.9295, 0.9335]]
llava_recall = [value * 100 for value in
                [0.969140625, 0.987890625, 0.989648438, 0.989257813, 0.987304688]]

# 设置大字体
plt.rcParams.update({'font.size': 12})

fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5))

colors = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeeb", " #f2a7da"]

ax1.plot(length, rstpreid_recall, marker='o', label='RSTPReid', color=colors[5], linestyle='-', linewidth=3, markersize=8)
ax1.plot(length, llava_recall, marker='o', label='LLaVA', color=colors[0], linestyle='-', linewidth=3, markersize=8)

ax1.set_xlabel('Sparse Representation Length k', fontsize=25)
ax1.set_ylabel('R@200', fontsize=25)
plt.xticks([10, 20, 30, 40, 50])
# plt.yticks([21, 22, 23, 24, 25])
ax1.tick_params(axis='both', which='major', labelsize=30)
# ax[0].set_xticks(x_axis)
ax1.legend(loc='center right', ncol=1, fontsize=25)
ax1.grid(True)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整整体布局以防止重叠
fig1.tight_layout()
for extension in ('pdf', 'png'):
    fig1.savefig(OUTPUT_DIR / f'rstpreid_length.{extension}', format=extension,
                 dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig1)
