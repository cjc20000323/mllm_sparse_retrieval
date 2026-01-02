import matplotlib.pyplot as plt

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
hybrid_recall_t2i = [85.9, 86.8, 86.7, 86.4, 86.4]
hybrid_recall_i2t = [91.2, 91.6, 91.9, 91.6, 91.3]
hybrid_recall_mean = [88.5, 89.2, 89.3, 89.0, 88.8]

# 设置大字体
plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

colors = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeeb", " #f2a7da"]

ax.plot(length, hybrid_recall_t2i, marker='o', label='t2i', color=colors[5], linestyle='-', linewidth=3, markersize=8)
ax.plot(length, hybrid_recall_i2t, marker='o', label='i2t', color=colors[4], linestyle='-', linewidth=3, markersize=8)
ax.plot(length, hybrid_recall_mean, marker='o', label='mean', color=colors[1], linestyle='-', linewidth=3, markersize=8)

ax.set_xlabel('Length', fontsize=25)
ax.set_ylabel('r@5', fontsize=25)
plt.xticks([10, 20, 30, 40, 50])
# ax[0].set_xticks(x_axis)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.grid(True)

ax.legend(['t2i', 'i2t', 'mean'], loc='lower right', ncol=1, fontsize=25)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整整体布局以防止重叠
plt.savefig('flickr_length.pdf', format='pdf')

hybrid_recall_t2i = [24.7, 22.6, 21.6, 22.0, 23.4]

# 设置大字体
plt.rcParams.update({'font.size': 12})

fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))

colors = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeeb", " #f2a7da"]

ax1.plot(length, hybrid_recall_t2i, marker='o', label='t2i', color=colors[5], linestyle='-', linewidth=3, markersize=8)

ax1.set_xlabel('Length', fontsize=25)
ax1.set_ylabel('r@5', fontsize=25)
plt.xticks([10, 20, 30, 40, 50])
plt.yticks([21, 22, 23, 24, 25])
ax1.tick_params(axis='both', which='major', labelsize=30)
# ax[0].set_xticks(x_axis)
ax1.legend(['t2i'], loc='lower right', ncol=1, fontsize=25)
ax1.grid(True)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整整体布局以防止重叠
plt.savefig('rstpreid_length.pdf', format='pdf')