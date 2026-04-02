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

length = [1, 2, 3, 4,5, 6, 7]
hybrid_recall_t2i = [85.2, 85.8, 85.9, 86.2, 86.7, 86.1, 86.3]
hybrid_recall_i2t = [91.3, 91.2, 91.1, 91.3, 91.9, 91.5, 91.5]
hybrid_recall_mean = [(hybrid_recall_i2t[i] + hybrid_recall_t2i[i]) / 2 for i in range(len(hybrid_recall_i2t))]

# 设置大字体
plt.rcParams.update({'font.size': 12})

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

colors = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeeb", " #f2a7da"]

ax.plot(length, hybrid_recall_t2i, marker='o', label='t2i', color=colors[0], linestyle='-', linewidth=3, markersize=8)
ax.plot(length, hybrid_recall_i2t, marker='o', label='i2t', color=colors[5], linestyle='-', linewidth=3, markersize=8)
ax.plot(length, hybrid_recall_mean, marker='o', label='mean', color=colors[1], linestyle='-', linewidth=3, markersize=8)

ax.set_xlabel('Perspective Number L', fontsize=25)
ax.set_ylabel('r@5', fontsize=25)
plt.xticks([1, 2, 3, 4, 5, 6, 7])
# ax[0].set_xticks(x_axis)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.grid(True)

ax.legend(['t2i', 'i2t', 'mean'], loc='lower right', fontsize=25)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整整体布局以防止重叠
plt.savefig('flickr_prompt.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)

hybrid_recall_t2i = [20.4, 20.2, 20.4, 20.5, 24.7, 21.5, 21.8]
hybrid_recall_t2i_cuhk = [12.1, 12.7, 13.0, 13.9, 15.4, 14.2, 13.8]

# 设置大字体
plt.rcParams.update({'font.size': 12})

fig1, ax1 = plt.subplots(1, 1, figsize=(8, 5))

colors = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeeb", " #f2a7da"]

ax1.plot(length, hybrid_recall_t2i, marker='o', label='t2i', color=colors[5], linestyle='-', linewidth=3, markersize=8)
ax1.plot(length, hybrid_recall_t2i_cuhk, marker='o', label='t2i', color=colors[0], linestyle='-', linewidth=3, markersize=8)

ax1.set_xlabel('Perspective Number L', fontsize=25)
ax1.set_ylabel('r@5', fontsize=25)
plt.xticks([1, 2, 3, 4, 5, 6, 7])
ax1.tick_params(axis='both', which='major', labelsize=30)
# ax[0].set_xticks(x_axis)
ax1.legend(['RSTPReid', 'CUHK-PEDES'], ncol=1, fontsize=25)
ax1.grid(True)
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整整体布局以防止重叠
plt.savefig('rstpreid_prompt.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)

