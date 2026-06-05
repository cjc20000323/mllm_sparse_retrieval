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

alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
hybrid_flickr_t2i_mistral = [59.7, 62.0, 63.5, 65.0, 65.5, 65.7, 65.0, 64.2, 62.9]
hybrid_flickr_i2t_mistral = [70.9, 73.1, 74.2, 75.0, 74.8, 75.4, 75.2, 74.5, 73.6]
hybrid_rstpreid_mistral = [9.4, 9.9, 10.7, 10.6, 11.2, 11.5, 11.1, 10.8, 10.6]

hybrid_flickr_t2i_llama = [53.2, 55.8, 58.0, 59.8, 61.1, 62.0, 62.2, 61.8, 60.3]
hybrid_flickr_i2t_llama = [58.0, 60.8, 64.3, 67.0, 69.9, 72.1, 72.3, 73.6, 73.1]
hybrid_rstpreid_llama = [7.0, 7.5, 7.9, 7.6, 7.7, 7.5, 7.8, 7.7]

fig, ax = plt.subplots(1, 1, figsize=(20, 5))

colors = ["#f57c6e", "#f2b56f", "#fae69e", "#84c3b7", "#88d8db", "#71b7ed", "#b8aeeb", " #f2a7da"]
ax.plot(alpha, hybrid_flickr_t2i_mistral, label='image retrieval', color=colors[0], linestyle='-', linewidth=3, marker='o', markersize=8)
ax.plot(alpha, hybrid_flickr_i2t_mistral, label='text retrieval', color=colors[1], linestyle='-', linewidth=3, marker='o', markersize=8)
# ax.plot(alpha, hybrid_rstpreid_mistral, label='rstpreid', color=colors[2], linestyle='-', linewidth=3, markersize=8)
# ax.plot(alpha, hybrid_flickr_t2i_llama, label='image retrieval llama', color=colors[2], linestyle='-', linewidth=3, markersize=8)
# ax.plot(alpha, hybrid_flickr_i2t_llama, label='text retrieval llama', color=colors[3], linestyle='-', linewidth=3, markersize=8)
ax.set_xlabel('alpha', fontsize=25)
ax.set_ylabel('recall@1', fontsize=25)
ax.tick_params(axis='both', which='major', labelsize=30)
ax.grid(True)
ax.legend(['image retrieval', 'text retrieval'], loc='lower right', fontsize=25)
plt.savefig('alpha_ablation.pdf', format='pdf', bbox_inches='tight', pad_inches=0.05)