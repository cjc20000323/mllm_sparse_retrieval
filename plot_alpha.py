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
hybrid_flickr_mistral = []
hybrid_coco_mistral = []