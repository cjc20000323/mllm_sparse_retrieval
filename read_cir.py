import json
import csv

from tqdm import tqdm

caption_list = []

with open('D:\数据集\合成检索\\fashion-iq\captions-20220326T130604Z-001\captions/cap.dress.val.json') as file:
    fashion_iq_dataset = json.load(file)

    for item in fashion_iq_dataset:
        caption_list.append(item['captions'])

    print(len(fashion_iq_dataset))

with open('D:\数据集\合成检索\\fashion-iq\captions-20220326T130604Z-001\captions/cap.shirt.val.json') as file:
    fashion_iq_dataset = json.load(file)

    for item in fashion_iq_dataset:
        caption_list.append(item['captions'])

    print(len(fashion_iq_dataset))

with open('D:\数据集\合成检索\\fashion-iq\captions-20220326T130604Z-001\captions/cap.toptee.val.json') as file:
    fashion_iq_dataset = json.load(file)

    for item in fashion_iq_dataset:
        caption_list.append(item['captions'])

    print(len(fashion_iq_dataset))

with open('D:\数据集\合成检索\\fashion-iq\image_splits-20220326T130551Z-001\image_splits/split.dress.val.json') as file:
    fashion_iq_dataset = json.load(file)

    print(len(fashion_iq_dataset))

with open('D:\数据集\合成检索\\fashion-iq\image_splits-20220326T130551Z-001\image_splits/split.shirt.val.json') as file:
    fashion_iq_dataset = json.load(file)

    print(len(fashion_iq_dataset))

with open('D:\数据集\合成检索\\fashion-iq\image_splits-20220326T130551Z-001\image_splits/split.toptee.val.json') as file:
    fashion_iq_dataset = json.load(file)

    print(len(fashion_iq_dataset))

# 一共15536个图像，可以被4整除