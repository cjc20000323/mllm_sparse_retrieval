import json
import csv

from tqdm import tqdm

caption_list = []

with open('D:\数据集\合成检索\\fashion-iq\captions-20220326T130604Z-001\captions/cap.dress.val.json') as file:
    fashion_iq_dataset = json.load(file)

    for item in fashion_iq_dataset:
        for caption in caption_list:
            for text in caption:
                for text_1 in item['captions']:
                    if text == text_1:
                        print(text)
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