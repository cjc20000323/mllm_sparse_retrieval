import json
import csv

from tqdm import tqdm

caption_list = []

with open('D:\数据集\文本行人识别\RSTPReid/data_captions.json') as file:
    cuhk_pedes_dataset = json.load(file)
    print(cuhk_pedes_dataset)
    for item in cuhk_pedes_dataset:
        print(item)
        print(item.keys())

    print(len(cuhk_pedes_dataset))

text_img_prompt = "Image of: <sent>"
caption = [["a cat.", "a dog?"]]
input_texts = [', '.join([cc.strip('.?, ') for cc in c]) for c in caption][0]
print(input_texts)