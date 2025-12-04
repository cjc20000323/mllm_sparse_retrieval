import json
import csv

from tqdm import tqdm

caption_list = []
test_count = 0
val_count = 0
caption_count = 0
with open('D:\数据集\文本行人识别\ICFG-PDES\ICFG-PEDES/ICFG-PEDES.json') as file:
    cuhk_pedes_dataset = json.load(file)
    print(cuhk_pedes_dataset)
    for item in cuhk_pedes_dataset:
        if item['split'] == 'test':
            test_count += 1
            if len(item['captions']) != 2:
                print(item['captions'])
            caption_count += len(item['captions'])
        elif item['split'] == 'val':
            val_count += 1

    print(len(cuhk_pedes_dataset))
    print(test_count)
    print(val_count)
    print(caption_count)

text_img_prompt = "Image of: <sent>"
caption = [["a cat.", "a dog?"]]
input_texts = [', '.join([cc.strip('.?, ') for cc in c]) for c in caption][0].lower()
print(input_texts)


# CUHK-PEDES 测试集3074个图像，验证集3078个图像
# ICFG-PEDES 测试集19848个图像，验证集0个图像
# RSTPReid 测试集1000个图像，验证集1000个图像