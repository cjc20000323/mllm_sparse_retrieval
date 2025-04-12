import numpy as np
from matplotlib import pyplot as plt
import csv
import random

data_dict = {}
data_save = [
    ['img', 'filename', 'caption', 'sentid']
]
with open('data/flickr/flickr_train.csv') as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == 'imgid':
            continue

        else:
            if row[0] not in data_dict:
                data_dict[row[0]] = [row]
            else:
                data_dict[row[0]].append(row)
key_list = list(data_dict.keys())
random.seed(0)

few_shot_sum = 100
indices = random.sample(range(1, len(data_dict.keys())), few_shot_sum)
print(indices)
for i in indices:
    k = key_list[i]
    v = data_dict[k]
    print(v)
    for data in v:
        data_save.append(data)

print(data_save)
with open(f'data/flickr/flickr_train_{few_shot_sum}.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data_save)  # 将数据逐行写入CSV文件