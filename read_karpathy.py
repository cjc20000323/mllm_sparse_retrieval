import json
import csv

from tqdm import tqdm

with open('karpathy/dataset_flickr30k.json') as file:
    coco_dataset = json.load(file)

    print(coco_dataset.keys())

train_output_data = [['imgid', 'filename', 'caption', 'sentid']]
val_output_data = [['imgid', 'filename', 'caption', 'sentid']]
test_output_data = [['imgid', 'filename', 'caption', 'sentid']]
train_count = 0
restval_count = 0
val_count = 0
test_count = 0
coco_data = coco_dataset['images']
sentid_set = set()
for data in tqdm(coco_data):
    # filepath = data['filepath']
    if train_count == 0:
        print(data.keys())
    sentids = data['sentids']
    for id in sentids:
        if id in sentid_set:
            print('There is something wrong.')
        else:
            sentid_set.add(id)
    filename = data['filename']
    imgid = data['imgid']
    split = data['split']
    sentences = []
    for sentence in data['sentences']:
        sentences.append(sentence['raw'])
    if split == 'train':
        train_count += 1
    elif split == 'restval':
        restval_count += 1
    elif split == 'val':
        val_count += 1
    else:
        test_count += 1
    for id in range(len(sentences)):
        if split == 'val':
            val_output_data.append([imgid, filename, sentences[id], sentids[id]])
        elif split == 'test':
            test_output_data.append([imgid, filename, sentences[id], sentids[id]])
        else:
            train_output_data.append([imgid, filename, sentences[id], sentids[id]])

print(train_count)
print(restval_count)
print(val_count)
print(test_count)
print(train_count + restval_count)

with open('data/flickr/flickr_train.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(train_output_data)  # 将数据逐行写入CSV文件

with open('data/flickr/flickr_val.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(val_output_data)  # 将数据逐行写入CSV文件

with open('data/flickr/flickr_test.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(test_output_data)  # 将数据逐行写入CSV文件