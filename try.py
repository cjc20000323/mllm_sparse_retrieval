import json
import csv

id_set = set()
with open(f'data/coco/coco_test.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[0] == 'imgid':
            continue
        else:
            id_set.add(row[4])

print(len(id_set))

with open('corpus_0.jsonl', 'r',  encoding="utf-8") as f:
    datas = f.readlines()

count = 0
id_set_1 = set()
for data in datas:
    dict_data = json.loads(data)
    count += 1
    id_set_1.add(dict_data['id'])
    if dict_data['id'] not in id_set:
        print(dict_data['id'])

print(count)

with open(f'data/coco/coco_test.csv', mode='r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[0] == 'imgid':
            continue

        else:
            if row[4] not in id_set_1:
                print(row[4])
