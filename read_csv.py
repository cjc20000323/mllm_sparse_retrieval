import csv

count = {}
text_dict = {}
# 打开CSV文件并读取
with open('data/coco/coco_test.csv', mode='r') as file:
    reader = csv.reader(file)

    # 遍历文件中的每一行
    for row in reader:
        if row[0] == 'imgid':
            continue
        else:
            if row[0] not in count.keys():
                count[row[0]] = 1
                text_dict[row[0]] = [row[3]]
            else:
                count[row[0]] += 1
                text_dict[row[0]].append(row[3])

for i in count.keys():
    if count[i] != 5:
        print(i)
        print(text_dict[i])

print(len(count.keys()))

