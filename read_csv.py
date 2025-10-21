import csv

count = {}
text_dict = {}
# 打开CSV文件并读取
with open('data/flickr/flickr_val.csv', mode='r') as file:
    reader = csv.reader(file)

    # 遍历文件中的每一行
    for row in reader:
        if row[0] == 'imgid':
            continue
        else:
            if row[0] not in count.keys():
                count[row[0]] = 1
                text_dict[row[0]] = [row[2]]
            else:
                count[row[0]] += 1
                text_dict[row[0]].append(row[2])

for i in count.keys():
    if count[i] != 5:
        print(i)
        print(text_dict[i])

print(len(count.keys()))
text_sum = 0
for i in text_dict.keys():
    text_sum += len(text_dict[i])
print(text_sum)

