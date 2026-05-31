import pandas as pd
from PIL import Image
from io import BytesIO

df_query = pd.read_parquet("D:/数据集/混合模态检索/UMRB-ReMuQ/query.parquet")
df_corpus = pd.read_parquet("D:/数据集/混合模态检索/UMRB-ReMuQ/corpus.parquet")

for idx, row in df_query.iterrows():
    print(row['id'])
    print(row['text'])
    print(row['image'])
    img = Image.open(BytesIO(row['image']["bytes"])).convert("RGB")
    img.show()
    break

corpus_set = set()
for idx, row in df_corpus.iterrows():
    if row['id'] in corpus_set:
        print(row['id'])
    corpus_set.add(row['id'])
print(len(corpus_set))