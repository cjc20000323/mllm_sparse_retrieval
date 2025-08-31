import os
import pickle
import torch
import random
import json
from tqdm import tqdm


def main():
    lookup_indices = []
    encoded = []
    jsonl_data = []

    for indice in tqdm(range(1000000)):
        lookup_indices.append(indice)
        encoded.append(torch.randn([1, 4096]).cpu().detach().float().numpy())
        vector = dict()
        for _ in range(128):
            token = random.randint(0, 128257)
            value = random.randint(0, 300)
            vector[token] = value
        jsonl_data.append(
            dict(
                id=indice,
                content="",
                vector=vector,
            )
        )

    os.makedirs(
        f'./dense_output/llava-hf-llama3-llava-next-8b-hf/flickr/image/no_filter/large/0_no_manual_128_sum_no_cluster_after_pad_all_disassembleeol_lower',
        exist_ok=True)
    os.makedirs(
        f'./sparse_output/llava-hf-llama3-llava-next-8b-hf/flickr/image/no_filter/large/0_no_manual_128_sum_no_cluster_after_pad_all_disassembleeol_lower',
        exist_ok=True)
    with open(os.path.join(
        f'./dense_output/llava-hf-llama3-llava-next-8b-hf/flickr/image/no_filter/large/0_no_manual_128_sum_no_cluster_after_pad_all_disassembleeol_lower',
        f'corpus_0.pkl'), 'wb') as f:
        pickle.dump((encoded, lookup_indices), f)

    with open(os.path.join(
        f'./sparse_output/llava-hf-llama3-llava-next-8b-hf/flickr/image/no_filter/large/0_no_manual_128_sum_no_cluster_after_pad_all_disassembleeol_lower',
        f'corpus_0.jsonl'), 'w') as f:
        for data in tqdm(jsonl_data):
            f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()