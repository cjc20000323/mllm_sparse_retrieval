import ast
import json

with open('llava-hf-llama3-llava-next-8b-hf_flickr_text_False_9_sparse_search_correct_results_0.txt', 'r', encoding='utf-8') as f:
    loaded_data_1 = json.load(f)

with open('llava-hf-llama3-llava-next-8b-hf_flickr_text_False_9_sparse_search_wrong_results_0.txt', 'r', encoding='utf-8') as f:
    loaded_data_2 = json.load(f)

with open('llava-hf-llama3-llava-next-8b-hf_flickr_text_False_9_hybrid_search_correct_results_0.txt', 'r', encoding='utf-8') as f:
    loaded_data_3 = json.load(f)

with open('llava-hf-llama3-llava-next-8b-hf_flickr_text_False_9_hybrid_search_wrong_results_0.txt', 'r', encoding='utf-8') as f:
    loaded_data_4 = json.load(f)

with open('llava-hf-llama3-llava-next-8b-hf_flickr_text_False_9_dense_search_correct_results_0.txt', 'r', encoding='utf-8') as f:
    loaded_data_5 = json.load(f)

with open('llava-hf-llama3-llava-next-8b-hf_flickr_text_False_9_dense_search_wrong_results_0.txt', 'r', encoding='utf-8') as f:
    loaded_data_6 = json.load(f)

print(len(loaded_data_3))
print(len(loaded_data_4))

count = [[], [], [], [], [], [], [], []]

for k, v in loaded_data_3.items():
    if k in loaded_data_1.keys() and k in loaded_data_5.keys():
        '''
        print(k)
        print(v)
        print(loaded_data_5[k])
        print(loaded_data_1[k])
        '''
        count[0].append(k)

    elif k in loaded_data_5.keys() and k in loaded_data_2.keys():
        '''
        print(k)
        print(v)
        print(loaded_data_5[k])
        print(loaded_data_2[k])
        '''
        count[1].append(k)

    elif k in loaded_data_6.keys() and k in loaded_data_1.keys():
        '''
        print(k)
        print(v)
        print(loaded_data_6[k])
        print(loaded_data_1[k])
        '''
        count[2].append(k)

    elif k in loaded_data_6.keys() and k in loaded_data_2.keys():
        print(k)
        print(v)
        print(loaded_data_6[k])
        print(loaded_data_2[k])
        count[3].append(k)

for k, v in loaded_data_4.items():
    if k in loaded_data_1.keys() and k in loaded_data_5.keys():
        '''
        print(k)
        print(v)
        print(loaded_data_5[k])
        print(loaded_data_1[k])
        '''
        count[4].append(k)

    elif k in loaded_data_5.keys() and k in loaded_data_2.keys():
        '''
        print(k)
        print(v)
        print(loaded_data_5[k])
        print(loaded_data_2[k])
        '''
        count[5].append(k)

    elif k in loaded_data_6.keys() and k in loaded_data_1.keys():
        '''
        print(k)
        print(v)
        print(loaded_data_6[k])
        print(loaded_data_1[k])
        '''
        count[6].append(k)

    elif k in loaded_data_6.keys() and k in loaded_data_2.keys():
        '''
        print(k)
        print(v)
        print(loaded_data_6[k])
        print(loaded_data_2[k])
        '''
        count[7].append(k)

for i in count:
    print(len(i))

print()
print(len(loaded_data_5))

for k, v in loaded_data_1.items():
    if k == v['results']:
        print(k)

for k, v in loaded_data_2.items():
    if k == v['results']:
        print(k)