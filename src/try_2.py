import torch
import numpy as np

a = torch.rand(4, 5)
print(a)
word_set = set()
top_k_values, top_k_indices = a.topk(2, dim=-1)
print(top_k_indices[0].tolist())
for top_k_indice_list in top_k_indices:
    print(top_k_indice_list)
    word_set.update(top_k_indice_list.tolist())
print(word_set)

'''
token_ids_in_text = torch.tensor([1, 3])
print(a[0].shape)
top_k_values, top_k_indices = a[:, token_ids_in_text].topk(2, dim=-1)
print(top_k_values)
print(top_k_indices)
values = np.rint(top_k_values.cpu().detach().float().numpy() * 100).astype(int)
print(values)
print(top_k_indices[0].numpy())
for indice_list, value_list in zip(token_ids_in_text[top_k_indices.cpu().detach().float().numpy()], values):
    print(indice_list)
    print(value_list)
    for indice, value in zip(indice_list, value_list):
        print(indice)
        print(value)
'''
from template import prompt_generation_from_text_prompt, prompt_generation_from_image_prompt
print(prompt_generation_from_text_prompt)
print(prompt_generation_from_image_prompt)

text1 = 'I love Wang Rui.'
new_prompt = prompt_generation_from_text_prompt.replace('<sent>', text1, 1)
print(new_prompt)
text2 = 'I want to fuck Wang Rui.'
new_prompt = new_prompt.replace('<sent>', text2, 1)
print(new_prompt)