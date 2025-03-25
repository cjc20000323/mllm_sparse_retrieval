import torch
import torch.nn.functional as F
import requests
from PIL import Image
from transformers import AutoTokenizer
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, LlavaProcessor, LlavaForConditionalGeneration, LlavaNextImageProcessor
from transformers.utils import PaddingStrategy

llama3_template = '<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n \n'

processor = LlavaNextProcessor.from_pretrained('./checkpoints/llava-hf-llava-v1.6-vicuna-7b-hf')
model = LlavaNextForConditionalGeneration.from_pretrained('./checkpoints/llava-hf-llava-v1.6-vicuna-7b-hf', torch_dtype=torch.float16).cuda()

img_prompt = llama3_template.format('<image>\nSummary above image in one word: ')
text_prompt = llama3_template.format('<sent>\nSummary above sentence in one word: ')

print(text_prompt)
print(img_prompt)

urls = []
images = Image.open('data/flickr/36979.jpg').convert('RGB')
image = processor.image_processor(images, return_tensors='pt')['pixel_values'].cuda()
# image = processor.image_processor(images, return_tensors='pt')
print(image.shape)

tokenizer = processor.tokenizer
input_id = tokenizer(text_prompt)['input_ids']
print(input_id)
print(tokenizer.decode(input_id))

texts = ['A dog sitting in the grass.']

text_inputs = processor([text_prompt.replace('<sent>', text) for text in texts], return_tensors="pt", padding=True).to('cuda')
print(text_inputs['input_ids'].shape)
img_inputs = processor(images=images, text=[img_prompt], return_tensors="pt", padding=True).to('cuda')
print(img_inputs.keys())
print(img_inputs['pixel_values'].shape)
print(img_inputs['input_ids'].shape)

with torch.no_grad():
    text_embs = model(**text_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
    print(text_embs.shape)
    # image = image.unsqueeze(0).cuda()
    # img_inputs = {'input_ids': img_inputs['input_ids'], 'attention_mask': img_inputs['attention_mask'], 'pixel_values': image, 'image_sizes': img_inputs['image_sizes']}
    img_embs = model(**img_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
    output = model(**img_inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, -1, :]
    print(img_embs.shape)
    print(output.shape)
    # print(output.shape)

    text_embs = F.normalize(text_embs, dim=-1)
    img_embs = F.normalize(img_embs, dim=-1)

print(text_embs @ img_embs.t())
with open('output.txt', 'w') as file:
    # 写入文本内容
    file.write(str(processor.tokenizer.vocab))

print("文件已成功写入。")
