import csv
import os

from torch.utils.data import Dataset
import torch
from PIL import Image
from arguments import coco_file_path, flickr_file_path
from template import llama3_template, text_prompt, img_prompt, text_prompt_no_one_word, img_prompt_no_one_word


class CrossModalRetrievalDataset(Dataset):

    def __init__(self, data_name, processor, split, mode):
        '''

        :param data_name: 指定数据集的名字，例如coco，flickr
        :param tokenizer: 指定模型的tokenizer
        :param processor:
        :param split: 说明当前的数据集是哪一部分的，是train,val还是test
        :param mode: 说明当前数据集取数据是1to1还是5to5
        '''
        super(CrossModalRetrievalDataset, self).__init__()
        self.data_name = data_name
        assert self.data_name in ['coco', 'flickr']
        self.split = split
        if self.data_name == 'coco':
            self.data_path = coco_file_path
        elif self.data_name == 'flickr':
            self.data_path = flickr_file_path
        else:
            ValueError('Data name is not in the candidates list.')
        self.img_dict = {}  # 保存数据集中图像id到图像的映射字典
        self.img_id_list = []  # 保存数据集中图像的id（是否要使用字典来直接映射）
        self.text_dict = {}  # 保存数据集中文本id到文本的映射字典
        self.text_id_list = []  # 保存数据集中文本的id（是否要使用字典来直接映射）
        self.img2text = {}  # 保存图像id到文本id的映射，表明搜索索引为id的图像时，希望查到的文本id
        self.text2img = {}  # 保存文本id到图像id的映射，报名搜索索引为id的文本时，希望查到的图像id
        self.img2filepath = {}  # 保存图像id的filepath字典
        # self.tokenizer = tokenizer  # 指定模型的tokenizer，分词并转成token id用
        self.processor = processor
        self.mode = mode  # mode为single的时候，长度按图像长度，获取文本时，找一个对应的就行，mode为full的时候，长度按文本数量来
        assert self.mode in ['single', 'full']
        with open(f'{self.data_path}' + f'{self.data_name}_{self.split}.csv', mode='r') as file:
            reader = csv.reader(file)
            # 遍历文件中的每一行
            for row in reader:
                if row[0] == 'imgid':
                    continue
                else:
                    if self.data_name == 'coco':
                        # 首先保存图像id和对应的图像文件名
                        if row[0] not in self.img_id_list:
                            self.img_id_list.append(row[0])
                        if row[0] not in self.img_dict.keys():
                            self.img_dict[row[0]] = row[2]

                        # 然后保存文本id和对应的文本
                        self.text_id_list.append(row[4])
                        self.text_dict[row[4]] = row[3]

                        # 保存图像id到文本id的映射以及到文件路径的映射
                        if row[0] not in self.img2text.keys():
                            self.img2text[row[0]] = [row[4]]
                        else:
                            self.img2text[row[0]].append(row[4])
                        if row[0] not in self.img2filepath.keys():
                            self.img2filepath[row[0]] = row[1]
                        # 文搜图是1对1的，所以这里应该不需要else，保存文本id到图像id的映射
                        self.text2img[row[4]] = row[0]
                    else:  # 这里处理的是flickr数据集
                        # 首先保存图像id和对应的图像文件名
                        if row[0] not in self.img_id_list:
                            self.img_id_list.append(row[0])
                        if row[0] not in self.img_dict.keys():
                            self.img_dict[row[0]] = row[1]

                        # 然后保存文本id和对应的文本
                        self.text_id_list.append(row[3])
                        self.text_dict[row[3]] = row[2]

                        # 保存图像id到文本id的映射，flickr没有到图像路径的映射，所以去掉
                        if row[0] not in self.img2text.keys():
                            self.img2text[row[0]] = [row[3]]
                        else:
                            self.img2text[row[0]].append(row[3])
                        # 文搜图是1对1的，所以这里应该不需要else，保存文本id到图像id的映射
                        self.text2img[row[3]] = row[0]

    def __len__(self):
        if self.mode == 'single':
            return len(self.img_id_list)
        elif self.mode == 'full':
            return len(self.text_id_list)
        else:
            ValueError('Mode is not either single or full.')

    def __getitem__(self, idx):
        '''
        这个数据集是想做图文检索，所以必然取出的数据会有图，应该不需要再进行分类讨论
        由于不像原始llava训练的数据集中包含有conversation字段，所以后续应该想办法适配一下，可能工作量较大
        '''
        if self.mode == 'single':
            img_id = self.img_id_list[idx]
            img_name = self.img_dict[img_id]
            if self.data_name == 'coco':
                img_file_path = self.img2filepath[img_id]
                image_path = f'./data/{self.data_name}/{img_file_path}/{img_name}'
            else:
                image_path = f'./data/{self.data_name}/flickr30k-images/{img_name}'
            text_id = self.img2text[img_id][0]  # 这个模式下，拿出第一个对应的文本即可
            text = self.text_dict[text_id]
            return text, image_path, text_id, img_id
        elif self.mode == 'full':
            text_id = self.text_id_list[idx]
            text = self.text_dict[text_id]
            img_id = self.text2img[text_id]
            img_name = self.img_dict[img_id]
            if self.data_name == 'coco':
                img_file_path = self.img2filepath[img_id]
                image_path = f'./data/{self.data_name}/{img_file_path}/{img_name}'
            else:
                image_path = f'./data/{self.data_name}/flickr30k-images/{img_name}'
            return text, image_path, text_id, img_id
        else:
            ValueError('Mode is not either single or full.')

        # 根据github上对E5-V的观察，他们似乎使用了直接从huggingface上加载processor处理文本和图像，但是这里要想获得图像

        # text_with_prompt = text_prompt.replace('<sent>', text)
        # text_input = self.processor(text_with_prompt, return_tensors="pt", padding=True)

        '''
        image = Image.open(image_path).convert('RGB')
        # image_tensor = self.processor.image_processor(image, return_tensors='pt')['pixel_values'][0]
        img_input = self.processor(images=image, text=[img_prompt],  return_tensors="pt", padding=True)
        print(img_input['pixel_values'].shape)
        if img_input['pixel_values'].shape == torch.Size([1, 3, 3, 336, 336]):
            print(image_path)
        print(img_input['input_ids'].shape)
        print(img_input['attention_mask'].shape)
        '''

        # image exist in the data
        # 这里将text原文输出出去，到外面在组成一个批次的张量，避免在这里形成张量大小不一致还要调整，使用原本的processor拆分后返回各个
        # return text, img_input['pixel_values'], img_input['input_ids'], img_input['attention_mask'], text_id, img_id
        # return text, img_input, text_id, img_id
        # return text, image_path, text_id, img_id

    def get_target(self, idx, query_type):
        if query_type == 'text':
            return self.text2img[idx]
        else:
            return self.img2text[idx]

    def get_text(self, idx):
        return self.text_dict[idx]


    def get_image(self, idx):
        return self.img_dict[idx]
