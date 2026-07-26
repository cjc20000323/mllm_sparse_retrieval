import csv
import os
import json

import pandas as pd
from torch.utils.data import Dataset
import torch
from PIL import Image
import tevatron.retriever.arguments
from arguments import coco_file_path, flickr_file_path, fashion_iq_file_path, cuhk_pedes_file_path, \
    icfg_pedes_flie_path, rstpreid_file_path, webqa_file_path, remuq_file_path, llava_file_path, edis_file_path
from template import llama3_template, text_prompt, img_prompt, text_prompt_no_one_word, img_prompt_no_one_word, \
    img_prompt_no_special_llava_v1_5, img_prompt_qwen_v2_5, img_prompt_intern_vl_v2_5
from tevatron.retriever.dataset import EncodeDataset

from dataclasses import dataclass
from transformers import ProcessorMixin, LlavaProcessor, Qwen2_5_VLProcessor

from tevatron.retriever.collator import TrainCollator
from arguments import dataset_path_prefix


@dataclass
class CrossModalRetrievalDataset(Dataset):

    def __init__(self, data_name, processor, split, mode, data_args=None):
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
        if data_args is not None:
            if data_args.use_few_shot:
                if data_args.dataset_suffix == 'no':
                    self.dataset_file = f'{self.data_path}' + f'{self.data_name}_{self.split}_{data_args.few_shot_sum}.csv'
                else:
                    self.dataset_file = f'{self.data_path}' + f'{self.data_name}_{self.split}_{data_args.few_shot_sum}_{data_args.dataset_suffix}.csv'
            else:
                self.dataset_file = f'{self.data_path}' + f'{self.data_name}_{self.split}.csv'
        else:
            self.dataset_file = f'{self.data_path}' + f'{self.data_name}_{self.split}.csv'
        print(self.dataset_file)
        with open(self.dataset_file, mode='r') as file:
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
                image_path = dataset_path_prefix + f'data/{self.data_name}/{img_file_path}/{img_name}'
            else:
                image_path = dataset_path_prefix + f'data/{self.data_name}/flickr30k-images/{img_name}'
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
                image_path = dataset_path_prefix + f'data/{self.data_name}/{img_file_path}/{img_name}'
            else:
                image_path = dataset_path_prefix + f'data/{self.data_name}/flickr30k-images/{img_name}'
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


class ComposedTextImageRetrievalDataset(Dataset):
    def __init__(self, data_name, processor, split, mode, data_args=None):
        '''

        :param data_name: 指定数据集的名字，例如coco，flickr
        :param tokenizer: 指定模型的tokenizer
        :param processor:
        :param split: 说明当前的数据集是哪一部分的，是train,val还是test
        :param mode: 说明当前数据集取数据是1to1还是5to5
        '''
        super(ComposedTextImageRetrievalDataset, self).__init__()
        self.data_name = data_name
        assert self.data_name in ['fashion-iq']
        self.split = split
        if self.data_name == 'fashion-iq':
            self.data_path = fashion_iq_file_path
        else:
            ValueError('Data name is not in the candidates list.')
        self.img_dict = {}  # 保存数据集中图像id到图像的映射字典
        self.img_id_list = []  # 保存数据集中图像的id（是否要使用字典来直接映射）,对于fashion-iq数据集，我们直接使用图像文件名做id
        self.text_dict = {}  # 保存数据集中文本id到文本的映射字典
        self.text_id_list = []  # 保存数据集中文本的id（是否要使用字典来直接映射），对于fashion-iq数据集，我们需要手动构造下id
        self.composed_id_list = []  # 保存合成图像的id
        self.composed2img = {}  # 保存基础图像id和修改文本id到目标图像id的映射
        # self.tokenizer = tokenizer  # 指定模型的tokenizer，分词并转成token id用
        self.processor = processor
        self.dress_type_dict = {}
        self.mode = mode  # mode为single的时候，长度按图像长度，获取文本时，找一个对应的就行，mode为full的时候，长度按文本数量来
        assert self.mode in ['composed', 'image']
        if data_args is not None:
            if data_args.use_few_shot:
                if data_args.dataset_suffix == 'no':
                    self.dataset_file = [
                        f'{self.data_path}' + f'captions-20220326T130604Z-001/captions/cap.dress.{self.split}.json',
                        f'{self.data_path}' + f'captions-20220326T130604Z-001/captions/cap.shirt.{self.split}.json',
                        f'{self.data_path}' + f'captions-20220326T130604Z-001/captions/cap.toptee.{self.split}.json']
                else:
                    self.dataset_file = [
                        f'{self.data_path}' + f'captions-20220326T130604Z-001/captions/cap.dress.{self.split}.json',
                        f'{self.data_path}' + f'captions-20220326T130604Z-001/captions/cap.shirt.{self.split}.json',
                        f'{self.data_path}' + f'captions-20220326T130604Z-001/captions/cap.toptee.{self.split}.json']
            else:
                self.dataset_file = [
                    f'{self.data_path}' + f'captions-20220326T130604Z-001/captions/cap.dress.{self.split}.json',
                    f'{self.data_path}' + f'captions-20220326T130604Z-001/captions/cap.shirt.{self.split}.json',
                    f'{self.data_path}' + f'captions-20220326T130604Z-001/captions/cap.toptee.{self.split}.json']
        else:
            self.dataset_file = [
                f'{self.data_path}' + f'captions-20220326T130604Z-001/captions/cap.dress.{self.split}.json',
                f'{self.data_path}' + f'captions-20220326T130604Z-001/captions/cap.shirt.{self.split}.json',
                f'{self.data_path}' + f'captions-20220326T130604Z-001/captions/cap.toptee.{self.split}.json']

        self.dataset_split_file = [
            f'{self.data_path}' + f'image_splits-20220326T130551Z-001/image_splits/split.dress.{self.split}.json',
            f'{self.data_path}' + f'image_splits-20220326T130551Z-001/image_splits/split.shirt.{self.split}.json',
            f'{self.data_path}' + f'image_splits-20220326T130551Z-001/image_splits/split.toptee.{self.split}.json']
        print(self.dataset_file)

        # 下面先把split中的图像保存下来
        for dataset_split_file in self.dataset_split_file:
            with open(dataset_split_file, mode='r') as file:
                reader = json.load(file)

                for id in reader:
                    self.img_id_list.append(id)
                    self.img_dict[id] = id + '.png'
                    if 'shirt' in dataset_split_file:
                        self.dress_type_dict[id] = 'shirt'
                    elif 'toptee' in dataset_split_file:
                        self.dress_type_dict[id] = 'toptee'
                    else:
                        self.dress_type_dict[id] = 'dress'

        # 然后处理it2t的映射关系以及文本的保存
        count = 0

        for dataset_file in self.dataset_file:
            with open(dataset_file, mode='r') as file:
                reader = json.load(file)

                for item in reader:
                    self.text_id_list.append(str(count))
                    self.text_dict[str(count)] = item['captions']
                    self.composed_id_list.append(item['candidate'] + '_' + str(count))
                    self.composed2img[item['candidate'] + '_' + str(count)] = item['target']
                    count += 1

        # print(self.composed2img.keys())
        # print(self.composed_id_list)

    def __len__(self):
        if self.mode == 'composed':
            return len(self.composed_id_list)
        else:
            return len(self.img_id_list)

    def __getitem__(self, idx):
        if self.mode == 'composed':
            composed_id = self.composed_id_list[idx]
            target_id = self.composed2img[composed_id]
            indice = composed_id.index('_')
            img_id = composed_id[:indice]
            text_id = composed_id[indice + 1:]
            img_name = self.img_dict[img_id]
            target_name = self.img_dict[target_id]
            text = [', '.join([cc.strip('.?, ') for cc in c]) for c in [self.text_dict[text_id]]][0].lower()
            # text = self.text_dict[text_id][0] + ' ' + self.text_dict[text_id][1]
            # target_name = self.img_dict[target_name]
            image_path = f'./data/{self.data_name}/images/images/{img_name}'
            target_path = f'./data/{self.data_name}/images/images/{target_name}'
            # target_id = self.composed2img[img_id]  # 这个模式下，拿出第一个对应的文本即可
            dress_type = self.dress_type_dict[img_id]
            if dress_type == 'toptee':
                dress_type = 'shirt'
        else:
            img_id = self.img_id_list[idx]
            img_name = self.img_dict[img_id]
            image_path = f'./data/{self.data_name}/images/images/{img_name}'
            count = 0
            text = ''
            target_path = ''
            text_id = ''
            composed_id = ''
            dress_type = self.dress_type_dict[img_id]
            if dress_type == 'toptee':
                dress_type = 'shirt'
        return text, image_path, target_path, text_id, img_id, composed_id, dress_type

    def get_image(self, idx):
        return self.img_dict[idx]

    def get_text(self, idx):
        return [', '.join([cc.strip('.?, ') for cc in c]) for c in [self.text_dict[idx]]][0].lower()

    def get_target(self, idx):
        return self.composed2img[idx]

    def get_dress_type(self, idx):
        dress_type = self.dress_type_dict[idx]
        return dress_type


class TextPersonRetrievalDataset(Dataset):
    def __init__(self, data_name, processor, split, mode, data_args=None):
        '''

        :param data_name: 指定数据集的名字，例如coco，flickr
        :param tokenizer: 指定模型的tokenizer
        :param processor:
        :param split: 说明当前的数据集是哪一部分的，是train,val还是test
        :param mode: 说明当前数据集取数据是1to1还是5to5
        '''
        super(TextPersonRetrievalDataset, self).__init__()
        self.data_name = data_name
        assert self.data_name in ['CUHK-PEDES', 'ICFG-PEDES', 'RSTPReid']
        self.split = split
        if self.data_name == 'CUHK-PEDES':
            self.data_path = cuhk_pedes_file_path
        elif self.data_name == 'ICFG-PEDES':
            self.data_path = icfg_pedes_flie_path
        elif self.data_name == 'RSTPReid':
            self.data_path = rstpreid_file_path
        else:
            ValueError('Data name is not in the candidates list.')
        self.img_dict = {}  # 保存数据集中图像id到图像的映射字典
        self.img_id_list = []  # 保存数据集中图像的id（是否要使用字典来直接映射）,对于行人检索数据集，可能需要手动构造下id
        self.text_dict = {}  # 保存数据集中文本id到文本的映射字典
        self.text_id_list = []  # 保存数据集中文本的id（是否要使用字典来直接映射），对于行人检索数据集，可能需要手动构造下id
        self.img2text = {}  # 保存图像id到文本id的映射，表明搜索索引为id的图像时，希望查到的文本id
        self.text2img = {}  # 保存文本id到图像id的映射，报名搜索索引为id的文本时，希望查到的图像id
        self.img2person = {}  # 保存数据集中图像id倒行人id的字典
        self.processor = processor
        self.mode = mode  # mode为single的时候，长度按图像长度，获取文本时，找一个对应的就行，mode为full的时候，长度按文本数量来

        img_id_count = 0
        text_id_count = 0

        assert self.mode in ['single', 'full']
        if data_args is not None:
            if self.data_name == 'CUHK-PEDES':
                self.dataset_file = f'{self.data_path}' + f'reid_raw.json'
            elif self.data_name == 'ICFG-PEDES':
                self.dataset_file = f'{self.data_path}' + f'ICFG-PEDES.json'
            elif self.data_name == 'RSTPReid':
                self.dataset_file = f'{self.data_path}' + f'data_captions.json'
            else:
                self.dataset_file = f'{self.data_path}' + f'reid_raw.json'
        else:
            if self.data_name == 'CUHK-PEDES':
                self.dataset_file = f'{self.data_path}' + f'reid_raw.json'
            elif self.data_name == 'ICFG-PEDES':
                self.dataset_file = f'{self.data_path}' + f'ICFG-PEDES.json'
            elif self.data_name == 'RSTPReid':
                self.dataset_file = f'{self.data_path}' + f'data_captions.json'
            else:
                self.dataset_file = f'{self.data_path}' + f'reid_raw.json'

        print(self.dataset_file)

        with open(self.dataset_file, mode='r') as file:
            reader = json.load(file)
            if self.data_name == 'CUHK-PEDES' or self.data_name == 'ICFG-PEDES':
                for item in reader:
                    if item['split'] == self.split:
                        self.img_id_list.append(str(img_id_count))
                        self.img_dict[str(img_id_count)] = item['file_path']
                        self.img2person[str(img_id_count)] = item['id']
                        for caption in item['captions']:
                            self.text_id_list.append(str(text_id_count))
                            self.text_dict[str(text_id_count)] = caption
                            if str(img_id_count) not in self.img2text.keys():
                                self.img2text[str(img_id_count)] = [str(text_id_count)]
                            else:
                                self.img2text[str(img_id_count)].append(str(text_id_count))
                            if str(text_id_count) not in self.text2img.keys():
                                self.text2img[str(text_id_count)] = str(img_id_count)
                            text_id_count += 1
                        img_id_count += 1
            elif self.data_name == 'RSTPReid':
                for item in reader:
                    if item['split'] == self.split:
                        self.img_id_list.append(str(img_id_count))
                        self.img_dict[str(img_id_count)] = item['img_path']
                        self.img2person[str(img_id_count)] = item['id']
                        for caption in item['captions']:
                            self.text_id_list.append(str(text_id_count))
                            self.text_dict[str(text_id_count)] = caption
                            if str(img_id_count) not in self.img2text.keys():
                                self.img2text[str(img_id_count)] = [str(text_id_count)]
                            else:
                                self.img2text[str(img_id_count)].append(str(text_id_count))
                            if str(text_id_count) not in self.text2img.keys():
                                self.text2img[str(text_id_count)] = str(img_id_count)
                            text_id_count += 1
                        img_id_count += 1
            else:
                pass

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
            if self.data_name == 'CUHK-PEDES' or self.data_name == 'ICFG-PEDES' or self.data_name == 'RSTPReid':
                image_path = dataset_path_prefix + f'/data/{self.data_name}/imgs/{img_name}'
            else:
                image_path = dataset_path_prefix + f'data/{self.data_name}/imgs/{img_name}'
            text_id = self.img2text[img_id][0]  # 这个模式下，拿出第一个对应的文本即可
            text = self.text_dict[text_id]
            return text, image_path, text_id, img_id
        elif self.mode == 'full':
            text_id = self.text_id_list[idx]
            text = self.text_dict[text_id]
            img_id = self.text2img[text_id]
            img_name = self.img_dict[img_id]
            if self.data_name == 'CUHK-PEDES' or self.data_name == 'ICFG-PEDES' or self.data_name == 'RSTPReid':
                image_path = dataset_path_prefix + f'data/{self.data_name}/imgs/{img_name}'
            else:
                image_path = dataset_path_prefix + f'data/{self.data_name}/imgs/{img_name}'
            return text, image_path, text_id, img_id
        else:
            ValueError('Mode is not either single or full.')

    def get_text(self, idx):
        return self.text_dict[idx]

    def get_image(self, idx):
        return self.img_dict[idx]

    def get_target_from_text(self, idx):
        # 因为行人检索任务是只有文搜图过程，因此输入文本的id，search中的各个run字典应该会以文本id为键值
        img_id = self.text2img[idx]
        return self.img2person[img_id]

    def get_target_from_image(self, idx):
        return self.img2person[idx]


class Text2ImagetextRetrievalDataset(Dataset):
    def __init__(self, data_name, processor, split, mode, data_args=None):
        super(Text2ImagetextRetrievalDataset, self).__init__()
        self.data_name = data_name
        assert self.data_name in ['webqa', 'edis']
        self.processor = processor
        self.split = split
        self.id2query = {}
        self.query_id_list = []
        self.id2candidate = {}  # 字典里面保存的还是字典，一个是候选图像一个是候选文本
        self.candidate_id_list = []
        self.query2candidate = {}  # 字典Key保存输入，保存输出
        if self.data_name == 'webqa':
            self.data_path = webqa_file_path
        elif self.data_name == 'edis':
            self.data_path = edis_file_path
        else:
            ValueError('Data name is not in the candidates list.')

        self.mode = mode
        if self.data_name == 'webqa':
            self.dataset_file = {
                'query': self.data_path + 'query-00000-of-00001.parquet',
                'rel': self.data_path + 'qrels-00000-of-00001.parquet',
                'corpus': [self.data_path + 'corpus-00000-of-00002.parquet',
                           self.data_path + 'corpus-00001-of-00002.parquet']
            }
        elif self.data_name == 'edis':
            self.dataset_file = {
                'query': self.data_path + 'query-00000-of-00001.parquet',
                'rel': self.data_path + 'qrels-00000-of-00001.parquet',
                'corpus': [self.data_path + 'corpus-00000-of-00004.parquet',
                           self.data_path + 'corpus-00001-of-00004.parquet',
                           self.data_path + 'corpus-00002-of-00004.parquet',
                           self.data_path + 'corpus-00003-of-00004.parquet',
                           ]
            }
        else:
            self.dataset_file = {
                'query': self.data_path + 'query-00000-of-00001.parquet',
                'rel': self.data_path + 'qrels-00000-of-00001.parquet',
                'corpus': [self.data_path + 'corpus-00000-of-00002.parquet',
                           self.data_path + 'corpus-00001-of-00002.parquet']
            }

        df_query = pd.read_parquet(self.dataset_file['query'])
        for idx, row in df_query.iterrows():
            self.id2query[row['id']] = row['text']
            self.query_id_list.append(row['id'])
        df_rel = pd.read_parquet(self.dataset_file['rel'])
        target_set = set()
        for idx, row in df_rel.iterrows():
            if row['query-id'] not in self.query2candidate.keys():
                self.query2candidate[row['query-id']] = [row['corpus-id']]
            else:
                self.query2candidate[row['query-id']].append(row['corpus-id'])
            target_set.add(row['corpus-id'])


        for corpus_path in self.dataset_file['corpus']:
            df_corpus = pd.read_parquet(corpus_path)
            for idx, row in df_corpus.iterrows():
                if row['id'] in target_set:
                    self.id2candidate[row['id']] = {'text': row['text'], 'image': row['image']}
                    self.candidate_id_list.append(row['id'])
                '''
                self.id2candidate[row['id']] = {'text': row['text'], 'image': row['image']}
                self.candidate_id_list.append(row['id'])
                '''

    def __len__(self):
        print(self.mode)
        if self.mode == 'query':
            return len(self.query_id_list)
        else:
            return len(self.candidate_id_list)

    def __getitem__(self, idx):
        if self.mode == 'query':
            query_id = self.query_id_list[idx]
            query_text = self.id2query[query_id]
            return query_text, query_id
        else:
            corpus_id = self.candidate_id_list[idx]
            corpus_text = self.id2candidate[corpus_id]['text']
            corpus_image = self.id2candidate[corpus_id]['image']["bytes"]
            return corpus_text, corpus_image, corpus_id

    def get_target(self, idx):
        return self.query2candidate[idx]

    def get_query(self, idx):
        return self.id2query[idx]

    def get_candidate(self, idx):
        return self.id2candidate[idx]


class Imagetext2TextRetrievalDataset(Dataset):
    def __init__(self, data_name, processor, split, mode, data_args=None):
        super(Imagetext2TextRetrievalDataset, self).__init__()
        self.data_name = data_name
        assert self.data_name in ['remuq', 'llava']
        self.split = split
        self.id2query = {}
        self.query_id_list = []
        self.id2candidate = {}  # 字典里面保存的还是字典，一个是候选图像一个是候选文本
        self.candidate_id_list = []
        self.query2candidate = {}  # 字典Key保存输入，保存输出
        if self.data_name == 'remuq':
            self.data_path = remuq_file_path
        elif self.data_name == 'llava':
            self.data_path = llava_file_path
        else:
            ValueError('Data name is not in the candidates list.')

        self.mode = mode
        if self.data_name == 'remuq':
            self.dataset_file = {
                'query': self.data_path + 'query.parquet',
                'rel': self.data_path + 'qrels.parquet',
                'corpus': self.data_path + 'corpus.parquet'
            }
        else:
            self.dataset_file = {
                'query': self.data_path + 'query.parquet',
                'rel': self.data_path + 'qrels.parquet',
                'corpus': self.data_path + 'corpus.parquet'
            }

        df_query = pd.read_parquet(self.dataset_file['query'])
        for idx, row in df_query.iterrows():
            self.id2query[row['id']] = {'text': row['text'], 'image': row['image']}
            self.query_id_list.append(row['id'])
        df_rel = pd.read_parquet(self.dataset_file['rel'])
        for idx, row in df_rel.iterrows():
            self.query2candidate[row['query-id']] = row['corpus-id']

        df_corpus = pd.read_parquet(self.dataset_file['corpus'])
        for idx, row in df_corpus.iterrows():
            self.id2candidate[row['id']] = row['text']
            self.candidate_id_list.append(row['id'])


    def __len__(self):
        if self.mode == 'query':
            return len(self.query_id_list)
        else:
            return len(self.candidate_id_list)

    def __getitem__(self, idx):
        if self.mode == 'query':
            query_id = self.query_id_list[idx]
            query_text = self.id2query[query_id]['text']
            query_image = self.id2query[query_id]['image']["bytes"]
            return query_text, query_image, query_id
        else:
            corpus_id = self.candidate_id_list[idx]
            corpus_text = self.id2candidate[corpus_id]
            return corpus_text, corpus_id

    def get_target(self, idx):
        return self.query2candidate[idx]

    def get_query(self, idx):
        return self.id2query[idx]

    def get_candidate(self, idx):
        return self.id2candidate[idx]


@dataclass
class PromptRepsTrainCollator:
    processor: ProcessorMixin
    model_args: tevatron.retriever.arguments.ModelArguments
    device: torch.device

    def __call__(self, features):
        texts = []
        imgs_path = []
        text_ids = []
        image_ids = []
        for feature in features:
            texts.append(feature[0])
            imgs_path.append(feature[1])
            text_ids.append(feature[2])
            image_ids.append(feature[3])

        if 'llava-hf-llava-1.5-7b-hf' in self.model_args.model_name_or_path or 'llava-hf-llava-v1.6-vicuna-7b-hf' in self.model_args.model_name_or_path:
            prompt = img_prompt_no_special_llava_v1_5
        elif 'Qwen2.5-VL-7B-Instruct' in self.model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in self.model_args.model_name_or_path:
            prompt = img_prompt_qwen_v2_5
        elif 'InternVL2_5-8B' in self.model_args.model_name_or_path or 'InternVL2_5-4B' in self.model_args.model_name_or_path:
            prompt = img_prompt_intern_vl_v2_5
        else:
            prompt = img_prompt
        raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
        img_inputs = self.processor(images=raw_images, text=[prompt] * len(imgs_path), return_tensors="pt",
                                    padding=True)
        imgs = img_inputs
        labels = torch.zeros(len(texts))

        return {'texts': texts, 'imgs': imgs, 'text_ids': text_ids, 'image_ids': image_ids, 'labels': labels}
