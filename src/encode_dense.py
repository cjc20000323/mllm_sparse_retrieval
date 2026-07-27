import math
import os
import pickle
import sys
from contextlib import nullcontext
from io import BytesIO

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data as Data
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode

torch.set_printoptions(threshold=sys.maxsize)
from PIL import Image
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
)
from transformers import CLIPModel, CLIPProcessor, Qwen2VLProcessor, \
    Qwen2VLForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration

from arguments import PromptRepsLLMDataArguments, ModelArguments
from arguments import TrainingArguments
from template import lamra_2_img_prompt, lamra_2_text_prompt, lamra_2_tbpr_prompt, lamra_2_5_text_prompt, \
    lamra_2_5_img_prompt, lamra_2_5_tbpr_prompt, vlm2vec_img_prompt, vlm2vec_text_prompt, vlm2vec_tbpr_prompt, \
    lamra_2_5_it2t_corpus_prompt, lamra_2_5_t2it_corpus_prompt, lamra_2_it2t_corpus_prompt, lamra_2_t2it_corpus_prompt
from dataset import CrossModalRetrievalDataset, TextPersonRetrievalDataset, Text2ImagetextRetrievalDataset, \
    Imagetext2TextRetrievalDataset
from models.blip_itm import blip_itm
from eva_clip import create_model_and_transforms, get_tokenizer
from sentence_transformers import SentenceTransformer
from peft import PeftModel

model_begin_indice = 28
path_prefix = '/root/autodl-fs/'


def blip_preprocess_image(raw_image, image_size, device):
    w, h = raw_image.size

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])
    image = transform(raw_image).unsqueeze(0).to(device)
    return image


def blip_load_image(image, image_size, device):
    raw_image = Image.open(str(image)).convert('RGB')
    return blip_preprocess_image(raw_image, image_size, device)


def qwen_safe_image(image, min_size=28):
    width, height = image.size
    if width >= min_size and height >= min_size:
        return image

    scale = max(min_size / width, min_size / height)
    resized_width = max(min_size, int(np.ceil(width * scale)))
    resized_height = max(min_size, int(np.ceil(height * scale)))
    resampling = getattr(Image, "Resampling", Image).BICUBIC
    return image.resize((resized_width, resized_height), resampling)


def patch_gme_remote_code(encoder):
    first_module = encoder._first_module()
    tokenize = getattr(first_module, "tokenize", None)
    if tokenize is None:
        return
    tokenize_func = getattr(tokenize, "__func__", tokenize)
    tokenize_func.__globals__["math"] = math


def main():
    parser = HfArgumentParser((ModelArguments, PromptRepsLLMDataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: PromptRepsLLMDataArguments
    training_args: TrainingArguments

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_map = "cuda"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(os.environ.get("WORLD_SIZE"))
    ddp = world_size != 1
    print(ddp)
    # if ddp and False:
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        # gradient_accumulation_steps = gradient_accumulation_steps // world_size

        if not dist.is_initialized():
            torch.distributed.init_process_group("nccl")
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        device_id = rank % torch.cuda.device_count()
        device = torch.device(device_id)
        torch.cuda.set_device(device)

        print(device)

    # 下面这部分指定采用的模型精度
    if training_args.bf16:
        torch_type = torch.bfloat16
    elif training_args.fp16:
        torch_type = torch.float16
    else:
        torch_type = torch.float32

    # 指定模型
    if 'eva' in model_args.model_name_or_path:
        encoder, _, processor = create_model_and_transforms('EVA02-CLIP-bigE-14-plus',
                                                            model_args.model_name_or_path + '/EVA02_CLIP_E_psz14_plus_s9B.pt',
                                                            force_custom_clip=True)
        encoder = encoder.to(device)
    elif 'clip' in model_args.model_name_or_path:
        encoder = CLIPModel.from_pretrained(model_args.model_name_or_path, device_map=device_map,
                                            torch_dtype=torch_type)
        processor = CLIPProcessor.from_pretrained(model_args.model_name_or_path)
    elif 'blip' in model_args.model_name_or_path:
        encoder = blip_itm(pretrained=model_args.model_name_or_path + '/model_large.pth', vit='large')
        encoder = encoder.to(device)
        processor = None
    elif 'gme' in model_args.model_name_or_path:
        encoder = SentenceTransformer(model_args.model_name_or_path, trust_remote_code=True, device=str(device))
        patch_gme_remote_code(encoder)
        processor = None
    elif 'Qwen2-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2-VL-2B-Instruct' in model_args.model_name_or_path:
        encoder = Qwen2VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                  device_map=device_map,
                                                                  torch_dtype=torch_type)
        processor = Qwen2VLProcessor.from_pretrained(model_args.model_name_or_path)
        conversation = [
            {

                "role": "user",
                "content": [
                    {"type": "text", "text": "\nSummary above image in one word: "},
                    {"type": "image", "image": '{}'},
                ],
            },
        ]
        prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        if dist.get_rank() == 0:
            print()
            print(prompt)
            print()
        input_id = processor(text=prompt,
                             return_tensors="pt",
                             padding=True).input_ids
        if dist.get_rank() == 0:
            print(input_id)
    elif 'LamRA' in model_args.model_name_or_path:
        if 'Qwen' in model_args.model_name_or_path:
            encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                         device_map=device_map,
                                                                         torch_dtype=torch_type)
            processor = Qwen2_5_VLProcessor.from_pretrained(model_args.model_name_or_path)
        else:
            encoder = Qwen2VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                      device_map=device_map,
                                                                      torch_dtype=torch_type)
            processor = Qwen2VLProcessor.from_pretrained(model_args.model_name_or_path)
    elif 'VLM2Vec' in model_args.model_name_or_path:
        if 'V2' in model_args.model_name_or_path:
            encoder = Qwen2VLForConditionalGeneration.from_pretrained('./checkpoints/Qwen-Qwen2-VL-2B-Instruct',
                                                                      device_map=device_map,
                                                                      torch_dtype=torch_type)
            processor = Qwen2VLProcessor.from_pretrained('./checkpoints/Qwen-Qwen2-VL-2B-Instruct')
        else:
            encoder = Qwen2VLForConditionalGeneration.from_pretrained('./checkpoints/Qwen-Qwen2-VL-7B-Instruct',
                                                                      device_map=device_map,
                                                                      torch_dtype=torch_type)
            processor = Qwen2VLProcessor.from_pretrained('./checkpoints/Qwen-Qwen2-VL-7B-Instruct')
        encoder = PeftModel.from_pretrained(
            encoder,
            model_args.model_name_or_path,
            torch_dtype=torch_type
        )
    else:
        encoder = CLIPModel.from_pretrained(model_args.model_name_or_path, device_map=device_map,
                                            torch_dtype=torch_type)
        processor = CLIPProcessor.from_pretrained(model_args.model_name_or_path)

    if training_args.task_type == 'tbpr':
        dataset = TextPersonRetrievalDataset(data_args.dataset_name, processor, data_args.dataset_split, 'single')
    elif training_args.task_type == 't2it':
        dataset = Text2ImagetextRetrievalDataset(data_args.dataset_name, processor, data_args.dataset_split, 'corpus')
    elif training_args.task_type == 'it2t':
        dataset = Imagetext2TextRetrievalDataset(data_args.dataset_name, processor, data_args.dataset_split, 'corpus')
    else:
        if training_args.encode_type == 'text':
            dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, data_args.dataset_split, 'full')
        else:
            dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, data_args.dataset_split, 'single')
    sampler = Data.DistributedSampler(dataset, num_replicas=dist.get_world_size(), shuffle=True, rank=dist.get_rank())
    test_dataloader = Data.DataLoader(dataset=dataset, sampler=sampler, pin_memory=True,
                                      batch_size=data_args.per_device_batch_size, shuffle=False)
    print(dataset.__len__())

    encoder = encoder.eval()

    encoded = []
    lookup_indices = []

    with torch.no_grad():
        sampler.set_epoch(0)

        if training_args.task_type == 'tbpr':
            for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                         total=len(test_dataloader)):
                if 'eva' in model_args.model_name_or_path:
                    image = [processor(Image.open(path)).unsqueeze(0).to(device) for path in imgs_path]
                    image = torch.cat(image)
                    reps = encoder.encode_image(image)

                elif 'clip' in model_args.model_name_or_path:
                    raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                    img_inputs = processor(images=raw_images, return_tensors="pt", padding=True)
                    imgs = img_inputs.to(device)
                    reps = encoder.get_image_features(imgs['pixel_values'])
                elif 'gme' in model_args.model_name_or_path:
                    raw_images = [qwen_safe_image(Image.open(path).convert('RGB')) for path in imgs_path]
                    reps = encoder.encode([dict(text="", image=image) for image in raw_images], convert_to_tensor=True)
                elif 'LamRA' in model_args.model_name_or_path:
                    raw_images = [qwen_safe_image(Image.open(path).convert('RGB')) for path in imgs_path]
                    if 'Qwen' in model_args.model_name_or_path:
                        img_inputs = processor(images=raw_images, text=[lamra_2_5_tbpr_prompt] * len(imgs_path),
                                               return_tensors="pt",
                                               padding=True)
                    else:
                        img_inputs = processor(images=raw_images, text=[lamra_2_tbpr_prompt] * len(imgs_path),
                                               return_tensors="pt",
                                               padding=True)
                    imgs = img_inputs.to(device)
                    output = encoder(**imgs, output_hidden_states=True, return_dict=True, use_cache=True)
                    reps = output.hidden_states[-1][:, -1, :]
                elif 'VLM2Vec' in model_args.model_name_or_path:
                    raw_images = [qwen_safe_image(Image.open(path).convert('RGB')) for path in imgs_path]
                    img_inputs = processor(images=raw_images, text=[vlm2vec_tbpr_prompt] * len(imgs_path),
                                           return_tensors="pt",
                                           padding=True)
                    imgs = img_inputs.to(device)
                    output = encoder(**imgs, output_hidden_states=True, return_dict=True)
                    reps = output.hidden_states[-1][:, -1, :]
                else:
                    raw_images = [blip_load_image(path, 384, device).to(device) for path in imgs_path]
                    raw_images = torch.cat(raw_images)
                    image_feat = encoder.visual_encoder(raw_images)
                    reps = encoder.vision_proj(image_feat[:, 0, :])

                # print(logits.shape)
                reps = F.normalize(reps, dim=-1)

                lookup_indices.extend(img_ids)

                encoded.append(reps.cpu().detach().float().numpy())
        elif training_args.task_type == 't2it':
            for batch_idx, (corpus_texts, corpus_images, corpus_ids) in tqdm(enumerate(test_dataloader),
                                                                             total=len(test_dataloader)):
                if 'gme' in model_args.model_name_or_path:
                    raw_images = [qwen_safe_image(Image.open(BytesIO(corpus_image)).convert("RGB")) for corpus_image in corpus_images]
                    reps = encoder.encode(
                        [dict(text=corpus_text, image=image) for corpus_text, image in zip(corpus_texts, raw_images)],
                        convert_to_tensor=True
                    )
                elif 'clip' in model_args.model_name_or_path:
                    raw_images = [Image.open(BytesIO(corpus_image)).convert("RGB") for corpus_image in corpus_images]
                    img_inputs = processor(images=raw_images, return_tensors="pt", padding=True)
                    imgs = img_inputs.to(device)
                    img_reps = encoder.get_image_features(imgs['pixel_values'])
                    text_inputs = processor(text=corpus_texts, return_tensors="pt", padding=True)
                    if text_inputs['input_ids'].shape[1] > 77:
                        text_inputs['input_ids'] = text_inputs['input_ids'][:, :77]
                        text_inputs['attention_mask'] = text_inputs['attention_mask'][:, :77]
                    text_reps = encoder.get_text_features(text_inputs['input_ids'].cuda(),
                                                     text_inputs['attention_mask'].cuda())
                    img_reps = F.normalize(img_reps, dim=-1)
                    text_reps = F.normalize(text_reps, dim=-1)
                    reps = img_reps + text_reps
                elif 'LamRA' in model_args.model_name_or_path:
                    raw_images = [qwen_safe_image(Image.open(BytesIO(corpus_image)).convert("RGB")) for corpus_image in corpus_images]
                    if 'Qwen' in model_args.model_name_or_path:
                        img_inputs = processor(images=raw_images,
                                               text=[lamra_2_5_t2it_corpus_prompt.replace('<sent>', corpus_text) for
                                                     corpus_text in corpus_texts],
                                               return_tensors="pt",
                                               padding=True)
                    else:
                        img_inputs = processor(images=raw_images,
                                               text=[lamra_2_t2it_corpus_prompt.replace('<sent>', corpus_text) for
                                                     corpus_text in corpus_texts],
                                               return_tensors="pt",
                                               padding=True)
                    imgs = img_inputs.to(device)
                    output = encoder(**imgs, output_hidden_states=True, return_dict=True, use_cache=True)
                    reps = output.hidden_states[-1][:, -1, :]
                else:
                    raw_images = [
                        blip_preprocess_image(Image.open(BytesIO(corpus_image)).convert("RGB"), 384, device)
                        for corpus_image in corpus_images
                    ]
                    raw_images = torch.cat(raw_images)
                    image_feat = encoder.visual_encoder(raw_images)
                    img_reps = encoder.vision_proj(image_feat[:, 0, :])
                    text_input = encoder.tokenizer(corpus_texts, padding='max_length', truncation=True, max_length=35,
                                                   return_tensors="pt").to(device)
                    text_output = encoder.text_encoder(text_input.input_ids,
                                                       attention_mask=text_input.attention_mask, mode='text')
                    text_reps = encoder.text_proj(text_output.last_hidden_state[:, 0, :])
                    img_reps = F.normalize(img_reps, dim=-1)
                    text_reps = F.normalize(text_reps, dim=-1)
                    reps = img_reps + text_reps
                # print(logits.shape)
                reps = F.normalize(reps, dim=-1)

                lookup_indices.extend(corpus_ids)

                encoded.append(reps.cpu().detach().float().numpy())
        elif training_args.task_type == 'it2t':
            for batch_idx, (corpus_texts, corpus_ids) in tqdm(enumerate(test_dataloader),
                                                              total=len(test_dataloader)):
                if 'gme' in model_args.model_name_or_path:
                    reps = encoder.encode([dict(text=t) for t in corpus_texts], convert_to_tensor=True)
                elif 'clip' in model_args.model_name_or_path:
                    text_inputs = processor(text=corpus_texts, return_tensors="pt", padding=True)
                    if text_inputs['input_ids'].shape[1] > 77:
                        text_inputs['input_ids'] = text_inputs['input_ids'][:, :77]
                        text_inputs['attention_mask'] = text_inputs['attention_mask'][:, :77]
                    reps = encoder.get_text_features(text_inputs['input_ids'].cuda(),
                                                     text_inputs['attention_mask'].cuda())
                elif 'LamRA' in model_args.model_name_or_path:
                    if 'Qwen' in model_args.model_name_or_path:
                        text_inputs = processor(
                            text=[lamra_2_5_it2t_corpus_prompt.replace('<sent>', text) for text in corpus_texts],
                            return_tensors="pt",
                            padding=True).to(device)
                    else:
                        text_inputs = processor(
                            text=[lamra_2_it2t_corpus_prompt.replace('<sent>', text) for text in corpus_texts],
                            return_tensors="pt",
                            padding=True).to(device)
                    output = encoder(**text_inputs, output_hidden_states=True, return_dict=True)
                    reps = output.hidden_states[-1][:, -1, :]
                else:
                    text_input = encoder.tokenizer(corpus_texts, padding='max_length', truncation=True, max_length=35,
                                                   return_tensors="pt").to(device)
                    text_output = encoder.text_encoder(text_input.input_ids,
                                                       attention_mask=text_input.attention_mask, mode='text')
                    reps = encoder.text_proj(text_output.last_hidden_state[:, 0, :])
                # print(logits.shape)
                reps = F.normalize(reps, dim=-1)

                lookup_indices.extend(corpus_ids)

                encoded.append(reps.cpu().detach().float().numpy())
        else:
            for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                         total=len(test_dataloader)):
                with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                    if len(texts) != data_args.per_device_batch_size:
                        print(len(texts))
                        print(dist.get_rank())
                    if training_args.encode_type == 'text':
                        if 'eva' in model_args.model_name_or_path:
                            tokenizer = get_tokenizer('EVA02-CLIP-bigE-14-plus')
                            text = tokenizer(texts).to(device)
                            reps = encoder.encode_text(text)
                        elif 'clip' in model_args.model_name_or_path:
                            text_inputs = processor(text=texts, return_tensors="pt", padding=True)
                            if text_inputs['input_ids'].shape[1] > 77:
                                text_inputs['input_ids'] = text_inputs['input_ids'][:, :77]
                                text_inputs['attention_mask'] = text_inputs['attention_mask'][:, :77]
                            reps = encoder.get_text_features(text_inputs['input_ids'].cuda(),
                                                             text_inputs['attention_mask'].cuda())
                        elif 'gme' in model_args.model_name_or_path:
                            reps = encoder.encode([dict(text=t) for t in texts], convert_to_tensor=True)
                        elif 'LamRA' in model_args.model_name_or_path:
                            if 'Qwen' in model_args.model_name_or_path:
                                text_inputs = processor(
                                    text=[lamra_2_5_text_prompt.replace('<sent>', text) for text in texts],
                                    return_tensors="pt",
                                    padding=True).to(device)
                            else:
                                text_inputs = processor(
                                    text=[lamra_2_text_prompt.replace('<sent>', text) for text in texts],
                                    return_tensors="pt",
                                    padding=True).to(device)
                            output = encoder(**text_inputs, output_hidden_states=True, return_dict=True)
                            reps = output.hidden_states[-1][:, -1, :]
                        elif 'VLM2Vec' in model_args.model_name_or_path:
                            text_inputs = processor(
                                text=[vlm2vec_text_prompt.replace('<sent>', text) for text in texts],
                                return_tensors="pt",
                                padding=True).to(device)
                            output = encoder(**text_inputs, output_hidden_states=True, return_dict=True)
                            reps = output.hidden_states[-1][:, -1, :]
                        else:
                            text_input = encoder.tokenizer(texts, padding='max_length', truncation=True, max_length=35,
                                                           return_tensors="pt").to(device)
                            text_output = encoder.text_encoder(text_input.input_ids,
                                                               attention_mask=text_input.attention_mask, mode='text')
                            reps = encoder.text_proj(text_output.last_hidden_state[:, 0, :])
                    else:
                        if 'eva' in model_args.model_name_or_path:
                            image = [processor(Image.open(path)).unsqueeze(0).to(device) for path in imgs_path]
                            image = torch.cat(image)
                            reps = encoder.encode_image(image)
                        elif 'clip' in model_args.model_name_or_path:
                            raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                            img_inputs = processor(images=raw_images, return_tensors="pt", padding=True)
                            imgs = img_inputs.to(device)
                            reps = encoder.get_image_features(imgs['pixel_values'])
                        elif 'gme' in model_args.model_name_or_path:
                            raw_images = [qwen_safe_image(Image.open(path).convert('RGB')) for path in imgs_path]
                            reps = encoder.encode([dict(text="", image=image) for image in raw_images],
                                                  convert_to_tensor=True)
                        elif 'LamRA' in model_args.model_name_or_path:
                            raw_images = [qwen_safe_image(Image.open(path).convert('RGB')) for path in imgs_path]
                            if 'Qwen' in model_args.model_name_or_path:
                                img_inputs = processor(images=raw_images, text=[lamra_2_5_img_prompt] * len(imgs_path),
                                                       return_tensors="pt",
                                                       padding=True)
                            else:
                                img_inputs = processor(images=raw_images, text=[lamra_2_img_prompt] * len(imgs_path),
                                                       return_tensors="pt",
                                                       padding=True)
                            imgs = img_inputs.to(device)
                            output = encoder(**imgs, output_hidden_states=True, return_dict=True, use_cache=True)
                            reps = output.hidden_states[-1][:, -1, :]
                        elif 'VLM2Vec' in model_args.model_name_or_path:
                            raw_images = [qwen_safe_image(Image.open(path).convert('RGB')) for path in imgs_path]
                            img_inputs = processor(images=raw_images, text=[vlm2vec_img_prompt] * len(imgs_path),
                                                   return_tensors="pt",
                                                   padding=True)
                            imgs = img_inputs.to(device)
                            output = encoder(**imgs, output_hidden_states=True, return_dict=True)
                            reps = output.hidden_states[-1][:, -1, :]
                        else:
                            raw_images = [blip_load_image(path, 384, device).to(device) for path in imgs_path]
                            raw_images = torch.cat(raw_images)
                            image_feat = encoder.visual_encoder(raw_images)
                            reps = encoder.vision_proj(image_feat[:, 0, :])

                    # print(logits.shape)
                    reps = F.normalize(reps, dim=-1)
                    if training_args.encode_type == 'text':
                        lookup_indices.extend(text_ids)
                    else:
                        lookup_indices.extend(img_ids)

                    encoded.append(reps.cpu().detach().float().numpy())

    encoded = np.concatenate(encoded)

    print(f'rank:{dist.get_rank()}, encoded length:{len(encoded)}')

    if data_args.is_filtered:
        filtered = "filter"
    else:
        filtered = "no_filter"

    if data_args.sparse_manual:
        manual = 'manual'
    else:
        manual = "no_manual"

    if model_args.use_output_embedding_cluster:
        cluster = f'cluster_{model_args.cluster_sum}'
    else:
        cluster = 'no_cluster'

    if data_args.sparse_value_mean:
        use_sparse_value_mean = 'mean'
    else:
        use_sparse_value_mean = 'no_mean'

    if training_args.task_type == 'tbpr':
        os.makedirs(
            f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',
            exist_ok=True)

        with open(os.path.join(
                f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',
                f'query.pkl') if data_args.encode_is_query else os.path.join(
            f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',
            f'corpus_{dist.get_rank()}.pkl'), 'wb') as f:
            pickle.dump((encoded, lookup_indices), f)


    elif training_args.task_type == 't2it':
        os.makedirs(

            f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',

            exist_ok=True)

        with open(os.path.join(

                f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',

                f'query.pkl') if data_args.encode_is_query else os.path.join(

            f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',

            f'corpus_{dist.get_rank()}.pkl'), 'wb') as f:

            pickle.dump((encoded, lookup_indices), f)

    elif training_args.task_type == 'it2t':
        os.makedirs(

            f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',

            exist_ok=True)

        with open(os.path.join(

                f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',

                f'query.pkl') if data_args.encode_is_query else os.path.join(

            f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',

            f'corpus_{dist.get_rank()}.pkl'), 'wb') as f:

            pickle.dump((encoded, lookup_indices), f)

    else:
        os.makedirs(
            f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',
            exist_ok=True)

        with open(os.path.join(
                f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',
                f'query.pkl') if data_args.encode_is_query else os.path.join(
            f'{data_args.dense_output_dir}/{model_args.model_name_or_path[model_begin_indice:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',
            f'corpus_{dist.get_rank()}.pkl'), 'wb') as f:
            pickle.dump((encoded, lookup_indices), f)

    # 训练结束后添加同步屏障
    dist.barrier()

    # 确保所有进程同步退出
    if dist.get_rank() == 0:
        # 主进程最后退出
        torch.distributed.destroy_process_group()
    else:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
