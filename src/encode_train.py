import json
import os
import pickle
import itertools

import numpy as np
from PIL import Image

from tqdm import tqdm
from transformers import (
    HfArgumentParser,
)
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoModel, \
    AutoProcessor, \
    AutoTokenizer, PhiForCausalLM, Phi3ForCausalLM, AutoModelForCausalLM
from arguments import PromptRepsLLMDataArguments, ModelArguments
import torch.distributed as dist
from arguments import TrainingArguments
from dataset import CrossModalRetrievalDataset
import torch
import torch.utils.data as Data
import torch.nn.functional as F
from peft import PeftModel, PeftConfig

from template import text_prompt, img_prompt, text_prompt_no_one_word, img_prompt_no_one_word, \
    img_prompt_no_special_llava_v1_5, text_prompt_no_special_llava_v1_5, text_prompt_qwen_v2_5, img_prompt_qwen_v2_5, \
    img_prompt_intern_vl_v2_5, text_prompt_intern_vl_v2_5
from model import MLLMRetrievalModel
from utils import split_model, load_image
from encode import get_img_valid_tokens_values, get_text_valid_tokens_values, get_filtered_ids


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
    if 'llava-hf-llava-1.5-7b-hf' in model_args.model_name_or_path:
        base_model = LlavaForConditionalGeneration.from_pretrained(model_args.model_name_or_path, device_map=device_map,
                                                                   torch_dtype=torch_type)
        processor = LlavaProcessor.from_pretrained(model_args.model_name_or_path)

    elif 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in model_args.model_name_or_path:
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                        device_map=device_map,
                                                                        torch_dtype=torch_type)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_args.model_name_or_path)
    elif 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
        # device_map = split_model('InternVL2_5-8B')
        base_model = AutoModel.from_pretrained(model_args.model_name_or_path,
                                               device_map=device_map,
                                               torch_dtype=torch_type,
                                               trust_remote_code=True,
                                               low_cpu_mem_usage=True,
                                               )
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path,
                                                  trust_remote_code=True, )
    else:
        base_model = LlavaNextForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                       device_map=device_map,
                                                                       torch_dtype=torch_type)
        processor = LlavaNextProcessor.from_pretrained(model_args.model_name_or_path)
        if 'royokong-e5-v' in model_args.model_name_or_path:
            setattr(processor, "patch_size", 14)  # hack for pass

    encoder = PeftModel.from_pretrained(
        base_model,  # 原始模型
        model_args.lora_model_path,  # LoRA 适配器目录
    )

    encoder = encoder.merge_and_unload()

    if training_args.encode_type == 'text':
        dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'full')
    else:
        dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'test', 'single')
    sampler = Data.DistributedSampler(dataset, num_replicas=dist.get_world_size(), shuffle=True, rank=dist.get_rank())
    test_dataloader = Data.DataLoader(dataset=dataset, sampler=sampler, pin_memory=True,
                                      batch_size=data_args.per_device_batch_size, shuffle=False)

    model = MLLMRetrievalModel(encoder)
    model = model.eval()
    print(model.is_ddp)

    encoded = []
    jsonl_data = []
    lookup_indices = []

    # 加载词表并获取过滤后的单词id，但目前尚不清楚filtered_ids是做什么的
    if 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
        vocab_dict = processor.get_vocab()
        filtered_ids = get_filtered_ids(processor)
    else:
        vocab_dict = processor.tokenizer.get_vocab()
        filtered_ids = get_filtered_ids(processor.tokenizer)
    vocab_dict = {v: k for k, v in vocab_dict.items()}

    with torch.no_grad():
        sampler.set_epoch(0)
        if 'llava-hf-llava-1.5-7b-hf' in model_args.model_name_or_path or 'llava-hf-llava-v1.6-vicuna-7b-hf' in model_args.model_name_or_path:
            prompt = img_prompt_no_special_llava_v1_5
        elif 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in model_args.model_name_or_path:
            prompt = img_prompt_qwen_v2_5
        elif 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
            prompt = img_prompt_intern_vl_v2_5
            if dist.get_rank() == 0:
                print(prompt)
        else:
            prompt = img_prompt
        for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                     total=len(test_dataloader)):
            if len(texts) != data_args.per_device_batch_size:
                print(len(texts))
                print(dist.get_rank())
            if training_args.encode_type == 'text':
                logits, reps = model.encode_data(texts, 'text', processor, device, model_args, data_args)
            else:
                # Preparation for inference
                if 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
                    prompt = processor.apply_chat_template(
                        img_prompt_intern_vl_v2_5, tokenize=False, add_generation_prompt=True
                    )
                    imgs = [load_image(path, max_num=12).to(torch_type).cuda() for path in imgs_path]
                    logits, reps = model.encode_data(imgs, 'image', processor, device, model_args, data_args)
                else:
                    if 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in model_args.model_name_or_path:
                        prompt = processor.apply_chat_template(
                            img_prompt_qwen_v2_5, tokenize=False, add_generation_prompt=True
                        )
                    raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                    img_inputs = processor(images=raw_images, text=[prompt] * len(imgs_path), return_tensors="pt",
                                           padding=True)
                    imgs = img_inputs.to(device)
                    logits, reps = model.encode_data(imgs, 'image', processor, device, model_args, data_args)

            # print(logits.shape)
            reps = F.normalize(reps, dim=-1)
            if dist.is_initialized():
                # reps_list = [[None] for _ in range(dist.get_world_size())]
                # logits_list = [[None] for _ in range(dist.get_world_size())]

                reps_list = [torch.zeros_like(reps) for _ in range(dist.get_world_size())]
                logits_list = [torch.zeros_like(logits) for _ in range(dist.get_world_size())]
                texts_list = [[None] for _ in range(dist.get_world_size())]
                text_ids_list = [[None] for _ in range(dist.get_world_size())]
                image_ids_list = [[None] for _ in range(dist.get_world_size())]

                '''
                texts_list = [[None] for _ in range(dist.get_world_size())]
                text_ids_list = [[None] for _ in range(dist.get_world_size())]
                image_ids_list = [[None] for _ in range(dist.get_world_size())]
                '''

                dist.all_gather(tensor_list=reps_list, tensor=reps.contiguous())
                dist.all_gather(tensor_list=logits_list, tensor=logits.contiguous())
                dist.all_gather_object(object_list=text_ids_list, obj=text_ids)
                dist.all_gather_object(object_list=image_ids_list, obj=img_ids)
                dist.all_gather_object(object_list=texts_list, obj=texts)

                batch_reps = torch.cat(reps_list)
                batch_logits = torch.cat(logits_list)
                batch_text_ids = list(itertools.chain(*text_ids_list))
                batch_image_ids = list(itertools.chain(*image_ids_list))
                batch_texts = list(itertools.chain(*texts_list))

                if dist.get_rank() == 0:
                    if training_args.encode_type == 'text':
                        lookup_indices.extend(batch_text_ids)
                    else:
                        lookup_indices.extend(batch_image_ids)
                    encoded.append(batch_reps.cpu().detach().float().numpy())

                    ids = batch_text_ids if training_args.encode_type == 'text' else batch_image_ids
                    if training_args.encode_type == 'text':
                        for id, logits, text in zip(ids, batch_logits, batch_texts):
                            vector = dict()
                            if 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
                                tokens, values = get_text_valid_tokens_values(text, processor, logits,
                                                                              vocab_dict,
                                                                              data_args,
                                                                              filtered_ids)
                            else:
                                tokens, values = get_text_valid_tokens_values(text, processor.tokenizer, logits,
                                                                              vocab_dict,
                                                                              data_args,
                                                                              filtered_ids)
                            for token, v in zip(tokens, values):
                                vector[token] = int(v)
                            jsonl_data.append(
                                dict(
                                    id=id,
                                    content="",
                                    vector=vector,
                                )
                            )
                    else:
                        for id, logits, text in zip(ids, batch_logits, batch_texts):
                            vector = dict()
                            if 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
                                tokens, values = get_img_valid_tokens_values(processor, logits, vocab_dict,
                                                                             data_args, filtered_ids)
                            else:
                                tokens, values = get_img_valid_tokens_values(processor.tokenizer, logits, vocab_dict,
                                                                             data_args, filtered_ids)
                            for token, v in zip(tokens, values):
                                vector[token] = int(v)
                            jsonl_data.append(
                                dict(
                                    id=id,
                                    content="",
                                    vector=vector,
                                )
                            )

        if dist.get_rank() == 0:
            encoded = np.concatenate(encoded)

            print(len(encoded))
            print(len(lookup_indices))
            print(len(jsonl_data))

            if data_args.is_filtered:
                filtered = "filter"
            else:
                filtered = "no_filter"

            if data_args.sparse_manual:
                manual = 'manual'
            else:
                manual = "no_manual"

            os.makedirs(
                f'{data_args.dense_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_train',
                exist_ok=True)
            os.makedirs(
                f'{data_args.sparse_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_train',
                exist_ok=True)

            with open(os.path.join(
                    f'{data_args.dense_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_train',
                    f'query.pkl') if data_args.encode_is_query else os.path.join(
                f'{data_args.dense_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_train',
                f'corpus_{data_args.dataset_shard_index}.pkl'), 'wb') as f:
                pickle.dump((encoded, lookup_indices), f)

            with open(os.path.join(
                    f'{data_args.sparse_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_train',
                    f'query.tsv') if data_args.encode_is_query else os.path.join(
                f'{data_args.sparse_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_train',
                f'corpus_{data_args.dataset_shard_index}.jsonl'), 'w') as f:
                for data in jsonl_data:
                    if data_args.encode_is_query:
                        id = data['id']
                        vector = data['vector']
                        query = " ".join([" ".join([str(token)] * freq) for token, freq in vector.items()])
                        if len(query.strip()) == 0:
                            continue
                        f.write(f'{id}\t{query}\n')
                    else:
                        f.write(json.dumps(data) + "\n")


if __name__ == "__main__":
    main()
