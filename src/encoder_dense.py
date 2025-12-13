import os
import pickle
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.data as Data

torch.set_printoptions(threshold=sys.maxsize)
from PIL import Image
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
)
from transformers import CLIPModel, CLIPProcessor, BlipModel, BlipProcessor

from arguments import PromptRepsLLMDataArguments, ModelArguments
from arguments import TrainingArguments
from dataset import CrossModalRetrievalDataset, TextPersonRetrievalDataset


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
    if 'clip' in model_args.model_name_or_path:
        encoder = CLIPModel.from_pretrained(model_args.model_name_or_path, device_map=device_map,
                                                                torch_dtype=torch_type)
        processor = CLIPProcessor.from_pretrained(model_args.model_name_or_path)
    elif 'blip' in model_args.model_name_or_path:
        encoder = BlipModel.from_pretrained(model_args.model_name_or_path, device_map=device_map,
                                            torch_dtype=torch_type)
        processor = BlipProcessor.from_pretrained(model_args.model_name_or_path)
    else:
        encoder = CLIPModel.from_pretrained(model_args.model_name_or_path, device_map=device_map,
                                            torch_dtype=torch_type)
        processor = CLIPProcessor.from_pretrained(model_args.model_name_or_path)


    if training_args.task_type == 'tbpr':
        dataset = TextPersonRetrievalDataset(data_args.dataset_name, processor, data_args.dataset_split, 'single')
    else:
        if training_args.encode_type == 'text':
            dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, data_args.dataset_split, 'full')
        else:
            dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, data_args.dataset_split, 'single')
    sampler = Data.DistributedSampler(dataset, num_replicas=dist.get_world_size(), shuffle=True, rank=dist.get_rank())
    test_dataloader = Data.DataLoader(dataset=dataset, sampler=sampler, pin_memory=True,
                                      batch_size=data_args.per_device_batch_size, shuffle=False)

    encoder = encoder.eval()

    encoded = []
    lookup_indices = []

    with torch.no_grad():
        sampler.set_epoch(0)

        if training_args.task_type == 'tbpr':
            for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                         total=len(test_dataloader)):
                raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                img_inputs = processor(images=raw_images, return_tensors="pt", padding=True)
                imgs = img_inputs.to(device)
                if 'clip' in model_args.model_name_or_path:
                    reps = encoder.get_image_features(imgs)
                else:
                    reps = encoder.get_image_features(imgs)

                # print(logits.shape)
                reps = F.normalize(reps, dim=-1)

                lookup_indices.extend(img_ids)

                encoded.append(reps.cpu().detach().float().numpy())
        else:
            for batch_idx, (texts, imgs_path, text_ids, img_ids) in tqdm(enumerate(test_dataloader),
                                                                         total=len(test_dataloader)):
                with torch.cuda.amp.autocast() if training_args.fp16 else nullcontext():
                    if len(texts) != data_args.per_device_batch_size:
                        print(len(texts))
                        print(dist.get_rank())
                    if training_args.encode_type == 'text':
                        text_inputs = processor(text=texts, return_tensors="pt", padding=True)
                        reps = encoder.get_text_features(text_inputs)
                    else:
                        raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
                        img_inputs = processor(images=raw_images, return_tensors="pt", padding=True)
                        imgs = img_inputs.to(device)
                        reps = encoder.get_image_features(imgs)

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
            f'{data_args.dense_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',
            exist_ok=True)
        os.makedirs(
            f'{data_args.sparse_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',
            exist_ok=True)

        with open(os.path.join(
                f'{data_args.dense_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',
                f'query.pkl') if data_args.encode_is_query else os.path.join(
            f'{data_args.dense_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.tbpr_type}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',
            f'corpus_{dist.get_rank()}.pkl'), 'wb') as f:
            pickle.dump((encoded, lookup_indices), f)

    else:
        os.makedirs(
            f'{data_args.dense_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',
            exist_ok=True)
        os.makedirs(
            f'{data_args.sparse_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',
            exist_ok=True)

        with open(os.path.join(
                f'{data_args.dense_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',
                f'query.pkl') if data_args.encode_is_query else os.path.join(
            f'{data_args.dense_output_dir}/{model_args.model_name_or_path[14:]}/{data_args.dataset_name}/{training_args.encode_type}/{filtered}/{model_args.calculate_type}/{data_args.prompt_type}/{data_args.dataset_split}/{data_args.num_expended_tokens}_{manual}_{data_args.sparse_length}_{data_args.sparse_value_type}_{cluster}_{data_args.reps_loc}_{model_args.eol_type}_{data_args.sparse_lower_or_upper}_{use_sparse_value_mean}',
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