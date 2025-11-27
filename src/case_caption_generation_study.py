import os

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from transformers import (
    HfArgumentParser,
)
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoModel, \
    AutoProcessor

from arguments import PromptRepsLLMDataArguments, ModelArguments
from arguments import TrainingArguments, PromptRepsLLMSearchArguments
from template import mistral_query_generation_paradigm_prompt, query_generation_paradigm_prompt, \
    mistral_query_generation_paradigm_prompt_1, query_generation_paradigm_prompt_1, \
    detailed_mistral_query_generation_paradigm_prompt, detailed_query_generation_paradigm_prompt, \
    detailed_query_generation_paradigm_prompt_1, detailed_mistral_query_generation_paradigm_prompt_1, \
    mistral_query_generation_paradigm_prompt_5, mistral_query_generation_paradigm_prompt_4, \
    query_generation_paradigm_prompt_4, query_generation_paradigm_prompt_5, query_generation_paradigm_prompt_2, \
    query_generation_paradigm_prompt_3, mistral_query_generation_paradigm_prompt_2, \
    mistral_query_generation_paradigm_prompt_3, query_generation_paradigm_prompt_6, query_generation_paradigm_prompt_7, \
    mistral_query_generation_paradigm_prompt_6, mistral_query_generation_paradigm_prompt_7, llama3_template, \
    llava_mistral_template
from scipy.stats import spearmanr


def main():
    parser = HfArgumentParser(
        (ModelArguments, PromptRepsLLMDataArguments, PromptRepsLLMSearchArguments, TrainingArguments))

    model_args, data_args, search_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: PromptRepsLLMDataArguments
    training_args: TrainingArguments
    search_args: PromptRepsLLMSearchArguments

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
        encoder = LlavaForConditionalGeneration.from_pretrained(model_args.model_name_or_path, device_map=device_map,
                                                                torch_dtype=torch_type)
        processor = LlavaProcessor.from_pretrained(model_args.model_name_or_path)

    elif 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in model_args.model_name_or_path:
        encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                     device_map=device_map,
                                                                     torch_dtype=torch_type)
        processor = Qwen2_5_VLProcessor.from_pretrained(model_args.model_name_or_path)
    elif 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
        # device_map = split_model('InternVL2_5-8B')
        encoder = AutoModel.from_pretrained(model_args.model_name_or_path,
                                            device_map=device_map,
                                            torch_dtype=torch_type,
                                            trust_remote_code=True,
                                            low_cpu_mem_usage=True,
                                            )
        processor = AutoProcessor.from_pretrained(model_args.model_name_or_path,
                                                  trust_remote_code=True, )
    else:
        encoder = LlavaNextForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                    device_map=device_map,
                                                                    torch_dtype=torch_type)
        processor = LlavaNextProcessor.from_pretrained(model_args.model_name_or_path)
        if 'royokong-e5-v' in model_args.model_name_or_path:
            setattr(processor, "patch_size", 14)  # hack for pass

    if data_args.reps_loc == 'after_pad':
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.padding = True

    encoder = encoder.eval()

    image_path = './data/flickr/flickr30k-images/499340051.jpg'
    text1 = 'An aerial view of a man sitting at a desktop computer.'
    text2 = 'A man in a black shirt sitting at a table with an open Apple laptop in front of him.'
    print(image_path)
    print(text1)
    print(text2)

    with torch.no_grad():
        if 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
            if search_args.rerank_template == 'caption_generation':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt
            elif search_args.rerank_template == 'what_caption_generation':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_1
            elif search_args.rerank_template == 'detailed_caption_generation':
                rerank_prompt_template = detailed_mistral_query_generation_paradigm_prompt
            elif search_args.rerank_template == 'detailed_caption_generation_1':
                rerank_prompt_template = detailed_mistral_query_generation_paradigm_prompt_1
            elif search_args.rerank_template == 'caption_generation_4':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_4
            elif search_args.rerank_template == 'caption_generation_5':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_5
            elif search_args.rerank_template == 'caption_generation_2':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_2
            elif search_args.rerank_template == 'caption_generation_3':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_3
            elif search_args.rerank_template == 'caption_generation_6':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_6
            elif search_args.rerank_template == 'caption_generation_7':
                rerank_prompt_template = mistral_query_generation_paradigm_prompt_7
            else:
                rerank_prompt_template = mistral_query_generation_paradigm_prompt
        else:
            if search_args.rerank_template == 'caption_generation':
                rerank_prompt_template = query_generation_paradigm_prompt
            elif search_args.rerank_template == 'what_caption_generation':
                rerank_prompt_template = query_generation_paradigm_prompt_1
            elif search_args.rerank_template == 'detailed_caption_generation':
                rerank_prompt_template = detailed_query_generation_paradigm_prompt
            elif search_args.rerank_template == 'detailed_caption_generation_1':
                rerank_prompt_template = detailed_query_generation_paradigm_prompt_1
            elif search_args.rerank_template == 'caption_generation_4':
                rerank_prompt_template = query_generation_paradigm_prompt_4
            elif search_args.rerank_template == 'caption_generation_5':
                rerank_prompt_template = query_generation_paradigm_prompt_5
            elif search_args.rerank_template == 'caption_generation_2':
                rerank_prompt_template = query_generation_paradigm_prompt_2
            elif search_args.rerank_template == 'caption_generation_3':
                rerank_prompt_template = query_generation_paradigm_prompt_3
            elif search_args.rerank_template == 'caption_generation_6':
                rerank_prompt_template = query_generation_paradigm_prompt_6
            elif search_args.rerank_template == 'caption_generation_7':
                rerank_prompt_template = query_generation_paradigm_prompt_7
            else:
                rerank_prompt_template = query_generation_paradigm_prompt

        text_input1 = rerank_prompt_template + text1
        text_input2 = rerank_prompt_template + text2
        print(text_input1)
        print(text_input2)

        raw_image = Image.open(image_path).convert('RGB')

        inputs_1 = processor(images=raw_image, text=text_input1, return_tensors="pt").to(
            encoder.device)

        inputs_2 = processor(images=raw_image, text=text_input2, return_tensors="pt").to(
            encoder.device)

        max_inputs_sum_1 = inputs_1['input_ids'].shape[1]
        labels_1 = processor(text=text1, return_tensors="pt")['input_ids'].squeeze().tolist()
        # 去掉label的第一个起始符
        print(labels_1[1:])
        labels_1_clone = labels_1[1:].copy()
        length_1 = len(labels_1[1:])
        labels_1 = [-100] * (max_inputs_sum_1 - len(labels_1[1:])) + labels_1[1:]

        max_inputs_sum_2 = inputs_2['input_ids'].shape[1]
        labels_2 = processor(text=text2, return_tensors="pt")['input_ids'].squeeze().tolist()
        # 去掉label的第一个起始符
        print(labels_2[1:])
        labels_2_clone = labels_2[1:].copy()
        length_2 = len(labels_2[1:])
        labels_2 = [-100] * (max_inputs_sum_2 - len(labels_2[1:])) + labels_2[1:]

        labels_view_1 = torch.tensor(labels_1).to(encoder.device)
        labels_view_2 = torch.tensor(labels_2).to(encoder.device)

        output_1 = encoder(**inputs_1, output_hidden_states=True, return_dict=True)
        logits_1 = output_1.logits
        shift_logits_1 = logits_1[..., :-1, :].contiguous()
        shift_labels_1 = labels_view_1[..., 1:].contiguous()

        output_2 = encoder(**inputs_2, output_hidden_states=True, return_dict=True)
        logits_2 = output_2.logits
        shift_logits_2 = logits_2[..., :-1, :].contiguous()
        shift_labels_2 = labels_view_2[..., 1:].contiguous()

        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        nll_1 = loss_func(shift_logits_1.view(-1, shift_logits_1.size(-1)), shift_labels_1.view(-1))
        nll_1 = nll_1.view(shift_labels_1.size())
        avg_nll_1 = torch.sum(nll_1)
        valid_tokens_1 = (labels_view_1 != -100).sum().float()
        avg_nll_1 /= valid_tokens_1

        nll_2 = loss_func(shift_logits_2.view(-1, shift_logits_2.size(-1)), shift_labels_2.view(-1))
        nll_2 = nll_2.view(shift_labels_2.size())

        avg_nll_2 = torch.sum(nll_2)
        valid_tokens_2 = (labels_view_2 != -100).sum().float()
        avg_nll_2 /= valid_tokens_2

        print(nll_1[-length_1:])
        print(nll_2[-length_2:])

        print(nll_1[-length_1:].shape)
        print(nll_2[-length_2:].shape)

        import matplotlib.pyplot as plt

        categories_1 = [processor.decode(labels_1_clone[i]) + str(i) for i in range(len(labels_1_clone))]
        categories_2 = [processor.decode(labels_2_clone[i]) + str(i) for i in range(len(labels_2_clone))]
        values_1 = nll_1[-length_1:].tolist()
        values_2 = nll_2[-length_2:].tolist()
        values_1 = [-x for x in values_1]
        values_2 = [-x for x in values_2]
        print(values_1)
        print(values_2)

        array_1 = np.array(values_1)
        array_2 = np.array(values_2)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharey=True)

        # 在第一个子图上绘制条形图
        bars1 = ax1.bar(categories_1, values_1, color='skyblue')
        for indice in range(len(bars1)):
            height = bars1[indice].get_height()  # 获取条形的高度，即y值
            # 在条形顶部中央位置添加文本
            ax1.text(bars1[indice].get_x() + bars1[indice].get_width() / 2,  # x坐标：条形中心
                    height - 0.3,  # y坐标：条形顶部上方一点
                    f'{values_1[indice]}',  # 要显示的文本
                    ha='center',  # 水平对齐：居中
                    va='bottom')  # 垂直对齐：底部与指定坐标对齐
        ax1.axhline(y=-avg_nll_1.item(), color='red', linestyle='--', linewidth=2, label=f'mean: {-avg_nll_1.item()}')
        ax1.set_ylabel('log likelihood')  # 通常只在第一个图上设置y轴标签

        # 在第二个子图上绘制条形图
        bars2 = ax2.bar(categories_2, values_2, color='lightgreen')
        ax2.axhline(y=-avg_nll_2.item(), color='red', linestyle='--', linewidth=2, label=f'mean: {-avg_nll_2.item()}')

        for indice in range(len(bars2)):
            height = bars2[indice].get_height()  # 获取条形的高度，即y值
            # 在条形顶部中央位置添加文本
            ax2.text(bars2[indice].get_x() + bars2[indice].get_width() / 2,  # x坐标：条形中心
                    height - 0.3,  # y坐标：条形顶部上方一点
                    f'{values_2[indice]}',  # 要显示的文本
                    ha='center',  # 水平对齐：居中
                    va='bottom')  # 垂直对齐：底部与指定坐标对齐

        ax1.legend()
        ax2.legend()
        plt.savefig(f'25235_{search_args.rerank_template}.png')

        if 'llava-hf-llava-v1.6-mistral-7b-hf' in model_args.model_name_or_path:
            if search_args.rerank_template == 'caption_generation':
                pure_text_prompt = llava_mistral_template.format('Please write a caption based on this image.')
            else:
                pure_text_prompt = llava_mistral_template.format('What is the caption of the above image?')
        else:
            if search_args.rerank_template == 'caption_generation':
                pure_text_prompt = llama3_template.format('Please write a caption based on this image.')
            else:
                pure_text_prompt = llama3_template.format('What is the caption of the above image?')

        text_input1 = pure_text_prompt + text1
        text_input2 = pure_text_prompt + text2

        print(text_input1)
        print(text_input2)

        inputs_1 = processor(text=text_input1, return_tensors="pt").to(
            encoder.device)

        inputs_2 = processor(text=text_input2, return_tensors="pt").to(
            encoder.device)

        max_inputs_sum_1 = inputs_1['input_ids'].shape[1]
        labels_1 = processor(text=text1, return_tensors="pt")['input_ids'].squeeze().tolist()
        # 去掉label的第一个起始符
        print(labels_1[1:])
        labels_1_clone = labels_1[1:].copy()
        length_1 = len(labels_1[1:])
        labels_1 = [-100] * (max_inputs_sum_1 - len(labels_1[1:])) + labels_1[1:]

        max_inputs_sum_2 = inputs_2['input_ids'].shape[1]
        labels_2 = processor(text=text2, return_tensors="pt")['input_ids'].squeeze().tolist()
        # 去掉label的第一个起始符
        print(labels_2[1:])
        labels_2_clone = labels_2[1:].copy()
        length_2 = len(labels_2[1:])
        labels_2 = [-100] * (max_inputs_sum_2 - len(labels_2[1:])) + labels_2[1:]

        labels_view_1 = torch.tensor(labels_1).to(encoder.device)
        labels_view_2 = torch.tensor(labels_2).to(encoder.device)

        output_1 = encoder(**inputs_1, output_hidden_states=True, return_dict=True)
        logits_1 = output_1.logits
        shift_logits_1 = logits_1[..., :-1, :].contiguous()
        shift_labels_1 = labels_view_1[..., 1:].contiguous()

        output_2 = encoder(**inputs_2, output_hidden_states=True, return_dict=True)
        logits_2 = output_2.logits
        shift_logits_2 = logits_2[..., :-1, :].contiguous()
        shift_labels_2 = labels_view_2[..., 1:].contiguous()

        loss_func = torch.nn.CrossEntropyLoss(reduction='none')
        nll_1 = loss_func(shift_logits_1.view(-1, shift_logits_1.size(-1)), shift_labels_1.view(-1))
        nll_1 = nll_1.view(shift_labels_1.size())
        avg_nll_1 = torch.sum(nll_1)
        valid_tokens_1 = (labels_view_1 != -100).sum().float()
        avg_nll_1 /= valid_tokens_1

        nll_2 = loss_func(shift_logits_2.view(-1, shift_logits_2.size(-1)), shift_labels_2.view(-1))
        nll_2 = nll_2.view(shift_labels_2.size())

        avg_nll_2 = torch.sum(nll_2)
        valid_tokens_2 = (labels_view_2 != -100).sum().float()
        avg_nll_2 /= valid_tokens_2

        print(nll_1[-length_1:])
        print(nll_2[-length_2:])

        print(nll_1[-length_1:].shape)
        print(nll_2[-length_2:].shape)

        import matplotlib.pyplot as plt

        categories_1 = [processor.decode(labels_1_clone[i]) + str(i) for i in range(len(labels_1_clone))]
        categories_2 = [processor.decode(labels_2_clone[i]) + str(i) for i in range(len(labels_2_clone))]
        values_1 = nll_1[-length_1:].tolist()
        values_2 = nll_2[-length_2:].tolist()
        values_1 = [-x for x in values_1]
        values_2 = [-x for x in values_2]
        print(values_1)
        print(values_2)

        array_3 = np.array(values_1)
        array_4 = np.array(values_2)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharey=True)

        # 在第一个子图上绘制条形图
        bars1 = ax1.bar(categories_1, values_1, color='skyblue')
        for indice in range(len(bars1)):
            height = bars1[indice].get_height()  # 获取条形的高度，即y值
            # 在条形顶部中央位置添加文本
            ax1.text(bars1[indice].get_x() + bars1[indice].get_width() / 2,  # x坐标：条形中心
                     height - 0.3,  # y坐标：条形顶部上方一点
                     f'{values_1[indice]}',  # 要显示的文本
                     ha='center',  # 水平对齐：居中
                     va='bottom')  # 垂直对齐：底部与指定坐标对齐
        ax1.axhline(y=-avg_nll_1.item(), color='red', linestyle='--', linewidth=2, label=f'mean: {-avg_nll_1.item()}')
        ax1.set_ylabel('log likelihood')  # 通常只在第一个图上设置y轴标签

        # 在第二个子图上绘制条形图
        bars2 = ax2.bar(categories_2, values_2, color='lightgreen')
        ax2.axhline(y=-avg_nll_2.item(), color='red', linestyle='--', linewidth=2, label=f'mean: {-avg_nll_2.item()}')

        for indice in range(len(bars2)):
            height = bars2[indice].get_height()  # 获取条形的高度，即y值
            # 在条形顶部中央位置添加文本
            ax2.text(bars2[indice].get_x() + bars2[indice].get_width() / 2,  # x坐标：条形中心
                     height - 0.3,  # y坐标：条形顶部上方一点
                     f'{values_2[indice]}',  # 要显示的文本
                     ha='center',  # 水平对齐：居中
                     va='bottom')  # 垂直对齐：底部与指定坐标对齐

        ax1.legend()
        ax2.legend()
        plt.savefig(f'25235_pure_{search_args.rerank_template}.png')

        rho1, p_value1 = spearmanr(array_1, array_3)
        rho2, p_value2 = spearmanr(array_2, array_4)

        print(f"斯皮尔曼相关系数: {rho1:.3f}, {rho2:.3f}")
        print(f"P值: {p_value1:.3f}, {p_value2:.3f}")



if __name__ == "__main__":
    main()

