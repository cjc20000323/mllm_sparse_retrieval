import os

import transformers
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    BitsAndBytesConfig
)
from transformers import LlavaProcessor, LlavaForConditionalGeneration, LlavaNextProcessor, \
    LlavaNextForConditionalGeneration, Qwen2_5_VLProcessor, Qwen2_5_VLForConditionalGeneration, AutoModel, \
    AutoProcessor, \
    AutoTokenizer, PhiForCausalLM, Phi3ForCausalLM, AutoModelForCausalLM, LlavaNextConfig
from arguments import PromptRepsLLMDataArguments, ModelArguments
import torch.distributed as dist
from arguments import TrainingArguments
from dataset import CrossModalRetrievalDataset, PromptRepsTrainCollator
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import math

from PIL import Image

from template import text_prompt, img_prompt, text_prompt_no_one_word, img_prompt_no_one_word, \
    img_prompt_no_special_llava_v1_5, text_prompt_no_special_llava_v1_5, text_prompt_qwen_v2_5, img_prompt_qwen_v2_5, \
    img_prompt_intern_vl_v2_5, text_prompt_intern_vl_v2_5
from model import MLLMRetrievalModel
from utils import split_model, load_image, find_all_linear_names
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from constant import llava_next_llama_8b_constant
from trainer import DenseEmbTrainer, allgather

from transformers import TrainerCallback


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def create_scheduler(args, optimizer):
    # base_warm_up = 2500
    # base_step = 2772 * 30  # math.ceil(2838361 / 1024)*30
    # base_rate = base_warm_up / base_step  # 0.03006253006253006
    if 'num_training_steps' not in args:
        args['num_training_steps'] = args['epochs'] * args['step_per_epoch']
    print("### num_training_steps, ", args['num_training_steps'], flush=True)

    args['num_warmup_steps'] = 100
    '''
    if isinstance(args['num_warmup_steps'], float):
        assert 0 <= args['num_warmup_steps'] < 1
        args['num_warmup_steps'] = int(args['num_training_steps'] * args['num_warmup_steps'])
    # args['num_warmup_steps'] = int(args['num_training_steps'] * base_rate)
    '''
    print("### num_warmup_steps, ", args['num_warmup_steps'], flush=True)

    if args['sched'] == 'linear':
        class lr_lambda_class:
            def __init__(self):
                pass

            def __call__(self, current_step):
                if current_step < args.num_warmup_steps:
                    return float(current_step) / float(max(1, args.num_warmup_steps))
                return max(
                    0.0, float(args.num_training_steps - current_step) / float(
                        max(1, args.num_training_steps - args.num_warmup_steps))
                )

        # def lr_lambda(current_step: int):
        #     if current_step < args.num_warmup_steps:
        #         return float(current_step) / float(max(1, args.num_warmup_steps))
        #     return max(
        #         0.0, float(args.num_training_steps - current_step) / float(
        #             max(1, args.num_training_steps - args.num_warmup_steps))
        #     )

        lr_scheduler = LambdaLR(optimizer, lr_lambda_class(), last_epoch=-1)

    else:
        raise NotImplementedError(f"args.sched == {args.sched}")

    return lr_scheduler


class CustomSaveCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir

    def on_epoch_end(self, args, state, control, **kwargs):
        # 获取当前epoch和模型
        epoch = int(state.epoch)
        model = kwargs.get("model")

        # 构造保存路径
        save_path = f"{self.output_dir}/epoch_{epoch}"

        # 无条件保存模型和分词器
        model.encoder.save_pretrained(save_path)


class LossPlotCallback(TrainerCallback):
    def __init__(self, model):
        self.model = model

    def on_train_end(self, args, state, control, **kwargs):
        steps, train_loss, eval_loss = [], [], []
        for log in state.log_history:
            if "loss" in log:
                steps.append(log["step"])
                train_loss.append(log["loss"])

        plt.plot(steps[:len(train_loss)], train_loss, label="Train Loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"./loss_curve_{self.model}.png")


from accelerate import Accelerator, DeepSpeedPlugin
import deepspeed


def main():
    parser = HfArgumentParser((ModelArguments, PromptRepsLLMDataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: PromptRepsLLMDataArguments
    training_args: TrainingArguments

    accelerator = Accelerator(deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config=training_args.deepspeed # 指向配置文件
        ))
    print(f"初始化成功! 分布式类型: {accelerator.state.distributed_type}")

    print("初始化后状态检查:")
    print(f"  num_processes: {getattr(accelerator.state, 'num_processes', '未定义')}")
    print(f"  process_index: {getattr(accelerator.state, 'process_index', '未定义')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_map = "cuda"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(os.environ.get("WORLD_SIZE"))
    ddp = world_size != 1
    print(ddp)

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
        if training_args.load_kbit == 4:
            encoder = LlavaNextForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                        quantization_config=BitsAndBytesConfig(
                                                                            load_in_4bit=True,
                                                                            bnb_4bit_compute_dtype=torch_type,
                                                                            bnb_4bit_use_double_quant=True,
                                                                            bnb_4bit_quant_type='nf4'
                                                                        ),
                                                                        device_map=device_map,
                                                                        torch_dtype=torch_type)
        else:
            encoder = LlavaNextForConditionalGeneration.from_pretrained(model_args.model_name_or_path,
                                                                        load_in_8bit=training_args.load_kbit == 8,
                                                                        load_in_4bit=training_args.load_kbit == 4,
                                                                        device_map=device_map,
                                                                        torch_dtype=torch_type)
        processor = LlavaNextProcessor.from_pretrained(model_args.model_name_or_path)
        if 'royokong-e5-v' in model_args.model_name_or_path:
            setattr(processor, "patch_size", 14)  # hack for pass

    if data_args.reps_loc == 'after_pad':
        processor.tokenizer.padding_side = "left"
        processor.tokenizer.padding = True

    lora_modules = []
    full_modules = []
    if model_args.lora:
        if training_args.load_kbit == 4 or training_args.load_kbit == 8:
            encoder = prepare_model_for_kbit_training(encoder)
        if 'llama3-llava-next-8b' in model_args.model_name_or_path:
            target_modules = find_all_linear_names(encoder, llava_next_llama_8b_constant['llm'])
            lora_modules.extend(target_modules)
        else:
            target_modules = find_all_linear_names(encoder, llava_next_llama_8b_constant['llm'])
            lora_modules.extend(target_modules)

        if training_args.train_vision_lora:
            if 'llama3-llava-next-8b' in model_args.model_name_or_path:
                target_modules = find_all_linear_names(encoder, llava_next_llama_8b_constant['vision_encoder'])
                lora_modules.extend(target_modules)
            else:
                target_modules = find_all_linear_names(encoder, llava_next_llama_8b_constant['vision_encoder'])
                lora_modules.extend(target_modules)

        if training_args.train_projector_lora:
            if 'llama3-llava-next-8b' in model_args.model_name_or_path:
                target_modules = find_all_linear_names(encoder, llava_next_llama_8b_constant['projector'])
                lora_modules.extend(target_modules)
            else:
                target_modules = find_all_linear_names(encoder, llava_next_llama_8b_constant['projector'])
                lora_modules.extend(target_modules)

        config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=lora_modules,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        encoder = get_peft_model(encoder, config)


    else:
        pass

    # encoder.enable_input_require_grads()
    model = MLLMRetrievalModel(encoder)
    # model.encoder.gradient_checkpointing_disable()
    # model.enable_gradient_checkpointing()

    model.encoder.gradient_checkpointing_enable()
    model.encoder.config.use_cache = False

    train_dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'train', 'single', data_args)

    sampler = Data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(), shuffle=True,
                                      rank=accelerator.process_index)
    train_dataloader = Data.DataLoader(dataset=train_dataset, sampler=sampler, pin_memory=True,
                                       batch_size=training_args.per_device_train_batch_size, shuffle=False)

    if 'llava-hf-llava-1.5-7b-hf' in model_args.model_name_or_path or 'llava-hf-llava-v1.6-vicuna-7b-hf' in model_args.model_name_or_path:
        prompt = img_prompt_no_special_llava_v1_5
    elif 'Qwen2.5-VL-7B-Instruct' in model_args.model_name_or_path or 'Qwen2.5-VL-3B-Instruct' in model_args.model_name_or_path:
        prompt = img_prompt_qwen_v2_5
    elif 'InternVL2_5-8B' in model_args.model_name_or_path or 'InternVL2_5-4B' in model_args.model_name_or_path:
        prompt = img_prompt_intern_vl_v2_5
    else:
        prompt = img_prompt

    if dist.get_rank() == 0:
        print('model.parameters()')
        print(model.parameters())

    params_with_grad = [param for param in model.parameters() if param.requires_grad]

    optimizer = torch.optim.AdamW(
        params_with_grad,
        lr=training_args.learning_rate,
        weight_decay=0.01
    )

    arg_sche = {'sched': 'linear', 'lr': training_args.learning_rate, 'epochs': training_args.num_train_epochs,
                'num_warmup_steps': 100, 'step_per_epoch': math.ceil(
            data_args.few_shot_sum / (training_args.per_device_train_batch_size * world_size))}
    arg_sche = AttrDict(arg_sche)
    lr_scheduler = create_scheduler(arg_sche, optimizer)

    model.encoder.gradient_checkpointing_enable()

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        [model, optimizer, train_dataloader, lr_scheduler]
    )
    loss_list = []
    steps = []
    step_count = 0

    for i in range(int(training_args.num_train_epochs)):

        # 创建带参数的进度条
        progress_bar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=f"Epoch {i + 1}/{training_args.num_train_epochs}",
            postfix={},  # 存储要显示的参数
            disable=not accelerator.is_main_process  # 仅主进程显示
        )
        for batch_idx, (texts, imgs_path, text_ids, img_ids) in progress_bar:
            step_count += 1
            raw_images = [Image.open(path).convert('RGB') for path in imgs_path]
            img_inputs = processor(images=raw_images, text=[prompt] * len(imgs_path), return_tensors="pt",
                                   padding=True)
            imgs = img_inputs.to(device)
            _, text_reps = model.encode_data(texts, 'text', processor, device, model_args, data_args)
            _, img_reps = model.encode_data(imgs, 'image', processor, device, model_args, data_args)
            text_reps = F.normalize(text_reps, dim=-1)
            img_reps = F.normalize(img_reps, dim=-1)
            all_image_reps = allgather(img_reps, dist.get_rank(), dist.get_world_size())
            all_text_reps = allgather(text_reps, dist.get_rank(), dist.get_world_size())

            loss_fct = nn.CrossEntropyLoss()
            logits = all_image_reps @ all_text_reps.t() / training_args.tau
            labels = torch.arange(all_text_reps.shape[0]).long().to(device)
            loss_i2t = loss_fct(logits, labels)
            loss_t2i = loss_fct(logits.t(), labels)
            loss = (loss_t2i + loss_i2t) / 2

            accelerator.backward(loss)

            # 梯度裁剪
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

            # 参数更新
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            all_loss = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
            dist.all_gather(tensor_list=all_loss, tensor=loss.contiguous())

            loss_tensor = torch.tensor(all_loss).mean()
            loss_list.append(loss_tensor)
            steps.append(step_count)

            # 在梯度累积同步点更新显示
            if accelerator.sync_gradients:
                # 获取当前学习率
                current_lr = optimizer.param_groups[0]['lr']

                # 更新进度条参数显示
                progress_bar.set_postfix({
                    'loss': f"{loss_tensor.item():.4f}",
                    'lr': f"{current_lr:.6f}"
                })

            model.encoder.save_pretrained(training_args.output_dir + f'/epoch_{i + 1}')

        # 每个epoch结束后更新并关闭进度条
        progress_bar.close()

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model.save_pretrained(training_args.output_dir)
    accelerator.end_training()
    plt.plot(steps[:len(loss_list)], loss_list, label="Train Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"./loss_curve_{training_args.output_dir[9:]}.png")


if __name__ == "__main__":
    main()
