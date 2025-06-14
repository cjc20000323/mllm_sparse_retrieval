import os

import transformers
from matplotlib import pyplot as plt
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
import torch.utils.data as Data
import torch.nn.functional as F

from template import text_prompt, img_prompt, text_prompt_no_one_word, img_prompt_no_one_word, \
    img_prompt_no_special_llava_v1_5, text_prompt_no_special_llava_v1_5, text_prompt_qwen_v2_5, img_prompt_qwen_v2_5, \
    img_prompt_intern_vl_v2_5, text_prompt_intern_vl_v2_5
from model import MLLMRetrievalModel
from utils import split_model, load_image, find_all_linear_names
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from constant import llava_next_llama_8b_constant
from trainer import DenseEmbTrainer

from transformers import TrainerCallback


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


def compute_metrics(eval_pred):
    predictions, labels = eval_pred


def main():
    from accelerate import Accelerator
    accelerator = Accelerator()

    parser = HfArgumentParser((ModelArguments, PromptRepsLLMDataArguments, TrainingArguments))

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_args: ModelArguments
    data_args: PromptRepsLLMDataArguments
    training_args: TrainingArguments

    print(training_args.local_loss)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_map = "cuda"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print(os.environ.get("WORLD_SIZE"))
    ddp = world_size != 1
    print(ddp)
    # if ddp and False:
    gradient_accumulation_steps = training_args.batch_size // training_args.per_device_train_batch_size
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        # gradient_accumulation_steps = gradient_accumulation_steps // world_size

        if not dist.is_initialized():
            torch.distributed.init_process_group("nccl")
        rank, world_size = torch.distributed.get_rank(), torch.distributed.get_world_size()
        device_id = rank % torch.cuda.device_count()
        device = torch.device(device_id)
        torch.cuda.set_device(device)
        gradient_accumulation_steps = gradient_accumulation_steps // dist.get_world_size()

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

        if dist.get_rank() == 0:
            print(lora_modules)

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

    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            print(f"{name} {param.requires_grad}")

        '''
        for name, param in model.named_parameters():
            print(f"Param ID: {id(param)}, Name: {name}")
        '''

    train_dataset = CrossModalRetrievalDataset(data_args.dataset_name, processor, 'train', 'single', data_args)

    data_collator = PromptRepsTrainCollator(processor, model_args, device)

    if training_args.train_mode == 'dense_emb':
        trainer = DenseEmbTrainer(
            model=model,
            train_dataset=train_dataset,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=training_args.per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=100,
                num_train_epochs=training_args.num_train_epochs,
                learning_rate=training_args.learning_rate,
                fp16=True if training_args.fp16 else False,
                bf16=True if training_args.bf16 else False,
                eval_strategy="no",
                save_strategy="no",
                eval_steps=None,
                output_dir=training_args.output_dir,
                save_total_limit=100,
                load_best_model_at_end=False,
                # ddp_find_unused_parameters=False if ddp else None,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=False,
                report_to=None,
                deepspeed=training_args.deepspeed,
                logging_steps=1,
                lr_scheduler_type="cosine",
                gradient_checkpointing_kwargs={"use_reentrant": False},
            ),
            data_collator=data_collator,
            callbacks=[CustomSaveCallback(training_args.output_dir), LossPlotCallback(training_args.output_dir[9:])]
        )
        if dist.get_rank() == 0:
            print('Trainer has been created.')
    else:
        trainer = DenseEmbTrainer(
            model=model,
            train_dataset=train_dataset,
            args=transformers.TrainingArguments(
                per_device_train_batch_size=training_args.per_device_train_batch_size,
                gradient_accumulation_steps=gradient_accumulation_steps,
                warmup_steps=100,
                num_train_epochs=training_args.num_train_epochs,
                learning_rate=training_args.learning_rate,
                fp16=True if training_args.fp16 else False,
                bf16=True if training_args.bf16 else False,
                eval_strategy="no",
                save_strategy="no",
                eval_steps=None,
                output_dir=training_args.output_dir,
                save_total_limit=100,
                load_best_model_at_end=False,
                # ddp_find_unused_parameters=False if ddp else None,
                ddp_find_unused_parameters=False if ddp else None,
                group_by_length=False,
                report_to=None,
                deepspeed=training_args.deepspeed,
                logging_steps=1,
                lr_scheduler_type="cosine",
                gradient_checkpointing_kwargs={"use_reentrant": False},
            ),
            data_collator=data_collator,
            callbacks=[CustomSaveCallback(training_args.output_dir), LossPlotCallback(training_args.output_dir[9:])]
        )
        if dist.get_rank() == 0:
            print('Trainer has been created.')

    trainer.model_args = model_args
    trainer.data_args = data_args
    trainer.device = device
    trainer.processor = processor
    trainer.gather_save_gradient = training_args.gather_save_gradient
    trainer.tau = training_args.tau
    trainer.local_loss = training_args.local_loss
    trainer.train()

    model.encoder.save_pretrained(training_args.output_dir)

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
