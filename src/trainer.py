import torch
import torch.nn as nn
from transformers import Trainer
import torch.distributed as dist

from arguments import PromptRepsLLMDataArguments

NIL_DATASET = True
from transformers import ProcessorMixin
import tevatron.retriever.arguments
import torch.nn.functional as F

class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, rank, world_size):
        output = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(output, tensor.contiguous())
        ctx.rank = rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank: ctx.batch_size * (ctx.rank + 1)],
            None,
            None
        )


allgather = AllGather.apply


class DenseEmbTrainer(Trainer):
    processor: ProcessorMixin = None
    model_args: tevatron.retriever.arguments.ModelArguments = None
    data_args: PromptRepsLLMDataArguments = None
    device: torch.device = None
    tau: float = 0.1
    gather_save_gradient: bool = True
    local_loss: bool = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        '''
        texts = inputs['texts']
        text_logits, text_reps = model.encode_data(texts, 'text', self.processor, self.device, self.model_args,
                                                   self.data_args)
        imgs = inputs['imgs'].to(self.device)
        img_logits, img_reps = model.encode_data(imgs, 'image', self.processor, self.device, self.model_args,
                                                 self.data_args)
        '''
        texts = inputs['texts']
        imgs = inputs['imgs'].to(self.device)
        output = model(texts, imgs, self.processor, self.device, self.model_args, self.data_args)
        text_reps, img_reps = output['text_reps'], output['img_reps']

        text_reps = F.normalize(text_reps, dim=-1)
        img_reps = F.normalize(img_reps, dim=-1)

        if dist.is_initialized():
            all_image_reps = allgather(img_reps, dist.get_rank(), dist.get_world_size())
            all_text_reps = allgather(text_reps, dist.get_rank(), dist.get_world_size())
        else:
            all_image_reps = allgather(img_reps, dist.get_rank(), dist.get_world_size())
            all_text_reps = allgather(text_reps, dist.get_rank(), dist.get_world_size())

            '''
            
            all_image_reps = [torch.zeros_like(img_reps) for _ in range(dist.get_world_size())]
            all_text_reps = [torch.zeros_like(text_reps) for _ in range(dist.get_world_size())]

            dist.all_gather(tensor_list=all_image_reps, tensor=img_reps.contiguous())
            dist.all_gather(tensor_list=all_text_reps, tensor=text_reps.contiguous())

            all_image_reps[dist.get_rank()] = img_reps
            all_text_reps[dist.get_rank()] = text_reps

            if self.gather_save_gradient:
                all_image_reps = torch.cat(all_image_reps)
                all_text_reps = torch.cat(all_text_reps)
            else:
                all_image_reps = torch.cat(all_image_reps).clone().detach()
                all_text_reps = torch.cat(all_text_reps).clone().detach()
                
            '''
        '''
        else:
            if self.gather_save_gradient:
                all_image_reps = img_reps
                all_text_reps = text_reps
            else:
                all_image_reps = img_reps.clone().detach()
                all_text_reps = text_reps.clone().detach()
        '''

        loss_fct = nn.CrossEntropyLoss()
        logits = all_image_reps @ all_text_reps.t() / self.tau
        labels = torch.arange(all_text_reps.shape[0]).long().to(self.device)
        loss_i2t = loss_fct(logits, labels)
        loss_t2i = loss_fct(logits.t(), labels)
        loss = (loss_t2i + loss_i2t) / 2

        '''
        if self.local_loss:
            i2t_sim = img_reps @ all_text_reps.t() / self.tau
            t2i_sim = text_reps @ all_image_reps.t() / self.tau

            labels = torch.arange(text_reps.size(0)).long().to(self.device)
            if dist.is_initialized():
                labels += dist.get_rank() * len(texts)

            loss_i2t = loss_fct(i2t_sim, labels)
            loss_t2i = loss_fct(t2i_sim, labels)
            loss = (loss_t2i + loss_i2t) / 2
        else:
            i2t_sim = all_image_reps @ all_text_reps.t() / self.tau
            t2i_sim = all_text_reps @ all_image_reps.t() / self.tau

            labels = torch.arange(all_text_reps.size(0)).long().to(self.device)
            loss_i2t = loss_fct(i2t_sim, labels)
            loss_t2i = loss_fct(t2i_sim, labels)
            loss = (loss_t2i + loss_i2t) / 2
        '''
        return loss
