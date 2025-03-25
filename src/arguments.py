from dataclasses import dataclass, field
from typing import Optional
import transformers

import tevatron.retriever.arguments

coco_file_path = './data/coco/'
flickr_file_path = './data/flickr/'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    q_flops_loss_factor: float = field(default=0.01)
    p_flops_loss_factor: float = field(default=0.01)
    eval_data_percentage: float = field(default=0.1)
    max_eval_samples: int = field(default=None)
    max_train_samples: int = field(default=None)
    save_early_checkpoints: bool = field(default=False)
    hybrid_training: bool = field(default=False)
    early_stopping_patience: int = field(default=None)
    encode_type: str = field(default='text')


@dataclass
class PromptRepsLLMDataArguments(tevatron.retriever.arguments.DataArguments):
    dense_output_dir: str = field(default='./dense_output/')
    sparse_output_dir: str = field(default='./sparse_output/')
    per_device_batch_size: int = field(default=4)
    encode_is_query: bool = field(default=False)


@dataclass
class PromptRepsLLMSearchArguments:
    passage_reps: str = field(default=None, metadata={"help": "Path to passage dense representations"})
    sparse_index: str = field(default=None, metadata={"help": "Path to passage sparse representations"})
    depth: int = field(default=1000)
    save_dir: str = field(default=None, metadata={"help": "Where to save the run files"})
    quiet: bool = field(default=True, metadata={"help": "Whether to print the progress"})
    use_gpu: bool = field(default=False, metadata={"help": "Whether to use GPU"})
    alpha: float = field(default=0.5, metadata={"help": "The weight for dense retrieval"})
    batch_size: int = field(default=128, metadata={"help": "Batch size for retrieval"})
    remove_query: bool = field(default=False, metadata={"help": "Whether to remove query id from the ranking"})
    threads: int = field(default=1, metadata={"help": "Number of threads for sparse retrieval"})
    query_type: str = field(default='text')
