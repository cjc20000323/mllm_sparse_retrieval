from dataclasses import dataclass, field
from typing import Optional, Literal
import transformers

import tevatron.retriever.arguments

coco_file_path = './data/coco/'
flickr_file_path = './data/flickr/'
fashion_iq_file_path = './data/fashion-iq/'
cuhk_pedes_file_path = './data/CUHK-PEDES/'
icfg_pedes_flie_path = './data/ICFG-PEDES/'
rstpreid_file_path = './data/RSTPReid/'


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
    load_kbit: int = field(default=4)
    train_vision_lora: bool = field(default=False)
    train_projector_lora: bool = field(default=False)
    train_mode: str = field(default='')
    gather_save_gradient: bool = field(default=True)
    tau: float = field(default=0.1)
    local_loss: bool = field(default=False)
    batch_size: int = field(default=32)
    task_type: str = field(default='ir')


@dataclass
class ModelArguments(tevatron.retriever.arguments.ModelArguments):
    lora_bias: Literal["none", "all", "lora_only"] = field(default="none")
    base_model_path: str = field(default='./checkpoints/llava-hf-llama3-llava-next-8b-hf')
    lora_model_path: str = field(default='./output/llava-hf-llama3-llava-next-8b-hf')
    use_output_embedding_cluster: bool = field(default=False)
    cluster_sum: int = field(default=8000)
    eol_type: str = field(default='prompteol')
    calculate_type: str = field(default='separate')
    # 当eol_type为all_disassembleeol时，稀疏特征和密集特征都用各方面prompt综合编码
    # 当eol_type为disassembleeol_concrete时，密集特征用原来的，稀疏特征由各方面选词，让后到原来的logit去找
    # 当eol_type为disassembleeol_separate时，密集特征用原来的，稀疏特征用各方面prompt综合编码


@dataclass
class PromptRepsLLMDataArguments(tevatron.retriever.arguments.DataArguments):
    dense_output_dir: str = field(default='./dense_output/')
    sparse_output_dir: str = field(default='./sparse_output/')
    per_device_batch_size: int = field(default=4)
    encode_is_query: bool = field(default=False)
    num_expended_tokens: int = field(default=0, metadata={"help": "Number of expended tokens. Default is 0, "
                                                                  "meaning exact term matching only."})
    is_filtered: bool = field(default=False)
    reps_loc: str = field(default='before_pad')
    sparse_manual: bool = field(default=False)
    sparse_length: int = field(default=128)
    text_sparse_length: int = field(default=128)
    image_sparse_length: int = field(default=128)
    use_few_shot: bool = field(default=False)
    few_shot_sum: int = field(default=200)
    use_cutoff_len: bool = field(default=False)
    cutoff_len: int = field(default=32)
    pad_to_multiple_of: int = field(default=8)
    dataset_suffix: str = field(default='no')
    sparse_value_type: str = field(default='replace')
    sparse_lower_or_upper: str = field(default='lower')
    prompt_type: str = field(default='prompt_5')
    sparse_value_mean: bool = field(default=False)
    sparse_type: str = field(default='single')
    tbpr_type: str = field(default='origin_type')
    print_sparse: bool = field(default=False)


@dataclass
class PromptRepsLLMSearchArguments:
    passage_reps: str = field(default=None, metadata={"help": "Path to passage dense representations"})
    sparse_index: str = field(default=None, metadata={"help": "Path to passage sparse representations"})
    val_passage_reps: str = field(default=None, metadata={"help": "Path to passage dense representations"})
    val_sparse_index: str = field(default=None, metadata={"help": "Path to val passage sparse representations"})
    depth: int = field(default=1000)
    save_dir: str = field(default=None, metadata={"help": "Where to save the run files"})
    quiet: bool = field(default=True, metadata={"help": "Whether to print the progress"})
    use_gpu: bool = field(default=False, metadata={"help": "Whether to use GPU"})
    alpha: float = field(default=0.5, metadata={"help": "The weight for dense retrieval"})
    retrieval_batch_size: int = field(default=0, metadata={"help": "Batch size for retrieval"})
    remove_query: bool = field(default=False, metadata={"help": "Whether to remove query id from the ranking"})
    threads: int = field(default=1, metadata={"help": "Number of threads for sparse retrieval"})
    query_type: str = field(default='text')
    beta: float = field(default=0.5, metadata={"help": 'The weight for sparse retrieval'})
    embedding_type: str = field(default='dense')
    first_stage_search_sum: int = field(default=200)
    use_candidate_sum: bool = field(default=False)
    rerank_num: int = field(default=20, metadata={"help": 'Number of candidates chosen for rerank'})
    rerank_type: str = field(default='pointwise', metadata={"help": 'How to rerank'})
    rerank_batch_size: int = field(default=1, metadata={"help": 'batch size for LLM input when rerank'})
    rerank_template: str = field(default='relevant')
    tuple_sum: int = field(default=20)
    modify_type: str = field(default='no')


@dataclass
class LogitInformationAnalysisArguments:
    logit_information_analysis_text: str = field(default=None)
    logit_information_analysis_image: str = field(default=None)
    logit_information_analysis_type: str = field(default='text')


@dataclass
class PromptGenerationArguments:
    prompt_generation_text: str = field(default=None)
    prompt_generation_image: str = field(default='None')
    prompt_generation_type: str = field(default='t2t')
    demonstration_num: int = field(default=1)
    prompt_generation_model: str = field(default=None)
    case_type: str = field(default='caption')
