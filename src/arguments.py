from dataclasses import dataclass, field
from typing import Optional, Literal
import transformers

import tevatron.retriever.arguments

model_begin_indice = 29
dataset_path_prefix = '/root/autodl-tmp/'
model_path_prefix = '/root/autodl-fs/'

coco_file_path = dataset_path_prefix + 'data/coco/'
flickr_file_path = dataset_path_prefix + 'data/flickr/'
fashion_iq_file_path = dataset_path_prefix + 'data/fashion-iq/'
cuhk_pedes_file_path =dataset_path_prefix + 'data/CUHK-PEDES/'
icfg_pedes_flie_path = dataset_path_prefix + 'data/ICFG-PEDES/'
rstpreid_file_path = dataset_path_prefix + 'data/RSTPReid/'
webqa_file_path = dataset_path_prefix + 'data/WebQA/'
remuq_file_path = dataset_path_prefix + 'data/UMRB-ReMuQ/'
llava_file_path = dataset_path_prefix + 'data/UMRB-LLaVA/'
edis_file_path = dataset_path_prefix + 'data/EDIS/'


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
    base_model_path: str = field(default=model_path_prefix + f'checkpoints/llava-hf-llama3-llava-next-8b-hf')
    lora_model_path: str = field(default='./output/llava-hf-llama3-llava-next-8b-hf')
    use_output_embedding_cluster: bool = field(default=False)
    cluster_sum: int = field(default=8000)
    eol_type: str = field(default='prompteol')
    calculate_type: str = field(default='separate')
    dspy_model_path: str = field(default=model_path_prefix + f'checkpoints/Meta-Llama-3.1-8B-Instruct')
    # 当eol_type为all_disassembleeol时，稀疏特征和密集特征都用各方面prompt综合编码
    # 当eol_type为disassembleeol_concrete时，密集特征用原来的，稀疏特征由各方面选词，让后到原来的logit去找
    # 当eol_type为disassembleeol_separate时，密集特征用原来的，稀疏特征用各方面prompt综合编码


@dataclass
class PromptRepsLLMDataArguments(tevatron.retriever.arguments.DataArguments):
    qwen_min_pixels: int = field(default=0, metadata={"help": "Qwen image minimum pixels; 0 uses checkpoint defaults"})
    qwen_max_pixels: int = field(default=0, metadata={"help": "Qwen image maximum pixels; 0 uses checkpoint defaults"})
    dense_output_dir: str = field(default='./dense_output/')
    sparse_output_dir: str = field(default='./sparse_output/')
    use_generated_validation_data: bool = field(
        default=False,
        metadata={
            "help": "Load the validation split produced by "
                    "generate_llava70b_dataset_text.py. Test data is never replaced."
        },
    )
    generated_validation_data_root: Optional[str] = field(
        default=None,
        metadata={
            "help": "Output root of generate_llava70b_dataset_text.py. Defaults to "
                    "GENERATED_LLAVA70B_DATA_ROOT or <project>/generated_llava70b_data."
        },
    )
    use_llava_next_training_prompts: bool = field(
        default=False,
        metadata={
            "help": "Mark generated validation data created with the "
                    "LLaVA-NeXT-style training prompts so its encoded "
                    "representations use a distinct output suffix."
        },
    )
    use_few_shot_examples: bool = field(
        default=False,
        metadata={
            "help": "Mark generated validation data created with multimodal "
                    "few-shot examples so encoded representations use a "
                    "distinct output suffix."
        },
    )
    few_shot_num: int = field(
        default=2,
        metadata={
            "help": "Number of fixed examples used to generate each validation item."
        },
    )
    few_shot_split: str = field(
        default='test',
        metadata={
            "help": "Source split of the generation examples: val or test."
        },
    )
    use_source_dataset_subset: bool = field(
        default=False,
        metadata={
            "help": "Mark that the validation data is a source-dataset subset "
                    "so encoded representations use a distinct output suffix."
        },
    )
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
    random_view_seed: int = field(
        default=42,
        metadata={
            "help": "Seed used by prompt_random_5 to select the same five vocabulary tokens in encode and search."
        },
    )
    prompt_generation: bool = field(default=False)
    prompt_generation_model: str = field(default='llama')
    sparse_value_mean: bool = field(default=False)
    # Kept for CLI compatibility; with prompteol this applies a fixed x5 sparse-weight multiplier.
    sparse_logits_multiply_view_count: bool = field(
        default=False,
        metadata={
            "help": "Use the original vocabulary logits multiplied by 5 as sparse weights. "
                    "Supported only for Llama3-8B and LLaVA-Mistral-7B separate prompteol encoding."
        },
    )
    sparse_type: str = field(default='single')
    tbpr_type: str = field(default='origin_type')
    cir_type: str = field(default='type')
    print_sparse: bool = field(default=False)
    composed_top_type: str = field(default='text')
    composed_top_k: int = field(default=3)


@dataclass
class PromptRepsLLMSearchArguments:
    passage_reps: str = field(default=None, metadata={"help": "Path to passage dense representations"})
    sparse_index: str = field(default=None, metadata={"help": "Path to passage sparse representations"})
    val_passage_reps: str = field(default=None, metadata={"help": "Path to passage dense representations"})
    val_sparse_index: str = field(default=None, metadata={"help": "Path to val passage sparse representations"})
    use_fixed_validation_dataset: bool = field(
        default=False,
        metadata={
            "help": "Use Flickr30K/RSTPReid/LLaVA as the validation dataset according to the test task type."
        },
    )
    validation_dataset_name: str = field(
        default=None,
        metadata={
            "help": "Dataset whose val split is used for DSPy prompt evaluation. "
                    "Overrides --use_fixed_validation_dataset when provided."
        },
    )
    depth: int = field(default=1000)
    val_depth: Optional[int] = field(
        default=None,
        metadata={
            "help": "Validation retrieval depth. Falls back to --depth when omitted."
        },
    )
    test_depth: Optional[int] = field(
        default=None,
        metadata={
            "help": "Test retrieval depth. Falls back to --depth when omitted."
        },
    )
    save_dir: str = field(default=None, metadata={"help": "Where to save the run files"})
    quiet: bool = field(default=True, metadata={"help": "Whether to print the progress"})
    use_gpu: bool = field(default=False, metadata={"help": "Whether to use GPU"})
    alpha: float = field(default=0.5, metadata={"help": "The weight for dense retrieval"})
    use_fixed_alpha: bool = field(
        default=False,
        metadata={
            "help": "Use --alpha directly for final hybrid fusion instead of selecting alpha on validation."
        },
    )
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
    rerank_batch_size: int = field(default=1, metadata={"help": 'Candidates per reranking model forward per GPU; must be positive'})
    rerank_template: str = field(default='relevant')
    tuple_sum: int = field(default=20)
    modify_type: str = field(default='no')
    use_rerank_prior_cache: bool = field(
        default=False,
        metadata={"help": "Precompute/load all text-only caption priors; requires modify_single and a supported text-candidate task"},
    )
    rerank_prior_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Shared prior cache directory; defaults to <project>/cache/rerank_prior"},
    )


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
    # prompt_generation_model: str = field(default=None)
    case_type: str = field(default='caption')
    dspy_strength: str = field(default='light')
    use_initial_dspy_aspects: bool = field(
        default=False,
        metadata={
            "help": "Use the project's task-specific retrieval aspects as GEPA round 1; "
                    "GEPA then reflects on retrieval failures, rewrites the Instruction, "
                    "and the highest-scoring evaluated aspect set is retained."
        },
    )
    force_regenerate_dspy_aspects: bool = field(
        default=False,
        metadata={"help": "Ignore the locked DSPy aspects file and generate a new one."},
    )
