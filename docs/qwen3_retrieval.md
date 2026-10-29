# Qwen3-VL-Instruct：PromptEOL 编码、检索与重排

`encode.py`、`search.py` 和 `rerank.py` 现在按检查点 `config.json` 中的
`model_type=qwen3_vl` 进入共享的原生 Qwen 流水线，使用
`Qwen3VLForConditionalGeneration` 和 `AutoProcessor`。本地模型目录改名不影响识别，
也没有按 8B 名称硬编码模型结构。范围是 Dense Instruct；Thinking、MoE 和视频不在此次适配范围。

共享实现放在 `src/qwen_vl_prompts.py`、`src/qwen_vl_encoding.py` 和
`src/qwen_vl_pipeline.py` 中，供 Qwen2、Qwen2.5 与 Qwen3 共用。代码统一通过 `qwen_vl_*` 导入，旧模块路径已移除。
现有索引格式和指纹保持不变，本次清理无需重建索引。
独立 FreeRet 脚本不在此次适配范围。三阶段正式流程与耗时估计见 [三阶段 Qwen 说明](three_stage_qwen.md)。

## 方法与提示词

| 配置 | 稠密表示 | 稀疏表示 |
| --- | --- | --- |
| `prompteol + separate` | 整体概括视角的隐藏状态，L2 归一化 | 同一整体视角的词表分布 |
| `disassembleeol_separate + concat` | 整体概括视角的隐藏状态，L2 归一化 | 各分解视角的词表分布，经 `log1p(relu)` 和既有聚合规则合并 |

英文指令、视角顺序、query/candidate 身份和重排公式沿用项目中的 LLaVA-Mistral 模板。
外层消息和视觉标记由检查点自身的 `apply_chat_template` 生成。没有引入 `<think>`、
`<|box_start|>` 或新 token 作为分隔符，也不会向 Qwen3 强加 Qwen2.5 的默认 system 消息。
整体视角和各分解视角均在原生 assistant 生成前缀末端取表示。

Qwen 的内容列表支持按顺序混排图像和文字，本实现保留各模板的原有顺序，
包括 `reverse_*` 的语义。无需因为迁移到 Qwen 而强制把 image 移到 text 前面；
不同顺序的实际检索效果仍需在同一实验配置下比较。

拼接方法只进行一次完整多模态前向。共享图像和原始文字上下文只处理一次，
各视角仅能关注共享上下文和自己的指令，不能看到其他视角或未来 token。
每条样本分别定位边界、排除 padding，并重置各视角的文字位置。
先用二维 mask 计算原生 MRoPE，再传入四维因果隔离 mask；原生 Qwen3 视觉编码、
DeepStack 特征注入和交错 MRoPE 均保留。使用 SDPA 执行此路径。

## 可运行示例：Flickr 图搜文

以下为 Bash 命令。在已有项目 GPU 环境中运行，依赖包括 Transformers、PyTorch、
Accelerate、DeepSpeed、项目原有的 FAISS、Pyserini/Java、pandas/openpyxl，
以及非手动文本稀疏扩展需要的 NLTK 数据。本地验证版本为 Transformers 4.57.3 / PyTorch 2.9.1。
普通图片路径和图片字节通过 PIL 与原生 processor 处理，不需要额外调用 `qwen-vl-utils`。
数据位置沿用 `src/arguments.py` 的配置。

选择一种方法，在新的空目录中重新编码。`bf16` 可按 GPU 能力换成 `fp16`。

```bash
MODEL=/root/autodl-fs/checkpoints/Qwen3-VL-8B-Instruct
MODEL_NAME=$(basename "$MODEL")
GPUS=1

# 方法一；使用方法二时替换下面两行。
EOL=prompteol
CALC=separate
# EOL=disassembleeol_separate
# CALC=concat

DENSE_ROOT=./qwen3_native_dense
SPARSE_ROOT=./qwen3_native_sparse
COMMON=(--model_name_or_path "$MODEL" --task_type ir
  --dataset_name flickr --dataset_split test
  --eol_type "$EOL" --calculate_type "$CALC" --prompt_type prompt_5
  --per_device_batch_size 1 --reps_loc after_pad --bf16
  --sparse_length 40 --sparse_value_type sum --sparse_lower_or_upper lower
  --num_expended_tokens 0
  --qwen_min_pixels 65536 --qwen_max_pixels 1048576)

# 1. 图搜文的候选库为 text。查询由检索入口在线编码。
deepspeed --num_gpus "$GPUS" src/encode.py "${COMMON[@]}" \
  --encode_type text --output_dir ./qwen3_encode_logs \
  --dense_output_dir "$DENSE_ROOT" --sparse_output_dir "$SPARSE_ROOT"

LEAF="0_no_manual_40_sum_no_cluster_after_pad_${EOL}_lower_no_mean"
RELATIVE="$MODEL_NAME/flickr/text/no_filter/$CALC/prompt_5/test/$LEAF"
DENSE="$DENSE_ROOT/$RELATIVE"
SPARSE="$SPARSE_ROOT/$RELATIVE"

# 2. 只把 collection/ 内的 JSONL 作为 Lucene 文档。
python -B src/qwen_sparse_index.py --sparse-root "$SPARSE" --threads 16

# 3. 固定 alpha；不在测试集上选择 alpha。
SEARCH=("${COMMON[@]}" --query_type image
  --passage_reps "$DENSE" --sparse_index "$SPARSE"
  --use_fixed_alpha --alpha 0.5 --depth 1000 --threads 16)
deepspeed --num_gpus "$GPUS" src/search.py "${SEARCH[@]}" \
  --output_dir ./qwen3_search_results

# 4. 检索后使用生成式评分重排。
deepspeed --num_gpus "$GPUS" src/rerank.py "${SEARCH[@]}" \
  --rerank_type pointwise --rerank_template what_caption_generation \
  --rerank_num 20 --rerank_batch_size 2 --modify_type no \
  --output_dir ./qwen3_rerank_results
```

文搜图时，将建库的 `--encode_type text` 和 `RELATIVE` 中的 `text` 改为 `image`，
检索的 `--query_type image` 改为 `text`。启用 `--sparse_manual` 后，叶目录中的长度字段
变为 `text_sparse_length_image_sparse_length`；编码程序会打印实际产物路径。
不同方法和方向使用各自的索引与结果目录。

`--qwen_min_pixels` / `--qwen_max_pixels` 的单位是图像像素面积，0 使用检查点默认值。
Qwen3 fast processor 的有效限制位于 `size.shortest_edge/longest_edge`，共享组件会统一这些设置，
并检查仅修改一端后是否仍然满足最小值不大于最大值。编码和检索必须使用相同设置。
示例的最大面积 1048576，在 patch_size=16、merge_size=2 时约对应 1024 个视觉 token。
四维 mask 的内存随序列长度平方增长，先用 batch size 1 检查实际显存占用。

## 任务配置

下表同时适用于两种方法；任务、数据集、方向和索引路径需要一同替换。

| 数据集 | task_type | 候选库 encode_type | query_type | 补充配置 |
| --- | --- | --- | --- | --- |
| COCO / Flickr | `ir` | `text` / `image` | `image` / `text` | 双向分别建库 |
| CUHK-PEDES / ICFG-PEDES / RSTPReid | `tbpr` | `image` | `text` | `--tbpr_type origin_type` 或 `type` |
| WebQA | `t2it` | `corpus` | `text` | 候选图文联合编码 |
| ReMuQ / LLaVA | `it2t` | `corpus` | `image` 或 `composed` | 查询图文联合编码 |
| Fashion-IQ | `cir` | `image` | `composed` | `--dataset_split val --cir_type type` 或 `type_1`；`--remove_query` 排除参考图 |

IR 支持 `prompt_1` 至 `prompt_7` 和 `prompt_random_5`，随机视角由 `random_view_seed` 固定。
TBPR `type` 使用对应行人视角；`origin_type` 使用基线定义的原始行人视角。
TBPR `type_1/type_2` 可用于 separate；concat 仅接受基线实际定义了对应视角列表的组合。
Fashion-IQ 和多模态任务使用各自的任务视角列表。
行人/CIR 的索引目录在 split 之后另有 `tbpr_type` / `cir_type` 一层。

按验证集选择 alpha 时，先使用一致配置建立验证集索引，然后移除 `--use_fixed_alpha`，
提供 `--val_passage_reps` 和 `--val_sparse_index`。支持 `--validation_dataset_name`、
`--use_fixed_validation_dataset` 和既有生成式验证集开关；有完整验证输入时，选择过程不使用测试集标签。
非固定模式下，两个 val 参数均未提供时，直接在当前目标 split（默认 test）选择 alpha；
独立 search 脚本仅在显式指定 val 路径时使用验证集。多卡只由 rank 0 扫描一次，
复用最佳结果，输出 `alpha_sweep.csv` 和 `best.csv`。

## 重排模式

| 任务 | 相关性模板 |
| --- | --- |
| IR | `relevant`、`old_relevant`、`query_relevant`、`origin_old_relevant`、对应的 `reverse_*`、`please_relevant`、`in_one_word_relevant`、`precise_caption`、`first_precise_caption`、`role_relevant`、`role_precise_caption`、`role_old_relevant`、`freeret` |
| TBPR | `relevant`、`old_relevant`、`query_relevant`、`origin_old_relevant`、对应的 `reverse_*`、`please_relevant`、`in_one_word_relevant`、`precise_caption`、`first_precise_caption`、`role_relevant`、`role_precise_caption`、`role_old_relevant`、`freeret` |
| WebQA | `relevant`、`old_relevant`、`query_relevant`、`origin_old_relevant` |
| ReMuQ / LLaVA | WebQA 的模式及 `answer_relevant` |
| Fashion-IQ | `relevant`、`old_relevant`、`query_relevant`、`origin_old_relevant`、`please_relevant`、`in_one_word_relevant`、`role_relevant`、`role_old_relevant` |

IR 生成式模板支持 `caption_generation`、`what_caption_generation`、
`caption_generation_2/3/4/5`、`detailed_caption_generation`、`detailed_caption_generation_1`。
TBPR 支持 `caption_generation`、`what_caption_generation`、`describe_caption_generation`。
WebQA、ReMuQ/LLaVA 支持 `caption_generation` 和 `what_caption_generation`。
WebQA/it2t 的 `relevant` 保留既有 `old_relevant` 别名语义。

相关性评分从实际 tokenizer 获取 Yes/No（`freeret` 为 A/B）的单 token ID，
对两者 logits 做 softmax。生成式评分为目标文本的平均 token 对数概率。
`modify_single` 和先验缓存保留既有适用范围：IR 图搜文、it2t。
缓存开关为 `--use_rerank_prior_cache --rerank_prior_cache_dir <path>`。
这里的 `freeret` 是重排模板名。

未知模板、listwise、CIR 生成式重排、依赖 LLaVA `<s>` 的 `caption_generation_6/7`，
以及没有整体稠密表示定义的 `CIR classify_type + concat` 会明确报错。
LLaVA 特定的输出聚类、稀疏 x5、fusion sparse_type、few-shot/自动生成提示词和计时估计
不属于这两个基础方法的适配范围。`--lora` 仍可合并已有兼容 adapter。

## 产物与验证

Qwen3 使用独立版本 `qwen3-prompteol-v1` 和 `qwen3_prompt_manifest.json`。
清单包含实际提示词示例、原生 chat template、词表指纹、图像 processor 类型及有效 size、
patch/merge 参数、模型/LoRA 身份和编码配置。检索前验证这些信息。
旧 Qwen3 产物需要重建；Qwen2.5 清单名称、字段和指纹计算保持兼容。
默认允许原目录重编码并清理对应旧分片；成功前不保留旧完成清单。
旧 `index/` 和 `collection/index/` 都会备份到叶目录的 `index.stale_<唯一后缀>/`，编码后需重建。
新索引位于 `collection/index/`；旧同级索引仍兼容，详见 [索引路径说明](qwen_search_scripts.md)。

编码保存 `corpus_<rank>.pkl` 与 `collection/corpus_<rank>.jsonl`。
检索输出 dense/sparse/hybrid TREC，重排另输出 rerank TREC，
Recall@K 写入 `recall_metrics.json` 和 `best.csv`，自动选优另有 `alpha_sweep.csv`。

```bash
python -B -m unittest discover -s tests -p 'test_qwen*_retrieval.py' -v
python -B -m unittest discover -s tests -p 'test_rerank*.py' -v
```

CPU 测试使用真实原生 processor 和随机初始化的小型 Qwen3-VL，启用两层 DeepStack。
覆盖独立/拼接前向等价、视觉与文字视角隔离、MRoPE、左右 padding、不同尺寸、尾批次、
全部任务的编码/重排、先验缓存、验证集 alpha、改名目录识别，以及本地保存后真实加载。
编码文件到检索重排的集成测试用内存替身代替 FAISS/Lucene，并 mock openpyxl 写入。

本地无 CUDA 与完整模型权重，尚未完成 8B 实际 GPU、真实 FAISS/Lucene 和完整数据集召回验证。
GPU 验收应对每个任务和两种方法执行小规模建库、真实 Lucene 索引、检索及重排，
核对查询 ID、分数有限性、清单中的提示词和 Recall@K 后，再运行完整数据集。

原生架构和处理器参考：[Transformers Qwen3-VL](https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/qwen3_vl)、
[Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)。
