# Qwen2-VL-Instruct：PromptEOL 编码、检索与重排

`encode.py`、`search.py` 和 `rerank.py` 按检查点 `config.json` 中的
`model_type=qwen2_vl` 进入共享 `qwen_vl_*` 流水线。模型使用原生
`Qwen2VLForConditionalGeneration` 和 `AutoProcessor`；本地目录改名、参数规模变化均不影响配置路由。
本次范围是 Qwen2-VL-Instruct 图像任务的两个基础方法，没有改动独立 dense、FreeRet、训练入口，
也没有重构 `model.py` 中的 `MLLMRetrievalModel` 接口。
三阶段正式流程与耗时估计已复用共享组件，见 [三阶段 Qwen 说明](three_stage_qwen.md)。

## 方法与提示词

| 配置 | 稠密表示 | 稀疏表示 |
| --- | --- | --- |
| `prompteol + separate` | 整体概括视角，L2 归一化 | 同一整体视角的词表分布 |
| `disassembleeol_separate + concat` | 整体概括视角，L2 归一化 | 分解视角的词表分布，经 `log1p(relu)` 和原有规则聚合 |

检索与重排以项目已有的 **LLaVA-Mistral** 为正文基准：复用相同的英文指令、视角列表、
query/candidate 身份、normal/reverse 图文顺序和评分定义。外层消息使用 Qwen2 检查点自己的
`apply_chat_template`，在原生 assistant 生成前缀末端取表示。不会人为添加 `<think>`、
`<|box_start|>`、LLaVA BOS 或新分隔 token，也不会统一强制 image 在 text 前面。

concat 方法只执行一次完整多模态前向。每条样本单独定位图像展开后的视角边界，
各视角共享图像和原始文字上下文，独立关注自己的指令。先用二维 padding mask 计算原生 MRoPE，
再使用四维因果隔离 mask，重置各视角文字位置，屏蔽其他视角和未来 token。此路径使用 SDPA。

## 可运行示例：Flickr 图搜文

以下为 Bash 命令。需要项目原有的 GPU 环境和数据路径，以及 PyTorch、Transformers、
Accelerate、DeepSpeed、FAISS、Pyserini/Java、pandas/openpyxl；非手动文本稀疏扩展还需要 NLTK 数据。
本地验证环境为 PyTorch 2.9.1 / Transformers 4.57.3。普通图片路径和字节通过 PIL 与原生 processor
处理，不需要额外调用 `qwen-vl-utils`。数据路径沿用 `src/arguments.py`。

选择一种方法，使用新的空输出目录建库；`bf16` 可按 GPU 能力替换为 `fp16`。

```bash
MODEL=/root/autodl-fs/checkpoints/Qwen2-VL-7B-Instruct
MODEL_NAME=$(basename "$MODEL")
GPUS=1

EOL=prompteol
CALC=separate
# 方法二：替换上面两行
# EOL=disassembleeol_separate
# CALC=concat

DENSE_ROOT=./qwen2_native_dense
SPARSE_ROOT=./qwen2_native_sparse
COMMON=(--model_name_or_path "$MODEL" --task_type ir
  --dataset_name flickr --dataset_split test
  --eol_type "$EOL" --calculate_type "$CALC" --prompt_type prompt_5
  --per_device_batch_size 1 --reps_loc after_pad --bf16
  --sparse_length 40 --sparse_value_type sum --sparse_lower_or_upper lower
  --num_expended_tokens 0
  --qwen_min_pixels 200704 --qwen_max_pixels 802816)

# 1. 编码候选文本；检索入口会在线编码查询图片。
deepspeed --num_gpus "$GPUS" src/encode.py "${COMMON[@]}" \
  --encode_type text --output_dir ./qwen2_encode_logs \
  --dense_output_dir "$DENSE_ROOT" --sparse_output_dir "$SPARSE_ROOT"

LEAF="0_no_manual_40_sum_no_cluster_after_pad_${EOL}_lower_no_mean"
RELATIVE="$MODEL_NAME/flickr/text/no_filter/$CALC/prompt_5/test/$LEAF"
DENSE="$DENSE_ROOT/$RELATIVE"
SPARSE="$SPARSE_ROOT/$RELATIVE"

# 2. 只索引 collection/ 内的 JSONL 文档。
python -B src/qwen_sparse_index.py --sparse-root "$SPARSE" --threads 16

# 3. 固定 alpha，不在测试集选择融合权重。
SEARCH=("${COMMON[@]}" --query_type image
  --passage_reps "$DENSE" --sparse_index "$SPARSE"
  --use_fixed_alpha --alpha 0.5 --depth 1000 --threads 16)
deepspeed --num_gpus "$GPUS" src/search.py "${SEARCH[@]}" \
  --output_dir ./qwen2_search_results

# 4. 使用相同查询编码与候选索引，随后进行生成式重排。
deepspeed --num_gpus "$GPUS" src/rerank.py "${SEARCH[@]}" \
  --rerank_type pointwise --rerank_template what_caption_generation \
  --rerank_num 20 --rerank_batch_size 2 --modify_type no \
  --output_dir ./qwen2_rerank_results
```

文搜图时，将建库的 `--encode_type text` 和 `RELATIVE` 中的 `text` 改成 `image`，
检索的 `--query_type image` 改成 `text`。启用 `--sparse_manual` 后，叶目录的长度字段
变成 `text_sparse_length_image_sparse_length`；编码程序会打印实际路径。

`qwen_min_pixels/qwen_max_pixels` 单位为像素面积，0 使用检查点默认值。
共享组件统一处理 fast processor 的 `size` 和 slow processor 的 `min_pixels/max_pixels`，
并检查只修改一个端点后是否仍满足最小值不大于最大值。编码、检索和重排使用相同设置。
示例的面积范围在 patch_size=14、merge_size=2 时约对应 256–1024 个视觉 token。
concat 的四维 mask 内存随序列长度平方增长，应先用 batch size 1 检查显存。

## 任务配置

下表同时适用于两种方法。每次更改任务、方法或方向，都使用对应的索引和结果目录。

| 数据集 | task_type | 候选库 encode_type | query_type | 补充设置 |
| --- | --- | --- | --- | --- |
| COCO / Flickr | `ir` | `text` / `image` | `image` / `text` | 双向分别建库 |
| CUHK-PEDES / ICFG-PEDES / RSTPReid | `tbpr` | `image` | `text` | `--tbpr_type origin_type` 或 `type` |
| WebQA | `t2it` | `corpus` | `text` | 候选图文联合编码 |
| ReMuQ / LLaVA | `it2t` | `corpus` | `image` 或 `composed` | 查询图文联合编码 |
| Fashion-IQ | `cir` | `image` | `composed` | `--dataset_split val --cir_type type` 或 `type_1`；`--remove_query` 排除参考图 |

IR 支持 `prompt_1` 至 `prompt_7` 和由 `random_view_seed` 固定的 `prompt_random_5`。
TBPR `origin_type` 使用原始行人视角，`type` 使用对应行人视角；`type_1/type_2` 的 concat
组合仅在基线确实定义对应视角列表时接受。CIR 和图文融合任务使用各自的任务视角。
行人/CIR 的索引路径在 split 后另有 `tbpr_type/cir_type` 一层。

按验证集选择 alpha 时，先用一致配置建立验证集索引，再移除 `--use_fixed_alpha`，提供
`--val_passage_reps` 和 `--val_sparse_index`。保留原有验证集选择和生成式验证集开关，
提供完整验证输入时，权重选择不使用测试集标签。非固定模式下，两个 val 参数均未提供时，
直接在当前目标 split（默认 test）选择 alpha；独立 search 脚本仅在显式指定 val 路径时使用验证集。
多卡只由 rank 0 扫描一次，复用最佳结果，输出 `alpha_sweep.csv` 和 `best.csv`。

## 重排支持矩阵

| 任务 | 相关性模式 |
| --- | --- |
| IR / TBPR | `relevant`、`old_relevant`、`query_relevant`、`origin_old_relevant`、对应 `reverse_*`、`please_relevant`、`in_one_word_relevant`、`precise_caption`、`first_precise_caption`、`role_relevant`、`role_precise_caption`、`role_old_relevant`、`freeret` |
| WebQA | `relevant`、`old_relevant`、`query_relevant`、`origin_old_relevant` |
| ReMuQ / LLaVA | WebQA 的模式及 `answer_relevant` |
| Fashion-IQ | `relevant`、`old_relevant`、`query_relevant`、`origin_old_relevant`、`please_relevant`、`in_one_word_relevant`、`role_relevant`、`role_old_relevant` |

IR 生成式模式支持 `caption_generation`、`what_caption_generation`、`caption_generation_2/3/4/5`、
`detailed_caption_generation`、`detailed_caption_generation_1`。TBPR 支持前两种标准模式与
`describe_caption_generation`；WebQA、ReMuQ/LLaVA 支持前两种标准模式。

相关性评分对 tokenizer 实际编码得到的 Yes/No logits 做二选一 softmax，`freeret` 使用 A/B。
生成式评分为目标文字的平均 token 对数概率。IR 图搜文与 it2t 保留 `modify_single` 和
`--use_rerank_prior_cache --rerank_prior_cache_dir <path>`。这里的 `freeret` 是重排模板名称。

未知模板、listwise、CIR 生成式重排、依赖 LLaVA BOS 的 `caption_generation_6/7`，以及
缺少整体稠密表示定义的 `CIR classify_type + concat` 会明确报错。输出聚类、稀疏 x5、
fusion sparse_type、few-shot/自动生成提示词、计时估计不在此次基础适配范围。
`--lora` 可合并已有兼容 adapter。

## 产物与验证

Qwen2 使用独立版本 `qwen2-prompteol-v1` 与 `qwen2_prompt_manifest.json`，与 Qwen2.5、Qwen3
清单分别校验。记录原生模板、词表、实际图像 processor/size/patch/merge、模型与编码设置。
以前的 Qwen2 产物需要重新建库；此次扩展保持 Qwen2.5、Qwen3 既有指纹和格式兼容。
默认允许原目录重编码并清理对应旧分片；成功前不保留旧完成清单。
旧 `index/` 和 `collection/index/` 都会备份到叶目录的 `index.stale_<唯一后缀>/`，编码后需重建。
新索引位于 `collection/index/`；旧同级索引仍兼容，详见 [索引路径说明](qwen_search_scripts.md)。

编码保存 `corpus_<rank>.pkl`、`collection/corpus_<rank>.jsonl`；检索输出 dense/sparse/hybrid TREC，
重排另输出 rerank TREC，Recall@K 写入 `recall_metrics.json` 和 `best.csv`，自动选优另有 `alpha_sweep.csv`。

```bash
python -B -m unittest discover -s tests -p 'test_qwen*_retrieval.py' -v
python -B -m unittest discover -s tests -p 'test_rerank*.py' -v
```

CPU 测试使用真实原生 Qwen2 processor 和随机初始化的小模型，覆盖拼接/独立等价、MRoPE、
视角隔离、左右 padding、图文混合批次、图像尺寸、尾批次、任务与重排模式矩阵、先验缓存、
索引清单、本地保存后加载，以及编码文件到检索重排的集成链路。集成测试使用 FAISS/Lucene
内存替身并 mock openpyxl 写入，不能替代真实检索服务和完整模型的验收。

本地无 CUDA 和完整权重，未完成 7B GPU、真实 FAISS/Lucene 与完整数据集召回验证。
GPU 验收应先对每项任务及两种方法执行小规模建库、真实索引、检索和重排，核对查询 ID、
分数有限性、实际提示词与 Recall@K，再运行完整数据集。

原生模型和处理器参考：[Transformers Qwen2-VL](https://huggingface.co/docs/transformers/v4.57.3/en/model_doc/qwen2_vl)。
