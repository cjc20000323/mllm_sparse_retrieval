# Qwen2.5-VL：PromptEOL 编码、检索与重排

`encode.py`、`search.py`、`rerank.py` 会读取本地检查点的 `config.json`，把
`model_type=qwen2_5_vl` 的模型送入共享适配组件。目录不必包含 3B/7B 等名称。
其他模型继续使用原有路径。独立的 `freeret_retrieval.sh` 不在此次适配范围。
三阶段正式流程与耗时估计已复用共享组件，见 [三阶段 Qwen 说明](three_stage_qwen.md)。

共享实现统一位于 `src/qwen_vl_prompts.py`、`src/qwen_vl_encoding.py` 和
`src/qwen_vl_pipeline.py`，同时供 Qwen2、Qwen2.5 与 Qwen3 使用。代码统一通过 `qwen_vl_*`
导入，旧模块路径已移除。本次清理不改变索引格式或指纹，无需重建索引。

## 方法与提示词

| 配置 | 稠密表示 | 稀疏表示 |
| --- | --- | --- |
| `prompteol + separate` | 整体概括视角 | 整体概括视角的词表分布 |
| `disassembleeol_separate + concat` | 整体概括视角 | 各分解视角的词表分布 |

英文正文、视角顺序和输入身份以当前 LLaVA-Mistral 模板为基准。Qwen 使用检查点的
`apply_chat_template`，不使用旧适配中的 `<tool_call>` 分隔符，也不添加新 token。
保留 `log(1+relu)`、L2 归一化、整数稀疏权重及 `replace/sum/max` 聚合。

拼接编码仅执行一次模型前向：每条样本单独定位视角，所有视角共享图像/原始文本上下文，
互相屏蔽各自指令。先用二维 padding mask 计算原生 MRoPE，再传四维隔离 mask；
各文本视角重置到共享上下文之后的位置。因此不同视角不会泄漏，图像位置编码仍保持原生定义。
该路径使用 SDPA，不支持忽略隔离 mask 的 FlashAttention 运行方式。

共享编码层按实际接口签名兼容不同 Transformers 版本的 `get_rope_index()`。
新版要求的 `mm_token_type_ids` 优先沿用 processor 的输出；缺失时，根据展开、padding
后的 token ID 补齐文本（0）、图像（1）和视频（2）标记，并按接口能力传入模型前向。
旧版模型不会收到不支持的模态标记参数。此兼容逻辑同时覆盖 Qwen2、Qwen2.5、Qwen3
及 encode、search、rerank、三阶段流程，无需新增启动参数。

Qwen 支持图文按内容列表顺序交错排列，没有“必须先图像”的格式限制。此次迁移保留原有
模板顺序，包括 `reverse_*` 的原有语义；某些 Mistral 模板本身的 normal/reverse 顺序相同，
本次没有重定义这些实验。格式可用不代表不同顺序的检索效果必然相同。

## 完整命令示例：Flickr 图搜文

在已有 GPU 环境运行，需要该项目原有的依赖：PyTorch、Transformers、Accelerate、
PEFT、NLTK 数据、FAISS、Pyserini/Java、pandas 和 openpyxl。本地验证版本为
PyTorch 2.9.1 / Transformers 4.57.3。普通图片路径和 Parquet 图片字节由 PIL 与原生
processor 处理，无须为图片额外安装 `qwen-vl-utils`。数据位置沿用 `arguments.py`。

以下是 Bash 命令。选择一套方法，在空输出根目录中重新建库。`bf16` 可按 GPU 能力换成 `fp16`。

```bash
MODEL=/root/autodl-fs/checkpoints/Qwen2.5-VL-7B-Instruct
MODEL_NAME=$(basename "$MODEL")
GPUS=1

# 方法一；使用方法二时替换下方两行。
EOL=prompteol
CALC=separate
# EOL=disassembleeol_separate
# CALC=concat

DENSE_ROOT=./qwen25_native_dense
SPARSE_ROOT=./qwen25_native_sparse
COMMON=(--model_name_or_path "$MODEL" --task_type ir
  --dataset_name flickr --dataset_split test
  --eol_type "$EOL" --calculate_type "$CALC" --prompt_type prompt_5
  --per_device_batch_size 1 --reps_loc after_pad --bf16
  --sparse_length 40 --sparse_value_type sum --sparse_lower_or_upper lower
  --num_expended_tokens 0)

# 1. 图搜文：候选库为 text。在线查询不需要提前保存 query.pkl。
deepspeed --num_gpus "$GPUS" src/encode.py "${COMMON[@]}" \
  --encode_type text --output_dir ./qwen_encode_logs \
  --dense_output_dir "$DENSE_ROOT" --sparse_output_dir "$SPARSE_ROOT"

LEAF="0_no_manual_40_sum_no_cluster_after_pad_${EOL}_lower_no_mean"
RELATIVE="$MODEL_NAME/flickr/text/no_filter/$CALC/prompt_5/test/$LEAF"
DENSE="$DENSE_ROOT/$RELATIVE"
SPARSE="$SPARSE_ROOT/$RELATIVE"

# 2. 仅 collection/ 中的 JSONL 是 Lucene 文档；不要把版本清单作为文档读入。
python -B src/qwen_sparse_index.py --sparse-root "$SPARSE" --threads 16

# 3. 固定 alpha 的在线检索；不会在测试集上选择 alpha。
SEARCH=("${COMMON[@]}" --query_type image
  --passage_reps "$DENSE" --sparse_index "$SPARSE"
  --use_fixed_alpha --alpha 0.5 --depth 1000 --threads 16)
deepspeed --num_gpus "$GPUS" src/search.py "${SEARCH[@]}" \
  --output_dir ./qwen25_search_results

# 4. 同一套候选库和查询编码，随后逐条重排。
deepspeed --num_gpus "$GPUS" src/rerank.py "${SEARCH[@]}" \
  --rerank_type pointwise --rerank_template what_caption_generation \
  --rerank_num 20 --rerank_batch_size 2 --modify_type no \
  --output_dir ./qwen25_rerank_results
```

可同时增加 `--qwen_min_pixels` / `--qwen_max_pixels` 控制图像分辨率，单位是像素数；
0 表示检查点默认值。例如 `--qwen_max_pixels 802816` 对应最多约 1024 个视觉 token。
编码与检索必须使用相同设置。拼接 mask 的内存随序列长度平方增长，建议先从 batch size 1 验证。

文搜图时把建库的 `--encode_type` 和索引路径中的 `text` 改为 `image`，把检索的
`--query_type image` 改为 `text`。如果启用 `--sparse_manual`，叶目录的长度字段变成
`text_sparse_length_image_sparse_length`；实际产物路径也会由编码程序打印。

## 其他任务

将上面命令的任务/数据集/方向同时替换，并重新建库：

| 数据集 | task_type | 候选库 encode_type | query_type | 补充参数 |
| --- | --- | --- | --- | --- |
| COCO/Flickr | `ir` | `text` 或 `image` | `image` 或 `text` | 双向分别建库 |
| CUHK-PEDES / ICFG-PEDES / RSTPReid | `tbpr` | `image` | `text` | `--tbpr_type origin_type` 或 `type` |
| WebQA | `t2it` | `corpus` | `text` | 候选图文联合编码 |
| ReMuQ / LLaVA | `it2t` | `corpus` | `image` 或 `composed` | 查询图文联合编码 |
| Fashion-IQ | `cir` | `image` | `composed` | `--cir_type type` 或 `type_1`，检索方向使用 `--remove_query` 排除参考图 |

行人/CIR 的索引目录在 split 后还包含 `tbpr_type` / `cir_type` 一层。
Fashion-IQ 使用数据集已有的 `val` 标注，应同步设置 `--dataset_split val`。

要按验证集选择 alpha，先用一致配置对相应验证集建库，然后移除 `--use_fixed_alpha`，
提供 `--val_passage_reps` 和 `--val_sparse_index`。支持 `--validation_dataset_name` 和
`--use_fixed_validation_dataset`；后者分别使用 Flickr、RSTPReid、WebQA、LLaVA 作同任务验证。
生成式验证数据开关只作用于验证集；测试数据不会被替换。
非固定模式下，Python 的两个 val 参数均未提供时，直接复用目标集分数扫描 alpha；
独立 `search_qwen.sh` 未显式指定 val 路径时始终使用目标集，显式错误 val 路径仍报错。
多卡只由 rank 0 选优并复用最佳结果，九组指标写入 `alpha_sweep.csv`，最佳目标集指标写入 `best.csv`。

## 重排支持矩阵

| 任务 | 有效模板 |
| --- | --- |
| 普通图文 | `relevant`、`old_relevant`、`query_relevant`、对应 `reverse_*`、`origin_old_relevant`、`please_relevant`、`in_one_word_relevant`、`precise_caption`、`first_precise_caption`、`role_*`、`freeret` |
| 行人 | 相关性、query、origin/old、reverse、please、precise、role、`freeret` |
| WebQA | `relevant`（Mistral 的 `old_relevant` 别名）、`old_relevant`、`query_relevant`、`origin_old_relevant` |
| ReMuQ/LLaVA | 同 WebQA，另有 `answer_relevant` |
| Fashion-IQ | `relevant`、`old_relevant`、`query_relevant`、`origin_old_relevant`、`please_relevant`、`in_one_word_relevant`、`role_relevant`、`role_old_relevant` |

普通图文生成式模板支持 `caption_generation`、`what_caption_generation`、
`caption_generation_2/3/4/5`、`detailed_caption_generation`、`detailed_caption_generation_1`。
行人支持 `caption_generation`、`what_caption_generation`、`describe_caption_generation`。
WebQA、ReMuQ/LLaVA 支持前两种标准生成式模板。生成式评分仍是目标文本的平均 token 对数概率。

`modify_single` 与先验缓存保留既有适用范围：IR 图搜文、it2t。
缓存需加 `--use_rerank_prior_cache --rerank_prior_cache_dir <path>`。
相关性评分使用 Qwen tokenizer 的 Yes/No；`freeret` 使用 A/B。
这里的 `freeret` 是本方法重排模板，不是独立 FreeRet 检索流水线。

下列选项会明确报错：未知模板、listwise、CIR 生成式重排，以及依赖 LLaVA `<s>` 的
`caption_generation_6/7`。CIR `classify_type + concat` 的旧实现没有定义此次方法所需的
整体稠密表示，因此不做静默替代；采用 `type` / `type_1`。
LLaVA 特定的 x5、输出聚类、fusion 稀疏类型、few-shot/自动生成提示词实验，以及其计时估算入口，
也不属于本次两个基础方法的适配范围。`--lora` 可加载并合并已有 Qwen adapter，不涉及训练迁移。

## 产物与验收

- 编码保存 `corpus_<rank>.pkl`、`collection/corpus_<rank>.jsonl` 和版本清单。
- 清单包含提示词、模型/LoRA 身份、方法、数据集 split、稀疏配置、分辨率及原生输入示例；查询核对清单，拒绝旧索引或配置混用。
- 默认允许原目录重新编码：清理对应旧分片、撤下旧完成清单，成功后写入新清单。
  语料重编码将旧 `index/` 和 `collection/index/` 都备份到叶目录的 `index.stale_<唯一后缀>/`；编码后需重建。
  新索引位于 `collection/index/`，旧同级索引仍兼容，详见 [索引路径说明](qwen_search_scripts.md)。
- 检索输出 dense/sparse/hybrid TREC，重排另输出 rerank TREC；Recall@K 写入 `recall_metrics.json` 和 `best.csv`，自动选优另有 `alpha_sweep.csv`。
- 不同方法、方向及模板应使用不同的结果目录。

CPU 测试使用真实 Qwen processor 和随机初始化的小型 Qwen，验证拼接/独立前向等价、视角隔离、
padding/尺寸/尾批次、任务模板、先验缓存以及编码文件到检索重排的集成链路。集成测试用内存
检索替身代替 Java/FAISS，并 mock 环境缺失的 openpyxl 写入，不能替代真实检索服务和完整权重验证。

```bash
python -m unittest discover -s tests -p 'test_qwen25_retrieval.py' -v
python -m unittest discover -s tests -p 'test_rerank*.py' -v
```

GPU 验收时，对每个任务和两种方法选择小规模数据，执行建库、真实 Lucene 索引、检索和重排，
核对全部查询 ID、有限分数、Recall@K 与清单中的实际提示词。随后再跑完整验证/测试集。
本地无 CUDA 与完整模型权重，未声称完成这一阶段或获得可用于论文比较的召回率。
