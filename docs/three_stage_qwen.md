# 三阶段检索中的 Qwen2、Qwen2.5 与 Qwen3-VL

正式三阶段入口和独立耗时估计入口均支持三代 Qwen 的 Instruct 检查点。
按原生配置识别模型，支持改名后的本地目录；Qwen3 范围为 Dense Instruct。
检索与重排可以使用不同模型。现有 LLaVA 默认配置、数据集范围、预计算 run 和计时规则保持不变。

## 计算与提示词

- 第一阶段 `sparse_concat` 使用 `disassembleeol_separate + concat`，每个 rank 只编码自己的查询。
  单条样本的图像与原始文本只输入一次，各视角通过因果 mask 独立读取共享上下文。
- 第一阶段同时产生 CPU dense 表示与 sparse topics。第二阶段复用这些表示，不重新加载查询模型，
  不重复前向；第二阶段查询编码时间记为 0。
- 第一阶段 `sparse_manual` 使用 `prompteol + separate + sparse_manual`；第二阶段仍固定 concat，
  两种表示不同，需要分别编码。此模式通过 Python CLI 使用，并分别提供匹配的第一、二阶段索引。
- 第三阶段使用原生 Qwen 模型和现有 `Reranker`，支持相关性模式和 `what_caption_generation` 等生成式模式。
  检索权重在加载重排权重前释放；重排本身仍按候选评分，不额外承诺跨候选视觉特征缓存。

复用 `qwen_vl_*` 的英文正文、视角列表和 LLaVA-Mistral normal/reverse 身份及图文顺序，
使用各检查点的原生 chat template。保留 Qwen MRoPE 和 Qwen3 DeepStack。
IR 图片查询不输入参考 caption；ReMuQ/LLaVA 保留查询图文；Fashion-IQ 保留修改文本与组合语义。

## 像素设置和索引校验

| Python CLI 参数 | 启动脚本环境变量 | 用途 |
| --- | --- | --- |
| `--retrieval-qwen-min-pixels` | `RETRIEVAL_QWEN_MIN_PIXELS` | 查询图像最小像素面积 |
| `--retrieval-qwen-max-pixels` | `RETRIEVAL_QWEN_MAX_PIXELS` | 查询图像最大像素面积 |
| `--ranker-qwen-min-pixels` | `RANKER_QWEN_MIN_PIXELS` | 重排图像最小像素面积 |
| `--ranker-qwen-max-pixels` | `RANKER_QWEN_MAX_PIXELS` | 重排图像最大像素面积 |

四个参数默认均为 0，使用对应检查点的默认值；CLI 参数可覆盖环境设置。
检索的像素设置必须与 `encode.py` 建库时的 `--qwen_min_pixels/--qwen_max_pixels` 一致。
重排像素限制独立设置，不改变已建候选索引。

Qwen 索引必须带有对应的 `qwen2_prompt_manifest.json`、`qwen25_prompt_manifest.json` 或
`qwen3_prompt_manifest.json`。第一、二阶段分别核对模型/LoRA、数据集、split、方法、提示词、
词表、图像处理和稀疏配置；配置元数据随查询表示保留，因此第二阶段复用分支也会校验。
有效的现有 Qwen 产物可以直接使用，无需为三阶段入口重新编码。没有清单的旧产物需要重建。
如果主动在原目录重新运行 Qwen encode，dense/sparse 编码文件会被替换，旧 `index/` 和
`collection/index/` 都会备份到叶目录下的 `index.stale_<唯一后缀>/`（在 collection 外）。
需先重新执行 sparse_index，再启动三阶段；编码中断时没有有效完成清单。
新索引默认位于 `collection/index/`；根目录及 collection 入口优先读取它，仅在该路径不存在时
回退旧同级 `index/`。显式传入索引目录时直接读取它，清单始终从编码叶目录读取。

第一阶段和第二阶段的候选限制、NumPy 稠密检索、Lucene impact 稀疏检索、min-max 归一化与
alpha 融合规则沿用现有实现。加载、校验、输出和资源清理不计入核心计算时间。

## Flickr 图搜文完整示例

以下 Bash 命令在已有 GPU、DeepSpeed、Transformers、Pyserini/Java 和 pandas/openpyxl 环境运行。
数据位置沿用项目设置，`DATA_ROOT` 可为三阶段流程指定数据根目录。
先选择模型，再用同一检索配置建库。也可直接使用符合这些配置的已有索引。

```bash
MODEL=/root/autodl-fs/checkpoints/Qwen3-VL-8B-Instruct
# MODEL=/root/autodl-fs/checkpoints/Qwen2.5-VL-7B-Instruct
# MODEL=/root/autodl-fs/checkpoints/Qwen2-VL-7B-Instruct
MODEL_NAME=$(basename "$MODEL")
MIN_PIXELS=65536
MAX_PIXELS=1048576

deepspeed --num_gpus 1 src/encode.py \
  --model_name_or_path "$MODEL" --task_type ir --encode_type text \
  --dataset_name flickr --dataset_split test \
  --eol_type disassembleeol_separate --calculate_type concat --prompt_type prompt_5 \
  --per_device_batch_size 1 --reps_loc after_pad --bf16 \
  --sparse_length 40 --sparse_value_type sum --sparse_lower_or_upper lower \
  --qwen_min_pixels "$MIN_PIXELS" --qwen_max_pixels "$MAX_PIXELS" \
  --dense_output_dir ./dense_output --sparse_output_dir ./sparse_output \
  --output_dir ./qwen_encode_logs

LEAF=0_no_manual_40_sum_no_cluster_after_pad_disassembleeol_separate_lower_no_mean
RELATIVE="$MODEL_NAME/flickr/text/no_filter/concat/prompt_5/test/$LEAF"
DENSE="./dense_output/$RELATIVE"
SPARSE="./sparse_output/$RELATIVE"

python -B src/qwen_sparse_index.py --sparse-root "$SPARSE" --threads 16

export BASE_MODEL="$MODEL_NAME"
export RETRIEVAL_MODEL_PATH="$MODEL" RANKER_MODEL_PATH="$MODEL"
export STAGE1_SPARSE_INDEX="$SPARSE" STAGE2_DENSE_INDEX="$DENSE" STAGE2_SPARSE_INDEX="$SPARSE"
export RETRIEVAL_QWEN_MIN_PIXELS="$MIN_PIXELS" RETRIEVAL_QWEN_MAX_PIXELS="$MAX_PIXELS"
export RANKER_QWEN_MIN_PIXELS="$MIN_PIXELS" RANKER_QWEN_MAX_PIXELS="$MAX_PIXELS"
export RETRIEVAL_DTYPE=bfloat16 RANKER_DTYPE=bfloat16
export RERANK_TEMPLATE=what_caption_generation

# 正式执行，输出三阶段 run、Recall 和计算耗时。
GPUS=1 DATASET=flickr OUTPUT_DIR=./qwen_three_stage \
  bash scripts/three_stage_retrieval_pipeline.sh

# 单独估计耗时，不输出部分 run 或 Recall。
NUM_PROCESSES=1 DATASET=flickr OUTPUT_DIR=./qwen_three_stage_estimate \
  bash scripts/estimate_three_stage_retrieval_time.sh
```

按 GPU 能力可将建库和运行的精度一起改为 FP16。concat mask 和完整词表 logits 都会占用显存，
应从 batch size 1 开始检查实际占用；减少重复图像编码不等于固定加速比。

文搜图需要用 `--encode_type image` 建库，并设置 `QUERY_TYPE=text`、对应的索引路径。
行人使用 `scripts/three_stage_retrieval_pipeline_tbpr.sh`，WebQA/ReMuQ/LLaVA 使用
`scripts/three_stage_retrieval_pipeline_multimodal.sh`；四个脚本都接收上表的像素设置。
这些启动脚本的数据集范围保持原样。Fashion-IQ 继续通过 Python CLI 使用 `--task-type cir
--query-type composed --dataset-split val`，并提供对应图像库索引。

相关性、reverse/role/precise、A/B 和生成式模式的支持矩阵与各代 Qwen 独立入口一致。
`modify_single` 与先验缓存仅保留既有 IR 图搜文、it2t 适用范围。Qwen3 Thinking/MoE、listwise、
CIR 生成式重排、`caption_generation_6/7` 和稀疏 x5 等不支持的实验会明确报错。

## 验证与计时边界

```bash
python -B -m unittest discover -s tests -p 'test_three_stage*.py' -v
python -B -m unittest discover -s tests -p 'test_qwen*_retrieval.py' -v
python -B -m unittest discover -s tests -p 'test_rerank*.py' -v
```

新增 CPU 测试使用三代真实原生 processor 与随机小模型，读取独立编码入口产生的实际产物，
执行 NumPy 受限检索和原生重排。验证两种编码方法、提示词一致性、IR caption 隔离、CIR 组合、
native ID、先验缓存、清单拒绝、查询复用前向次数、模型释放和耗时估计。Lucene 使用内存替身，
不能替代真实 Java 索引验收。

耗时估计维持原有规则：完整执行第一、二阶段，仅在第三阶段稳定后提前停止；缓存准备与模型加载
不进入观察器，复用时第二阶段查询编码时间为零。详细规则见 [耗时估计说明](three_stage_time_estimation.md)。

本地没有 CUDA 与完整权重，尚未完成真实 GPU、Lucene 和完整数据集召回/耗时验收。
应先对每个模型及任务执行小规模真实索引验证，再运行完整数据集。
