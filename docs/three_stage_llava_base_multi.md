# LLaVA base 多模型、多数据集三阶段检索

三个新增脚本独立调用 `src/three_stage_retrieval_pipeline.py`，原单模型脚本保持不变。
需要提前准备每个模型、数据集的候选编码结果和 Lucene 索引。

| 脚本 | 默认数据集 | 默认任务数 |
|---|---|---:|
| `three_stage_llava_base_ir_multi.sh` | `flickr coco` | 4 |
| `three_stage_llava_base_tbpr_multi.sh` | `CUHK-PEDES ICFG-PEDES RSTPReid` | 6 |
| `three_stage_llava_base_fused_multi.sh` | `webqa remuq llava` | 6 |

默认 `BASE_MODELS` 包含两个模型：
`llava-hf-llava-v1.6-mistral-7b-hf` 和 `llava-hf-llama3-llava-next-8b-hf`。
先遍历一个模型的全部数据集，再处理下一个模型，所有任务串行运行。

```bash
BASE_MODELS="llava-hf-llava-v1.6-mistral-7b-hf llava-hf-llama3-llava-next-8b-hf" \
DATASETS="CUHK-PEDES ICFG-PEDES RSTPReid" \
CHECKPOINT_ROOT=/root/autodl-fs/checkpoints \
STAGE1_DEPTH=200 STAGE2_DEPTH=200 RERANK_NUM=20 \
bash scripts/three_stage_llava_base_tbpr_multi.sh
```

非空 `BASE_MODEL` 优先于 `BASE_MODELS`；非空 `DATASET` 优先于 `DATASETS`。
如环境中已有这些单值变量，可先 `unset BASE_MODEL DATASET` 再执行批次。
IR 通过 `QUERY_TYPE=image|text` 选择本次运行的检索方向，默认 `image`。
保留原配置默认值：base、4 GPU、float16、稀疏长度 40、`caption_generation` 重排前 5 项。

## 路径与覆盖

模型默认从 `CHECKPOINT_ROOT/<模型名称>` 加载；每个 ranker 默认使用当前检索模型。
在新脚本配置区编辑 `MODEL_PATHS`、`RANKER_MODEL_PATHS`、`ARTIFACT_MODEL_NAMES` 可以分别覆盖：

```bash
declare -A MODEL_PATHS=([my_llava]="/models/checkpoint with spaces")
declare -A RANKER_MODEL_PATHS=([my_llava]="/models/ranker")
declare -A ARTIFACT_MODEL_NAMES=([my_llava]="encoded_model_directory")
```

然后用 `BASE_MODELS="my_llava ..."` 选择对应名称。表项可独立省略。
`MODEL_PATH`、`RETRIEVAL_MODEL_PATH`、`RANKER_MODEL_PATH` 和 `ARTIFACT_MODEL_NAME` 的全局覆盖仅用于单模型。

各组合的 `DENSE_INDEXES`、`SPARSE_INDEXES`、`STAGE1_INDEXES`，以及验证集的
`VAL_DENSE_INDEXES`、`VAL_SPARSE_INDEXES`、`VALIDATION_DATASETS`，均以 `模型名称|数据集` 为键：

```bash
declare -A DENSE_INDEXES=(["my_llava|CUHK-PEDES"]="/data/dense/leaf")
declare -A SPARSE_INDEXES=(["my_llava|CUHK-PEDES"]="/data/sparse/leaf/collection")
```

包含图像的候选表征路径添加 `_llava_base`；纯文本候选沿用原路径。
稀疏输入支持 leaf、collection、index 和 collection/index，优先使用 collection/index。
批次输出由 `LLAVA_OUTPUT_ROOT` 自动区分模型、数据集、方向、alpha 和分辨率，
深度目录前新增 `<modify_type>_prior_<缓存状态>`，末级为 `stage1_<STAGE1_DEPTH>_stage2_<STAGE2_DEPTH>`。
例如 `modify_single_prior_false/stage1_200_stage2_200`、`no_prior_false/stage1_200_stage2_200`；
开启缓存的校正任务使用 `modify_single_prior_true`。自动与手动模式均添加标记，精确 `OUTPUT_DIR` 不变。
共享 `STAGE*_INDEX`、验证集路径和精确 `OUTPUT_DIR` 只用于一个模型与一个数据集的组合。

## Alpha、prior 与失败处理

`SELECT_HYBRID_ALPHA=false` 使用固定 `HYBRID_ALPHA`（默认 0.5），忽略验证集配置。
设为 `true` 后，有成对验证集索引则在验证集选择；未提供时在目标 split 选择，
继续由 Python 保存各 alpha 与 best 结果到 CSV。

`AUTO_MODIFY_BY_QUERY_TYPE=true` 默认开启，按每个任务最终的查询类型自动配置：

| 任务 | query_type | modify_type |
|---|---|---|
| IR 图搜文 | `image` | `modify_single` |
| IR 文搜图 | `text` | `no` |
| TBPR、WebQA | `text` | `no` |
| ReMuQ、LLaVA | `image` / `composed` | `modify_single` |

自动模式优先于全局 `MODIFY_TYPE`。每个任务仅运行自动选择的一种模式，批次仍为模型 × 数据集。
IR 每次通过 `QUERY_TYPE` 选择一个方向；不增加双向循环。
缓存默认关闭；自动 `no` 任务始终关闭缓存，`modify_single` 任务使用全局 `USE_RERANK_PRIOR_CACHE`。
因此混合模态批次可以同时处理 WebQA 的 `no` 和 ReMuQ/LLaVA 的 `modify_single`：

```bash
DATASETS="webqa remuq llava" USE_RERANK_PRIOR_CACHE=true \
bash scripts/three_stage_llava_base_fused_multi.sh
```

此时 WebQA 为 `no_prior_false`，ReMuQ/LLaVA 为 `modify_single_prior_true`。
每个任务都会打印 query_type、自动开关、实际 modify_type 和缓存状态。

设置 `AUTO_MODIFY_BY_QUERY_TYPE=false` 可恢复手动配置，使用全局 `MODIFY_TYPE`（默认 `no`）。
手动校正仍只支持 IR 图搜文、ReMuQ、LLaVA；TBPR/WebQA 不支持。校正需要 caption-generation 模板，
prior 缓存需要 `modify_single`。未校正配置示例：

```bash
AUTO_MODIFY_BY_QUERY_TYPE=false MODIFY_TYPE=no USE_RERANK_PRIOR_CACHE=false \
bash scripts/three_stage_llava_base_ir_multi.sh
```

全部组合在第一次启动 GPU 前完成文件、索引和输出检查。重复模型、数据集、表征目录名称被拒绝；
已有结果目录需要 `OVERWRITE=true`。任何运行任务失败都会立即停止，并返回该任务的退出码。
