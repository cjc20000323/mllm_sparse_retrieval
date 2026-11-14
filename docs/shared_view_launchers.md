# prompt_shared_5 的 IR / TBPR / fused 配套脚本

九个入口均默认使用 `prompt_shared_5`，只支持 `llava_mistral` 和 `llava_llama`：

| 阶段 | IR | TBPR | fused |
|---|---|---|---|
| 编码候选库 | `scripts/encode_ir.sh` | `scripts/encode_tbpr.sh` | `scripts/encode_fused.sh` |
| 建立稀疏索引 | `scripts/sparse_index_ir.sh` | `scripts/sparse_index_tbpr.sh` | `scripts/sparse_index_fused.sh` |
| 三阶段检索 | `scripts/three_stage_retrieval_pipeline_ir.sh` | `scripts/three_stage_retrieval_pipeline_tbpr.sh` | `scripts/three_stage_retrieval_pipeline_fused.sh` |

TBPR 默认 CUHK-PEDES、`TBPR_TYPE=type`；fused 默认 WebQA。每次默认只运行一个模型和一个数据集。
IR 默认 Flickr，并依次运行图搜文和文搜图；可选择 COCO，或设置 `DATASETS="flickr coco"` 批量运行。
默认编码设置为 `disassembleeol_separate + concat`、长度 40、sum、lower、after_pad、关闭过滤和均值。
默认使用 1 GPU、float16、batch size 1；建库使用 CPU，线程数 16。FreeRet 相关开关默认关闭。

## 配置和顺序

在各 `.sh` 文件顶部编辑变量，或用同名环境变量覆盖。三个阶段需使用相同的模型路径、
`PROMPT_TYPE`、数据集、split、`TBPR_TYPE` 和编码参数。

```bash
bash scripts/encode_ir.sh
bash scripts/sparse_index_ir.sh
bash scripts/three_stage_retrieval_pipeline_ir.sh

bash scripts/encode_tbpr.sh
bash scripts/sparse_index_tbpr.sh
bash scripts/three_stage_retrieval_pipeline_tbpr.sh

bash scripts/encode_fused.sh
bash scripts/sparse_index_fused.sh
bash scripts/three_stage_retrieval_pipeline_fused.sh
```

首先设置本机的 `CHECKPOINT_ROOT`、`DATA_ROOT`、`DENSE_OUTPUT_ROOT`、`SPARSE_OUTPUT_ROOT`。
默认检查点根目录为 `/root/autodl-fs/checkpoints`，数据根目录为 `/root/autodl-tmp`，
表征输出为当前项目的 `dense_output` / `sparse_output`。
数据目录结构为 `DATA_ROOT/data/<数据集目录>/...`。

编码通过轻量启动入口设置数据根目录，再调用原 `encode.py` 的 `main()`；
不改变原有编码算法，也不修改 `encode_adaptive_views.py`。
这里只编码候选：TBPR 为 `image`，fused 为 `corpus`；查询由三阶段入口在线编码。
IR 图搜文（`QUERY_TYPE=image`）的候选为 `text`；文搜图（`QUERY_TYPE=text`）的候选为 `image`。
两个方向的候选库按模态分别保存，结果按 `image_to_text` / `text_to_image` 分目录，IR 不加入 `TBPR_TYPE` 路径。
当 `LLAVA_IMAGE_MODE=base` 时，只有图像候选库增加 `_llava_base` 后缀，文本候选库不增加该后缀。
WebQA 使用 t2it / text 查询，ReMuQ 和 LLaVA 使用 it2t / image 查询（包含图像和文本）；
后两者也可显式设置 `QUERY_TYPE=composed`。

## 批量运行

支持两个模型族；IR 支持 Flickr/COCO，TBPR 和 fused 各支持三个数据集。非空的 `MODEL_FAMILY` / `DATASET` 单值优先于复数列表。
要使用列表，保留单值变量为空：

```bash
MODEL_FAMILIES="llava_mistral llava_llama" \
DATASETS="flickr coco" \
bash scripts/encode_ir.sh

MODEL_FAMILIES="llava_mistral llava_llama" \
DATASETS="CUHK-PEDES ICFG-PEDES RSTPReid" \
bash scripts/encode_tbpr.sh

MODEL_FAMILIES="llava_mistral llava_llama" \
DATASETS="webqa remuq llava" \
bash scripts/sparse_index_fused.sh
```

同样的配置适用于另外两个阶段。批量按列表顺序串行执行，运行前检查全部组合的必要输入。
某一工作进程失败时立即停止，并保留已完成产物。

单模型可设置 `MODEL_PATH`；多模型分别使用 `MODEL_PATH_LLAVA_MISTRAL`、`MODEL_PATH_LLAVA_LLAMA`。
自定义检查点沿用标准编码入口原有的模型识别规则。产物的模型目录名取实际检查点目录名。

IR 的三个脚本都默认设置 `QUERY_TYPES="image text"`，依次运行图搜文、文搜图。
非空的 `QUERY_TYPE` 覆盖整个方向列表。只跑 COCO 文搜图时，三个阶段统一设置：

```bash
export DATASET=coco
export QUERY_TYPE=text
bash scripts/encode_ir.sh
bash scripts/sparse_index_ir.sh
bash scripts/three_stage_retrieval_pipeline_ir.sh
```

恢复双向时使用 `unset QUERY_TYPE`；也可通过 `QUERY_TYPES="text image"` 调整顺序。
IR 启动前检查 `DATA_ROOT/data/flickr/flickr_<split>.csv` 或 `DATA_ROOT/data/coco/coco_<split>.csv`。
检查对象由实际 `DATASET_SPLIT` 决定，所有组合检查通过后才启动第一个工作进程。

## 查看命令和指定已有产物

```bash
DRY_RUN=true bash scripts/encode_ir.sh
DATASET=coco bash scripts/three_stage_retrieval_pipeline_ir.sh --dry-run
DRY_RUN=true bash scripts/encode_tbpr.sh
bash scripts/three_stage_retrieval_pipeline_fused.sh --dry-run
bash scripts/sparse_index_tbpr.sh --help
```

Dry-run 不加载模型、不检查数据是否已经存在、不启动工作进程，也不创建输出。
实际建库会跳过已有 `segments_*` 的索引；存在但不完整的索引会报错，不会覆盖。
索引只读取 `corpus_*.jsonl`，目标为候选稀疏目录的 `index/`。

单组合建库可设置 `SPARSE_CORPUS_DIR`。单组合三阶段可设置 `DENSE_CORPUS_DIR`、`SPARSE_CORPUS_DIR`，
或 `STAGE1_SPARSE_INDEX`、`STAGE2_DENSE_INDEX`、`STAGE2_SPARSE_INDEX`。
支持实验目录及相应的 index / collection 路径别名。
编码只接受输出根目录，不接受显式候选或索引目录；日志目录用 `ENCODE_OUTPUT_DIR`。

三阶段保留原生 Python 命令行参数覆盖能力，例如：

```bash
bash scripts/three_stage_retrieval_pipeline_tbpr.sh \
  --dataset-name ICFG-PEDES --tbpr-type type --prompt-type prompt_shared_5 \
  --stage1-depth 1000 --stage2-depth 100 --output-dir ./my_results
```

参与目录计算的命令行参数会先应用，再推导路径；显式 `OUTPUT_DIR` 保持指定值。
显式候选、索引、验证集或结果路径只允许单个模型/数据集/方向组合；IR 默认双向时需先设置 `QUERY_TYPE`。
新模式默认结果目录带 `prompt_shared_5`，避免与旧模式结果混用。
显式改回 `PROMPT_TYPE=prompt_5` 时沿用旧提示词，并保留原 TBPR 结果目录布局。

三阶段默认 `sparse_concat -> hybrid -> pointwise rerank`，深度均为 1000、alpha 为 0.5，
重排模板为原有 `caption_generation`、重排数为 5。可在配置区调整；原重排提示词不变。

## 自动配置重排修正

IR / fused / TBPR 三个标准三阶段入口默认使用 `MODIFY_TYPE=auto` 和 `USE_RERANK_PRIOR_CACHE=auto`。
启动器根据每个任务的最终查询方向和重排模板分别解析，不会把某个方向的值沿用到下一个方向。

| 任务与方向 | caption generation 模板下的 modify_type | prior cache |
|---|---|---|
| IR 图搜文（`QUERY_TYPE=image`） | `modify_single` | `true` |
| IR 文搜图（`QUERY_TYPE=text`） | `no` | `false` |
| ReMuQ / LLaVA 图文搜文（`image` 或 `composed`） | `modify_single` | `true` |
| WebQA 文搜图文 | `no` | `false` |
| TBPR 文搜行人图像 | `no` | `false` |

只有模板名称包含 `caption_generation` 时才自动启用修正，例如 `caption_generation`、`what_caption_generation`；
使用 `relevant` 等其他模板时，自动值为 `no`，缓存自动关闭。
命令预览会显示每个任务实际传入的 `--modify_type`，以及是否包含 `--use-rerank-prior-cache`。

`MODIFY_TYPE=no` 或 `MODIFY_TYPE=modify_single` 可以手动覆盖自动选择。
若需恢复本次更改前的默认评分行为，设置 `MODIFY_TYPE=no`：

```bash
MODIFY_TYPE=no bash scripts/three_stage_retrieval_pipeline_ir.sh
DATASET=remuq MODIFY_TYPE=modify_single bash scripts/three_stage_retrieval_pipeline_fused.sh
```

原生命令行的 `--modify_type` / `--modify-type` 手动值优先于环境变量。
命令行指定的重排模板、类型和查询方向会先应用，再进行自动解析及缓存校验。
`auto` 仅用于脚本或环境变量配置，底层 Python 入口仍接收原有的 `no` / `modify_single` 和实际缓存布尔开关：

```bash
bash scripts/three_stage_retrieval_pipeline_ir.sh --modify-type no
bash scripts/three_stage_retrieval_pipeline_ir.sh --rerank-template relevant --dry-run
```

自动配置不改变候选库、索引或结果目录，缓存沿用现有实现、目录和评分公式。
`USE_RERANK_PRIOR_CACHE=auto` 根据最终的 `modify_type` 解析：`modify_single` 开启，`no` 关闭。
这也适用于手动指定的 `MODIFY_TYPE` 及命令行覆盖后的值。
因此 IR 双向可以直接运行；fused 混合数据集也会逐任务分别配置，不需要拆开执行。

显式 `USE_RERANK_PRIOR_CACHE=false` 可单独关闭缓存，保留 `modify_single` 的评分修正；
显式 `true` 则要求全部任务启用缓存，继续兼容 `1/0`、`yes/no` 等原有布尔写法。
原生命令行 `--use-rerank-prior-cache` 显式开启优先于环境变量设置：

```bash
bash scripts/three_stage_retrieval_pipeline_ir.sh
DATASETS="webqa remuq llava" bash scripts/three_stage_retrieval_pipeline_fused.sh
USE_RERANK_PRIOR_CACHE=false bash scripts/three_stage_retrieval_pipeline_ir.sh
QUERY_TYPE=image bash scripts/three_stage_retrieval_pipeline_ir.sh --use-rerank-prior-cache
```

缓存仍只支持 IR 图搜文及 it2t 图文搜文，要求 `pointwise + caption_generation + modify_single`。
自动模式若解析到不兼容组合，或显式 `true` 遇到 IR 文搜图、TBPR、WebQA、关闭修正等情况，
会在启动任何工作进程前报错。若要使用不带缓存的现有修正方式，可设置 `USE_RERANK_PRIOR_CACHE=false`。
