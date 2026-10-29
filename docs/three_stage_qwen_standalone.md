# Qwen 三阶段独立启动脚本

`scripts/three_stage_qwen.sh` 直接调用 DeepSpeed 和
`src/three_stage_retrieval_pipeline.py`，不调用其他 Shell 启动器。可从任意目录启动，
在脚本顶部配置参数，或通过环境变量覆盖。已有脚本保持原样。

默认使用 Qwen2.5-VL-7B-Instruct、Flickr 图搜文、test、1 GPU、float16，
固定 dense 权重 `HYBRID_ALPHA=0.5`，sparse 权重为 `1-alpha`。
第一、二阶段固定使用 `disassembleeol_separate+concat`，复用第一阶段的查询编码，
候选深度均为 1000；第三阶段默认使用 `what_caption_generation` 重排前 5 个候选。

## 启动

先准备匹配的候选库 dense 表示和 Lucene 索引；脚本不会执行 encode 或 sparse_index。
查询由三阶段入口在线编码，无需提前生成 query.pkl。

```bash
# 使用默认配置运行一个数据集。
bash scripts/three_stage_qwen.sh

# 使用 Qwen3 顺序运行两个数据集，分别保存结果。
QWEN_FAMILY=qwen3 DATASETS="flickr coco" bash scripts/three_stage_qwen.sh

# Qwen2 文搜图，需要预先编码 image 候选库。
QWEN_FAMILY=qwen2 DATASET=flickr IR_QUERY_TYPE=text bash scripts/three_stage_qwen.sh

# 改为重排前 20 个候选、固定 alpha=0.7。
RERANK_NUM=20 HYBRID_ALPHA=0.7 DATASET=coco bash scripts/three_stage_qwen.sh
```

显式 `DATASET` 优先于 `DATASETS`；未配置时只运行 Flickr。批量数据集用空白分隔，
不允许重复或未知名称。全部任务先经过预检查，再按顺序启动；首个失败停止后续任务并保留退出码。

| 数据集 | 任务 | 查询方向 |
| --- | --- | --- |
| flickr / coco | ir | `IR_QUERY_TYPE=image` 图搜文；`text` 文搜图 |
| CUHK-PEDES / ICFG-PEDES / RSTPReid | tbpr | 文本搜行人图像；`TBPR_TYPE=origin_type` 或 `type` |
| webqa | t2it | 文本搜图文候选 |
| remuq / llava | it2t | 图文查询搜文本候选 |

`QWEN_FAMILY=qwen2|qwen25|qwen3` 默认分别选择 Qwen2-VL-7B-Instruct、
Qwen2.5-VL-7B-Instruct、Qwen3-VL-8B-Instruct，检查点根目录默认
`/root/autodl-fs/checkpoints`。`QWEN_MODEL_PATH` 可指定其他实际路径；目录名称可自定义，
候选库路径使用该目录的 basename。重排模型默认相同，可用 `RANKER_MODEL_PATH` 单独设置。

## 候选库路径与编码参数

`DENSE_OUTPUT_ROOT`、`SPARSE_OUTPUT_ROOT` 默认是项目下 `dense_output/`、`sparse_output/`。
按编码端规则拼接模型目录名、数据集、候选模态、过滤配置、concat、提示词、split 和稀疏参数；
行人任务在 split 后增加 `TBPR_TYPE`。稀疏入口默认以 `collection` 结尾。

解析索引时优先使用 `collection/index`，只有该路径不存在时兼容旧同级 `index`。
新目录存在但缺少 `segments_*` 时直接报错。也可显式指向新旧索引目录，读取所指定的目录。
Qwen 配置清单由 Python 入口从编码叶目录校验。

默认稀疏长度 40、`prompt_5`、随机种子 42、after_pad、sum、lower、非手动、不取均值、不过滤。
这些设置必须与编码一致。这个启动器不支持把第一、二阶段切换为 prompteol/separate 或手动稀疏。
检索表示位置由 `RETRIEVAL_REPS_LOC` 控制，未设时沿用 `REPS_LOC`。

例如服务器上已使用自定义模型目录、长度 30，并把编码产物保存到了 autodl-fs：

```bash
QWEN_MODEL_PATH=/root/autodl-fs/checkpoints/Qwen-Qwen2.5-VL-7B-Instruct \
DENSE_OUTPUT_ROOT=/root/autodl-fs/dense_output \
SPARSE_OUTPUT_ROOT=/root/autodl-fs/sparse_output \
SPARSE_LENGTH=30 DATASET=flickr \
bash scripts/three_stage_qwen.sh
```

单任务可以使用 `STAGE2_DENSE_INDEX`、`STAGE2_SPARSE_INDEX` 显式覆盖；
`STAGE1_SPARSE_INDEX` 默认与第二阶段共用。dense 输入支持目录或单个 corpus*.pkl。
批量任务使用脚本中的路径表，避免把一个数据集的库误用于其他任务：

```bash
declare -A DENSE_INDEXES=([flickr]="/my/flickr/dense" [coco]="/my/coco/dense")
declare -A SPARSE_INDEXES=([flickr]="/my/flickr/sparse/collection" [coco]="/my/coco/sparse/collection")
```

将这些配置填入脚本中已有的对应声明；不要在外部 Shell 导出关联数组。
未填写的任务继续使用根目录派生路径。另有 `STAGE1_INDEXES` 可单独覆盖第一阶段索引。

数据根目录 `DATA_ROOT` 默认 `/root/autodl-tmp`，数据标注位于其 `data/` 子目录。
多卡参数是 `GPUS`、`NUM_NODES`，默认均为 1；`DEEPSPEED_BIN` 默认 deepspeed。
`RETRIEVAL_DTYPE`、`RANKER_DTYPE` 可分别配置；未指定时使用 `DTYPE` 或 float16。
检索和重排各有 `*_QWEN_MIN_PIXELS`、`*_QWEN_MAX_PIXELS`，未指定时使用
`QWEN_MIN_PIXELS`、`QWEN_MAX_PIXELS` 或 0（检查点默认值）。检索像素设置必须与编码一致。

## alpha 与输出

默认固定权重，不读取或传递验证输入。开启自动选优时：

```bash
# 未显式提供验证索引：直接在目标 test 上扫描 0.1～0.9。
SELECT_HYBRID_ALPHA=true DATASET=flickr bash scripts/three_stage_qwen.sh

# 使用显式验证索引选择权重，再用于目标集检索和重排。
SELECT_HYBRID_ALPHA=true DATASET=coco \
VALIDATION_DATASET_NAME=flickr VALIDATION_SPLIT=val \
VAL_PASSAGE_REPS=/path/to/flickr/val/dense \
VAL_SPARSE_INDEX=/path/to/flickr/val/sparse/collection \
bash scripts/three_stage_qwen.sh
```

验证 dense/sparse 必须成对提供，且验证与目标任务类型一致；路径损坏继续报错，
不会回退到 test。未指定验证输入时，即使默认 val 目录存在也不会自动读取。
批量任务可填写 `VAL_DENSE_INDEXES`、`VAL_SPARSE_INDEXES` 和 `VALIDATION_DATASETS` 表，
键为目标数据集。批量共用环境变量指定的验证路径时必须显式设置 `VALIDATION_DATASET_NAME`。

自动模式沿用现有三阶段扫描逻辑，复用检索分数，以选定权重执行一次目标重排；
`stage2_alpha_sweep.csv` 和 `alpha_selection.json` 在重排前保存。
有验证集时报告验证和目标两部分的各权重指标，目标指标不会改写验证集选择的权重。

默认输出规则：

```text
<QWEN_OUTPUT_ROOT>/<编码相对路径>/<方向>/three_stage/<alpha标签>/
```

`QWEN_OUTPUT_ROOT` 默认项目下 `three_stage_qwen_results/`。
固定权重标签如 `alpha_0.5`；自动选优使用 `alpha_val` 或 `alpha_test`（其他目标 split 使用对应名称）。
单任务可用 `OUTPUT_DIR` 覆盖最终路径；批量模式仅使用 `QWEN_OUTPUT_ROOT` 分别派生路径。
终端会打印每个任务的实际索引、结果路径和执行命令。

输出沿用 Python 三阶段入口的各阶段 TREC/JSON、`recall_metrics.json/.xlsx`、
`timing_metrics.json/.xlsx` 和 `pipeline_metadata.json`。
`OVERWRITE` 默认 false；如需复用现有结果目录，显式设置 `OVERWRITE=true`。
该开关只传递给 Python，启动脚本不会删除候选库或结果文件。

`MODIFY_TYPE=modify_single` 及先验缓存遵循现有适用范围：IR 图搜文和 it2t 的生成式重排。
`USE_RERANK_PRIOR_CACHE=true` 要求同时启用 modify_single，缓存目录可用 `RERANK_PRIOR_CACHE_DIR` 指定。

## 验证

启动器测试使用真实 Bash、临时路径和记录参数的 DeepSpeed 替身，命令交给真实三阶段参数解析器检查。
覆盖三代模型、数据集方向、批量执行、参数与路径一致性、新旧索引、固定/自动 alpha、验证输入、
失败停止及覆盖参数；不会加载完整模型或启动 Java。
实际运行仍需要项目的 GPU、DeepSpeed、Transformers、Pyserini/Java 和报告依赖 pandas/openpyxl 环境。
