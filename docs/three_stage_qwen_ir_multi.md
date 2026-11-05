# Qwen IR 多模型、多数据集三阶段检索

新增 `scripts/three_stage_qwen_ir_multi.sh`，独立调用 Python 三阶段入口，已有脚本保持不变。
默认运行 Qwen2-VL-7B、Qwen2.5-VL-7B、Qwen3-VL-8B × Flickr、COCO × 图搜文、文搜图，共 12 个串行任务。
顺序为模型 → 数据集 → 查询方向。先准备各模型的 text、image 两套候选编码和对应的稀疏索引。

```bash
QWEN_FAMILIES="qwen2 qwen25 qwen3" DATASETS="flickr coco" \
IR_QUERY_TYPE="" QUERY_TYPE="" IR_QUERY_TYPES="image text" \
bash scripts/three_stage_qwen_ir_multi.sh
```

- `QWEN_FAMILY`、`DATASET` 非空时分别优先选择一个系列和一个数据集。
- `IR_QUERY_TYPES` 默认 `image text`，也可设为 `text image` 调整顺序，或只填一个方向。
  空列表、重复方向和其他值会报错。
- 非空 `IR_QUERY_TYPE` 优先选择一个方向；`QUERY_TYPE` 是单值别名，两者显式冲突时报错。
  `image` 为图搜文，读取 text 候选；`text` 为文搜图，读取 image 候选。
  如果旧环境已经设置单值参数，先清空它们再启用双向列表。
- 默认 1 GPU、float16、prompt_5、disassembleeol_separate+concat、稀疏长度 40，
  `what_caption_generation` 重排前 5 项。可设置阶段深度、批大小和检索/重排的独立精度、像素预算。

## 模型与路径

`QWEN_CHECKPOINT_ROOT` 默认 `/root/autodl-fs/checkpoints`，三个默认子目录为
`Qwen2-VL-7B-Instruct`、`Qwen2.5-VL-7B-Instruct`、`Qwen3-VL-8B-Instruct`。
若实际目录名不同，在脚本配置区编辑 `MODEL_PATHS`；候选路径使用实际 checkpoint 的 basename。
每个 ranker 默认使用当前检索模型，可通过 `RANKER_MODEL_PATHS` 按系列覆盖。
`QWEN_MODEL_PATH`、`RANKER_MODEL_PATH` 的全局覆盖仅用于选中一个系列时。

```bash
declare -A MODEL_PATHS=([qwen25]="/models/Qwen-Qwen2.5-VL-7B-Instruct")
declare -A RANKER_MODEL_PATHS=([qwen25]="/models/other-ranker")
```

`DENSE_OUTPUT_ROOT`、`SPARSE_OUTPUT_ROOT` 默认项目目录下的 dense_output、sparse_output。
候选路径遵循现有 Qwen `artifact_dirs()`；检索和编码的提示词、seed、稀疏及像素配置须一致。
稀疏输入支持编码叶目录、collection、index、collection/index，优先选择 collection/index。

批次覆盖使用 `family|dataset|query_type` 为键的 `DENSE_INDEXES`、`SPARSE_INDEXES`、`STAGE1_INDEXES`，例如：

```bash
declare -A DENSE_INDEXES=(["qwen25|flickr|image"]="/data/text/dense/leaf" ["qwen25|flickr|text"]="/data/image/dense/leaf")
declare -A SPARSE_INDEXES=(["qwen25|flickr|image"]="/data/text/sparse/leaf/collection" ["qwen25|flickr|text"]="/data/image/sparse/leaf/collection")
```

旧 `family|dataset` 键仅在选中一个查询方向时兼容；三段精确键优先。
双向运行遇到适用的旧键会报错，要求明确补上方向，避免错误复用相反方向的候选表征。
`STAGE*_INDEX` 和精确 `OUTPUT_DIR` 的全局覆盖仅用于一个系列、一个数据集、一个方向的组合。
默认输出为 `QWEN_OUTPUT_ROOT/<候选相对路径>/<方向>/three_stage/<alpha模式>/`
`<modify_type>_prior_<缓存状态>/stage1_<数量>_stage2_<数量>`。
其中 `QWEN_OUTPUT_ROOT` 默认项目 three_stage_qwen_results，显式 `OUTPUT_DIR` 保持原样。

## Alpha 和 prior

`SELECT_HYBRID_ALPHA=false` 默认使用固定 `HYBRID_ALPHA=0.5` 并忽略验证集输入。
设为 true 时，在成对验证集索引上选择 alpha；未指定时在目标 split 选择，继续保存各 alpha 和 best CSV。
批次验证集使用 `VAL_DENSE_INDEXES`、`VAL_SPARSE_INDEXES`、`VALIDATION_DATASETS` 表，键同样为 `family|dataset|query_type`。
每个方向独立选择 alpha，读取对应候选模态的验证集表征；固定 alpha 模式忽略验证集表。
全局 `VAL_PASSAGE_REPS`、`VAL_SPARSE_INDEX`、`VALIDATION_DATASET_NAME` 仅用于单个任务。

`AUTO_MODIFY_BY_QUERY_TYPE=true` 默认开启：image → modify_single，text → no，优先于全局 MODIFY_TYPE。
自动 no 任务关闭缓存；校正任务使用 `USE_RERANK_PRIOR_CACHE`（默认 false）。
输出目录中的 `prior_true/false` 标记缓存开关。图搜文启用校正缓存示例：

```bash
IR_QUERY_TYPE=image USE_RERANK_PRIOR_CACHE=true \
bash scripts/three_stage_qwen_ir_multi.sh
```

关闭自动模式后使用全局 `MODIFY_TYPE`（默认 no）。校正只支持图搜文和 caption-generation 模板，
缓存还要求 modify_single。每个任务打印有效查询类型、模式和缓存状态。

脚本在启动 GPU 前检查全部组合。空/重复列表、目录冲突、缺少 checkpoint 或索引会报错；
已有输出需要 `OVERWRITE=true`。任一运行任务失败立即停止并返回该任务退出码。
