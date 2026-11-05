# 普通编码的 FreeRet 配套脚本

每个脚本每次运行一个模型、一个数据集，不使用 adaptive 视角选择。两个开关默认均为 `false`，可以修改脚本顶部配置或通过环境变量设置。

| 任务 | 编码 | 稀疏索引 | 三阶段检索 |
|---|---|---|---|
| IR | `encode_freeret_ir.sh` | `sparse_index_freeret_ir.sh` | `three_stage_retrieval_freeret_ir.sh` |
| TBPR | `encode_freeret_tbpr.sh` | `sparse_index_freeret_tbpr.sh` | `three_stage_retrieval_freeret_tbpr.sh` |
| fused | `encode_freeret_fused.sh` | `sparse_index_freeret_fused.sh` | `three_stage_retrieval_freeret_fused.sh` |

这 9 个文件都位于 `scripts/`。原 `encode.sh`、`sparse_index.sh`、`three_stage_retrieval_pipeline.sh` 保留。

## 配置

- `MODEL_FAMILY=llava_mistral` 为默认值，另支持 `llava_llama`、`qwen2`、`qwen25`、`qwen3`。`CHECKPOINT_ROOT` 默认 `/root/autodl-fs/checkpoints`，`MODEL_PATH` 可以覆盖；Qwen 默认目录名采用项目的 `Qwen-Qwen…` 命名，可用 `MODEL_PATH` 指定实际目录。重排模型默认跟随 `MODEL_PATH`。
- IR 默认 Flickr 图→文，编码文本候选；设置 `QUERY_TYPE=text` 切换为文→图，编码图像候选。COCO 使用 `DATASET=coco`。
- TBPR 默认 CUHK-PEDES 文→图；另支持 ICFG-PEDES、RSTPReid。`TBPR_TYPE` 默认 `origin_type`，可以改为 `type`，三个阶段需保持一致。
- fused 默认 WebQA（文本→图文）；ReMuQ、LLaVA 使用图文查询→文本候选。`QUERY_TYPE=image` 或 `composed` 均保留查询文本，候选编码使用 `ENCODE_TYPE=corpus`。
- `ENCODE_TYPE` 留空时按检索方向推导；显式设置时必须匹配。使用单值 `MODEL_FAMILY`、`DATASET`，不使用批量列表变量。
- 默认 `prompt_5`、`disassembleeol_separate+concat`、非手工稀疏、长度 40、`after_pad`、float16、batch size 1、1 GPU。其他配置在各脚本中展开。
- 编码沿用 `src/arguments.py` 配置的数据位置，默认 `/root/autodl-tmp/data`；检索的 `DATA_ROOT` 必须指向同一数据树。编码脚本没有新增改变数据根目录的参数。
- 三阶段检索默认深度均为 1000，`HYBRID_ALPHA=0.5`，重排默认 `caption_generation`、`MODIFY_TYPE=no`、前 5 项。可以设置独立验证集及 alpha 扫描参数；该组脚本不自动改变重排计分模式。

## 两个开关

| DENSE | PROMPT_CONSTRAINTS | 效果及目录后缀 |
|---|---|---|
| false | false | 原表征及提示词，原目录 |
| true | false | FreeRet pre-MLP 密集层，原提示词；`_dense_freeret_pre_mlp` |
| true | true | FreeRet 密集层、输入前一次任务描述及各视角约束；`_dense_freeret_pre_mlp_freeret_constraints_v2` |

`USE_FREERET_PROMPT_CONSTRAINTS=true` 要求 `USE_FREERET_DENSE=true`。三个阶段应使用同一组合；目录与模式元数据不匹配时会报错。未标记的显式产物目录会按开关补齐后缀，已经标记的正确目录不会重复添加。

任务描述取自 `freeret_prompt_catalog.py` 的 `task_description`，放在原聊天前缀之后、图文输入之前，仅加入一次。Summary、任务约束、聊天模板和实际重排提示词保持原样。v1 产物不含任务描述，需重新编码并建立 v2 索引；开启约束时不能混用旧产物。关闭约束时保持原行为。

另可设置 `USE_FREERET_DENSE=true USE_FREERET_FULL_PROMPT=true` 启用完整 FreeRet 提示词。新开关默认 `false`；开启后替代现有 Summary 提示词，整体 dense 使用原 FreeRet 模板，sparse 在语义规则中写入各视角，其他规则保留。此时旧 `USE_FREERET_PROMPT_CONSTRAINTS` 不再叠加内容。目录后缀为 `_dense_freeret_pre_mlp_freeret_full_prompt_v1`，三个阶段都需使用同一新开关，并重新编码、建索引。实际重排评分提示词不变。

例如为 TBPR 开启两个模式，依次执行：

```bash
USE_FREERET_DENSE=true USE_FREERET_PROMPT_CONSTRAINTS=true bash scripts/encode_freeret_tbpr.sh
USE_FREERET_DENSE=true USE_FREERET_PROMPT_CONSTRAINTS=true bash scripts/sparse_index_freeret_tbpr.sh
USE_FREERET_DENSE=true USE_FREERET_PROMPT_CONSTRAINTS=true bash scripts/three_stage_retrieval_freeret_tbpr.sh
```

切换数据集、模型、稀疏参数或图像模式时，三个阶段使用一致设置。`LLAVA_IMAGE_MODE=base` 仅支持 LLaVA-NeXT；Qwen 使用 `anyres`。结果路径含检索方向、split、`stage1_<depth>_stage2_<depth>` 和 FreeRet 后缀。

## 路径及预检

`DENSE_OUTPUT_ROOT`、`SPARSE_OUTPUT_ROOT` 指定编码根目录；`OUTPUT_DIR` 指运行日志或检索结果，不是编码根目录。索引及检索可用 `DENSE_CORPUS_DIR`、`SPARSE_CORPUS_DIR` 指向已有产物，检索还支持显式 Stage-1/2 路径。

LLaVA 的普通分片及索引位于 `<实验目录>/corpus_*.jsonl`、`<实验目录>/index`；Qwen 位于 `<实验目录>/collection/corpus_*.jsonl`、`<实验目录>/collection/index`。索引输入会隔离为语料分片，不包含 manifest 或其他文件。已有索引默认拒绝覆盖。

```bash
DRY_RUN=true USE_FREERET_DENSE=true USE_FREERET_PROMPT_CONSTRAINTS=true bash scripts/encode_freeret_ir.sh
DRY_RUN=true DATASET=remuq MODEL_FAMILY=qwen25 bash scripts/three_stage_retrieval_freeret_fused.sh
bash scripts/sparse_index_freeret_ir.sh --help
```

dry-run 检查配置并打印最终目录和启动命令，不要求模型或产物存在，不加载权重、不写编码文件、不建索引。正式检索在启动 DeepSpeed 前检查 checkpoint、数据标注、dense 分片、Lucene 分段和 FreeRet 元数据。脚本只接受 `--help`，其余参数通过变量配置。
