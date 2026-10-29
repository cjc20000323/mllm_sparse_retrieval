# 独立 Qwen 编码脚本

`scripts/encode_qwen.sh` 仿照 `encode.sh`，在顶部配置参数，末尾直接调用
`deepspeed ... src/encode.py`。它不调用其他编码脚本，也不读取 `qwen_retrieval_common.sh`。

默认任务为 Qwen2.5-VL-7B-Instruct、Flickr、`TASK_TYPE=ir`、`ENCODE_TYPE=text`、
test、1 GPU、float16、`disassembleeol_separate + concat`。文本候选库对应三阶段默认的图搜文。

```bash
# 直接使用默认配置，单次只编码 Flickr 文本库。
bash scripts/encode_qwen.sh

# 切换 Qwen 系列和数据集。qwen2 / qwen25 / qwen3 均可使用 QWEN_MODEL_PATH 指定检查点。
QWEN_FAMILY=qwen3 DATASET=coco bash scripts/encode_qwen.sh

# 文搜图需要预先编码图像候选库。IR_QUERY_TYPE 不会替代显式 ENCODE_TYPE。
DATASET=flickr TASK_TYPE=ir ENCODE_TYPE=image bash scripts/encode_qwen.sh

# 行人和多模态任务显式设置 task_type 与候选库模态。
DATASET=CUHK-PEDES TASK_TYPE=tbpr ENCODE_TYPE=image bash scripts/encode_qwen.sh
DATASET=webqa TASK_TYPE=t2it ENCODE_TYPE=corpus bash scripts/encode_qwen.sh
DATASET=remuq TASK_TYPE=it2t ENCODE_TYPE=corpus bash scripts/encode_qwen.sh

# 整体概括方法；CALCULATE_TYPE 自动设为 separate。
DATASET=flickr EOL_TYPE=prompteol bash scripts/encode_qwen.sh
```

参数可通过环境变量覆盖，也可直接修改脚本顶部的默认值。保留 `QWEN_CHECKPOINT_ROOT`、
`QWEN_MODEL_PATH`、`DTYPE`、`PER_DEVICE_BATCH_SIZE`、像素上下限及稀疏配置。
`SPARSE_LENGTH`、`TEXT_SPARSE_LENGTH`、`IMAGE_SPARSE_LENGTH` 默认都为 40，保证与三阶段默认配置一致。
如果未设置 `DATASET`，旧的单值 `DATASETS=flickr` 仍可使用；`DATASETS="flickr coco"` 会报错，
需要分两次运行。显式 `DATASET` 优先于旧变量。脚本仅接受 `-h/--help` 命令行选项。

生成验证集时可设置 `USE_GENERATED_VALIDATION_DATA=true` 和 `GENERATED_VALIDATION_DATA_ROOT`，
脚本会将 split 设为 val。`PROMPT_TYPE=prompt_random_5` 直接交给 Qwen 适配层处理，
随机种子使用 `RANDOM_VIEW_SEED`，默认 42；不会调用 LLaVA 的随机视角脚本。

产物目录不变：稠密表示在 `DENSE_OUTPUT_ROOT`（默认项目 `dense_output/`），
稀疏文档在 `SPARSE_OUTPUT_ROOT`（默认项目 `sparse_output/`）。内部仍按模型目录名、数据集、
模态、方法、提示词、split 和稀疏配置组织，行人任务还包含 `TBPR_TYPE`。
`OUTPUT_DIR` 默认 `./output`，是运行输出目录，不控制候选库表示的位置。
默认允许在原目录重新编码，无需增加覆盖参数。Python 编码入口由 rank 0 清理本次模式对应的
旧分片，避免 GPU 数量减少后残留旧数据。开始重跑前撤下完成清单，全部 rank 完成后重新写入；
失败运行不会保留上一轮的有效清单，可修复原因后继续执行原命令。

语料库重新编码时，旧 `index/` 和新 `collection/index/` 都会备份到编码叶目录下的
`index.stale_<唯一后缀>/`，位于 `collection` 外，终端打印备份位置。
编码完成后重新运行 `sparse_index_qwen.sh`，生成新的 `collection/index/`，再执行检索或三阶段流程。
备份索引不会自动作为当前索引使用。查询模式仅覆盖 `query.pkl`、`query.tsv`，不会备份语料索引。
其他数据集、配置目录和非编码文件保持原样。

建 Lucene、重排脚本继续读取公共配置，默认仍处理 Flickr 和 COCO。
`search_qwen.sh` 已改为独立脚本，和编码一样单次运行一个 `DATASET`。未显式指定 val 路径时
直接在 test 选优，由 rank 0 扫描一次并输出 alpha_sweep.csv、best.csv；显式指定时才使用 val。
详见 [独立 Qwen 检索脚本](qwen_search_scripts.md)。
如希望直接使用固定权重，可按下面的示例显式启用固定 alpha：

```bash
export QWEN_FAMILY=qwen25 DATASET=flickr DATASETS=flickr
export TASK_TYPE=ir ENCODE_TYPE=text IR_QUERY_TYPE=image
export EOL_TYPE=disassembleeol_separate

bash scripts/encode_qwen.sh
bash scripts/sparse_index_qwen.sh
USE_FIXED_ALPHA=true ALPHA=0.5 bash scripts/search_qwen.sh
bash scripts/rerank_qwen.sh
# 或在完成 encode、sparse_index 后执行三阶段：
bash scripts/three_stage_retrieval_pipeline_qwen.sh
```

若要同时准备两个数据集的候选库，分别执行 `DATASET=flickr` 与 `DATASET=coco` 的编码，
之后可用 `DATASETS="flickr coco"` 启动建索引、重排或三阶段的批量任务；独立检索需分别运行
`DATASET=flickr` 和 `DATASET=coco`。各阶段的模型、方法、提示词、稀疏和像素配置应保持一致。
