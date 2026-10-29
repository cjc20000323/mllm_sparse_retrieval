# 独立 Qwen 检索脚本

`scripts/search_qwen.sh` 仿照 `search.sh`：顶部集中配置参数，末尾直接执行
`deepspeed ... src/search.py`，不调用其他启动脚本。可编辑顶部配置，也可通过环境变量覆盖。

默认配置为 Qwen2.5-VL-7B-Instruct、单个 Flickr、图搜文、test、1 GPU、float16、
`disassembleeol_separate + concat`。未显式配置 val 路径时直接使用目标 split（默认 test）选优，
即使默认 val 目录已经存在，也不会自动加载它。

## 启动与 alpha 模式

```bash
# 先准备匹配的 test dense 表示及 Lucene 索引；不指定 val 就直接在 test 选优。
bash scripts/search_qwen.sh

# 如需固定权重，显式启用固定 alpha；不读取验证集索引。
USE_FIXED_ALPHA=true ALPHA=0.5 bash scripts/search_qwen.sh

# 切换模型和数据集；qwen2、qwen25、qwen3 均支持 QWEN_MODEL_PATH。
QWEN_FAMILY=qwen3 DATASET=coco bash scripts/search_qwen.sh

# 文搜图：需要匹配的图像候选库及索引。
DATASET=flickr IR_QUERY_TYPE=text bash scripts/search_qwen.sh

# 与之前 sparse_length=30 的编码保持一致。
SPARSE_LENGTH=30 bash scripts/search_qwen.sh
```

`USE_FIXED_ALPHA=false` 是默认模式，在选定数据集上扫描 0.1～0.9，步长 0.1，
以 R@1、R@5、R@10 之和选择最佳值，并用于目标 split；同分（含全零）取较小的 alpha。
没有显式 val 路径时，不探测默认验证目录，不传递 val 参数，复用目标集已算出的
dense/sparse 分数选优，不重新编码查询。显式指定任一 val 路径时，未指定的另一项仍按默认
目录补齐，两者必须完整；错误路径、损坏索引或清单不匹配继续报错。
`USE_FIXED_ALPHA=true` 直接使用 `ALPHA`，其默认值为 0.9，忽略全部 val 配置。

多卡仍按查询分片编码和检索，但汇总后的融合、选优与召回评估只由 rank 0 执行。
无 val 时全局只做九次候选融合与 hybrid 评估，dense/sparse 基准指标各计算一次，
缓存最佳候选用于最终 TREC 和报告，不再重算最佳 alpha。显式 val 模式扫描 val，
然后仅计算所选权重在目标集的结果。共享重排路径接收已缓存候选分片，查询不会被重复分片。
融合权重为 dense=alpha、sparse=1-alpha；`USE_GPU` 默认 false，仅控制 FAISS 是否使用 GPU。

单次只处理一个 `DATASET`，默认 flickr。未设置时兼容单值 `DATASETS`，多值会报错；显式
`DATASET` 优先。COCO/Flickr 的方向由 `IR_QUERY_TYPE=image|text` 控制。行人数据集自动使用
tbpr/文本搜图，webqa 使用 t2it，remuq/llava 使用 it2t，与原 Qwen 入口的任务映射一致。
`EOL_TYPE=prompteol` 自动选择 separate；默认的 disassembleeol_separate 对应 concat。

## 输入与输出路径

`DENSE_OUTPUT_ROOT`、`SPARSE_OUTPUT_ROOT` 默认分别为项目的 `dense_output/`、`sparse_output/`。
脚本按编码端的规则拼接模型目录名、数据集、候选库模态、过滤配置、方法、提示词、split 和
稀疏参数；行人任务还包含 `TBPR_TYPE`。测试集和显式 val 的 sparse 路径均以 `collection` 结尾。
脚本预检查及 Python 搜索器优先解析到 `collection/index/`，并从编码叶目录读取 Qwen 清单。
仅当 `collection/index` 不存在时兼容旧的同级 `index/`；新目录存在但缺少 `segments_*` 时直接报错。

也可直接指定路径，适合使用新的编码根目录或单独保存验证集：

```bash
PASSAGE_REPS="/path/to/test/dense" \
SPARSE_INDEX="/path/to/test/sparse/collection" \
VAL_PASSAGE_REPS="/path/to/val/dense" \
VAL_SPARSE_INDEX="/path/to/val/sparse/collection" \
OUTPUT_DIR="/path/to/search/results" \
bash scripts/search_qwen.sh
```

上述路径必须对应同一数据集和相同编码配置。dense 可指向包含 `corpus_*.pkl` 的目录或单个
分片文件。sparse 兼容编码根目录、旧 `index/`、`collection/` 和 `collection/index/`，脚本统一转换为
`collection` 路径后传给 Python；不会重复追加 `collection`，也支持末尾斜杠和路径中的空格。
Python 内部读取已建 Lucene 索引，不会直接搜索 JSONL。缺少索引时提示先运行
`sparse_index_qwen.sh`；无需为这次路径兼容重新编码或重建有效的现有索引。
多分片候选库应传入 dense 目录，Python 仍会核对分片数量与配置清单。

终端同时打印配置的 `collection` 路径和实际 Lucene `index` 路径。直接调用 Python 时，
`--sparse_index`、`--val_sparse_index` 也兼容上述四种路径写法；直接指定 `index` 目录会读取该目录。

新建索引统一使用下面的布局，已有索引不会自动搬迁：

```text
<编码叶目录>/
├── qwen25_prompt_manifest.json  # Qwen2 / Qwen3 使用各自的清单名
└── collection/
    ├── corpus_0.jsonl
    └── index/
        └── segments_*
```

`sparse_index_qwen.sh` 调用 `src/qwen_sparse_index.py`，仅暂存 `corpus_*.jsonl`（优先硬链接，
失败则复制）。它在 `collection` 外构建临时索引，成功并检查 `segments_*` 后放入
`collection/index`，成功或失败都清理临时目录。现有目标目录不会被覆盖；旧同级索引可以保留。
已编码的数据可直接建库，无需再次 encode：

```bash
python -B src/qwen_sparse_index.py --sparse-root "/path/to/sparse/leaf" --threads 16
```

默认结果目录为 `QWEN_OUTPUT_ROOT`（项目下 `qwen_retrieval_results/`）加上编码相对路径，
再加 `<检索方向>/search/alpha_val`（验证集选优）、`alpha_test`（test 选优，其他目标 split
使用对应名称）；固定模式末尾为 `alpha_<ALPHA>`。`OUTPUT_DIR` 可直接覆盖。
结果沿用现有 Python 实现：`dense_run.trec`、`sparse_run.trec`、`hybrid_run.trec`、
`recall_metrics.json`、`alpha_sweep.csv`、`best.csv` 和 Qwen 配置清单。本次运行不再写 `best.xlsx`。
JSON 和终端报告额外包含 `alpha_selection`，
记录 `source=fixed|validation|target`、实际 `dataset` 和 `split`；固定模式后二者为 null。
终端打印实际输入、输出路径及选择结果。

`alpha_sweep.csv` 将九组权重放在同一张表中，每组含 dense、sparse、hybrid 三种方法；
基准行复用已缓存指标。`best.csv` 保存最佳权重对应的目标集结果。显式使用 val 时，
汇总表的 dataset/split 对应验证集，best 表对应目标集；固定模式只写 best 表。
两份 CSV 使用 UTF-8 BOM，包含 dataset、split、selection_source、alpha、beta、method、
category、selection_score、selected 以及该数据集全部 `Recall@K` 列，召回率范围为 0～100。
`category=all` 表示总体，Fashion-IQ 另有 dress、shirt、toptee 行；选优仍按类别宏平均指标。
selection_score 仅 hybrid 行填写，并记录选择来源上的得分；selected 标记最佳 hybrid。
共享重排入口的 best 表另含 rerank 行。

## 编码配置与生成验证集

模型、EOL 方法、提示词、`RANDOM_VIEW_SEED`（默认 42）、稀疏参数及像素设置需与编码时一致。
`SPARSE_LENGTH`、`TEXT_SPARSE_LENGTH`、`IMAGE_SPARSE_LENGTH` 默认均为 40；修改根目录或显式
索引路径不会自动更改这些参数，配置清单由 Python 检查。

生成验证集可使用 `USE_GENERATED_VALIDATION_DATA=true` 和 `GENERATED_VALIDATION_DATA_ROOT`，
并提前准备相应的 val 编码及索引。该开关仅用于验证流程，目标 `DATASET_SPLIT` 保持不变；
固定 alpha 和目标集回退模式均不传递生成验证集参数。生成验证集的路径遵循编码端 split=val 的规则，
不额外添加 `_generated_val`，可通过验证索引路径覆盖项选择单独保存的产物。

脚本仅接受 `-h/--help` 命令行选项。`rerank_qwen.sh` 和 `sparse_index_qwen.sh` 仍使用公共配置，
其批量数据集及默认 alpha 行为保持原样；运行各阶段时分别设置一致的配置。

直接调用 `src/search.py` 的 Qwen 分支时，两个 val 参数均不传表示在当前目标 split 选优；
如显式传入验证输入，必须同时提供完整的 `--val_passage_reps` 和 `--val_sparse_index`。
