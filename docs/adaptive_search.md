# Adaptive views 全库检索

`src/search_adaptive_views.py` 使用 `encode_adaptive_views.py` 保存的视角进行在线查询编码，
分别在完整候选库上执行 dense、sparse 检索，再融合为 hybrid。不重新选择视角、不编码候选库、
不建立索引、不执行三阶段候选筛选或重排，不支持 overlap 产物。原 `search.py` 保持原样。

## 批量启动

```bash
bash scripts/search_adaptive_views_ir.sh
bash scripts/search_adaptive_views_tbpr.sh
bash scripts/search_adaptive_views_fused.sh
```

三个脚本分别默认覆盖 Flickr/COCO、CUHK-PEDES/ICFG-PEDES/RSTPReid、WebQA/ReMuQ/LLaVA，
以及 Qwen2、Qwen2.5、Qwen3、LLaVA-Llama、LLaVA-Mistral。IR 默认双向检索。
TBPR 是文本搜行人图像，WebQA 是文本搜图文，ReMuQ/LLaVA 是**图文共同查询文本库**。

配置均在各 `.sh` 中展开，也可通过环境变量覆盖：

```bash
MODEL_FAMILIES='llava_mistral qwen25' DATASETS='RSTPReid ICFG-PEDES' \
  DATA_ROOT=/root/gpufree-share CHECKPOINT_ROOT=/root/gpufree-share/checkpoints \
  DENSE_OUTPUT_ROOT=/root/gpufree-share/dense_output \
  SPARSE_OUTPUT_ROOT=/root/gpufree-share/sparse_output \
  DRY_RUN=true bash scripts/search_adaptive_views_tbpr.sh
```

`MODEL_FAMILY`、`DATASET`、`IR_QUERY_TYPE` 的非空单值分别覆盖复数列表。批量任务顺序运行，
每个任务内部通过 `GPUS`/`NUM_NODES` 分片查询。`DRY_RUN=true` 校验配置和文件、打印实际命令，
不加载模型权重、不写结果；配置和 processor 仍需要可读取。

自动定位要求模型/数据根路径、精度、图像处理、稀疏参数和视角选择配置与编码时一致，
包括 `VIEW_EXAMPLE_MODE`、采样数量、PMI、层级候选、打乱 run ID、累计概率模式及阈值。
这些采样参数只用于寻找原缓存，检索过程中不会重新采样或选择。默认值对齐普通 adaptive 编码脚本。
无法找到精确实验时不会自动选取最新实验或重新选择视角。

Qwen 必须使用 `LLAVA_IMAGE_MODE=anyres`。编码时使用 LLaVA `base` 的实验，需设置 `base`，
并把模型列表限定为 LLaVA。`USE_FREERET_DENSE`、`USE_FREERET_PROMPT_CONSTRAINTS`、
`USE_FREERET_FULL_PROMPT` 必须与编码一致，完整 FreeRet 提示词优先。

## STC 稀疏词表

三个 search 脚本已显式列出并转发 STC 的全部八项参数，默认值与编码端一致：

| 参数 | 默认值 |
| --- | --- |
| `USE_STC_SPARSE` | `false` |
| `STC_ALGORITHM` | `agglomerative` |
| `STC_N_CLUSTERS` | `16000` |
| `STC_SEED` | `42` |
| `STC_BATCH_SIZE` | `4096` |
| `STC_MAX_ITER` | `100` |
| `STC_WEIGHT_SCALE` | `10000` |
| `STC_CACHE_ROOT` | 空；自动定位使用 `${SPARSE_OUTPUT_ROOT}/stc_vocab` |

例如候选使用 MiniBatchKMeans、16000 个语义簇编码后，保持其他编码参数和目录一致：

```bash
USE_STC_SPARSE=true STC_ALGORITHM=minibatch_kmeans STC_N_CLUSTERS=16000 \
  MODEL_FAMILY=llava_mistral DATASET=RSTPReid \
  bash scripts/search_adaptive_views_tbpr.sh
```

自动定位从相同模型的现有聚类缓存计算 STC 产物路径。手动指定候选目录后，查询使用该产物的
`stc_vocab/` 副本，原缓存目录可以不存在。检索从不重聚类，也不通过忽略指纹来兼容不一致的词表。
模型、tokenizer、映射指纹、聚类算法、语义簇数和概率量化规则均参与校验。MiniBatchKMeans
还校验种子、批大小和迭代轮数；层次聚类不使用这三个参数。

直接 Python 调用时，未提供的 STC 开关和参数从候选 manifest 继承；显式冲突会报告具体字段。
shell 脚本的 `USE_STC_SPARSE=false` 是显式设置，检索 STC 产物时须改为 `true`。
同时提供 dense/sparse 时两份完整配置须一致；dense-only、sparse-only 也支持 STC 产物。
独立验证库绑定自己的映射副本，并要求与目标库具有相同 STC 映射指纹和量化配置。

查询按编码端相同的“原始 logits → softmax → 簇内概率求和 → top-K → 跨视角合并 → 整数量化”
流程生成词项，dense、sparse、hybrid 指标仍写入原有 CSV。STC 配置位于
`run_metadata.json` 的 `configuration.encoding.stc_sparse` 中，结果目录增加 `_stc_...` 标记。
关闭 STC 时不增加编码指纹字段，原结果指纹与路径保持原行为。

## 明确指定候选目录

仅单个模型、数据集、方向允许手动覆盖：

```bash
MODEL_FAMILY=llava_mistral DATASET=RSTPReid \
  DENSE_CORPUS_DIR=/path/from/encoding/Adaptive_dense \
  SPARSE_CORPUS_DIR=/path/from/encoding/Adaptive_sparse \
  bash scripts/search_adaptive_views_tbpr.sh
```

任一目录非空即进入手动模式：只填写 dense 或 sparse 时只输出该路；两者都填写时输出三路。
脚本中的编码参数仍会校验，需与原实验一致。选择文件默认从 `adaptive_views_manifest.json`
读取，也可通过 `VIEW_SELECTION_FILE` 显式指定完全相同的概率选择记录。
`VIEW_SELECTION_ROOT` 用于按完整指纹恢复缺失引用；不会接受只有短指纹相近的其他文件。
稀疏索引必须预先完成，程序检查真实 `collection/index` 或明确指定的 `index` 内的 `segments_*`。
默认优先使用 `collection/index`；缺失时可只读兼容根目录旧 `index`。带 Adaptive 清单的根目录
JSONL 分片也能校验，但 search 不复制分片或建立索引。缺少有效索引时会给出对应的
`python -B src/adaptive_views.py index --sparse-root ...` 命令；也可先运行对应任务的
`scripts/sparse_index_adaptive_{ir,tbpr,fused}.sh`，自动保留原分片并整理到 `collection` 后建索引。
已有不完整的 `collection/index` 会报错，不回退到旧索引。

也可直接运行 Python；省略的编码参数、FreeRet 和 STC 选项从候选清单继承，显式冲突则报错：

```bash
python -B src/search_adaptive_views.py \
  --passage_reps /path/to/adaptive_dense \
  --sparse_index /path/to/adaptive_sparse \
  --adaptive_data_root /root/gpufree-share \
  --query_type text --depth 1000 --alpha 0.5
```

两个目录必须属于同一编码实验和完整选择指纹。旧累计概率 v1 的编码产物需要按当前 v2
重新编码并建索引，不能直接用于检索；overlap 产物会被拒绝。

## 融合权重

默认 `SELECT_HYBRID_ALPHA=false`，dense 权重为 `HYBRID_ALPHA=0.5`，sparse 权重为 `1-alpha`。
融合使用项目现有的每查询 min-max 归一化；每路在全库检索 `DEPTH` 个候选，再融合候选并集。

设置 `SELECT_HYBRID_ALPHA=true` 后扫描 0.1～0.9，按 `R@1 + R@5 + R@10` 最大值选择，
平分取较小 alpha，同时报告 R@20。只有一条检索分支时不能自动选权重。
单任务可提供 `VAL_PASSAGE_REPS`、`VAL_SPARSE_INDEX`，可选 `VAL_VIEW_SELECTION_FILE`、
`VALIDATION_DATASET_NAME` 和 `VALIDATION_SPLIT`，在独立验证库上选择 alpha，再用于目标集。
验证库加载自己的锁定视角，但必须使用相同任务和模型/表征配置。
未提供验证库时在目标集选择，报告明确标记 `selection_source=target`，不会将其标为验证集结果。

## 输出

默认目录为：

```text
adaptive_search_results/<模型目录名>/<数据集>/<检索方向>/<split>/<检索指纹>/
```

FreeRet 模式按项目现有命名规则附加相应后缀。指纹包含完整编码/选择指纹、输入目录、
检索深度、方向和固定/自动权重策略。单任务可通过 `OUTPUT_DIR` 覆盖。

每个任务输出：

- `dense_run.json/.trec`、`sparse_run.json/.trec`、`hybrid_run.json/.trec`：实际启用分支的排名。
- `recall_metrics.csv/.json`：各分支的 Recall@1、@5、@10、@20，数值单位为百分比。
- `alpha_sweep.csv`：仅自动选择时输出，记录九个权重、指标、选择来源及选中项。
- `run_metadata.json`：运行状态、完整配置和指纹、选择文件、视角及概率、查询提示词示例、最终权重与指标。

批量汇总位于 `SEARCH_OUTPUT_ROOT/.batches/<批次ID>/batch_summary.csv`，每完成一个任务就更新。
缺失输入的任务标记 skipped；没有可运行任务时返回非零。配置错误或执行失败返回非零并保留
已完成的报告。默认拒绝已有输出目录，显式设置 `OVERWRITE=true` 可重跑。

Jupyter 可直接读取：

```python
import pandas as pd
display(pd.read_csv("/path/to/task/recall_metrics.csv"))
```
