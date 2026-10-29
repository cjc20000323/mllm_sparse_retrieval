# 三阶段 alpha 自动选择

原有固定权重模式保持默认，`HYBRID_ALPHA=0.5` 控制二阶段 dense 权重。
设置 `SELECT_HYBRID_ALPHA=true` 后扫描 0.1 至 0.9，步长 0.1；sparse 权重为 `1-alpha`。
普通任务按 R@1、R@5、R@10 均值选优，Fashion-IQ 按 dress、shirt、toptee 三类的
R@10、R@50 均值选优。并列（包括全零）时选择较小的 alpha。

```bash
# 不提供验证索引：在当前目标 split（默认 test）选择，然后执行一次重排。
SELECT_HYBRID_ALPHA=true bash scripts/three_stage_retrieval_pipeline.sh

# 提供验证索引：在验证集选择，用该权重执行目标集检索和一次重排。
# 这里示例使用 Flickr val 为 COCO test 选择 alpha，数据根目录仍为 DATA_ROOT。
DATASET=coco SELECT_HYBRID_ALPHA=true \
  VALIDATION_DATASET_NAME=flickr VALIDATION_SPLIT=val \
  VAL_PASSAGE_REPS="/path/to/flickr/text/val/dense" \
  VAL_SPARSE_INDEX="/path/to/flickr/text/val/sparse" \
  bash scripts/three_stage_retrieval_pipeline.sh
```

通用、TBPR、多模态及 Qwen 批量启动器均支持此开关。前三个脚本也接受
`--select-hybrid-alpha`、`--val-passage-reps`、`--val-sparse-index`、
`--validation-dataset-name`、`--validation-split`、`--val-stage1-sparse-index`、
`--val-stage1-run`。验证数据集默认当前数据集，split 默认 val，必须属于同一任务。
验证索引必须成对提供；缺文件、标签或配置不匹配会报错，不会静默回退到测试集。

验证流程沿用第一阶段的方法和候选深度，再在其候选范围内执行二阶段融合。
`sparse_concat` 默认让验证集第一、二阶段共用 `VAL_SPARSE_INDEX`；
`sparse_manual` 需另提供 `VAL_STAGE1_SPARSE_INDEX`。CLIP/BLIP 使用原有第一阶段模型。
也可通过 `VAL_STAGE1_RUN` 指定独立的验证集第一阶段结果；目标集的 run/外部命令不会被验证集复用。
验证集不执行重排。每个 split 的二阶段编码和 dense/sparse 检索仅执行一次，九个权重复用分数；
`sparse_concat` 继续复用第一阶段的查询表示，`sparse_manual` 保留两阶段各自的编码方法。

Qwen 批量入口可在新验证配置表中，按**目标数据集**填写：

```bash
declare -A VAL_DENSE_INDEXES=([flickr]="/flickr/val/dense" [coco]="/coco/val/dense")
declare -A VAL_SPARSE_INDEXES=([flickr]="/flickr/val/sparse" [coco]="/coco/val/sparse")
declare -A VALIDATION_DATASETS=([flickr]="flickr" [coco]="coco")
```

这些表位于 `scripts/three_stage_retrieval_pipeline_qwen.sh` 内，可直接修改。
`VAL_STAGE1_INDEXES`、`VAL_STAGE1_RUNS` 表用于按目标数据集覆盖验证第一阶段输入。
未设置验证索引的任务在自己的目标集选择。若多个任务共用环境变量指定的验证索引，
必须显式设置 `VALIDATION_DATASET_NAME`，且该验证集应与每个任务的类型一致。

自动输出目录追加 `alpha_select`，显式 `OUTPUT_DIR` 保持原样。输出包括：

- `stage2_alpha_sweep.csv`：UTF-8 BOM 汇总表，包含 dataset、split、validation/target 用途、
  alpha/beta、dense/sparse/hybrid、类别、查询数、全部 R@K、选择得分和 selected。
  召回率取值 0–1；dense/sparse 为每个 alpha 重复列出的对照。selection_score 只用于 hybrid。
- `alpha_selection.json`：选择来源、实际 dataset/split、指标定义、九个得分、最终权重和验证流程计时。
- `alpha_validation/`：有验证集时保存验证集第一阶段、dense、sparse 以及选中 hybrid 的 TREC/JSON。
- 现有二阶段 hybrid 文件和召回报告：使用最终权重；第三阶段只重排这些候选。

有验证集时，目标集也扫描并保存九个权重的召回率，但不会改写验证集选出的 alpha。
CSV 和选择结果在第三阶段前落盘。各进程只接收选中的权重，继续处理自己的查询分片。
验证流程计时独立保存在 `alpha_selection.json`；现有三个阶段的核心计时对应目标流程，
召回评估和文件输出仍排除在核心计时之外。

该模式要求实际执行二阶段检索与第三阶段重排，不接受 `--stage2-run` 或 `--ranker-run`。
耗时估计入口只支持固定 alpha，会明确拒绝自动选择开关。
