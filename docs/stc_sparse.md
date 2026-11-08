# Adaptive Views 的 STC 稀疏词表压缩

普通 Adaptive Views 的编码、稀疏索引和三阶段检索支持同一开关。IR、TBPR、fused 三组脚本均可批量运行多个模型和数据集。默认 `USE_STC_SPARSE=false`，原提示词、词项权重、指纹和产物路径保持原行为。

```bash
# RSTPReid 示例；内存充足时可选 STC 原版 agglomerative。
USE_STC_SPARSE=true STC_ALGORITHM=minibatch_kmeans \
MODEL_FAMILIES=llava_mistral DATASETS=RSTPReid \
bash scripts/encode_adaptive_views_tbpr.sh

# 使用与编码相同的模型、路径、视角选择及 STC 配置建立索引。
USE_STC_SPARSE=true STC_ALGORITHM=minibatch_kmeans \
MODEL_FAMILIES=llava_mistral DATASETS=RSTPReid \
bash scripts/sparse_index_adaptive_tbpr.sh

USE_STC_SPARSE=true STC_ALGORITHM=minibatch_kmeans \
MODEL_FAMILIES=llava_mistral DATASETS=RSTPReid \
bash scripts/three_stage_retrieval_pipeline_adaptive_views_tbpr.sh
```

将脚本后缀 `tbpr` 替换为 `ir` 或 `fused` 可使用另外两类任务。编码和索引的其他参数仍需对应。三阶段检索既支持第二阶段 hybrid，也支持 `STAGE2_METHOD=dense`；后一种情况下第一阶段仍使用 STC 稀疏表征。

## 参数

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `USE_STC_SPARSE` | `false` | 开启压缩 |
| `STC_ALGORITHM` | `agglomerative` | `agglomerative` / `minibatch_kmeans` |
| `STC_N_CLUSTERS` | `16000` | 非数字 token 的语义簇数 |
| `STC_SEED` | `42` | MiniBatchKMeans 种子 |
| `STC_BATCH_SIZE` | `4096` | MiniBatchKMeans 批大小 |
| `STC_MAX_ITER` | `100` | MiniBatchKMeans 最大迭代轮数 |
| `STC_WEIGHT_SCALE` | `10000` | 聚合概率转换为整数的倍率；最大 1000000 |
| `STC_CACHE_ROOT` | `${SPARSE_OUTPUT_ROOT}/stc_vocab` | 模型级缓存 |

各脚本显式列出这些参数，并通过普通变量和集中传入的子进程环境转发。Python 入口接受对应的小写下划线参数；检索入口也接受连字符参数。

## 计算与边界

输入、输出 embedding 分别 L2 归一化后拼接，再整体归一化。层次聚类采用余弦距离和 complete linkage。MiniBatchKMeans 使用随机初始化、三次初始化尝试及固定种子。簇编号按最小成员 token ID 规范化。

特殊 token、解码为空白的 token 不进入簇。含数字的 token 各占独立维度，避免把不同数字合并。因此最终维度可能略大于 `STC_N_CLUSTERS`；日志和 manifest 记录实际数量。停用词作为普通 token，不采用原 STC 不确定性评分中的全匹配处理。

每个视角的原始 logits 在真实 tokenizer ID 上用 float32 softmax；输出头中未定义的填充行不参与分母。再排除不可用 token 并应用 `IS_FILTERED` 等词项过滤，不重新归一化。簇内概率求和后才选择前 `SPARSE_LENGTH` 个簇；该参数现在表示每视角的簇数，而非 token 数。

各视角按 `SPARSE_VALUE_TYPE=sum/max/replace` 合并，`SPARSE_VALUE_MEAN=true` 时除以实际视角数。最后乘 `STC_WEIGHT_SCALE` 并四舍六入五成双取整，正概率保留项至少为 1。Lucene 词项形如 `stc_000001`。整数化会引入量化误差，稀疏及 hybrid 指标需要重新评估。

该模式保留提示词、FreeRet 表征设置、一次拼接前向及视角级掩码，密集向量计算不变。它使用 STC 的固定 embedding 聚类和概率质量聚合思想，不包含生成后缀匹配或答案不确定性评分。不支持 overlap 编码和 Top-K 实验入口。

独立全库检索也支持 STC：使用 `search_adaptive_views_ir.sh`、`search_adaptive_views_tbpr.sh` 或
`search_adaptive_views_fused.sh`，并设置与编码一致的 STC 开关和参数。查询复用候选产物附带的
映射，不重新聚类。直接运行 `src/search_adaptive_views.py` 并指定候选目录时，省略的 STC 选项
从 manifest 继承；显式冲突会报错。配置及示例见 [Adaptive Search](adaptive_search.md)。

## 缓存与产物

首次运行编码脚本时先在独立 CPU 单进程建立聚类缓存，再启动多卡编码。缓存命中时不加载模型权重；模型、适配器、tokenizer、提取精度及聚类配置参与缓存标识，与数据集和视角选择无关。层次聚类仍有平方规模的 CPU 内存需求，MiniBatchKMeans 避免完整两两距离矩阵；本实现不保存平方规模的布尔同簇矩阵。

直接运行 Python 多卡编码前，需先预计算：

```bash
python -B src/stc_sparse.py build \
  --model-name-or-path /path/to/checkpoint --dtype float16 \
  --stc-algorithm minibatch_kmeans --stc-n-clusters 16000 \
  --stc-cache-root /path/to/sparse_output/stc_vocab
```

缓存位于 `<STC_CACHE_ROOT>/<checkpoint名称>/<配置指纹前16位>/`，包含：

- `manifest.json`：身份、聚类配置、实际维度和映射校验指纹。
- `token_to_cluster.npz`：真实词表 ID、有效 token ID 与簇编号。
- `cluster_members.json`：每个簇的全部成员及规模，便于检查聚类质量。

编码的 dense、sparse 目录都增加 `_stc_<算法>_k<簇数>_<模式指纹>` 标记；dense 的数值计算相同，目录分开以保证完整 manifest 一致。每份产物附带 `stc_vocab/` 映射副本，三阶段检索使用候选产物锁定的副本。建索引和检索不会重新聚类；校验失败不会退回普通词表。

视角概率与选择缓存不受 STC 开关影响，已有普通视角选择可以复用。`DRY_RUN=true` 的批量编码不预计算、不加载模型、不写产物；独立 builder 的 `--dry-run` 只检查已有缓存或报告预计算位置。
