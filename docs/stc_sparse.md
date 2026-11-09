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
| `STC_WEIGHT_SCALE` | `100` | 与非 STC 相同的整数化倍率；启用 STC 时仅接受 100 |
| `STC_CACHE_ROOT` | `${SPARSE_OUTPUT_ROOT}/stc_vocab` | 模型级缓存 |

各脚本显式列出这些参数，并通过普通变量和集中传入的子进程环境转发。Python 入口接受对应的小写下划线参数；检索入口也接受连字符参数。

## 计算与边界

输入、输出 embedding 分别 L2 归一化后拼接，再整体归一化。层次聚类采用余弦距离和 complete linkage。MiniBatchKMeans 使用随机初始化、三次初始化尝试及固定种子。簇编号按最小成员 token ID 规范化。

特殊 token、解码为空白的 token 不进入簇。STC v2 还会在分块检查后排除输入或输出 embedding 任一侧范数为零的 token，包含数字 token；不额外设置“接近零”的阈值。
保留的数字 token 各占独立维度，避免把不同数字合并。因此最终维度可能略大于 `STC_N_CLUSTERS`；日志和 manifest 记录实际数量。停用词作为普通 token，不采用原 STC 不确定性评分中的全匹配处理。

日志会报告输入／输出各自的零向量数量、合并后的排除数量、剩余词项数量，以及最多 20 个 token 的 ID、词项和解码文本示例。
`manifest.json` 的 `embedding_filter` 保存两侧及合并后的完整排除 ID、数量与规则，`token_to_cluster.npz` 的有效 token 列表不包含这些词项。
NaN/Inf（含非有限范数）继续报错，并指出矩阵及 token；不会被零向量排除掩盖，也不会发布完成清单。
过滤后剩余语义 token 少于 `STC_N_CLUSTERS` 时，需降低簇数，程序不会自动改变它。

STC 与非 STC 共用同一份按模型族区分的稀疏计算流程。提示词、激活位置和精度相同，先计算 `log1p(ReLU(raw_logit))`，再按原算法选择每个视角的 Top-K token。`SPARSE_LENGTH` 统一表示每视角 token 数，不再表示簇数；不会聚合全词表后挑选簇，也不会为了补齐簇数而选择额外 token。
各模型族保留原有过滤、大小写归并顺序。STC 额外在选中 token 后、词项合并前排除特殊、空白、零向量及不可映射 ID，不回填低排名 token。这样同名词项不会混入已排除 ID 的权重。

每个选中 token 先乘 100 并取整，随后沿用原算法的词项／视角 `sum/max/replace` 合并；`SPARSE_VALUE_MEAN=true` 时最后按实际视角数做整数除法。非正整数按原规则丢弃，不再强制提升到 1。仅在该完整流程结束后，将最终词项映射到簇，同簇整数权重固定求和，不再缩放、截断、做均值或再次应用 max/replace。Lucene 词项形如 `stc_000001`。

规范化后同名词项若对应多个簇，固定采用最小有效 token ID 对应的簇，查询和候选遵循相同规则。压缩后的非零维度可能少于选中的 token 数。除排除不可映射项外，末端投影保持总整数权重，不会从未选中的簇成员引入额外贡献。
所有脚本与 Python 默认倍率统一为 100；显式使用旧的 `STC_WEIGHT_SCALE=10000` 会报错，需同步修改编码、索引和检索配置。

该模式保留提示词、FreeRet 表征设置、一次拼接前向及视角级掩码，密集向量计算不变。它借用 STC 的固定 embedding 聚类，将稀疏权重恢复为原始的 `log1p(ReLU)`，不包含生成后缀匹配或答案不确定性评分。不支持 overlap 编码和 Top-K 实验入口。

独立全库检索也支持 STC：使用 `search_adaptive_views_ir.sh`、`search_adaptive_views_tbpr.sh` 或
`search_adaptive_views_fused.sh`，并设置与编码一致的 STC 开关和参数。查询复用候选产物附带的
映射，不重新聚类。直接运行 `src/search_adaptive_views.py` 并指定候选目录时，省略的 STC 选项
从 manifest 继承；显式冲突会报错。配置及示例见 [Adaptive Search](adaptive_search.md)。

## 缓存与产物

首次运行编码脚本时先在独立 CPU 单进程建立聚类缓存，再启动多卡编码。缓存命中时不加载模型权重；模型、适配器、tokenizer、提取精度及聚类配置参与缓存标识，与数据集和视角选择无关。层次聚类仍有平方规模的 CPU 内存需求，MiniBatchKMeans 避免完整两两距离矩阵；本实现不保存平方规模的布尔同簇矩阵。

新建映射统一使用 STC v2，请求中包含 `zero_norm_policy=exclude_input_or_output_zero_norm_v1`，
因此缓存和产物指纹与旧规则隔离；无需新增脚本开关或改变 `DTYPE`，原来的八项 STC 参数名称保留，倍率限定为 100。
自动查找和构建不会回退到旧 v1 缓存，也不会覆盖它。显式绑定的 v1 映射副本仍可严格校验并读取；映射版本与稀疏权重版本独立。

新编码配置记录 `method=token_topk_quantize_merge_then_cluster_sum`、`weight_transform=log1p_relu`、`weighting_version=3`、逐 token 的 x100 整数化及同名词项映射规则，因此编码目录、查询缓存和检索结果会生成新指纹。聚类请求及映射指纹不变，已有 v2 聚类缓存可直接复用，无需重新聚类；视角选择文件同样可以复用。
已有 softmax 产物和“先汇总簇再 Top-K”的 ReLU 产物均需重新编码、建立新索引。索引、search 和三阶段检索会拒绝这些旧权重配置并提示重编码；手动指定旧目录也不会绕过校验。原有产物不会被覆盖。

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
