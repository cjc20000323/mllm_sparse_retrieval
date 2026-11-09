# Adaptive Views 的 LENS 质心稀疏头

本模式移植 ACL 2025 论文 [Enhancing Lexicon-Based Text Embeddings with Large Language Models](https://aclanthology.org/2025.acl-long.930/) 的 LM-head 聚类方法。它只聚类输出头的原始行向量，以各簇算术均值作为新输出头；不拼接输入 embedding、不做 L2 归一化，不引入论文的训练、双向注意力或全序列池化，因此不等同于完整 LENS 复现。

对原始输出矩阵 `W` 建簇后得到质心矩阵 `C`，编码时直接计算 `cluster_logits = hidden @ C.T`。若原 head 有 bias，则同时取同簇 bias 的均值。这里的 hidden 是原 LM head 实际接收的隐藏状态，独立于 FreeRet dense 的取层设置。

## 脚本配置

在对应任务的编码、索引、search 和三阶段脚本中使用相同设置，例如：

```bash
USE_LENS_SPARSE=true
USE_STC_SPARSE=false
LENS_ALGORITHM=kmeans
LENS_N_CLUSTERS=8000
LENS_SEED=42
LENS_N_INIT=1
LENS_MAX_ITER=300
LENS_BATCH_SIZE=4096
LENS_CACHE_ROOT="/root/gpufree-share/lens_heads"
```

支持的脚本包括 `encode_adaptive_views_{ir,tbpr,fused}.sh`、`sparse_index_adaptive_{ir,tbpr,fused}.sh`、`search_adaptive_views_{ir,tbpr,fused}.sh` 和 `three_stage_retrieval_pipeline_adaptive_views_{ir,tbpr,fused}.sh`，以及现有普通单任务和三阶段批量包装脚本。模型、数据目录、视角选择、编码精度及其他编码参数也需一致。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `USE_LENS_SPARSE` | `false` | 使用质心输出头；与 STC 互斥 |
| `LENS_ALGORITHM` | `kmeans` | 标准 K-means；可选近似实现 `minibatch_kmeans` |
| `LENS_N_CLUSTERS` | `8000` | 输出簇维度，必须得到指定数量的非空簇 |
| `LENS_SEED` | `42` | 聚类随机种子 |
| `LENS_N_INIT` | `1` | 初始化尝试次数，使用 K-means++ |
| `LENS_MAX_ITER` | `300` | 最大迭代次数 |
| `LENS_BATCH_SIZE` | `4096` | 仅 MiniBatchKMeans 使用 |
| `LENS_CACHE_ROOT` | 空 | 默认 `${SPARSE_OUTPUT_ROOT}/lens_heads` |

初始化次数、种子和迭代上限是工程默认值，均进入缓存配置。Qwen 大词表的标准 K-means 预计算可能较慢；MiniBatch 是单独记录指纹的可选近似实现。

## 编码与查询

- 聚类仅处理真实 tokenizer ID，排除特殊、空白、零输出向量；NaN/Inf 或非有限范数报错。数字正常参与聚类，不采用 STC 的数字独立维度规则。`IS_FILTERED` 等原词项过滤在建簇前应用。
- 表征前向期间临时切换到质心 head；正常结束和异常退出均恢复原 head。输入 embedding、模型词表配置、视角选择、提示词、一次拼接前向及视角掩码保持原样。
- 每个视角对簇 logits 执行 `log1p(ReLU)`，选 Top-K 簇，逐项乘 100 并取整，再按 `sum/max/replace` 合并。开启均值时最后进行整数除法，丢弃非正整数。
- 此模式的 `SPARSE_LENGTH` 表示每视角保留的簇数。词项为 `lens_000001` 等簇编号，不再套用原 token 的文本过滤。
- Dense 的取层与归一化逻辑不变，原生、类 FreeRet、完整 FreeRet 提示词模式均可组合使用。三阶段的重排使用恢复后的原始 head。
- 本次不支持 overlap 编码或 Top-K 实验入口，显式请求会报错。

## 缓存与产物

编码脚本在启动 DeepSpeed/NCCL 前，先由 CPU 单进程预计算或复用质心缓存。也可手动预计算：

```bash
python -B src/lens_sparse.py build \
  --model-name-or-path /path/to/checkpoint --dtype float16 \
  --lens-algorithm kmeans --lens-n-clusters 8000 \
  --lens-cache-root /path/to/lens_heads
```

缓存位于 `<LENS_CACHE_ROOT>/<checkpoint名称>/<配置指纹前16位>/`，包含：

- `manifest.json`：模型、适配器、tokenizer、精度、过滤、聚类设置及内容校验。
- `head.npz`：float32 质心、可选 bias、token ID 与簇映射。
- `cluster_members.json`：每簇的全部 token 成员。

每份编码产物附带不可变的 `lens_head/` 副本。Search 和三阶段查询读取候选库锁定的质心，不能通过改变缓存目录替换该 head，也不会重新聚类。原中心必须同时用于查询和候选。

编码、查询缓存及检索结果以独立 `_lens_<算法>_k<维度>_<指纹>` 标记隔离。LENS、STC 和普通产物不能混用；切换到 LENS 后需重新编码并建立 `collection/index`。关闭 LENS 时，原模式的路径和指纹保持原样。

`DRY_RUN=true` 不加载模型权重、不构建质心、不写文件。完整运行仍需在实际模型和数据上评估 Recall，本适配不保证复现论文的训练后指标。
