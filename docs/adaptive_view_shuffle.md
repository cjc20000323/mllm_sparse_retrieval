# 候选视角顺序打乱

`encode_adaptive_views.py` 和 `encode_adaptive_views_overlap.py` 的配套脚本支持：

```bash
SHUFFLE_VIEW_OPTIONS="${SHUFFLE_VIEW_OPTIONS-false}"
```

默认 `false`，保留原来的候选顺序、选择缓存、指纹和路径。该开关与 `USE_FREERET_DENSE`、`USE_FREERET_PROMPT_CONSTRAINTS` 独立，可以组合使用。

## 开启方式

例如 IR overlap 批量编码：

```bash
SHUFFLE_VIEW_OPTIONS=true bash scripts/encode_adaptive_views_overlap_ir.sh
```

也可以直接修改脚本开头的默认值。单任务 adaptive/overlap 脚本以及 IR、TBPR、Fused 批量脚本均支持。直接调用 Python 使用 `--view_selection_shuffle true`。

每次启动编码会生成新的 16 位运行标识。批量任务中，每个“模型＋数据集”组合分别生成一套排列，同组合的所有选择轮次、文本和图像任务共享这套排列。多 GPU 由 rank 0 同步运行标识。批量调度器通过内部参数把同一个运行标识传给子任务，正常使用时无需手动设置编码内部参数。

排列使用独立随机流，原有 `VIEW_SELECTION_SEED` 和每轮样本采样规则保持不变。跨启动生成新实验；同批次同组合的图像、文本任务可以复用这次实验的选择结果。开启打乱时，`VIEW_SELECTION_FILE` 应留空，或指向独立目录中尚不存在的新文件；不会复用或覆盖已有锁定文件。

## 新字母与原视角

随机排列视角内容后，自上而下重新分配 A–T。例如：

```text
A. tools and equipment       -> 原视角 R
B. main people or objects    -> 原视角 A
...
```

模型按新 A–T 计算概率。程序先将概率还原为原视角顺序，再执行原来的多轮平均、Top-K 和 overlap 选择。最终 `selected` 中仍保存原视角标识及内容，原候选池和任务映射不变。

开启 PMI 时，整个组合只计算一次无 examples 先验。先验和全部条件概率使用同一排列，并在对齐后进行 PMI 校正。overlap 的排序、频次计数和最终视角继续使用原视角标识。

## 保存位置与后续使用

日志会打印运行标识和完整映射。选择缓存目录及编码实验目录增加 `_shuffle_<运行标识>`，原有 `_llava_base` 和 FreeRet 标记保留。例如：

```text
view_selection/<模型>/<数据集>/<缓存指纹>_shuffle_<运行标识>/view_selection.json
view_selection/<模型>/<数据集>/<缓存指纹>_shuffle_<运行标识>/option_mapping.json
.../adaptive_views/<视角指纹>_shuffle_<运行标识>/corpus_0.pkl
.../adaptive_views/<视角指纹>_shuffle_<运行标识>/collection/corpus_0.jsonl
```

`view_selection.json` 的 `option_shuffle` 保存运行标识、映射和映射指纹；`option_mapping.json` 保存相同映射。每一项包含 `display_label`、`canonical_label`、`aspect`。逐轮记录同时保存展示顺序和原视角顺序的概率，以及新字母对应的 top1。PMI 还记录展示顺序的先验与原始条件概率。

索引脚本自动从 manifest 和选择文件识别、验证映射；它会按原规则扫描匹配模型、数据集的各个实验：

```bash
bash scripts/sparse_index_adaptive_views_overlap_ir.sh
```

批量三阶段检索使用编码打印的运行标识定位该次实验，例如：

```bash
VIEW_SELECTION_RUN_ID=0123456789abcdef bash scripts/three_stage_retrieval_pipeline_adaptive_views_overlap_ir_multi.sh
```

TBPR、Fused 和普通 adaptive 的批量检索入口同样支持 `VIEW_SELECTION_RUN_ID`。单任务检索仍可以直接指定 `VIEW_SELECTION_FILE`；程序从文件识别映射，也可用 `VIEW_SELECTION_RUN_ID` 额外核对。检索不会生成新排列，也不会自动选取“最新一次”实验。模型、采样、PMI、overlap、稀疏参数及两个 FreeRet 开关仍应与编码时匹配。

开启打乱后的运行标识、映射均参与选择、视角和编码指纹。即使两次最终选择相同，也会保存为不同实验；查询编码和结果目录依据同一个锁定选择文件区分实验。

使用独立验证集选择 hybrid 权重时，`--val-view-selection-file` 可指向另一运行的选择文件，或未打乱的选择文件。该文件按自身元数据及对应编码产物校验，不继承主实验的运行标识；验证集预检也不会重新标记结果目录。权重扫描报告仍保存在主实验目录。省略该参数时，验证集继续复用主实验的选择文件及运行标识。
