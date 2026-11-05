# 稀疏重合率视角选择

入口是 `src/encode_adaptive_views_overlap.py`，启动脚本是
`scripts/encode_adaptive_views_overlap.sh`。三类任务另有独立的批量启动脚本，见下表。
支持 Qwen2/2.5/3-VL、LLaVA-NeXT Llama/Mistral，以及 ITR、TBPR、混合模态任务。

## 启动

```bash
USE_VIEW_OVERLAP_SELECTION=true \
MODEL_FAMILY=qwen25 DATASET=flickr \
VIEW_SELECTION_MODALITY=both \
VIEW_SELECTED_COUNT=5 VIEW_OVERLAP_THRESHOLD=0.1 \
VIEW_OVERLAP_SAMPLES=20 VIEW_OVERLAP_SEED=42 \
bash scripts/encode_adaptive_views_overlap.sh
```

`USE_VIEW_OVERLAP_SELECTION=false`（默认）在新建选择时沿用原有概率 top-5 流程；复用时使用文件中锁定的视角集合。
也可以编辑新脚本中对应配置。`VIEW_SELECTED_COUNT` 支持 1～20，默认 5。
`VIEW_SELECTION_ROUNDS` 等原参数继续控制第一步的多轮概率评估。

`USE_VIEW_PMI=true` 可与 overlap 同时启用。每个模型/数据集选择过程先计算一次不带 examples 的先验，
每轮使用 `softmax(log p(v|examples) - log p0(v))` 校正概率，再按原模态权重聚合。
overlap 的种子视角、探测顺序和次数相同的排序均使用校正后的全局概率。
最终仍按 overlap 入选次数选择指定数量，校正不作用于语料向量或检索分数。

| USE_VIEW_PMI | USE_VIEW_OVERLAP_SELECTION | 新建选择的流程 | 选择版本 |
|---|---|---|---|
| false | false | 原概率前五 | 12 |
| false | true | 原概率排序＋overlap 投票 | 16 |
| true | false | PMI 校正概率前五 | 14 |
| true | true | PMI 校正排序＋overlap 投票 | 17 |

PMI 未设置时，新选择默认关闭；指定已有 `VIEW_SELECTION_FILE` 时继承其记录。
显式 true/false 与文件冲突会报错。四种模式的缓存和编码产物指纹隔离，普通版本 12 和 PMI 版本 14 保持不变。
新 overlap 配置包含 `rule="candidate_fraction_lt"`，该标记参与选择缓存及视角指纹。
旧 overlap 版本 13、PMI＋overlap 版本 15 不再兼容，必须重新选择视角、编码并建库；旧文件不会自动删除或改写。

## 多模型、多数据集批量编码

| 脚本 | 默认数据集 | 默认候选编码模态 | 五个模型的任务数 |
|---|---|---|---|
| `scripts/encode_adaptive_views_overlap_ir.sh` | flickr、coco | image、text | 20 |
| `scripts/encode_adaptive_views_overlap_tbpr.sh` | CUHK-PEDES、ICFG-PEDES、RSTPReid | image | 15 |
| `scripts/encode_adaptive_views_overlap_fused.sh` | webqa、remuq、llava | corpus | 15 |

三个批量脚本默认 `USE_VIEW_OVERLAP_SELECTION=true`，原单任务脚本仍默认 false。
配置区完整列出模型路径、数据与输出目录、GPU/精度、稀疏编码、图像处理、采样及 overlap 参数，
实际 DeepSpeed 命令也在各自文件内展开。可直接用普通变量赋值修改，环境变量可覆盖默认值。
默认五个基座为 Qwen2/2.5/3-VL 和 LLaVA-NeXT Llama/Mistral，GPU=1、float16、稀疏长度 40、
TBPR `origin_type`、图像模式 `anyres`；概率阶段 20 轮，默认 `VIEW_EXAMPLE_MODE=auto` 按上下文预算填充示例。
`VIEW_EXAMPLE_MAX_TOKENS=0` 使用模型上限，也可设置更小预算；文本和图像分别填充，保留一个回答 token。
固定模式仍支持每轮文本 8 条、图像 2 张；复用旧产物时设置 `VIEW_EXAMPLE_MODE=fixed` 并匹配原采样参数。
自动策略和预算参与缓存及产物指纹，索引/三阶段自动定位需同步设置。
详见 [自动示例说明](adaptive_views.md#按上下文长度自动填充示例)。
overlap 默认选 5 项、阈值 0.1、每个启用模态最多 20 条、独立种子 42。
`LLAVA_IMAGE_MODE=base` 仅允许 LLaVA-NeXT 模型族。

```bash
# 同时启用 PMI 和 overlap，批量编码两个基座、两个数据集。
MODEL_FAMILIES="qwen25 qwen3" DATASETS="flickr coco" USE_VIEW_PMI=true \
bash scripts/encode_adaptive_views_overlap_ir.sh

# 预检和打印任务，不加载模型、不写编码及日志目录。
DRY_RUN=true bash scripts/encode_adaptive_views_overlap_tbpr.sh

# 混合模态候选库编码；WebQA=t2it，ReMuQ/LLaVA=it2t。
MODEL_FAMILY=qwen25 DATASETS="webqa remuq llava" \
bash scripts/encode_adaptive_views_overlap_fused.sh
```

`MODEL_FAMILY`、`DATASET`、`ENCODE_TYPE` 的非空单值覆盖对应复数列表。
全局 `MODEL_PATH` 只允许一个模型族；多模型使用逐模型路径变量。`OUTPUT_DIR` 只允许一个编码任务。
`VIEW_SELECTION_FILE` 只允许一个模型/数据集组合，可跨其编码模态复用。
使用该变量时，PMI、选择模态及四个 overlap 参数未设置就继承文件；需要覆盖时显式赋值，冲突会报错。
批量调度先预检整个矩阵，缺少模型配置或数据标注的任务跳过；无可运行任务返回 2，实际编码失败立即停止。
运行 `bash scripts/encode_adaptive_views_overlap_ir.sh --help` 查看变量说明。
内部调度复用 `adaptive_encode_batch.py --encoder overlap`，普通批量入口默认仍为 `ordinary`。

## 筛选方法

1. 使用与普通/PMI 编码共用的 A～T 共 20 个视角和 last-token logits 概率评估，得到一次全局概率排序。
2. 使用独立随机流重新采样，样本可与概率阶段重合。`text` 只采文本，`image` 只采图像，
   `both` 分别采样文本与图像，各取至多 `VIEW_OVERLAP_SAMPLES` 条，不组成图文对。
3. 每条样本先接受概率最高的视角，按现有隔离 EOL 编码、激活转换、token 过滤与权重量化方法
   提取 top-30 稀疏向量。过滤、去重和零权重剔除后可能少于 30 个有效 token。
4. 按全局概率顺序逐一探测其他视角。重合率为
   `当前候选与已接受视角 token 并集的交集大小 / 当前候选的有效 token 数`，不按权重加权。
   分母是过滤、去重后的实际数量，不固定为 30。**严格小于阈值才接受**，等于或高于阈值则丢弃。
   空候选跳过，比率为 `N/A`，不做除零运算。只在接受时更新并集，权重沿用配置的聚合方法。
5. 每条样本达到指定视角数量或全部候选处理完就停止。统计各视角的入选次数，最终按
   **次数降序 → 全局概率降序 → A～T 顺序**，选出 `VIEW_SELECTED_COUNT` 个视角。

最高概率视角的向量为空时，该样本不投票。若获得过投票的不同视角不足目标数量，
保存 `view_selection_diagnostics.json` 并停止，不写入不完整的语料编码结果；可提高阈值放宽筛选后重试。
默认阈值仍为 0.1；阈值 0 时种子之后没有候选可入选，阈值 1 时完全重合的候选仍不入选。
所有样本种子向量均为空时，应检查 logits 或过滤配置。

探测阶段固定 top-30；**最终完整语料编码仍由 `SPARSE_LENGTH` 等原参数控制**，默认 40。
模型、图像分辨率、模态和最终视角数量均沿用现有编码实现。

当前候选池将原 N 的 `environment, geographical locations, weather or places` 合并到 C，
新 N 为 `emotions and facial expressions`（情绪与面部表情）。G 扩展为
`wearing of people or objects, including clothing and accessories`，R 改为 `tools and equipment`；
这些调整继续保留。本次 H 改为 `time`，J 改为 `written words and symbols`（文字与符号），
删除独立的 objects 候选；TBPR type 的原物品位置仅在参考表中对应到 A，不增加 A 的历史别名。
相对版本 9/10/11，其他 18 项保持不变。
R 的历史来源原文保留用于校验，不参与选项提示词拼接。
C 合并原 C/N 的历史来源；J、N 是人工新增项，`aliases` 和 `sources` 均为空。
完整选项及任务对应表见 [adaptive_views.md](adaptive_views.md)。WebQA 时间映射为 H（I、M、H、E、A），
ReMuQ 为 A、M、E、I、H；TBPR type 的原 objects 位置参考记录为 A（F、D、A、G、E），
A 仍使用 `main people or objects`。已有版本 12/14 中该位置为 null 的选择文件仍可只读复用，
不会自动改写，无需重新编码或建索引。候选池保持不变；旧 overlap 版本 13/15 受新筛选规则影响，需要重新生成。LLaVA 两侧仍为 A、B、C、D、E。
这些对应表不限制可选范围，所有任务仍对全部 20 项评分，新 N 也参与排序和 overlap 探测。

## 结果、复用与后续流程

overlap 模式默认实时打印每个样本的模态、编号、ID，以及每个被测试视角的描述、概率、
sparse tokens/权重、重合率（与已有并集的交集数 / 当前候选的有效 token 数）、阈值和接受/丢弃结果。
同时打印更新后的已选视角及 token 并集与聚合权重；token 按权重降序排列，初始视角的 overlap 标为 `N/A (seed)`。
空种子向量会标注拒绝原因。每条样本完成后，打印全部 20 个视角的累计入选次数。
全部采样结束后，打印最终视角，以及完整 20 项次数排名、全局概率和最终入选标记；
排名依次按入选次数、全局概率降序，同分沿用 A～T 顺序。候选不足时也打印已有统计并标注 `incomplete selection`。

两个入口共用的前置分类阶段，每轮都会打印样本 ID、token 数、最高概率视角及全部 20 项概率，
按概率降序、同分按 A～T 顺序展示，保留 8 位小数。分类结束后、overlap 探测开始前，
打印文本均值、图像均值和加权总分的完整排名；未启用的模态显示 `N/A`。
关闭 overlap 开关时，同样输出这些分类日志与最终选中的五个视角。
日志仅由主进程即时刷新输出，无需增加参数，也不新增日志文件或改变选择 JSON 与指纹。
复用显式指定的或自动缓存的选择文件时，打印复用路径、已保存的聚合排名、锁定视角与已有统计，
不重放逐轮日志或重新计算分类与 overlap。

成功后打印选择 JSON、dense/sparse 输出目录与索引命令。选择 JSON 保存全局概率、实际采样 ID、
每个被测试视角的稀疏 token/权重、交集数、候选分母、重合率、入选记录、token 并集、总次数和最终视角。
新算法配置与采样 ID 参与视角指纹，目录继续采用 `adaptive_views/<指纹前16位>`。

复用时设置 `VIEW_SELECTION_FILE=/实际路径/view_selection.json`。
开启新模式并复用新选择文件时，未指定的数量、阈值、样本数、随机种子继承文件；
显式配置冲突会报错。复用文件不会重新评估概率或重做探测。
当前普通、overlap、PMI、PMI＋overlap 选择文件版本分别为 12、16、14、17。旧版本 1～11 与当前候选池不兼容，13/15 与当前 overlap 规则不兼容，
需使用新选择缓存或文件路径，重新选择、编码和建索引；旧文件与产物保留，不自动改写或迁移。

按编码完成时打印的 sparse 目录建索引：

```bash
python -B src/adaptive_views.py index --sparse-root "/实际路径/adaptive_views/指纹" --threads 16
```

也可使用三个只处理 overlap 系列结果的批量索引入口：

| 任务 | 脚本 | 默认数据集 | 默认候选模态 |
|---|---|---|---|
| IR | `scripts/sparse_index_adaptive_views_overlap_ir.sh` | flickr、coco | image、text |
| TBPR | `scripts/sparse_index_adaptive_views_overlap_tbpr.sh` | CUHK-PEDES、ICFG-PEDES、RSTPReid | image |
| Fused | `scripts/sparse_index_adaptive_views_overlap_fused.sh` | webqa、remuq、llava | corpus |

默认遍历五个模型族和所有匹配的检查点、指纹；每个指纹独立输出到 `collection/index`。
脚本配置使用普通变量赋值，也可以通过环境覆盖默认值。非空 `MODEL_FAMILY`、`DATASET`、`ENCODE_TYPE`
覆盖对应复数列表。TBPR 可指定 `ENCODE_TYPE=text`，已有 `origin_type/type` 变体均可处理。
`SPARSE_OUTPUT_ROOT` 必须与编码时一致；默认 `<项目>/sparse_output`，split 默认 test，线程默认 16。
`THREADS` 优先于兼容别名 `SPARSE_THREADS`，`PYTHON_BIN` 选择当前 Pyserini 环境的解释器。

```bash
# IR：两个模型、两个数据集，同时为 overlap 与 PMI+overlap 建库。
MODEL_FAMILIES="qwen25 qwen3" DATASETS="flickr coco" \
bash scripts/sparse_index_adaptive_views_overlap_ir.sh

# TBPR：只为 PMI+overlap 建库。
PMI_MODE=on bash scripts/sparse_index_adaptive_views_overlap_tbpr.sh

# Fused：只预检非 PMI 的 overlap 结果，不写索引或输出目录。
PMI_MODE=off DRY_RUN=true bash scripts/sparse_index_adaptive_views_overlap_fused.sh
```

如果清单 `selection_file` 保存的旧绝对路径失效，索引会按完整 `selection_fingerprint` 查找原文件的副本。
默认优先搜索 `<SPARSE_OUTPUT_ROOT>/view_selection`；如果它不存在，或单任务输入是指纹叶目录，
则定位产物祖先目录中最近的 `view_selection`。也可显式指定一个有界的 JSON 搜索目录：

```bash
VIEW_SELECTION_ROOT="/实际路径/视角文件备份" DRY_RUN=true \
bash scripts/sparse_index_adaptive_views_overlap_ir.sh

python -B src/adaptive_views.py index --sparse-root "/实际路径/adaptive_views/指纹" \
  --view-selection-root "/实际路径/视角文件备份" --threads 16
```

三个新脚本配置区均列出 `VIEW_SELECTION_ROOT`，空值表示自动定位；批量入口的相对路径按项目根目录解析。
单任务 Python 的 `--view-selection-root` 优先于同名环境配置，相对路径按当前工作目录解析。
搜索目录只在原引用文件不存在时使用，每批每个目录仅扫描一次；支持副本更名，多个有效同内容副本按路径排序选择第一份。
日志打印旧引用与实际匹配路径。目录名中的选择缓存指纹和编码产物视角指纹本来就不同；
恢复必须同时匹配完整选择指纹、文件内容校验值和视角指纹，并继续校验模型、数据集、视角、模态及 PMI/overlap 配置。
仅有相同的短指纹前缀、选中视角或视角指纹均不能代替原选择文件。
查找忽略无关的损坏候选；原引用文件存在但损坏或不一致时仍直接报错，不用其他副本绕过校验。
找回仅供本次运行使用，不改写清单、选择文件或编码目录；DRY_RUN 也不写入文件。
找不到有效副本时会报告搜索范围和期望完整指纹，需要恢复原 JSON 或重新编码。Sparse 的 `collection/corpus_*.jsonl` 分片仍必须完整。

新入口固定使用共享索引程序的 `--overlap-only`，通过清单引用的选择文件及指纹识别模式，
无需填写视角指纹或设置 `USE_VIEW_OVERLAP_SELECTION`。`PMI_MODE=all`（默认）处理 16/17，
`off` 仅处理 16，`on` 仅处理 17。普通 12 和仅 PMI 的 14 会跳过，汇总为 `skipped_non_overlap`；
版本或候选池不兼容的选择文件（含旧 13/15）计入 `skipped_incompatible`，打印实际版本、候选数量和差异标签。
这些统计仅包含模型、数据集、模态、split 和 PMI 筛选范围内的结果。
已有有效索引跳过；损坏或不一致的清单、选择文件及目标分片会在建库前报错。
全部没有可用目标时返回错误，全部已建库时成功退出。实际建库失败立即停止，保留原退出码。
需要 CPU、Pyserini 和 Java，不加载模型权重或运行 GPU 编码。原三个通用索引脚本继续处理全部兼容模式。

然后使用已有 adaptive 三阶段脚本，传入刚刚打印的路径，并保持模型、分辨率、稀疏配置一致：

```bash
VIEW_SELECTION_FILE="/实际路径/view_selection.json" \
DENSE_CORPUS_DIR="/实际dense目录" SPARSE_CORPUS_DIR="/实际sparse目录" \
MODEL_FAMILY=qwen25 DATASET=flickr \
bash scripts/three_stage_retrieval_pipeline_adaptive_views.sh
```

该步骤读取锁定的视角集合编码查询，继续检索与重排；无需再次开启视角选择。
批量检索时，三个 `three_stage_retrieval_pipeline_adaptive_views_{ir,tbpr,fused}.sh` 使用
`USE_VIEW_PMI=true USE_VIEW_OVERLAP_SELECTION=true` 精确定位组合模式缓存，采样与编码参数须一致。
三个 `sparse_index_adaptive_{ir,tbpr,fused}.sh` 的 `PMI_MODE=on` 同时匹配版本 14 和 17 的完成产物。
索引入口校验清单 `selection_file` 引用的选择文件及指纹；需保留原文件或完整匹配的副本。单任务指定旧 13/15 结果会报错，
批量扫描会打印原因，跳过所有版本或候选池不兼容的结果，并在 `skipped_incompatible` 汇总；全部不可用则返回错误。
直接指定不兼容选择文件进行编码、检索或单任务建库仍报错；如需使用该实验，应以一致的当前代码重新选择、编码和建库。
不会自动修改版本、指纹或清单引用；JSON 损坏、文件缺失、指纹不一致及有效目标分片损坏仍中止整批预检。
复用及检索不会重新计算先验、分类或 overlap 探测。

## 三类任务的 overlap 批量检索入口

新增三个独立脚本，配置与实际 DeepSpeed 命令完整展开，使用普通变量赋值并集中传入子进程：

| 任务 | 脚本 | 默认范围 |
|---|---|---|
| IR | `scripts/three_stage_retrieval_pipeline_adaptive_views_overlap_ir_multi.sh` | 五个模型、flickr/coco、双向，共 20 个任务 |
| TBPR | `scripts/three_stage_retrieval_pipeline_adaptive_views_overlap_tbpr_multi.sh` | 五个模型、三个 PEDES 数据集、文本搜图像，共 15 个任务 |
| Fused | `scripts/three_stage_retrieval_pipeline_adaptive_views_overlap_fused_multi.sh` | 五个模型、WebQA/ReMuQ/LLaVA，共 15 个任务 |

WebQA 为文本搜图文；ReMuQ/LLaVA 为图文搜文本，均读取对应的 corpus 编码库。
新入口要求 `USE_VIEW_OVERLAP_SELECTION=true`（默认）；关闭时明确报配置错误，普通任务使用原入口。
`USE_VIEW_PMI=false` 默认定位版本 16，设为 true 定位版本 17；一次运行只使用指定的一套编码与选择配置。
原三个通用批量检索入口的默认值及定位行为保持不变。

```bash
# IR：两个模型、两个数据集，默认双向检索 overlap 产物。
MODEL_FAMILIES="qwen25 qwen3" DATASETS="flickr coco" \
bash scripts/three_stage_retrieval_pipeline_adaptive_views_overlap_ir_multi.sh

# TBPR：与 PMI+overlap 编码和索引配套。
USE_VIEW_PMI=true bash scripts/three_stage_retrieval_pipeline_adaptive_views_overlap_tbpr_multi.sh

# Fused：仅预检；视角文件备份可以位于其他目录。
DRY_RUN=true VIEW_SELECTION_ROOT="/实际路径/视角文件备份" \
bash scripts/three_stage_retrieval_pipeline_adaptive_views_overlap_fused_multi.sh
```

完整配置包括检索/重排模型路径、数据与输出根目录、GPU/精度、稀疏设置、图像处理、概率采样、
overlap 数量/阈值/采样，以及三阶段深度、alpha、重排等参数。非空单值覆盖复数列表；所有编码相关参数必须与编码时一致。
默认 float16、稀疏长度 40、anyres、TBPR origin_type；概率阶段 20 轮、auto、预算 0、种子 42，
fixed 模式按每轮文本 8/图像 2 抽样；示例模式与预算须匹配编码端。
overlap 默认 5 项、阈值 0.1、每模态 20 条、种子 42。
what_caption_generation 的 modify_type 自动逐任务设置：IR 图搜文、ReMuQ/LLaVA 使用 modify_single，其余方向使用 no。

原视角缓存存在时直接校验读取；缺失时，新入口固定启用共享调度器的 `--recover-missing-selection`。
恢复仅检查当前模型、数据集、候选模态及完整编码参数对应目录中的完成清单。
清单原引用必须关联当前计算出的 `view_selection/<模型>/<数据集>/<缓存键前16位>/view_selection.json`；
引用前面的旧绝对路径前缀可不同，但不能仅凭相同视角或相似目录猜测实验。
匹配清单必须具有正确的配置指纹、规范产物目录及完整选择指纹；若有多个不同的完整选择指纹，明确报歧义错误。
确定唯一实验后，通过 `VIEW_SELECTION_ROOT` 或自动定位的缓存目录按完整指纹恢复原 JSON，继续校验采样、split、PMI/overlap、
dense/sparse 完成清单、分片数及 Lucene 索引一致性。恢复的实际路径传给检索程序，不修改旧清单或缓存，不重新选择视角或建库。

没有对应清单、有效原 JSON、编码库或索引的任务会跳过并汇总；全部不可运行时返回错误。
损坏、指纹不一致及歧义会在整批预检时报错；实际检索失败立即停止并保留退出码。
DRY_RUN 只加载配置和 processor 做预检，不加载模型权重、不写结果；OVERWRITE 默认 false。
结果继续按视角/编码指纹、stage1/stage2 深度和检索指纹隔离；恢复后的实际 JSON 路径按现有规则记录在 batch_job.json 并参与检索结果指纹。
## 候选选项顺序打乱

overlap 编码支持 `SHUFFLE_VIEW_OPTIONS=true`，新字母的概率会还原到原视角后再进行 overlap 筛选和频次统计。默认关闭。映射记录、PMI 配合及运行标识的使用见 [候选视角顺序打乱](adaptive_view_shuffle.md)。

