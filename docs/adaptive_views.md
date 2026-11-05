# 数据集视角选择：编码、建库与三阶段检索

新增入口为 `src/encode_adaptive_views.py` 和
`src/three_stage_retrieval_pipeline_adaptive_views.py`。它们完整复制原入口后增加自选视角模式，
不修改原文件。关闭该模式时继续使用原有编码与检索行为。

支持 LLaVA-NeXT Llama3/Mistral，以及 Qwen2、Qwen2.5、Qwen3-VL Instruct。
ITR 对应现有 `ir`，TBPR 对应 `tbpr`，fused-modal 对应 `t2it/it2t`；新模式不包含 CIR。

## 选择方法

从 `template.py` 中 ITR、TBPR origin/type、WebQA、ReMuQ 和 LLaVA 融合任务的默认人工五视角
提取描述，补充文本／图像 `prompt_6` 的第六项（原因）和 `mistral_generation` 的第七项（动植物），
得到旧版 20 项来源目录，再将近义表述合并；版本 4 从已有模板追加六个视角 O～T，
版本 5 精简 P／Q／R 的模型可见描述。当前版本将原 C/N 的环境描述合并到 C，
并将 N 定义为人工新增的“情绪与面部表情”；clothing/accessories 从 R 的模型描述移到 G，
R 改为工具与设备。本次将时间从 J 移到 H，删除独立的 `objects` 候选，
并将 J 定义为人工新增的“文字与符号”，最终候选池仍为 20 项。原始固定任务模板保持不变，
不导入补充来源的整组模板，也不在运行时生成新视角。

下表“旧编号”均指版本 2 的 A～T；当前候选池编号仍为 A～T。本次 H/J 调整以版本 9/10/11 的候选池为基准，
其余 18 项保持不变，之前的 C/G/N/R 调整继续保留。

| 新编号 | 视角 | 统一英文表述 | 旧编号 |
|---|---|---|---|
| A | 主要人物或物体 | `main people or objects` | A、Q |
| B | 人物或物体间的关系 | `relations, such as belongings or spatial position, between main people or objects` | B |
| C | 环境、地理位置、天气或场所 | `environment, geographical locations, weather or places` | C、R |
| D | 动作或运动 | `actions or movements of main people or objects` | D、I |
| E | 外观和装饰细节 | `appearance of main people or objects, including color, material, decoration, pattern and shape` | E、L、P |
| F | 性别 | `gender of people` | F、H |
| G | 穿戴，包括服装和配饰 | `wearing of people or objects, including clothing and accessories` | G、K |
| H | 时间 | `time` | O |
| I | 事件和活动 | `events and actions` | M |
| J | 文字与符号 | `written words and symbols` | 无，版本 12 人工新增 |
| K | 行为原因或动机 | `reason why main people or objects might be in this position and doing this thing` | S |
| L | 动植物 | `animals and plants` | T |
| M | 环境、地理位置及建筑 | `environment and geographical locations, such as architecture, buildings or structures` | N |
| N | 情绪与面部表情 | `emotions and facial expressions` | 无，版本 9 人工新增 |
| O | 群体或群组 | `group` | 无，版本 4 新增 |
| P | 年龄和族群 | `age and ethnicity` | 无，版本 4 新增、版本 5 精简 |
| Q | 姿态 | `poses` | 无，版本 4 新增、版本 5 精简 |
| R | 工具与设备 | `tools and equipment` | 无，版本 4 新增、当前版本调整 |
| S | 职业和活动 | `professions and activities` | 无，版本 4 新增 |
| T | 人物体型和身高 | `body type, height of person` | 无，版本 4 新增 |

**原 C/N 的环境来源统一归到新 C；M 继续保留建筑相关描述；新 N 表示情绪与面部表情。**
动作 D 与事件 I、外观 E 与穿着 G 仍分别保留；独立物品候选已删除。
TBPR type 的原物品位置仅在参考对应表中记为 A，不把 objects 加到 A 的历史别名，也不挂到 H/J。
H 的历史来源为时间。导出的候选池中，
`aliases` 保存历史编号、原文及来源模板，`sources` 汇总该合并项的全部来源；
人工新增 J、N 的 `aliases=[]`、`sources=[]`。O～T 的 `aliases.label` 为 `null`，保留完整描述及
两侧来源，不伪造历史编号。当前 P 不包含 gender，Q 不包含 actions/movements；
G 明确包含 clothing/accessories，R 的模型可见描述仅为 tools and equipment。

六项来源均是已有文本／图像模板中的指定条目；原文用于校验和溯源，模型实际使用上表的统一描述：

| 编号 | 模板来源（文本／图像各一份） | 条目位置（从 1 开始） |
|---|---|---|
| O | ITR `llama_generation` | 第 7 项 |
| P | ITR `mistral_generation` | 第 1 项 |
| Q | ITR `mistral_generation` | 第 2 项 |
| R | ITR `mistral_generation` | 第 3 项 |
| S | ITR `mistral_generation` | 第 6 项 |
| T | TBPR `mistral_generation` | 第 4 项 |

P／Q／R 的来源原文分别为 `gender, age and ethnicity`、`actions, movements and poses`、
`objects, tools, clothing and accessories`。这些原文仅保留在来源记录中，选择提示词及正式编码指令
使用 `age and ethnicity`、`poses`、`tools and equipment`，文本、图像和融合输入一致。
R 的历史来源原文仍保留 clothing/accessories，用于溯源和原模板校验，不会作为 R 的选项描述再次输入模型。

原始任务的五视角按以下顺序映射；这是来源说明，自选模型仍可从全部 20 项中选择：

| 原始任务／侧别 | 新编号顺序 |
|---|---|
| ITR：Flickr／COCO，文本与图像 | A、B、C、D、E |
| TBPR：origin_type，文本与图像 | A、F、D、G、E |
| TBPR：type，文本与图像 | F、D、A、G、E |
| WebQA：t2it，查询与候选 | I、M、H、E、A |
| ReMuQ：it2t，查询与候选 | A、M、E、I、H |
| llava 数据集，图文查询 | A、B、C、D、E |
| llava 数据集，文本候选 | A、B、C、D、E |

TBPR type 的原 `objects` 位置参考对应到 A（`main people or objects`），
新生成选择 JSON 的 `task_views` 在原位置记录 `A`，两侧均为 F、D、A、G、E。
这仅调整来源参考表；A 的描述、历史别名、20 项候选池、视角指纹和编码目录规则不变，其他任务映射也不变。
已有版本 12/14 文件中该位置的历史 `null` 仍可只读复用，不自动改写，原有编码和索引可继续使用。
旧 overlap 版本 13/15 因筛选规则更新不再兼容，需要重新选择、编码并建索引。
新生成 JSON 仍按原规则计算自身指纹；原模板的 objects 原文保留用于校验，其他保留视角缺少映射仍会报错。
J、K、L、N 以及 O～T 是补充候选，不属于上述原始任务的默认五视角。
该对应表只用于原模板溯源，IR、TBPR 和混合模态任务的自适应选择仍覆盖全部 20 项。
可在不加载模型时查看完整候选池及合并来源：

```bash
python -B src/adaptive_views.py pool
```

视角选择支持 `--view_selection_modality text|image|both`，脚本使用环境变量 `VIEW_SELECTION_MODALITY`。
新选择默认 `both`；未指定模式且复用已有选择文件时，继承文件中记录的模式。

| 模式 | 选择阶段输入及执行顺序 | 最终视角概率 |
|---|---|---|
| `text` | 仅抽取纯文本组，完成 R 轮文本评分 | 文本 R 轮概率均值 |
| `image` | 仅抽取纯图像组，完成 R 轮图像评分 | 图像 R 轮概率均值 |
| `both` | 先完成全部 R 轮文本评分，再完成全部 R 轮图像评分 | 文本均值和图像均值各占 0.5 |

每个启用模态每轮随机抽取一组完整样本，组成一个选择提示词。
不打乱句子内部词序或图像像素，不给出图文配对关系或召回标签。
选择器使用被测编码模型本身，不使用 DSPy 或外部模型。
纯文本选择不读取图像文件；纯图像选择只提供图像，不混入 caption、问题或其他样本文本。
双模态的两路独立对全部 20 项评分，图像评分不依赖文本结果，也不做文本预筛选。

模型对最后有效输入位置的 logits 进行评分，只取 A～T 字母 token，在这些选项内部做 softmax。
`both` 模式的概率聚合为：

```text
score(v) = 0.5 × mean(text_probabilities[:, v])
         + 0.5 × mean(image_probabilities[:, v])
```

按得分取前五，同分按字母顺序；选中顺序同时用于查询与候选库。
默认每个启用模态 20 轮、种子 42，`VIEW_EXAMPLE_MODE=auto` 按模型上下文预算自动决定每轮示例数量。
每轮先生成可复现的随机排列，选取能够容纳的示例前缀；轮内不重复，轮间允许重复。
`VIEW_EXAMPLE_MODE=fixed` 保留原采样方式，默认文本每轮 8 条、图像每轮 2 张，样本不足时使用实际数量。
采样模型使用 eval/inference 模式，随机性来自样本抽取和顺序。
默认从当前编码 split 抽样，可用 `VIEW_SELECTION_SPLIT` 指定其他 split。
仅构建和校验启用模态的样本池；启用的文本池合并该任务查询/候选侧文本，启用的图像池来自其中实际存在的图像。
缺少启用模态时明确报错，不切换模式。未启用模态的组大小不会参与采样、校验或缓存标识。

```bash
VIEW_SELECTION_MODALITY=text bash scripts/encode_adaptive_views.sh
VIEW_SELECTION_MODALITY=image bash scripts/encode_adaptive_views.sh
VIEW_SELECTION_MODALITY=both bash scripts/encode_adaptive_views.sh
```

模式只控制视角选择；后续候选库编码、查询编码和重排仍按原任务所需模态执行。

### 按上下文长度自动填充示例

`encode_adaptive_views.py` 和 `encode_adaptive_views_overlap.py` 及配套的八个编码脚本共用自动示例逻辑。
`VIEW_EXAMPLE_MAX_TOKENS=0` 默认读取模型文本配置的 `max_position_embeddings`（必要时读取顶层配置）；
正整数预算与模型上限取较小值，并预留一个回答 token。例如设置 8192 时，输入最多 8191 个有效 token。
文本和图像两路独立填充；预算计算使用真实 processor 处理后的完整输入，包含聊天模板、示例编号、
Candidate perspectives、回答前缀以及按实际图像分辨率展开的视觉 token，不使用字符数估算，也不截断输入。

长度探测先指数扩展、再二分定位可容纳的示例前缀；确认追加下一条超限后停止，或在样本池耗尽时停止。
探测仅做预处理，最后接受的输入直接复用于一次模型评分。基础提示或第一条示例已超限时报告长度、预算和样本 ID。
每轮日志和 `view_selection.json` 的 `rounds` 记录真实示例数、ID、输入长度、预算、停止原因及下一条加入后的长度。
PMI 先验仍不含示例；overlap 的独立探测样本数仍由 `VIEW_OVERLAP_SAMPLES` 控制。
自动模式下 `VIEW_TEXTS_PER_ROUND`、`VIEW_IMAGES_PER_ROUND` 不决定示例数量。

```bash
# 默认按模型声明的上下文上限填充
bash scripts/encode_adaptive_views_tbpr.sh
# 下调预算以控制显存占用；与 PMI、层级视角和 overlap 等开关可组合
VIEW_EXAMPLE_MAX_TOKENS=8192 bash scripts/encode_adaptive_views_overlap_ir.sh
# 复用旧固定数量实验时，编码、索引和自动定位检索均指定 fixed
VIEW_EXAMPLE_MODE=fixed VIEW_TEXTS_PER_ROUND=30 VIEW_IMAGES_PER_ROUND=10 \
bash scripts/encode_adaptive_views_tbpr.sh
```

Python 对应参数为 `--view_example_mode auto|fixed` 和 `--view_example_max_tokens 0`。
自动策略版本、请求预算、有效预算及模型上下文上限参与选择缓存、视角和编码产物指纹。
固定模式保留旧采样载荷和指纹，旧 JSON 按 fixed 解释；与显式选择文件的模式或预算冲突会报错。
索引和三阶段批量脚本需与编码使用相同的模式及预算；读取显式完成清单的入口从文件中验证策略。
模型上限是声明的序列容量；显存不足时可主动下调预算。Qwen 像素预算仍独立控制图像展开长度。
候选字母必须是单 token，否则报错。

### PMI 先验校正

普通编码脚本可开启 `USE_VIEW_PMI=true`，对应 Python 参数 `--view_selection_pmi true`：

```bash
USE_VIEW_PMI=true VIEW_SELECTION_MODALITY=both bash scripts/encode_adaptive_views.sh
```

保留原选择指令、A～T 候选和模型对话模板，删除整个 Text/Image examples 段，
不输入样本文本、图片或图像占位符。新选择仅由 rank 0 额外前向一次，得到两种模态共用的先验 `p0(v)`。
每轮校正为：

```text
PMI(v, examples) = log p(v | examples) - log p0(v)
corrected_probabilities = softmax(PMI)
```

计算使用候选 logits 的 `log_softmax`，避免原概率下溢后再取对数或除零。
先对每轮校正结果归一化，再取轮次均值；`both` 仍将两种模态的均值各占 50%，最后选前五。
采样种子、样本和候选顺序均不改变。PMI 仅用于视角选择，不校正后续向量、检索分数或重排分数。

日志打印全部 20 项先验概率，逐轮显示原概率、先验、PMI 和校正概率，排名及 top1 使用校正概率。
版本 14（PMI）和 17（PMI＋overlap）的选择 JSON 在 `calibration` 保存方法配置、无样本提示词、token 数、先验概率与对数概率；
各轮新增 `text_raw_probabilities` / `image_raw_probabilities` 和 `text_pmi_scores` / `image_pmi_scores`。
原 `text_probabilities` / `image_probabilities`、模态均值和最终 `scores` 在该模式下均表示校正后的概率，
所有概率和分数列表仍按 A～T 的候选池顺序保存。

未设置开关时，新选择默认关闭；显式复用文件时继承其中的状态。显式 `true` 或 `false` 与文件冲突时会报错，
应取消开关以继承，或指定新的选择文件。开启 PMI 时使用独立选择缓存和产物指纹，
即使选中同样的五个视角也不会覆盖未校正产物；关闭时使用普通模式版本 12。
当前候选池下的普通/overlap/PMI/PMI＋overlap 文件分别为版本 12/16/14/17。复用版本 14、17 文件时不重新计算先验或采样，现有三阶段检索直接读取锁定视角，
无需添加 PMI 检索参数。部署时须同步更新共享的 `adaptive_views.py` 和 `adaptive_view_encoding.py`。
overlap 入口也支持此开关。只开 PMI 时使用版本 14；两个开关同时开启时使用版本 17，
将每轮 PMI 校正概率的聚合排名用于 overlap 探测，并用它打破入选次数相同的平局。
最终视角由 overlap 入选次数决定，可选 1～20 项，不要求等于概率前五。详见 [overlap 使用说明](adaptive_views_overlap.md)。
overlap 重合率为“与已接受视角 token 并集的交集数 / 当前候选有效 token 数”，严格小于阈值才接受。
默认阈值 0.1；候选不足时提高阈值放宽筛选。新规则标记 `candidate_fraction_lt` 参与缓存及产物指纹。

## 第一步：选择视角并编码候选库

启动器 `scripts/encode_adaptive_views.sh` 直接运行新编码入口。示例为 Qwen2.5 文搜图：

```bash
export MODEL_FAMILY=qwen25
export MODEL_PATH=/root/autodl-fs/checkpoints/Qwen-Qwen2.5-VL-7B-Instruct
export DATASET=coco IR_QUERY_TYPE=text DATA_ROOT=/root/autodl-tmp
export DTYPE=float16 SPARSE_LENGTH=30 TBPR_TYPE=type
export DENSE_OUTPUT_ROOT=/root/autodl-fs/dense_output
export SPARSE_OUTPUT_ROOT=/root/autodl-fs/sparse_output

VIEW_SELECTION_ROUNDS=20 VIEW_EXAMPLE_MODE=auto VIEW_EXAMPLE_MAX_TOKENS=8192 \
GPUS=2 bash scripts/encode_adaptive_views.sh
```

`MODEL_FAMILY` 支持 `qwen2|qwen25|qwen3|llava_llama|llava_mistral`；默认 qwen25。
`MODEL_PATH` 可覆盖实际检查点；选择和编码必须使用同一模型、processor、精度与像素配置。
`DATASET` 默认 flickr，方向控制与现有脚本一致。行人任务自动使用图像候选库；
WebQA/ReMuQ/LLaVA 自动使用 corpus。`ENCODE_TYPE` 可单独覆盖。

新模式固定 `disassembleeol_separate+concat`、非手动 sparse。仍保留一个整体视角用于 dense，
选中的五个分解视角用于 sparse。输入图像只放在共享上下文中一次，分解视角使用隔离注意力。
第三阶段仍用 `what_caption_generation`，不增加五视角重排。

选择阶段默认由 rank 0 实时打印每轮模态、轮次、样本 ID、实际 token 数、最高概率视角，
以及全部 20 个视角的标签、描述和概率。概率按降序展示，同分沿用 A～T 顺序，保留 8 位小数。
每轮完成立即刷新输出，无需新增参数。分类结束后打印完整聚合排名，包括 `text_mean`、
`image_mean` 和加权 `score`；未启用的模态显示 `N/A`。选择文件保存后、语料编码开始前，
打印其路径和最终选中的五个视角。日志不改变选择结果、JSON 内容或指纹，也不额外生成日志文件。

复用显式指定的或自动缓存的选择文件时，打印 `reused locked selection`、实际路径、
已保存的聚合排名和锁定视角，不重放逐轮结果或重新分类。

编码结束仍会打印：

```text
View selection: .../view_selection.json
Adaptive dense: .../adaptive_views/<视角指纹>/
Adaptive sparse: .../adaptive_views/<视角指纹>/
Build index: python -B src/adaptive_views.py index --sparse-root "..."
```

自动选择文件位于 sparse 输出根目录下的 `view_selection/<模型>/<数据集>/<选择配置指纹>/`。
保存 `view_selection.json` 和 `view_pool.json`。选择清单包含模型、数据集、采样参数、每轮样本 ID、
实际 token 数、各轮概率、启用模态的均值、最终得分和 top-5；`selection_modality` 和 `modality_weights`
记录选择模式及实际权重，`task_views` 保存原始任务各侧的有序新编号映射。
未启用模态的概率、均值、token 数为 `null`，每轮样本 ID 为空列表，采样组大小为 `null`。
多卡仅 rank 0 选择和保存，完成后广播给其他编码进程。

编码产物沿用原目录规则，在实验叶目录后增加 `adaptive_views/<视角指纹>/`，不覆盖旧方法产物。
模式同时纳入自动选择缓存标识和视角指纹；三种模式即使选中相同的五项，也会使用独立的选择缓存及编码目录。
该目录内保存 `corpus_<rank>.pkl`、`collection/corpus_<rank>.jsonl` 和
`adaptive_views_manifest.json`。所有分片完成后才写完成清单。
重跑同一自选配置会替换对应分片，将已有新旧索引备份到 collection 外。

如果需要编码另一个模态或同一数据集的另一个 split，显式复用刚才的选择文件：

```bash
VIEW_SELECTION_FILE=/实际路径/view_selection.json \
DATASET_SPLIT=val bash scripts/encode_adaptive_views.sh
```

显式选择文件只读复用，不重新采样；采样设置以文件记录为准。未指定模式时自动继承文件模式，
显式指定不同模式时在加载模型权重前报错。三阶段检索直接读取选择文件中的模式，不增加模式启动参数。
模型、processor、数据集或视角池
不匹配时会输出具体字段差异。改变采样设置并希望重新选择时，省略该文件变量，生成新的配置目录。

普通模式使用选择文件版本 12，overlap 模式使用版本 16，PMI 模式使用版本 14，PMI＋overlap 使用版本 17；
它们共用 H 为时间、J 为文字与符号的 20 项候选池，并保留此前 C/G/N/R 的调整。
版本 1～11 的旧选择文件均不能直接复用，旧 overlap 版本 13/15 也不能复用，
即使手动修改版本号也会因候选池不匹配而报错。旧文件和产物保留，不自动迁移；
使用新候选池须重新选择视角、编码和建索引，PMI 先验也会重新计算。
选择文件、产物指纹和完成清单明确记录当前配置，不通过改写旧元数据代替重新采样。
重新选择时省略 `VIEW_SELECTION_FILE` 或指定新的文件路径，再执行编码和建库。
新候选池会生成新的自动选择目录和编码产物指纹，原有选择文件与产物保留。

### 多模型、多数据集批量编码

以下入口按“模型→数据集→编码模态”的列表顺序串行执行。三个 `.sh` 各自完整列出配置和实际 DeepSpeed 命令；
共享 Python 模块只负责预检、任务遍历及结果汇总，每个任务回到对应脚本的内部单任务分支执行：

| 脚本 | 默认数据集 | 默认候选编码模态 |
|---|---|---|
| `scripts/encode_adaptive_views_ir.sh` | `flickr coco` | `image text` |
| `scripts/encode_adaptive_views_tbpr.sh` | `CUHK-PEDES ICFG-PEDES RSTPReid` | `image` |
| `scripts/encode_adaptive_views_fused.sh` | `webqa remuq llava` | `corpus` |

默认模型列表为 `MODEL_FAMILIES="qwen2 qwen25 qwen3 llava_llama llava_mistral"`，
支持修改脚本配置区，也支持环境变量覆盖。IR 的 `ENCODE_TYPES` 表示要编码的候选模态，
默认同时生成图像库、文本库；查询方向由调度器按每个编码模态确定，不使用全局 `IR_QUERY_TYPE`。
TBPR 也可显式指定 `ENCODE_TYPES="image text"`，混合模态仅接受 `corpus`。

```bash
# 先预检、查看任务和实际启动命令，不加载模型、不写输出
MODEL_FAMILIES="qwen25 qwen3" DATASETS="flickr coco" DRY_RUN=true \
bash scripts/encode_adaptive_views_ir.sh

# IR 两个模型、两个数据集，启用 PMI；每个任务使用两张卡
MODEL_FAMILIES="qwen25 qwen3" DATASETS="flickr coco" GPUS=2 USE_VIEW_PMI=true \
DENSE_OUTPUT_ROOT=/root/autodl-fs/dense_output SPARSE_OUTPUT_ROOT=/root/autodl-fs/sparse_output \
bash scripts/encode_adaptive_views_ir.sh

# 多数据集 TBPR 与混合模态编码
MODEL_FAMILIES="qwen2 qwen25 qwen3" bash scripts/encode_adaptive_views_tbpr.sh
MODEL_FAMILIES="llava_llama llava_mistral" bash scripts/encode_adaptive_views_fused.sh

# 各模型可使用不同的自定义检查点目录，目录名称须互不相同
MODEL_FAMILIES="qwen25 qwen3" DATASET=flickr \
MODEL_PATH_QWEN25="/models/my qwen25" MODEL_PATH_QWEN3="/models/my qwen3" \
bash scripts/encode_adaptive_views_ir.sh
```

非空的 `MODEL_FAMILY`、`DATASET`、`ENCODE_TYPE` 优先于对应列表；如终端之前导出了单值变量，
需取消该变量才能恢复多值遍历。列表使用空白分隔，空列表、重复项、不支持的配置在启动前报错。
三个批量入口仅接受 `-h/--help` 参数，其他设置通过环境变量或配置区修改。
`--run-one` 是调度器内部使用的单任务分支，不用于用户手动配置批量运行。

每个脚本的配置区完整展开模型与任务、五种检查点目录、输入输出目录、运行资源、稀疏编码、图像处理、
视角采样及 PMI 参数。可以直接在对应 `.sh` 修改默认值；单值模型、数据集、模态和输出覆盖默认留空，
避免缩小批量范围。配置使用普通 Bash 变量赋值，无需逐项 `export`；启动 Python 调度器时，
脚本通过一次 `env` 集中传参，直接修改脚本中的配置同样生效。
PMI 和选择模态提供普通赋值的注释示例，仅在已设置时传入，未设置时继续继承已有选择文件。
脚本后半部分还逐项写出与 `encode_adaptive_views.sh` 相同的固定参数：
`adaptive_views`、`eol_type=disassembleeol_separate`、`calculate_type=concat`、`sparse_manual=false`、
`prompt_type=prompt_5`、`random_view_seed=42`、`cir_type=type`，以及精度、布尔开关的条件处理。
其中 `random_view_seed` 是原编码兼容字段，视角选择采样使用独立的 `VIEW_SELECTION_SEED`。

`CHECKPOINT_ROOT` 默认 `/root/autodl-fs/checkpoints`，沿用单任务脚本中的五种检查点目录名称。
可分别设置 `MODEL_PATH_QWEN2`、`MODEL_PATH_QWEN25`、`MODEL_PATH_QWEN3`、
`MODEL_PATH_LLAVA_LLAMA`、`MODEL_PATH_LLAVA_MISTRAL`；全局 `MODEL_PATH` 优先，但仅允许选中一个模型族。
每个模型族在单次运行中对应一个检查点。`LLAVA_IMAGE_MODE=base` 仅允许选择 LLaVA-NeXT 模型族。

`DATA_ROOT` 默认 `/root/autodl-tmp`，`DATASET_SPLIT` 默认 `test`，`VIEW_SELECTION_SPLIT` 默认跟随编码 split。
`GPUS`、`NUM_NODES` 控制当前任务使用的资源，模型之间不并行。
原脚本的精度、稀疏、像素预算、采样和 PMI 环境设置继续有效。
未指定 `USE_VIEW_PMI` 或 `VIEW_SELECTION_MODALITY` 时保留原来的选择文件继承规则，不强行设为关闭或双模态。

批量入口会在启动第一个模型前检查整个任务矩阵，核对检查点的 `config.json` 和编码、选择 split 所需的标注文件。
不逐一读取图像或检查权重内容。缺少模型配置或标注的组合跳过并汇总路径；所有组合均不可运行时返回非零状态。
非法配置、冲突的模型目录名称、输出根目录为文件等问题会提前报错。
真实编码任务失败后立即停止并保留退出码和已完成的任务。

**每个可运行组合都重新编码，不依据已有完成清单跳过。** 相同产物目录中的旧分片由原编码程序替换，
已有索引按原规则备份到 `index.stale_*`。自动视角缓存仍会复用；同一模型、数据集和选择设置下，
IR 图像与文本编码共享同一套视角结果，不因切换编码模态重新选择。
`VIEW_SELECTION_FILE` 只允许一个模型与数据集组合使用，但允许该组合编码两个模态；
多个组合运行时省略该变量，分别使用自动选择缓存。

dense/sparse 产物目录与指纹保持原规则。`ENCODE_LOG_ROOT` 默认 `./adaptive_encode_logs`，
各任务的 `output_dir` 为 `<日志根>/<模型族>/<检查点目录名>/<数据集>/<split>/<编码模态>`。
全局 `OUTPUT_DIR` 仅允许一个编码任务使用。所有相对目录按项目根解析，可从其他目录使用脚本绝对路径启动。
`PYTHON_BIN` 可指定当前编码环境解释器，`BASH_BIN`、`DEEPSPEED_BIN` 可覆盖对应程序。
运行结束输出完成、跳过、失败及剩余任务数量；不会自动建索引或执行检索，也不调用 overlap 编码入口。

## 第二步：单独建立稀疏索引

使用编码输出的实际 sparse 目录：

```bash
python -B src/adaptive_views.py index --sparse-root /实际路径/adaptive_views/视角指纹 --threads 16
```

建库仅暂存 `corpus_*.jsonl`，在 collection 外完成，再发布到 `collection/index`；
不会把选择文件或完成清单当作 Lucene 文档，已有目标索引不会被覆盖。

### 多数据集、多模型批量建库

三个入口默认按编码参数定位选择缓存，读取锁定视角的指纹，精确匹配每个模型、数据集、split 和编码模态的产物。
需要编码时的本地模型配置和 processor，不加载模型权重、不读取原数据，也不重新选择视角。
`TARGET_SPARSE_DIR` 留空启用自动定位；非空时只校验和索引指定的完整实验目录。

| 脚本 | 默认数据集 | 默认编码模态 |
|---|---|---|
| `scripts/sparse_index_adaptive_ir.sh` | `flickr coco` | `image text` |
| `scripts/sparse_index_adaptive_tbpr.sh` | `CUHK-PEDES` | `image` |
| `scripts/sparse_index_adaptive_fused.sh` | `webqa remuq llava` | `corpus` |

IR/fused 默认 `MODEL_FAMILIES="qwen2 qwen25 qwen3 llava_llama llava_mistral"`；TBPR 默认 `llava_mistral`。
三个入口均默认 `DATASET_SPLIT=test`、`THREADS=16`。
脚本配置区可直接编辑，也可用环境变量覆盖：

```bash
# 先查看 IR 的待建任务；不调用 Java、不创建索引
MODEL_FAMILIES="qwen25 qwen3" DATASETS="flickr coco" \
SPARSE_OUTPUT_ROOT=/root/autodl-fs/sparse_output DRY_RUN=true \
bash scripts/sparse_index_adaptive_ir.sh

# 批量为 PMI 产物建库
MODEL_FAMILIES="qwen25 qwen3" DATASETS="flickr coco" USE_VIEW_PMI=true \
SPARSE_OUTPUT_ROOT=/root/autodl-fs/sparse_output \
bash scripts/sparse_index_adaptive_ir.sh

# 多数据集 TBPR 和混合模态
MODEL_FAMILIES="qwen2 qwen25 qwen3" DATASETS="CUHK-PEDES ICFG-PEDES RSTPReid" \
LLAVA_IMAGE_MODE=anyres bash scripts/sparse_index_adaptive_tbpr.sh
MODEL_FAMILIES="llava_llama llava_mistral" bash scripts/sparse_index_adaptive_fused.sh
```

非空的 `MODEL_FAMILY`、`DATASET`、`ENCODE_TYPE` 优先于对应列表 `MODEL_FAMILIES`、`DATASETS`、`ENCODE_TYPES`。
列表使用空白分隔，空列表、重复项和不支持的值会报错。IR 支持 `image/text`，TBPR 默认 `image`、也可指定 `text`，
混合模态只接受 `corpus`。每个组合匹配当前明确指定的一套编码配置；`TBPR_TYPE`、稀疏参数和逐模型检查点路径须与编码一致。
自动模式用 `USE_VIEW_PMI`、`USE_VIEW_OVERLAP_SELECTION` 等开关匹配对应实验。
`PMI_MODE=all|on|off` 默认 `all`，用于手动模式或后端扫描模式的筛选；自动模式显式设置时须与 `USE_VIEW_PMI` 一致。
`THREADS` 优先于兼容别名 `SPARSE_THREADS`。`PYTHON_BIN` 可指定当前编码/索引环境的解释器；不需要 GPU。
仅支持 `-h/--help` 参数，其他配置通过环境变量或脚本配置区设置。
可从其他工作目录使用脚本绝对路径启动，相对 `SPARSE_OUTPUT_ROOT` 按项目根目录解析。

整批预检通过后，按模型、数据集、模态的列表顺序执行，每个组合只建立与编码参数一致的指纹索引。
可分别配置普通、PMI、overlap 或 PMI＋overlap；其他配置的历史实验不会参与预检。只有该指纹的 `collection/corpus_*.jsonl`
进入索引，输出固定为同目录的 `collection/index`，不会混合不同指纹。
已有 `segments_*` 的索引视为已建库并跳过；优先检查 `collection/index`，仅当其不存在时才回退旧 `index`。
已有残缺索引、指纹不符、损坏的完成清单或分片数不符会在启动建库前报错，不自动覆盖。
索引会校验完成清单 `selection_file` 引用的选择文件及指纹，需保留原文件或完整匹配的副本。
原引用不存在时，按清单的完整 `selection_fingerprint` 从 `<SPARSE_OUTPUT_ROOT>/view_selection` 查找原 JSON 副本；
默认目录不存在时，定位产物祖先目录最近的 `view_selection`。可设置 `VIEW_SELECTION_ROOT` 指定搜索目录，
单任务 Python 也支持 `--view-selection-root`，该参数优先于环境变量。
命中后仍检查文件校验值、视角指纹、模型、数据集及配置；多个同内容副本按路径排序选择第一份。
每批每个搜索目录仅扫描一次，打印实际使用路径，不修改清单、文件或指纹。原路径存在但内容损坏时仍报错。
找不到匹配原文件时报告搜索范围及完整指纹，需恢复备份或重新编码；相同短目录指纹或相同视角不能代替完整匹配。
后端扫描模式中，选择文件版本不受支持或候选池不一致的产物（包括旧 overlap 13/15）会打印原因并跳过，汇总为 `skipped_incompatible`；
日志包含实际版本、支持版本、实际/期望候选数量及差异候选标签。仅有不兼容结果时返回无匹配错误。
这些产物仍不可用于编码或检索，不会自动改写或迁移；如需使用，应以同一套当前代码重新选择、编码和建库。
JSON 损坏、文件缺失、指纹不符及有效目标分片损坏仍会中止预检，不能作为“不兼容”跳过。
自动定位缺少对应缓存或完成产物时，会报告期望路径并中止整批预检；已全部建库时正常成功退出。
后端扫描模式保留未编码组合的跳过汇总。任一建库失败即停止后续任务并保留退出码，已完成的索引保留。
汇总会列出新建、跳过、试运行计划和缺失组合数量。

若只想为 overlap 系列产物建库，使用新增的
`scripts/sparse_index_adaptive_views_overlap_ir.sh`、
`scripts/sparse_index_adaptive_views_overlap_tbpr.sh`、
`scripts/sparse_index_adaptive_views_overlap_fused.sh`。
这三个入口固定开启 `--overlap-only`：`PMI_MODE=all` 默认处理 overlap 16 和 PMI＋overlap 17，
`off` 仅处理 16，`on` 仅处理 17；普通 12、仅 PMI 14 跳过并汇总为 `skipped_non_overlap`。
它们沿用相同的多模型、多数据集配置、单值覆盖、线程、DRY_RUN 和已有索引跳过规则，
配置使用普通变量赋值并集中传入子进程；无需 GPU、无需重新选择视角。
这些独立 overlap 入口仍按完成清单扫描；三个通用入口已改为按编码参数精确定位。
示例与完整说明见 [overlap 索引说明](adaptive_views_overlap.md)。

## 第三步：三阶段检索与重排

overlap 专用批量入口为 `three_stage_retrieval_pipeline_adaptive_views_overlap_{ir,tbpr,fused}_multi.sh`。
三个脚本要求开启 overlap，默认选择版本 16；`USE_VIEW_PMI=true` 时选择版本 17。
完整展开多模型、多数据集配置与启动命令，并逐任务自动适配检索方向和 modify_type。
它们严格按编码配置定位；缓存缺失时，先匹配关联该缓存键且编码配置一致的清单，再用清单的完整选择指纹找回原 JSON。
可用 `VIEW_SELECTION_ROOT` 指定备份目录；多个不同实验匹配时报歧义错误，不选择最新指纹。
恢复只读，dense/sparse 与索引仍须完整。原入口默认定位与输出行为保持不变。
示例及详细规则见 [overlap 批量检索说明](adaptive_views_overlap.md)。

新入口 `scripts/run_adaptive_views_three_stage.sh` 读取已编码、已建库的产物，
先在启动模型前检查选择文件、dense 分片、完成清单与 Lucene 索引，再调用现有三阶段流程。
它兼容普通 adaptive 和 overlap 选择结果，支持 sparse 产物目录、`collection` 或 `index` 路径。
缺少索引时打印手动建库命令并退出；不会自动编码、重新选择视角或建库。

保持第一步导出的模型、数据集、方向、精度、稀疏参数和像素设置，再填写实际路径：

```bash
export VIEW_SELECTION_FILE=/实际路径/view_selection.json
export DENSE_CORPUS_DIR=/实际路径/dense叶目录/adaptive_views/视角指纹
export SPARSE_CORPUS_DIR=/实际路径/sparse叶目录/adaptive_views/视角指纹

STAGE1_DEPTH=200 STAGE2_DEPTH=200 RERANK_NUM=20 GPUS=2 \
OUTPUT_DIR=./three_stage_adaptive_results/coco_qwen25 \
bash scripts/run_adaptive_views_three_stage.sh
```

查看完整参数示例：`bash scripts/run_adaptive_views_three_stage.sh --help`。
三个必填输入是 `VIEW_SELECTION_FILE`、`DENSE_CORPUS_DIR`、`SPARSE_CORPUS_DIR`；
可选 `STAGE1_SPARSE_INDEX` 默认使用同一个 sparse 产物。
`--view-selection-file`、`--stage1-sparse-index`、`--stage2-dense-index` 和 `--stage2-sparse-index`
可覆盖对应环境变量，重复参数以最后一个为准，检查和检索使用同一组有效路径。
其他参数透传至原启动器，例如 `--stage1-depth 200 --stage2-depth 200 --rerank-num 20`。
相对路径按项目根目录解析，可从任意工作目录使用脚本绝对路径启动。
`PYTHON_BIN` 默认取当前环境的 `python`，可显式指定环境解释器用于产物检查。
默认 `GPUS=1`、两阶段深度均为 `1000`、`RERANK_NUM=5`、`HYBRID_ALPHA=0.5`；
`OUTPUT_DIR`、`--output-dir` 与覆盖保护沿用现有行为，CUDA/cuDNN 库路径继承当前环境。
原入口 `scripts/three_stage_retrieval_pipeline_adaptive_views.sh` 仍可直接使用。

第一阶段使用所选五视角稀疏检索，第二阶段复用查询表示进行 hybrid 排序，第三阶段只对
`RERANK_NUM` 个候选执行原有生成式重排。启动前使用 processor 与文件清单核对候选库，
配置不一致时在查询前向之前失败并列出差异。

默认固定 `HYBRID_ALPHA=0.5`。`SELECT_HYBRID_ALPHA=true` 开启现有 0.1～0.9 扫描；
没有显式验证输入时使用目标集。显式验证索引须成对提供。
验证与目标是同一数据集时可共用选择文件；跨数据集验证须提供 `VAL_VIEW_SELECTION_FILE`。
所有结果、alpha CSV/TREC、查询复用和核心计时定义沿用原流程。
三阶段运行的选择文件及视角指纹另记入 `pipeline_metadata.json`。

### 多模型、多数据集批量三阶段检索

以下完整参数入口以 `three_stage_retrieval_pipeline_adaptive_views.sh` 为基础，各自列出全部配置和实际 DeepSpeed 命令：

| 新脚本 | 默认数据集 | 检索方向／候选编码 |
|---|---|---|
| `scripts/three_stage_retrieval_pipeline_adaptive_views_ir.sh` | `flickr coco` | 图搜文使用 `text` 库；文搜图使用 `image` 库 |
| `scripts/three_stage_retrieval_pipeline_adaptive_views_tbpr.sh` | `CUHK-PEDES ICFG-PEDES RSTPReid` | 文搜行人图像，使用 `image` 库 |
| `scripts/three_stage_retrieval_pipeline_adaptive_views_fused.sh` | `webqa remuq llava` | WebQA 文本搜图文；ReMuQ/LLaVA 图文搜文本，使用 `corpus` 库 |

配置采用普通 Bash 变量赋值，启动 Python 调度器时通过一次 `env` 集中传参；直接编辑脚本即可生效。
五种基座检查点、逐模型重排路径、目录、资源、稀疏参数、图像处理、视角选择和三阶段参数全部展开。
重排模型路径留空时跟随当前检索模型。共享调度器负责精确匹配、预检和遍历，每个任务调用本脚本的内部
`--run-one` 分支执行完整命令；用户入口只接受 `-h/--help`。

三个新入口固定使用 `what_caption_generation`，每个任务独立按检索方向自动决定 `modify_type`：

| 任务与检索方向 | `modify_type` |
|---|---|
| IR：图搜文 | `modify_single` |
| IR：文搜图 | `no` |
| TBPR：文本搜行人图像 | `no` |
| WebQA：文本搜图文 | `no` |
| ReMuQ/LLaVA：图文搜文本 | `modify_single` |

该规则始终启用，不提供手动开关，外部 `MODIFY_TYPE` 或其他脚本的 `AUTO_MODIFY_BY_QUERY_TYPE` 不覆盖它。
调度器通过内部 `JOB_MODIFY_TYPE` 逐任务传递实际值，任务顺序、上一个方向和外部同名变量不会改变选择。
日志会打印 `what_caption_generation modify_type=...`；实际值纳入新入口的检索结果指纹，
并记录到 `batch_job.json` 的 `configuration.modify_type` 及流水线已有的 `pipeline_metadata.json` 的 `stage3.modify_type`。
因此自动生成的结果目录与以前三个入口全部使用 `no` 的结果分开；视角选择缓存、编码指纹和 dense/sparse/index 路径保持一致。
这只启用已有重排修正算法，不额外启用重排先验缓存，与编码阶段的视角 PMI 是两个独立设置。

新脚本默认 `OVERWRITE=false`：结果目录已存在会在预检时报错；设置 `OVERWRITE=true` 后允许重新执行并覆写。
`DRY_RUN=true` 只读取配置、processor 和已有产物，打印指纹与任务，不加载模型权重、不启动检索、不写结果。
默认五种模型、所有数据集和 IR 双向检索对应 20／15／15 个任务；缺失输入的任务会跳过并汇总。

```bash
# IR：两个模型、两个数据集、双向检索，先预览任务
MODEL_FAMILIES="qwen25 qwen3" DATASETS="flickr coco" DRY_RUN=true \
DENSE_OUTPUT_ROOT=/root/autodl-fs/dense_output SPARSE_OUTPUT_ROOT=/root/autodl-fs/sparse_output \
bash scripts/three_stage_retrieval_pipeline_adaptive_views_ir.sh

# TBPR：使用 PMI 编码结果，采样及稀疏配置须与编码时一致
MODEL_FAMILIES="qwen25 qwen3" DATASETS="CUHK-PEDES ICFG-PEDES RSTPReid" USE_VIEW_PMI=true \
bash scripts/three_stage_retrieval_pipeline_adaptive_views_tbpr.sh

# fused：多个模型、多个数据集；显式允许覆写已有结果
MODEL_FAMILIES="llava_llama llava_mistral" DATASETS="webqa remuq llava" OVERWRITE=true \
bash scripts/three_stage_retrieval_pipeline_adaptive_views_fused.sh
```

每次运行按指定编码参数精确匹配已有实验；模式、采样、路径、精度和稀疏配置的要求见下文。
自定义独立验证集及非自动缓存位置的显式产物路径继续使用原单任务入口。

原有三个批量入口继续兼容，保持 `modify_type=no`、原结果指纹及默认 `OVERWRITE=true` 的行为：

| 脚本 | 默认数据集 | 检索方向／候选编码 |
|---|---|---|
| `scripts/three_stage_adaptive_ir_multi.sh` | `flickr coco` | 图搜文使用 `text` 库；文搜图使用 `image` 库 |
| `scripts/three_stage_adaptive_tbpr_multi.sh` | `CUHK-PEDES ICFG-PEDES RSTPReid` | 文搜图，使用 `image` 库 |
| `scripts/three_stage_adaptive_fused_multi.sh` | `webqa remuq llava` | 按任务检索，使用 `corpus` 库 |

默认 `MODEL_FAMILIES="qwen2 qwen25 qwen3 llava_llama llava_mistral"`，IR 默认
`IR_QUERY_TYPES="image text"`。非空 `MODEL_FAMILY`、`DATASET`、`IR_QUERY_TYPE` 分别优先于对应列表。
可编辑三个脚本的配置区，也可通过环境变量覆盖；仅接受 `-h/--help` 命令行参数。

```bash
# 先预览指定编码配置的检索任务：不加载权重、不启动检索、不写结果
MODEL_FAMILIES="qwen25 qwen3" DATASETS="flickr coco" DRY_RUN=true \
DENSE_OUTPUT_ROOT=/root/autodl-fs/dense_output SPARSE_OUTPUT_ROOT=/root/autodl-fs/sparse_output \
bash scripts/three_stage_adaptive_ir_multi.sh

# 检索 PMI 编码结果，选择设置和 sparse 参数须与编码时一致
MODEL_FAMILIES="qwen25 qwen3" DATASETS="flickr coco" USE_VIEW_PMI=true \
VIEW_SELECTION_MODALITY=both SPARSE_LENGTH=40 GPUS=2 \
DENSE_OUTPUT_ROOT=/root/autodl-fs/dense_output SPARSE_OUTPUT_ROOT=/root/autodl-fs/sparse_output \
bash scripts/three_stage_adaptive_ir_multi.sh

# TBPR：固定 hybrid alpha=0.5
MODEL_FAMILIES="qwen25 qwen3" TBPR_TYPE=origin_type \
bash scripts/three_stage_adaptive_tbpr_multi.sh

# 混合模态：检索 overlap 编码结果，并开启当前数据集的 alpha 扫描
MODEL_FAMILIES="llava_llama llava_mistral" USE_VIEW_OVERLAP_SELECTION=true \
VIEW_SELECTED_COUNT=5 VIEW_OVERLAP_THRESHOLD=0.1 VIEW_OVERLAP_SAMPLES=20 VIEW_OVERLAP_SEED=42 \
SELECT_HYBRID_ALPHA=true bash scripts/three_stage_adaptive_fused_multi.sh
```

每次批量运行选择一套编码超参数。普通模式默认 `USE_VIEW_PMI=false`、`USE_VIEW_OVERLAP_SELECTION=false`；
两个开关可同时开启，精确定位版本 17 的编码结果。选择参数默认与编码入口相同：模态 `both`、20 轮、
种子 42、自动填充示例、预算 0（模型上限）；fixed 模式默认每轮文本 8 条、图像 2 张。
`VIEW_SELECTION_SPLIT` 默认跟随 `DATASET_SPLIT=test`。
overlap 默认选 5 个视角、阈值 0.1、每模态采样 20 条、独立种子 42。
`DTYPE`、`REPS_LOC`、稀疏长度、聚合、过滤、像素预算和 LLaVA 图像模式也须与编码一致。

定位过程分两步：先加载本地模型配置和 processor，按编码端同一函数计算选择缓存键，
定位 `SPARSE_OUTPUT_ROOT/view_selection/<模型>/<数据集>/<选择缓存指纹>/view_selection.json`；
再读取并校验锁定视角，用原来的视角指纹和编码目录生成函数得到 dense/sparse 路径。
不会扫描全部实验、选择最新指纹或回退其他配置，也不重新确定视角、计算 PMI 先验或执行 overlap 探测。
单任务的 `VIEW_SELECTION_FILE`、`DENSE_CORPUS_DIR`、`SPARSE_CORPUS_DIR` 和 `STAGE1_SPARSE_INDEX`
由批量调度器针对每个组合重新计算。非自动缓存位置的显式选择文件可继续使用单任务入口。

模型身份包含检查点绝对路径及相关文件的大小、修改时间，选择缓存还包含 `DATA_ROOT` 的绝对路径；
这些信息需要与编码时保持一致。找不到缓存时会打印期望路径，供核对模型、数据路径和超参数。
预演仍需要可用的模型配置、tokenizer/processor 文件及当前检索环境，但不读取模型权重。

模型路径沿用 `CHECKPOINT_ROOT` 和 `MODEL_PATH_<FAMILY>`；可用 `RANKER_MODEL_PATH_<FAMILY>`
单独覆盖各模型族的重排检查点，默认与当前检索模型相同。后缀使用大写，如 `MODEL_PATH_QWEN25`、
`RANKER_MODEL_PATH_LLAVA_LLAMA`。全局 `MODEL_PATH`、`RANKER_MODEL_PATH` 仅允许一个模型族时使用。
`RETRIEVAL_DTYPE`、`RANKER_DTYPE` 默认跟随 `DTYPE`；两阶段 LLaVA 模式也可分别覆盖。

默认两阶段检索深度为 1000，`RERANK_NUM=5`，`HYBRID_ALPHA=0.5`。
`SELECT_HYBRID_ALPHA=true` 使用现有的当前数据集扫描；自定义验证集输入继续使用单任务入口。
查询编码在线执行并在第二阶段复用，第三阶段固定为 `what_caption_generation`。
新入口按上表自动选择 `modify_type`；原单任务脚本和旧批量入口继续使用固定的 `no`。
任务按模型、数据集、方向串行执行；`GPUS`、`NUM_NODES` 控制当前任务。

整批预检完成后才启动第一个任务。缺少模型、标注、对应缓存、编码目录或尚未建立的索引会跳过并汇总；
全部不可运行时报错。已有清单损坏、指纹不符、分片不完整或残缺索引会提前报错。
实际检索失败后立即停止，保留已完成结果并传递原退出码。不会自动编码候选库或建立索引。

原有 `three_stage_adaptive_*_multi.sh` 默认每次重新检索，使用 `OVERWRITE=true`；
新的 `three_stage_retrieval_pipeline_adaptive_views_{ir,tbpr,fused}.sh` 默认 `OVERWRITE=false`。
两类入口都通过该开关控制是否允许覆写，设为 `false` 时已有结果目录将报错。
`ADAPTIVE_OUTPUT_ROOT` 默认 `./three_stage_adaptive_results`，
结果目录为 `<根>/<模型>/<数据集>/<方向>/<split>/adaptive_views/<视角指纹>/<编码指纹>/stage1_<深度>_stage2_<深度>/<检索指纹>`。
目录中的指纹取前 16 位，日志和成功任务的 `batch_job.json` 保存完整指纹、实际输入路径及参数。
检索指纹区分 alpha、重排模型、重排数量等设置；同一配置重跑使用同一路径。
全局 `OUTPUT_DIR` 只允许单个任务使用；相对路径均从项目根解析，支持从其他工作目录启动。

支持通过新入口直接使用 CLI：编码端开关为 `--adaptive_views`，三阶段开关为 `--adaptive-views`，
选择文件分别为 `--view_selection_file` 和 `--view-selection-file`。
三阶段自选模式执行全部阶段，不接受预计算 run；关闭模式后保留原入口的预计算 run 功能。
结果目录默认受原有覆盖保护，显式设置 `OVERWRITE=true` 或 `--overwrite` 才复用。

## 索引按编码参数自动定位

`scripts/sparse_index_adaptive_{ir,tbpr,fused}.sh` 的 `TARGET_SPARSE_DIR` 默认留空，使用后端的
`--exact-encoding` 模式。先用编码时的模型配置和 processor 身份、数据根路径、采样和视角模式
计算选择缓存位置，再读取 `view_selection.json` 中的完整 `views_fingerprint`，复用编码函数计算
稀疏目录并验证完成清单。不会遍历其他实验，因此旧实验的失效引用不会挡住当前配置的新产物。

```bash
# 首先预演，确认打印的缓存路径、完整指纹和输入目录
DRY_RUN=true bash scripts/sparse_index_adaptive_tbpr.sh
DRY_RUN=true bash scripts/sparse_index_adaptive_ir.sh
DRY_RUN=true bash scripts/sparse_index_adaptive_fused.sh
# 确认配置与编码一致后建立索引
bash scripts/sparse_index_adaptive_tbpr.sh
# 也可直接指定编码日志里的 Adaptive sparse 目录
TARGET_SPARSE_DIR="/path/to/adaptive_views/<fingerprint>_llava_base" \
bash scripts/sparse_index_adaptive_tbpr.sh
```

TBPR 脚本默认匹配 `llava_mistral / CUHK-PEDES / image / test`、`TBPR_TYPE=type`、
稀疏长度 30、`after_pad`、`LLAVA_IMAGE_MODE=base`、`USE_HIERARCHICAL_VIEWS=true`，
PMI、overlap 和两种 FreeRet 开关默认关闭。采样默认 20 轮、自动示例模式、预算 0、种子 42、
选择模态 `both`；选择 split 默认跟随编码 split。模型和 sparse 根目录默认位于
`/root/gpufree-share` 下，`DATA_ROOT` 默认为 `/root/autodl-tmp`。
IR 和 fused 默认跟随各自的编码脚本：五种模型、稀疏长度 40、`float16`、`after_pad`、`anyres`、
`TBPR_TYPE=origin_type`，层级视角、PMI、overlap、打乱和 FreeRet 开关默认关闭。
IR 默认 `flickr coco` 的 `image text`；fused 默认 `webqa remuq llava` 的 `corpus`，
对应 WebQA=`t2it`、ReMuQ/LLaVA=`it2t`。
两者检查点根目录为 `/root/autodl-fs/checkpoints`，dense/sparse 根目录分别为 `./dense_output` 和 `./sparse_output`，
`DATA_ROOT=/root/autodl-tmp`，采样默认同为 20 轮、auto、预算 0、种子 42。
这些参数须与实际编码相同；旧固定数量产物须设置 `VIEW_EXAMPLE_MODE=fixed`，并同步原来的每轮文本和图像数量。
`MODEL_FAMILIES`、`DATASETS` 支持空格分隔的多模型、多数据集列表，单值变量优先。

`USE_VIEW_PMI`、`USE_HIERARCHICAL_VIEWS`、`USE_VIEW_OVERLAP_SELECTION` 及 overlap 参数、
FreeRet 开关、图像像素预算、编码精度、稀疏参数均在配置区展开。Qwen 应使用 `LLAVA_IMAGE_MODE=anyres`。
选项打乱模式须传入编码时的 `VIEW_SELECTION_RUN_ID`，索引不创建新 run ID。
`PMI_MODE` 是原扫描模式的筛选项；自动模式由 `USE_VIEW_PMI` 定位，两者显式冲突时报错。

自动定位只读取本地模型配置、tokenizer/processor、选择缓存和 sparse 产物，不加载模型权重，
不读取原始数据或 dense 产物，也不重新选择视角。缺少缓存会报告预期路径和关键设置并停止，
不会自动选择最新实验。模型路径、相关文件修改时间和 `DATA_ROOT` 都影响缓存键。
编码使用自定义 `VIEW_SELECTION_FILE` 时，请用 `TARGET_SPARSE_DIR` 指定产物。
手动模式沿用原清单校验和按完整指纹恢复缺失引用的机制，不需要本地模型配置。
有效已有索引仍跳过；预演不建索引。独立 overlap 索引脚本保留原扫描行为。

## 在 Jupyter 中查看三阶段结果

`three_stage_retrieval_pipeline.py` 和 `three_stage_retrieval_pipeline_adaptive_views.py` 的表格结果现在为
`recall_metrics.csv`、`timing_metrics.csv`，替代原来的同名 XLSX 文件。
保留原有字段、召回率的 0～1 数值、各阶段与分量耗时以及逐阶段保存行为；JSON 和已有的 alpha 扫描 CSV 继续输出。
CSV 使用带 BOM 的 UTF-8 编码，不需要 `openpyxl`，结果清单也记录 CSV 文件名。

```python
import pandas as pd

recall = pd.read_csv("/path/to/results/recall_metrics.csv")
timing = pd.read_csv("/path/to/results/timing_metrics.csv")
display(recall, timing)
```

此更改应用于后续运行；已经生成的 XLSX 不会自动转换或删除。

## 验证范围

测试覆盖 A～T 池、其余 17 项不变、P／Q／R 的精简选项正文与三种模态编码指令、
20 项旧来源的唯一归属、O～T 的两侧完整来源与空历史编号、
C/N 环境来源合并、新 N 情绪视角、G 包含服装配饰及 R 工具设备、H 时间与新 J 文字符号、
本次其余 18 项不变、TBPR type 原物品位置参考映射到 A 及历史 null 文件只读兼容、J/K/L/N 及 O～T 选取和编码、
旧版本拒绝、最后有效 token、选项 softmax、等权均值、可重复采样、清单校验、
三种选择模式的概率公式、R/R/2R 次调用及先文本后图像顺序、未启用模态隔离、模式继承与冲突、缓存和产物目录隔离、
多卡选择/失败传播，以及五种小型真实模型的文本、图像、融合编码、变长批次和多图选择。
端到端测试使用实际编码产物和检索替身验证第二阶段不重复前向、第三阶段模板不变。
大型 GPU、NCCL 多卡和真实 Java 索引仍需在服务器上进行小规模验收。
## 候选选项顺序打乱

两个 adaptive 编码入口及配套批量脚本支持 `SHUFFLE_VIEW_OPTIONS=true`；每次启动生成独立运行标识，每个模型/数据集固定一套重新编号的映射。默认关闭。详细行为、保存位置和后续检索方式见 [候选视角顺序打乱](adaptive_view_shuffle.md)。

## Recall@20 输出

共享检索评估默认输出 `Recall@1/5/10/20/100/200/300/400`，dense、sparse、hybrid 及同一流程中的重排结果均包含 Recall@20；Fashion-IQ 保留原来已包含 20 的指标列表。控制台、CSV、JSON 和逐查询诊断同步增加该指标，沿用各程序原有数值单位。正确候选位于第 20 名时计为命中，第 21 名时不计；少于 20 个候选时使用现有返回结果。

Recall@20 仅增加输出，不参与 alpha 优选或 DSPy 优化目标/反馈。原生 FreeRet 会在自定义 `--metrics_ks` 中补入 20 并去重，保留其他 K 值。已有编码和索引可以继续使用；本次修改不回填历史报告。重新运行检索或评估时，沿用原有输出目录及覆盖规则，生成包含 Recall@20 的结果表。

## CSV 结果与 Jupyter

检索、重排、验证集、两阶段、DSPy 和独立 dense 检索原来生成的 XLSX 表格统一改为 CSV，例如 `best.csv`、`dense.csv`、`0_1_0_9_test.csv`。输出目录、原表头、行顺序和数值单位保持不变，使用 UTF-8 BOM 编码，不写额外索引列，无需 `openpyxl` 或其他 Excel 写入引擎。已有三阶段和 Qwen CSV、JSON、TREC 输出保持原样。

```python
import pandas as pd

metrics = pd.read_csv("best.csv")
display(metrics)
```

原来以整数 K 值命名的表头，读入 CSV 后列名是字符串，例如 `metrics["20"]`；原本使用 `Recall@20`、`R@20` 的报告保留这些表头。

相似度统计只生成一个 `dense_sparse_similarity_test.csv`，用首列 `table_type` 区分原来的三个工作表，保留它们的字段并集；某类记录不适用的列为空。

```python
report = pd.read_csv("dense_sparse_similarity_test.csv")
summary = report[report["table_type"] == "summary"]
per_query = report[report["table_type"] == "per_query"]
metadata = report[report["table_type"] == "metadata"]
```

历史 XLSX 文件不会被删除或批量转换。共享写入接口收到旧 Excel 后缀路径时，会写到同名 CSV；相似度统计函数返回实际 CSV 路径。无需重新编码或建立索引，重新运行检索/评估即可生成 CSV。

