# 保留现有提示词，切换 FreeRet 同位置的 dense 表征

七个编码、搜索、重排和三阶段入口，以及对应的批量脚本，都支持：

```bash
USE_FREERET_DENSE="${USE_FREERET_DENSE-false}"
```

默认 `false`，沿用原来的表征、路径和指纹。设为 `true` 时，只改变 dense 隐藏状态的来源。

本文下述“提示词/sparse 不变”指 `USE_FREERET_PROMPT_CONSTRAINTS=false`；额外约束模式见文末。

支持的基座是 `llava_llama`、`llava_mistral`、`qwen2`、`qwen25`、`qwen3`。其他基座开启时会报不支持，关闭时保持原行为。

## 具体变化

开启后，在最后一个解码层 `post_attention_layernorm` 的输入处截取状态，即该层 attention 加残差之后、该层 MLP 之前，与本仓库 `freeret_retrieval.py` 的位置一致。

现有提示文本、图像处理、attention mask、取 token/EOL 的位置、单/多视角聚合、归一化和 dense 维数均保留。整个模型仍完成原来的前向计算，词表 logits 不变；sparse 的激活、top-K、权重、过滤和合并规则也不变。

PMI、overlap 和视角选择缓存继续使用原来的算法及缓存键。该开关不改变最终选择的视角，也不修改重排的 A/B、Yes/No 或 caption likelihood 评分。`rerank.py` 中的前置 dense/hybrid 检索会使用新 dense。

## 一套实验要保持同一个开关

例如 TBPR overlap：

```bash
# 重新编码候选，获取新 dense；sparse 数值保持原算法。
USE_FREERET_DENSE=true bash scripts/encode_adaptive_views_overlap_tbpr.sh

# 若编码时已复用有效索引，则无需再次建立该索引；批量索引会跳过已有索引。
USE_FREERET_DENSE=true bash scripts/sparse_index_adaptive_views_overlap_tbpr.sh

USE_FREERET_DENSE=true bash scripts/three_stage_retrieval_pipeline_adaptive_views_overlap_tbpr_multi.sh
```

IR、Fused 入口同样使用这个变量，模型、数据、稀疏参数、图像模式、PMI 和 overlap 参数仍需与编码时一致。普通 `encode.sh`、`search.sh`、`rerank.sh` 及原生 Qwen、LLaVA、三阶段批量入口也支持该开关。可直接在脚本配置区赋值，不必 `export`。

直接调用 Python 时使用 `--use_freeret_dense true`；三阶段入口也接受 `--use-freeret-dense true`。显式 `false` 与省略开关一致。

## 路径与复用

所有新模式的编码产物、索引、查询产物、日志及检索/重排结果，其实验目录都追加：

```text
_dense_freeret_pre_mlp
```

例如：

```text
原 dense：  .../adaptive_views/<视角指纹>/corpus_0.pkl
新 dense：  .../adaptive_views/<视角指纹>_dense_freeret_pre_mlp/corpus_0.pkl

原 sparse： .../adaptive_views/<视角指纹>/collection/corpus_0.jsonl
新 sparse： .../adaptive_views/<视角指纹>_dense_freeret_pre_mlp/collection/corpus_0.jsonl
新索引：    .../adaptive_views/<视角指纹>_dense_freeret_pre_mlp/collection/index
```

原有 `_llava_base` 等标注保留，新模式标注追加在其后。自定义输出目录也会幂等追加标注，已经带标注的路径不会重复追加。

新的编码配置及检索指纹记录 `dense_representation={source: freeret_pre_mlp, version: 1}`。目录旁的 `.dense_representation.json` 保存模式信息；读取时校验，避免新旧 dense 混用。关闭模式兼容已有无该字段的产物。

编码仍需要前向计算来生成新 dense。完成后，如果对应原模式 sparse 已存在，会校验配置和全部文档的稀疏内容；一致时可独立复制 sparse 文件和有效 Lucene 索引到新目录。复制不使用共享可写链接，不覆盖原模式产物。原模式不存在时正常生成，新索引缺失时再执行索引脚本。

视角选择文件、候选池及其缓存键保持共用，因此视角指纹本身不会仅因 dense 开关而改变。新的 dense 编码指纹和检索结果指纹会改变。已有原模式 dense 不能直接用于开启模式的 dense/hybrid 检索，需要重新编码。

## 为整体和各视角增加 FreeRet 约束

配套脚本增加 `USE_FREERET_PROMPT_CONSTRAINTS`，默认 `false`。七个 Python 入口均支持 `--use_freeret_prompt_constraints true`，三阶段入口也支持 `--use-freeret-prompt-constraints true`。

| USE_FREERET_DENSE | USE_FREERET_PROMPT_CONSTRAINTS | 行为 |
|---|---|---|
| false | false | 原隐藏层、原提示词 |
| true | false | FreeRet 取值节点、原提示词 |
| true | true | FreeRet 取值节点、输入前一次任务描述、每个表征指令前任务约束 |
| false | true | 启动时报错 |

开启后，现有聊天前缀之后、实际图文输入之前，加入对应数据集的 `task_description`，每条样本只出现一次，结尾统一为一个换行。整体 dense 和 `disassembleeol_separate + concat` 的每个视角 sparse 块仍使用对应查询/候选角色的 FreeRet `query_instruction` / `target_instruction`。原 Summary 的措辞、标点、尾部空格和所在的对话包装保留。例如 Flickr 图像输入的用户正文：

```text
You are required to determine if the provided caption matches the given image.
<image>
Based on the given image, output exactly one word that follows these rules:
 - Output a single lowercase noun that best captures the semantics of the image.
 - You may use world knowledge and paraphrase if needed.
 - The word must be a complete, standalone word — not a function word, preposition, symbol, or fragment.
 - Favor the caption of the image if possible.
Summary above image in one word: 
```

各视角共享开头的任务描述，不再各自重复描述；仍使用相同任务约束，最后一行保留该视角的原指令，例如 `Summary the tools and equipment in above image in one word: `。约束中的引用标签转换为普通文本，避免增加图像占位符。图文输入、聊天/系统前缀和第三阶段的重排提示词保持原样，原生 `freeret_retrieval.py` 的提示词也不变。

支持 Flickr、COCO、CUHK-PEDES、ICFG-PEDES、RSTPReid、WebQA、ReMuQ 和 LLaVA。提示词目录与原 FreeRet 程序共用；其中 TBPR、Flickr、ReMuQ 等继续使用项目已有的适配版本。没有对应约束版本的数据集开启时会报错。

编码、索引和检索使用一致的两个开关，例如：

```bash
USE_FREERET_DENSE=true USE_FREERET_PROMPT_CONSTRAINTS=true bash scripts/encode_adaptive_views_overlap_tbpr.sh
USE_FREERET_DENSE=true USE_FREERET_PROMPT_CONSTRAINTS=true bash scripts/sparse_index_adaptive_views_overlap_tbpr.sh
USE_FREERET_DENSE=true USE_FREERET_PROMPT_CONSTRAINTS=true bash scripts/three_stage_retrieval_pipeline_adaptive_views_overlap_tbpr_multi.sh
```

也可以直接修改脚本开头的两个变量默认值。新模式实验目录后缀是 `_dense_freeret_pre_mlp_freeret_constraints_v2`，已有图像模式后缀保留。编码配置和模式文件记录约束版本、任务描述及放置规则的指纹，查询缓存也区分新旧提示词。旧 v1 产物保留；批量扫描索引时跳过并汇总，显式指定旧目录时会提示重新编码。关闭约束开关时，原提示词、路径和缓存键不变。

任务描述和约束会改变 dense 和 sparse 数值，需要重新编码并建立对应索引；不会复制无约束模式或旧 v1 的 sparse/索引。普通 adaptive/PMI 的选项分类提示词、采样方案和选择缓存可继续复用；overlap 的稀疏探测使用相同的新表征提示词，其配置和缓存随版本更新，需要重新选视角。显式提供旧约束版本的 overlap 选择文件时会提示不兼容。

## 完整 FreeRet 提示词模式

所有配套编码、search、rerank、普通/adaptive 三阶段检索及索引脚本增加：

```bash
USE_FREERET_DENSE=true
USE_FREERET_FULL_PROMPT=true
```

`USE_FREERET_FULL_PROMPT` 默认 `false`。为 `true` 时要求 FreeRet dense 已开启，并优先使用完整提示词；`USE_FREERET_PROMPT_CONSTRAINTS` 不再重复添加任务描述或约束。该旧开关的真假不影响完整模式的路径和有效配置。为 `false` 时，沿用已有两个开关的全部行为。

整体 dense 使用当前 `freeret_retrieval.py` 的完整编码提示词：原任务描述、带标签的输入、规则、相同模型的聊天包装和 assistant generation prefix。保留目录中的原始标点与换行，不追加项目的 Summary 或人工 EOL。TBPR、Flickr、ReMuQ、LLaVA 仍以本仓库已有的 FreeRet 适配模板为准。

多视角 sparse 只替换语义规则中的目标。例如 Flickr 图像 clothing 视角：

```text
 - Output a single lowercase noun that best captures the clothing of the <image>.
```

其他规则不变，层级视角的 `such as` 部分完整保留。每个视角采用独立完整提示词前向，在有效序列末尾取 FreeRet 隐状态或 logits；稀疏激活、过滤、top-K、权重和合并保持现有算法。完整模式会增加前向次数。

overlap 继续分别抽取纯文本和纯图像来计算重合率，不恢复图文配对。此步骤使用单模态输入和规则变体，去掉缺失模态字段，不插入假图片；实际编码、在线查询仍使用该任务完整的 FreeRet 输入。视角分类、PMI、采样和实际重排评分提示词保持原样。

新模式目录后缀为 `_dense_freeret_pre_mlp_freeret_full_prompt_v1`。元数据及查询缓存记录完整模板、聊天包装、视角规则和单模态 overlap 规则。需要重新编码和建立索引，不能借用旧 Summary 或 constraints 模式的产物。普通 adaptive 视角选择缓存可以复用；overlap 需要重新选择视角。未支持的任务或无法确定语义的自定义视角会明确报错。

例如 TBPR overlap：

```bash
USE_FREERET_DENSE=true USE_FREERET_FULL_PROMPT=true bash scripts/encode_adaptive_views_overlap_tbpr.sh
USE_FREERET_DENSE=true USE_FREERET_FULL_PROMPT=true bash scripts/sparse_index_adaptive_views_overlap_tbpr.sh
USE_FREERET_DENSE=true USE_FREERET_FULL_PROMPT=true bash scripts/three_stage_retrieval_pipeline_adaptive_views_overlap_tbpr_multi.sh
```

普通编码可使用相应的 `encode_freeret_<任务>.sh`、`sparse_index_freeret_<任务>.sh` 和 `three_stage_retrieval_freeret_<任务>.sh`。Python 参数为 `--use_freeret_full_prompt true`，三阶段入口也接受 `--use-freeret-full-prompt true`。
