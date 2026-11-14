# 通用五视角：prompt_shared_5

这是独立的新模式，默认仍为 `prompt_5`。原有模式、原有模板、自适应视角选择规则不变；
不需要修改 `encode_adaptive_views.py`。新模式只用于标准编码、搜索、重排入口和标准三阶段入口中的检索向量编码。
`relevant`、caption generation 等重排评分提示词不变。

## 指定方式

TBPR / fused 的六个配套入口默认已设为本模式，见 [配套启动脚本](shared_view_launchers.md)。

在 `scripts/encode.sh`、`scripts/search.sh`、`scripts/rerank.sh` 或
`scripts/three_stage_retrieval_pipeline.sh` 的配置区将 `PROMPT_TYPE` 的默认值改成 `prompt_shared_5`，
也可通过环境变量覆盖：

```bash
PROMPT_TYPE=prompt_shared_5 bash scripts/encode.sh
PROMPT_TYPE=prompt_shared_5 bash scripts/search.sh
PROMPT_TYPE=prompt_shared_5 bash scripts/rerank.sh
PROMPT_TYPE=prompt_shared_5 bash scripts/three_stage_retrieval_pipeline.sh
```

这些脚本的模型、数据集和产物目录仍按各自配置区指定。搜索和重排的编码参数必须与候选库一致。
其他已有支持 `PROMPT_TYPE` 的标准任务启动脚本也可传递该值。

直接运行 `encode.py`、`search.py` 或 `rerank.py` 时，在原命令中指定：

```bash
--prompt_type prompt_shared_5 --eol_type disassembleeol_separate --calculate_type concat
```

标准三阶段入口使用 `--prompt-type prompt_shared_5`。稀疏 Stage 1 使用 `sparse_concat`；
也可使用原有 CLIP/BLIP 或预计算 run。在线 `sparse_manual` 对应 separate 编码，不支持新模式。
新模式只支持 `ir/tbpr/t2it/it2t` 和 `disassembleeol_separate + concat`，其他组合提前报错。

## 五个视角与提示词

严格沿用 Flickr/COCO 当前五视角的顺序和英文措辞；`<input>` 为 `sentence`、`image` 或 `image and sentence`：

| 顺序 | 视角 | 提示词 |
|---|---|---|
| 1 | 人物或物体 | `Summary the people or objects in above <input> in one word: ` |
| 2 | 关系 | `Summary the relations, such as belongings or spatial position, between main people or objects in above <input> in one word: ` |
| 3 | 环境 | `Summary the environment, weather or places in above <input> in one word: ` |
| 4 | 动作 | `Summary the actions or movements of main people or objects in above <input> in one word: ` |
| 5 | 外观 | `Summary the appearance, such as color, material, decoration and so on, of main people or objects in above <input> in one word: ` |

全局摘要为 `Summary above <input> in one word: `。保留各模型的对话格式、图像占位符、
输入顺序、全局摘要加五视角的 concat 布局，以及原有向量合并方法。

| 任务 | 查询 | 候选 |
|---|---|---|
| Flickr/COCO | 根据检索方向使用 sentence 或 image | 对应的 image 或 sentence |
| TBPR | sentence | image |
| WebQA / t2it | sentence | image and sentence |
| ReMuQ、LLaVA / it2t | image and sentence | sentence |

新模式下，TBPR 的 `origin_type/type/type_1/type_2` 不再改变这组固定视角。
fused 不额外添加时间、事件等旧视角。已有 FreeRet 约束开关仍按原方式包装指令。

## 产物

沿用现有 `prompt_type` 路径及配置指纹，新产物位于 `prompt_shared_5` 命名空间；
有提示词清单的产物会记录实际的新提示词。切换到新模式需重新编码候选库并建立对应索引，
检索时使用匹配的 `PROMPT_TYPE` 和产物目录。旧模式的路径和指纹保持原行为。
