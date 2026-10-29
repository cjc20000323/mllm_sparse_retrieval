# DSPy 受限视角池模式

默认 `DSPY_ASPECT_MODE=free` 保留原有自由优化。设置 `DSPY_ASPECT_MODE=pool` 后，
每轮只允许从 JSON 候选池原样选择 3～6 个不同视角，通过验证集检索表现优化视角子集。
池中可填写单词或完整视角短语；模型不能新增、改写、组合词条或追加后缀。

编辑 `configs/dspy_aspect_pool_ir.json`，例如：

```json
{
  "aspects": ["object", "person", "action", "relation", "scene", "appearance", "color", "material"],
  "initial_aspects": ["object", "relation", "scene", "action", "appearance"]
}
```

`aspects` 至少包含 3 个非空字符串；首尾空白和大小写统一后不能重复。
`initial_aspects` 可省略，默认使用文件中的前 5 项，池不足 5 项时全部使用。
显式初始组合也必须是池内的 3～6 个不同词条。输出和实际编码顺序统一按 `aspects` 的文件顺序排列。

```bash
# 原始验证集，沿用该启动脚本现有模型和文本反思配置。
DSPY_ASPECT_MODE=pool \
DSPY_ASPECT_POOL_FILE=./configs/dspy_aspect_pool_ir.json \
bash scripts/dspy_prompt_schema_generation_encode_and_search.sh

# 生成验证集，沿用现有 LLaVA 服务和多模态错误样本反思配置。
DSPY_ASPECT_MODE=pool \
DSPY_ASPECT_POOL_FILE=./configs/dspy_aspect_pool_ir.json \
bash scripts/dspy_prompt_schema_generation_encode_and_search_generated_validation.sh
```

Python 对应选项是 `--dspy_aspect_mode pool --dspy_aspect_pool_file <文件>`。
文件在优化模型调用和编码权重加载前校验。池文件按任务分别维护，脚本不会自动替换候选词。
池模式可配合现有文本或视觉反思；多模态反思仍只支持 Flickr/COCO 的 IR 任务。

`USE_INITIAL_DSPY_ASPECTS=true` 时，第一轮使用 JSON 初始组合并实际计算验证集得分；
关闭热启动时，模型在池内生成第一组组合。池模式每一轮（包括第一轮）都根据选中词条
构造编码提示词，不使用原任务中与所选词不一致的手写完整模板。

GEPA 的 light/medium/heavy 预算仍为 10/18/27 个唯一且完成验证的候选。
若池中合法组合少于预算，则评估全部合法组合后结束，例如 3 个词只有 1 种组合，
4 个词共有 5 种组合（4 个三词子集和 1 个四词子集）。不同排列不会重复计分。
模型输出非法或重复时最多尝试 4 次，之后从尚未使用的合法组合中确定性选择一个作为兜底；
该候选也必须完成真实检索评估，不能使用虚构得分。服务错误不会伪装成成功的优化轮次。

最终选取已评估候选中得分最高者，同分保留较早完成者；未遍历全部组合时不保证池内全局最优。
选择指标沿用原有验证检索目标，视觉失败诊断仍会作为选择子集的反思依据。

锁文件、最终视角 JSON、编码及检索产物使用 `_pool_<内容指纹>` 区分。
修改池内容、顺序或有效初始组合会产生新指纹，避免复用其他配置的产物；
仅修改 JSON 排版、大小写或首尾空白不会改变规范化后的身份。
自由模式继续使用原有产物命名。

每轮 `dspy_rounds.jsonl` 和最终锁文件会记录池内容、指纹、初始组合、请求和实际轮数。
`candidate_source` 区分 `initial`、`model`、`pool_fallback`；最终选择信息中的
`dspy_gepa_stop_condition` 区分 `round_budget` 与 `pool_exhausted`。
多模态反思日志也记录同一池身份，所有进程在评估前使用主进程广播的合法组合。
