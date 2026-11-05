# 图片维度候选视角模式

`USE_HIERARCHICAL_VIEWS=false` 为默认值，沿用原 A–T 的 20 项候选。设置为 `true` 后使用以下 A–J 候选；一级维度是主体，二级维度组成 `such as` 部分，不加入原图的典型元素和参考工作。

| 选项 | 一级维度 | 完整描述 |
|---|---|---|
| A | 场景 | scene, such as location, indoor or outdoor setting and terrain |
| B | 自然环境 | natural environment, such as weather, lighting, season and time |
| C | 人物 | people, such as number of people, appearance and pose |
| D | 衣着 | clothing, such as category, color, material and style |
| E | 物体 | objects, such as category, number, appearance and state |
| F | 动作 | actions, such as behavior, participants, tools and targets |
| G | 关系 | relations, such as spatial relations, interactions and ownership |
| H | 视觉属性 | visual attributes, such as color, shape, size and material |
| I | 空间布局 | spatial layout, such as foreground and background, distance and direction |
| J | 高层语义 | high-level semantics, such as events, activities and functions |

IR、TBPR、混合模态共用同一候选池。普通模式仍取前 5 项；overlap 默认仍取 5 项，`VIEW_SELECTED_COUNT` 范围为 1–10。PMI、选项打乱、FreeRet 两个开关可以继续组合使用。

单任务及三类任务批量编码脚本均支持该开关。例如：

```bash
USE_HIERARCHICAL_VIEWS=true bash scripts/encode_adaptive_views_ir.sh
USE_HIERARCHICAL_VIEWS=true bash scripts/encode_adaptive_views_overlap_tbpr.sh
USE_HIERARCHICAL_VIEWS=true USE_VIEW_PMI=true SHUFFLE_VIEW_OPTIONS=true bash scripts/encode_adaptive_views_overlap_fused.sh
```

也可以直接修改对应脚本开头的默认值为 `true`。直接调用 Python 时使用 `--adaptive_views --view_selection_hierarchical true`。仅查看候选列表可执行 `python -B src/adaptive_views.py pool --hierarchical`。

选择后，稀疏编码保留完整描述，例如：

```text
Summary the clothing, such as category, color, material and style in above image in one word:
```

文本输入使用 `sentence`，图文输入使用 `image and sentence`。全局摘要模板和密集表征方法不因这个开关改变；FreeRet 约束开关启用时，继续在这些提示词前添加原有约束。

新模式的选择文件及编码 manifest 记录 `view_pool_mode="hierarchical_v1"`；完整候选池仍保存在 `pool` 和 `view_pool.json` 中。此标识参与选择缓存、视角、编码及批量检索指纹，产物使用各自指纹目录。新模式的 `task_views` 为空，不把旧模板字母映射误用为新的 A–J。

编码复用 `VIEW_SELECTION_FILE` 时，开关必须与文件记录一致。关闭开关的旧模式继续使用原格式和指纹；不带 `view_pool_mode` 的文件按原候选池校验。

稀疏索引脚本从 manifest 和选择文件自动识别候选池，无需新增索引开关。单任务三阶段检索也按选择文件自动识别；批量三阶段检索需使用与编码一致的 `USE_HIERARCHICAL_VIEWS=true` 定位缓存，例如：

```bash
USE_HIERARCHICAL_VIEWS=true bash scripts/three_stage_retrieval_pipeline_adaptive_views_ir.sh
USE_HIERARCHICAL_VIEWS=true bash scripts/three_stage_retrieval_pipeline_adaptive_views_overlap_tbpr_multi.sh
```

使用 PMI、overlap 或打乱时，检索仍需匹配编码时的原有配置；打乱运行通过编码日志中的 `VIEW_SELECTION_RUN_ID` 定位。独立验证集按自己的选择文件识别候选池，可与主实验使用不同模式。
