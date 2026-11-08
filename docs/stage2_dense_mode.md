# 三阶段检索的 dense 二阶段模式

普通和 adaptive 三阶段流水线均支持：

```bash
STAGE2_METHOD=dense bash scripts/three_stage_retrieval_pipeline.sh
STAGE2_METHOD=dense bash scripts/three_stage_retrieval_pipeline_adaptive_views_tbpr.sh
```

对应 Python 参数为 `--stage2-method dense`（也接受 `--stage2_method dense`）。
默认 `STAGE2_METHOD=hybrid`，保留原来的二阶段融合行为。
现有普通、Qwen、LLaVA、FreeRet、adaptive 及 overlap 的三阶段启动脚本均传递此配置，
包括 IR、TBPR、fused 和多模型多数据集入口。overlap 的视角选择算法保持原样。

## 候选范围与计算

流程为：一阶段召回 → 一阶段候选内 dense 排序 → 三阶段重排。
二阶段不会重新在全库召回候选，只对每个查询的一阶段前 `STAGE1_DEPTH` 个候选计算密集相似度，
再取前 `STAGE2_DEPTH` 个结果交给原有三阶段。相似度与归一化设置沿用现有 dense 检索逻辑。

dense 模式不执行二阶段 Lucene 稀疏检索、hybrid 融合或 alpha 扫描，
`HYBRID_ALPHA`、`SELECT_HYBRID_ALPHA` 及 alpha 验证库输入会被忽略并在日志中说明。
二阶段只要求 dense 编码产物；一阶段使用 sparse_manual/sparse_concat 时，其自身的稀疏索引仍然必需。

查询编码与提示词保持原样，支持原有 FreeRet 开关、adaptive 锁定视角和视角级掩码。
一阶段 sparse_concat 生成的密集查询向量继续直接复用，因此无需重新编码候选库或重建索引。
查询编码本身仍沿用原有联合 dense/sparse 实现，新模式改变的是二阶段使用的分数。

## 输出与兼容

- 最终二阶段排名写为 `stage2_dense_run.trec` 和 `stage2_dense_run.json`。
- `recall_metrics.csv/.json` 中二阶段标为 `dense`，包含 R@1、R@5、R@10、R@20。
- `pipeline_metadata.json` 的 `stage2.method` 为 `dense`，`alpha` 和二阶段 sparse 路径为 `null`。
- 不生成二阶段 hybrid/sparse 排名或 alpha 扫描报告；耗时仅记录实际执行的步骤。
- 三阶段继续使用原有重排方式，输入换成二阶段的 dense 排名。

自动生成的结果目录包含 `_stage2_dense` 标记，FreeRet 后缀仍保留在末尾；
adaptive 批量检索指纹包含新二阶段方法，并排除没有使用的 alpha 参数。
编码指纹、视角选择指纹和默认 hybrid 结果路径保持原样。
手动 `OUTPUT_DIR` 仍是精确目录，已有输出需按原规则显式开启 `OVERWRITE=true` 才可覆写。

普通流水线的 `--stage2-run` 在 dense 模式下应提供预计算的 dense 排名；程序先过滤到一阶段候选，
再取二阶段深度，并以 dense 标签保存和评估。adaptive 模式原有的预计算输入限制保持不变。
