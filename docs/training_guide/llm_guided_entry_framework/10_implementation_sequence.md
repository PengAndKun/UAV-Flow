# Implementation Sequence

## 1. 文档目标

这份文档用于回答一个很实际的问题：

- 后面到底先写什么
- 每一步做到什么算通过
- 怎样避免一边写一边推翻前面设计

目标是把 `Phase 2.5` 从文档设计落成可执行开发计划。

---

## 2. 总原则

实现顺序建议遵循：

1. 先固定 teacher
2. 再固定 state
3. 再固定 dataset
4. 最后再进训练

不要反过来：

- 不要先写 PPO
- 不要先写 student 网络
- 不要先写复杂蒸馏损失

原因很简单：

- 如果 teacher schema 不稳，后面所有训练都会学坏
- 如果 state builder 结构不稳，后面模型接口会反复改
- 如果 dataset export 不稳，实验就没法复现

---

## 3. 推荐开发顺序

建议按 6 步走。

---

## 4. Step 1：固定 teacher schema

对应文档：
- [07_teacher_schema_spec.md](07_teacher_schema_spec.md)

### 4.1 目标

把当前 `Pure LLM / Fusion + LLM` 输出统一收成固定 schema。

### 4.2 要做的事

实现一个 teacher 结果规范化模块，负责：

- 读取 LLM 原始输出
- 校验字段合法性
- 映射到：
  - `entry_state`
  - `subgoal`
  - `action_hint`
  - `target_candidate_id`
  - `risk_level`
  - `reason`
  - `confidence`

### 4.3 通过标准

满足以下条件即可通过：

1. 对同一批样本，teacher 输出字段完整
2. 输出值全部落在固定集合里
3. 不再出现“window + cross_entry”这类明显冲突
4. 每条 teacher 结果都能保存成统一 JSON

### 4.4 推荐产物

- `teacher_output.json`
- `teacher_validation.json`

---

## 5. Step 2：构建 teacher 样本集

对应文档：
- [04_stepc_labeling_standard.md](04_stepc_labeling_standard.md)
- [05_pure_llm_baseline_test.md](05_pure_llm_baseline_test.md)

### 5.1 目标

先拿一批静态样本把 teacher 跑通并检查质量。

### 5.2 要做的事

1. 继续从 `phase2_multimodal_fusion_analysis/results/fusion_*/labeling/` 收样本
2. 对每个样本跑：
   - `Pure LLM baseline`
   - 或 `Fusion + LLM teacher`
3. 保存：
   - teacher 输出
   - teacher 校验结果
   - 人工标注对照

### 5.3 通过标准

达到下面任意一版可以进入下一步：

- `20 ~ 30` 个样本的 pilot set 完成
- 这批样本里 teacher 输出整体可读、可解析、可比较

### 5.4 推荐先做的小目标

优先覆盖 4 类：

- `enterable_open_door`
- `front_blocked_detour`
- `window_visible_keep_search`
- `no_entry_confirmed`

---

## 6. Step 3：实现 entry_state_builder

对应文档：
- [06_agent_state_schema.md](06_agent_state_schema.md)
- [08_entry_state_builder_spec.md](08_entry_state_builder_spec.md)

### 6.1 目标

把一个 `fusion` 样本包变成固定格式的 `entry_state.json`。

### 6.2 要做的事

实现状态构造器，负责：

1. 读取：
   - `yolo_result.json`
   - `depth_result.json`
   - `fusion_result.json`
   - `state_excerpt.json`
   - `pose_history_summary.json`
   - `teacher_output.json`
2. 生成：
   - `global_state`
   - `candidates`
   - `teacher_targets`
   - `metadata`

### 6.3 通过标准

1. 对任意一个已存在样本包，都能稳定生成 `entry_state.json`
2. 对候选不足 `K` 的情况，能自动补零候选
3. 有 teacher 和无 teacher 两种样本都能导出

### 6.4 强制检查项

必须检查：

- `top-K` 长度固定
- 每个 candidate 字段完整
- 所有归一化字段有明确值
- `teacher_available` 状态正确

---

## 7. Step 4：实现 dataset export

对应文档：
- [09_distillation_dataset_export_spec.md](09_distillation_dataset_export_spec.md)

### 7.1 目标

把零散样本包整理成正式训练集。

### 7.2 要做的事

实现数据集导出器，负责：

1. 遍历已有 `fusion_*` 目录
2. 读取对应的 `entry_state.json` 和 `teacher_output.json`
3. 做过滤：
   - 缺文件
   - invalid teacher
   - 低 confidence teacher
4. 生成：
   - `train.jsonl`
   - `val.jsonl`
   - `manifest.json`
   - `quality_report.json`

### 7.3 通过标准

1. 可以稳定导出一个完整 `train / val` 数据集
2. 每条样本都能追溯回源目录
3. 类别分布和 teacher 有效率能统计出来

---

## 8. Step 5：先做 BC 初始化，不先上 PPO

### 8.1 目标

先验证 student 能不能从 teacher 学到：

- `entry_state`
- `subgoal`
- `action_hint`

### 8.2 要做的事

第一版训练先做：

- classification / imitation

建议 student 先只用：

- `global_state`
- `topK candidate structured features`
- `teacher labels`

先不要：

- 加复杂 ROI encoder
- 加 PPO
- 加 sequence model

### 8.3 通过标准

如果 student 在验证集上已经能较稳定预测：

- `entry_state`
- `subgoal`
- `action_hint`

说明这套状态设计是成立的。

---

## 9. Step 6：再加 semantic distillation 和 PPO

### 9.1 目标

在 BC 跑通之后，再往上叠：

- `teacher_reason_embedding`
- PPO 微调

### 9.2 为什么放到最后

因为这两部分复杂度更高：

- `teacher_reason_embedding`
  会引入额外的文本编码器和蒸馏损失
- `PPO`
  会引入 rollout、奖励函数、策略稳定性问题

如果在前面基础没稳时先加它们，会很难定位问题。

### 9.3 通过标准

这一步通过的标志是：

1. student 比纯 BC 更稳
2. 复杂场景下：
   - `front_blocked_detour`
   - `window_visible_keep_search`
   的误判率下降

---

## 10. 推荐的最小里程碑

建议按 3 个小里程碑推进。

### M1：Teacher 可用

完成：

- teacher schema 固定
- pilot 样本集完成
- teacher 输出可验证

### M2：State 可用

完成：

- entry_state_builder 跑通
- 任意样本可生成 `entry_state.json`
- dataset export 跑通

### M3：Student 可用

完成：

- BC student 跑通
- 验证集上能学到合理 `subgoal/action_hint`

只要先做到 `M3`，就已经足够进入论文里的第一轮核心实验。

---

## 11. 不建议现在就做的事情

为了避免路线发散，当前阶段不建议先做：

- 复杂长序列 Transformer
- 端到端整图视觉 backbone 重训
- 大规模 PPO 先行
- 多任务一锅端联合训练
- 把完整搜人任务也绑进来一起训

这些都应该在：

- teacher 稳定
- state 稳定
- dataset 稳定

之后再做。

---

## 12. 当前最推荐的下一步

如果按这份顺序继续，最建议下一步先实现：

1. `teacher schema validator`
2. `entry_state_builder`

也就是说：

- 先固定 teacher 输出
- 再固定 student 输入

这是后面所有训练代码最稳定的基础。

---

## 13. 一句话总结

推荐顺序是：

`Teacher -> State -> Dataset -> BC -> Distill -> PPO`

不要反过来。
