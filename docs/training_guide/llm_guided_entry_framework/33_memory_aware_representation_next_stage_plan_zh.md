# 33. Memory-Aware Representation 下一阶段推进计划

## 1. 文档目的

当前阶段已经完成了 memory-aware 数据采集、LLM teacher 标注、可训练样本导出、以及第一版 memory-aware representation distillation 训练。

这份文档用于明确一个关键决策：

**下一步暂时不进入 action / policy / RL 训练，而是先把 `LLM + YOLO/RGB + Depth + Memory` 蒸馏出的表示模型做实、导出、验证，并确定它能否作为后续策略训练的状态输入。**

也就是说，当前阶段的目标不是让无人机直接学会控制，而是先获得一个稳定的入口搜索状态表示：

```text
RGB + Depth + YOLO + Target House Context + Memory
-> z_entry
-> target-conditioned state / subgoal / entry association evidence
-> later policy state
```

其中 `z_entry` 是后续 action 训练或强化学习要消费的状态表示。

---

## 2. 当前进度锚点

### 2.1 数据集

当前可用的 memory-aware dataset：

```text
E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\phase2_5_memory_aware_dataset_v3_20260427_20260427_141237
```

基本统计：

```text
total_exported = 130
train_count = 105
val_count = 25
memory_available_count = 130
memory_source_counts.after_snapshot = 130
llm_teacher_available_count = 130
teacher_source_priority = llm_teacher_validated
```

目标条件化标签分布：

```text
target_house_not_in_view = 44
non_target_house_entry_visible = 42
target_house_no_entry_after_full_coverage = 27
target_house_entry_blocked = 9
target_house_entry_approachable = 6
target_house_entry_visible = 1
target_house_geometric_opening_needs_confirmation = 1
```

这说明数据已经能覆盖三类关键状态：

- 目标房屋尚未进入视野
- 看到非目标房屋入口，需要过滤
- 目标房屋完整搜索后确认没有可用入口
- 目标入口可接近或被阻挡

但 `target_house_entry_approachable`、`target_house_entry_visible`、`target_house_geometric_opening_needs_confirmation` 仍然偏少，后续需要继续补采。

### 2.2 已训练模型

当前模型：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\runs\memory_aware_v5_pilot_20260427
```

最佳 checkpoint：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\runs\memory_aware_v5_pilot_20260427\checkpoints\best.pt
```

关键配置：

```text
global_input_dim = 93
candidate_input_dim = 44
top_k = 3
representation_dim = 128
train_examples = 105
val_examples = 25
best_epoch = 14
epochs_completed = 26
```

### 2.3 验证结果

评估目录：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\evals\memory_aware_v5_pilot_20260427_eval
```

验证集结果：

| 任务 | Accuracy | Macro F1 | 说明 |
|---|---:|---:|---|
| `target_conditioned_state` | 0.96 | 0.706 | 当前最重要，说明状态识别基本可用 |
| `target_conditioned_subgoal` | 1.00 | 0.500 | 子目标方向很稳，但类别仍不均衡 |
| `target_conditioned_action_hint` | 0.56 | 0.354 | 暂不作为当前阶段核心指标 |
| `target_conditioned_target_candidate_id` | 0.84 | 0.392 | 候选入口关联已有基础，但还需更多样本 |

当前结论：

**这个模型已经可以作为 memory-aware state representation 的第一版候选，但还不能直接进入 action 训练。**

---

## 3. 为什么现在不做 action 训练

action 训练是最后阶段，因为它依赖三个前提：

1. 状态表示要稳定。
2. 目标入口和目标房屋的关联要可信。
3. 搜索完成、继续观察、靠近入口、切换房屋这些高层状态要能区分。

当前 `target_conditioned_action_hint` 准确率只有 `0.56`，主要原因不是模型完全不可用，而是当前采集数据里的 action 更像人工操作轨迹，不是严格策略标签。

例如：

```text
yaw_left / yaw_right / right / forward
```

这些动作受到操作者习惯、视角调整、局部避障、采样时机影响较大。

所以当前阶段不要把 action 当成主目标，否则会把表示模型带偏。

更合理的阶段划分是：

```text
阶段 A：训练 memory-aware representation
阶段 B：导出 representation embedding
阶段 C：验证 representation 是否能区分搜索状态和入口归属
阶段 D：补采薄弱状态数据
阶段 E：冻结或半冻结 representation，进入 policy / RL / BC
```

当前正处于阶段 B 和阶段 C。

---

## 4. 下一步核心任务

### 4.1 任务一：导出 representation embedding

需要新增一个导出脚本：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\export_representation_embeddings.py
```

输入：

```text
--checkpoint_path
--export_dir
--split train / val / all
--output_dir
--device
--batch_size
--run_name
```

默认使用：

```text
checkpoint_path =
E:\github\UAV-Flow\phase2_5_representation_distillation\runs\memory_aware_v5_pilot_20260427\checkpoints\best.pt

export_dir =
E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\phase2_5_memory_aware_dataset_v3_20260427_20260427_141237
```

输出建议放到：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\embeddings\memory_aware_v5_pilot_20260427
```

### 4.2 embedding 输出格式

建议至少导出三类文件：

```text
embeddings.npy
metadata.jsonl
embedding_summary.json
```

其中 `embeddings.npy` 保存纯向量：

```text
shape = [num_samples, 128]
```

`metadata.jsonl` 每行对应一个样本：

```json
{
  "sample_id": "memory_capture_20260427_001426_step0278_auto_step",
  "split": "val",
  "source_session": "memory_episode_20260427_000537_search_house_2",
  "target_house_id": "002",
  "representation_index": 17,
  "true_target_conditioned_state": "target_house_no_entry_after_full_coverage",
  "pred_target_conditioned_state": "target_house_no_entry_after_full_coverage",
  "true_target_conditioned_subgoal": "complete_no_entry_search",
  "pred_target_conditioned_subgoal": "complete_no_entry_search",
  "no_entry_completion_prob": 0.94,
  "state_probs": {
    "target_house_not_in_view": 0.01,
    "target_house_no_entry_after_full_coverage": 0.94
  },
  "subgoal_probs": {
    "complete_no_entry_search": 0.96
  },
  "candidate_id_probs": {
    "none": 0.91
  },
  "entry_search_status": "no_entry_found_after_full_coverage",
  "full_coverage_ready": true,
  "has_reliable_entry": false,
  "memory_source": "after_snapshot"
}
```

`embedding_summary.json` 用于快速检查：

```json
{
  "sample_count": 130,
  "embedding_dim": 128,
  "split_counts": {
    "train": 105,
    "val": 25
  },
  "state_accuracy": 0.96,
  "subgoal_accuracy": 1.0,
  "no_entry_true_count": 27,
  "no_entry_val_count": 5,
  "mean_no_entry_prob_when_true": 0.0,
  "mean_no_entry_prob_when_false": 0.0
}
```

这里的 `mean_no_entry_prob_when_true / false` 后续由脚本实际计算。

---

## 5. 表示质量验证应该看什么

导出 embedding 后，不要马上进入控制训练，而是先做表示验证。

### 5.1 验证一：状态可分性

目标：

确认 `z_entry` 能把不同 target-conditioned state 分开。

重点看：

```text
target_house_not_in_view
non_target_house_entry_visible
target_house_entry_blocked
target_house_entry_approachable
target_house_no_entry_after_full_coverage
```

推荐检查：

- confusion matrix
- 每类 embedding centroid
- 类内距离和类间距离
- 最近邻样本是否语义一致
- UMAP / t-SNE 可视化

合格标准：

```text
target_house_no_entry_after_full_coverage 不应和 target_house_not_in_view 混在一起
non_target_house_entry_visible 不应和 target_house_entry_approachable 混在一起
target_house_entry_blocked 和 target_house_entry_approachable 可以接近，但不能完全重叠
```

### 5.2 验证二：no-entry 完整搜索判断

这是 memory-aware 表示的核心能力之一。

目标：

确认模型不是因为“没看到门”就输出完成搜索，而是结合 memory 判断：

```text
full_coverage_ready = true
has_reliable_entry = false
entry_search_status = no_entry_found_after_full_coverage
```

需要检查：

- true no-entry 样本的 `no_entry_completion_prob`
- false no-entry 样本的 `no_entry_completion_prob`
- 搜索中间阶段是否被误判为完成
- house 2 这种绕一圈无门样本是否稳定

合格标准：

```text
true no-entry 平均概率明显高于 false no-entry
搜索中间阶段不能过早触发 complete_no_entry_search
```

### 5.3 验证三：入口候选关联

目标：

确认模型能区分：

```text
目标房屋入口
非目标房屋入口
窗户 / 假入口
几何开口但未确认入口
```

重点看：

- `target_conditioned_target_candidate_id`
- `has_reliable_entry`
- `candidate_entry_count`
- `best_entry_id`
- `association_confidence`

当前 `target_conditioned_target_candidate_id` accuracy 为 `0.84`，说明有基础，但不够强。

后续需要特别补采：

```text
同一画面中同时有目标房屋入口和非目标房屋入口
目标房屋附近有 window 干扰
目标入口只在侧面出现
目标入口被遮挡后再通过绕行确认
```

### 5.4 验证四：时序一致性

memory 的价值不在单帧，而在 episode。

所以要检查同一个 episode 内预测是否合理演化：

```text
target_house_not_in_view
-> target_house_entry_blocked
-> target_house_entry_approachable
-> approach_target_entry
```

或者无入口搜索：

```text
target_house_not_in_view
-> keep_search_target_house
-> target_house_no_entry_after_full_coverage
-> complete_no_entry_search
-> switch_to_next_house
```

合格标准：

```text
同一 episode 内状态不能高频跳变
complete_no_entry_search 只能出现在覆盖充分之后
approach_target_entry 应该出现在可靠入口稳定之后
```

---

## 6. 当前最需要补充的数据

基于当前分布，下一轮优先补这几类。

### 6.1 目标入口可接近样本

当前：

```text
target_house_entry_approachable = 6
```

建议补到：

```text
>= 30
```

采集方式：

- 目标 house 正门逐步靠近
- 目标 house 侧门逐步靠近
- 门从画面边缘逐渐移动到中心
- 从 blocked 视角绕到 approachable 视角

### 6.2 目标入口被阻挡样本

当前：

```text
target_house_entry_blocked = 9
```

建议补到：

```text
>= 30
```

采集方式：

- 门前有遮挡物
- 深度显示不可通行
- 画面看得到门，但路径不稳定
- 需要左绕或右绕才能确认入口

### 6.3 几何开口待确认样本

当前：

```text
target_house_geometric_opening_needs_confirmation = 1
target_house_entry_visible = 1
```

建议补到：

```text
每类 >= 15
```

采集方式：

- 远距离看到疑似门洞
- 低置信度 YOLO 门
- depth 有开口但 RGB 不明确
- RGB 像入口但 depth 不支持直接进入

### 6.4 目标房屋无入口完整搜索样本

当前：

```text
target_house_no_entry_after_full_coverage = 27
```

这类已经可用，但还需要扩展到更多 house。

采集方式：

- 绕目标 house 一圈
- 每个 front/left/right sector 至少观察一次
- 不要一开始就判定无入口
- 最后一帧才应触发 `complete_no_entry_search`

---

## 7. 下一步实现顺序

### 7.1 第一步：写 embedding exporter

新增：

```text
phase2_5_representation_distillation/export_representation_embeddings.py
```

目标：

```text
checkpoint + exported dataset -> 128-d z_entry + predictions + metadata
```

完成标准：

```text
embeddings.npy 存在
metadata.jsonl 行数 = 130
embedding_summary.json 中 embedding_dim = 128
train + val 样本都能导出
no_entry_completion_prob 可以被统计
```

### 7.2 第二步：写 representation analyzer

新增：

```text
phase2_5_representation_distillation/analyze_representation_embeddings.py
```

目标：

```text
embedding 输出 -> 表示质量报告
```

建议输出：

```text
state_centroid_distance.csv
nearest_neighbors.jsonl
no_entry_probability_report.json
episode_temporal_consistency_report.json
embedding_projection.png
```

### 7.3 第三步：决定是否补采 v4 数据

如果 analyzer 显示：

```text
approachable / blocked 不可分
no-entry 容易过早触发
non-target entry 和 target entry 混淆
candidate id 仍不稳定
```

则先补采数据，不进入 action。

如果 analyzer 显示：

```text
state 可分
no-entry 稳定
candidate association 可用
episode 内状态演化合理
```

则可以进入下一阶段：

```text
冻结 representation
定义 policy state interface
准备 action / RL / BC 数据
```

---

## 8. 后续 policy state interface 草案

后续 action 训练不要直接吃原始 RGB / depth。

推荐状态输入：

```text
s_policy = [
  z_entry,
  target_state_probs,
  target_subgoal_probs,
  candidate_id_probs,
  memory_summary_features,
  relative_pose_to_target_house,
  relative_pose_to_best_entry
]
```

其中：

```text
z_entry: 128-d representation
target_state_probs: 目标状态概率
target_subgoal_probs: 高层子目标概率
candidate_id_probs: 入口候选概率
memory_summary_features: 搜索覆盖、候选数量、是否已完整搜索等
relative_pose_to_target_house: UAV 与目标 house 的相对位置
relative_pose_to_best_entry: UAV 与最佳入口的相对位置
```

这一步只是接口定义，不代表现在开始训练 action。

---

## 9. 当前阶段的通过标准

在进入 action 训练前，至少满足：

```text
1. embedding exporter 可稳定导出所有样本
2. representation_dim = 128
3. target_conditioned_state val accuracy >= 0.90
4. target_conditioned_subgoal val accuracy >= 0.85
5. no-entry completion 在 val 上无明显误触发
6. non-target entry 与 target entry 不严重混淆
7. episode 内预测状态演化合理
8. 至少补足 approachable / blocked / geometric opening 的薄弱样本
```

当前已经满足：

```text
target_conditioned_state val accuracy = 0.96
target_conditioned_subgoal val accuracy = 1.00
no-entry completion val support = 5
```

当前尚未完全满足：

```text
embedding exporter 未实现
embedding analyzer 未实现
approachable / blocked / geometric opening 数据仍偏少
candidate association 仍需更多复杂场景验证
```

---

## 10. 一句话总结

当前最正确的推进方向是：

**先把 `LLM + YOLO/RGB + Depth + Memory` 蒸馏成稳定、可导出、可验证的 `z_entry` 表示；确认这个表示能区分目标房屋、目标入口、非目标入口、无入口完整搜索之后，再把它作为后续 action / RL / BC 的状态输入。**

所以，下一步工程任务不是 action 训练，而是：

```text
export_representation_embeddings.py
-> analyze_representation_embeddings.py
-> 根据表示质量决定补采 v4 数据
-> 再进入 policy state interface 和 action 训练
```

