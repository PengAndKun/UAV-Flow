# 34. Memory-Aware Representation V5 实验报告

## 1. 实验目的

本轮实验的目的不是训练动作控制策略，而是验证当前 `LLM + YOLO/RGB + Depth + Memory` 蒸馏出来的表示模型是否已经可以作为后续策略训练的状态输入。

核心问题是：

```text
z_entry 是否能稳定表达：
1. 当前是否看到目标房屋
2. 入口是否属于目标房屋
3. 目标入口是否可接近或被阻挡
4. 目标房屋是否已经完整搜索但无可用入口
5. 当前应该继续搜索、接近入口、绕行，还是切换房屋
```

因此，本轮评价重点是：

```text
target_conditioned_state
target_conditioned_subgoal
target_conditioned_target_candidate_id
no-entry full-coverage 判断
episode 内时序一致性
```

`target_conditioned_action_hint` 只作为诊断指标，不作为当前阶段通过/失败标准。

---

## 2. 实验输入

### 2.1 数据集

```text
E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\phase2_5_memory_aware_dataset_v3_20260427_20260427_141237
```

数据统计：

```text
total = 130
train = 105
val = 25
memory_source = after_snapshot
llm_teacher = validated
```

标签分布：

| 类别 | 数量 |
|---|---:|
| `target_house_not_in_view` | 44 |
| `non_target_house_entry_visible` | 42 |
| `target_house_no_entry_after_full_coverage` | 27 |
| `target_house_entry_blocked` | 9 |
| `target_house_entry_approachable` | 6 |
| `target_house_entry_visible` | 1 |
| `target_house_geometric_opening_needs_confirmation` | 1 |

### 2.2 模型

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\runs\memory_aware_v5_pilot_20260427\checkpoints\best.pt
```

模型配置：

```text
global_input_dim = 93
candidate_input_dim = 44
top_k = 3
representation_dim = 128
```

---

## 3. 已执行命令

### 3.1 导出全部 embedding

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python E:\github\UAV-Flow\phase2_5_representation_distillation\export_representation_embeddings.py `
  --checkpoint_path E:\github\UAV-Flow\phase2_5_representation_distillation\runs\memory_aware_v5_pilot_20260427\checkpoints\best.pt `
  --export_dir E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\phase2_5_memory_aware_dataset_v3_20260427_20260427_141237 `
  --run_name memory_aware_v5_pilot_20260427 `
  --split all `
  --device cpu `
  --batch_size 32
```

输出：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\embeddings\memory_aware_v5_pilot_20260427
```

### 3.2 分别导出 train / val

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python E:\github\UAV-Flow\phase2_5_representation_distillation\export_representation_embeddings.py `
  --checkpoint_path E:\github\UAV-Flow\phase2_5_representation_distillation\runs\memory_aware_v5_pilot_20260427\checkpoints\best.pt `
  --export_dir E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\phase2_5_memory_aware_dataset_v3_20260427_20260427_141237 `
  --run_name memory_aware_v5_pilot_20260427_val `
  --split val `
  --device cpu `
  --batch_size 32
```

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python E:\github\UAV-Flow\phase2_5_representation_distillation\export_representation_embeddings.py `
  --checkpoint_path E:\github\UAV-Flow\phase2_5_representation_distillation\runs\memory_aware_v5_pilot_20260427\checkpoints\best.pt `
  --export_dir E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\phase2_5_memory_aware_dataset_v3_20260427_20260427_141237 `
  --run_name memory_aware_v5_pilot_20260427_train `
  --split train `
  --device cpu `
  --batch_size 32
```

### 3.3 分析 embedding

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:MPLBACKEND='Agg'
python E:\github\UAV-Flow\phase2_5_representation_distillation\analyze_representation_embeddings.py `
  --embedding_dir E:\github\UAV-Flow\phase2_5_representation_distillation\embeddings\memory_aware_v5_pilot_20260427_val `
  --nearest_k 5
```

---

## 4. 实验结果

### 4.1 All split

输出目录：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\embeddings\memory_aware_v5_pilot_20260427
```

结果：

| 指标 | 数值 |
|---|---:|
| sample_count | 130 |
| embedding_dim | 128 |
| target_conditioned_state_accuracy | 0.9923 |
| target_conditioned_subgoal_accuracy | 0.9923 |
| target_conditioned_action_hint_accuracy | 0.7000 |
| target_conditioned_target_candidate_id_accuracy | 0.9077 |
| mean_no_entry_prob_when_true | 0.9715 |
| mean_no_entry_prob_when_false | 0.0057 |
| no_entry_separation_margin_mean | 0.9658 |
| temporal_warning_count | 0 |

### 4.2 Val split

输出目录：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\embeddings\memory_aware_v5_pilot_20260427_val
```

结果：

| 指标 | 数值 |
|---|---:|
| sample_count | 25 |
| embedding_dim | 128 |
| target_conditioned_state_accuracy | 0.9600 |
| target_conditioned_subgoal_accuracy | 1.0000 |
| target_conditioned_action_hint_accuracy | 0.5600 |
| target_conditioned_target_candidate_id_accuracy | 0.8400 |
| mean_no_entry_prob_when_true | 0.9724 |
| mean_no_entry_prob_when_false | 0.0064 |
| no_entry_separation_margin_mean | 0.9660 |
| temporal_warning_count | 0 |

### 4.3 Val label distribution

| 类别 | Val 数量 |
|---|---:|
| `target_house_not_in_view` | 9 |
| `non_target_house_entry_visible` | 8 |
| `target_house_no_entry_after_full_coverage` | 5 |
| `target_house_entry_blocked` | 2 |
| `target_house_entry_approachable` | 1 |

这个分布说明 val 集里 `blocked` 和 `approachable` 仍然很少，所以它们当前的准确率只能作为初步参考，不能作为最终结论。

---

## 5. No-Entry 判断分析

No-entry 是当前记忆模块最关键的能力之一，因为它不是单帧视觉判断，而是：

```text
目标房屋被多视角搜索
覆盖达到要求
没有可靠入口
然后才触发 complete_no_entry_search
```

Val split 结果：

```text
true_no_entry_count = 5
false_no_entry_count = 20
true_no_entry_prob.mean = 0.9724
false_no_entry_prob.mean = 0.0064
false_high_prob_count = 0
true_low_prob_count = 0
```

结论：

**当前模型对 `target_house_no_entry_after_full_coverage` 的区分非常清楚，没有出现明显过早触发 no-entry completion 的问题。**

这说明 memory 特征确实被模型用上了，而不是只靠单帧“没看到门”来判断。

---

## 6. Episode 时序一致性

Val split 共覆盖 3 个 episode：

| Episode | 样本数 | 起始状态 | 结束状态 | no-entry 数量 | warning |
|---|---:|---|---|---:|---:|
| `memory_episode_20260426_101452_search_house_1` | 4 | `non_target_house_entry_visible` | `target_house_entry_approachable` | 0 | 0 |
| `memory_episode_20260426_234211_search_house_1` | 16 | `target_house_not_in_view` | `target_house_not_in_view` | 0 | 0 |
| `memory_episode_20260427_000537_search_house_2` | 5 | `target_house_no_entry_after_full_coverage` | `target_house_no_entry_after_full_coverage` | 5 | 0 |

结论：

```text
temporal_warning_count = 0
```

也就是说，val 中没有出现：

- 无覆盖证据却触发 no-entry
- no-entry 在 episode 中过早出现
- no-entry 与 reliable entry 同时冲突

---

## 7. 当前主要错例

### 7.1 状态错例

Val 中只有 1 个 target-conditioned state 错例：

```text
sample_id = memory_capture_20260426_234255_step0001_auto_yaw
true_state = target_house_not_in_view
pred_state = target_house_geometric_opening_needs_confirmation
true_subgoal = reorient_to_target_house
pred_subgoal = reorient_to_target_house
```

解释：

这个错例发生在早期 yaw 阶段，模型把“目标房屋未入视野”误判为“疑似几何开口待确认”。但子目标仍然预测为 `reorient_to_target_house`，所以对高层搜索流程影响较小。

### 7.2 Candidate ID 错例

Val 中 candidate id accuracy 为 `0.84`，主要错例集中在：

```text
non_target_house_entry_visible
candidate_0 / candidate_1 / null_candidate 混淆
```

这说明当前模型已经能判断“不是目标入口”，但对“这个非目标入口是否对应 top-k 里的第几个 candidate”还不够稳定。

在现阶段这不是最大问题，因为策略首先需要知道：

```text
ignore_non_target_entry
```

而不是必须精确选择非目标 candidate id。

### 7.3 Action hint 错例

Val 中 action accuracy 为 `0.56`，主要混淆：

```text
yaw_left <-> yaw_right
right -> yaw_right
```

这符合预期，因为 action label 很受人工控制轨迹、采样时机和局部视角影响。

当前阶段不要用这个结果否定表示模型，也不要马上进入 action 训练。

---

## 8. Centroid 距离观察

Val split 中，几个关键状态的 centroid 距离：

| 状态 A | 状态 B | 距离 |
|---|---|---:|
| `target_house_no_entry_after_full_coverage` | `target_house_entry_blocked` | 13.46 |
| `target_house_no_entry_after_full_coverage` | `target_house_entry_approachable` | 13.27 |
| `target_house_no_entry_after_full_coverage` | `non_target_house_entry_visible` | 13.11 |
| `target_house_not_in_view` | `target_house_entry_approachable` | 12.68 |
| `target_house_entry_blocked` | `non_target_house_entry_visible` | 9.20 |

观察：

- `no_entry_after_full_coverage` 和其他类距离都很大，表示清晰。
- `blocked` 与 `non_target_entry_visible` 距离相对更近，说明还需要补采复杂场景。
- `approachable` 在 val 中只有 1 条，所以 centroid 参考价值有限。

---

## 9. 实验结论

当前 V5 memory-aware representation 的结论是：

**可以作为第一版可用状态表示，但还不建议进入 action 训练。**

已经成立的能力：

- 能区分 `target_house_not_in_view` 和 `non_target_house_entry_visible`
- 能稳定识别 `target_house_no_entry_after_full_coverage`
- 能输出稳定的 `complete_no_entry_search`
- episode 内没有明显 no-entry 过早触发
- `z_entry` 维度固定为 128，可以作为后续 policy state 的核心输入

仍需加强的能力：

- `target_house_entry_approachable` 样本太少
- `target_house_entry_blocked` 样本太少
- `target_house_geometric_opening_needs_confirmation` 几乎没有覆盖
- candidate id 在非目标入口场景下仍有混淆
- action hint 目前不适合作为核心监督目标

---

## 10. 下一轮实验建议

### 10.1 先补采 V6 数据，而不是训练 action

下一轮最应该补的是：

| 优先级 | 类别 | 当前数量 | 目标数量 |
|---|---|---:|---:|
| P0 | `target_house_entry_approachable` | 6 | >= 30 |
| P0 | `target_house_entry_blocked` | 9 | >= 30 |
| P1 | `target_house_geometric_opening_needs_confirmation` | 1 | >= 15 |
| P1 | `target_house_entry_visible` | 1 | >= 15 |
| P1 | 混合目标/非目标入口同屏 | 不足 | >= 30 |

### 10.2 推荐采集场景

建议围绕 house 1 / house 2 / 新 house 增加以下 episode：

```text
search_house_1_front_approach_long
search_house_1_side_approach_left
search_house_1_side_approach_right
search_house_1_blocked_then_detour
search_house_1_window_distractor_same_view
search_house_2_full_no_entry_repeat
search_house_3_target_vs_non_target_same_frame
```

每个 episode 推荐：

```text
capture_count >= 20
至少包含 5 个目标入口稳定帧
至少包含 3 个过渡帧
至少包含 3 个干扰帧
如果是 no-entry，最后 5 帧才进入 complete_no_entry_search
```

### 10.3 下一步工程动作

建议下一步做：

```text
1. 补采 V6 memory sessions
2. 跑 memory-aware LLM teacher batch
3. 重新 export dataset
4. 训练 memory_aware_v6
5. 导出 z_entry
6. 对比 V5 / V6 representation quality
```

对比指标：

```text
val target_conditioned_state_accuracy
val target_conditioned_subgoal_accuracy
no_entry_separation_margin
approachable vs blocked centroid distance
non_target vs target entry confusion
candidate id accuracy
episode temporal warning count
```

---

## 11. 当前阶段判断

如果目标是“验证 memory 是否有用”，当前 V5 已经给出积极证据。

如果目标是“进入 action/RL 控制训练”，当前还差一轮补采。

最稳妥路线是：

```text
V5 表示验证通过
-> V6 补采薄弱状态
-> V6 表示训练与对比
-> 冻结 z_entry
-> 再进入 action / BC / RL
```

