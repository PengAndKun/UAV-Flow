# 35. Memory Ablation 实验报告

## 1. 实验目的

第 34 份报告已经说明 V5 memory-aware representation 在 `target_conditioned_state`、`target_conditioned_subgoal` 和 no-entry full-coverage 判断上表现较好。

但还需要回答一个更关键的问题：

```text
模型是真的利用了 memory，还是只靠单帧 RGB / YOLO / depth 特征碰巧学到了标签？
```

因此，本轮实验做 memory ablation：

```text
同一个 checkpoint
同一个 dataset split
只改变输入特征
观察 target-conditioned state、subgoal 和 no-entry 判断是否下降
```

如果去掉 memory 后 no-entry 判断显著下降，就说明结构化记忆确实参与了表示学习。

---

## 2. 实验脚本

新增脚本：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\evaluate_representation_ablation.py
```

支持的 ablation mode：

| Mode | 含义 |
|---|---|
| `none` | 原始完整输入 |
| `zero_memory` | 将所有 `memory_*` 全局特征置零 |
| `zero_candidates` | 将 top-k candidate features 和 mask 置零 |
| `zero_memory_and_candidates` | 同时移除 memory 与 candidate features |

注意：

这里不是重新训练模型，而是用同一个训练好的模型做输入消融评估。

---

## 3. 实验命令

### 3.1 Val split

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python E:\github\UAV-Flow\phase2_5_representation_distillation\evaluate_representation_ablation.py `
  --checkpoint_path E:\github\UAV-Flow\phase2_5_representation_distillation\runs\memory_aware_v5_pilot_20260427\checkpoints\best.pt `
  --export_dir E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\phase2_5_memory_aware_dataset_v3_20260427_20260427_141237 `
  --run_name memory_aware_v5_pilot_20260427_val_memory_ablation `
  --split val `
  --ablations none zero_memory zero_candidates zero_memory_and_candidates `
  --device cpu `
  --batch_size 32
```

输出：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\ablations\memory_aware_v5_pilot_20260427_val_memory_ablation
```

### 3.2 Train split

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python E:\github\UAV-Flow\phase2_5_representation_distillation\evaluate_representation_ablation.py `
  --checkpoint_path E:\github\UAV-Flow\phase2_5_representation_distillation\runs\memory_aware_v5_pilot_20260427\checkpoints\best.pt `
  --export_dir E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\phase2_5_memory_aware_dataset_v3_20260427_20260427_141237 `
  --run_name memory_aware_v5_pilot_20260427_train_memory_ablation `
  --split train `
  --ablations none zero_memory zero_candidates zero_memory_and_candidates `
  --device cpu `
  --batch_size 32
```

输出：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\ablations\memory_aware_v5_pilot_20260427_train_memory_ablation
```

---

## 4. Val split 结果

| Mode | State Acc | Subgoal Acc | Action Acc | Candidate Acc | No-entry True Prob | No-entry False Prob | No-entry Margin | True Low |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 0.96 | 1.00 | 0.56 | 0.84 | 0.9724 | 0.0064 | 0.9660 | 0 |
| `zero_memory` | 0.60 | 0.48 | 0.36 | 0.80 | 0.0073 | 0.0147 | -0.0074 | 5 |
| `zero_candidates` | 0.92 | 0.84 | 0.56 | 0.88 | 0.9717 | 0.0103 | 0.9614 | 0 |
| `zero_memory_and_candidates` | 0.08 | 0.12 | 0.52 | 0.84 | 0.0095 | 0.0190 | -0.0095 | 5 |

Val split 的关键变化：

```text
zero_memory:
target_conditioned_state_accuracy: 0.96 -> 0.60
target_conditioned_subgoal_accuracy: 1.00 -> 0.48
no_entry_true_prob: 0.9724 -> 0.0073
no_entry_margin: 0.9660 -> -0.0074
true_low_prob_count: 0 -> 5
```

解释：

Val 里有 5 个真实 no-entry 样本。移除 memory 后，这 5 个样本全部变成 no-entry 低置信。

这说明：

**模型判断 no-entry full-coverage 主要依赖 memory，而不是只靠单帧视觉。**

---

## 5. Train split 结果

| Mode | State Acc | Subgoal Acc | Action Acc | Candidate Acc | No-entry True Prob | No-entry False Prob | No-entry Margin | True Low |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `none` | 1.0000 | 0.9905 | 0.7333 | 0.9238 | 0.9713 | 0.0055 | 0.9658 | 0 |
| `zero_memory` | 0.8190 | 0.8000 | 0.5238 | 0.8952 | 0.1534 | 0.0064 | 0.1471 | 18 |
| `zero_candidates` | 0.9714 | 0.9333 | 0.7333 | 0.9143 | 0.9711 | 0.0179 | 0.9531 | 0 |
| `zero_memory_and_candidates` | 0.4762 | 0.5143 | 0.4190 | 0.8952 | 0.0301 | 0.0157 | 0.0145 | 22 |

Train split 的关键变化：

```text
zero_memory:
target_conditioned_state_accuracy: 1.0000 -> 0.8190
target_conditioned_subgoal_accuracy: 0.9905 -> 0.8000
no_entry_true_prob: 0.9713 -> 0.1534
true_low_prob_count: 0 -> 18
```

Train 中 no-entry 样本更多。去掉 memory 后，虽然不是完全崩掉，但 no-entry 置信度大幅下降，22 个 no-entry 中有 18 个变成低置信。

---

## 6. 结果解释

### 6.1 Memory 对 no-entry 判断是必要的

No-entry full-coverage 本质不是单帧视觉任务，而是时序搜索任务：

```text
看过哪些 sector
是否完成覆盖
是否有 reliable entry
是否存在 rejected candidate
是否应该切换 house
```

这些信息主要来自 memory features。

实验中：

```text
val none no-entry margin = 0.9660
val zero_memory no-entry margin = -0.0074
```

这说明移除 memory 后，模型几乎失去了区分 no-entry completion 的能力。

### 6.2 Candidate features 对 no-entry 不是主要来源

Val 中：

```text
zero_candidates no-entry margin = 0.9614
```

与完整输入的 `0.9660` 非常接近。

这说明 no-entry 判断主要不是来自当前帧 candidate，而是来自 memory。

这也符合任务定义：

```text
当前帧没有入口 != 完整搜索后没有入口
```

只有 memory 才能表达“已经搜索充分”。

### 6.3 Candidate features 对 subgoal 有辅助作用

虽然 zero_candidates 对 no-entry 影响很小，但会让 subgoal accuracy 从：

```text
val: 1.00 -> 0.84
train: 0.9905 -> 0.9333
```

这说明 candidate features 对 `approach / detour / ignore` 这类局部子目标仍然有帮助。

### 6.4 Memory 和 candidate 同时去掉会严重崩溃

Val 中：

```text
zero_memory_and_candidates:
state_acc = 0.08
subgoal_acc = 0.12
```

说明当前模型确实在融合两类信息：

```text
memory = 搜索历史和任务进度
candidate = 当前帧入口候选与几何证据
```

二者缺一会退化，尤其是 memory 对任务级状态更关键。

---

## 7. 对论文方法的意义

这轮实验可以支持以下论文表述：

```text
The proposed memory-aware representation does not merely encode instantaneous visual detections.
By ablating structured memory features at inference time, the model loses its ability to distinguish
full-coverage no-entry states, demonstrating that the hierarchical search memory provides essential
temporal and task-progress information for target-conditioned entry grounding.
```

中文可以写成：

```text
该表示并非仅依赖单帧视觉检测。当在推理阶段移除结构化记忆特征后，模型几乎失去了识别“目标房屋已完整搜索但无可用入口”状态的能力，说明层次化搜索记忆为目标条件化入口定位提供了必要的时序与任务进度信息。
```

这可以作为后续论文中的一个重要消融实验。

---

## 8. 当前结论

V5 memory ablation 的结论非常明确：

```text
1. Memory 是 no-entry full-coverage 判断的关键来源。
2. Candidate features 更偏当前帧入口候选与局部子目标判断。
3. 单帧视觉/候选特征不能替代结构化搜索记忆。
4. 当前 memory-aware representation 的设计方向是成立的。
```

因此，现在可以说：

**V5 表示模型不仅表现好，而且确实使用了 memory。**

---

## 9. 下一步建议

下一步不建议直接进入 action 训练。

建议继续做 V6 数据补采与训练：

```text
1. 补采 target_house_entry_approachable
2. 补采 target_house_entry_blocked
3. 补采 geometric_opening_needs_confirmation
4. 补采目标入口和非目标入口同屏干扰
5. 重新训练 memory_aware_v6
6. 再跑同样的 embedding analysis 和 memory ablation
```

V6 通过标准：

```text
val target_conditioned_state_accuracy >= 0.90
val target_conditioned_subgoal_accuracy >= 0.85
val no_entry_margin >= 0.70
zero_memory 后 no_entry_margin 明显下降
approachable / blocked 支持数 >= 5 each in val
```

