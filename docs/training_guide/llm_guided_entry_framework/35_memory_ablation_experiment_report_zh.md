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

正式实验统计要求：

```text
每个 ablation mode 重复 10 次
报告 mean / variance / std
```

当前实现已经支持：

```text
--repeats 10
```

但需要注意，当前是 `model.eval()` 下的推理阶段消融，且没有随机采样和 dropout，因此 10 次重复是确定性的，方差理论上为 0。若论文最终需要非零方差，应进一步做：

```text
10 个不同 seed 的重新训练 checkpoint
或 10 个不同随机 train/val split
```

当前已经进一步完成这两类正式实验，并将结果汇总在本文件第 10 节。

自动化脚本：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\run_formal_ablation_experiments.py
```

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
  --batch_size 32 `
  --repeats 10
```

输出：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\ablations\memory_aware_v5_pilot_20260427_val_memory_ablation_10repeat
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
  --batch_size 32 `
  --repeats 10
```

输出：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\ablations\memory_aware_v5_pilot_20260427_train_memory_ablation_10repeat
```

---

## 4. Val split 结果，10 次重复

| Mode | State Acc Mean | State Var | Subgoal Mean | Subgoal Var | No-entry Margin Mean | No-entry Margin Var | True Low Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none` | 0.9600 | ~0 | 1.0000 | 0 | 0.9660 | ~0 | 0 |
| `zero_memory` | 0.6000 | ~0 | 0.4800 | ~0 | -0.0074 | 0 | 5 |
| `zero_candidates` | 0.9200 | ~0 | 0.8400 | ~0 | 0.9614 | ~0 | 0 |
| `zero_memory_and_candidates` | 0.0800 | ~0 | 0.1200 | ~0 | -0.0095 | ~0 | 5 |

更完整的 10 次统计已经写入：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\ablations\memory_aware_v5_pilot_20260427_val_memory_ablation_10repeat\ablation_summary.json
```

Val split 的关键变化：

```text
zero_memory，10 次平均:
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

## 5. Train split 结果，10 次重复

| Mode | State Acc Mean | State Var | Subgoal Mean | Subgoal Var | No-entry Margin Mean | No-entry Margin Var | True Low Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none` | 1.0000 | 0 | 0.9905 | 0 | 0.9658 | 0 | 0 |
| `zero_memory` | 0.8190 | 0 | 0.8000 | ~0 | 0.1471 | 0 | 18 |
| `zero_candidates` | 0.9714 | ~0 | 0.9333 | 0 | 0.9531 | 0 | 0 |
| `zero_memory_and_candidates` | 0.4762 | 0 | 0.5143 | 0 | 0.0145 | 0 | 22 |

更完整的 10 次统计已经写入：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\ablations\memory_aware_v5_pilot_20260427_train_memory_ablation_10repeat\ablation_summary.json
```

Train split 的关键变化：

```text
zero_memory，10 次平均:
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

---

## 10. 正式 10 Seed / 10 Random Split 实验

上面的 10 次重复是同一 checkpoint 的确定性推理消融，因此方差约等于 0。

为了满足正式实验要求，已经额外完成：

```text
seed_retrain: 10 个不同训练 seed，固定原始 split
random_split: 10 个不同随机 split，固定训练 seed
```

运行命令：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python E:\github\UAV-Flow\phase2_5_representation_distillation\run_formal_ablation_experiments.py `
  --base_export_dir E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\phase2_5_memory_aware_dataset_v3_20260427_20260427_141237 `
  --experiment_name memory_aware_v5_formal_ablation_10seed_10split `
  --num_repeats 10 `
  --start_seed 202604270 `
  --experiment_modes seed_retrain random_split `
  --inference_repeats 1
```

输出目录：

```text
E:\github\UAV-Flow\phase2_5_representation_distillation\ablations\memory_aware_v5_formal_ablation_10seed_10split_20260427_171221
```

核心文件：

```text
formal_ablation_summary.json
formal_ablation_runs.jsonl
formal_ablation_table.csv
```

同时生成：

```text
10 个 seed_retrain checkpoint
10 个 random_split checkpoint
10 个 random split dataset
```

### 10.1 不同训练 Seed，固定 Split

| Mode | State Acc Mean | State Acc Std | Subgoal Mean | Subgoal Std | No-entry Margin Mean | No-entry Margin Std | True Low Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none` | 0.8760 | 0.0605 | 0.8960 | 0.0599 | 0.9826 | 0.0110 | 0.0 |
| `zero_memory` | 0.6160 | 0.0480 | 0.6080 | 0.0711 | 0.0523 | 0.1234 | 4.5 |
| `zero_candidates` | 0.7120 | 0.1647 | 0.7200 | 0.1486 | 0.9798 | 0.0137 | 0.0 |
| `zero_memory_and_candidates` | 0.2360 | 0.1368 | 0.2640 | 0.1592 | 0.0791 | 0.1959 | 4.1 |

相对完整输入的平均下降：

| Ablation | State Delta Mean | State Delta Std | Subgoal Delta Mean | Subgoal Delta Std | No-entry Margin Delta Mean | No-entry Margin Delta Std |
|---|---:|---:|---:|---:|---:|---:|
| `zero_memory` | -0.2600 | 0.0881 | -0.2880 | 0.0909 | -0.9304 | 0.1272 |
| `zero_candidates` | -0.1640 | 0.1245 | -0.1760 | 0.1148 | -0.0028 | 0.0053 |
| `zero_memory_and_candidates` | -0.6400 | 0.1649 | -0.6320 | 0.1695 | -0.9035 | 0.2039 |

结论：

```text
在固定 split 下，换 10 个训练 seed 后，zero_memory 仍然让 no-entry margin 平均下降 0.9304。
candidate ablation 对 no-entry margin 的影响接近 0。
```

### 10.2 不同 Random Split，固定训练 Seed

| Mode | State Acc Mean | State Acc Std | Subgoal Mean | Subgoal Std | No-entry Margin Mean | No-entry Margin Std | True Low Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `none` | 0.9760 | 0.0265 | 0.9640 | 0.0120 | 0.9649 | 0.0242 | 0.0 |
| `zero_memory` | 0.7920 | 0.0392 | 0.7800 | 0.0447 | 0.1623 | 0.1531 | 4.1 |
| `zero_candidates` | 0.9280 | 0.0531 | 0.9160 | 0.0418 | 0.9667 | 0.0244 | 0.0 |
| `zero_memory_and_candidates` | 0.5920 | 0.0909 | 0.6080 | 0.0891 | 0.0612 | 0.2018 | 4.5 |

相对完整输入的平均下降：

| Ablation | State Delta Mean | State Delta Std | Subgoal Delta Mean | Subgoal Delta Std | No-entry Margin Delta Mean | No-entry Margin Delta Std |
|---|---:|---:|---:|---:|---:|---:|
| `zero_memory` | -0.1840 | 0.0480 | -0.1840 | 0.0367 | -0.8026 | 0.1456 |
| `zero_candidates` | -0.0480 | 0.0431 | -0.0480 | 0.0431 | 0.0018 | 0.0089 |
| `zero_memory_and_candidates` | -0.3840 | 0.0824 | -0.3560 | 0.0809 | -0.9037 | 0.1925 |

结论：

```text
在 10 个不同 random split 下，zero_memory 仍然让 no-entry margin 平均下降 0.8026。
zero_candidates 对 no-entry margin 不造成稳定下降。
```

### 10.3 正式重复实验结论

两组正式重复实验共同支持：

```text
memory 是 no-entry full-coverage 判断的必要信息源。
candidate features 更主要服务于当前帧入口候选与局部子目标。
该结论对训练 seed 和 random split 都比较稳定。
```

这比单次消融更适合作为论文实验结果。
