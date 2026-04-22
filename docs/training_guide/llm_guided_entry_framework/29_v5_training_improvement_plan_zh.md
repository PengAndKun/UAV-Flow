# 29. V5 训练改进计划（中文）

## 1. 当前结论

在 `v4 memory-aware` 中，我们已经把 `memory_features` 接进了训练输入。  
对比 `v3 reviewed`，当前结果呈现出一个很清晰的趋势：

- 普通任务头明显提升
- 动作头明显提升
- target-conditioned 候选选择也有提升
- 但 target-conditioned 的高层状态/子任务略有下降

简化对比如下：

| 指标 | v3 | v4 | 变化 |
|---|---:|---:|---:|
| entry_state | 0.606 | 0.788 | +0.182 |
| subgoal | 0.576 | 0.636 | +0.061 |
| action_hint | 0.515 | 0.667 | +0.152 |
| target_conditioned_state | 0.905 | 0.857 | -0.048 |
| target_conditioned_subgoal | 0.810 | 0.762 | -0.048 |
| target_conditioned_action_hint | 0.476 | 0.762 | +0.286 |
| target_conditioned_target_candidate_id | 0.381 | 0.476 | +0.095 |

这说明：

**memory 当前更强地帮助了“低层动作和候选选择”，但还没有完全转化成“更强的高层 target-conditioned 语义判断”。**

## 2. V5 的目标

`v5` 不应该只是“再训一遍”，而应该解决一个明确问题：

**在保留 v4 动作和候选优势的同时，把 target-conditioned 高层判断重新拉回来。**

也就是同时提升：

1. `target_conditioned_state`
2. `target_conditioned_subgoal`
3. 保留 `target_conditioned_action_hint`
4. 保留 `target_conditioned_target_candidate_id`

## 3. 为什么 v4 会出现这种现象

我认为主要有 4 个原因。

### 3.1 memory 现在先拼到了 global branch

这一步很稳，但也有副作用：

- 模型更容易把 memory 当作“动作上下文”
- 而不一定把它学成“高层语义证据”

所以更容易先提升：
- 动作
- 候选 id

### 3.2 当前 loss 仍然不够偏向高层 target-conditioned 任务

虽然当前 selection metric 已经偏向 target-conditioned 任务，  
但训练损失本身对：

- `target_conditioned_state`
- `target_conditioned_subgoal`

的强调还不够强。

### 3.3 稀有类仍然太少

目前最稀缺的类仍然包括：

- `target_house_entry_approachable`
- `approach_target_entry`
- `detour_left_to_target_entry`
- `hold`

这会让模型更容易学会“保守动作”，而不是高质量的 target-conditioned 细分判断。

### 3.4 memory 特征目前是轻量摘要

当前输入的是：

- `observed_sector_count`
- `current_sector_low_yield_flag`
- `last_best_entry_status`
- `previous_action`
- `previous_subgoal`

这对动作有帮助，但对“目标入口到底是不是 approach-worthy”还不够细。

## 4. V5 总体思路

我建议 `v5` 分成两部分：

### Part A：训练侧改进

目标：
- 在不改太多结构的前提下，让高层 target-conditioned 任务权重更高

### Part B：输入侧改进

目标：
- 让 memory 更偏“高层语义记忆”
- 而不只是“动作上下文记忆”

## 5. V5 训练改进项

### 5.1 提高 target-conditioned state/subgoal loss 权重

当前建议：

- `target_state`: 从 `1.2` 提到 `1.5 ~ 1.6`
- `target_subgoal`: 从 `1.2` 提到 `1.5 ~ 1.6`
- `target_action`: 保持 `0.8`
- `target_candidate`: 保持 `0.8` 或略降到 `0.7`

原因：
- 先把“高层对不对”放在“细粒度动作对不对”前面

### 5.2 curriculum 再细化一层

当前是三阶段：

1. 只训状态
2. 加子任务
3. 全头训练

`v5` 建议改成四阶段：

1. `entry_state + target_conditioned_state`
2. `subgoal + target_conditioned_subgoal`
3. `target_candidate_id + target_action_hint`
4. 全头联合微调

这样高层 target-conditioned 语义会在候选动作之前先稳定。

### 5.3 best checkpoint 的 selection metric 再偏高层

建议把 `selection_metric` 再调成：

- `target_conditioned_state`
- `target_conditioned_subgoal`

权重更高，  
让 best checkpoint 优先按“高层 target-conditioned 任务”选，而不是被动作头带偏。

### 5.4 增加 head-level early signal monitoring

建议在 `metrics.json` 里重点盯：

- `acc_target_state`
- `acc_target_subgoal`
- `acc_target_action`
- `acc_target_candidate`

如果发现：
- target action 上升
- 但 target state 持续下降

就说明模型又在“动作先行、语义后退”。

## 6. V5 输入改进项

### 6.1 增加更高层的 memory features

当前建议补这几类特征：

- `memory_target_candidate_exists`
- `memory_target_candidate_blocked_count`
- `memory_target_candidate_rejected_count`
- `memory_current_sector_is_repeated_low_yield`
- `memory_last_decision_override_applied`
- `memory_last_decision_override_reason`

这些比单纯的：
- `previous_action`
- `previous_subgoal`

更接近高层 target-conditioned 语义。

### 6.2 弱化过强的动作型 memory 依赖

如果发现模型太依赖：

- `previous_action`
- `previous_subgoal`

可以做一个小 ablation：

- `v5a`: 全 memory
- `v5b`: 去掉 `previous_action`
- `v5c`: 去掉 `previous_action + previous_subgoal`

看高层 target-conditioned 指标是否反而更稳。

### 6.3 预留 memory branch 升级路线

如果 `v5` 仍然表现出：
- 动作好
- 高层状态一般

那 `v6` 就建议把 memory 从 `global_features` 里拆出来，做成独立 branch：

- `global branch`
- `candidate branch`
- `memory branch`

但 `v5` 还不建议直接跳这一步，先用最小改动验证更合适。

## 7. 数据侧改进项

### 7.1 继续补稀有类

最该继续补：

- `target_house_entry_approachable`
- `approach_target_entry`
- `detour_left_to_target_entry`
- `hold`

### 7.2 新增时序 memory 样本

后续采集时，建议让更多样本带：

- `episode_id`
- `step_index`
- `memory_snapshot_before`
- `memory_snapshot_after`

这样 `working/episodic memory` 后面才能真正学起来。

## 8. V5 实验设计建议

建议至少做 3 组：

### V5-A：Loss Reweight

只改：
- loss 权重
- selection metric

不改输入

目的：
- 看高层 target-conditioned 任务能不能直接回来

### V5-B：Loss Reweight + Memory Feature Enrichment

在 V5-A 基础上，再加高层 memory 特征

目的：
- 看 memory 是否能真正帮助 target-conditioned 语义

### V5-C：Ablation

比较：

1. `v4 memory`
2. `v5-a`
3. `v5-b`

重点看：

- `target_conditioned_state`
- `target_conditioned_subgoal`
- `target_conditioned_action_hint`
- `target_conditioned_target_candidate_id`

## 9. V5 成功标准

我建议把成功标准定成：

1. `target_conditioned_state >= 0.90`
2. `target_conditioned_subgoal >= 0.82`
3. `target_conditioned_action_hint >= 0.75`
4. `target_conditioned_target_candidate_id >= 0.45`
5. 普通 `entry_state` 不低于 `0.75`

也就是说：
- 高层恢复
- 低层不明显退步

## 10. 一句话结论

`v5` 的核心不是“再堆更多 memory”，而是：

**把当前 memory-aware 模型从“动作增强版”继续推向“高层 target-conditioned 语义增强版”。**

这轮最值得先做的，是：

1. 提高 `target_state / target_subgoal` loss 权重  
2. 调整 curriculum  
3. 增加更高层的 semantic-memory features  
4. 保守验证，不急着上独立 memory branch

