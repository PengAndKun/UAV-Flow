# 26. Memory-Aware 数据采集规范（中文）

## 1. 这份文档解决什么问题

前面我们已经把 `entry_search_memory` 的运行时逻辑接进了：

- `fusion_entry_analysis.py`
- `entry_search_memory.py`

而且已经验证了 5 类记忆作用：

1. 历史被拒绝候选降权
2. `last_best_entry_id` 跟踪加权
3. 重复低收益 sector 降权
4. 低收益 sector 的高层换边搜索
5. 持续 blocked 候选的高层转向搜索

但是这里还有一个现实问题：

**记忆本质上需要时序。**

如果早期采集的数据主要是很多“彼此独立的静态帧”，那么：

- `semantic_memory` 可以部分重建
- `working_memory` 很难准确重建
- `episodic_memory` 更难完整重建

所以这份文档的目标是把下面三件事固定下来：

1. 旧数据怎样做“近似时序重建”
2. 新数据以后应该怎样按真正时序采
3. 训练前怎样把 memory snapshot 带进数据集

---

## 2. 当前阶段我们怎么处理“旧数据没有记忆”这个问题

### 2.1 现在已经解决的部分

当前系统已经支持：

- 按顺序回放 `results/fusion_*/`
- 每处理一条样本，就更新一次 `entry_search_memory.json`
- 下一条样本再读取这份 memory

这意味着：

- 旧数据可以重建出一份**近似的 `semantic_memory`**
- 并且这份近似 memory 已经能真实影响 fusion 和 target-conditioned 决策

这也是我们目前已经实际验证通过的部分。

### 2.2 现在还没有完美解决的部分

旧数据如果缺少以下字段，就无法完美重建完整记忆：

- `episode_id`
- `step_index`
- `sample_timestamp`
- `previous_action`
- `previous_memory_snapshot`

所以我们现在要明确一个边界：

### 当前过渡策略

- **在线运行时 memory：真时序**
- **旧离线数据 memory：顺序回放重建的近似时序**

这不是缺陷，而是当前阶段合理的工程折中。

---

## 3. 为什么多模态记忆一定要带时序

你的系统不是单纯分类任务，而是：

1. 找目标房屋
2. 找目标房屋入口
3. 判断入口能不能进
4. 记住哪些位置已经看过
5. 不重复试同一个 blocked 候选

这些都依赖“前一帧做过什么、当前帧又看到什么”。

因此 memory-aware 数据至少要能回答下面 4 个问题：

1. 这条样本属于哪个连续 episode？
2. 这条样本在 episode 中是第几步？
3. 前一时刻系统记住了什么？
4. 当前样本之后 memory 又如何更新？

如果没有这 4 个问题的答案，那么模型只能学到：

- 单帧感知

而不能稳定学到：

- “记住自己已经看过什么”
- “为什么现在要换扇区”
- “为什么不能再盯这扇 blocked 门”

---

## 4. 推荐采用的 memory-aware 数据采集结构

### 4.1 顶层组织单位

以后每条采样数据不再只看成独立样本，而应该属于：

- `episode`
- `step`

推荐组织结构：

```text
episode/
  step_0001/
  step_0002/
  step_0003/
```

或者保留当前 `fusion_*/labeling/` 风格，但必须在元数据里加：

- `episode_id`
- `step_index`

### 4.2 每条样本必须新增的核心字段

推荐每个样本最少保存：

```json
{
  "episode_id": "house_001_entry_search_20260421_01",
  "step_index": 12,
  "sample_timestamp": 1776742201.52,
  "task_label": "search house entry",
  "current_house_id": "",
  "target_house_id": "001",
  "previous_action": "yaw_left",
  "previous_target_conditioned_subgoal": "keep_search_target_house",
  "memory_snapshot_path": "labeling/entry_search_memory_snapshot.json"
}
```

其中最重要的是：

- `episode_id`
- `step_index`
- `previous_action`
- `memory_snapshot_path`

---

## 5. 每条样本需要保存哪种 memory snapshot

### 5.1 推荐保存完整 memory snapshot

每次处理完一条样本，都建议在该样本目录里保存：

- `entry_search_memory_snapshot_before.json`
- `entry_search_memory_snapshot_after.json`

如果第一版想轻一点，至少保留：

- `entry_search_memory_snapshot.json`

默认建议先保存 **after snapshot**，因为它更适合后面做监督和回放。

### 5.2 为什么要存 before / after

如果只存当前帧感知结果，那么你只能学：

- 当前看到了什么

如果同时有 before / after，就能学：

- 当前输入如何改变 memory
- 当前决策为什么发生变化

这对以后想训练：

- memory-aware policy
- memory update module

会非常关键。

---

## 6. 推荐的目录级文件新增方案

对于当前每个：

- `phase2_multimodal_fusion_analysis/results/fusion_xxx/labeling/`

建议后续新增这些文件：

- `sample_metadata.json`
- `entry_search_memory_snapshot_before.json`
- `entry_search_memory_snapshot_after.json`
- `temporal_context.json`

推荐含义如下：

### `sample_metadata.json`

记录：

- `episode_id`
- `step_index`
- `sample_timestamp`
- `current_house_id`
- `target_house_id`
- `task_label`

### `temporal_context.json`

记录：

- `previous_action`
- `previous_target_conditioned_state`
- `previous_target_conditioned_subgoal`
- `previous_target_conditioned_action_hint`
- `previous_best_candidate_id`

### `entry_search_memory_snapshot_before.json`

记录当前样本处理前的三层 memory。

### `entry_search_memory_snapshot_after.json`

记录当前样本处理后的三层 memory。

---

## 7. 新数据采集时应该怎样落地

### 7.1 从现在开始，采样必须按 episode 采

不要再把新样本全部当成完全独立的静态帧。

建议每次任务都定义：

- `episode_id`
- `target_house_id`
- `entry-search mission type`

例如：

- `house_001_entry_search_ep_0007`

然后这个 episode 下连续保存：

- 起始朝向
- 观察 sector 的变化
- 候选门的变化
- 每一步动作
- 每一步 memory snapshot

### 7.2 每一步都建议保存的最小字段

每一步至少保存：

- 当前 `rgb.png`
- 当前 `depth_cm.png`
- 当前 `fusion_result.json`
- 当前 `teacher_output.json`（如果有）
- 当前 `entry_search_memory_snapshot_after.json`
- `sample_metadata.json`
- `temporal_context.json`

这样后面无论做：

- teacher 监督
- memory-aware state builder
- representation distillation

都不会再缺时序支撑。

---

## 8. 旧数据应该怎样补救

### 8.1 旧数据不推倒重来

之前的 `results/` 不需要作废。

建议把旧数据分两层使用：

#### 第一层：继续作为单帧感知 + semantic memory 数据

旧数据仍然适合：

- 多模态融合
- target-conditioned fusion
- teacher validator
- entry state builder
- semantic memory aware 训练

#### 第二层：只做“近似 episode 重建”

可按以下方式重建：

1. 按目录时间排序
2. 按 `task_label + target_house_id` 分组
3. 尽量推断相邻样本是否属于同一个 entry-search episode

这个重建版本只建议用于：

- `semantic_memory`
- 少量 `working_memory`

不要把它当成完整 `episodic_memory` 真值。

### 8.2 旧数据推荐新增的兼容字段

即使不能完整重建 episode，也建议给旧数据补：

- `pseudo_episode_id`
- `pseudo_step_index`
- `memory_snapshot_path`
- `memory_reconstruction_mode`

例如：

```json
{
  "pseudo_episode_id": "replay_house_001_20260413",
  "pseudo_step_index": 5,
  "memory_snapshot_path": "labeling/entry_search_memory_snapshot.json",
  "memory_reconstruction_mode": "offline_replay_semantic_only"
}
```

这样后面训练器就能区分：

- 真时序样本
- 离线重建样本

---

## 9. 训练时怎样使用这些 memory 数据

### 9.1 第一阶段建议

第一阶段训练优先使用：

- `semantic_memory`

转成特征后可包括：

- `observed_sector_count`
- `current_sector_observation_count`
- `current_sector_low_yield_flag`
- `blocked_entry_count`
- `rejected_entry_count`
- `approachable_entry_count`
- `last_best_entry_exists`
- `last_best_entry_status`
- `blocked_attempt_count`

### 9.2 第二阶段建议

当新数据里已经稳定带上：

- `episode_id`
- `step_index`
- `previous_action`
- `memory_snapshot_before/after`

再逐步把下面这些带进训练：

- `working_memory.recent_actions`
- `working_memory.recent_target_decisions`
- `episodic_memory` 的检索摘要

### 9.3 为什么不建议一开始就吃完整 episodic memory

因为：

- 旧数据没有完整 episodic 真值
- 新数据的 episodic 采样刚开始积累
- 一开始强行接入，会让模型学得很散

所以更稳的顺序是：

1. 单帧 + `semantic_memory`
2. 单帧 + `semantic_memory + short working_memory`
3. 再考虑 episodic retrieval

---

## 10. 推荐的字段优先级

### P0：必须补

- `episode_id`
- `step_index`
- `sample_timestamp`
- `target_house_id`
- `memory_snapshot_path`

### P1：强烈建议补

- `previous_action`
- `previous_target_conditioned_subgoal`
- `previous_best_candidate_id`
- `entry_search_memory_snapshot_before.json`
- `entry_search_memory_snapshot_after.json`

### P2：后续增强

- `episodic_snapshot_ids`
- `retrieved_memory_summary`
- `working_memory_summary_vector`

---

## 11. 当前我们已经做到哪一步

截至当前阶段，系统已经实现：

1. `entry_search_memory.json` 的运行时读写
2. `semantic_memory` 接入 fusion
3. 候选级 memory-aware 排序
4. 高层 target-conditioned memory-aware override
5. 顺序回放式的 memory 重建

但还没有完全实现：

1. 每条样本自动保存 before / after memory snapshot
2. 数据集级 `episode_id / step_index` 强制字段
3. 训练器直接读取 `memory_snapshot`

所以这份文档定义的就是接下来要补的标准。

---

## 12. 一句话结论

对当前这个多模态融合入口搜索任务，最合理的 memory-aware 数据策略是：

- **旧数据：顺序回放重建 `semantic_memory`**
- **新数据：从现在开始按 `episode + step + memory_snapshot` 真时序采**
- **训练：先用 `semantic_memory`，再逐步接 `working/episodic`**

不要试图把早期静态数据强行包装成完整长时记忆真值。  
正确做法是：

- 承认旧数据的边界
- 用它继续训练 semantic-memory-aware 模型
- 同时把新采样流程升级成真正的时序 memory-aware 数据链

