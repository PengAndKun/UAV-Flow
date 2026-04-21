# 27. Memory Snapshot Export 规范（中文）

## 1. 这份文档解决什么问题

在 [26_memory_aware_data_collection_spec_zh.md](26_memory_aware_data_collection_spec_zh.md) 里，我们已经明确：

- 旧数据主要通过顺序回放重建 `semantic_memory`
- 新数据需要按真正时序采集
- 每条样本应逐步带上：
  - `episode_id`
  - `step_index`
  - `previous_action`
  - `memory_snapshot`

但如果后续要真正让训练器使用 memory，还需要解决一个更具体的问题：

**这些 memory snapshot 到底怎样进入 `entry_state_builder` 和导出链？**

这份文档专门定义：

1. `before / after memory snapshot` 的推荐文件结构
2. `entry_state_builder` 应读取哪些字段
3. `distillation_dataset_export` 应导出哪些 memory 特征
4. 哪些字段是第一阶段必须导出，哪些可以后续再增强

---

## 2. 设计目标

### 2.1 我们不是把整份 memory JSON 原样喂给模型

运行时的完整 memory 文件适合：

- 系统决策
- 回放
- 调试

但训练时更适合：

- 从完整 memory snapshot 中抽取结构化特征
- 再导出成固定长度或可控长度的训练字段

所以这里的总体策略是：

### 存储层

保存完整：

- `entry_search_memory_snapshot_before.json`
- `entry_search_memory_snapshot_after.json`

### 训练层

只导出经过整理的：

- `memory_features`
- `memory_targets`
- `memory_metadata`

---

## 3. 推荐的目录级文件结构

对于每个：

- `phase2_multimodal_fusion_analysis/results/fusion_xxx/labeling/`

后续推荐新增或稳定保存：

- `sample_metadata.json`
- `temporal_context.json`
- `entry_search_memory_snapshot_before.json`
- `entry_search_memory_snapshot_after.json`

推荐职责如下：

### `sample_metadata.json`

保存样本级别元信息：

```json
{
  "episode_id": "house_001_entry_search_ep_0007",
  "step_index": 12,
  "sample_timestamp": 1776742201.52,
  "task_label": "search target house entry",
  "current_house_id": "",
  "target_house_id": "001"
}
```

### `temporal_context.json`

保存与前一时刻相关的上下文：

```json
{
  "previous_action": "yaw_left",
  "previous_target_conditioned_state": "target_house_entry_visible",
  "previous_target_conditioned_subgoal": "keep_search_target_house",
  "previous_target_conditioned_action_hint": "yaw_left",
  "previous_best_candidate_id": "0"
}
```

### `entry_search_memory_snapshot_before.json`

保存当前样本处理前的三层 memory：

- `working_memory`
- `episodic_memory`
- `semantic_memory`

### `entry_search_memory_snapshot_after.json`

保存当前样本处理后的三层 memory。

---

## 4. `before / after` snapshot 的作用分工

### 4.1 `before snapshot`

更适合表达：

- 当前决策时，系统已经记住了什么

也就是说，它更适合作为：

- 模型输入的一部分

### 4.2 `after snapshot`

更适合表达：

- 当前处理完之后，memory 被更新成了什么

它更适合作为：

- 后续回放
- memory update 监督
- 训练后分析

### 4.3 第一阶段建议

如果只做第一版 memory-aware 训练，我建议：

- `entry_state_builder` 先重点读取 `before snapshot`
- `export` 里保留 `after snapshot` 路径和摘要

这样最稳，因为：

- 决策使用 `before`
- 分析与未来增强使用 `after`

---

## 5. `entry_state_builder` 应如何接 memory snapshot

## 5.1 新增输入

后续 `entry_state_builder.py` 读取每条样本时，应优先尝试读取：

- `entry_search_memory_snapshot_before.json`
- `entry_search_memory_snapshot_after.json`

如果没有，再退化为：

- 当前目录引用的 `entry_search_memory.json`
- 或旧版仅 `semantic_memory` 摘要

### 5.2 `entry_state.json` 中建议新增的结构

推荐新增顶层字段：

```json
{
  "memory_context": {
    "available": true,
    "source": "before_snapshot",
    "episode_id": "house_001_entry_search_ep_0007",
    "step_index": 12,
    "semantic_memory": {},
    "working_memory": {},
    "memory_features": {}
  }
}
```

其中：

- `semantic_memory` 不一定要完整展开
- `working_memory` 也不一定完整展开
- 训练真正消费的是 `memory_features`

---

## 6. 第一阶段推荐导出的 `memory_features`

为了兼顾稳定性和可训练性，我建议第一阶段先只导出：

### 6.1 `semantic_memory` 特征

```json
{
  "observed_sector_count": 2,
  "entry_search_status": "searching_entry",
  "last_best_entry_exists": 1,
  "candidate_entry_count": 3,
  "approachable_entry_count": 0,
  "blocked_entry_count": 1,
  "rejected_entry_count": 2,
  "current_sector_observation_count": 3,
  "current_sector_low_yield_flag": 1,
  "current_sector_best_target_match_score": 0.03,
  "last_best_entry_status": "blocked_temporary",
  "last_best_entry_attempt_count": 4
}
```

这些字段最适合当前阶段，因为：

- 和现在已经接好的 memory-aware fusion 高度一致
- 对旧数据也比较容易重建
- 不会因为缺完整 episode 而太脆弱

### 6.2 `working_memory` 特征

第一阶段先导轻量摘要，不直接导整个动作序列：

```json
{
  "recent_action_count": 3,
  "previous_action": "yaw_left",
  "previous_subgoal": "keep_search_target_house",
  "previous_best_candidate_id": "0"
}
```

### 6.3 不建议第一阶段直接导出的内容

暂时不要一开始就直接导：

- 完整 `episodic_memory` 列表
- 全量 snapshot 文本
- 长序列 action history

因为：

- 旧数据不完整
- 新数据还在补采
- 容易让训练结构过重

---

## 7. `distillation_dataset_export` 应该怎样导出

## 7.1 JSONL 记录中建议新增的字段

后续每条导出样本建议新增：

```json
{
  "memory_available": true,
  "memory_source": "before_snapshot",
  "memory_snapshot_before_path": "...",
  "memory_snapshot_after_path": "...",
  "memory_features": {
    "observed_sector_count": 2,
    "entry_search_status": "searching_entry",
    "last_best_entry_exists": 1,
    "candidate_entry_count": 3,
    "blocked_entry_count": 1,
    "rejected_entry_count": 2,
    "current_sector_observation_count": 3,
    "current_sector_low_yield_flag": 1,
    "last_best_entry_status": "blocked_temporary",
    "last_best_entry_attempt_count": 4,
    "previous_action": "yaw_left",
    "previous_subgoal": "keep_search_target_house"
  }
}
```

### 7.2 `manifest.json` 中建议新增的统计

```json
{
  "memory_snapshot_before_count": 0,
  "memory_snapshot_after_count": 0,
  "memory_feature_available_count": 0,
  "memory_source_counts": {
    "before_snapshot": 0,
    "offline_replay_semantic_only": 0,
    "none": 0
  }
}
```

这样后面你一眼就能看出：

- 当前训练集到底有多少样本真正带 memory
- 哪些只是离线回放近似 memory

---

## 8. 建议增加的 `memory_targets`

除了 `memory_features`，后续还可以增加一组可选的 `memory_targets`，用于训练更强的 memory-aware 学生模型。

第一阶段建议先留接口，不强制启用。

推荐字段例如：

```json
{
  "memory_targets": {
    "should_shift_sector": 1,
    "should_stop_retry_blocked_entry": 1,
    "preferred_alternate_sector": "front_left",
    "last_best_entry_consistency": 1
  }
}
```

这些 target 的价值在于：

- 可以显式监督模型学会“何时该换扇区”
- 可以显式监督模型学会“何时不该再盯 blocked 门”

但这一步建议在 memory snapshot 流程稳定后再启用。

---

## 9. 向后兼容策略

因为当前数据集不是所有样本都已经带上了 memory snapshot，所以必须设计兼容模式。

### 9.1 推荐兼容优先级

`entry_state_builder` / `export` 读取顺序建议：

1. `entry_search_memory_snapshot_before.json`
2. `entry_search_memory_snapshot_after.json`
3. 样本处理时落下的 `entry_search_memory` 摘要
4. 没有 memory，则标记 `memory_available = false`

### 9.2 兼容输出建议

即使没有 snapshot，也建议保留统一字段：

```json
{
  "memory_available": false,
  "memory_source": "none",
  "memory_features": {}
}
```

这样训练器的输入接口可以稳定，不会因为有无 memory 样本而反复改。

---

## 10. 第一阶段推荐做法

如果我们要把这套方案尽快落地，我建议第一阶段只做下面这些：

### Step A

在采样或重处理流程里，为每个样本新增：

- `sample_metadata.json`
- `temporal_context.json`
- `entry_search_memory_snapshot_after.json`

### Step B

在 `entry_state_builder.py` 中先读取：

- `entry_search_memory_snapshot_after.json`

并提取轻量 `semantic_memory` 特征。

### Step C

在 `distillation_dataset_export.py` 中导出：

- `memory_available`
- `memory_source`
- `memory_snapshot_after_path`
- `memory_features`

### Step D

等新的 episode 式采样数据积累起来后，再升级到：

- `before + after snapshot`
- `working_memory` 摘要
- `memory_targets`

---

## 11. 不建议现在立刻做的事

这几件事现在不建议马上上：

1. 把整份 `episodic_memory` 原样喂给训练器
2. 直接做长序列 Transformer memory encoder
3. 假设旧静态数据天然有完整时序
4. 强行把所有样本都补齐完整 before/after 真时序

原因不是做不到，而是：

- 现在最重要的是让 memory-aware 数据链先稳定
- 然后再逐步增加复杂度

---

## 12. 一句话结论

后续 `memory snapshot` 最合理的接法是：

- **存储层**：保存完整 `before/after memory snapshot`
- **状态层**：在 `entry_state_builder` 中抽取轻量 `memory_features`
- **导出层**：在 `export` 中统一导出 `memory_features + memory_paths + memory_source`

第一阶段先聚焦：

- `semantic_memory`
- 轻量 `working_memory` 摘要

不要一开始就把完整 episodic 长序列塞进训练器。

