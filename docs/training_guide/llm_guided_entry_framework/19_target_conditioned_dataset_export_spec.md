# Target-Conditioned Dataset Export Specification

## 1. 目的

本文件定义：

- `target-conditioned dataset export`

应该如何从现有 `labeling/` 样本包中导出后续可训练的数据集。

目标是让后续训练同时支持：

1. 普通入口相关任务
2. 目标房屋条件入口任务
3. 表示蒸馏
4. 行为克隆 / PPO 初始化

---

## 2. 导出在整条链中的位置

建议的数据链顺序仍然是：

1. `fusion`
2. `teacher_validator`
3. `entry_state_builder`
4. `dataset_export`
5. `representation_distillation_trainer`
6. `policy_trainer`

也就是说，dataset export 不负责重新推理，而负责：

- 统一筛样本
- 统一拆 train/val
- 统一生成训练清单

---

## 3. 输入文件要求

每个 `labeling/` 目录至少需要：

- `entry_state.json`
- `teacher_output.json`
- `teacher_validation.json`

若启用 target-conditioned 版本，还建议存在：

- `fusion_result.json`
- `labeling_manifest.json`
- 原始图像副本
  - `rgb.png`
  - `depth_cm.png`
  - `depth_preview.png`

---

## 4. 导出目标

建议导出两套监督：

### 4.1 普通监督

用于兼容旧实验：

- `entry_state`
- `subgoal`
- `action_hint`
- `target_candidate_id`

### 4.2 目标条件监督

用于新主线：

- `target_conditioned_state`
- `target_conditioned_subgoal`
- `target_conditioned_action_hint`
- `target_conditioned_target_candidate_id`

后续训练建议以：

- 目标条件监督

为主，
普通监督为辅助。

---

## 5. 样本过滤规则

建议导出阶段明确区分三类样本：

### 5.1 `valid`

满足：

- `teacher_validation.status = valid`
- teacher 置信度满足阈值

这些样本可直接进入正式训练。

### 5.2 `weak_valid`

满足：

- `teacher_validation.status = weak_valid`

这些样本建议：

- 默认不进入主训练集
- 可作为附加训练或后续 ablation 使用

### 5.3 `invalid`

满足：

- `teacher_validation.status = invalid`

这些样本应直接过滤。

---

## 6. 推荐阈值

第一版建议：

- `allow_weak_valid = false`
- `min_teacher_confidence = 0.55`
- `min_target_confidence = 0.55`（若已提供）

若后续 target-conditioned teacher 还偏保守，可单独放宽：

- `min_target_confidence = 0.50`

但第一版主实验建议阈值先严一点。

---

## 7. train / val 拆分规则

强烈建议：

- **不要按帧随机拆**

推荐至少按：

- `fusion_xxx run_dir`

拆分，
更进一步可按：

- `house_id`

拆分。

### 第一版推荐

- `train : val = 8 : 2`
- 按 `run_dir` 分组拆分

理由：

- 同一 `run_dir` 内样本相似度高
- 随机打散会导致严重泄漏

---

## 8. 类别均衡策略

当前环境天然会导致类别不平衡，
所以导出器建议保留两层统计：

### 8.1 原始分布

- 所有符合过滤条件的样本分布

### 8.2 实际导出分布

- train/val 中各类样本数量

建议重点统计：

- `target_conditioned_state`
- `target_conditioned_subgoal`
- `target_conditioned_action_hint`

如果某些类极少，例如：

- `target_house_not_in_view`
- `non_target_house_entry_visible`
- `target_house_entry_blocked`

则应在报告中显式标出来，供后续补采。

---

## 9. 推荐导出目录结构

建议沿用当前 export 风格，但新增 target-conditioned 统计：

```text
phase2_5_distillation_dataset_xxx/
├─ manifest.json
├─ quality_report.json
├─ train.jsonl
├─ val.jsonl
├─ train_ids.txt
├─ val_ids.txt
└─ samples/
   ├─ sample_000001/
   │  ├─ entry_state.json
   │  ├─ teacher_output.json
   │  ├─ teacher_validation.json
   │  └─ metadata.json
   └─ ...
```

若后续需要支持 ROI 图像特征蒸馏，可再加：

- `rgb.png`
- `depth_cm.png`

的快捷副本或引用路径。

---

## 10. `manifest.json` 推荐新增字段

当前 `manifest.json` 建议新增：

- `target_conditioned_state_counts`
- `target_conditioned_subgoal_counts`
- `target_conditioned_action_hint_counts`
- `target_conditioned_enabled`

推荐结构：

```json
{
  "export_dir": "...",
  "results_root": "...",
  "total_exported": 0,
  "train_count": 0,
  "val_count": 0,

  "entry_state_counts": {},
  "subgoal_counts": {},
  "action_hint_counts": {},

  "target_conditioned_state_counts": {},
  "target_conditioned_subgoal_counts": {},
  "target_conditioned_action_hint_counts": {},

  "allow_weak_valid": false,
  "min_teacher_confidence": 0.55,
  "target_conditioned_enabled": true
}
```

---

## 11. `quality_report.json` 推荐新增字段

建议新增：

- `skipped_counts_by_target_conditioned_reason`
- `missing_target_conditioned_fields_count`

用于排查：

- 为什么某些样本无法进入新主线训练集

例如：

- 没有 `target_conditioned_state`
- `target_candidate_id` 缺失
- `target_confidence` 太低

---

## 12. `train.jsonl / val.jsonl` 推荐结构

每条样本建议包含 4 块：

### 12.1 `state`

- 即 `entry_state.json` 里的：
  - `global_state`
  - `candidates`

### 12.2 `teacher_targets`

保留完整 teacher targets，
包括：

- 普通 targets
- target-conditioned targets

### 12.3 `metadata`

- `sample_id`
- `run_dir`
- `task_label`
- `house info`
- `source paths`

### 12.4 `quality`

- `teacher_status`
- `teacher_score`
- `teacher_confidence`
- `target_confidence`

---

## 13. 推荐导出样本示例

```json
{
  "sample_id": "fusion_20260414_xxx",
  "state": {
    "global_state": {},
    "candidates": []
  },
  "teacher_targets": {
    "entry_state": "enterable_open_door",
    "subgoal": "approach_entry",
    "action_hint": "forward",

    "target_conditioned_state": "target_house_entry_approachable",
    "target_conditioned_subgoal": "approach_target_entry",
    "target_conditioned_action_hint": "forward",
    "target_conditioned_target_candidate_id": 0
  },
  "metadata": {
    "run_dir": "...",
    "task_label": "...",
    "target_house_id": 1
  },
  "quality": {
    "teacher_status": "valid",
    "teacher_score": 0.92,
    "teacher_confidence": 0.88,
    "target_confidence": 0.85
  }
}
```

---

## 14. 导出器对旧样本的兼容策略

为了兼容旧样本，建议：

### 若缺失 target-conditioned teacher 字段

则：

- 不报错
- 标记：
  - `target_conditioned_enabled = false`
  - 或该样本只进入旧任务训练集

### 若新实验显式要求 target-conditioned 样本

则：

- 过滤掉这类旧样本

也就是说，导出器应支持两种模式：

1. `legacy_or_target_conditioned`
2. `target_conditioned_only`

---

## 15. 第一版推荐导出模式

我建议第一版新增一个导出模式：

- `target_conditioned_only = false` 默认

先让旧样本还能进来，
但在 manifest 中明确标记：

- 哪些样本有 target-conditioned supervision
- 哪些没有

等 target-conditioned 样本足够多，再切到：

- `target_conditioned_only = true`

---

## 16. 与后续训练的关系

### 16.1 表示蒸馏训练

建议使用：

- 普通 targets
- target-conditioned targets

做多头训练。

### 16.2 policy 训练

建议优先使用：

- `target_conditioned_subgoal`
- `target_conditioned_action_hint`

因为它们更符合真实任务目标。

### 16.3 实验比较

后续可以直接做：

- `legacy export`
  vs
- `target-conditioned export`

比较二者在任务完成率上的差异。

---

## 17. 第一版实现建议

第一版导出器升级时，建议只做：

1. 新增 target-conditioned 统计
2. 新增 target-conditioned targets 导出
3. 保持原有目录结构不变
4. 保持原有 JSONL 主结构不变

不要第一版就：

- 改整个 export 目录组织
- 拆多种复杂任务包

先保证：

- 原脚本小改即可跑通

---

## 18. 一句话总结

target-conditioned dataset export 的核心不是重新发明一套导出系统，而是：

- **在现有导出链上，把目标房屋条件监督显式带进去**

这样后续训练才能真正学习：

- 不是“看到门就走”
- 而是“看到目标房屋的门才走”

