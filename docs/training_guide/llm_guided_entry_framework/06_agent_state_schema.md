# Agent State Schema

## 1. 文档目标

这份文档用于固定 `Phase 2.5` 中：

- `LLM-guided teacher` 的输出结构
- `student policy` 的输入状态结构
- `YOLO candidate + depth ROI + LLM distilled signals` 的统一样本格式

目标不是写论文叙述，而是给后续代码实现提供一个**稳定的数据接口规范**。

---

## 2. 总体设计原则

### 2.1 不重新做整图语义学习

当前系统已经有：

- `YOLO26`
  - 检测 `open door / door / close door / window`
- depth 几何分析
  - 计算前障碍、距离、宽度、可穿越性
- fusion 判断
  - 输出入口状态

因此 student 不应该再从整张 RGB 图重新学习“哪里是门”。

更合理的做法是：

1. `YOLO` 先提出候选区域
2. `depth` 对候选区域做几何验证
3. `LLM teacher` 提供高层语义指导
4. student 学会：
   - 选哪个候选
   - 当前该靠近、绕行、搜索，还是穿过

### 2.2 采用候选级状态，而不是纯整图状态

每一帧状态由三部分组成：

1. 全局状态 `global_state`
2. 候选入口列表 `candidates`
3. teacher 蒸馏信号 `teacher_targets`

### 2.3 固定 top-K

建议每帧只保留前 `K=3` 个候选。

排序规则建议：

1. `open door`
2. `door`
3. `close door`
4. `window`
5. 按 confidence 降序

如果候选不足 `K`，用零填充。

---

## 3. Teacher Output Schema

teacher 的输出应该固定为结构化 JSON，而不是松散文本。

推荐 schema：

```json
{
  "entry_state": "enterable_open_door",
  "subgoal": "approach_entry",
  "action_hint": "forward",
  "target_candidate_id": 0,
  "risk_level": "low",
  "reason": "Open door is visible and traversable but still far, approach first.",
  "confidence": 0.84
}
```

### 3.1 字段定义

#### `entry_state`

类型：
- `string`

取值集合：
- `enterable_open_door`
- `enterable_door`
- `visible_but_blocked_entry`
- `front_blocked_detour`
- `window_visible_keep_search`
- `geometric_opening_needs_confirmation`
- `no_entry_confirmed`

#### `subgoal`

类型：
- `string`

取值集合：
- `keep_search`
- `approach_entry`
- `align_entry`
- `detour_left`
- `detour_right`
- `cross_entry`
- `backoff_and_reobserve`

#### `action_hint`

类型：
- `string`

取值集合：
- `forward`
- `yaw_left`
- `yaw_right`
- `left`
- `right`
- `backward`
- `hold`

#### `target_candidate_id`

类型：
- `int`

含义：
- 当前 top-K 候选中，teacher 认为最重要的候选索引

范围：
- `0 ~ K-1`
- 如果没有明确候选，可用 `-1`

#### `risk_level`

类型：
- `string`

取值集合：
- `low`
- `medium`
- `high`

#### `reason`

类型：
- `string`

要求：
- 一句短解释
- 长度建议不超过 `24` 个英文词或对应短中文句

#### `confidence`

类型：
- `float`

范围：
- `[0, 1]`

---

## 4. Teacher Reason Embedding

### 4.1 作用

`reason` 文本本身适合人看，但不适合直接给 student。

因此建议将 `reason` 编码成固定维度向量：

- `teacher_reason_embedding`

### 4.2 推荐方案

推荐使用一个稳定的文本编码器，而不是直接依赖 LLM 内部 hidden state。

建议形式：

1. teacher 生成 `reason`
2. 用固定文本编码器得到 embedding
3. 缓存到样本文件里

### 4.3 推荐维度

建议：

- 第一版：`128` 或 `256` 维

不建议第一版就上太大维度。

### 4.4 推荐存储方式

每条样本存：

- `teacher_reason_text`
- `teacher_reason_embedding`

例如：

```json
{
  "teacher_reason_text": "Open door is visible and traversable but still far, approach first.",
  "teacher_reason_embedding": [0.12, -0.03, ...]
}
```

---

## 5. Student Input State Schema

student 输入状态建议为：

```json
{
  "global_state": { ... },
  "candidates": [ ... ],
  "teacher_targets": { ... }
}
```

---

## 6. Global State

### 6.1 字段定义

```json
{
  "pose_x": 0.0,
  "pose_y": 0.0,
  "pose_z": 0.0,
  "yaw_deg": 0.0,
  "front_obstacle_present": 0,
  "front_min_depth_cm": 230.0,
  "front_obstacle_severity": "clear",
  "target_house_id": 1,
  "current_house_id": 1,
  "target_distance_cm": 620.0,
  "target_bearing_deg": -18.0,
  "movement_enabled": 1,
  "history_actions": ["forward", "yaw_left"]
}
```

### 6.2 必选字段

- `pose_x`
- `pose_y`
- `pose_z`
- `yaw_deg`
- `front_obstacle_present`
- `front_min_depth_cm`
- `target_house_id`
- `current_house_id`

### 6.3 可选字段

- `target_distance_cm`
- `target_bearing_deg`
- `movement_enabled`
- `history_actions`

### 6.4 归一化建议

- `yaw_deg`
  - 转成 `sin(yaw), cos(yaw)` 更稳
- `front_min_depth_cm`
  - 除以 `1200`
- `target_distance_cm`
  - 除以场景最大有效距离，如 `2000`
- `target_bearing_deg`
  - 转成 `sin/cos`

---

## 7. Candidate State

每个候选使用统一结构。

```json
{
  "candidate_id": 0,
  "class_name": "open door",
  "class_onehot": [1, 0, 0, 0],
  "confidence": 0.89,
  "bbox_cx": 0.53,
  "bbox_cy": 0.44,
  "bbox_w": 0.18,
  "bbox_h": 0.39,
  "bbox_area_ratio": 0.0702,
  "aspect_ratio": 0.46,
  "entry_distance_cm": 280.0,
  "surrounding_depth_cm": 190.0,
  "clearance_depth_cm": 310.0,
  "depth_gain_cm": 90.0,
  "opening_width_cm": 112.0,
  "traversable": 1,
  "crossing_ready": 1,
  "candidate_rank": 0
}
```

### 7.1 YOLO 语义部分

- `class_name`
- `class_onehot`
- `confidence`
- `bbox_cx`
- `bbox_cy`
- `bbox_w`
- `bbox_h`
- `bbox_area_ratio`
- `aspect_ratio`
- `candidate_rank`

### 7.2 Depth ROI 几何部分

- `entry_distance_cm`
- `surrounding_depth_cm`
- `clearance_depth_cm`
- `depth_gain_cm`
- `opening_width_cm`
- `traversable`
- `crossing_ready`

### 7.3 归一化建议

- `confidence`
  - 保持 `[0, 1]`
- `bbox_*`
  - 全部归一化到 `[0, 1]`
- `entry_distance_cm`
  - 除以 `1200`
- `surrounding_depth_cm`
  - 除以 `1200`
- `clearance_depth_cm`
  - 除以 `1200`
- `depth_gain_cm`
  - 除以 `400`
- `opening_width_cm`
  - 除以 `300`
- `traversable`
  - `0/1`
- `crossing_ready`
  - `0/1`

### 7.4 候选缺失时的填充

当候选数少于 `K` 时：

- `candidate_id = -1`
- 所有连续值填 `0`
- `class_onehot = [0, 0, 0, 0]`
- `valid_mask = 0`

建议每个候选再加：

- `valid_mask`

---

## 8. Teacher Targets

teacher targets 是训练时使用的目标，不一定全部在推理时在线提供。

```json
{
  "entry_state": "enterable_open_door",
  "subgoal": "approach_entry",
  "action_hint": "forward",
  "target_candidate_id": 0,
  "risk_level": "low",
  "teacher_reason_text": "Open door is visible and traversable but still far, approach first.",
  "teacher_reason_embedding": [ ... ]
}
```

### 8.1 训练时使用

用于：

- `entry_state` 分类损失
- `subgoal` 分类损失
- `action_hint` 分类损失
- `target_candidate_id` 分类损失
- `teacher_reason_embedding` 蒸馏损失

### 8.2 推理时使用

推理阶段通常不直接输入完整 teacher targets，
而是使用训练好的 student 自己预测：

- 当前 entry state
- 当前 subgoal
- 当前 action hint

---

## 9. 单条样本推荐格式

推荐每条样本保存成：

```json
{
  "sample_id": "fusion_20260409_101400",
  "frame_id": "step_0001",
  "rgb_path": ".../rgb.png",
  "depth_cm_path": ".../depth_cm.png",
  "depth_preview_path": ".../depth_preview.png",
  "global_state": { ... },
  "candidates": [ ... ],
  "teacher_targets": { ... },
  "metadata": {
    "task_label": "search the house for people",
    "target_house_id": 1,
    "current_house_id": 1
  }
}
```

---

## 10. 推荐网络接口

student policy 网络建议输入：

- `global_state_tensor`
- `candidate_tensor`
- 可选：
  - `candidate_rgb_roi_tensor`
  - `candidate_depth_roi_tensor`

输出建议至少有 4 个 head：

1. `entry_state_head`
2. `subgoal_head`
3. `action_hint_head`
4. `policy_head`

可选第 5 个：

5. `semantic_projection_head`
   - 对齐 `teacher_reason_embedding`

---

## 11. 第一版最小实现建议

第一版不要一下做太复杂。

建议先做：

1. 不加 ROI encoder
2. 只用结构化状态
3. 只蒸馏：
   - `entry_state`
   - `subgoal`
   - `action_hint`
4. `teacher_reason_embedding` 作为第二阶段增强

也就是说，第一版 student 输入就可以是：

- `global_state`
- `topK candidates`

这已经足够验证：

- 候选级状态表示是不是有效
- LLM guidance 是否能被 student 学到

---

## 12. 后续代码实现建议

后面写代码时，建议对应拆成 3 个模块：

1. `teacher_schema.py`
   - 定义 teacher 输出字段和类别映射

2. `entry_state_builder.py`
   - 从 `YOLO + depth + fusion + pose` 构建 student 输入状态

3. `distillation_dataset_export.py`
   - 把现有样本包导出成训练可读的 JSONL / NPZ

---

## 13. 一句话总结

这一步的核心不是重新训练一个整图视觉 backbone，
而是：

- 用 `YOLO` 先找候选
- 用 `depth` 验证候选的几何可行性
- 用 `LLM teacher` 提供高层语义指导
- 把三者蒸馏成一个可部署的候选级 student 表示
