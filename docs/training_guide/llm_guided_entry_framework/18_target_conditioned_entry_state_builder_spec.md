# Target-Conditioned Entry State Builder Specification

## 1. 目的

本文件定义：

- `entry_state_builder`

在引入 `target-conditioned fusion` 和 `target-conditioned teacher` 之后，应该如何构建新的训练状态。

它回答的核心问题是：

- 后续 student / policy 到底应该看到什么状态，才能学会“找目标房屋入口并进入”，而不是只会“看到门就靠近”。

---

## 2. 设计目标

新的 `entry_state.json` 必须同时满足：

1. **保留原有普通入口信息**
   - 兼容旧实验和旧 debug

2. **显式加入目标房屋约束**
   - 让状态能表达“这个候选是不是目标房屋的候选”

3. **支持 teacher 蒸馏**
   - 能直接监督：
     - 普通 teacher 目标
     - target-conditioned teacher 目标

4. **支持后续 RL / BC**
   - 输入必须是固定长度、可训练、可归一化的结构

---

## 3. 新的总体结构

推荐保留当前 `entry_state.json` 的四层结构，但扩展字段：

```json
{
  "sample_id": "",
  "global_state": {},
  "candidates": [],
  "teacher_targets": {},
  "metadata": {}
}
```

变化重点在：

- `global_state`
- `candidates`
- `teacher_targets`

---

## 4. `global_state` 新字段

旧版 `global_state` 主要表达：

- pose
- 前方障碍
- target distance / bearing

target-conditioned 版本建议新增：

### 4.1 目标房屋可见性

- `target_house_in_fov`
  - `0/1`

- `target_house_expected_side`
  - onehot 或离散 id
  - 建议编码：
    - `left = 0`
    - `center = 1`
    - `right = 2`
    - `out_of_view = 3`

- `target_house_expected_image_x`
  - 若可估计，归一化到 `[0, 1]`
  - 若未知可填 `0.5` 并配合 mask

### 4.2 目标房屋关系

- `target_house_distance_cm`
- `target_house_distance_norm`
- `target_house_bearing_deg`
- `target_house_bearing_sin`
- `target_house_bearing_cos`

### 4.3 目标约束 mask

- `target_house_known_mask`
  - 当前样本是否有明确目标房屋信息

- `target_house_bbox_available_mask`
  - 当前目标房屋是否有 `map_bbox_image`

---

## 5. `candidates` 新字段

这是本次升级最关键的部分。

每个候选不仅要表示：

- 它是什么
- 它能不能过

还要表示：

- 它是不是目标房屋的候选

### 5.1 保留的旧字段

每个候选继续保留：

- `candidate_id`
- `valid_mask`
- `class_name`
- `class_onehot`
- `confidence`
- `bbox_cx`
- `bbox_cy`
- `bbox_w`
- `bbox_h`
- `bbox_area_ratio`
- `aspect_ratio`
- `entry_distance_cm`
- `surrounding_depth_cm`
- `clearance_depth_cm`
- `depth_gain_cm`
- `opening_width_cm`
- `traversable`
- `crossing_ready`

### 5.2 新增目标条件字段

建议新增：

- `candidate_target_match_score`
- `candidate_semantic_score`
- `candidate_geometry_score`
- `candidate_total_score`

- `candidate_house_id`
- `candidate_is_target_house_entry`
- `candidate_target_side_match`
- `candidate_center_in_target_bbox`
- `candidate_near_target_bbox`

#### 推荐编码

- `candidate_target_match_score`
  - `[0,1]`

- `candidate_semantic_score`
  - `[0,1]`

- `candidate_geometry_score`
  - `[0,1]`

- `candidate_total_score`
  - `[0,1]`

- `candidate_is_target_house_entry`
  - `0/1`

- `candidate_target_side_match`
  - `0.0 / 0.5 / 1.0`

- `candidate_center_in_target_bbox`
  - `0/1`

- `candidate_near_target_bbox`
  - `0/1`

### 5.3 为什么这些字段值得保留

因为它们能让 student 直接学到：

- 哪个候选属于目标房屋
- 为什么属于
- 属于目标房屋之后，语义和几何是否还支持进入

这比只给一个总分更容易学，也更便于后续消融。

---

## 6. `teacher_targets` 新字段

旧版 teacher target 主要是：

- `entry_state`
- `subgoal`
- `action_hint`
- `target_candidate_id`

现在建议拆成两层：

### 6.1 普通 teacher target

保留：

- `entry_state`
- `subgoal`
- `action_hint`
- `target_candidate_id`
- `risk_level`
- `teacher_reason_text`
- `confidence`

### 6.2 目标条件 teacher target

新增：

- `target_conditioned_state`
- `target_conditioned_subgoal`
- `target_conditioned_action_hint`
- `target_conditioned_target_candidate_id`
- `target_conditioned_reason_text`
- `target_conditioned_confidence`

### 6.3 mask 字段

建议增加：

- `teacher_available`
- `target_conditioned_teacher_available`

这样后续训练时可以选择：

- 只训旧头
- 只训目标条件头
- 两者联合训

---

## 7. 推荐输出示例

### 7.1 `global_state`

```json
{
  "pose_x": 2000.0,
  "pose_y": 85.3,
  "pose_z": 225.0,
  "yaw_deg": 12.0,
  "yaw_sin": 0.2079,
  "yaw_cos": 0.9781,
  "front_obstacle_present": 0,
  "front_min_depth_cm": 230.0,
  "target_house_id": 1,
  "current_house_id": -1,
  "target_house_in_fov": 1,
  "target_house_expected_side": 1,
  "target_house_expected_image_x": 0.53,
  "target_house_distance_cm": 860.0,
  "target_house_bearing_deg": 6.0
}
```

### 7.2 单个 candidate

```json
{
  "candidate_id": 0,
  "valid_mask": 1,
  "class_name": "open door",
  "confidence": 0.91,
  "bbox_cx": 0.54,
  "bbox_cy": 0.61,
  "bbox_w": 0.13,
  "bbox_h": 0.32,
  "entry_distance_cm": 420.0,
  "opening_width_cm": 138.0,
  "traversable": 1,
  "crossing_ready": 0,
  "candidate_target_match_score": 0.82,
  "candidate_semantic_score": 0.95,
  "candidate_geometry_score": 0.76,
  "candidate_total_score": 0.84,
  "candidate_is_target_house_entry": 1,
  "candidate_target_side_match": 1.0,
  "candidate_center_in_target_bbox": 1,
  "candidate_near_target_bbox": 1
}
```

### 7.3 `teacher_targets`

```json
{
  "teacher_available": 1,
  "entry_state": "enterable_open_door",
  "subgoal": "approach_entry",
  "action_hint": "forward",
  "target_candidate_id": 0,

  "target_conditioned_teacher_available": 1,
  "target_conditioned_state": "target_house_entry_approachable",
  "target_conditioned_subgoal": "approach_target_entry",
  "target_conditioned_action_hint": "forward",
  "target_conditioned_target_candidate_id": 0,
  "target_conditioned_confidence": 0.91
}
```

---

## 8. 固定长度设计建议

后续训练为了方便，推荐：

- 固定 `top-K = 3`

即：

- 只保留前 3 个候选
- 不足则 zero-pad

每个候选结构固定，便于：

- BC
- distillation
- PPO

---

## 9. 候选排序建议

旧版排序可能更偏：

- class priority
- confidence

新版本建议保留两种排序方式中的一种：

### 方案 A：仍按语义排序

优点：

- 稳定
- 与旧版兼容

缺点：

- 目标候选可能不在第一个

### 方案 B：按 `candidate_total_score` 排序

优点：

- 更接近最终目标房屋入口任务

缺点：

- 会让分布随 fusion 改动而飘

### 推荐

第一版建议：

- `candidates` 仍按原始 `YOLO semantic priority + confidence` 排序
- 目标相关分数作为候选字段保留

这样更稳，后面也容易做对比实验。

---

## 10. 与旧数据的兼容

为了兼容旧数据，建议 builder 支持：

### 10.1 若没有 target-conditioned fusion 字段

则：

- 新增字段全部回退默认值
- 不报错

例如：

- `target_house_in_fov = 0`
- `candidate_target_match_score = 0`
- `target_conditioned_teacher_available = 0`

### 10.2 若没有 target-conditioned teacher 字段

则：

- 仅保留普通 teacher targets

这样我们可以渐进式升级数据管线。

---

## 11. 与蒸馏训练的关系

后续 student encoder 建议输入：

- `global_state`
- `top-K candidates`

并输出两个层次的 head：

### 11.1 普通 head

- `entry_state`
- `subgoal`
- `action_hint`

### 11.2 目标条件 head

- `target_conditioned_state`
- `target_conditioned_subgoal`
- `target_conditioned_action_hint`
- `target_candidate_id`

这样可以直接比较：

- 只学普通入口
- 学目标条件入口

哪个更有效。

---

## 12. 第一版实现建议

第一版 builder 升级时，建议只做：

1. 扩 `global_state`
2. 扩 `candidates`
3. 扩 `teacher_targets`
4. 保持原有文件名仍叫 `entry_state.json`

先不要：

- 重命名整个数据结构
- 改太多现有导出脚本接口

这样过渡最稳。

---

## 13. 一句话总结

`target-conditioned entry_state_builder` 的核心不是让状态变更复杂，而是：

- **让状态显式包含“这个候选是否属于目标房屋”的信息**

只有这样，后续 student / policy 才能学会：

- 忽略非目标房屋门
- 优先寻找目标房屋入口
- 再决定靠近、绕行还是进入

