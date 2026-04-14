# Target-Conditioned Fusion Schema

## 1. 目的

本文件定义：

- `target-conditioned multimodal entry fusion`

在代码层面应输出的字段结构。

目标是统一后续这些模块之间的接口：

1. `fusion_entry_analysis`
2. `teacher_validator`
3. `entry_state_builder`
4. `distillation_dataset_export`
5. `representation_distillation_trainer`

也就是说，后续所有实现都应尽量以本文件为准，而不是各模块自行猜字段。

---

## 2. 设计原则

新 schema 应满足 4 个要求：

1. **兼容旧字段**
   - 旧版 `final_entry_state / recommended_subgoal / recommended_action_hint` 先保留

2. **显式表达目标约束**
   - 明确区分：
     - 普通入口判断
     - 目标房屋条件入口判断

3. **便于 teacher 蒸馏**
   - 字段应直接支持：
     - `entry_state`
     - `subgoal`
     - `action_hint`
     - `target_candidate_id`

4. **便于调试**
   - 需要保留：
     - 中间候选分数
     - 目标匹配分数
     - 目标房屋是否在视野中

---

## 3. 顶层结构

推荐 `fusion_result.json` 顶层继续保留：

```json
{
  "fusion_version": "phase2_target_conditioned_v1",
  "sample_id": "...",
  "inputs": {},
  "yolo": {},
  "depth": {},
  "fusion": {}
}
```

其中新设计主要落在：

- `fusion`

---

## 4. `fusion` 顶层字段

推荐结构：

```json
{
  "final_entry_state": "",
  "recommended_subgoal": "",
  "recommended_action_hint": "",
  "decision_reason": "",

  "target_conditioned_state": "",
  "target_conditioned_subgoal": "",
  "target_conditioned_action_hint": "",
  "target_conditioned_reason": "",

  "crossing_ready": false,
  "risk_level": "",

  "target_context": {},
  "semantic_detection": {},
  "semantic_depth_assessment": {},
  "best_target_candidate": {},
  "candidate_target_scores": [],
  "debug": {}
}
```

说明：

- 旧字段：
  - `final_entry_state`
  - `recommended_subgoal`
  - `recommended_action_hint`
  - `decision_reason`
- 新字段：
  - `target_conditioned_state`
  - `target_conditioned_subgoal`
  - `target_conditioned_action_hint`
  - `target_conditioned_reason`

---

## 5. 旧字段的定位

旧字段继续表示：

- **不带目标房屋约束的普通入口判断**

也就是：

- 如果当前画面里最显著候选是一个 `open_door`
- 旧字段可以仍然输出：
  - `enterable_open_door`

但它不保证这个门属于目标房屋。

所以从现在开始，旧字段应该主要作为：

- debug/reference

而不是最终任务执行依据。

---

## 6. 新字段的定位

新字段表示：

- **带目标房屋约束的最终任务决策**

这组字段才是后面策略训练和 agent 交互应优先使用的输出：

- `target_conditioned_state`
- `target_conditioned_subgoal`
- `target_conditioned_action_hint`
- `target_conditioned_reason`

---

## 7. `target_context` 结构

推荐：

```json
{
  "target_house_id": "",
  "current_house_id": "",
  "target_house_distance_cm": 0.0,
  "target_house_bearing_deg": 0.0,
  "target_house_in_fov": false,
  "target_house_expected_side": "out_of_view",
  "target_house_expected_image_x": null
}
```

字段解释：

- `target_house_id`
  - 当前任务目标房屋

- `current_house_id`
  - UAV 当前所在房屋，若未知可为空

- `target_house_distance_cm`
  - UAV 到目标房屋中心或入口参考点的距离

- `target_house_bearing_deg`
  - 目标房屋相对 UAV 机头方向的 bearing

- `target_house_in_fov`
  - 目标房屋是否应在当前视野中

- `target_house_expected_side`
  - `{left, center, right, out_of_view}`

- `target_house_expected_image_x`
  - 若能估算，给出目标房屋在图像中的预期横向位置

---

## 8. `semantic_detection` 结构

这个字段保留当前最终被选中的语义候选：

```json
{
  "candidate_id": 0,
  "class_name": "open door",
  "confidence": 0.91,
  "xyxy": [x1, y1, x2, y2]
}
```

它表示：

- 普通语义层最强候选

不一定等于目标候选。

---

## 9. `semantic_depth_assessment` 结构

这个字段表示：

- 对 `semantic_detection` 对应 ROI 做 depth 验证后的结果

推荐：

```json
{
  "source": "semantic_region",
  "entry_distance_cm": 0.0,
  "surrounding_depth_cm": 0.0,
  "clearance_depth_cm": 0.0,
  "depth_gain_cm": 0.0,
  "opening_width_cm": 0.0,
  "traversable": false,
  "crossing_ready": false,
  "rgb_bbox_xyxy": [x1, y1, x2, y2]
}
```

---

## 10. `candidate_target_scores` 结构

这是目标条件融合最核心的新字段。

推荐每个候选存成：

```json
{
  "candidate_id": 0,
  "class_name": "open door",
  "confidence": 0.91,
  "xyxy": [x1, y1, x2, y2],

  "candidate_target_match_score": 0.0,
  "candidate_semantic_score": 0.0,
  "candidate_geometry_score": 0.0,
  "candidate_total_score": 0.0,

  "candidate_house_id": "",
  "candidate_is_target_house_entry": false,

  "entry_distance_cm": 0.0,
  "opening_width_cm": 0.0,
  "traversable": false,
  "crossing_ready": false
}
```

### 字段含义

- `candidate_target_match_score`
  - 候选与目标房屋匹配程度

- `candidate_semantic_score`
  - 候选在语义上像入口的程度

- `candidate_geometry_score`
  - 候选在深度几何上可通行的程度

- `candidate_total_score`
  - 最终排序分数

- `candidate_house_id`
  - 候选若能归属到某栋房屋，则填房屋 id

- `candidate_is_target_house_entry`
  - 当前候选是否被判定为目标房屋入口候选

---

## 11. `best_target_candidate` 结构

推荐：

```json
{
  "candidate_id": 0,
  "class_name": "open door",
  "confidence": 0.91,
  "xyxy": [x1, y1, x2, y2],
  "candidate_target_match_score": 0.82,
  "candidate_total_score": 0.88,
  "candidate_is_target_house_entry": true,
  "entry_distance_cm": 420.0,
  "opening_width_cm": 138.0,
  "traversable": true,
  "crossing_ready": false
}
```

如果当前没有可靠目标候选，则：

```json
{
  "candidate_id": -1,
  "candidate_is_target_house_entry": false
}
```

---

## 12. `risk_level` 枚举

推荐：

- `low`
- `medium`
- `high`

含义：

- `low`
  - 当前目标候选清晰、无遮挡、可接近

- `medium`
  - 目标候选存在但还远、还需确认、或几何证据一般

- `high`
  - 前障碍近、目标不明确、或存在明显误入非目标房屋风险

---

## 13. `target_conditioned_state` 枚举

推荐第一版先固定为：

- `target_house_not_in_view`
- `target_house_visible_keep_search`
- `target_house_entry_visible`
- `target_house_entry_approachable`
- `target_house_entry_blocked`
- `non_target_house_entry_visible`
- `target_house_geometric_opening_needs_confirmation`

### 解释

- `target_house_not_in_view`
  - 当前视野内没有足够证据表明目标房屋在前方

- `target_house_visible_keep_search`
  - 目标房屋在视野中，但入口还未确认

- `target_house_entry_visible`
  - 看到了目标房屋入口候选，但还不适合直接接近

- `target_house_entry_approachable`
  - 目标房屋入口已被确认，应该靠近

- `target_house_entry_blocked`
  - 目标房屋入口存在，但当前被障碍挡住

- `non_target_house_entry_visible`
  - 当前看到的是其他房屋的门，不应该进

- `target_house_geometric_opening_needs_confirmation`
  - 目标房屋区域内存在几何开口，但语义还不够强

---

## 14. `target_conditioned_subgoal` 枚举

推荐：

- `reorient_to_target_house`
- `keep_search_target_house`
- `approach_target_entry`
- `align_target_entry`
- `detour_left_to_target_entry`
- `detour_right_to_target_entry`
- `cross_target_entry`
- `ignore_non_target_entry`
- `backoff_and_reobserve`

---

## 15. `target_conditioned_action_hint` 枚举

保持和现有动作集合兼容：

- `forward`
- `yaw_left`
- `yaw_right`
- `left`
- `right`
- `backward`
- `hold`

说明：

- `subgoal` 表达高层意图
- `action_hint` 表达下一步粗粒度动作方向

---

## 16. 推荐决策映射

### 情况 A：目标房屋不在视野内

```json
{
  "target_conditioned_state": "target_house_not_in_view",
  "target_conditioned_subgoal": "reorient_to_target_house",
  "target_conditioned_action_hint": "yaw_left"
}
```

或 `yaw_right`，取决于 bearing。

### 情况 B：看见了别的房屋的门

```json
{
  "target_conditioned_state": "non_target_house_entry_visible",
  "target_conditioned_subgoal": "ignore_non_target_entry",
  "target_conditioned_action_hint": "yaw_left"
}
```

### 情况 C：目标房屋门存在但很远

```json
{
  "target_conditioned_state": "target_house_entry_approachable",
  "target_conditioned_subgoal": "approach_target_entry",
  "target_conditioned_action_hint": "forward"
}
```

### 情况 D：目标房屋门被挡住

```json
{
  "target_conditioned_state": "target_house_entry_blocked",
  "target_conditioned_subgoal": "detour_left_to_target_entry",
  "target_conditioned_action_hint": "left"
}
```

---

## 17. 与 `teacher_output.json` 的对应关系

teacher 层建议后续也新增目标条件字段：

```json
{
  "entry_state": "",
  "subgoal": "",
  "action_hint": "",

  "target_conditioned_state": "",
  "target_conditioned_subgoal": "",
  "target_conditioned_action_hint": "",
  "target_candidate_id": -1
}
```

其中：

- 旧字段作为通用 teacher 输出
- 新字段作为最终任务 teacher 输出

后续蒸馏时，建议以新字段优先。

---

## 18. 与 `entry_state.json` 的对应关系

`entry_state_builder` 后续应至少补充：

### `global_state`

- `target_house_in_fov`
- `target_house_expected_side`
- `target_house_bearing_deg`

### `candidates`

- `candidate_target_match_score`
- `candidate_is_target_house_entry`
- `candidate_house_id`

### `teacher_targets`

- `target_conditioned_state`
- `target_conditioned_subgoal`
- `target_conditioned_action_hint`
- `target_candidate_id`

---

## 19. 最小实现建议

第一版实现时，建议只做这几件事：

1. 新增 `target_context`
2. 新增 `candidate_target_scores`
3. 新增 `best_target_candidate`
4. 新增 `target_conditioned_state / subgoal / action_hint / reason`
5. 保留旧字段不动

这样改动范围可控，也便于回归比较。

---

## 20. 一句话总结

这份 schema 的核心思想是：

- 旧字段继续描述“普通入口判断”
- 新字段专门描述“目标房屋条件入口判断”

从现在开始，真正面向任务执行、teacher 蒸馏和策略训练的主输出，应逐步转向：

- `target_conditioned_state`
- `target_conditioned_subgoal`
- `target_conditioned_action_hint`

