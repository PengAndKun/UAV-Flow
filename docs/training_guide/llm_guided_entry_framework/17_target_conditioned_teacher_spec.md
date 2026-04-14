# Target-Conditioned Teacher Specification

## 1. 目的

本文件定义：

- `target-conditioned teacher`

在 `Phase 2 / Phase 2.5` 路线中的职责、输入、输出和约束规则。

它要解决的问题是：

- teacher 不再只回答“图里有没有可进入门”
- 而是回答“图里有没有**目标房屋**的可进入门，以及下一步该怎么做”

这份规范是后续这些模块的接口依据：

1. `teacher_validator`
2. `entry_state_builder`
3. `distillation_dataset_export`
4. `representation_distillation_trainer`

---

## 2. 为什么 teacher 也必须升级

如果 fusion 已经升级成：

- `target-conditioned multimodal entry fusion`

但 teacher 仍然只输出普通：

- `approach_entry`
- `keep_search`
- `cross_entry`

就会出现一个问题：

- teacher 给出的动作语义缺少“目标房屋约束”

例如：

### 错误例子

当前目标是：

- `House_1`

画面里看到的是：

- `House_2` 的一个 `open_door`

如果 teacher 只看普通入口语义，很可能会输出：

- `approach_entry`

这对普通入口任务是合理的，
但对“进入目标房屋”任务是错误的。

所以 teacher 也必须升级为：

- **target-conditioned teacher**

---

## 3. teacher 的新角色

升级后的 teacher 应同时承担两层职责：

### 3.1 普通入口解释层

这一层仍然保留，用于：

- 兼容旧流程
- 帮助分析“画面里最显著的入口是什么”

### 3.2 目标房屋任务层

这一层是新的主任务层，用于：

- 判断当前入口是否属于目标房屋
- 决定该忽略非目标入口、重新转向、还是接近目标入口

后续真正用于蒸馏和策略训练时，应优先使用：

- 目标房屋任务层输出

---

## 4. 输入设计

target-conditioned teacher 的输入，建议包含 6 类信息。

### 4.1 图像输入

- `rgb.png`
- `depth_preview.png`

### 4.2 YOLO 摘要

至少应包含：

- top-K candidate
- class
- confidence
- bbox

### 4.3 depth 摘要

至少应包含：

- `front_obstacle`
- `front_min_depth_cm`
- `entry_distance_cm`
- `opening_width_cm`
- `traversable`
- `crossing_ready`

### 4.4 普通 fusion 摘要

至少应包含：

- `final_entry_state`
- `recommended_subgoal`
- `recommended_action_hint`

### 4.5 目标条件 fusion 摘要

这是关键新增部分，至少应包含：

- `target_house_id`
- `current_house_id`
- `target_house_in_fov`
- `target_house_expected_side`
- `best_target_candidate`
- `candidate_target_scores`
- `target_conditioned_state`

### 4.6 当前全局状态

至少应包含：

- `uav_pose`
- `uav_yaw`
- `task_label`
- `target_house_id`
- 最近一小段动作历史（可选）

---

## 5. teacher 输出总体结构

建议 teacher 输出同时保留两层：

```json
{
  "entry_state": "",
  "subgoal": "",
  "action_hint": "",
  "risk_level": "",
  "reason": "",
  "confidence": 0.0,

  "target_conditioned_state": "",
  "target_conditioned_subgoal": "",
  "target_conditioned_action_hint": "",
  "target_candidate_id": -1,
  "target_conditioned_reason": "",
  "target_confidence": 0.0
}
```

其中：

- 第一层是旧的普通输出
- 第二层是新的目标条件输出

后续蒸馏时，建议优先使用第二层。

---

## 6. 普通输出字段的定位

保留这些旧字段：

- `entry_state`
- `subgoal`
- `action_hint`
- `risk_level`
- `reason`
- `confidence`

它们的作用是：

- 描述画面中普通入口判断结果
- 与历史结果兼容
- 供 debug 使用

但它们不再是最终策略主监督。

---

## 7. 目标条件输出字段

### 7.1 `target_conditioned_state`

推荐枚举：

- `target_house_not_in_view`
- `target_house_visible_keep_search`
- `target_house_entry_visible`
- `target_house_entry_approachable`
- `target_house_entry_blocked`
- `non_target_house_entry_visible`
- `target_house_geometric_opening_needs_confirmation`

### 7.2 `target_conditioned_subgoal`

推荐枚举：

- `reorient_to_target_house`
- `keep_search_target_house`
- `approach_target_entry`
- `align_target_entry`
- `detour_left_to_target_entry`
- `detour_right_to_target_entry`
- `cross_target_entry`
- `ignore_non_target_entry`
- `backoff_and_reobserve`

### 7.3 `target_conditioned_action_hint`

推荐保持与现有动作集合兼容：

- `forward`
- `yaw_left`
- `yaw_right`
- `left`
- `right`
- `backward`
- `hold`

### 7.4 `target_candidate_id`

表示：

- 当前 top-K 中最应该关注的、属于目标房屋的候选 id

如果没有可靠目标候选：

- `-1`

### 7.5 `target_conditioned_reason`

要求：

- 简短
- 针对目标房屋任务
- 不解释无关细节

例如：

- `Target house is not yet centered; rotate left to reacquire it.`
- `Visible open door belongs to the target house but is still far away.`
- `The visible entry is blocked by a near obstacle; detour right.`
- `The current visible door likely belongs to a non-target house; ignore it.`

### 7.6 `target_confidence`

范围：

- `[0, 1]`

表示：

- teacher 对目标条件决策的整体信心

---

## 8. teacher 的核心判断顺序

target-conditioned teacher 不应直接从图像跳到动作，而应按下面顺序推理：

### Step 1. 目标房屋是否在视野中

如果：

- `target_house_in_fov = false`

则优先输出：

- `target_house_not_in_view`
- `reorient_to_target_house`

### Step 2. 当前最显著候选是否属于目标房屋

如果：

- 看到的是非目标房屋门

则应输出：

- `non_target_house_entry_visible`
- `ignore_non_target_entry`

### Step 3. 目标候选语义上是否像入口

如果：

- 目标候选是 `window`

则不应建议进入

### Step 4. 目标候选几何上是否可接近 / 可通过

只有此时才进入：

- `approach_target_entry`
- `detour_left/right_to_target_entry`
- `cross_target_entry`

---

## 9. teacher 输出的最小规则约束

以下规则建议做成 validator 的硬约束。

### 9.1 目标不在视野内时不能直接 approach/cross

如果：

- `target_conditioned_state = target_house_not_in_view`

则：

- `target_conditioned_subgoal` 不能是 `approach_target_entry`
- 不能是 `cross_target_entry`

### 9.2 非目标房屋入口不能进入

如果：

- `target_conditioned_state = non_target_house_entry_visible`

则：

- `target_conditioned_action_hint` 不能是 `forward`

### 9.3 高障碍时不能要求直接穿越

如果：

- 前障碍 `severity = high`

则：

- `target_conditioned_subgoal` 不能是 `cross_target_entry`

### 9.4 window 不能当目标入口

如果：

- 目标候选语义为 `window`

则：

- `target_conditioned_state` 不能是 `target_house_entry_approachable`

---

## 10. 普通 teacher 与目标 teacher 的对应关系

推荐把两层关系理解成：

### 普通 teacher

回答：

- 图里“像门的东西”是什么

### 目标 teacher

回答：

- 图里“属于目标房屋的门”是什么

因此允许出现：

### 示例 A

```json
{
  "entry_state": "enterable_open_door",
  "subgoal": "approach_entry",
  "target_conditioned_state": "non_target_house_entry_visible",
  "target_conditioned_subgoal": "ignore_non_target_entry"
}
```

这在目标房屋任务里是合理的，
因为普通入口存在，但不属于目标房屋。

---

## 11. 与蒸馏目标的关系

后续表示蒸馏时，建议把 teacher 目标拆成两组：

### 11.1 普通蒸馏目标

- `entry_state`
- `subgoal`
- `action_hint`

### 11.2 目标条件蒸馏目标

- `target_conditioned_state`
- `target_conditioned_subgoal`
- `target_conditioned_action_hint`
- `target_candidate_id`

其中第二组应作为主监督。

---

## 12. 第一版 teacher 输出建议

第一版实现时，不要让 teacher 太复杂，建议：

1. 保留旧输出不动
2. 在其基础上新增目标条件字段
3. target reason 尽量简短，长度控制在 `<= 160 chars`
4. `target_candidate_id` 不确定时直接返回 `-1`
5. `target_confidence < 普通 confidence` 也是允许的

也就是说：

- 第一版 teacher 可以“更保守”
- 只要目标条件逻辑明确就好

---

## 13. 推荐样例

### 样例 1：目标房屋不在视野中

```json
{
  "target_conditioned_state": "target_house_not_in_view",
  "target_conditioned_subgoal": "reorient_to_target_house",
  "target_conditioned_action_hint": "yaw_left",
  "target_candidate_id": -1
}
```

### 样例 2：看到非目标房屋的 open door

```json
{
  "target_conditioned_state": "non_target_house_entry_visible",
  "target_conditioned_subgoal": "ignore_non_target_entry",
  "target_conditioned_action_hint": "yaw_right",
  "target_candidate_id": -1
}
```

### 样例 3：目标房屋门已确认但还远

```json
{
  "target_conditioned_state": "target_house_entry_approachable",
  "target_conditioned_subgoal": "approach_target_entry",
  "target_conditioned_action_hint": "forward",
  "target_candidate_id": 0
}
```

### 样例 4：目标房屋门被挡住

```json
{
  "target_conditioned_state": "target_house_entry_blocked",
  "target_conditioned_subgoal": "detour_left_to_target_entry",
  "target_conditioned_action_hint": "left",
  "target_candidate_id": 1
}
```

---

## 14. 一句话总结

target-conditioned teacher 的核心不是再解释“图里有没有门”，而是：

- **图里这个门，是不是目标房屋的门，以及下一步该不该对它采取动作。**

因此它必须在旧 teacher 输出之外，新增一层：

- `target_conditioned_state / subgoal / action_hint / target_candidate_id`

后续真正用于策略蒸馏和训练时，应优先依赖这层输出。

