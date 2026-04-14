# Target-Conditioned Entry Fusion

## 1. 为什么需要这一层

当前 `Phase 2` 的融合已经能比较稳定地回答：

- 图里有没有 `open door / door / close door / window`
- 这个候选在几何上是否可穿越
- 当前前方是否有障碍物

但在真实任务里，这还不够。

当前系统的真实目标不是：

- 找到任意一个可进入门

而是：

- **找到目标房屋的可进入门**

在一条空旷街道、且同时存在多栋房屋的场景中，以下情况非常常见：

1. 画面里同时出现多栋房屋
2. 多栋房屋都存在 door-like opening
3. 某个非目标房屋的门比目标房屋的门更显眼
4. 仅靠 `YOLO + depth` 会优先选择“最像门”或“最近门”

这就会导致错误行为：

- 无人机去靠近别的房屋的门
- 在非目标房屋前反复判断“可进入”
- 目标房屋明明在附近，但策略没有被显式约束去找它

所以需要把原来的普通融合升级成：

- **target-conditioned multimodal entry fusion**

也就是：

- 先问“这是不是目标房屋的入口”
- 再问“它能不能进入”

而不是反过来。

---

## 2. 新问题定义

给定：

- 当前 UAV 位姿
- 地图中已标注的房屋集合
- 当前目标房屋 `target_house_id`
- 当前 RGB + depth 观测

系统需要输出：

1. 当前目标房屋是否在视野中
2. 哪个候选最可能属于目标房屋
3. 这个目标候选是否是有效入口
4. 当前应该：
   - 靠近目标入口
   - 绕行到目标入口
   - 转向重新寻找目标房屋
   - 继续搜索

也就是说，融合层要从：

- `entry-centric`

升级成：

- `target-house-conditioned entry-centric`

---

## 3. 普通 fusion 不够的原因

现有融合大致是：

1. `YOLO` 给候选
2. `depth` 判断候选几何可通行性
3. `fusion` 输出最终入口状态

这个逻辑有一个默认假设：

- 画面里的最佳门候选，就是当前该关注的门候选

但在多房屋场景中，这个假设不成立。

普通 fusion 容易犯的错误包括：

### 3.1 选错房屋

画面里：

- `house_2` 的 open door 更大、更近、更清晰
- `house_1` 才是目标房屋

普通 fusion 往往会输出：

- `enterable_open_door`
- `approach_entry`

但这是错误目标。

### 3.2 只看局部最优，不看任务约束

从局部几何看：

- 某个门可通过

但从全局任务看：

- 那扇门不属于目标房屋

所以这类候选应该被降权，甚至直接过滤。

### 3.3 无法把“找目标房屋”和“找入口”统一起来

实际任务顺序应该是：

1. 先对准目标房屋
2. 再在目标房屋上找入口
3. 再判断入口是否可进入

而不是：

1. 先找到一个入口
2. 再事后判断是不是目标房屋

---

## 4. 核心思想

目标条件融合的核心是：

- **地图先验决定“该看哪栋房子”**
- **YOLO 决定“图里有哪些 door/window 候选”**
- **depth 决定“这些候选几何上能不能通过”**
- **融合模块决定“目标房屋的最佳候选是什么”**

一句话概括：

- **先做目标约束**
- **再做入口判断**

---

## 5. 输入信息设计

新的融合层建议显式接入 4 类输入。

### 5.1 地图 / 房屋先验

来自：

- `houses_config.json`
- 地图配准结果
- 当前任务的 `target_house_id`

至少需要：

- `target_house_id`
- `current_house_id`
- `target_house_center_x`
- `target_house_center_y`
- `target_house_bbox_image`
- `target_house_polygon`（后续可选）
- `target_house_front_door_point`（后续可选）
- `target_house_front_facing_yaw`（后续可选）

### 5.2 UAV 全局状态

至少需要：

- `uav_x`
- `uav_y`
- `uav_z`
- `uav_yaw`
- `movement_enabled`

以及由地图关系推导出的：

- `target_house_distance_cm`
- `target_house_bearing_deg`
- `target_house_in_fov`
- `target_house_expected_side`

其中：

- `target_house_expected_side ∈ {left, center, right, out_of_view}`

### 5.3 YOLO 候选

对每个候选保留：

- `class_name`
- `confidence`
- `bbox`
- `candidate_rank`

并补充与目标房屋关系：

- `candidate_target_match_score`
- `candidate_house_id`
- `candidate_is_target_house_entry`

### 5.4 Depth ROI 几何特征

依然按候选 ROI 提取：

- `entry_distance_cm`
- `opening_width_cm`
- `depth_gain_cm`
- `clearance_depth_cm`
- `traversable`
- `crossing_ready`
- `front_obstacle_present`
- `front_min_depth_cm`

---

## 6. 新的中间变量

建议在 fusion 内部增加以下中间变量。

### 6.1 目标房屋视野先验

- `target_house_in_view`
- `target_house_expected_bearing_deg`
- `target_house_expected_image_x`
- `target_house_expected_side`

这部分回答：

- 从地图和 UAV 姿态推算，目标房屋此刻大概应该出现在画面哪里

### 6.2 候选-目标匹配分数

对每个候选计算：

- `candidate_target_match_score`

这个分数不是几何可通过分数，而是：

- **这个候选属于目标房屋的概率/匹配程度**

它可以由这些量组合得到：

- 候选 bbox 中心方向，与 `target_house_bearing_deg` 的一致性
- 候选 bbox 是否落在目标房屋的预期图像区域附近
- 候选对应的地图投影是否更接近目标房屋而不是其他房屋
- 候选的语义类别是否 door-like

### 6.3 目标候选选择

在所有候选中，不直接选最大或最高置信度，而是选：

- `best_target_candidate`

原则是：

```text
best_target_candidate
= argmax(candidate_target_match_score + semantic_score + geometric_score)
```

其中：

- `candidate_target_match_score` 优先级最高
- `semantic_score` 次之
- `geometric_score` 最后验证

---

## 7. 推荐决策顺序

建议把 fusion 顺序改成下面 5 步。

### Step 1. 先判断目标房屋是否在视野中

如果：

- `target_house_in_fov = false`

则不应该继续做“穿门判断”，而应该优先输出：

- `target_house_not_in_view`
- `reorient_to_target_house`

### Step 2. 找到最可能属于目标房屋的候选

对所有 `YOLO` 候选计算：

- `candidate_target_match_score`

如果最高分仍然很低，则输出：

- `keep_search_target_house`

### Step 3. 对目标候选做语义判断

如果目标候选是：

- `window`

则输出：

- `target_house_window_visible_keep_search`

而不是可进入入口。

### Step 4. 再对目标候选做 depth 几何验证

这里只验证：

- 最像目标房屋入口的候选 ROI

而不是全图最深开口。

### Step 5. 最终输出目标条件状态

最终输出应该不再是普通：

- `enterable_open_door`
- `window_visible_keep_search`

而是目标条件版本：

- `target_house_entry_visible`
- `target_house_entry_approachable`
- `target_house_entry_blocked`
- `non_target_house_entry_visible`
- `target_house_not_in_view`
- `keep_search_target_house`

---

## 8. 新的 fusion 输出字段建议

建议在 `fusion_result.json` 中新增：

### 8.1 目标房屋相关字段

- `target_house_id`
- `current_house_id`
- `target_house_distance_cm`
- `target_house_bearing_deg`
- `target_house_in_fov`
- `target_house_expected_side`

### 8.2 候选-目标匹配字段

- `best_target_candidate_id`
- `best_target_candidate_class`
- `best_target_candidate_confidence`
- `best_target_candidate_match_score`
- `best_target_candidate_is_target_house_entry`

### 8.3 目标条件决策字段

- `target_conditioned_state`
- `target_conditioned_subgoal`
- `target_conditioned_action_hint`
- `target_conditioned_reason`

### 8.4 与旧状态并存字段

为了兼容旧逻辑，建议先保留：

- `final_entry_state`
- `recommended_subgoal`
- `recommended_action_hint`

但新训练和新决策应逐步转向使用：

- `target_conditioned_state`

---

## 9. 推荐状态枚举

建议新增以下目标条件状态：

### 9.1 目标房屋可见但尚未确认入口

- `target_house_visible_keep_search`

### 9.2 目标房屋入口可见

- `target_house_entry_visible`

### 9.3 目标房屋入口可接近

- `target_house_entry_approachable`

### 9.4 目标房屋入口被挡

- `target_house_entry_blocked`

### 9.5 视野中只有非目标房屋入口

- `non_target_house_entry_visible`

### 9.6 目标房屋暂时不在视野内

- `target_house_not_in_view`

---

## 10. 和已有 teacher / distillation 的关系

这一层加入后，teacher 也应升级成目标条件版本。

也就是说，teacher 输出不再只说：

- `approach_entry`

而应说：

- `approach_target_entry`
- `reorient_to_target_house`
- `ignore_non_target_entry`
- `detour_to_target_entry`

对蒸馏来说，这非常重要，因为 student 最终学到的不只是：

- 这是不是一个门

而是：

- 这是不是**目标房屋**的门

所以 `entry_state_builder` 后面也应增加：

- `target_house_in_fov`
- `target_house_expected_side`
- `candidate_target_match_score`
- `candidate_is_target_house_entry`

---

## 11. 和表示蒸馏的关系

这一步会直接影响：

- [`13_representation_distillation_plan.md`](13_representation_distillation_plan.md)

因为蒸馏对象将从普通 `entry-centric representation` 升级成：

- **target-conditioned entry-centric representation**

也就是说，student 表示不只编码：

- door / window / traversable

还要编码：

- 是否属于目标房屋
- 当前应不应该忽略该候选

这会让后面的局部策略明显更稳。

---

## 12. 为什么这一步合理

你现在的任务环境确实具有这些特点：

- 街道空旷
- 房屋很多
- 多个 door-like opening 会同时出现

因此类别不平衡并不完全是数据问题，而是：

- **任务本身就更容易先看到非目标房屋或被挡入口**

所以把地图房屋先验显式并入融合，不是“额外 trick”，而是：

- 任务定义本身所要求的必要条件

换句话说：

- 如果任务是“进入目标房屋”
- 那么 fusion 里就必须有 “target house conditioning”

---

## 13. 第一版实现建议

第一版不要过度复杂，建议这样做：

1. 先不做复杂投影几何
2. 先用：
   - `target_house_center`
   - `uav_pose`
   - `uav_yaw`
   - `candidate_bbox_center`
   做一个简化版 `candidate_target_match_score`
3. 先把目标条件状态加进 `fusion_result.json`
4. 先观察这一步能否明显减少“去错房屋门口”的情况

如果第一版有效，再加：

- house polygon
- front door prior
- 更精细的 image-plane projection

---

## 14. 实验建议

这一层加入后，建议在实验里单独比较：

1. `Plain fusion`
   - 只用 YOLO + depth

2. `Target-conditioned fusion`
   - YOLO + depth + target house prior

比较指标：

- target-house entry selection accuracy
- wrong-house approach rate
- target-house search efficiency
- target entry approach success rate

这会很直观地证明：

- 加入地图目标先验后，系统不再只会找“最近门”，而是会找“目标房屋门”

---

## 15. 一句话总结

这一步的核心不是再提升普通门检测精度，而是：

- **把“目标房屋约束”显式纳入入口融合**

也就是：

- 先判断“是不是目标房屋的候选入口”
- 再判断“这个候选入口能不能进入”

这会让整个系统从：

- 通用入口发现

升级成：

- **目标房屋定向入口发现**

这一步是后续局部技能学习和表示蒸馏真正对准任务目标的关键。
