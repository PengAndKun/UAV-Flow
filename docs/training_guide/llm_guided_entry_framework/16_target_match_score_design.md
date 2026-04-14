# Target Match Score Design

## 1. 目的

本文件专门定义：

- `candidate_target_match_score`

的设计目标、输入组成、推荐计算方式和第一版实现范围。

它回答的核心问题是：

- **当前画面中的某个 door/window 候选，究竟有多大概率属于目标房屋？**

这是 `target-conditioned fusion` 的核心中间量。

---

## 2. 为什么这个分数重要

在当前任务里，系统不是要回答：

- 画面中有没有门

而是要回答：

- 画面中有没有**目标房屋的门**

所以必须有一个单独分数，把“候选像门”与“候选属于目标房屋”区分开。

如果没有这个分数，融合模块会天然偏向：

- 最大的候选
- 最近的候选
- 置信度最高的候选

这在多房屋街景中会非常容易把 UAV 引到错误房屋门口。

---

## 3. 定义

对每个候选 `c_i`，定义：

```text
candidate_target_match_score(i) ∈ [0, 1]
```

它表示：

- 候选 `c_i` 属于 `target_house_id` 的匹配程度

这个分数不表示：

- 几何可穿越性
- 入口语义可信度

它只回答：

- **这个候选是不是目标房屋上的候选**

---

## 4. 与其他分数的分工

建议把总决策拆成 3 类分数：

### 4.1 目标匹配分数

- `candidate_target_match_score`

回答：

- 候选是否来自目标房屋

### 4.2 语义分数

- `candidate_semantic_score`

回答：

- 候选在语义上是否像有效入口

例如：

- `open door > door > close door > window`

### 4.3 几何分数

- `candidate_geometry_score`

回答：

- 候选在深度上是否可接近/可穿越

例如：

- 是否前方被挡
- 宽度是否足够
- 深度增益是否明显

### 4.4 总分

最终排序可以写成：

```text
candidate_total_score
= α * candidate_target_match_score
+ β * candidate_semantic_score
+ γ * candidate_geometry_score
```

其中：

- `α > β >= γ`

也就是：

- **目标归属优先于语义**
- **语义优先于几何**

---

## 5. 输入组成

第一版 `candidate_target_match_score` 建议由这 4 类量组成。

### 5.1 UAV -> 目标房屋相对方位

已知：

- `uav_x, uav_y, uav_yaw`
- `target_house_x, target_house_y`

计算：

- `target_house_bearing_deg`

它表示目标房屋相对机头的方向。

这一步回答：

- 目标房屋应该在左、中、右的哪一侧

### 5.2 候选 bbox 的图像位置

已知：

- 候选 bbox 中心点 `bbox_cx`

可估计：

- 候选在图像中的水平位置

这一步回答：

- 候选看起来在左、中、右哪一侧

### 5.3 房屋地图先验

如果地图里已有：

- `target_house_bbox_image`

则可以直接比较：

- 候选 bbox 是否落在目标房屋地图框附近

这一步在第一版里非常有价值，因为实现简单、解释清楚。

### 5.4 候选类别先验

虽然 `candidate_target_match_score` 主要不负责语义，但第一版允许一个弱语义先验：

- `window` 往往不应成为目标入口候选
- `open door / door` 可以轻微加成

注意：

- 这只是弱约束，不能替代 `candidate_semantic_score`

---

## 6. 第一版推荐的 4 个子分数

第一版建议把 `candidate_target_match_score` 分成下面 4 项：

```text
candidate_target_match_score
= w1 * S_bearing
+ w2 * S_image_side
+ w3 * S_house_bbox
+ w4 * S_class_prior
```

其中推荐：

- `w1 = 0.35`
- `w2 = 0.25`
- `w3 = 0.30`
- `w4 = 0.10`

也就是：

- 方位和地图框最重要
- 类别先验最弱

---

## 7. `S_bearing` 设计

### 7.1 定义

比较：

- 目标房屋相对 bearing
- 候选在图像中的相对方向

先把候选 bbox 中心转成一个简化图像方向：

- `left`
- `center`
- `right`

同时把目标房屋 bearing 也粗分成：

- `left`
- `center`
- `right`
- `out_of_view`

### 7.2 第一版评分

推荐简单打分：

- 完全一致：`1.0`
- 相邻区域：`0.5`
- 明显相反：`0.0`
- `out_of_view`：`0.0`

### 7.3 为什么第一版先离散化

因为：

- 简单
- 稳定
- 容易调试

后续再升级成连续角差评分。

---

## 8. `S_image_side` 设计

如果已有：

- `target_house_expected_side`

则可以直接比较：

- 候选 bbox 所在侧
- 目标房屋预期所在侧

评分可与 `S_bearing` 一样：

- 一致：`1.0`
- 相邻：`0.5`
- 相反：`0.0`

第一版里：

- `S_bearing` 和 `S_image_side` 很接近

但仍建议保留两个字段，因为：

- 一个来自连续几何 bearing
- 一个来自离散视觉预期

后面好分析贡献。

---

## 9. `S_house_bbox` 设计

如果：

- `houses_config.json` 里有 `target_house.map_bbox_image`

那么这个分数最直接。

### 9.1 第一版方法

比较：

- 候选 bbox 与目标房屋 bbox 的 IoU
- 或候选中心是否落在目标房屋 bbox 内

推荐第一版：

```text
if center_in_target_bbox:
    S_house_bbox = 1.0
elif center_near_target_bbox:
    S_house_bbox = 0.5
else:
    S_house_bbox = 0.0
```

其中：

- `center_near_target_bbox`
  表示候选中心距离目标房屋框边界在一个可调 margin 内

### 9.2 为什么第一版不用复杂投影

因为你已经有：

- 校准后的地图
- 房屋框

这已经足够支撑第一版目标匹配。

---

## 10. `S_class_prior` 设计

第一版建议给弱先验：

- `open door`: `1.0`
- `door`: `0.8`
- `close door`: `0.4`
- `window`: `0.1`

注意：

- 这部分权重必须低
- 否则它会偷走 `semantic_score` 的工作

它只应该用于：

- 在目标归属近似相同时打破平局

---

## 11. 第一版完整推荐公式

推荐：

```text
candidate_target_match_score
= 0.35 * S_bearing
+ 0.25 * S_image_side
+ 0.30 * S_house_bbox
+ 0.10 * S_class_prior
```

最后 clamp 到：

- `[0, 1]`

---

## 12. 与 `candidate_semantic_score` 的边界

为了避免重复计算，建议：

### `candidate_target_match_score`

只负责：

- 候选属于目标房屋的可能性

### `candidate_semantic_score`

只负责：

- 候选本身像不像可进入入口

例如：

- `open door`
- `door`
- `window`
- `close door`

### `candidate_geometry_score`

只负责：

- 候选几何上是否可接近、可通过

---

## 13. 最终候选选择建议

最终不要直接按 `candidate_target_match_score` 选，而是：

### Step 1

先过滤：

- `candidate_target_match_score < τ_target`

例如：

- `τ_target = 0.45`

### Step 2

在剩余候选中，再按：

```text
candidate_total_score
= α * candidate_target_match_score
+ β * candidate_semantic_score
+ γ * candidate_geometry_score
```

推荐第一版：

- `α = 0.5`
- `β = 0.3`
- `γ = 0.2`

---

## 14. 特殊情形处理

### 14.1 目标房屋不在视野内

若：

- `target_house_in_fov = false`

则所有候选：

- `candidate_target_match_score = 0`

最终直接走：

- `target_house_not_in_view`

### 14.2 画面里只有非目标房屋门

若：

- 某些候选语义很强
- 但 `candidate_target_match_score` 很低

则输出：

- `non_target_house_entry_visible`

而不是普通 `approach_entry`

### 14.3 地图框缺失

若某栋房屋没有 `map_bbox_image`：

则第一版只用：

- `S_bearing`
- `S_image_side`
- `S_class_prior`

此时：

- `S_house_bbox = 0`

---

## 15. 推荐调试输出

为了便于调试，每个候选建议输出：

- `S_bearing`
- `S_image_side`
- `S_house_bbox`
- `S_class_prior`
- `candidate_target_match_score`

这样后面出现“为什么系统选错门”时，可以直接看是哪一项失真。

---

## 16. 对 teacher 和蒸馏的价值

这一步一旦加入，teacher 的高层判断会明显更稳定，因为它不再只对着“画面里最像门的东西”，而是对着：

- **目标房屋的候选入口**

这会直接提升：

- `target_conditioned_state`
- `target_conditioned_subgoal`
- `target_candidate_id`

的蒸馏质量。

---

## 17. 第一版实现范围建议

第一版只做：

1. `S_bearing`
2. `S_image_side`
3. `S_house_bbox`
4. `S_class_prior`

先不做：

- 复杂 3D 投影
- house polygon 精细落点
- doorway-facing prior
- temporal smoothing

因为第一版的目标是：

- 先证明“目标房屋条件融合”有效

而不是一开始就把系统做得很重。

---

## 18. 后续可升级方向

第二版可以继续加：

1. `polygon overlap score`
2. `front_door_prior score`
3. `temporal consistency score`
4. `cross-frame candidate tracking`
5. `map-projected image x prediction`

但这些都应在第一版跑通后再加。

---

## 19. 一句话总结

`candidate_target_match_score` 的核心作用是：

- **把“候选像门”与“候选属于目标房屋”分开**

第一版最合理的做法是：

- 先用 `bearing + expected side + house bbox + weak class prior`

做一个简单、可解释、可调试的匹配分数。

