# Representation Distillation Plan

## 1. 目标定位

这一步的目标不是直接训练一个端到端飞控策略，而是先训练一个：

- 融合 `LLM teacher`
- 融合 `YOLO / RGB` 候选语义
- 融合 `Depth ROI` 几何可通行性

的 **入口相关表示模型**。

后续再把这个表示作为 `student policy / RL policy` 的状态输入。

一句话概括：

- **先蒸馏表示**
- **再训练策略**

而不是：

- 直接从原始多模态输入端到端学控制

---

## 2. 为什么要走这条路线

当前系统已经具备三类互补信息：

1. `YOLO26`
   - 能稳定检测：
     - `open door`
     - `door`
     - `close door`
     - `window`

2. `Depth`
   - 能提供：
     - `entry_distance_cm`
     - `opening_width_cm`
     - `depth_gain_cm`
     - `traversable`
     - `crossing_ready`
     - `front_obstacle`

3. `LLM / VLM teacher`
   - 能提供：
     - `entry_state`
     - `subgoal`
     - `action_hint`
     - `risk_level`
     - `reason`

如果直接端到端学控制，会有几个问题：

- 数据量需求大
- 训练不稳定
- 很难解释错误来源
- 容易把 teacher 的高层语义信息浪费掉

所以更合理的做法是：

1. 把现有感知和 teacher 信号整理成统一监督
2. 先蒸馏出一个紧凑的 `entry-centric representation`
3. 再基于这个表示训练局部策略

---

## 3. 本阶段到底蒸馏什么

这里蒸馏的不是“整图通用视觉特征”，而是：

- **目标入口相关的局部多模态表示**

也就是说，student 表示要回答的是：

1. 当前最值得关注的候选入口是谁
2. 它是门还是窗
3. 它是否可接近 / 可穿越
4. 当前应该靠近、绕行、还是继续搜索

所以蒸馏对象应该围绕：

- `top-K YOLO candidates`
- `candidate-aligned depth ROI`
- `LLM teacher high-level semantics`

而不是整张图的无差别表征。

---

## 4. 总体架构

建议结构分为 4 个模块：

1. `Candidate proposal`
2. `Candidate-level multimodal encoding`
3. `Teacher-guided distillation head`
4. `Policy input state`

可以抽象成：

```text
RGB + YOLO -> top-K candidates
Depth -> candidate-aligned ROI geometry
LLM teacher -> structured semantic guidance
------------------------------------------
Distilled entry representation z_entry
------------------------------------------
Policy / BC / PPO
```

---

## 5. 输入设计

### 5.1 RGB / YOLO 分支

这里不建议重新做整图目标检测，而是直接使用已经训练好的 `YOLO26` 作为：

- 候选区域提议器
- 候选语义分类器

每个候选至少保留：

- `class_name`
- `class_onehot`
- `confidence`
- `bbox_cx`
- `bbox_cy`
- `bbox_w`
- `bbox_h`
- `bbox_area_ratio`
- `aspect_ratio`

如果后续引入 ROI encoder，再额外取：

- `rgb_roi_feature`

### 5.2 Depth 分支

Depth 不再独立全图盲判，而是：

- 以 `YOLO` 候选框为锚点
- 在对应深度区域提取几何特征

每个候选保留：

- `entry_distance_cm`
- `surrounding_depth_cm`
- `clearance_depth_cm`
- `depth_gain_cm`
- `opening_width_cm`
- `traversable`
- `crossing_ready`
- `front_obstacle_present`
- `front_min_depth_cm`

如果后续引入 ROI encoder，再额外取：

- `depth_roi_feature`

### 5.3 Global 分支

保留策略训练所需的全局状态：

- `pose_x`
- `pose_y`
- `pose_z`
- `yaw_deg`
- `yaw_sin`
- `yaw_cos`
- `target_house_id`
- `current_house_id`
- `target_distance_cm`
- `target_bearing_deg`
- `target_bearing_sin`
- `target_bearing_cos`
- `movement_enabled`
- `history_actions`

### 5.4 Teacher 分支

Teacher 不是直接当在线控制器，而是作为监督来源。

建议蒸馏这几类 teacher 信号：

- `entry_state`
- `subgoal`
- `action_hint`
- `target_candidate_id`
- `risk_level`
- `teacher_reason_text`
- `teacher_reason_embedding`

---

## 6. Student 表示形式

建议 student 先输出一个统一的中间表示：

- `z_entry`

这个表示应编码：

1. 当前全局环境是否安全
2. 哪个候选最重要
3. 候选的语义类别
4. 候选的几何可通行性
5. 当前 teacher 建议的高层动作方向

推荐形式：

```text
z_entry = f(global_state, topK_candidate_features, teacher_guidance)
```

第一版不要求过大：

- `128` 或 `256` 维就够

---

## 7. 蒸馏目标

这一步不只蒸馏单一动作标签，而是多头蒸馏。

建议至少蒸馏 5 类目标：

### 7.1 `entry_state` 分类

监督 student 判断：

- `enterable_open_door`
- `enterable_door`
- `visible_but_blocked_entry`
- `front_blocked_detour`
- `window_visible_keep_search`
- `geometric_opening_needs_confirmation`
- `no_entry_confirmed`

### 7.2 `subgoal` 分类

监督 student 学高层子任务：

- `keep_search`
- `approach_entry`
- `align_entry`
- `detour_left`
- `detour_right`
- `cross_entry`
- `backoff_and_reobserve`

### 7.3 `action_hint` 分类

监督 student 学粗粒度动作方向：

- `forward`
- `yaw_left`
- `yaw_right`
- `left`
- `right`
- `backward`
- `hold`

### 7.4 `target_candidate_id` 分类

监督 student 学会：

- 当前 top-K 中哪个候选最值得关注

### 7.5 `teacher_reason_embedding` 回归

监督 student 吸收 LLM 的高层语义解释。

建议：

- 不直接蒸馏大语言模型内部 hidden state
- 而是蒸馏 `teacher_reason_text` 的稳定文本向量

---

## 8. 推荐损失函数

建议总损失写成：

```text
L = λ1 * L_entry_state
  + λ2 * L_subgoal
  + λ3 * L_action_hint
  + λ4 * L_target_candidate
  + λ5 * L_reason_embedding
```

如果后面再接 policy 学习：

```text
L_total = L_distill + λ6 * L_policy
```

推荐第一版策略：

- `λ1, λ2, λ3, λ4` 相对高
- `λ5` 稍低

原因：

- 先保证结构化决策可学
- 再让语义 embedding 做增强监督

---

## 9. 训练流程建议

建议拆成三阶段：

### Stage 1: 表示蒸馏预训练

输入：

- `entry_state.json`
- `teacher_output.json`
- `teacher_validation.json`

目标：

- 训练 `entry-centric student encoder`

输出：

- 蒸馏好的 `z_entry`

### Stage 2: 行为克隆初始化

输入：

- `z_entry`
- `teacher subgoal / action_hint`

目标：

- 让 student 先学会 teacher 的粗决策

### Stage 3: PPO 微调

输入：

- `z_entry`
- 环境交互反馈

目标：

- 学到比 teacher 更鲁棒的局部门进入与绕障策略

---

## 10. 为什么这条路线更稳

相比直接端到端学控制，这条路线有几个明显优势：

### 10.1 样本效率更高

因为：

- 低层语义检测已经由 `YOLO` 提供
- 低层几何已经由 `Depth` 提供
- 高层决策已经由 `LLM teacher` 提供

student 不需要从零学一切。

### 10.2 错误更可解释

如果后续策略失败，可以分层排查：

- 是 YOLO 候选错了
- 还是 depth ROI 错了
- 还是 teacher 错了
- 还是 student 蒸馏失败了

### 10.3 更适合部署

最终部署时：

- 不需要在线 LLM
- 不需要重型 teacher
- 只保留 student encoder + policy

这比纯 LLM 在线控制稳定得多。

---

## 11. 和论文主线的关系

如果论文主线是：

- 多模态入口感知
- LLM-guided teacher
- distillation
- local entry policy

那么这份表示蒸馏计划正好处在中间桥梁位置。

它解决的是：

- 如何把 `LLM + YOLO + Depth` 的异构知识
- 变成一个可训练、可部署、可用于 RL 的统一状态表示

这一步是整篇方法成立的关键。

---

## 12. 第一版实现建议

第一版不要过度复杂化，建议先做：

1. 不加 ROI visual encoder
2. 只用结构化状态
3. 蒸馏：
   - `entry_state`
   - `subgoal`
   - `action_hint`
   - `target_candidate_id`
4. `teacher_reason_embedding` 先留接口，不急着第一天就接上

也就是说，第一版先验证：

- 这套候选级状态表示是不是够用

如果第一版已经明显有效，再引入：

- RGB ROI encoder
- Depth ROI encoder
- text embedding distillation

---

## 13. 后续代码模块建议

建议后面实现时分成这几个模块：

1. `teacher_validator`
   - 清洗 teacher 信号

2. `entry_state_builder`
   - 构建候选级状态

3. `distillation_dataset_export`
   - 导出 train/val

4. `representation_distillation_trainer`
   - 训练 student 表示

5. `entry_policy_trainer`
   - 基于 student 表示训练 policy

---

## 14. 一句话总结

这条路线的核心不是：

- 直接让大模型控飞

而是：

- **先把 `LLM + YOLO + Depth` 的知识蒸馏成一个入口相关表示**
- **再用这个表示作为局部策略的状态输入**

这是比端到端控制更稳、更可解释、也更适合部署的方案。
