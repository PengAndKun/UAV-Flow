# Representation Distillation Trainer Specification

## 1. 目标

本文件定义 `Phase 2.5` 中表示蒸馏训练器的实现规范。

这一阶段的目标不是直接训练端到端飞控，而是先训练一个：

- 融合 `YOLO / RGB` 候选入口语义
- 融合 `Depth ROI` 几何可通行信息
- 融合 `LLM teacher` 高层引导

的紧凑 student 表示模型。

后续局部策略学习、行为克隆和 PPO 微调，都基于这份表示继续进行。

一句话概括：

- **先学会看懂入口相关状态**
- **再学会如何行动**

---

## 2. 输入数据来源

训练器默认读取：

- `phase2_multimodal_fusion_analysis/exports/.../train.jsonl`
- `phase2_multimodal_fusion_analysis/exports/.../val.jsonl`

每条 JSONL 样本应来自：

- [entry_state.json](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/results/fusion_20260413_164730_new3/labeling/entry_state.json)
- [teacher_output.json](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/results/fusion_20260413_164730_new3/labeling/teacher_output.json)
- [teacher_validation.json](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/results/fusion_20260413_164730_new3/labeling/teacher_validation.json)

训练器只依赖导出后的统一 schema，不直接回读 `results/` 下的原始分析文件。

---

## 3. 训练目标定位

训练器的第一版只做 **representation distillation pretrain**，不直接做 PPO。

### 3.1 训练目标

Student 需要学会：

- 当前是否存在目标相关入口
- 当前最值得关注的候选是谁
- 当前该靠近、绕行、重定向还是继续搜索
- 当前是否已经满足穿越条件

### 3.2 不在本阶段完成的事情

本阶段先不做：

- 连续控制回归
- 长时序 value learning
- 端到端导航
- 全任务 house-search

---

## 4. 输入结构

训练器读取的状态由三部分组成：

### 4.1 `global_state`

至少包括：

- UAV pose / yaw 编码
- 全局前障碍信息
- `target_house_id`
- `target_house_in_fov`
- `target_house_expected_side`
- `target_house_expected_image_x`
- `target_distance_cm`
- `target_bearing_deg / sin / cos`
- `current_house_id`

### 4.2 `candidates`

固定长度 `top-K`，第一版建议：

- `K = 3`

每个 candidate 至少包括：

- YOLO 语义
  - `class_id`
  - `class_onehot`
  - `confidence`
  - `bbox cx/cy/w/h`
- 深度几何
  - `entry_distance_cm`
  - `opening_width_cm`
  - `depth_gain_cm`
  - `traversable`
  - `crossing_ready`
- target-conditioned 信息
  - `candidate_target_match_score`
  - `candidate_total_score`
  - `candidate_is_target_house_entry`
  - `candidate_target_side_match`
  - `candidate_center_in_target_bbox`
  - `candidate_near_target_bbox`

### 4.3 `teacher_targets`

训练 supervision 来自 teacher，分两套：

- 普通监督
  - `entry_state`
  - `subgoal`
  - `action_hint`
- 目标条件监督
  - `target_conditioned_state`
  - `target_conditioned_subgoal`
  - `target_conditioned_action_hint`
  - `target_conditioned_target_candidate_id`
  - `target_conditioned_confidence`

---

## 5. Student 网络设计

第一版建议使用 **结构化多分支 MLP**，而不是一上来就上大视觉 backbone。

## 5.1 整体结构

```text
global_state  ------> global_encoder ------
                                          \
candidates[0..K-1] -> candidate_encoder --- fusion_mlp -> z_entry
                                          /
teacher aux mask -------------------------

z_entry -> heads
```

### 5.2 `global_encoder`

输入：

- 全局数值特征向量

结构建议：

- `Linear -> LayerNorm -> ReLU -> Linear`

输出维度建议：

- `64`

### 5.3 `candidate_encoder`

每个候选共享同一个 encoder。

输入：

- 单个 candidate 的结构化特征

结构建议：

- `Linear -> LayerNorm -> ReLU -> Linear`

输出维度建议：

- `64`

对 `K=3` 个候选可做：

- 逐个编码
- 再拼接
- 或做简单 attention pooling

第一版建议直接：

- `concat(candidate_1, candidate_2, candidate_3)`

实现最稳。

### 5.4 `fusion_mlp`

输入：

- `global_feature`
- `candidate_feature_concat`

输出：

- `z_entry`

输出维度建议：

- `128` 或 `256`

第一版优先：

- `128`

---

## 6. Head 设计

第一版建议至少训练 5 个 head。

### 6.1 普通入口状态 head

分类目标：

- `entry_state`

### 6.2 普通子任务 head

分类目标：

- `subgoal`

### 6.3 普通动作提示 head

分类目标：

- `action_hint`

### 6.4 目标条件状态 head

分类目标：

- `target_conditioned_state`

### 6.5 目标条件子任务 head

分类目标：

- `target_conditioned_subgoal`

### 6.6 目标条件动作提示 head

分类目标：

- `target_conditioned_action_hint`

### 6.7 目标候选索引 head

分类目标：

- `target_conditioned_target_candidate_id`

第一版可以把无目标候选统一映射成：

- `-1 -> ignore_index`

或者

- `extra_null_class`

建议第一版用：

- `extra_null_class`

训练更稳定。

---

## 7. Loss 设计

第一版总损失建议写成：

```text
L_total =
  λ1 * L_entry_state
  + λ2 * L_subgoal
  + λ3 * L_action_hint
  + λ4 * L_target_state
  + λ5 * L_target_subgoal
  + λ6 * L_target_action
  + λ7 * L_target_candidate
```

### 7.1 推荐初始权重

- `λ1 = 1.0`
- `λ2 = 1.0`
- `λ3 = 0.7`
- `λ4 = 1.2`
- `λ5 = 1.2`
- `λ6 = 0.8`
- `λ7 = 0.8`

理由：

- 目标条件任务是主线，略高权重
- 动作提示比状态略弱，避免过早过拟合动作分布

### 7.2 类别不平衡处理

当前数据明显不平衡，因此建议：

- 每个分类 head 使用 class weight
- 或 focal loss

第一版建议：

- 先用 `CrossEntropy + class_weight`

更容易解释，也更稳定。

---

## 8. 训练顺序

第一版建议按下面顺序训练：

### Step 1

只训：

- `entry_state`
- `target_conditioned_state`

目的是先让 student 学会“看懂场景”。

### Step 2

再加：

- `subgoal`
- `target_conditioned_subgoal`

目的是让表示学会高层决策。

### Step 3

最后再加：

- `action_hint`
- `target_conditioned_action_hint`
- `target_candidate_id`

目的是让表示具备局部可执行性。

也就是说：

- **先状态**
- **再子任务**
- **最后动作**

不要一开始所有头一起硬训。

---

## 9. 训练超参数建议

第一版建议：

- optimizer: `AdamW`
- learning rate: `1e-3`
- weight decay: `1e-4`
- batch size: `32`
- epochs: `80`
- early stopping patience: `12`

如果样本还只有 `100~200` 左右，建议：

- batch size 先用 `16` 或 `32`
- 不要过大

---

## 10. 验证指标

### 10.1 主指标

- `entry_state accuracy`
- `target_conditioned_state accuracy`
- `target_conditioned_subgoal accuracy`
- `target_conditioned_action_hint accuracy`

### 10.2 更关键的类级指标

必须单独报：

- `target_house_entry_approachable`
- `target_house_entry_blocked`
- `non_target_house_entry_visible`
- `target_house_not_in_view`

建议同时报：

- per-class precision
- per-class recall
- per-class F1

### 10.3 候选选择指标

- `target_candidate_id accuracy`

如果这个指标太低，说明：

- 表示学会了“状态”
- 但还没学会“具体盯哪个候选”

---

## 11. 第一版成功标准

我建议第一版不要定太高目标，先看这些：

- `target_conditioned_state accuracy >= 0.75`
- `target_conditioned_subgoal accuracy >= 0.70`
- `non_target_house_entry_visible` 的 precision 足够高
- `target_house_entry_blocked` 不要被大量误判成 `approach`

如果达不到，不要立刻换模型，先检查：

- 数据类别分布
- teacher 质量
- target-conditioned 匹配规则

---

## 12. 输出内容

训练器第一版建议输出：

- `checkpoints/best.pt`
- `checkpoints/last.pt`
- `metrics.json`
- `per_class_report.json`
- `train_curve.png`
- `confusion_matrix_target_state.png`
- `confusion_matrix_target_subgoal.png`

---

## 13. 与后续模块的关系

这一步完成后，后面两条线都可以接：

### 13.1 BC / imitation line

用 `z_entry` 作为状态表示，继续训练：

- policy head
- action classifier

### 13.2 RL line

用蒸馏好的 encoder 初始化 RL policy：

- 冻结 encoder 先训 head
- 再部分解冻联合微调

推荐顺序：

1. `representation distillation`
2. `behavior cloning`
3. `PPO fine-tuning`

---

## 14. 第一版不建议做的事

为了避免范围过大，第一版训练器先不要做：

- teacher reason embedding 蒸馏
- ROI 图像 encoder
- 整图 transformer encoder
- 时序 transformer
- 多帧历史堆叠

这些都可以作为 v2。

第一版先把：

- 结构化 target-conditioned 状态
- teacher 多头监督
- 稳定训练闭环

做扎实更重要。

---

## 15. 实现建议

实现顺序建议：

1. `dataset reader`
2. `feature vector builder`
3. `student network`
4. `multi-head loss`
5. `trainer`
6. `evaluator`

也就是：

- 先把输入读对
- 再把监督对齐
- 最后才是训练 loop

这会比直接从 trainer 开始写更稳。
