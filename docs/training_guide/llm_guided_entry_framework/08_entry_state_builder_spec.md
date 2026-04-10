# Entry State Builder Specification

## 1. 文档目标

这份文档定义 `entry_state_builder` 的职责：

- 从现有多模态结果中构建 student 输入状态
- 把 `YOLO + depth ROI + fusion + pose + teacher output` 统一整理成固定格式

它是后续训练代码的**状态构造器规范**。

---

## 2. 模块输入

`entry_state_builder` 的输入建议来自已有结果包：

- `rgb.png`
- `depth_cm.png`
- `yolo_result.json`
- `depth_result.json`
- `fusion_result.json`
- `state_excerpt.json`
- `pose_history_summary.json`
- `teacher_output.json`

如果 teacher 还没跑，也应支持“无 teacher”模式。

---

## 3. 模块输出

输出一个固定的 `entry_state.json`，结构如下：

```json
{
  "sample_id": "...",
  "global_state": { ... },
  "candidates": [ ... ],
  "teacher_targets": { ... },
  "metadata": { ... }
}
```

---

## 4. 构建流程

### Step 1: 读取全局状态

来源：
- `state_excerpt.json`
- `pose_history_summary.json`

提取：
- 当前 pose
- yaw
- 当前房屋/目标房屋
- 最近动作
- target 距离和方位

### Step 2: 读取 YOLO 候选

来源：
- `yolo_result.json`

提取：
- `class_name`
- `confidence`
- `bbox`

并排序：
- `open door`
- `door`
- `close door`
- `window`
- 同类按 confidence 降序

### Step 3: 对齐 depth ROI

来源：
- `fusion_result.json`
- `depth_result.json`

原则：
- 优先使用已经在 fusion 里为语义候选计算出的 ROI 几何特征
- 不再重新从整张深度图盲目提全局 opening

### Step 4: 合成候选状态

对每个 top-K 候选，生成统一字段：

- 语义字段
- 几何字段
- 候选排序字段
- `valid_mask`

### Step 5: 接 teacher targets

来源：
- `teacher_output.json`

若缺失：
- 用空 teacher 占位
- 仍可导出样本，但标记为 `teacher_available = 0`

---

## 5. Global State 规范

推荐输出：

```json
{
  "pose_x": 2000.0,
  "pose_y": 85.3,
  "pose_z": 225.0,
  "yaw_deg": -1.7,
  "yaw_sin": -0.03,
  "yaw_cos": 0.99,
  "front_obstacle_present": 0,
  "front_min_depth_cm": 230.0,
  "front_obstacle_severity": "clear",
  "target_house_id": 1,
  "current_house_id": 1,
  "target_distance_cm": 620.0,
  "target_bearing_deg": -18.0,
  "target_bearing_sin": -0.31,
  "target_bearing_cos": 0.95,
  "history_actions": ["forward", "yaw_left"]
}
```

### 5.1 必须保证的字段

- `pose_x`
- `pose_y`
- `pose_z`
- `yaw_deg`
- `front_obstacle_present`
- `front_min_depth_cm`

### 5.2 推荐自动生成字段

- `yaw_sin`
- `yaw_cos`
- `target_bearing_sin`
- `target_bearing_cos`

这样后续网络不需要自己学角度周期性。

---

## 6. Candidate 规范

每个 candidate 建议格式：

```json
{
  "candidate_id": 0,
  "valid_mask": 1,
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

### 6.1 候选来源优先级

1. `fusion.semantic_detection`
2. `fusion.semantic_depth_assessment`
3. `yolo_result.detections`

如果一个候选没有深度 ROI 结果：
- 几何字段填 `0`
- 但语义字段仍然保留

### 6.2 候选对齐规则

如果 YOLO 检测框和 depth ROI 匹配：
- 直接写入该候选

如果一个 YOLO 候选没有匹配到 depth 结果：
- 视为“语义可见但几何未知”

这样可以支持：
- `geometric_opening_needs_confirmation`

---

## 7. Teacher Targets 规范

teacher targets 建议直接内嵌为：

```json
{
  "teacher_available": 1,
  "entry_state": "enterable_open_door",
  "subgoal": "approach_entry",
  "action_hint": "forward",
  "target_candidate_id": 0,
  "risk_level": "low",
  "teacher_reason_text": "Open door is visible and traversable but still far, approach first.",
  "teacher_reason_embedding": [ ... ],
  "confidence": 0.84
}
```

如果 teacher 缺失：

```json
{
  "teacher_available": 0,
  "entry_state": "",
  "subgoal": "",
  "action_hint": "",
  "target_candidate_id": -1,
  "risk_level": "",
  "teacher_reason_text": "",
  "teacher_reason_embedding": [],
  "confidence": 0.0
}
```

---

## 8. 填充与裁剪策略

### 8.1 top-K 固定长度

建议 `K=3`。

如果候选多于 `K`：
- 只保留前 `K`

如果候选少于 `K`：
- 用零候选补齐

### 8.2 零候选格式

```json
{
  "candidate_id": -1,
  "valid_mask": 0,
  "class_name": "",
  "class_onehot": [0, 0, 0, 0],
  "confidence": 0.0,
  "bbox_cx": 0.0,
  "bbox_cy": 0.0,
  "bbox_w": 0.0,
  "bbox_h": 0.0,
  "bbox_area_ratio": 0.0,
  "aspect_ratio": 0.0,
  "entry_distance_cm": 0.0,
  "surrounding_depth_cm": 0.0,
  "clearance_depth_cm": 0.0,
  "depth_gain_cm": 0.0,
  "opening_width_cm": 0.0,
  "traversable": 0,
  "crossing_ready": 0,
  "candidate_rank": -1
}
```

---

## 9. 归一化职责

建议 `entry_state_builder` 做两层输出：

### 9.1 原始值层

便于人工检查：
- `entry_distance_cm = 280.0`
- `opening_width_cm = 112.0`

### 9.2 归一化值层

便于训练：
- `entry_distance_norm`
- `opening_width_norm`
- `depth_gain_norm`

这样后续训练代码不需要自己再猜归一化规则。

---

## 10. 元数据

建议附带：

```json
{
  "metadata": {
    "sample_id": "...",
    "task_label": "...",
    "source_run_dir": "...",
    "target_house_id": 1,
    "current_house_id": 1,
    "teacher_source": "anthropic_claude_scene_result.json",
    "fusion_source": "fusion_result.json"
  }
}
```

---

## 11. 输出文件建议

每个样本建议至少输出：

- `entry_state.json`
- 可选：`entry_state_compact.npz`

其中：
- JSON 方便人工看
- NPZ 方便训练读

---

## 12. 一句话总结

`entry_state_builder` 的本质是：

- 把已有结果包整理成一个统一、固定、可训练的候选级状态表示。
