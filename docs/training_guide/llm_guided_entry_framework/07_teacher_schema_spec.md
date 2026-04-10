# Teacher Schema Specification

## 1. 文档目标

这份文档用于固定 `LLM-guided teacher` 的：

- 输入来源
- 输出字段
- 字段约束
- 合法取值集合
- 质检规则

目标是让 teacher 输出变成一个**稳定、可验证、可蒸馏**的数据接口，而不是一组随 prompt 漂移的自然语言。

---

## 2. Teacher 的角色边界

teacher 在本项目中的职责是：

1. 解释当前局部观测
2. 判断当前入口状态
3. 选择当前局部子任务
4. 提供粗粒度动作提示
5. 给出简洁解释，供后续蒸馏

teacher 不负责：

- 高频低层闭环控制
- 逐毫秒避障
- 直接替代 student policy

一句话：

- `teacher` 给方向
- `student` 负责高频执行

---

## 3. Teacher 输入规范

teacher 每次推理建议接收三类输入。

### 3.1 原始视觉输入

- `rgb.png`
- `depth_preview.png`

说明：

- `rgb` 负责语义判断
- `depth_preview` 负责让 teacher 理解大致几何布局

### 3.2 结构化感知摘要

建议包含：

- `front_obstacle.present`
- `front_obstacle.front_min_depth_cm`
- `front_obstacle.severity`
- `top_k_yolo_candidates`
- `semantic_depth_assessment`
- `chosen_depth_candidate`
- `entry_distance_cm`
- `opening_width_cm`
- `traversable`
- `crossing_ready`
- `final_entry_state`

### 3.3 控制上下文

建议包含：

- `pose`
- `yaw`
- `target_house_id`
- `current_house_id`
- `recent_actions`
- `current_task_label`

---

## 4. Teacher 输出主 Schema

推荐固定为：

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

---

## 5. 字段定义

### 5.1 `entry_state`

类型：
- `string`

取值：
- `enterable_open_door`
- `enterable_door`
- `visible_but_blocked_entry`
- `front_blocked_detour`
- `window_visible_keep_search`
- `geometric_opening_needs_confirmation`
- `no_entry_confirmed`

要求：
- 必填
- 必须来自固定集合

### 5.2 `subgoal`

类型：
- `string`

取值：
- `keep_search`
- `approach_entry`
- `align_entry`
- `detour_left`
- `detour_right`
- `cross_entry`
- `backoff_and_reobserve`

要求：
- 必填
- 必须与 `entry_state` 相容

### 5.3 `action_hint`

类型：
- `string`

取值：
- `forward`
- `yaw_left`
- `yaw_right`
- `left`
- `right`
- `backward`
- `hold`

要求：
- 必填
- 必须是粗粒度动作，不得输出连续数值控制量

### 5.4 `target_candidate_id`

类型：
- `int`

范围：
- `-1`
- `0..K-1`

含义：
- `-1` 表示当前没有可靠目标候选
- `>=0` 表示当前最值得关注的候选索引

### 5.5 `risk_level`

类型：
- `string`

取值：
- `low`
- `medium`
- `high`

### 5.6 `reason`

类型：
- `string`

要求：
- 一句短解释
- 不超过约 24 个英文词或同等信息量
- 不要引入图中没有明确证据支持的额外场景幻想

### 5.7 `confidence`

类型：
- `float`

范围：
- `[0, 1]`

---

## 6. Teacher 输出一致性规则

为了防止 teacher 输出自相矛盾，建议加规则检查。

### 6.1 `entry_state` 与 `subgoal`

#### 合理组合

- `enterable_open_door` -> `approach_entry / align_entry / cross_entry`
- `enterable_door` -> `approach_entry / align_entry`
- `visible_but_blocked_entry` -> `detour_left / detour_right / backoff_and_reobserve`
- `front_blocked_detour` -> `detour_left / detour_right / backoff_and_reobserve`
- `window_visible_keep_search` -> `keep_search`
- `geometric_opening_needs_confirmation` -> `keep_search / approach_entry`
- `no_entry_confirmed` -> `keep_search`

#### 不合理组合示例

- `window_visible_keep_search` + `cross_entry`
- `front_blocked_detour` + `forward`
- `no_entry_confirmed` + `target_candidate_id >= 0` 且高置信度

### 6.2 `subgoal` 与 `action_hint`

#### 合理组合

- `approach_entry` -> `forward / yaw_left / yaw_right`
- `align_entry` -> `yaw_left / yaw_right / left / right`
- `detour_left` -> `left / yaw_left`
- `detour_right` -> `right / yaw_right`
- `cross_entry` -> `forward`
- `backoff_and_reobserve` -> `backward / hold`
- `keep_search` -> `yaw_left / yaw_right / hold`

---

## 7. Teacher 输出质量分级

建议把每条 teacher 输出分成三档：

### 7.1 `valid`

满足：
- schema 完整
- 所有字段合法
- 规则一致性通过

### 7.2 `weak_valid`

满足：
- schema 完整
- 字段合法
- 但 `confidence` 偏低或 `reason` 含糊

### 7.3 `invalid`

出现以下任一情况：
- schema 不完整
- 输出值不在合法集合内
- 自相矛盾
- 明显与前障碍和语义证据冲突

---

## 8. 推荐保存格式

每次 teacher 推理建议保存：

```json
{
  "sample_id": "fusion_20260409_101400",
  "teacher_model": "claude-sonnet-4-6",
  "input_refs": {
    "rgb": "labeling/rgb.png",
    "depth_preview": "labeling/depth_preview.png",
    "fusion_result": "labeling/fusion_result.json"
  },
  "teacher_output": {
    "entry_state": "enterable_open_door",
    "subgoal": "approach_entry",
    "action_hint": "forward",
    "target_candidate_id": 0,
    "risk_level": "low",
    "reason": "Open door is visible and traversable but still far, approach first.",
    "confidence": 0.84
  },
  "validation": {
    "status": "valid",
    "issues": []
  }
}
```

---

## 9. 和后续模块的接口关系

### 输入给 `entry_state_builder`

teacher 输出中的：

- `entry_state`
- `subgoal`
- `action_hint`
- `target_candidate_id`
- `risk_level`

将直接进入 student 训练样本构造。

### 输入给 `distillation_dataset_export`

teacher 输出中的：

- `reason`
- `confidence`
- `validation.status`

将用于：

- 生成 `teacher_reason_embedding`
- 过滤低质量 teacher 样本

---

## 10. 第一版实现建议

第一版建议只强制要求：

- `entry_state`
- `subgoal`
- `action_hint`
- `target_candidate_id`
- `risk_level`
- `reason`
- `confidence`

先不要在第一版里塞更多字段。

目标是：

- schema 稳定
- 样本可批量产出
- 后续 student 能直接用

---

## 11. 一句话总结

teacher schema 的关键不是“说得多丰富”，而是：

- 字段固定
- 含义稳定
- 规则自洽
- 能直接蒸馏
