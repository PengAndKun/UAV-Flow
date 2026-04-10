# Teacher Validator Specification

## 1. 文档目标

这份文档定义 `teacher validator` 的职责：

- 校验 `LLM-guided teacher` 输出是否符合 schema
- 判断样本质量等级
- 输出结构化验证结果

目标是保证：

- 进入蒸馏训练集的 teacher 样本是可控的
- 无效或冲突样本不会污染 student 训练

---

## 2. 为什么必须有 validator

teacher 是 LLM 输出，不是严格程序逻辑，因此天然存在这些风险：

- 字段缺失
- 值不在合法集合内
- 自相矛盾
- 和已有感知证据冲突
- 表达模糊，无法蒸馏

如果没有 validator：

- 坏 teacher 会直接进入数据集
- student 会学到冲突规则
- 后面很难区分是模型设计问题还是 teacher 数据问题

所以 validator 不是可选项，而是 teacher 路线的必要基础。

---

## 3. 输入

validator 建议接收三类输入：

### 3.1 teacher 输出

例如：

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

### 3.2 感知依据

建议读取：

- `fusion_result.json`
- `yolo_result.json`
- `depth_result.json`
- `state_excerpt.json`

### 3.3 配置

validator 建议支持：

- 类别映射表
- 合法取值集合
- 置信度阈值
- 是否严格模式

---

## 4. 输出

validator 输出建议为：

```json
{
  "status": "valid",
  "score": 0.92,
  "errors": [],
  "warnings": [],
  "normalized_teacher_output": { ... }
}
```

字段说明：

- `status`
  - `valid`
  - `weak_valid`
  - `invalid`
- `score`
  - `[0, 1]`
- `errors`
  - 硬错误列表
- `warnings`
  - 软警告列表
- `normalized_teacher_output`
  - 经过标准化后的 teacher 结果

---

## 5. 校验分层

建议分 4 层校验。

### 5.1 Level 1: Schema 校验

检查：

- 必填字段是否存在
- 字段类型是否正确
- 枚举值是否合法
- `confidence` 是否在 `[0, 1]`
- `target_candidate_id` 是否在合法范围

### 5.2 Level 2: 内部一致性校验

检查：

- `entry_state` 和 `subgoal` 是否一致
- `subgoal` 和 `action_hint` 是否一致
- `risk_level` 是否和行为方向明显冲突

### 5.3 Level 3: 感知证据一致性校验

检查：

- teacher 是否与 fusion 结果强冲突
- teacher 是否违反前障碍优先原则
- teacher 是否把 `window` 当成进入目标

### 5.4 Level 4: 质量校验

检查：

- `reason` 是否过长或过空
- `confidence` 是否过低
- 输出是否过于模糊

---

## 6. 硬错误规则

这些错误一旦出现，直接判为 `invalid`。

### 6.1 Schema 缺失

- 缺少：
  - `entry_state`
  - `subgoal`
  - `action_hint`
  - `target_candidate_id`
  - `risk_level`
  - `reason`
  - `confidence`

### 6.2 非法枚举值

例如：

- `subgoal = move_forward_now`
- `risk_level = urgent`

### 6.3 明显自相矛盾

例如：

- `window_visible_keep_search` + `cross_entry`
- `front_blocked_detour` + `forward`
- `no_entry_confirmed` + `target_candidate_id=0` 且高置信度

### 6.4 与硬证据强冲突

例如：

- `front_obstacle.severity = high`
  但 teacher 仍输出：
  - `cross_entry`
  - `action_hint = forward`

或者：

- top semantic class 明确是 `window`
  但 teacher 输出：
  - `enterable_open_door`

---

## 7. 软警告规则

这些不一定判 invalid，但要记 warning。

### 7.1 低置信度

例如：

- `confidence < 0.55`

### 7.2 解释过短

例如：

- `reason = "maybe okay"`

### 7.3 解释过长

例如：

- 超过建议长度很多
- 写成大段推理

### 7.4 目标候选不明确

例如：

- 有明显 door-like 候选
  但 `target_candidate_id = -1`

### 7.5 子任务偏保守但不冲突

例如：

- `enterable_open_door`
  但输出 `keep_search`

这不一定错误，但应记录 warning。

---

## 8. 推荐评分机制

建议从 `1.0` 开始扣分。

### 硬错误

每个硬错误：
- 直接判 `invalid`
- `score <= 0.3`

### 警告

每个 warning 可扣：
- `0.05 ~ 0.15`

### 状态划分建议

- `valid`
  - 无硬错误
  - `score >= 0.80`
- `weak_valid`
  - 无硬错误
  - `0.55 <= score < 0.80`
- `invalid`
  - 存在硬错误
  - 或 `score < 0.55`

---

## 9. 标准化步骤

validator 不只是报错，还应做 normalization。

建议标准化内容：

- class / subgoal / action_hint 全部转成固定小写 token
- 同义词归并
  - `open_door` -> `open door`
  - `turn_left` -> `yaw_left`
- `reason` 去首尾空格
- `confidence` 转 float

输出：

- `normalized_teacher_output`

这样后面 dataset export 不需要再做一次清洗。

---

## 10. 与感知结果的最小对齐规则

建议先只用最稳定的几条感知规则做校验：

### 10.1 前障碍优先

如果：

- `front_obstacle.present = true`
- 且 `severity = high`

则 teacher 不应输出：

- `cross_entry`
- `forward` 作为主动作

### 10.2 窗口不可作为可进入目标

如果：

- top semantic class = `window`

则 teacher 不应输出：

- `enterable_open_door`
- `enterable_door`
- `cross_entry`

### 10.3 无候选时不能强推进入

如果：

- `entry_visible = false`
- 且 `final_entry_state = no_entry_confirmed`

则 teacher 不应输出：

- `cross_entry`
- `target_candidate_id >= 0` 且高置信度

---

## 11. 推荐输出文件

每条样本建议写：

- `teacher_output.json`
- `teacher_validation.json`

其中 `teacher_validation.json` 结构建议：

```json
{
  "status": "weak_valid",
  "score": 0.71,
  "errors": [],
  "warnings": [
    "low_confidence",
    "target_candidate_missing"
  ],
  "normalized_teacher_output": {
    "entry_state": "geometric_opening_needs_confirmation",
    "subgoal": "keep_search",
    "action_hint": "yaw_right",
    "target_candidate_id": -1,
    "risk_level": "medium",
    "reason": "Opening is visible but semantic confirmation is weak.",
    "confidence": 0.58
  }
}
```

---

## 12. 第一版实现建议

第一版只实现下面这些就够：

1. schema 校验
2. 枚举合法性检查
3. 前障碍优先检查
4. `window` 不可进入检查
5. 低置信度 warning
6. 统一 normalized 输出

先不要一上来做太复杂的规则图谱。

---

## 13. 一句话总结

teacher validator 的作用不是“证明 LLM 很聪明”，而是：

- 把 teacher 输出变成可控、可筛、可蒸馏的数据源
