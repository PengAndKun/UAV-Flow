# 37. LLM 多建筑控制与搜索完成判定规范

本文档定义下一步需要实现的两个核心能力：

1. 明确 `search task finished` 的触发条件，尤其是“找到目标房屋入口并靠近到约 3m”即完成当前 house 搜索任务。
2. 增加一个任务分析入口，使自然语言任务如“先探索 house 1 再探索 house 3”可以被 LLM 解析成有序多建筑目标队列，并驱动控制器依次切换目标。

这份文档先约定技术规范，不直接修改代码。

---

## 1. 当前问题

在 `memory_episode_20260428_121539_search house 1 and house 2` 的简单测试中，LLM 控制链路已经跑通：

- LLM 可以读取 memory-aware fusion 结果。
- LLM 可以输出动作符号。
- `w/a/s/d/r/f` 可以连续执行。
- `q/e` 可以单步执行并触发立即截图。
- LLM 控制文件可以独立保存到 `captures_remote/llm_control_sessions/`。

但是决策逻辑暴露出一个关键问题：

当前 fusion/depth 中的 `traversable=0` 或 `cross_ready=0` 被 LLM 理解得过于保守，导致它看到 house 1 的 open door 时反复左右绕行，而不是向入口靠近。

这需要重新区分两个概念：

- **Approach readiness**：是否可以朝目标入口靠近。
- **Cross readiness**：是否已经足够近、足够对齐，可以穿过入口或进入室内。

对于当前阶段的数据采集，我们的目标不是让 UAV 真正穿门进入，而是让 UAV 找到目标 house 的入口并靠近到约 3m。因此：

`cross_ready=0` 不应该阻止 `approach_target_entry`。

---

## 2. 搜索任务完成定义

### 2.1 当前阶段的任务完成目标

当前 LLM 控制采集阶段，每个 house 的任务目标定义为：

给定目标 house，UAV 在室外搜索该 house 的入口；当确认目标入口并靠近到入口附近约 3m 时，认为当前 house 搜索任务完成，然后切换到下一个目标 house。

这里的“完成”不是“进入房屋内部”，而是：

- 目标 house 已被正确锁定。
- 目标 house 的入口候选已被识别。
- UAV 已靠近到足够近的入口前方位置。
- 后续可以把该状态作为策略训练或进入动作训练的起点。

### 2.2 Found-entry completion

推荐第一版完成条件：

```json
{
  "finish_type": "target_entry_reached",
  "target_house_id": "001",
  "entry_id": "H001_E02",
  "entry_distance_cm": 300.0,
  "distance_threshold_cm": 300.0,
  "association_confidence": 0.65,
  "observation_count": 2,
  "target_match_score": 0.5
}
```

触发条件建议：

- `target_house_id == current target house`
- 当前最佳入口是 door-like 类别：`open door / door / close door`
- 入口关联到当前目标 house
- `entry_distance_cm <= 300`
- `association_confidence >= 0.55`
- `observation_count >= 2`
- 最近若干帧没有明显判为 `non_target_house_entry_visible`

如果视觉和 depth 证据较强，可以允许较宽松版本：

- `entry_distance_cm <= 350`
- `target_match_score >= 0.6`
- `entry_association == target_house_entry`

这适合无人机停在门前 3m 左右，准备切换到下一个 house。

### 2.3 Approach-ready 与 cross-ready 的区别

当前最重要的修正是：

`cross_ready=0` 只表示“不要穿门”，不表示“不要靠近”。

建议规则：

```text
if target entry visible
and entry belongs to target house
and distance_cm > finish_distance_cm
and front obstacle is not severe:
    action = forward
```

也就是说，如果 house 1 的 open door 已经可见，并且它属于 house 1，那么即使 depth 认为入口宽度不足、还不能穿越，也应该允许 UAV 向前靠近。

只有以下情况才不应该 forward：

- 当前门不是目标 house 的门。
- 当前主要检测是 window。
- 前方近距离有严重障碍。
- UAV 已经离门很近，继续 forward 可能撞门或撞墙。
- 入口归属不稳定，需要先转向补观察。

### 2.4 No-entry completion

对于像 house 2 这种“绕一圈没找到门”的情况，需要另一个完成条件：

```json
{
  "finish_type": "no_entry_after_full_coverage",
  "target_house_id": "002",
  "visited_coverage_ratio": 0.75,
  "observed_coverage_ratio": 0.5,
  "total_observations": 8,
  "reliable_entry_count": 0
}
```

触发条件建议沿用已有 memory 逻辑：

- `visited_coverage_ratio >= 0.75`
- `observed_coverage_ratio >= 0.5`
- `total_observations >= 8`
- 没有可靠 target-house entry
- 可疑 window / non-target entry 已被记录或拒绝

完成后状态应为：

```json
{
  "house_task_status": "NO_ENTRY_CONFIRMED",
  "finish_type": "no_entry_after_full_coverage",
  "next_action": "switch_to_next_house"
}
```

### 2.5 需要人工复核的完成失败状态

如果出现以下情况，不应该自动切换到下一个 house，而应进入 `NEEDS_REVIEW`：

- LLM 连续多次左右摆动，没有靠近或扩大视角。
- target house 不在视野，但连续多次朝同一方向转向仍无法找回。
- fusion 判断 `enterable_door`，但 target-conditioned 判断 `target_house_not_in_view`。
- 入口距离小于 3m，但入口归属置信度低。
- 同一入口在 memory 中既被标为 target，又被标为 non-target。

---

## 3. 多建筑搜索任务

### 3.1 任务输入

面板中应增加一个新的按钮，例如：

`Analyze LLM Task`

该按钮对应一个小窗口或区域，包含：

- 自然语言任务输入框
- API base URL
- API key
- model
- `Analyze Task` 按钮
- 解析结果预览
- `Apply Plan` 按钮

示例输入：

```text
先探索 house 1，再探索 house 3。
```

或：

```text
Search house 1 first, then house 3. If house 1 entry is reached within 3m, switch to house 3.
```

### 3.2 LLM 任务解析输出

LLM 不应该直接输出动作，而是先输出结构化任务计划：

```json
{
  "plan_id": "llm_task_plan_20260428_123000",
  "task_text": "先探索 house 1，再探索 house 3。",
  "ordered_targets": [
    {
      "order": 1,
      "house_id": "001",
      "house_alias": "house 1",
      "goal": "search_entry",
      "finish_condition": "target_entry_reached_or_no_entry_after_full_coverage",
      "status": "pending"
    },
    {
      "order": 2,
      "house_id": "003",
      "house_alias": "house 3",
      "goal": "search_entry",
      "finish_condition": "target_entry_reached_or_no_entry_after_full_coverage",
      "status": "pending"
    }
  ],
  "execution_policy": {
    "entry_reached_distance_cm": 300,
    "max_decisions_per_house": 40,
    "allow_no_entry_completion": true,
    "stop_on_needs_review": true
  },
  "reason": "The user explicitly requested sequential exploration of house 1 and house 3."
}
```

### 3.3 House ID 映射

LLM 解析任务时必须读取已有 house registry，而不是凭空猜测。

输入给 LLM 的 house registry 应至少包含：

```json
{
  "available_houses": [
    {
      "house_id": "001",
      "house_name": "House_1",
      "aliases": ["house 1", "House_1", "001"]
    },
    {
      "house_id": "003",
      "house_name": "House_3",
      "aliases": ["house 3", "House_3", "003"]
    }
  ]
}
```

如果用户输入无法匹配到 house，应返回：

```json
{
  "status": "needs_user_review",
  "unmatched_targets": ["house 13"],
  "reason": "house 13 is not present in the current house registry."
}
```

---

## 4. 执行状态机

多建筑执行建议使用两层状态机。

### 4.1 Plan-level 状态

```text
IDLE
-> PLAN_ANALYZED
-> EXECUTING
-> COMPLETED
```

异常分支：

```text
EXECUTING
-> PAUSED_BY_USER
-> NEEDS_REVIEW
-> ABORTED
```

### 4.2 House-level 状态

每个目标 house 的状态：

```text
PENDING
-> ACTIVE
-> ENTRY_SEARCHING
-> ENTRY_REACHED
-> DONE
```

无入口完成分支：

```text
ENTRY_SEARCHING
-> NO_ENTRY_CONFIRMED
-> DONE
```

异常分支：

```text
ENTRY_SEARCHING
-> NEEDS_REVIEW
```

### 4.3 自动切换逻辑

当当前 house 满足完成条件：

```text
if current_house.finish_type in {
    target_entry_reached,
    no_entry_after_full_coverage
}:
    mark current house done
    select next pending house
    set target_house_id to next house
    reset per-house local decision counter
    continue LLM control
```

如果没有下一个 house：

```text
mark plan completed
stop LLM control
```

---

## 5. LLM 控制策略修正

### 5.1 新的动作优先级

当目标入口已可见时：

```text
if target entry visible and distance_cm > 300 and front path is clear:
    prefer forward
```

当入口已接近：

```text
if target entry distance_cm <= 300:
    finish current house
```

当入口可见但确实被遮挡：

```text
if target entry visible and severe obstacle in front:
    detour left/right or backoff
```

当只看到 window：

```text
ignore window and continue search
```

当看到非目标门：

```text
do not approach; reorient to current target house
```

### 5.2 防左右振荡规则

需要加入一个简单 anti-oscillation 规则：

如果最近 4 次动作类似：

```text
left, left, right, right
```

或：

```text
right, right, left, left
```

说明横向绕行没有带来新信息，此时禁止继续同样摆动，优先：

- forward 靠近
- backward 拉开视角
- yaw 到新 sector
- stop and needs_review

### 5.3 q/e 转向规则

`q/e` 仍保持单步动作。

原因：

- 转向会显著改变画面。
- 需要立即重新截图分析。
- 不应让 LLM 在旧图像上连续转多次。

执行规则：

```text
if action in {q, e}:
    repeat = 1
    execute yaw
    capture immediately
    use new capture for next decision
```

---

## 6. 数据保存结构

建议保留两套数据目录。

### 6.1 原始感知与 memory capture

```text
captures_remote/
  memory_collection_sessions/
    memory_episode_xxx/
      memory_fusion_captures/
        memory_capture_xxx/
          labeling/
            sample_metadata.json
            temporal_context.json
            fusion_result.json
            yolo_result.json
            depth_result.json
            entry_search_memory_snapshot_before.json
            entry_search_memory_snapshot_after.json
            fusion_overlay.png
```

### 6.2 LLM 控制与任务计划文件

```text
captures_remote/
  llm_control_sessions/
    memory_episode_xxx/
      task_plan.json
      task_plan_response.json
      execution_trace.json
      decision_0001_memory_capture_xxx/
        llm_control_prompt.json
        llm_control_response.json
        llm_control_decision.json
```

新增 `execution_trace.json` 用于记录多建筑任务执行过程：

```json
{
  "plan_id": "llm_task_plan_20260428_123000",
  "episode_id": "memory_episode_20260428_123000_search_house_1_then_3",
  "current_target_index": 1,
  "events": [
    {
      "event_type": "target_started",
      "house_id": "001",
      "step_index": 0
    },
    {
      "event_type": "target_finished",
      "house_id": "001",
      "finish_type": "target_entry_reached",
      "entry_distance_cm": 286.5,
      "step_index": 18
    },
    {
      "event_type": "target_started",
      "house_id": "003",
      "step_index": 18
    }
  ]
}
```

---

## 7. 面板功能设计

### 7.1 新增按钮

在 `Memory Collection` 或 `LLM Control Pilot` 区域新增：

```text
Analyze LLM Task
```

打开窗口：

```text
LLM Task Planner
```

窗口包含：

- Task Text
- API Base URL
- API Key
- Model
- Analyze Task
- Apply Plan
- Current Plan Preview

### 7.2 与已有 LLM Control 的关系

`Analyze LLM Task` 只负责生成目标队列，不直接控制无人机。

`Start LLM Control` 负责执行当前 plan：

1. 读取 `task_plan.json`
2. 设置第一个 target house
3. 进入 LLM control loop
4. 达到当前 house 完成条件后自动切换下一个 target house
5. 所有目标完成后自动停止

### 7.3 手动覆盖

操作员必须可以随时：

- Stop LLM Control
- 手动切换 target house
- 手动标记当前 house finished
- 手动标记 needs_review

---

## 8. Prompt 更新要点

### 8.1 Task planner prompt

任务解析 LLM 的职责：

- 解析自然语言任务。
- 匹配合法 house id。
- 生成 ordered target list。
- 不输出低层动作。
- 不创造不存在的 house。

关键规则：

```text
You are a UAV task planner, not the low-level controller.
Parse the user's natural language instruction into an ordered list of house search targets.
Only use house ids from the provided house registry.
Return strict JSON only.
```

### 8.2 Control prompt

控制 LLM 的职责：

- 当前只控制一个 active target house。
- 判断是否应继续搜索、靠近入口、绕行、转向或完成当前 house。
- 不能自行跳到下一个 house，必须通过完成条件触发 plan executor 切换。

新增关键规则：

```text
If a target-house door/open door is visible and associated with the current target house,
and the UAV is farther than 3 meters from the entry,
prefer approaching forward when the front path is not severely blocked.

cross_ready=false means the UAV should not enter or cross the doorway yet.
It does not mean the UAV cannot approach the doorway.

The current house search is finished when the UAV is within 3 meters of a reliable
target-house entry candidate.
```

---

## 9. 后续编码顺序

建议分四步实现。

### Step 1：任务计划文档化与本地 JSON

新增 task planner 窗口，只生成：

- `task_plan.json`
- `task_plan_response.json`

暂时不自动控制。

### Step 2：plan executor 接入 target house 切换

让 `Start LLM Control` 能读取 task plan，并在开始时自动设置第一个目标 house。

### Step 3：search finished 判定

在每轮 capture 后检查：

- found-entry completion
- no-entry completion
- needs-review

满足完成后写入 `execution_trace.json`，并切换到下一个 house。

### Step 4：更新 control prompt 和 anti-oscillation

修正 LLM prompt：

- `cross_ready=0` 不阻止 approach。
- 看到可靠目标入口且距离大于 3m 时优先 forward。
- 左右摆动多次后禁止继续摆动。

---

## 10. 结论

下一版系统应该从“单目标 LLM 控制”升级为“任务计划驱动的多建筑搜索控制”。

核心变化是：

1. 每个 house 有明确完成条件。
2. 靠近目标入口到约 3m 即完成当前 house 搜索。
3. `cross_ready` 只用于判断是否能穿门，不用于阻止向入口靠近。
4. 自然语言任务先解析成 ordered target plan。
5. LLM control loop 只执行当前 active house，完成后由 plan executor 切换下一个目标。

这样可以同时满足数据采集、可解释实验和后续策略训练三方面需求。
