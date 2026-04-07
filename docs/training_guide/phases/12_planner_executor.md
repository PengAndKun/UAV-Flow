## Planner-Driven 自主探索执行器设计

### 设计目标

将当前系统从"LLM 提议高层搜索意图 + 人类驱动运动"转变为：

```
planner/LLM 拥有稀疏探索意图
    │
    ▼
server 自动执行有界探索段
    │
    ▼
人类主要监督并在需要时接管
```

### 当前系统边界

当前 runtime 已支持：
- mission-aware planner 输出
- archive 检索上下文
- 本地 reflex 推理
- safety gating
- takeover 日志
- person evidence 日志

**尚未实现**的部分：
1. planner 没有持久执行循环——只返回语义搜索意图和候选航点
2. 运动仍然由用户触发——`manual` 模式下用户是运动源，`assist_step` 只追加一步
3. 没有 planner 拥有的 episode 状态机——无"当前自主段"概念
4. 航点进度不会触发自主继续——可以请求新计划但不会自行执行
5. 安全和证据未耦合到自主运行控制器——都存在但无统一协调

### 设计原则

1. **LLM 保持稀疏**：不发出逐帧电机控制，只负责高层搜索语义
2. **执行有界**：第一版永远不能无限运行，执行有限搜索段后重新评估
3. **安全优先于自主**：任何安全阻塞、重复进度失败或接管请求停止段
4. **可观测性必需**：每个自主段必须记录原因、进度、停止原因和证据结果
5. **分阶段推出**：先有界段执行器，再连续自动搜索

### 执行器运行时状态

新增 `planner_executor_runtime` 对象：

```json
{
  "mode": "segment",
  "active": true,
  "state": "running",
  "run_id": "run_001",
  "segment_id": "seg_003",
  "mission_id": "mission_search_house_1",
  "trigger": "panel_button",
  "current_plan_id": "plan_007",
  "current_search_subgoal": "search_room",
  "target_waypoint": [2360.0, 85.0, 225.0],
  "step_budget": 20,
  "steps_executed": 12,
  "blocked_count": 0,
  "replan_count": 1,
  "last_action": "forward",
  "last_progress_cm": 18.5,
  "last_stop_reason": null,
  "last_stop_detail": null,
  "started_at": "2025-03-26T14:30:00",
  "updated_at": "2025-03-26T14:30:45"
}
```

#### 状态定义

| 状态 | 含义 |
|------|------|
| `idle` | 空闲，等待启动 |
| `starting` | 正在初始化段 |
| `running` | 自主执行中 |
| `replanning` | 请求新计划中 |
| `blocked` | 被安全门控阻塞 |
| `waiting_confirmation` | 等待人工确认 |
| `completed` | 段正常完成 |
| `aborted` | 段被中止 |
| `takeover` | 人工接管中 |

### 两阶段推出

#### Stage A：计划段执行器（先做这个）

行为：server 执行一个有界自主段

```
获取或复用当前计划
  │
  ▼
用 reflex/局部启发式跟随当前航点
  │
  ▼
在以下条件之一时停止：
  ├─ 航点到达
  ├─ 步数预算耗尽
  ├─ 无进度阈值超过
  ├─ 高风险门控反复触发
  ├─ 人员证据需要确认
  └─ 人工接管开始
```

**API 设计**：

```
POST /execute_plan_segment
  请求: { task_label, step_budget, refresh_plan, trigger }
  响应: { status, planner_executor_runtime, segment_summary, state }

POST /stop_planner_executor
  停止当前段

GET /planner_executor
  查询执行器状态
```

**面板控制**：
- `Execute Plan Segment` 按钮
- `Stop Executor` 按钮
- 执行器状态行

**为什么先做 Stage A**：比连续自主循环容易调试——运行时间有限、日志确定性、故障分析简单、安全风险低。

#### Stage B：连续自动搜索执行器（Stage A 稳定后再做）

行为：后台自主循环

```
while mission_active:
  1. 确保存在任务
  2. 确保有当前计划
  3. 执行一个低层动作
  4. 检查进度/证据/安全
  5. 需要时重新规划
  6. 在完成、接管或失败时停止
```

执行模式（`planner_execute_mode`）：
- `manual` — 完全手动
- `segment` — 有界段执行
- `auto_search` — 连续自动搜索

### 每步执行策略

执行器永远不向 LLM 请求低层动作。每步：

```
1. 读取 current_plan
2. 将语义目标转换为局部执行目标
3. 请求 reflex_runtime
4. 门控 reflex 动作
5. reflex 不可用时回退到局部航点跟随启发式
6. 应用恰好一步移动
7. 重新测量进度
```

动作源优先级：
1. 接受的 reflex 动作（本地策略有信心时）
2. 局部启发式动作（朝活跃航点）
3. hold / stop / replan

### 重规划与停止条件

#### 重规划触发

| 条件 | 说明 |
|------|------|
| 航点到达 | 当前航点已接近 |
| 区域标记为 observed | 当前区域已扫描 |
| 新建 suspect 区域 | 检测到可疑区域 |
| 证据状态变化 | PEF 证据分变化 |
| N 步无进度 | 连续多步距离未减少 |
| N 步偏航误差高 | 连续多步无法对准 |
| archive 建议更好的 cell | 检索发现优先扩展 cell |

#### 硬停止触发

| 条件 | 说明 |
|------|------|
| 接管激活 | 人工按下接管 |
| 重复安全门控阻塞 | 多次被安全阻止 |
| 重复高风险动作 | 连续多步高风险 |
| planner 不可用且无局部回退 | 无法生成动作 |
| 人员确认 + 任务要求报告 | 无需继续探索 |

### 搜索子目标特定执行规则

| 子目标 | 行为特点 |
|--------|---------|
| `search_house` | 更广的扫描行为，更大步数预算，房间/区域转换时重规划 |
| `search_room` | 较小步数预算，紧贴优先区域 |
| `search_frontier` | 偏向探索和覆盖增益，到达前沿或视野打开时停止 |
| `approach_suspect_region` | 减少步数预算，降速/更紧门控，风险上升时快速重规划 |
| `confirm_suspect_region` | 允许观察导向动作，允许短偏航/垂直扫描，证据状态变化时停止 |

### 安全集成

使用现有安全信号：
- `risk_score`
- reflex 置信度阈值
- shield 触发
- takeover 状态
- blocked action 原因

**新增安全计数器**（在 `planner_executor_runtime` 内）：

| 计数器 | 用途 |
|--------|------|
| `consecutive_blocked_steps` | 连续被阻塞步数 |
| `consecutive_no_progress_steps` | 连续无进度步数 |
| `consecutive_high_risk_steps` | 连续高风险步数 |

任一超过阈值 → 段停止，记录清晰原因。

### 证据集成

执行器必须响应 `person_evidence_runtime`：

```
if suspect 出现:
  → 停止段 或 重规划到 approach_suspect_region

if confirmed_present:
  → 停止探索 → 标记 run 成功

if confirmed_absent:
  → 标记区域为 checked → 继续搜索并重规划
```

这是使循环成为真正搜索导向（而非通用导航）的关键。

### 执行器日志

新增日志流 `phase4_executor_logs`，每段记录：

```json
{
  "mission_id": "mission_search_house_1",
  "task_label": "search the house for people",
  "planner_source": "heuristic",
  "planner_model": null,
  "active_plan_id": "plan_007",
  "active_search_subgoal": "search_room",
  "step_budget": 20,
  "executed_step_count": 18,
  "progress_trace": [
    {"step": 1, "action": "forward", "progress_cm": 19.2},
    {"step": 2, "action": "yaw_right", "progress_cm": 0.0}
  ],
  "blocked_reasons": [],
  "replan_reasons": ["waypoint_reached"],
  "takeover_status": false,
  "evidence_before": {"suspect": false, "confidence": 0.12},
  "evidence_after": {"suspect": true, "confidence": 0.68},
  "final_stop_reason": "replan_suspect_detected"
}
```

### 执行器实现计划

| 步骤 | 内容 | 文件 |
|------|------|------|
| Step 1 | 添加运行时状态（不动） | `runtime_interfaces.py`, `uav_control_server.py` |
| Step 2 | 添加 `POST /execute_plan_segment`（有界同步段，默认 `step_budget=5`） | `uav_control_server.py` |
| Step 3 | 添加面板控件 | `uav_control_panel.py` |
| Step 4 | 记录每个段和停止原因的日志 | `phase4_executor_logs/` |
| Step 5 | Stage A 稳定后添加后台 `auto_search` | `uav_control_server.py` |

### 验收标准

Stage A 成功的标准：
- 任务如 `search the house for people` 可以运行一个有界计划段而无需键盘操作
- 段能清晰说明是因为 success / no progress / safety block / replan / takeover 而停止
- planner source/model 在日志中可见
- 证据状态在段前/后都被捕获
- 人类可以随时停止段

