# Phase 2 验收报告

更新时间：2026-03-18

关联文件：

- `UAV-Flow-Eval/uav_control_server.py`
- `UAV-Flow-Eval/uav_control_panel.py`
- `UAV-Flow-Eval/planner_server.py`
- `UAV-Flow-Eval/runtime_interfaces.py`
- `docs/uav_phase_progress_log.md`

---

## 1. 验收目标

本轮 Phase 2 的验收目标是确认以下能力已经跑通：

- 将逐步动作预测升级为结构化高层规划
- 支持外部 planner 服务
- 支持本地 fallback planner
- 支持在 panel 中显示 planner 调试信息
- 支持按 `K` 步自动触发 planner
- 支持将规划结果写入 capture 元数据

本次验收针对的是“工程版 Phase 2 闭环”，不是最终论文版完整 planner 系统。

---

## 2. 验收范围

本次验收覆盖：

- `GET /plan`
- `POST /plan`
- `POST /request_plan`
- `planner_runtime`
- `last_plan_request`
- `heuristic fallback planner`
- `external planner server`
- `k_step` 自动触发模式
- panel 中的 planner 状态显示
- capture 中的 `plan / depth / camera_info / runtime_debug`

---

## 3. 验收环境

运行环境：

- Unreal 环境：`UnrealTrack-SuburbNeighborhood_Day-ContinuousColor-v0`
- 控制服务：`uav_control_server.py`
- 控制面板：`uav_control_panel.py`
- 外部 planner：`planner_server.py`

关键模式：

- `preview_mode = first_person`
- `planner_auto_mode = k_step`
- `planner_interval_steps = 5`

---

## 4. 验收过程与结果

### 4.1 外部 planner 验收

验收方式：

- 启动 `planner_server.py`
- 启动 `uav_control_server.py` 并连接 `planner_url`
- 在 panel 中请求 planner

验收结果：

- 通过

证据：

- panel 中显示：
  - `planner=external_heuristic_planner`
  - `status=ok`
  - `source=external`
  - `subgoal=turn_right` 或 `subgoal=move_forward`
  - `sector=...`
  - `conf=...`
  - `wp=(...)`
- 说明结构化 plan 已由外部 planner 返回并被 UI 正常展示

---

### 4.2 fallback planner 验收

验收方式：

- 关闭 `planner_server.py`
- 在 panel 中再次请求 planner

验收结果：

- 通过

证据：

- panel 中显示：
  - `planner=heuristic_fallback`
  - `status=fallback`
  - `source=local_heuristic`

说明：

- 外部 planner 不可用时，系统能够自动退回本地 fallback planner

---

### 4.3 自动 `k_step` 规划验收

验收方式：

- 打开 `planner_auto_mode = k_step`
- 持续使用 `W` 等键移动无人机
- 观察控制服务日志与 panel 状态

验收结果：

- 通过

日志证据：

- 控制服务日志中出现：
  - `Auto planner triggered at step=196 next_trigger_step=201 planner=external_heuristic_planner subgoal=move_forward`
  - `Auto planner triggered at step=201 next_trigger_step=206 planner=external_heuristic_planner subgoal=move_forward`

UI 证据：

- panel 中显示：
  - `trigger=step_interval`
  - `auto=k_step`
  - `next=121`
  - `next=206`

说明：

- planner 已不再仅靠手动 `Request Plan`
- server 可以按配置的步长自动触发高层规划

---

### 4.4 capture 元数据验收

验收方式：

- 执行 capture
- 检查生成的 `bundle.json`

验收结果：

- 通过

已验证文件：

- [capture_20260317_232502_move right 3 meters_bundle.json](/E:/github/UAV-Flow/captures_remote/capture_20260317_232502_move%20right%203%20meters_bundle.json)

关键字段检查结果：

- `task_label`：存在
- `depth`：存在
- `camera_info`：存在
- `plan`：存在
- `runtime_debug`：存在
- `plan.schema_version = phase2.plan.v1`

说明：

- Phase 2 的高层规划结果已经能落入采集数据，后续可用于分析或训练

---

## 5. 验收结论

结论：

- 当前工程版 Phase 2 验收通过

当前已通过的能力：

- 结构化 planner 协议
- 外部 planner 接入
- fallback planner
- planner 调试显示
- `k_step` 自动规划触发
- capture 元数据写入

当前通过的是：

- Phase 2 的工程实现闭环

尚未包含的是：

- 更强的模型级 planner
- 更复杂的 replan 策略
- 基于 archive 的后续 Phase 3 能力

---

## 6. 建议进入下一步

建议下一步进入 Phase 3，重点转向：

- goal-conditioned archive
- reflex navigator
- risk / shield / runtime debug 扩展

如果仍在 Phase 2 内继续打磨，可优先处理：

- 自动 replan 条件
- 更丰富的 planner 输入摘要
- 更稳定的 waypoint 执行状态
