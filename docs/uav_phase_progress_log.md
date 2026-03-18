# UAV 阶段进展记录

本文档用于记录每个阶段已经完成的工作、当前状态、下一阶段目标。

- 当前更新时间：2026-03-17
- 记录方式：每完成一个阶段，就在本文档中追加一节
- 关联文档：
  - `docs/uav_phase_upgrade_roadmap.md`
  - `docs/hierarchical_multimodal_uav_plan.md`

---

## Phase 1 已完成：单机控制与深度可观测基线

### 阶段目标

建立一个可重复使用的 UAV 控制底座，先把“环境启动 + 无人机控制 + 主视角 + 深度图 + 数据保存”这条链路打通，并收成适合提交到仓库的版本。

### 本阶段完成内容

#### 1. `UAV-Flow-Eval/uav_control_server.py`

已完成：

- 保留 Unreal 环境启动能力
- 保留单机 UAV 控制能力
- 保留主视角 RGB 画面接口
- 保留深度图接口和深度原图输出
- 保留状态接口
- 保留截图与采集接口
- 保留 `task label` 和 `plan` 基础接口
- 保留单 UAV 可见、额外内部 agent 隐藏逻辑

当前主要接口：

- `GET /state`
- `GET /frame`
- `GET /depth_frame`
- `GET /depth_raw.png`
- `GET /camera_info`
- `GET /plan`
- `POST /move_relative`
- `POST /capture`
- `POST /task`
- `POST /plan`
- `POST /request_plan`
- `POST /runtime_debug`
- `POST /shutdown`

#### 2. `UAV-Flow-Eval/uav_control_panel.py`

已完成：

- 保留主视角窗口
- 保留深度图窗口
- 保留键盘控制和按钮控制
- 保留截图按钮
- 保留 `task label`
- 保留 `request plan`
- 删除点云、mapping、额外地图窗口逻辑

当前键位：

- `W/S/A/D`：前后左右
- `R/F`：上升下降
- `Q/E`：偏航旋转
- `C`：截图
- `V`：刷新主视角和深度图
- `P`：请求高层规划

#### 3. 深度链路

已完成：

- 深度图预览渲染
- 深度原始 16-bit PNG 输出
- `camera_info` 生成
- RGB + depth + depth preview + camera_info 的本地保存

#### 4. 上传前清理

已完成：

- 将主控制链路中的 pointcloud / radar / mapping 代码从 `uav_control_server.py` 与 `uav_control_panel.py` 中移除
- 保留深度图作为当前主传感器链路
- 保留最小可运行、可维护、可提交版本

### 本阶段交付物

- 可运行的 UAV 控制服务
- 可运行的远程控制面板
- 主视角 + 深度图双窗口
- 可保存的 RGB/depth 数据包

### 本阶段验收结论

当前 Phase 1 可以视为完成，形成了后续阶段继续扩展的稳定底座。

---

## Phase 2 进行中：稀疏高层规划替代逐步动作预测

### 阶段目标

把系统从“逐步动作控制”升级成“低频高层规划 + 高频底层执行”。

核心原则：

- 大模型或 planner 不再每一步都输出动作
- `uav_control_server.py` 继续只做执行器
- 规划结果以结构化方式表达，而不是连续控制量

### Phase 2 计划完成内容

#### 1. 统一高层规划输出格式

规划器输出不再是：

- `[dx, dy, dz, dyaw]`

而改成：

- `semantic_subgoal`
- `sector_id`
- `candidate_waypoints`
- `planner_confidence`
- `should_replan`

#### 2. 保持 `uav_control_server.py` 为执行器

本阶段要求：

- server 不负责复杂语义推理
- server 只负责：
  - 接收 planner 结果
  - 存储当前 `plan state`
  - 按当前 waypoint/子目标执行
  - 回传状态和调试信息

#### 3. 外部 planner 服务接入

计划新增：

- 一个独立 planner 服务或脚本

作用：

- 输入：RGB、深度摘要、位姿、任务文本
- 输出：结构化高层规划结果

建议优先实现两种模式：

- `heuristic fallback planner`
- `external planner service`

#### 4. 控制面板增加 planner 调试信息

面板中需要稳定显示：

- 当前子目标
- 当前扇区编号
- 置信度
- 当前第一个 waypoint

#### 5. 稀疏调用机制

需要增加：

- 每 `K` 步或满足条件时再调用一次 planner
- 执行过程中默认沿当前规划继续走
- 支持后续扩展为 `replan trigger`

### Phase 2 子计划拆分与达成度

#### 子计划 2.1：定义统一 planner 协议

目标：

- 固定 planner 的请求和响应 schema
- 让 `server / panel / 外部 planner` 三端使用同一套结构

主要任务：

- 定义 planner request 字段
- 定义 planner response 字段
- 固定 waypoint 数据结构
- 固定 `semantic_subgoal / sector_id / confidence / should_replan` 等字段语义

当前达成度：

- `90%`

当前状态：

- 已完成协议初版，后续若引入更强 planner 只需在现有 schema 上扩展

#### 子计划 2.2：实现本地 fallback planner

目标：

- 在没有外部 planner 服务时，系统仍然能输出结构化高层计划
- 保证第二阶段可以先闭环跑起来

主要任务：

- 基于当前位姿生成启发式 waypoint
- 生成默认 `semantic_subgoal`
- 生成默认 `sector_id`
- 生成默认 `planner_confidence`

当前达成度：

- `100%`

当前状态：

- 已完成基础 fallback planner，可在外部 planner 不可用时输出结构化 plan

#### 子计划 2.3：接入独立 external planner 服务

目标：

- 将高层规划从 `uav_control_server.py` 中解耦出去
- 使 server 只承担执行器角色

主要任务：

- 新建独立 planner 脚本或服务
- 接收 RGB、深度摘要、位姿、任务标签
- 返回结构化 `plan`
- 明确 planner 超时与失败回退机制

当前达成度：

- `80%`

当前状态：

- 已新增独立 `planner_server.py`，当前为启发式外部 planner，后续可替换为模型版 planner

#### 子计划 2.4：补齐 server 侧计划执行状态

目标：

- 让 `uav_control_server.py` 能稳定维护当前 plan state
- 为后续 waypoint 执行与 replan 打基础

主要任务：

- 明确 `current_plan` 更新逻辑
- 明确 `request_plan` 触发逻辑
- 在 `/state` 中稳定暴露 planner 状态
- 为后续加入 `plan progress` / `current waypoint index` 预留字段

当前达成度：

- `65%`

当前状态：

- 已在 server 中加入 `current_plan`、`planner_runtime`、`last_plan_request` 等运行时状态

#### 子计划 2.5：补齐 panel 侧 planner 调试显示

目标：

- 让面板能清楚展示当前高层规划结果
- 让用户在调试时直接看到 planner 是否正常工作

主要任务：

- 显示 `semantic_subgoal`
- 显示 `sector_id`
- 显示 `planner_confidence`
- 显示第一个 `candidate_waypoint`
- 显示 planner 请求成功/失败状态

当前达成度：

- `70%`

当前状态：

- 已在 panel 中显示 planner 名称、子目标、扇区、置信度、waypoint、planner 状态与延迟

#### 子计划 2.6：实现稀疏调用与阶段验收闭环

目标：

- 验证“低频高层规划 + 高频底层执行”这条链路可以真正运行

主要任务：

- 定义 planner 调用步长 `K`
- 明确何时沿当前 plan 执行
- 明确何时请求新 plan
- 跑通一轮基础闭环测试
- 记录 planner 调用频率与时延

当前达成度：

- `80%`

当前状态：

- 已支持 `k_step` 自动规划模式，server 可按步数自动触发 planner；当前剩余工作主要是更系统的闭环验证与参数调优

### Phase 2 具体开发清单

#### A. planner 输入输出协议

需要明确：

- 请求字段
  - `task_label`
  - `frame_id`
  - `timestamp`
  - `pose`
  - `depth`
  - `camera_info`
  - `image_b64`
- 响应字段
  - `plan_id`
  - `planner_name`
  - `generated_at`
  - `semantic_subgoal`
  - `sector_id`
  - `candidate_waypoints`
  - `planner_confidence`
  - `should_replan`
  - `debug`

#### B. waypoint 表达

本阶段建议固定使用：

- `x`
- `y`
- `z`
- `yaw`
- `radius`
- `semantic_label`

#### C. server 侧逻辑

本阶段要做：

- 明确 `request_plan` 的触发逻辑
- 明确 fallback planner 和 external planner 的切换逻辑
- 明确 `current_plan` 的更新时机
- 为下一阶段预留 waypoint 执行状态

#### D. panel 侧逻辑

本阶段要做：

- 让 `request plan` 变成稳定调试入口
- 将 `plan` 显示得更清楚
- 让用户能明显看到：
  - 当前计划是什么
  - 当前是否在重规划
  - planner 是否成功响应

### Phase 2 交付物

目标交付：

- 一个结构化 planner 接口
- 一个可独立运行的 planner 服务或脚本
- 一个稳定的 fallback planner
- 控制面板中的 planner 调试显示
- 基于“稀疏规划”的基础闭环

### Phase 2 验收标准

完成标准：

- planner 输出不再是逐步动作，而是结构化计划
- `uav_control_server.py` 不承担高层语义推理
- `uav_control_panel.py` 能稳定显示 planner 状态
- 系统能跑“每 K 步规划一次，其余时间按当前 plan 执行”的闭环

### Phase 2 当前轮验收结果

验收时间：

- `2026-03-18`

验收结论：

- 当前工程版 `Phase 2` 验收通过

本轮验收已确认：

- 外部 planner 模式通过
- fallback planner 模式通过
- `k_step` 自动规划模式通过
- panel 中 planner 状态显示通过
- capture 中 `plan / depth / camera_info / runtime_debug` 写入通过

关键证据摘要：

- 外部 planner 模式下，panel 显示：
  - `planner=external_heuristic_planner`
  - `status=ok`
  - `source=external`
- fallback 模式下，panel 显示：
  - `planner=heuristic_fallback`
  - `status=fallback`
  - `source=local_heuristic`
- 自动规划模式下，控制服务日志出现：
  - `Auto planner triggered at step=196 ...`
  - `Auto planner triggered at step=201 ...`
- panel 中显示：
  - `trigger=step_interval`
  - `auto=k_step`
  - `next=...`

当前建议：

- 可以将当前工程线视为 `Phase 2` 已完成
- 下一步建议进入 `Phase 3`

关联验收文档：

- `docs/phase2_acceptance_report.md`

---

## 后续记录模板

后续每个阶段建议按以下格式追加：

### Phase N 已完成：阶段名称

#### 阶段目标

#### 本阶段完成内容

#### 本阶段交付物

#### 本阶段验收结论

### Phase N+1 进行中：阶段名称

#### 阶段目标

#### 计划完成内容

#### 具体开发清单

#### 交付物

#### 验收标准
