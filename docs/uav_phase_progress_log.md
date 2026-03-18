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

- `100%`

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

## Phase 3 进行中：archive runtime 与 reflex 执行接口打底

### 阶段目标

在不破坏当前 `主视角 + 深度图 + planner` 稳定链路的前提下，先把 Phase 3 的第一批运行时能力补起来：

- 引入可在线记录的 goal-conditioned archive runtime
- 开始填充 `archive_cell_id / local_policy_action / risk_score`
- 让 `server / panel / capture metadata` 都能观察到 archive 状态
- 为后续 reflex navigator 和 archive 检索打下统一运行时接口

### Phase 3 第一批子计划拆分与达成度

#### 子计划 3.1：定义 archive cell schema 与量化规则

目标：

- 把 archive 从“概念”变成可运行的数据结构
- 固定 cell 的基本组成方式，避免后续 server/panel/capture 各写一套

主要任务：

- 定义 `task_label + semantic_subgoal + quantized pose + depth signature` 的 cell 结构
- 固定 `cell_id` 生成规则
- 固定最近访问列表与基础 transition 统计

当前达成度：

- `100%`

当前状态：

- 已新增 `UAV-Flow-Eval/archive_runtime.py`
- 已完成首版 cell schema、pose bin、depth signature、transition 计数与 recent cells 管理

#### 子计划 3.2：接入 server 侧 archive 注册与状态暴露

目标：

- 让 `uav_control_server.py` 在运行时自动登记当前 cell
- 让 `/state` 和独立调试接口可以直接看到 archive 状态

主要任务：

- 在 observation refresh / move / plan update / task update 时同步 archive
- 在 `/state` 中暴露 `archive`
- 增加独立 `GET /archive` 接口

当前达成度：

- `85%`

当前状态：

- server 已接入 `ArchiveRuntime`
- 已能在 `/state` 中返回 `archive.current_cell_id / cell_count / transition_count / top_cells`
- 已新增 `GET /archive`
- 后续还需要补更明确的 archive retrieval / progress 指标

#### 子计划 3.3：补齐 panel 侧 archive 调试显示

目标：

- 让控制面板在 Phase 3 里不仅能看 planner，也能看 archive 当前状态
- 让后续 archive/reflex 调试不必只盯日志

主要任务：

- 新增 archive 状态栏
- 显示当前 `archive_cell_id`
- 显示访问次数、cell 总数、transition 数量、近期 cell 摘要

当前达成度：

- `80%`

当前状态：

- panel 已新增 `Archive ...` 状态行
- 已显示 `cell / visits / cells / transitions / recent / hint`
- 后续还可以继续加 retrieval candidates 与 archive 命中来源说明

#### 子计划 3.4：加入基础 risk 估计与 reflex 占位接口

目标：

- 在真正训练 reflex navigator 之前，先把运行时需要的风险位和低层动作位打通
- 给后续 safety / local policy 留出统一接口

主要任务：

- 基于深度图中心区域生成 heuristic `risk_score`
- 更新 `runtime_debug.risk_score`
- 维持 `runtime_debug.local_policy_action`
- 保留 `shield_triggered` 作为后续安全头占位

当前达成度：

- `70%`

当前状态：

- `local_policy_action` 已在移动时稳定更新
- 已新增基于深度中心区域的 heuristic `risk_score`
- 已开始更新 `shield_triggered`
- 当前仍是 debug 级 heuristic，尚未真正影响控制决策

#### 子计划 3.5：扩展 capture metadata 为 archive-aware 数据包

目标：

- 让当前手动采集链路开始记录 archive 相关信息
- 为后续离线 archive 构建和 reflex 数据集准备字段

主要任务：

- 在 capture metadata 中加入 `archive`
- 保留 `plan + runtime_debug + archive` 一起写入 bundle

当前达成度：

- `75%`

当前状态：

- capture bundle 已开始写入 `archive`
- 返回给 panel 的 capture 结果中也会带上 `archive`
- 后续还需要补“成功/失败局部轨迹标签”等更适合训练的数据字段

### Phase 3 第二批子计划拆分与达成度

#### 子计划 3.6：补齐 archive retrieval 输出与命中状态

目标：

- 不只是“记录 cell”，还要能给出当前最相关的历史 cell 候选
- 为后续 planner / reflex 共享 archive 检索结果打基础

主要任务：

- 为 retrieval candidate 增加显式 score
- 在 archive state 中暴露 `active_retrieval`
- 保持 `planner context / state / panel` 使用一致的 retrieval 结果

当前达成度：

- `75%`

当前状态：

- archive runtime 已新增 `retrieval_score`
- 已开始暴露 `active_retrieval`
- 后续还需要补更细的命中原因、成功率和轨迹片段摘要

#### 子计划 3.7：建立 reflex runtime interface

目标：

- 在真实 reflex navigator 训练前，先固定运行时接口
- 让 server 能给出“当前建议动作 / 航点误差 / 风险 / retrieval 命中”的统一状态

主要任务：

- 新增 `reflex_runtime` 结构
- 计算 `waypoint_distance / yaw_error / vertical_error / progress`
- 给出 heuristic `suggested_action`
- 暴露独立 `GET /reflex` 接口

当前达成度：

- `80%`

当前状态：

- server 已新增 `reflex_runtime`
- 已开始根据 waypoint 和 risk 生成 heuristic `suggested_action`
- 已新增 `GET /reflex`
- 当前仍是 heuristic stub，尚未接入真实蒸馏策略

#### 子计划 3.8：panel 侧补齐 retrieval / reflex 可视化

目标：

- 让控制面板能直接看见 archive 命中与 reflex 建议动作
- 降低后续 Phase 3 联调时只靠日志排查的成本

主要任务：

- 新增 reflex 状态栏
- 显示建议动作、航点距离、yaw 误差、progress
- 显示 retrieval 命中的 cell 摘要

当前达成度：

- `75%`

当前状态：

- panel 已新增 `Reflex ...` 状态行
- 已开始显示 `suggested / wp_dist / yaw_err / progress / retrieval`
- 后续可以再补动作来源、命中 score 和 shield 触发提示

### Phase 3 当前轮已落地代码

- `UAV-Flow-Eval/archive_runtime.py`
  - 新增 Phase 3 archive runtime 骨架
- `UAV-Flow-Eval/uav_control_server.py`
  - 接入 archive runtime
  - 新增 `GET /archive`
  - 在 `/state` 与 `capture` 中写入 archive 状态
  - 加入基于深度图的 heuristic `risk_score`
- `UAV-Flow-Eval/uav_control_panel.py`
  - 新增 archive 调试显示行
  - 新增 reflex 调试显示行

### Phase 3 第二批补充代码

- `UAV-Flow-Eval/archive_runtime.py`
  - retrieval candidate 已开始附带 `retrieval_score`
  - archive state 已开始暴露 `active_retrieval`
- `UAV-Flow-Eval/uav_control_server.py`
  - 已新增 `reflex_runtime`
  - 已新增 `GET /reflex`
  - planner request context 已开始带 `archive + reflex_runtime`
- `UAV-Flow-Eval/uav_control_panel.py`
  - 已显示 retrieval 命中与 reflex 建议动作摘要

### Phase 3 第三批子计划拆分与达成度

#### 子计划 3.9：建立 external reflex policy 协议

目标：

- 让 reflex navigator 像 planner 一样，可以被独立服务替换
- 避免后续接轻量策略模型时再改主控制链路

主要任务：

- 定义 `phase3.reflex_request.v1`
- 定义标准化 `reflex_runtime` 输出
- 保证 external policy 与 local heuristic 使用同一 schema

当前达成度：

- `85%`

当前状态：

- 已在 `runtime_interfaces.py` 中新增 reflex request / runtime payload helper
- 已新增 `UAV-Flow-Eval/reflex_policy_server.py`
- 当前 external reflex service 还是 heuristic stub，后续可直接替换成模型版服务

#### 子计划 3.10：在 server 中接入 external reflex policy + fallback

目标：

- 让 `uav_control_server.py` 可以优先请求外部 local policy
- 外部服务异常时回退到本地 heuristic reflex，不影响实验链路

主要任务：

- 新增 `--reflex_policy_url / --reflex_policy_endpoint`
- 新增 `POST /request_reflex`
- 支持 `reflex_auto_mode=on_move`
- 保持 external 结果优先，本地 heuristic 只兜底

当前达成度：

- `80%`

当前状态：

- server 已支持 external reflex policy URL
- 已新增 `POST /request_reflex` 和 `GET /reflex`
- 已支持 `reflex_auto_mode=on_move`
- 已处理 “state refresh 覆盖 external reflex” 的运行时问题

#### 子计划 3.11：panel 侧补齐 external reflex 调试入口

目标：

- 让面板能直接验证 external reflex 是否工作
- 降低后续联调 local policy 服务的摩擦

主要任务：

- 新增 `Request Reflex` 入口
- 显示 `policy / source / latency / should_execute`
- 在 UI 中保留当前 heuristic 与 external 结果的统一显示方式

当前达成度：

- `75%`

当前状态：

- panel 已新增 `Request Reflex`
- 已显示 `policy / source / lat / exec`
- 后续还可以继续补 policy error 状态与最近一次请求结果摘要

### Phase 3 第四批子计划拆分与达成度

#### 子计划 3.12：扩展 capture 为 reflex 训练样本

目标：

- 让当前 capture bundle 不只是“调试快照”，而开始具备训练样本的结构
- 让后续 reflex dataset 构建先复用现有手动链路

主要任务：

- 固定 `phase3.capture_bundle.v1`
- 在 bundle 中新增 `reflex_sample`
- 将 waypoint / risk / retrieval / suggested_action 一起写入

当前达成度：

- `80%`

当前状态：

- capture bundle 已新增 `dataset_schema_version`
- 已新增 `reflex_sample`
- 已用真实样本 `capture_20260318_151608_bundle.json` 验证：
  - `dataset_schema_version=phase3.capture_bundle.v1`
  - 已写入 `archive / reflex_runtime / reflex_sample`
  - `reflex_sample` 中已包含非默认 `suggested_action / retrieval_cell_id / waypoint_distance_cm`
- 后续还需要补更明确的成功/失败标签与 episode-level 统计

#### 子计划 3.13：新增离线 reflex replay / summary 脚本

目标：

- 在不连接 Unreal 的前提下，也能检查当前采集数据是否适合 reflex 学习
- 为后续数据集分析和回放验证提供轻量入口

主要任务：

- 读取 `*_bundle.json`
- 输出 chronological replay lines
- 汇总 suggested action / risk / waypoint distance / retrieval 命中统计
- 支持按 task 过滤和导出 JSON

当前达成度：

- `100%`

当前状态：

- 已新增 `UAV-Flow-Eval/reflex_replay.py`
- 已支持 bundle 目录扫描、回放摘要、统计导出
- 已用真实采样目录 `captures_remote` 验证：
  - `reflex_replay.py` 能正确读出 2026-03-18 新样本
  - 回放结果已出现非默认 `suggested_action=yaw_right`
  - 回放结果已出现非默认 `retrieval_cell_id`
- 当前为离线汇总版，后续可继续扩成 episode replay / visualization

#### 子计划 3.14：补强 external reflex 调试入口

目标：

- 让 external reflex service 的联调链路更完整
- 让 panel 可直接验证 external/local heuristic 的切换结果

主要任务：

- `Request Reflex` 手动触发
- UI 中显示 `source / policy / latency / should_execute`
- 保持自动 `on_move` 和手动请求共存

当前达成度：

- `100%`

当前状态：

- panel 已支持 `Request Reflex`
- reflex 状态栏已显示 `source / policy / latency / exec`
- server 已保留 external 优先、本地 heuristic fallback 的运行策略
- 已通过真实联调验证：
  - `Auto reflex policy updated ... source=external`
  - capture 时 external reflex 状态可正确写入 bundle

### Phase 3 第三批补充代码

- `UAV-Flow-Eval/runtime_interfaces.py`
  - 已新增 `build_reflex_request`
  - 已新增 `coerce_reflex_runtime_payload`
- `UAV-Flow-Eval/reflex_policy_server.py`
  - 已新增独立 external reflex policy stub 服务
- `UAV-Flow-Eval/uav_control_server.py`
  - 已支持 external reflex policy + fallback
  - 已新增 `POST /request_reflex`
  - 已支持 `reflex_auto_mode=on_move`
- `UAV-Flow-Eval/uav_control_panel.py`
  - 已新增 `Request Reflex` 手动入口
  - 已显示 `source / policy / latency / exec`

### Phase 3 第四批补充代码

- `UAV-Flow-Eval/runtime_interfaces.py`
  - 已新增 `build_reflex_sample`
- `UAV-Flow-Eval/uav_control_server.py`
  - capture bundle 已新增 `dataset_schema_version`
  - capture bundle 已新增 `reflex_sample`
- `UAV-Flow-Eval/reflex_replay.py`
  - 已新增离线 replay / summary 脚本
- `UAV-Flow-Eval/uav_control_panel.py`
  - 已补强 external reflex 的手动触发与状态显示

### Phase 3 第五批子计划拆分与达成度

#### 子计划 3.15：建立 episode 级数据组织与 manifest 导出

目标：

- 让当前 capture 样本不再只是零散文件，而能按任务和时间自动整理成 episode
- 为后续训练、回放和误差分析提供稳定的 episode 入口

主要任务：

- 按 `task_label + 时间间隔` 自动分组 episode
- 为每个 episode 生成 manifest
- 输出全局 `episode_index.json`

当前达成度：

- `100%`

当前状态：

- 已新增 `UAV-Flow-Eval/reflex_dataset_builder.py`
- 已支持从 `captures_remote` 自动生成：
  - `phase3_dataset_export/episode_index.json`
  - `phase3_dataset_export/episodes/<episode_id>/episode_manifest.json`
- 已用真实样本验证当前可正确生成 `episode_0001_move_right_3_meters`

#### 子计划 3.16：导出训练友好的 reflex dataset JSONL

目标：

- 让现有 capture bundle 可直接导出成训练/离线分析更容易消费的平铺样本格式
- 减少后续 local policy 训练前的数据清洗成本

主要任务：

- 从 bundle 中提取 `pose / waypoint / retrieval / risk / action`
- 生成统一 `phase3.dataset_sample.v1`
- 输出全局 `reflex_dataset.jsonl`

当前达成度：

- `100%`

当前状态：

- `reflex_dataset_builder.py` 已支持导出 `phase3.dataset_sample.v1`
- 已生成：
  - `phase3_dataset_export/reflex_dataset.jsonl`
- 当前样本中已包含：
  - `executed_action`
  - `suggested_action`
  - `retrieval_cell_id`
  - `waypoint_distance_cm`
  - `risk_score`
- 后续可继续扩展为图像拷贝、分片存储或 episode-level tensor 导出

#### 子计划 3.17：新增 retrieval 质量统计报告

目标：

- 用结构化指标判断 archive retrieval 是否真的提供了目标相关的帮助
- 为后续从 heuristic stub 切到真实 local policy 前提供可比指标

主要任务：

- 统计 retrieval hit rate
- 统计 same-task / same-subgoal 命中率
- 统计 retrieval score / visit count / risk 对比
- 支持按 task 过滤导出 JSON

当前达成度：

- `100%`

当前状态：

- 已新增 `UAV-Flow-Eval/retrieval_quality_report.py`
- 已用真实样本验证输出：
  - `retrieval_hit_rate=1.0`
  - `same_task_hit_rate=1.0`
  - `same_subgoal_hit_rate=1.0`
  - `avg_retrieval_score=4.29175`

### Phase 3 第六批子计划拆分与达成度

#### 子计划 3.18：定义可训练 local policy artifact 格式

目标：

- 让当前 external reflex service 不再只能跑 heuristic stub
- 先建立一个稳定的“训练产物 -> 服务加载 -> runtime 推理”接口

主要任务：

- 固定 `phase3.reflex_policy_artifact.v1`
- 定义 feature names / means / stds / action prototypes
- 保证 artifact 能被独立服务直接加载

当前达成度：

- `100%`

当前状态：

- 已新增 `UAV-Flow-Eval/reflex_policy_model.py`
- 已固定 artifact schema：
  - `feature_names`
  - `feature_means`
  - `feature_stds`
  - `actions.<action>.prototype`
- 已生成真实 artifact：
  - `phase3_dataset_export/prototype_reflex_policy.json`

#### 子计划 3.19：新增 local policy 训练脚本骨架

目标：

- 让当前 Phase 3 JSONL 数据可以直接产出一个可加载的 local policy 模型文件
- 先跑通训练闭环，再逐步替换成真实神经网络训练

主要任务：

- 读取 `reflex_dataset.jsonl`
- 提取固定 feature 向量
- 训练 prototype-based local policy
- 输出 artifact JSON

当前达成度：

- `100%`

当前状态：

- 已新增 `UAV-Flow-Eval/train_reflex_policy.py`
- 已用真实数据验证训练输出：
  - `phase3_dataset_export/prototype_reflex_policy.json`
- 当前训练器为 prototype baseline，后续可替换为 MLP / Transformer / diffusion local policy

#### 子计划 3.20：让 reflex policy server 支持加载真实 artifact

目标：

- 让 `reflex_policy_server.py` 既能跑 heuristic stub，也能跑训练得到的 local policy artifact
- 保持主控制链路不变，只替换 policy server 即可

主要任务：

- 新增 `--model_artifact`
- 优先走 artifact 推理，保留 heuristic fallback
- 在 `/health` 和 `/schema` 中暴露当前 policy mode

当前达成度：

- `100%`

当前状态：

- `UAV-Flow-Eval/reflex_policy_server.py` 已支持 `--model_artifact`
- 已支持 `prototype_model` 模式
- 已用本地 smoke test 验证：
  - artifact 可被成功加载
  - runtime 输出可返回 `suggested_action=yaw_right`
- 已完成主链联调验证：
  - `uav_control_server.py` 启动时已出现 `Startup reflex policy synced ... source=external_model`
  - panel 中已显示：
    - `mode=prototype_policy`
    - `policy=prototype_reflex_policy`
    - `source=external_model`

### 当前结论

- Phase 3 第一批运行时骨架已开始落地
- 当前系统已经从“planner-only debug”升级为“planner + archive runtime + risk debug + reflex service-ready”基础形态
- 当前已经开始具备“可采集、可回放、可替换 local policy”的工程条件
- 当前已经开始具备“可组织成 episode、可导出训练 JSONL、可评估 retrieval 质量”的离线数据基础
- 当前已经开始具备“可训练 baseline local policy artifact、可由 policy server 直接加载”的最小训练闭环
- 还没有进入真正的神经网络 reflex navigator 训练与执行闭环

### Phase 3 当前轮验收结果（第四批）

验收结论：

- `通过`

本轮验收范围：

- 子计划 `3.12`：capture bundle 是否已经具备 reflex 训练样本结构
- 子计划 `3.13`：离线 replay / summary 是否已经能正确读取真实样本
- 子计划 `3.14`：external reflex 调试入口与运行时状态是否已经打通

本轮验收证据：

- 真实样本：
  - `captures_remote/capture_20260318_151608_bundle.json`
  - `captures_remote/capture_20260318_151608_camera_info.json`
- bundle 中已存在：
  - `dataset_schema_version=phase3.capture_bundle.v1`
  - `archive`
  - `reflex_runtime`
  - `reflex_sample`
- `reflex_sample` 中已出现非默认运行时字段：
  - `suggested_action=yaw_right`
  - `retrieval_cell_id=move_right_3_meters__turn_right__x5_y1_z0_yaw4__fd1`
  - `waypoint_distance_cm=205.625`
- `reflex_replay.py --capture_dir captures_remote --limit 5` 已能正确读出新样本：
  - `suggest=yaw_right`
  - `retrieval=move_right_3_meters__turn_right__x5_y1_z0_yaw4__fd1`
- 运行日志已出现：
  - `Auto reflex policy updated at step=... source=external suggested=yaw_right`
  - `Captured RGB/depth bundle: ...capture_20260318_151608_bundle.json`

当前判断：

- Phase 3 第四批的数据采集、离线回放和 external reflex 调试链路已经验通
- 可以进入 Phase 3 下一批：episode 级数据组织、retrieval 质量统计、真实 local policy 替换

### Phase 3 当前轮验收结果（第六批）

验收结论：

- `通过`

本轮验收范围：

- 子计划 `3.18`：可训练 local policy artifact 格式是否已经固定
- 子计划 `3.19`：训练脚本是否已经能从现有 JSONL 产出可加载 artifact
- 子计划 `3.20`：模型版 reflex policy 是否已经接入主控制链路

本轮验收证据：

- 训练输出：
  - `phase3_dataset_export/prototype_reflex_policy.json`
- 训练命令已成功运行：
  - `train_reflex_policy.py --dataset_jsonl ... --output_path ...`
- reflex 服务已成功加载 artifact：
  - `Loaded reflex model artifact ... policy_name=prototype_reflex_policy samples=1`
- 主服务启动日志已出现：
  - `Startup reflex policy synced policy=prototype_reflex_policy source=external_model suggested=yaw_right`
- panel 已显示模型版 reflex 状态：
  - `mode=prototype_policy`
  - `policy=prototype_reflex_policy`
  - `source=external_model`

当前判断：

- Phase 3 第六批已经完成从“离线数据集”到“可加载的 local policy artifact”再到“主链运行时接入”的最小闭环
- 当前 prototype baseline 已经不再只是独立脚本，而是能通过 `reflex_policy_server.py` 驱动 `uav_control_server.py`

### 下一步建议

- 下一批优先做更丰富的训练样本采集和多任务/多动作数据覆盖
- 再把当前 prototype baseline 替换成真实轻量 local policy 网络
- 然后进入 Phase 3 的训练、回放和评估闭环

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
