# UAV-Flow 论文实验操作手册

## 1. 文档目的

这份文档用于统一当前论文实验的完整操作流程，覆盖：

- `Heuristic Baseline`
- `Gemini Lite`
- `Gemini Flash`
- `planner-only` 实验
- `planner-driven execute_plan_segment` 实验
- 结果记录与验收
- 常见错误与排查

当前系统的实验主线是：

1. 大语言模型负责高层搜索规划
2. 本地执行器负责低层动作执行
3. 通过 `capture bundle + search log + progress log` 保留实验证据

---

## 2. 当前推荐实验分组

### 2.1 高层规划对比

用于论文中的核心对比：

1. `Heuristic Baseline`
2. `Gemini Lite + llm_only`
3. `Gemini Flash + llm_only`

目标：

- 比较高层搜索规划质量
- 比较 token 消耗
- 比较延迟
- 比较 mission/search 语义一致性

### 2.2 有限自动探索执行

用于验证 LLM plan 是否真的能驱动探索：

1. `Gemini Lite + Execute Plan Segment`
2. `Gemini Flash + Execute Plan Segment`

目标：

- 验证 planner-owned bounded execution
- 观察 segment 是否能在不手动逐步按键的情况下推进搜索
- 观察 stop reason 是否合理

---

## 3. 输出文件位置

实验中主要关注这些输出：

- capture bundle：
  - [captures_remote](/E:/github/UAV-Flow/captures_remote)
- person evidence / search logs：
  - [phase4_search_logs](/E:/github/UAV-Flow/phase4_search_logs)
- takeover logs：
  - [phase4_takeover_logs](/E:/github/UAV-Flow/phase4_takeover_logs)
- 总进度记录：
  - [uav_phase_progress_log.md](/E:/github/UAV-Flow/docs/uav_phase_progress_log.md)

---

## 4. 实验前准备

### 4.1 纯 planner 实验推荐配置

如果当前目标是只比较高层 planner，不让低层自动执行干扰，推荐：

- `uav_control_server.py` 使用：
  - `--reflex_execute_mode manual`
- 可以不启动 `reflex_policy_server.py`
- 或者即使没启动 reflex server，也允许 control server 回退到 `local_heuristic`

这是目前最稳、最适合论文对比的配置。

### 4.2 全链路实验推荐配置

如果当前目标是跑完整低层策略链路，再额外启动：

```powershell
python E:\github\UAV-Flow\UAV-Flow-Eval\reflex_policy_server.py --host 127.0.0.1 --port 5022 --model_artifact E:\github\UAV-Flow\phase3_dataset_export\mlp_reflex_policy_v4.json --policy_name mlp_reflex_policy_v4
```

然后在 `uav_control_server.py` 启动参数中加入：

- `--reflex_policy_url http://127.0.0.1:5022`
- `--reflex_policy_endpoint /reflex_policy`

如果不启动 reflex server，而 control server 又带了这个 URL，就会出现：

- `Reflex policy request failed ... WinError 10061`

这不代表 planner 坏了，只代表低层外部 reflex 没连上。

---

## 5. 启动顺序

### 5.1 启动 planner server

#### Heuristic Baseline

```powershell
python E:\github\UAV-Flow\UAV-Flow-Eval\planner_server.py --host 127.0.0.1 --port 5021 --planner_name external_heuristic_planner --planner_mode heuristic --planner_route_mode heuristic_only --log_level INFO
```

#### Gemini Lite

```powershell
python E:\github\UAV-Flow\UAV-Flow-Eval\planner_server.py --host 127.0.0.1 --port 5021 --planner_name external_llm_planner --planner_mode llm --planner_route_mode llm_only --llm_api_style google_genai_sdk --llm_base_url google-genai-sdk --llm_model gemini-3.1-flash-lite-preview --llm_input_mode text --log_level INFO
```

#### Gemini Flash

```powershell
python E:\github\UAV-Flow\UAV-Flow-Eval\planner_server.py --host 127.0.0.1 --port 5021 --planner_name external_llm_planner --planner_mode llm --planner_route_mode llm_only --llm_api_style google_genai_sdk --llm_base_url google-genai-sdk --llm_model gemini-3-flash-preview --llm_input_mode text --log_level INFO
```

### 5.2 启动 control server

推荐用于论文 planner 实验的纯 planner 版本：

```powershell
python E:\github\UAV-Flow\UAV-Flow-Eval\uav_control_server.py --env_bin_win E:\github\UAV-Flow\UnrealEnv\UE4_ExampleScene_Win\UE4_ExampleScene\Binaries\Win64\UE4_ExampleScene.exe --viewport_mode free --preview_mode first_person --planner_url http://127.0.0.1:5021 --planner_endpoint /plan --planner_auto_mode k_step --planner_interval_steps 5 --reflex_execute_mode manual --fixed_spawn_pose_file E:\github\UAV-Flow\uav_fixed_spawn_pose.json --capture_dir E:\github\UAV-Flow\captures_remote --search_log_dir E:\github\UAV-Flow\phase4_search_logs
```

如果你要启用外部 reflex server，就在上面追加：

- `--reflex_policy_url http://127.0.0.1:5022`
- `--reflex_policy_endpoint /reflex_policy`
- `--reflex_auto_mode on_move`

### 5.3 启动 panel

```powershell
python E:\github\UAV-Flow\UAV-Flow-Eval\uav_control_panel.py --host 127.0.0.1 --port 5020 --timeout_s 20
```

`--timeout_s 20` 是推荐值，原因是：

- LLM 请求可能接近 5 到 10 秒
- panel 如果超时太短，会导致：
  - `WinError 10053`
  - `Client disconnected before JSON response was sent`

---

## 6. Panel 使用流程

### 6.1 Planner Routing 面板

当前支持直接在 panel 中切换：

- `Heuristic Baseline`
- `Gemini Lite`
- `Gemini Flash`
- `Search Hybrid`
- `Anthropic Qwen Next`
- `Anthropic Sonnet`
- `Anthropic Opus`

推荐操作：

1. 选择 `Preset`
2. 对于 Gemini：
   - `API Style = google_genai_sdk`
   - 在 `API Key` 输入框填入 key
3. 点 `Apply Manual`
4. 观察右下角配置摘要

Gemini 正常时应看到：

- `mode=llm`
- `route=llm_only`
- `api=google_genai_sdk`
- `model=...`
- `key_cfg=1`
- `enabled=1`
- `fallback=0`

### 6.2 任务设置

当前论文主任务统一用：

```text
search the house for people
```

操作：

1. 在 `Task Label` 输入上面的任务
2. 点 `Set Task`
3. 然后再请求 planner

如果不先 `Set Task`，系统容易回到：

- `semantic_navigation`
- `search_frontier`
- `scope=local`

这会使实验口径偏离 `person_search`。

---

## 7. 标准实验流程

### 7.1 Heuristic Baseline

1. 启动 heuristic planner server
2. 启动 control server
3. 启动 panel
4. 在 panel 里选择 `Heuristic Baseline`
5. `Task Label = search the house for people`
6. 点 `Set Task`
7. 点 `Request Plan`
8. 轻微移动 2 到 3 步
9. 再点 `Request Plan`
10. 点 `Capture`

验收标准：

- `Plan detail=heuristic`
- `route=heuristic_only`
- `model=-`
- `tokens=0`
- `Mission type=person_search`

### 7.2 Gemini Lite

1. 启动 `Gemini Lite` planner server
2. 启动 control server
3. 启动 panel
4. 选择 `Gemini Lite`
5. 填 `API Key`
6. 点 `Apply Manual`
7. `Task Label = search the house for people`
8. 点 `Set Task`
9. 点 `Request Plan`
10. 轻微移动 2 到 3 步
11. 再点 `Request Plan`
12. 点 `Capture`

验收标准：

- `detail=llm_planner`
- `route=llm_only`
- `model=gemini-3.1-flash-lite-preview`
- `fallback=0`
- `tokens>0`
- `Mission type=person_search`
- `subgoal=search_house`
- `scope=house`

### 7.3 Gemini Flash

流程与 `Gemini Lite` 相同，只是切换为：

- `Preset = Gemini Flash`
- `model = gemini-3-flash-preview`

验收标准：

- `detail=llm_planner`
- `route=llm_only`
- `model=gemini-3-flash-preview`
- `fallback=0`
- `tokens>0`

---

## 8. planner-driven execute_plan_segment 实验

### 8.1 当前功能定位

`Execute Plan Segment` 是当前的第一版 LLM plan 驱动探索执行器。

它不是无限自动搜索，而是：

- 执行一个有步数上限的自动探索段
- 可选先刷新一次 planner
- 然后按当前 plan 连续走若干步
- 在安全或任务事件出现时停止

### 8.2 操作流程

1. 启动 `Gemini Lite` 或 `Gemini Flash`
2. 启动 control server
3. 启动 panel
4. `Task Label = search the house for people`
5. 点 `Set Task`
6. 点 `Execute Plan Segment`

### 8.3 观察指标

重点看 panel 里的 `PlanExec` 行：

- `mode=execute_plan_segment`
- `steps=...`
- `blocked=...`
- `stop=...`

常见 stop reason：

- `budget_exhausted`
- `waypoint_reached`
- `blocked_motion`
- `takeover_active`
- `evidence_state_changed`

### 8.4 当前建议

如果你现在在做论文主表，先把 `Execute Plan Segment` 放在“系统能力验证”或“自主执行补充实验”里。  
主实验仍然建议先比较：

- Heuristic Baseline
- Gemini Lite
- Gemini Flash

---

## 9. 结果记录规范

每次实验建议至少保留：

1. 一张 panel 截图
2. 一个 `capture bundle`
3. 必要时保留 `search_session_*.jsonl`

bundle 文件重点检查这些字段：

- `task_label`
- `mission.mission_type`
- `mission.search_scope`
- `search_runtime.current_search_subgoal`
- `plan.debug.source`
- `plan.debug.api_style`
- `plan.debug.model_name`
- `plan.debug.route_mode`
- `plan.debug.usage.total_token_count`
- `reflex_runtime.source`
- `planner_executor_runtime`

建议将关键样本统一保存到论文实验记录表中：

| 组别 | 模型 | 任务 | fallback | token | latency | mission_type | subgoal | capture |
|---|---|---|---|---:|---:|---|---|---|
| Heuristic | - | search the house for people | 0 | 0 | x ms | person_search | search_house / frontier | bundle path |
| Gemini Lite | gemini-3.1-flash-lite-preview | search the house for people | 0 | x | x ms | person_search | search_house | bundle path |
| Gemini Flash | gemini-3-flash-preview | search the house for people | 0 | x | x ms | person_search | search_house | bundle path |

---

## 10. 常见错误与排查

### 10.1 `WinError 10053`

表现：

- `Client disconnected before JSON response was sent`
- `Client disconnected before binary response was sent`

含义：

- 客户端先断开了，server 回包时发现连接没了
- 常见原因是 panel timeout 太短，或预览请求被刷新掉

处理：

- 用：
  - `python ...\\uav_control_panel.py --timeout_s 20`
- 这是非致命问题，通常不影响无人机当前状态

### 10.2 `WinError 10061`

表现：

- `Reflex policy request failed ... 由于目标计算机积极拒绝，无法连接`

含义：

- 外部 reflex policy server 没启动
- 或 `5022` 没监听

处理：

- 纯 planner 实验：可以忽略，让其回退到 `local_heuristic`
- 全链路实验：启动 `reflex_policy_server.py`

### 10.3 Gemini 返回 `403`

表现：

- `Method doesn't allow unregistered callers`

含义：

- key 没送到请求
- 或仍然走旧 REST 路径

处理：

- 确保 panel 中：
  - `API Style = google_genai_sdk`
  - `API Key` 已填写
  - `key_cfg=1`

### 10.4 Gemini 返回坏 JSON

表现：

- `LLM planner returned invalid JSON`

当前状态：

- 解析器已经做了 JSON 修复与字段兜底
- `Gemini Flash` 这条链已经验证通过

如果仍偶发出现：

- 先重新请求一次
- 优先保留 bundle 和 planner server 日志

---

## 11. 当前推荐论文实验顺序

建议按这个顺序实际跑论文实验：

1. `Heuristic Baseline`
2. `Gemini Lite`
3. `Gemini Flash`
4. `Gemini Lite + Execute Plan Segment`
5. `Gemini Flash + Execute Plan Segment`

主表优先放前 3 组。  
后 2 组更适合放“自主执行验证”。

---

## 12. 当前结论

截至当前版本，系统已经具备：

- 高层 LLM search planner
- 官方 Gemini SDK 接入
- 可切换的 planner routing
- person-search mission/search runtime
- person evidence logging
- bounded planner-driven exploration segment execution
- panel 内统一配置与实验记录能力

这已经足够支撑下一步正式论文实验。
---

## 13. Pure LLM Action Experiments

This project now also supports a denser action-level LLM experiment path.

Use this mode when you want the API to participate before each macro action, rather than only at sparse high-level planning points.

### 13.1 Recommended startup

1. Start `planner_server.py` with `Gemini Lite` or `Gemini Flash`.
2. Start [uav_control_server.py](/E:/github/UAV-Flow/UAV-Flow-Eval/uav_control_server.py) with:
   - `--reflex_execute_mode manual`
3. Start [uav_control_panel.py](/E:/github/UAV-Flow/UAV-Flow-Eval/uav_control_panel.py)
4. In panel:
   - choose `Gemini Lite` or `Gemini Flash`
   - fill `API Key`
   - click `Apply Manual`

### 13.2 Single-step pure LLM action

1. Set:
   - `Task Label = search the house for people`
2. Click:
   - `Set Task`
3. Click:
   - `Request LLM Action`
4. Inspect:
   - `LLMAct`
   - `View LLM Reply`
   - `View LLM Prompt`
5. If the action looks reasonable, click:
   - `Execute LLM Action`

Expected panel signals:
- `LLMAct status=ok` or `executed`
- `source=llm_action`
- `model=gemini-...`
- `tokens > 0`
- `preview=...`

### 13.3 Multi-step pure LLM action segment

This is the main mode for “API decides several consecutive actions”.

Suggested configuration:
- `Seg Steps = 5`
- `Plan Every = 0`

Procedure:
1. Set:
   - `Task Label = search the house for people`
2. Click:
   - `Set Task`
3. Click:
   - `Execute LLM Action Segment`

Meaning:
- each step asks the LLM action endpoint once
- each step executes one bounded macro action
- no extra high-level planner refresh unless `Plan Every > 0`

If you want both:
- high-level replanning
- per-step pure LLM action

then use:
- `Seg Steps = 5`
- `Plan Every = 1`

This mode means:
- before each executed action, the planner can also be refreshed once
- then the action endpoint decides the next concrete macro action

### 13.4 What to record

For each pure LLM action run, keep:
1. one panel screenshot
2. one `capture bundle`
3. when useful, planner/control logs

Important bundle fields:
- `task_label`
- `mission.mission_type`
- `plan.debug.source`
- `plan.debug.model_name`
- `llm_action_runtime.status`
- `llm_action_runtime.suggested_action`
- `llm_action_runtime.confidence`
- `llm_action_runtime.stop_condition`
- `llm_action_runtime.usage.total_token_count`
- `planner_executor_runtime.mode`
- `planner_executor_runtime.executed_steps`
- `planner_executor_runtime.last_stop_reason`

### 13.5 Suggested comparison groups

| Group | Control mode | API participation | Notes |
|---|---|---|---|
| Planner-only | high-level planner only | sparse | current main method |
| Plan Segment | high-level planner + segment executor | medium | bounded autonomous segment |
| LLM Action Segment | pure LLM action segment | dense | API decides each macro action |

Interpretation guidance:
- `Planner-only` is the clean main method
- `Plan Segment` shows planner-driven autonomy
- `LLM Action Segment` is the closest baseline to “LLM directly controls exploration”
