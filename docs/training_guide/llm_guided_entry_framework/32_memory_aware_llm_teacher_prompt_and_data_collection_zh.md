# 32. Memory-Aware LLM Teacher Prompt 与数据采集计划

## 1. 文档目的

当前系统已经具备：

- YOLO/RGB 入口检测
- Depth 可通行性分析
- target-conditioned fusion
- house-centered memory
- candidate entry 多帧证据
- memory session validator

所以下一阶段不应该再只采“单帧图片分类数据”，而应该采集：

**带目标房屋、入口候选、深度几何、历史记忆和 LLM teacher 判断的入口搜索 episode。**

这份文档定义两件事：

1. 下一轮应该怎样采集 memory-aware 数据
2. LLM teacher prompt 应该怎样升级

目标是让后续 representation distillation 学到：

```text
RGB + Depth + YOLO + Memory + Target Context -> target-conditioned entry representation
```

而不是只学一个单帧门窗分类器。

---

## 2. 当前方向确认

当前推荐方向是：

```text
LLM + YOLO/RGB + Depth + Memory -> Teacher Label / Teacher Representation
Teacher Representation -> Student Representation z_entry
z_entry -> 后续策略训练状态
```

其中：

- YOLO/RGB 负责识别 `door / open door / close door / window`
- Depth 负责判断入口是否可通行、是否 blocked、是否 crossing ready
- Memory 负责维护入口是否被多帧确认、属于哪个 house、是否已经搜索过
- LLM 负责高层语义解释、冲突仲裁和 teacher label 生成

也就是说，LLM 不再只是看一张图猜动作，而是读取结构化多模态摘要：

```text
target house context
YOLO detections
Depth evidence
candidate memory
search progress
previous decisions
```

再输出稳定的 teacher 判断。

---

## 3. 下一轮数据采集目标

### 3.1 采集单元

推荐以 episode 为基本单位，而不是孤立 frame。

一个 episode 应对应：

```text
search_house_<id>_<scenario>
```

例如：

```text
search_house_1_front_approach
search_house_1_left_detour
search_house_1_right_detour
search_house_1_window_distractor
search_house_1_blocked_entry
```

每个 episode 内通过 auto capture 采集多个 memory capture。

每个 capture 至少包含：

```text
RGB
Depth
YOLO result
Depth result
Fusion result
Memory snapshot before
Memory snapshot after
Sample metadata
Temporal context
Fusion overlay
```

### 3.2 每个 house 的最小采集组合

对每个目标 house，建议采 5 类 episode：

| 场景 | 目的 |
|---|---|
| `front_approach` | 学习从正面发现并接近目标入口 |
| `left_detour` | 学习入口偏左或需要左侧绕行 |
| `right_detour` | 学习入口偏右或需要右侧绕行 |
| `window_distractor` | 学习过滤窗户和非入口开口 |
| `blocked_entry` | 学习目标入口被遮挡时不要盲目前进 |

每个 episode 建议：

```text
capture_count >= 8
best_entry_observation_count >= 3
source_frames >= 3
bbox_history >= 3
association_evidence non-zero
validator status = PASS
```

### 3.3 优先补采类别

根据前面训练和混淆矩阵分析，优先补：

```text
target_house_entry_approachable
approach_target_entry
target_house_entry_blocked
detour_left_to_target_entry
detour_right_to_target_entry
non_target_house_entry_visible
ignore_non_target_entry
hold
```

其中最关键的是：

- `target_house_entry_approachable`
- `approach_target_entry`
- `target_house_entry_blocked`
- `non_target_house_entry_visible`

这几类直接决定模型是否能区分：

```text
应该接近目标入口
应该继续搜索
应该绕行
应该过滤非目标入口或窗口
```

---

## 4. 采集操作流程

### 4.1 启动服务

推荐启动 server：

```powershell
python E:\github\UAV-Flow\UAV-Flow-Eval\uav_control_server_basic.py `
  --host 127.0.0.1 `
  --port 5020 `
  --env_bin_win E:\github\UAV-Flow\UnrealEnv\UE4_ExampleScene_Win\UE4_ExampleScene\Binaries\Win64\UE4_ExampleScene.exe `
  --viewport_mode free `
  --preview_mode first_person `
  --fixed_spawn_pose_file E:\github\UAV-Flow\uav_fixed_spawn_pose.json `
  --capture_dir E:\github\UAV-Flow\captures_remote
```

推荐启动 panel：

```powershell
python E:\github\UAV-Flow\UAV-Flow-Eval\uav_control_panel_basic.py `
  --host 127.0.0.1 `
  --port 5020 `
  --timeout_s 8 `
  --state_interval_ms 1500 `
  --preview_interval_ms 1500 `
  --depth_interval_ms 1800
```

注意：

如果改过 server 代码，必须重启 server。

否则 panel 可能仍然连接旧进程，新的 memory 字段或 validator 逻辑不会生效。

### 4.2 开始 episode

在 panel 中：

1. 选择 `House ID`
2. 填写 `Task Label`
3. 在 `Memory Collection` 中填写 episode label
4. 点击 `Start Episode`
5. 打开 `Open Memory Window`
6. 选择 auto capture

推荐 auto capture：

```text
Auto Mode = step
Step Interval = 5
```

如果 UAV 移动较慢，也可以：

```text
Auto Mode = time
Time(s) = 3 或 5
```

### 4.3 采集过程中观察

采集时重点看：

```text
Auto Capture: running
episode_id 非空
step_index 增长
last capture time 更新
last capture run dir 更新
```

如果状态栏一直不变，说明 auto capture 没有真正启动。

### 4.4 停止并验证

采完后点击：

```text
Stop Episode
```

然后运行：

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_validate_memory_collection_session.py `
  --session_dir E:\github\UAV-Flow\captures_remote\memory_collection_sessions\<memory_episode_xxx>
```

合格结果应类似：

```text
[memory-validator] status=PASS captures=11 valid_captures=11 target=001
[memory-validator] start=1 stop=1 final_step=81
[memory-validator] search_status=APPROACHING_ENTRY entry_status=entry_found best_entry_id=0
[memory-validator] best=0:open door obs=10 frames=8 bbox=6 assoc=1.0 dist=0.9585 view=0.9242 app=0.9273 geom=0.7533
```

只有 `PASS` session 才建议进入训练集。

---

## 5. LLM Teacher Prompt 升级原则

### 5.1 旧 prompt 的问题

旧 prompt 更偏单帧判断：

```text
看当前 RGB / depth / detection，判断入口状态和动作。
```

它的问题是：

- 不知道目标 house 是哪个
- 不知道某个入口是否已经被多帧确认
- 不知道当前看到的是目标入口还是非目标入口
- 不知道之前是否已经搜索过这个方向
- 容易把 window 当成可进入入口

### 5.2 新 prompt 的目标

新 prompt 应该是：

```text
Memory-aware target-conditioned teacher prompt
```

也就是让 LLM 读取：

- 当前目标 house
- 当前无人机所在或接近的 house
- YOLO 检测
- Depth 可通行性
- candidate entry memory
- best entry evidence
- previous action / subgoal
- search progress

然后输出：

```text
target_conditioned_state
target_conditioned_subgoal
target_conditioned_action_hint
target_candidate_id
entry association decision
reason
```

---

## 6. LLM Teacher 输入字段

### 6.1 Task Context

```json
{
  "target_house_id": "001",
  "current_house_id": "001",
  "episode_id": "memory_episode_20260426_101452_search_house_1",
  "step_index": 79,
  "task": "search target house entrance and approach it safely"
}
```

### 6.2 Pose / Motion Context

```json
{
  "uav_pose": {
    "x": 0.0,
    "y": 0.0,
    "z": 120.0,
    "yaw": 15.0
  },
  "previous_action": "yaw_right",
  "previous_subgoal": "keep_search_target_house"
}
```

### 6.3 YOLO Evidence

```json
{
  "detections": [
    {
      "candidate_id": "0",
      "class_name": "open door",
      "confidence": 0.93,
      "bbox": [123.0, 81.0, 220.0, 260.0]
    },
    {
      "candidate_id": "2",
      "class_name": "window",
      "confidence": 0.81,
      "bbox": [310.0, 90.0, 380.0, 180.0]
    }
  ]
}
```

### 6.4 Depth Evidence

```json
{
  "best_depth_candidate": {
    "candidate_id": "0",
    "entry_distance_cm": 420.0,
    "opening_width_cm": 108.0,
    "traversable": true,
    "crossing_ready": false,
    "front_obstacle": false
  }
}
```

### 6.5 Memory Evidence

```json
{
  "search_status": "APPROACHING_ENTRY",
  "entry_search_status": "entry_found",
  "candidate_entry_count": 2,
  "best_entry_id": "0",
  "candidate_entries": [
    {
      "entry_id": "0",
      "entry_type": "open door",
      "status": "approachable",
      "observation_count": 10,
      "source_frame_count": 8,
      "bbox_history_count": 6,
      "association_confidence": 1.0,
      "association_evidence": {
        "distance_score": 0.9585,
        "view_consistency_score": 0.9242,
        "appearance_score": 0.9273,
        "geometry_score": 0.7533
      },
      "is_best_candidate": true
    },
    {
      "entry_id": "2",
      "entry_type": "window",
      "status": "window_rejected",
      "observation_count": 1,
      "is_best_candidate": false
    }
  ]
}
```

---

## 7. LLM Teacher 输出 JSON

LLM 必须输出严格 JSON：

```json
{
  "target_conditioned_state": "target_house_entry_approachable",
  "target_conditioned_subgoal": "approach_target_entry",
  "target_conditioned_action_hint": "forward",
  "target_candidate_id": "0",
  "entry_association": "target_house_entry",
  "memory_decision": "reuse_confirmed_best_entry",
  "confidence": 0.92,
  "reason": "Candidate 0 is an open door repeatedly observed across frames with strong distance, view, appearance, and geometry evidence. Candidate 2 is a rejected window and should not be approached."
}
```

### 7.1 字段定义

| 字段 | 含义 |
|---|---|
| `target_conditioned_state` | 当前目标房屋入口状态 |
| `target_conditioned_subgoal` | 高层子目标 |
| `target_conditioned_action_hint` | 建议动作 |
| `target_candidate_id` | 被选中的入口候选 id |
| `entry_association` | 当前入口与目标 house 的关系 |
| `memory_decision` | 是否使用历史 best entry、继续观察或拒绝候选 |
| `confidence` | teacher 置信度 |
| `reason` | 简短可解释理由 |

### 7.2 允许的 `target_conditioned_state`

```text
target_house_not_in_view
target_house_entry_visible
target_house_entry_approachable
target_house_entry_blocked
non_target_house_entry_visible
target_house_geometric_opening_needs_confirmation
```

### 7.3 允许的 `target_conditioned_subgoal`

```text
reorient_to_target_house
keep_search_target_house
approach_target_entry
align_target_entry
detour_left_to_target_entry
detour_right_to_target_entry
cross_target_entry
ignore_non_target_entry
backoff_and_reobserve
```

### 7.4 允许的 `target_conditioned_action_hint`

```text
forward
yaw_left
yaw_right
left
right
backward
hold
```

### 7.5 允许的 `entry_association`

```text
target_house_entry
non_target_house_entry
window_or_non_entry
uncertain_entry
no_entry
```

### 7.6 允许的 `memory_decision`

```text
reuse_confirmed_best_entry
update_best_entry
continue_observing
shift_search_sector
reject_window
reject_non_target_entry
no_memory_available
```

---

## 8. Prompt 模板

### 8.1 System Prompt

```text
You are a memory-aware UAV entry-search teacher.

Your job is to produce target-conditioned supervision labels for a lightweight student policy.
You are not directly controlling the UAV.

Use the provided YOLO/RGB evidence, depth traversability evidence, target-house context, and structured memory.

Rules:
1. Do not invent candidate ids. Select target_candidate_id only from provided candidates, or use null.
2. A window must not be selected as a target entry.
3. Prefer a door/open door/close door candidate only if it is associated with the target house.
4. Strong memory evidence includes repeated observations, bbox history, high association confidence, and non-zero distance/view/appearance/geometry evidence.
5. If the target house is not in view, output target_house_not_in_view and reorient_to_target_house.
6. If the best target-house entry is visible but blocked, output target_house_entry_blocked and choose detour/backoff/search, not forward.
7. If the best target-house entry is approachable but not crossing-ready, output approach_target_entry.
8. If the entry is aligned and crossing-ready, output cross_target_entry.
9. If only a non-target entry or window is visible, output non_target_house_entry_visible and ignore_non_target_entry.
10. Return strict JSON only.
```

### 8.2 User Prompt

```text
Task:
Find and approach the entrance of the target house.

Target context:
{target_context_json}

YOLO/RGB evidence:
{yolo_summary_json}

Depth evidence:
{depth_summary_json}

Memory evidence:
{memory_summary_json}

Previous decision context:
{temporal_context_json}

Return JSON with:
- target_conditioned_state
- target_conditioned_subgoal
- target_conditioned_action_hint
- target_candidate_id
- entry_association
- memory_decision
- confidence
- reason
```

---

## 9. 判定规则

### 9.1 选择 `approach_target_entry`

当满足：

```text
best entry 是 door/open door/close door
associated_house_id == target_house_id
observation_count >= 2
association_confidence 高
depth traversable = true
not crossing_ready
```

输出：

```json
{
  "target_conditioned_state": "target_house_entry_approachable",
  "target_conditioned_subgoal": "approach_target_entry",
  "target_conditioned_action_hint": "forward"
}
```

### 9.2 选择 `cross_target_entry`

当满足：

```text
best entry 是目标 house entry
depth traversable = true
crossing_ready = true
center alignment good
```

输出：

```json
{
  "target_conditioned_state": "target_house_entry_approachable",
  "target_conditioned_subgoal": "cross_target_entry",
  "target_conditioned_action_hint": "forward"
}
```

### 9.3 选择 `detour_left/right_to_target_entry`

当满足：

```text
目标 entry 可见
但 front obstacle 或 depth geometry 显示 blocked
entry 已被多次观察
```

输出：

```json
{
  "target_conditioned_state": "target_house_entry_blocked",
  "target_conditioned_subgoal": "detour_right_to_target_entry",
  "target_conditioned_action_hint": "right"
}
```

左右方向根据入口 bbox 中心和目标方位决定。

### 9.4 选择 `ignore_non_target_entry`

当满足：

```text
可见候选是 window
或 door-like candidate 与目标 house 不匹配
```

输出：

```json
{
  "target_conditioned_state": "non_target_house_entry_visible",
  "target_conditioned_subgoal": "ignore_non_target_entry",
  "target_conditioned_action_hint": "hold"
}
```

### 9.5 选择 `keep_search_target_house`

当满足：

```text
目标 house 在视野内
但没有稳定目标入口
或当前 sector 观察还不充分
```

输出：

```json
{
  "target_conditioned_state": "target_house_entry_visible",
  "target_conditioned_subgoal": "keep_search_target_house",
  "target_conditioned_action_hint": "yaw_left"
}
```

### 9.6 选择 `reorient_to_target_house`

当满足：

```text
target_house_in_fov = false
```

输出：

```json
{
  "target_conditioned_state": "target_house_not_in_view",
  "target_conditioned_subgoal": "reorient_to_target_house",
  "target_conditioned_action_hint": "yaw_left"
}
```

---

## 10. 示例

### 10.1 示例 A：目标 open door 已确认，可接近

输入摘要：

```json
{
  "target_house_id": "001",
  "best_entry": {
    "entry_id": "0",
    "entry_type": "open door",
    "status": "approachable",
    "observation_count": 10,
    "source_frame_count": 8,
    "bbox_history_count": 6,
    "association_confidence": 1.0,
    "association_evidence": {
      "distance_score": 0.9585,
      "view_consistency_score": 0.9242,
      "appearance_score": 0.9273,
      "geometry_score": 0.7533
    }
  },
  "depth": {
    "traversable": true,
    "crossing_ready": false
  }
}
```

期望输出：

```json
{
  "target_conditioned_state": "target_house_entry_approachable",
  "target_conditioned_subgoal": "approach_target_entry",
  "target_conditioned_action_hint": "forward",
  "target_candidate_id": "0",
  "entry_association": "target_house_entry",
  "memory_decision": "reuse_confirmed_best_entry",
  "confidence": 0.92,
  "reason": "Candidate 0 is a repeatedly observed open door with strong target-house association and traversable geometry, but it is not crossing-ready yet, so the UAV should approach it."
}
```

### 10.2 示例 B：看到 window，应该忽略

输入摘要：

```json
{
  "target_house_id": "001",
  "best_visible_candidate": {
    "entry_id": "2",
    "entry_type": "window",
    "status": "window_rejected"
  }
}
```

期望输出：

```json
{
  "target_conditioned_state": "non_target_house_entry_visible",
  "target_conditioned_subgoal": "ignore_non_target_entry",
  "target_conditioned_action_hint": "hold",
  "target_candidate_id": null,
  "entry_association": "window_or_non_entry",
  "memory_decision": "reject_window",
  "confidence": 0.9,
  "reason": "The visible candidate is a window and memory marks it as rejected, so it should not be selected as the target entrance."
}
```

### 10.3 示例 C：目标入口 blocked，需要绕行

输入摘要：

```json
{
  "best_entry": {
    "entry_id": "0",
    "entry_type": "open door",
    "status": "blocked_temporary",
    "observation_count": 4
  },
  "depth": {
    "traversable": false,
    "front_obstacle": true
  }
}
```

期望输出：

```json
{
  "target_conditioned_state": "target_house_entry_blocked",
  "target_conditioned_subgoal": "detour_right_to_target_entry",
  "target_conditioned_action_hint": "right",
  "target_candidate_id": "0",
  "entry_association": "target_house_entry",
  "memory_decision": "shift_search_sector",
  "confidence": 0.84,
  "reason": "The target entry is visible and associated with the target house, but depth evidence indicates that the current path is blocked, so the UAV should detour rather than move forward."
}
```

---

## 11. LLM Teacher 数据保存建议

每次调用 LLM 后，建议在对应 capture 的 `labeling/` 目录下保存：

```text
llm_teacher_prompt.json
llm_teacher_response.json
llm_teacher_label.json
```

当前已经实现 prompt builder 与 teacher label validator：

```text
phase2_multimodal_fusion_analysis/memory_aware_llm_teacher_prompt_builder.py
phase2_multimodal_fusion_analysis/run_memory_aware_llm_teacher_prompt_builder.py
phase2_multimodal_fusion_analysis/memory_aware_llm_teacher_batch.py
phase2_multimodal_fusion_analysis/run_memory_aware_llm_teacher_batch.py
phase2_multimodal_fusion_analysis/memory_aware_llm_teacher_label_validator.py
phase2_multimodal_fusion_analysis/run_memory_aware_llm_teacher_label_validator.py
```

单个 capture 生成 prompt：

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_memory_aware_llm_teacher_prompt_builder.py `
  --labeling_dir E:\github\UAV-Flow\captures_remote\memory_collection_sessions\<episode>\memory_fusion_captures\<capture>\labeling `
  --overwrite
```

整个 episode 批量生成 prompt：

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_memory_aware_llm_teacher_prompt_builder.py `
  --session_dir E:\github\UAV-Flow\captures_remote\memory_collection_sessions\<episode> `
  --overwrite
```

生成结果：

```text
labeling/llm_teacher_prompt.json
```

生成 prompt 后，可以先 dry-run 检查哪些样本会访问 LLM：

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_memory_aware_llm_teacher_batch.py `
  --session_dir E:\github\UAV-Flow\captures_remote\memory_collection_sessions\<episode> `
  --model <model_name> `
  --dry_run
```

确认无误后再真正调用 LLM：

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_memory_aware_llm_teacher_batch.py `
  --session_dir E:\github\UAV-Flow\captures_remote\memory_collection_sessions\<episode> `
  --model <model_name>
```

默认行为：

1. 使用 `ANTHROPIC_BASE_URL` 和 `ANTHROPIC_AUTH_TOKEN`
2. 已经存在 `llm_teacher_label_validated.json` 的样本会自动跳过
3. 每个样本会写入 `llm_teacher_response.json`
4. 每个样本会写入 `llm_teacher_label.json`
5. 默认自动调用 validator 生成 `llm_teacher_label_validated.json`

其中：

### `llm_teacher_prompt.json`

保存实际送入 LLM 的结构化输入。

### `llm_teacher_response.json`

保存原始模型返回。

### `llm_teacher_label.json`

保存 LLM 解析后的 teacher label。

通过 validator 后，建议额外生成：

```text
labeling/llm_teacher_label_validated.json
```

这个文件保存规范化标签、错误、警告、fallback label 和验证上下文，更适合作为后续 dataset exporter 的入口。

建议结构：

```json
{
  "valid": true,
  "model_name": "xxx",
  "target_conditioned_state": "target_house_entry_approachable",
  "target_conditioned_subgoal": "approach_target_entry",
  "target_conditioned_action_hint": "forward",
  "target_candidate_id": "0",
  "entry_association": "target_house_entry",
  "memory_decision": "reuse_confirmed_best_entry",
  "confidence": 0.92,
  "reason": "...",
  "source_files": {
    "fusion_result": "fusion_result.json",
    "memory_before": "entry_search_memory_snapshot_before.json",
    "memory_after": "entry_search_memory_snapshot_after.json"
  }
}
```

---

## 12. Teacher 验证规则

LLM 输出后必须做 validator。现在已经实现：

```text
phase2_multimodal_fusion_analysis/memory_aware_llm_teacher_label_validator.py
phase2_multimodal_fusion_analysis/run_memory_aware_llm_teacher_label_validator.py
```

它的作用不是重新替代 LLM，而是把 LLM 输出变成可训练、可追踪、可回退的 teacher label。

至少检查：

1. 输出是否为合法 JSON
2. label 是否在允许集合内
3. `target_candidate_id` 是否存在于候选列表
4. `window` 是否被错误选为 target entry
5. 如果 `target_conditioned_subgoal = approach_target_entry`，best entry 是否 door-like
6. 如果 depth blocked，是否错误输出 `forward`
7. 如果 `target_house_not_in_view`，是否错误输出 `approach_target_entry`
8. 如果 memory best entry 很强，LLM 是否无理由改选其他候选

单个 capture 验证：

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_memory_aware_llm_teacher_label_validator.py `
  --labeling_dir E:\github\UAV-Flow\captures_remote\memory_collection_sessions\<episode>\memory_fusion_captures\<capture>\labeling `
  --write_validated
```

指定 prompt 与 label 验证：

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_memory_aware_llm_teacher_label_validator.py `
  --prompt_json E:\github\UAV-Flow\captures_remote\memory_collection_sessions\<episode>\memory_fusion_captures\<capture>\labeling\llm_teacher_prompt.json `
  --label_json E:\github\UAV-Flow\captures_remote\memory_collection_sessions\<episode>\memory_fusion_captures\<capture>\labeling\llm_teacher_label.json `
  --write_validated
```

整个 episode 批量验证：

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_memory_aware_llm_teacher_label_validator.py `
  --session_dir E:\github\UAV-Flow\captures_remote\memory_collection_sessions\<episode> `
  --write_validated
```

生成结果：

```text
labeling/llm_teacher_label_validated.json
```

如果 LLM 输出不合法，validator 会回退到 rule-based fusion label，并记录：

```json
{
  "valid": false,
  "status": "FAIL",
  "fallback_label": {
    "target_conditioned_state": "...",
    "target_conditioned_subgoal": "...",
    "target_conditioned_action_hint": "...",
    "target_candidate_id": "..."
  },
  "errors": [],
  "warnings": []
}
```

其中 `FAIL` 样本不建议直接进入训练集；`WARN` 样本可以进入人工抽查队列。

### 12.1 接入 distillation dataset exporter

当前 dataset exporter 已支持读取 `llm_teacher_label_validated.json`。

默认推荐使用 `auto` 模式：

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_distillation_dataset_export.py `
  --results_root E:\github\UAV-Flow\captures_remote\memory_collection_sessions `
  --export_name phase2_5_memory_llm_distillation_dataset `
  --llm_teacher_mode auto
```

`auto` 模式含义：

1. 如果样本里存在 `llm_teacher_label_validated.json` 且状态为 `PASS/WARN`，优先使用 LLM validated label 覆盖 `target_conditioned_*` teacher 字段
2. 如果没有 LLM validated label，则继续使用旧的 `teacher_output.json/teacher_validation.json`
3. 如果二者都没有，则该样本不会进入训练集

如果只想导出已经完成 LLM teacher 验证的样本，可以使用：

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_distillation_dataset_export.py `
  --results_root E:\github\UAV-Flow\captures_remote\memory_collection_sessions `
  --export_name phase2_5_memory_llm_distillation_dataset `
  --llm_teacher_mode require
```

导出后的 `train.jsonl/val.jsonl` 会额外包含：

```text
llm_teacher_available
llm_teacher_status
teacher_source_priority
entry_association
memory_decision
llm_teacher_validated_path
```

同时，导出副本中的 `entry_state.json` 也会同步更新 `teacher_targets.target_conditioned_*` 字段，确保训练端读取的是 LLM teacher 后的 target-conditioned 标签。

---

## 13. 与 representation distillation 的关系

LLM teacher 的输出不是最终在线控制器。

它的作用是生成更高质量的监督信号：

```text
RGB / Depth / YOLO / Memory -> Teacher labels
```

然后 student 学：

```text
RGB / Depth / YOLO / Memory -> z_entry -> labels
```

最终部署时可以不在线调用 LLM，而是用蒸馏后的 representation 模型提供状态。

推荐训练目标：

```text
target_conditioned_state
target_conditioned_subgoal
target_conditioned_action_hint
target_candidate_id
entry_association
memory_decision
teacher_confidence
```

---

## 14. 下一步执行顺序

建议按下面顺序推进：

1. 按本文采集更多 `PASS` memory sessions
2. 使用 prompt builder 生成 `llm_teacher_prompt.json`
3. 让 LLM 读取 `YOLO + Depth + Fusion + Memory + Target Context`
4. 保存 `llm_teacher_response.json` 与 `llm_teacher_label.json`
5. 使用 teacher label validator 生成 `llm_teacher_label_validated.json`
6. 将 `PASS/WARN` 的 LLM teacher label 接入 dataset exporter
7. 训练 V5-B memory-aware representation distillation

---

## 15. 当前阶段结论

现在的数据采集目标已经从：

```text
采 RGB/Depth 图片
```

升级为：

```text
采带 memory 的目标房屋入口搜索 episode
```

LLM prompt 也应该从：

```text
单帧视觉判断
```

升级为：

```text
读取 YOLO + Depth + Memory + Target Context 的 teacher prompt
```

一句话总结：

**下一阶段的核心不是让 LLM 替代 YOLO/Depth/Memory，而是让 LLM 基于这些结构化证据生成更稳定的 teacher label，再蒸馏到轻量表示模型中。**
