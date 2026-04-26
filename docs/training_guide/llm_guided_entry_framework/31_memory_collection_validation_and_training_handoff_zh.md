# 31. Memory 采集验证与训练交接说明

## 1. 文档目的

这份文档用于承接前面的 memory-aware collection 工作。

当前阶段的核心结论是：

**结构化记忆采集链路已经打通，并且 candidate entry 已经具备可用于训练的入口级证据。**

也就是说，现在不只是保存 episode 状态，而是已经开始记录：

- 哪个 house 是当前目标
- 哪些入口候选被观察到
- 每个入口候选来自哪些帧
- 每个入口候选的 bbox 历史
- 每个入口与目标 house 的关联证据
- 当前唯一 best entry 是哪一个
- 当前任务建议是继续搜索、绕行、还是 approach target entry

这一步完成后，下一步就可以从“采集功能验证”进入“可训练样本导出和 V5 训练改进”。

---

## 2. 当前已验证样本

目前可以作为标准参考样本的是：

```text
E:\github\UAV-Flow\captures_remote\memory_collection_sessions\memory_episode_20260426_101452_search_house_1
```

该 episode 的最终状态：

```text
episode_id = memory_episode_20260426_101452_search_house_1
step_index = 81
target_house_id = 001
current_house_id = 001
search_status = APPROACHING_ENTRY
entry_search_status = entry_found
candidate_entry_count = 2
best_entry_id = 0
decision_hint = approach_target_entry
```

最终 best entry：

```text
entry_id = 0
entry_type = open door
status = approachable
observation_count = 10
source_frames = 8
bbox_history = 6
association_confidence = 1.0
distance_score = 0.9585
view_consistency_score = 0.9242
appearance_score = 0.9273
geometry_score = 0.7533
is_best_candidate = true
```

同时存在一个被拒绝的 window candidate：

```text
entry_id = 2
entry_type = window
status = window_rejected
is_best_candidate = false
```

这个结果很重要，因为它证明系统已经可以同时维护：

- 正目标入口：`open door`
- 非入口干扰物：`window`
- 唯一 best entry
- 多帧 bbox 历史
- 非零 association evidence

---

## 3. 合格 memory episode 的判定标准

一个 memory collection episode 至少需要满足下面几类条件。

### 3.1 生命周期完整

episode 目录下应有：

```text
entry_search_memory_snapshot_start.json
entry_search_memory_snapshot_stop.json
memory_fusion_captures/
```

如果没有 `entry_search_memory_snapshot_stop.json`，说明 episode 没有被正常停止。

这种样本不是完全不能用，但不建议作为标准训练样本。

### 3.2 capture 链路完整

每个 `memory_fusion_captures/memory_capture_*` 下的 `labeling/` 目录应包含：

```text
sample_metadata.json
temporal_context.json
state.json
yolo_result.json
depth_result.json
fusion_result.json
fusion_overlay.png
entry_search_memory_snapshot_before.json
entry_search_memory_snapshot_after.json
```

如果缺少 `fusion_result.json`，说明本次捕捉没有完成 fusion analyze。

如果缺少 `entry_search_memory_snapshot_after.json`，说明 memory 没有完成更新后的落盘。

### 3.3 target house 对齐

`sample_metadata.json` 中应能看到：

```json
{
  "target_house_id": "001",
  "current_house_id": "001",
  "memory_step_index": 79,
  "capture_source": "auto_step"
}
```

允许前几帧 `current_house_id` 为空，因为 UAV 可能还没进入 house registry 的判定范围。

但稳定观察阶段，`target_house_id` 不应为空。

### 3.4 candidate entry 证据完整

最终 stop snapshot 中，目标 house 的 best entry 应满足：

```text
entry_id 非空
entry_type 是 door / open door / close door 之一
observation_count >= 2
source_frames >= 2
bbox_history >= 2
association_confidence > 0
association_evidence.distance_score > 0
association_evidence.view_consistency_score > 0
association_evidence.appearance_score > 0
is_best_candidate = true
```

其中 `bbox_history` 默认最多保留 6 条，`source_frames` 默认最多保留 8 条。

所以如果 `observation_count` 大于 8，但 `source_frames = 8`，这是正常现象。

### 3.5 best entry 唯一

同一个 house 的所有 `candidate_entries` 中，只允许一个：

```json
"is_best_candidate": true
```

如果出现多个 `true`，说明 best entry 唯一性没有生效。

当前代码已经修复该问题，新的标准样本中：

```text
best_true_count = 1
```

### 3.6 状态演化合理

一个合理的入口搜索 episode 通常会出现类似演化：

```text
target_house_entry_blocked
-> target_house_entry_blocked with keep_search_target_house
-> target_house_entry_approachable
-> approach_target_entry
```

这说明 memory 不只是静态记录，还在支撑搜索过程：

- 早期看到入口但路径不佳
- 中期继续换角度或绕行
- 后期确认可接近入口
- 最终给出 `approach_target_entry`

---

## 4. 当前 memory 字段对训练的意义

### 4.1 `source_frames`

表示这个入口候选在哪些观测帧里被看到过。

训练意义：

- 判断入口是否经过多帧确认
- 区分单帧误检和稳定入口
- 给时序 memory feature 提供基础

### 4.2 `bbox_history`

表示同一入口候选在不同帧中的检测框历史。

训练意义：

- 可用于判断入口是否稳定
- 可用于计算 bbox drift
- 可用于判断 UAV 是否在接近入口
- 可用于后续图像 crop feature 提取

### 4.3 `association_evidence`

当前包含：

```json
{
  "distance_score": 0.9585,
  "view_consistency_score": 0.9242,
  "appearance_score": 0.9273,
  "language_score": 0.0,
  "geometry_score": 0.7533,
  "memory_similarity_score": 0.0
}
```

训练意义：

- `distance_score` 表示入口距离与目标 house 空间关系是否合理
- `view_consistency_score` 表示入口位置与目标 house 视角是否一致
- `appearance_score` 表示 YOLO/RGB 语义置信是否支持入口判断
- `geometry_score` 表示 depth 是否支持可通行入口
- `language_score` 预留给后续 LLM/VLM 语言约束
- `memory_similarity_score` 表示和历史候选是否一致

### 4.4 `is_best_candidate`

表示当前 house 下唯一最优入口候选。

训练意义：

- 可作为 target candidate selection 的监督信号
- 可用于过滤非入口干扰物
- 可用于构建 `target_conditioned_target_candidate_id`

### 4.5 `search_status`

例如：

```text
ENTRY_CANDIDATE_FOUND
ENTRY_ASSOCIATED
APPROACHING_ENTRY
```

训练意义：

- 可作为 high-level search progress feature
- 可帮助模型区分“刚看到入口”和“已经准备接近入口”

---

## 5. 下一步训练数据导出建议

### 5.1 样本级输入

每个训练样本建议从一个 `memory_capture_*` 目录中读取：

```text
labeling/state.json
labeling/yolo_result.json
labeling/depth_result.json
labeling/fusion_result.json
labeling/entry_search_memory_snapshot_before.json
labeling/entry_search_memory_snapshot_after.json
labeling/sample_metadata.json
labeling/temporal_context.json
```

其中：

- `before` 表示决策前 memory
- `fusion_result` 表示本帧多模态融合结果
- `after` 表示决策后 memory

如果要训练“根据当前状态做决策”，优先使用：

```text
state + yolo + depth + memory_before -> fusion labels
```

如果要训练“记忆更新模型”，可以使用：

```text
memory_before + current observation -> memory_after
```

### 5.2 推荐标签

继续沿用 target-conditioned labels：

```text
target_conditioned_state
target_conditioned_subgoal
target_conditioned_action_hint
target_conditioned_target_candidate_id
```

同时新增或强化 memory labels：

```text
best_entry_id
best_entry_type
best_entry_status
best_entry_association_confidence
best_entry_target_match_score
best_entry_observation_count
best_entry_source_frame_count
best_entry_bbox_history_count
best_entry_is_approachable
candidate_entry_count
rejected_window_count
```

### 5.3 推荐 memory features

V5 可以优先加入轻量结构化特征：

```text
memory_candidate_entry_count
memory_best_entry_observation_count
memory_best_entry_source_frame_count
memory_best_entry_bbox_history_count
memory_best_entry_association_confidence
memory_best_entry_distance_score
memory_best_entry_view_consistency_score
memory_best_entry_appearance_score
memory_best_entry_geometry_score
memory_best_entry_is_open_door
memory_best_entry_is_window
memory_rejected_window_count
memory_search_status_id
memory_entry_search_status_id
memory_previous_subgoal_id
memory_previous_action_id
```

这些特征比早期的 `previous_action / previous_subgoal` 更接近研究问题，因为它们直接描述：

**目标 house 的入口是否已经被稳定识别、验证和关联。**

---

## 6. 推荐的后续工作顺序

### Step 1：使用 session validator

当前已经新增脚本：

```text
phase2_multimodal_fusion_analysis/validate_memory_collection_session.py
```

输入：

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\validate_memory_collection_session.py `
  --session_dir E:\github\UAV-Flow\captures_remote\memory_collection_sessions\memory_episode_20260426_101452_search_house_1
```

输出：

```text
PASS / WARN / FAIL
capture_count
has_start_snapshot
has_stop_snapshot
best_true_count
best_entry_id
best_entry_type
bbox_history_count
source_frame_count
association_evidence_nonzero
missing_files
```

这样每次采集后不用手动打开 JSON。

目前标准样本验证结果应类似：

```text
[memory-validator] status=PASS captures=11 valid_captures=11 target=001
[memory-validator] search_status=APPROACHING_ENTRY entry_status=entry_found best_entry_id=0
[memory-validator] best=0:open door obs=10 frames=8 bbox=6 assoc=1.0
```

### Step 2：批量验证所有 memory sessions

建议再加一个批处理模式：

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\validate_memory_collection_session.py `
  --sessions_root E:\github\UAV-Flow\captures_remote\memory_collection_sessions
```

用来统计：

```text
valid_sessions
invalid_sessions
total_captures
target_house_distribution
entry_type_distribution
state_distribution
subgoal_distribution
```

### Step 3：导出 memory-aware distillation dataset

在现有 distillation dataset exporter 上增加 memory session 输入。

建议导出：

```text
rgb_path
depth_path
yolo_features
depth_features
target_context_features
memory_features
labels
```

### Step 4：训练 V5-B

推荐 V5-B 配置：

```text
base = v5 loss reweight
extra_input = enriched memory features
primary target = target_conditioned_state + target_conditioned_subgoal
secondary target = target_action + target_candidate
```

### Step 5：做 ablation

至少比较：

```text
no_memory
simple_memory
enriched_memory
enriched_memory_without_previous_action
```

重点看：

```text
target_conditioned_state
target_conditioned_subgoal
target_conditioned_action_hint
target_conditioned_target_candidate_id
```

---

## 7. 采集更多数据时的建议

### 7.1 每个 house 至少采三类 episode

对每个目标 house，建议采：

```text
search_house_X_front
search_house_X_left_detour
search_house_X_right_detour
```

如果有门口遮挡，再加：

```text
search_house_X_blocked_entry
```

如果窗户容易误检，再加：

```text
search_house_X_window_distractor
```

### 7.2 每个 episode 的最小数量

建议每个 episode 至少：

```text
capture_count >= 8
best_entry_observation_count >= 3
source_frames >= 3
bbox_history >= 3
```

### 7.3 保留失败样本

不要只保留成功接近入口的 episode。

训练需要同时看到：

```text
target_house_entry_visible
target_house_entry_blocked
non_target_house_entry_visible
target_house_entry_approachable
```

否则模型会只学会 approach，而不会学会继续搜索和过滤窗口。

---

## 8. 当前阶段结论

当前 memory V1 已经达到“可以进入训练数据导出”的最低要求。

最关键的通过点是：

1. episode 生命周期完整
2. capture 链路完整
3. target house 对齐
4. candidate entry 有多帧历史
5. bbox history 已经写入
6. association evidence 已经非零
7. best entry 已经唯一
8. final decision 可以从 blocked/search 转向 approachable/approach

下一步不建议继续只手工看 JSON。

应该先实现 session validator，然后批量验证更多 memory sessions，再把合格 session 接进 representation distillation dataset exporter。

一句话总结：

**现在的重点已经从“记忆功能是否能跑通”转到“怎样把合格记忆 episode 稳定导出成可训练样本”。**
