# 38. 确定性分段搜索记忆重设计方案

## 1. 文档目标

本文档重新设计当前的 entry search memory，使它从“粗粒度 coverage ratio 记忆”升级为“RGB/YOLO + Depth 同步证据驱动的确定性分段搜索记忆”。

核心目标是：

```text
一旦某个 house 的某个外立面分段已经被可靠 RGB/depth 观测确认，就立即把该分段标记为搜索完成，后续控制器和 LLM 不应再重复探索这个分段。
```

这次先写技术方案，不直接改代码。后续代码实现应按本文档分阶段接入。

---

## 2. 当前问题

当前 memory 已经能记录：

- 当前 target house
- perimeter coverage
- face coverage
- candidate entries
- searched / unsearched 状态

但实际 LLM 控制时仍出现重复探索，原因是现在的 memory 更像“连续覆盖率统计”，还没有足够强地表达：

```text
这个 house 的这一段外墙已经被 RGB/YOLO + Depth 确认搜索过，因此不要再回头看。
```

也就是说，当前系统容易出现：

- 已经看过关闭门，后续又反复靠近关闭门。
- 已经扫过一个墙面区间，后续因为 LLM 不确定又回头采样。
- 只用 coverage ratio 判断不够直观，不能直接告诉控制器“下一段应该去哪”。
- `observed_coverage_ratio` 能说明看过多少，但不够说明“哪些具体段已经完成搜索”。

新的设计应把“是否探索完成”从模糊累计值变成明确的 segment 状态。

---

## 3. 核心思想

### 3.1 以 house face segment 作为最小记忆单元

把每个 house 的外立面拆成若干可搜索分段：

```text
house
  face_front
    segment_0
    segment_1
    segment_2
    segment_3
  face_left
    segment_0
    segment_1
    segment_2
    segment_3
  face_right
  face_back
```

第一版建议每个 face 拆成 `4` 段。这样足够表达“沿着一面墙搜索到哪里”，又不会复杂到需要精确 3D 重建。

### 3.2 RGB/YOLO 决定语义，Depth 决定几何状态

每次 memory capture 后已有同步数据：

- `rgb.png`
- `depth_cm.png`
- `yolo_result.json`
- `depth_result.json`
- `fusion_result.json`
- UAV pose / yaw
- target house id

由于 RGB 和 depth 是同步图像，YOLO bbox 可以直接映射到 depth 图同一像素区域。

因此每个候选框都可以做：

```text
YOLO semantic class
+ depth ROI statistics
+ surrounding wall depth statistics
+ target house / face / segment assignment
= deterministic entry/window/closed/open evidence
```

### 3.3 “100% 完成”是工程确定性，不是绝对世界真值

这里的“100% 确认该区域已经探索过”应定义为：

```text
在当前传感器、当前视角和当前规则阈值下，该 segment 已经获得足够可靠的可解释证据，可以被控制器视为完成搜索，不应再主动重复访问。
```

它不是说真实世界绝对没有误差，而是说工程状态机可以把该 segment 关闭。

如果证据不足，不能强行标记完成，应进入：

- `observed_uncertain`
- `partially_observed`
- `needs_review`
- `occluded`

---

## 4. Segment 状态定义

建议每个 segment 使用以下状态机：

```text
unseen
  -> observed_uncertain
  -> searched_no_entry
  -> searched_closed_entry
  -> searched_window_only
  -> entry_found_open
  -> blocked_or_occluded
  -> needs_review
```

推荐语义如下：

| 状态 | 含义 | 后续是否重复搜索 |
|---|---|---|
| `unseen` | 尚未被有效观测 | 需要搜索 |
| `observed_uncertain` | 看过但证据不足 | 可以补观察 |
| `searched_no_entry` | 该段墙面清晰可见，无门或可进入口 | 不再主动重复搜索 |
| `searched_closed_entry` | 发现关闭门或不可进入门 | 不再重复靠近，除非任务要求开门 |
| `searched_window_only` | 仅发现窗户，无可进入门 | 不再主动重复搜索 |
| `entry_found_open` | 发现可进入或可能可进入入口 | 转入接近入口逻辑 |
| `blocked_or_occluded` | 该段被树、墙、车等遮挡 | 需要换视角或飞高 |
| `needs_review` | 规则矛盾，需要人工或 LLM 复核 | 不直接完成 |

---

## 5. 单帧如何更新 segment

### 5.1 输入

每次 capture 使用以下输入：

```json
{
  "target_house_id": "002",
  "uav_pose": {
    "x": 1448.0,
    "y": 1276.0,
    "z": 225.0,
    "yaw": 122.9
  },
  "rgb_path": "rgb.png",
  "depth_path": "depth_cm.png",
  "yolo_result_path": "yolo_result.json",
  "fusion_result_path": "fusion_result.json",
  "house_geometry": {
    "house_id": "002",
    "bbox_world": {}
  }
}
```

### 5.2 判断当前图像覆盖了哪些 segment

V1 可以先不用精确相机投影，采用轻量几何近似：

1. 根据 UAV 世界坐标和 house bbox，判断 UAV 当前位于 house 的哪一侧。
2. 得到当前主要观测 face，例如 `front / left / right / back`。
3. 根据 UAV 沿该 face 的相对位置，确定当前主要覆盖的 segment。
4. 根据图像中目标 house 的横向占比，决定是否同时覆盖相邻 segment。

示例：

```json
{
  "visible_face_id": "front",
  "primary_segment_index": 2,
  "visible_segments": [
    {"face_id": "front", "segment_index": 1, "visibility_ratio": 0.35},
    {"face_id": "front", "segment_index": 2, "visibility_ratio": 0.82},
    {"face_id": "front", "segment_index": 3, "visibility_ratio": 0.41}
  ]
}
```

后续 V2 可以用更精确的相机内参和 depth projection 做 segment 投影。

### 5.3 YOLO bbox 同步到 depth ROI

对每个 YOLO bbox：

```text
rgb bbox = [x1, y1, x2, y2]
depth bbox = same pixel range on depth_cm.png
```

计算：

- `roi_depth_median_cm`
- `roi_depth_p10_cm`
- `roi_depth_p90_cm`
- `roi_valid_ratio`
- `surrounding_wall_depth_median_cm`
- `depth_gap_cm = roi_depth_median_cm - surrounding_wall_depth_median_cm`
- `far_pixel_ratio`
- `near_pixel_ratio`
- `vertical_clearance_ratio`
- `lower_band_near_ratio`
- `center_band_far_ratio`

这里最重要的是比较 bbox 内部和周围墙面的深度差异。

---

## 6. 门/窗状态的确定性规则

### 6.1 Open door / 可进入入口

候选满足以下条件时，可以标为 `entry_found_open`：

```text
YOLO class in {open door, door}
and roi_valid_ratio >= 0.60
and center_band_far_ratio >= threshold
and depth_gap_cm >= threshold
and vertical_clearance_ratio >= threshold
and bbox is not too high like a window
```

解释：

- RGB/YOLO 提供“这是 door-like object”的语义。
- depth ROI 证明这个区域后方明显更深。
- 中心区域有足够远深度，说明不是一扇贴墙的关闭门。
- 垂直可通行区域足够，说明不是小窗户或窄缝。

### 6.2 Closed door / 关闭门

候选满足以下条件时，可以标为 `searched_closed_entry`：

```text
YOLO class in {close door, door}
and roi_valid_ratio >= 0.60
and abs(depth_gap_cm) <= small_gap_threshold
and center_band_far_ratio is low
and surrounding wall depth is consistent
```

解释：

- YOLO 认为它是门。
- depth 看起来和墙面接近，没有明显向内延伸。
- 因此该 segment 已搜索完成，但不是可进入入口。

### 6.3 Window / 窗户

候选满足以下条件时，可以标为 `searched_window_only`：

```text
YOLO class == window
or bbox vertical position is high
or bbox aspect/size resembles window
and depth ROI does not show body-height traversable opening
```

解释：

- 窗户是有用证据，因为它说明该段外立面已经看清。
- 但窗户不应触发 entry approach。

### 6.4 No candidate but visible wall

如果某个 segment 没有 YOLO 门窗候选，但满足：

```text
segment_visibility_ratio >= threshold
and target house wall region is visible
and image is not heavily occluded
and depth valid ratio is high
```

则可以标为：

```text
searched_no_entry
```

这点非常关键：不是只有看到门/窗才算搜索过。看清楚一段墙且没有入口，也应该关闭该 segment。

### 6.5 Occlusion / 不确定

如果图像中该 segment 被遮挡：

```text
tree / fence / parked car / wall / too close / depth invalid
```

则不能标记完成，应为：

```text
blocked_or_occluded
```

此时控制器可以选择：

- 换视角
- 后退
- 飞高
- 沿边移动到下一个观察位置

---

## 7. Segment Memory Schema

建议在每个 house 的 `semantic_memory` 里新增：

```json
{
  "segment_search_memory": {
    "version": "segment_search_memory_v1",
    "segment_count_per_face": 4,
    "faces": {
      "front": {
        "segments": [
          {
            "segment_id": "H002_front_0",
            "state": "searched_no_entry",
            "is_search_complete": true,
            "completion_confidence": 1.0,
            "last_observed_step": 31,
            "last_observed_time": "2026-05-01T10:12:30",
            "observation_count": 2,
            "visibility_ratio_best": 0.84,
            "evidence_type": "visible_wall_no_entry",
            "best_evidence": {
              "capture_id": "memory_capture_20260501_101230_step0031_llm_control",
              "labeling_dir": ".../labeling",
              "yolo_classes": [],
              "depth_valid_ratio": 0.91,
              "occlusion_ratio": 0.05
            },
            "candidate_entries": []
          }
        ]
      }
    },
    "summary": {
      "total_segments": 16,
      "complete_segments": 7,
      "uncertain_segments": 2,
      "entry_found_segments": 1,
      "remaining_segments": 6,
      "next_unsearched_segment": {
        "face_id": "right",
        "segment_index": 1
      }
    }
  }
}
```

### 7.1 Candidate evidence schema

每个 segment 内候选可以这样记录：

```json
{
  "candidate_id": "H002_front_2_cand_0003",
  "source_class": "close door",
  "bbox_xyxy": [220, 180, 330, 410],
  "semantic_confidence": 0.87,
  "depth_roi": {
    "roi_depth_median_cm": 510.0,
    "surrounding_wall_depth_median_cm": 505.0,
    "depth_gap_cm": 5.0,
    "roi_valid_ratio": 0.93,
    "center_band_far_ratio": 0.04,
    "vertical_clearance_ratio": 0.05
  },
  "deterministic_state": "closed_entry",
  "segment_update": "searched_closed_entry",
  "evidence_confidence": 1.0,
  "reason": "YOLO close door with no far-depth opening inside ROI."
}
```

---

## 8. Segment 完成判定

### 8.1 可以直接完成的情况

以下状态应设置：

```json
{
  "is_search_complete": true,
  "completion_confidence": 1.0
}
```

适用情况：

- `searched_no_entry`
- `searched_closed_entry`
- `searched_window_only`
- `entry_found_open`

### 8.2 不能直接完成的情况

以下状态不应完成：

- `unseen`
- `observed_uncertain`
- `blocked_or_occluded`
- `needs_review`

### 8.3 多候选冲突

如果同一个 segment 中同时出现：

- 一个候选像 open door
- 另一个候选像 window
- depth 证据互相冲突

则该 segment 不应直接完成，应设置：

```json
{
  "state": "needs_review",
  "is_search_complete": false,
  "reason": "Conflicting RGB/depth evidence in the same segment."
}
```

---

## 9. House 搜索完成判定

### 9.1 找到可进入入口

如果任意 segment 状态为：

```text
entry_found_open
```

并且入口属于当前 target house，则当前 house 可进入 approach 阶段。

当 UAV 靠近到约 3m：

```text
entry_distance_cm <= 300
```

当前 house 搜索任务完成：

```json
{
  "finish_type": "target_entry_reached",
  "next_action": "switch_to_next_house"
}
```

### 9.2 没找到入口但所有必需 segment 已完成

如果：

```text
required_segments_complete_ratio >= 1.0
and reliable_open_entry_count == 0
```

则可以判定：

```json
{
  "finish_type": "no_entry_after_segment_completion",
  "next_action": "switch_to_next_house"
}
```

### 9.3 有遮挡 segment 时

如果某些 segment 是 `blocked_or_occluded`：

```text
complete_segments < required_segments
```

则不能说该 house 完全没有入口，只能进入：

```text
continue_observe / change_viewpoint / temporary_search_altitude
```

如果多次尝试仍无法观测，可以设置：

```json
{
  "finish_type": "no_entry_after_reachable_segment_completion",
  "unresolved_segments": ["H002_back_1"],
  "needs_review": true
}
```

---

## 10. 对 LLM 控制的影响

LLM 不应再自己猜“哪里搜索过”。它应该读取结构化 segment memory summary。

Prompt 中应新增：

```json
{
  "segment_search_memory_summary": {
    "target_house_id": "002",
    "complete_segments": [
      "front_0",
      "front_1",
      "front_2"
    ],
    "entry_found_segments": [],
    "blocked_or_occluded_segments": [
      "right_0"
    ],
    "next_unsearched_segment": {
      "face_id": "front",
      "segment_index": 3,
      "recommended_motion": "advance_forward_then_yaw"
    },
    "do_not_revisit_segments": [
      "front_0",
      "front_1",
      "front_2"
    ]
  }
}
```

LLM 控制规则应明确：

```text
Never intentionally return to a segment whose is_search_complete=true unless the system reports needs_review or explicit user override.
```

控制器应优先使用确定性 memory override：

1. 如果 LLM 想回头看已完成 segment，override 为前往 `next_unsearched_segment`。
2. 如果 LLM 想靠近关闭门，且该 segment 为 `searched_closed_entry`，override 为沿 face 前进到下一段。
3. 如果 LLM 想继续左右抖动，而当前 segment 已完成，override 为 `w` 或 yaw 到下一 segment。
4. 如果 `entry_found_open`，优先 center + approach，不再继续搜索其他 segment。

---

## 11. 和现有 memory 的关系

当前已有：

- `perimeter_coverage`
- `face_coverage`
- `searched_sectors`
- `candidate_entries`

新的 `segment_search_memory` 不一定立刻替代它们，而是作为更强的显式层。

建议关系：

```text
searched_sectors: 旧版粗粒度扇区状态
perimeter_coverage: 旧版环绕覆盖统计
face_coverage: 当前 face/segment 覆盖统计
segment_search_memory: 新版确定性完成标记
candidate_entries: 入口实例表
```

后续如果新版稳定，可以让 `face_coverage` 主要服务于几何覆盖统计，而让 `segment_search_memory` 负责决策级完成判断。

---

## 12. 实现阶段计划

### Phase A：schema 与离线回放

目标：

- 不接控制器。
- 只对已有 memory session 回放生成 segment memory。

需要新增或修改：

- `phase2_multimodal_fusion_analysis/entry_search_memory.py`
- 新增 `segment_search_memory` schema
- 新增离线脚本，例如 `rebuild_segment_search_memory_from_session.py`

Done 标准：

- 输入一个已有 `memory_episode_*`
- 输出每个 capture 后的 segment 状态变化
- 生成 `segment_search_memory_report.json`
- 能看出哪些 segment 被标记完成

### Phase B：接入 fusion 更新

目标：

- 每次 `memory_capture_analyze` 后自动更新 segment memory。

需要修改：

- `phase2_multimodal_fusion_analysis/fusion_entry_analysis.py`
- `phase2_multimodal_fusion_analysis/entry_search_memory.py`
- `UAV-Flow-Eval/uav_control_server_basic.py`

Done 标准：

- 每次 capture 后 `entry_search_memory_snapshot_after.json` 包含 `segment_search_memory`
- `fusion_result.json` 中包含 `segment_search_memory_summary`
- 同一 segment 完成后，不会被重复标为未搜索

### Phase C：接入 LLM prompt 和控制 override

目标：

- LLM 每次决策能看到已完成 segment。
- 控制器可以拦截明显重复探索动作。

需要修改：

- `phase2_multimodal_fusion_analysis/memory_aware_llm_teacher_prompt_builder.py`
- `UAV-Flow-Eval/uav_control_panel_basic.py`

Done 标准：

- Prompt 中出现 `do_not_revisit_segments`
- LLM 日志中能看到 `next_unsearched_segment`
- 如果 LLM 选择返回已完成 segment，控制器记录 override reason

### Phase D：可视化与人工验证

目标：

- 在 memory inspector 或 map window 中显示 segment 搜索状态。

建议颜色：

- 灰色：`unseen`
- 黄色：`observed_uncertain`
- 绿色：`searched_no_entry / searched_window_only / searched_closed_entry`
- 蓝色：`entry_found_open`
- 红色：`needs_review`
- 橙色：`blocked_or_occluded`

Done 标准：

- 可以直观看到 house 哪些边、哪些段已完成。
- 复盘时能解释 UAV 为什么不再回头搜索某段。

---

## 13. 测试方案

### 13.1 单帧规则测试

选取已有 capture：

- open door
- close door
- window
- wall-only
- occluded by tree

对每张图输出：

```json
{
  "visible_segments": [],
  "candidate_evidence": [],
  "segment_updates": []
}
```

检查规则是否符合人工直觉。

### 13.2 session replay 测试

对以下类型 session 做回放：

- house 1 正门找到 open door
- house 1 侧面找到 open door
- house 2 绕行无入口
- house 2 被墙/树遮挡的绕行
- house 1 到 house 3 多建筑任务

统计：

- segment 完成数量随时间变化
- 重复访问已完成 segment 次数
- no-entry 判定是否过早
- open-entry 是否被正确保留为 approach 目标

### 13.3 LLM 控制对比

对比：

- 原始 memory
- segment memory only
- segment memory + control override

指标：

- 重复探索次数
- 完成一个 no-entry house 的决策步数
- 找到 open door 后继续偏离的次数
- 撞墙/危险 lateral 次数

---

## 14. 关键风险

### 14.1 过早关闭 segment

如果 segment 还没看清就标记完成，会漏掉入口。

缓解：

- 要求 `visibility_ratio_best >= threshold`
- 要求 `depth_valid_ratio >= threshold`
- 遮挡比例高时只能 `blocked_or_occluded`
- 模糊或冲突时 `needs_review`

### 14.2 depth ROI 受视角影响

门在图像边缘或视角斜的时候，depth gap 可能不稳定。

缓解：

- bbox 接近图像边缘时降低置信度
- 要求门尽量居中后再做最终 open/closed 判定
- 允许多帧证据合并

### 14.3 window 和 open door 混淆

窗户也可能有较深 depth。

缓解：

- 使用 YOLO class
- 使用 bbox 高度、垂直位置、离地比例
- 使用 body-height traversable corridor 判断

### 14.4 house segment 归属错误

如果当前看到的是邻近 house，segment 更新会污染 target house memory。

缓解：

- segment update 必须经过 target-house association gate
- 当前 house 不确定时，只写 `needs_review` 或 `unassigned_evidence`
- 不把 non-target house 的门写入 target house 的完成证据

---

## 15. 下一步代码任务清单

建议按以下顺序执行：

1. 在 `entry_search_memory.py` 中新增 `segment_search_memory` 默认 schema。
2. 写一个纯函数 `classify_yolo_depth_candidate(...)`，输入 YOLO bbox 和 depth ROI，输出 open/closed/window/no-entry evidence。
3. 写一个纯函数 `assign_visible_house_segments(...)`，输入 UAV pose、yaw、house bbox、图像上下文，输出 visible segments。
4. 写一个更新器 `update_segment_search_memory(...)`。
5. 先写离线 replay 脚本验证历史 session。
6. 验证通过后，再接入 `memory_capture_analyze` 在线更新。
7. 最后把 summary 接入 LLM prompt 和控制 override。

---

## 16. 当前离线测试版

已经先实现一个不接入在线控制器的测试脚本：

- [segment_search_memory_replay.py](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/segment_search_memory_replay.py)
- [run_segment_search_memory_replay.py](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/run_segment_search_memory_replay.py)

运行示例：

```powershell
C:\Users\Administrator\miniconda3\envs\unrealcv\python.exe `
  E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_segment_search_memory_replay.py `
  --session_dir E:\github\UAV-Flow\captures_remote\memory_collection_sessions\memory_episode_20260430_222749_to_2_search_round_room_3
```

快速测试前 N 个 capture：

```powershell
C:\Users\Administrator\miniconda3\envs\unrealcv\python.exe `
  E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_segment_search_memory_replay.py `
  --session_dir E:\github\UAV-Flow\captures_remote\memory_collection_sessions\memory_episode_20260430_235309_to_2_search_round_room_8 `
  --max_captures 12
```

输出文件默认写入：

```text
<memory_episode_dir>/segment_search_memory_report.json
```

报告中最重要的字段：

- `segment_memories`
- `segment_memories.<house_id>.summary`
- `timeline`
- `timeline[].updates`
- `summary.repeated_completed_segment_observations`
- `summary.skip_reasons`

当前测试版规则特点：

- 如果 target house 不在视野中，不更新 segment，只记录 skip reason。
- 如果同一 segment 同时出现 `close door` 与 depth-open 证据，优先标记 `needs_review`，不直接判定为 open entry。
- 如果 segment 已经是 complete，再次被观测会记录 `repeat_completed_observation=true`，用于统计重复探索。
- 当前只做离线 report，不修改共享 `entry_search_memory.json`。

---

## 17. 当前批量验证工具

为了判断这套规则是否能稳定覆盖不同采集 episode，已经增加批量 replay 汇总脚本：

- [batch_segment_search_memory_replay.py](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/batch_segment_search_memory_replay.py)
- [run_batch_segment_search_memory_replay.py](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/run_batch_segment_search_memory_replay.py)

快速测试前 3 个 episode：

```powershell
C:\Users\Administrator\miniconda3\envs\unrealcv\python.exe `
  E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_batch_segment_search_memory_replay.py `
  --limit 3 `
  --max_captures 12 `
  --overwrite_reports
```

完整批量验证所有历史 episode：

```powershell
C:\Users\Administrator\miniconda3\envs\unrealcv\python.exe `
  E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_batch_segment_search_memory_replay.py `
  --overwrite_reports
```

输出目录默认是：

```text
E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\results\segment_search_memory_batch
```

每次运行会生成三类文件：

- `segment_search_memory_batch_summary_*.json`：完整总报告。
- `segment_search_memory_batch_sessions_*.csv`：episode 级汇总。
- `segment_search_memory_batch_houses_*.csv`：house 级 segment 状态汇总。

批量汇总重点看这些指标：

- `repeated_completed_segment_observations`：是否仍在重复探索已完成 segment。
- `needs_review_count`：YOLO 与 depth 规则冲突或证据不足的数量。
- `complete_segment_count / total_segments`：目标 house 的分段搜索完成度。
- `entry_found_open_count`：规则确认的可进入入口数量。
- `skip_reasons`：target house 不在视野、缺少数据等跳过原因。

这个批处理仍然是离线验证工具，不直接影响控制器状态。只有当批量结果稳定后，才建议把 segment summary 接入 LLM prompt 和在线 memory update。

---

## 18. 重复探索诊断工具

批量 replay 之后，可以进一步定位“到底是哪一个 segment 被反复搜索”：

- [analyze_segment_search_repetition.py](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/analyze_segment_search_repetition.py)
- [run_analyze_segment_search_repetition.py](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/run_analyze_segment_search_repetition.py)

运行：

```powershell
C:\Users\Administrator\miniconda3\envs\unrealcv\python.exe `
  E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_analyze_segment_search_repetition.py
```

输出仍然写入：

```text
E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\results\segment_search_memory_batch
```

生成文件：

- `segment_search_repetition_diagnostics_*.json`
- `segment_search_repetition_segments_*.csv`
- `segment_search_repetition_sessions_*.csv`

其中 `segment_search_repetition_segments_*.csv` 是最适合人工检查的表。每一行对应一个被重复观察的 completed segment，包含：

- `session_name`
- `house_id`
- `face_id`
- `segment_index`
- `first_complete_capture`
- `first_complete_state`
- `repeat_count`
- `repeat_step_preview`
- `likely_issue`

`likely_issue` 是一个粗略归因标签，例如：

- `open_entry_not_used_as_stop_or_approach_gate`
- `closed_entry_revisited_after_completion`
- `window_or_wall_segment_revisited`
- `no_entry_segment_revisited`
- `long_span_repeat_possible_memory_prompt_or_control_issue`

这个诊断工具的作用是把“LLM 为什么回头搜”从主观观察变成可统计问题。后续接入控制器时，可以直接把高重复 segment 作为 memory override 的优先目标：如果一个 segment 已经 completed 且 repeat_count 增长，就强制 planner 转向 next segment，而不是继续让 LLM 犹豫。

---

## 19. 控制器最小接入版本

当前已经在 [uav_control_panel_basic.py](/E:/github/UAV-Flow/UAV-Flow-Eval/uav_control_panel_basic.py) 中加入一个最小接入版本。

这个版本不是完整在线 memory 重构，而是在每次 LLM control 决策前做三件事：

1. 自动刷新当前 memory episode 的 `segment_search_memory_report.json`。
2. 把目标 house 的 segment summary 压缩进 LLM prompt。
3. 如果发现 `entry_found_open`，优先触发 approach/stop gate。

新增 prompt 字段：

```json
{
  "segment_search_memory": {
    "do_not_revisit_segments": [],
    "next_unsearched_segments": [],
    "entry_found_open_gate": {},
    "current_capture": {}
  }
}
```

字段语义：

- `do_not_revisit_segments`：RGB/YOLO + Depth 已经确认完成搜索的 house-face segment，不应该再主动重复探索。
- `next_unsearched_segments`：下一批应该继续探索的 segment。
- `entry_found_open_gate.present=true`：已经发现 open-entry segment，后续策略应优先 re-center / approach / stop。
- `current_capture.repeat_completed_observation=true`：当前截图又看到了已完成 segment，说明应该离开该区域去下一个未搜索 segment。

新增 rule override：

- `segment_entry_found_open_stop_gate`：open entry 已满足实际/代理到达条件，直接停止当前 house 搜索。
- `segment_entry_found_open_align_gate`：open entry 在图像边缘，先对准入口。
- `segment_entry_found_open_approach_gate`：open entry 已确认但还没到达，直接转为安全前进接近。

每次 LLM decision 的 artifact 目录中也会额外保存：

```text
segment_search_memory_control_summary.json
```

这样重跑实验后，可以直接检查当前 prompt 是否正确看到了：

- 哪些 segment 不要再搜
- 下一步应该搜哪些 segment
- 是否触发了 open-entry approach/stop gate

---

## 20. 一句话总结

新版搜索记忆的重点不是继续让 LLM 猜“我是不是看过这里”，而是让 RGB/YOLO + Depth 规则直接把 house 外立面分段变成可关闭的搜索单元：

```text
看清一段，就关闭一段；发现开门，就转入接近；发现关门/窗/无入口，就不再重复搜索该段。
```

这会让 house search 从“LLM 反复犹豫式探索”变成“确定性进度驱动的分段搜索”。
