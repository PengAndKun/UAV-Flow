# Entry Search Memory Schema

## 1. 文档目标

这份文档把 [23_entry_search_memory_design_zh.md](23_entry_search_memory_design_zh.md) 里的“三层多模态记忆”设计，进一步落成一份可实现的 JSON schema 规范。

本 schema 的目标是统一：

- 运行时记忆写入格式
- reviewer / 调试 / 可视化读取格式
- 后续 teacher / state builder / dataset export 的输入格式
- 训练时 memory-to-feature 的转换边界

一句话说：

- 本文档回答“记忆应该长成什么样”
- 而不是“控制器怎样用它”

---

## 2. 设计原则

### 2.1 以 `house_id` 为顶层组织

记忆不是全局一锅粥，而是：

- 每个目标房屋一份 `entry_search_memory`

顶层建议：

```json
{
  "version": "v1",
  "updated_at": "2026-04-20 10:30:00",
  "memories": {
    "001": {},
    "002": {}
  }
}
```

### 2.2 三层记忆并存

每个 `house_id` 下包含：

- `working_memory`
- `episodic_memory`
- `semantic_memory`

### 2.3 运行时存 JSON，训练时转张量

schema 的职责是：

- 存原始结构化信息

它**不是**训练时最终直接喂模型的张量格式。

后续应由：

- `entry_state_builder`
- `feature_builder`

把这份 memory schema 转成固定长度向量。

---

## 3. 顶层结构

推荐顶层文件名：

- `entry_search_memory.json`

推荐结构：

```json
{
  "version": "v1",
  "updated_at": "2026-04-20 10:30:00",
  "current_target_house_id": "001",
  "memories": {
    "001": {
      "house_id": "001",
      "house_name": "House_1",
      "working_memory": {},
      "episodic_memory": [],
      "semantic_memory": {}
    }
  }
}
```

---

## 4. House-Level Memory Schema

每个 `house_id` 下建议统一结构如下：

```json
{
  "house_id": "001",
  "house_name": "House_1",
  "house_status": "IN_PROGRESS",
  "target_match_active": true,
  "created_at": "2026-04-20 10:00:00",
  "updated_at": "2026-04-20 10:30:00",
  "working_memory": {},
  "episodic_memory": [],
  "semantic_memory": {}
}
```

字段说明：

- `house_id: string`
- `house_name: string`
- `house_status: string`
  - 来源于现有 `HouseRegistry`
  - 例如：`UNSEARCHED / IN_PROGRESS / EXPLORED / PERSON_FOUND`
- `target_match_active: bool`
  - 表示当前是否正在作为主要搜索目标
- `created_at: string`
- `updated_at: string`

---

## 5. Working Memory Schema

### 5.1 作用

Working memory 是短时缓存，主要服务当前数步内决策。

### 5.2 推荐结构

```json
{
  "target_house_id": "001",
  "current_house_id": "001",
  "last_best_entry_id": "entry_001_02",
  "recent_actions": ["yaw_left", "hold", "forward"],
  "recent_target_decisions": [
    {
      "target_conditioned_state": "target_house_entry_visible",
      "target_conditioned_subgoal": "keep_search_target_house",
      "target_conditioned_action_hint": "yaw_left",
      "timestamp": 1770000001.0
    }
  ],
  "top_candidates": [
    {
      "candidate_id": "entry_001_02",
      "class_name": "door",
      "confidence": 0.88,
      "target_match_score": 0.82,
      "geometry_score": 0.71,
      "distance_cm": 420.0,
      "opening_width_cm": 110.0,
      "traversable": true,
      "crossing_ready": false
    }
  ]
}
```

### 5.3 字段定义

#### 顶层字段

- `target_house_id: string`
- `current_house_id: string`
- `last_best_entry_id: string`
- `recent_actions: string[]`
  - 最多保留最近 `3~5` 步
- `recent_target_decisions: object[]`
  - 最多保留最近 `3~5` 条
- `top_candidates: object[]`
  - 建议最多保留 `top-3`

#### `recent_target_decisions[*]`

- `target_conditioned_state: string`
- `target_conditioned_subgoal: string`
- `target_conditioned_action_hint: string`
- `timestamp: float`

#### `top_candidates[*]`

- `candidate_id: string`
- `class_name: string`
- `confidence: float`
- `target_match_score: float`
- `geometry_score: float`
- `distance_cm: float`
- `opening_width_cm: float`
- `traversable: bool`
- `crossing_ready: bool`

### 5.4 Working memory 更新规则

- 新帧到来时，覆盖 `top_candidates`
- `recent_actions` 保持固定窗口
- `recent_target_decisions` 保持固定窗口
- `last_best_entry_id` 优先记录当前最有希望的目标入口

---

## 6. Episodic Memory Schema

### 6.1 作用

Episodic memory 保存关键多模态观察快照，用于：

- 避免重复搜索
- 供后续 teacher 或 LLM retrieval 使用
- 供人工审核和调试使用

### 6.2 推荐结构

```json
[
  {
    "snapshot_id": "snap_001_0007",
    "house_id": "001",
    "sector_id": "front_center",
    "pose": {
      "x": 3200.0,
      "y": 120.0,
      "z": 580.0,
      "yaw": 15.0
    },
    "view_type": "front_view",
    "rgb_path": "rgb.png",
    "depth_preview_path": "depth_preview.png",
    "fusion_overlay_path": "fusion_overlay.png",
    "sample_id": "fusion_20260420_001",
    "candidate_entry_id": "entry_001_02",
    "observation_result": "blocked_entry",
    "target_conditioned_state": "target_house_entry_blocked",
    "target_conditioned_subgoal": "detour_left_to_target_entry",
    "target_conditioned_action_hint": "left",
    "timestamp": 1770000012.0
  }
]
```

### 6.3 字段定义

- `snapshot_id: string`
- `house_id: string`
- `sector_id: string`
- `pose: object`
  - `x: float`
  - `y: float`
  - `z: float`
  - `yaw: float`
- `view_type: string`
  - 推荐取值：
    - `front_view`
    - `left_oblique`
    - `right_oblique`
    - `close_entry_view`
- `rgb_path: string`
- `depth_preview_path: string`
- `fusion_overlay_path: string`
- `sample_id: string`
  - 对应某个 `labeling` 样本或 fusion 样本 id
- `candidate_entry_id: string`
- `observation_result: string`
  - 推荐取值：
    - `no_entry`
    - `window_only`
    - `blocked_entry`
    - `approachable_entry`
    - `non_target_entry_visible`
- `target_conditioned_state: string`
- `target_conditioned_subgoal: string`
- `target_conditioned_action_hint: string`
- `timestamp: float`

### 6.4 Episodic memory 保留策略

不建议保存所有帧。

推荐保留条件：

- target-conditioned 状态发生变化
- 新出现高分 candidate entry
- 某个扇区第一次被观察
- 某个 candidate 从 `unverified` 变成 `blocked / approachable / non_target`

### 6.5 去重策略

如果新 snapshot 满足：

- 与旧 snapshot 距离差 `< 150~200cm`
- yaw 差 `< 20~30deg`
- 且 `target_conditioned_state` 相同

则可判为重复，不必新增。

---

## 7. Semantic Memory Schema

### 7.1 作用

Semantic memory 存长期聚合知识，是 planner / policy 最值得直接读取的一层。

### 7.2 推荐结构

```json
{
  "entry_search_status": "searching_entry",
  "last_best_entry_id": "entry_001_02",
  "search_summary": {
    "observed_sector_count": 3,
    "candidate_entry_count": 2,
    "approachable_entry_count": 0,
    "blocked_entry_count": 1,
    "rejected_entry_count": 1
  },
  "searched_sectors": {
    "front_left": {
      "observed": true,
      "observation_count": 2,
      "last_visit_time": 1770000010.0,
      "best_entry_state": "window_only",
      "best_target_conditioned_subgoal": "keep_search_target_house",
      "best_target_match_score": 0.42
    },
    "front_center": {
      "observed": true,
      "observation_count": 3,
      "last_visit_time": 1770000012.0,
      "best_entry_state": "blocked_entry",
      "best_target_conditioned_subgoal": "detour_left_to_target_entry",
      "best_target_match_score": 0.82
    }
  },
  "candidate_entries": [
    {
      "candidate_id": "entry_001_02",
      "semantic_class": "door",
      "target_match_score": 0.82,
      "distance_cm": 420.0,
      "opening_width_cm": 110.0,
      "status": "blocked_temporary",
      "attempt_count": 1,
      "last_checked_time": 1770000012.0
    }
  ]
}
```

### 7.3 `entry_search_status`

推荐枚举：

- `not_started`
- `searching_entry`
- `entry_found`
- `entered_house`
- `entry_search_exhausted`

### 7.4 `search_summary`

便于快速查看整体进度。

字段建议：

- `observed_sector_count: int`
- `candidate_entry_count: int`
- `approachable_entry_count: int`
- `blocked_entry_count: int`
- `rejected_entry_count: int`

### 7.5 `searched_sectors`

第一版推荐固定扇区：

- `front_left`
- `front_center`
- `front_right`
- `left_side`
- `right_side`

每个扇区字段：

- `observed: bool`
- `observation_count: int`
- `last_visit_time: float | null`
- `best_entry_state: string`
- `best_target_conditioned_subgoal: string`
- `best_target_match_score: float`

### 7.6 `candidate_entries`

每个候选建议字段：

- `candidate_id: string`
- `semantic_class: string`
- `target_match_score: float`
- `distance_cm: float`
- `opening_width_cm: float`
- `status: string`
- `attempt_count: int`
- `last_checked_time: float`

推荐 `status` 枚举：

- `unverified`
- `non_target`
- `window_rejected`
- `blocked_temporary`
- `blocked_confirmed`
- `approachable`
- `entered`

---

## 8. 完整示例

```json
{
  "version": "v1",
  "updated_at": "2026-04-20 10:30:00",
  "current_target_house_id": "001",
  "memories": {
    "001": {
      "house_id": "001",
      "house_name": "House_1",
      "house_status": "IN_PROGRESS",
      "target_match_active": true,
      "created_at": "2026-04-20 10:00:00",
      "updated_at": "2026-04-20 10:30:00",
      "working_memory": {
        "target_house_id": "001",
        "current_house_id": "001",
        "last_best_entry_id": "entry_001_02",
        "recent_actions": ["yaw_left", "hold", "forward"],
        "recent_target_decisions": [
          {
            "target_conditioned_state": "target_house_entry_visible",
            "target_conditioned_subgoal": "keep_search_target_house",
            "target_conditioned_action_hint": "yaw_left",
            "timestamp": 1770000001.0
          }
        ],
        "top_candidates": [
          {
            "candidate_id": "entry_001_02",
            "class_name": "door",
            "confidence": 0.88,
            "target_match_score": 0.82,
            "geometry_score": 0.71,
            "distance_cm": 420.0,
            "opening_width_cm": 110.0,
            "traversable": true,
            "crossing_ready": false
          }
        ]
      },
      "episodic_memory": [
        {
          "snapshot_id": "snap_001_0007",
          "house_id": "001",
          "sector_id": "front_center",
          "pose": {
            "x": 3200.0,
            "y": 120.0,
            "z": 580.0,
            "yaw": 15.0
          },
          "view_type": "front_view",
          "rgb_path": "rgb.png",
          "depth_preview_path": "depth_preview.png",
          "fusion_overlay_path": "fusion_overlay.png",
          "sample_id": "fusion_20260420_001",
          "candidate_entry_id": "entry_001_02",
          "observation_result": "blocked_entry",
          "target_conditioned_state": "target_house_entry_blocked",
          "target_conditioned_subgoal": "detour_left_to_target_entry",
          "target_conditioned_action_hint": "left",
          "timestamp": 1770000012.0
        }
      ],
      "semantic_memory": {
        "entry_search_status": "searching_entry",
        "last_best_entry_id": "entry_001_02",
        "search_summary": {
          "observed_sector_count": 3,
          "candidate_entry_count": 2,
          "approachable_entry_count": 0,
          "blocked_entry_count": 1,
          "rejected_entry_count": 1
        },
        "searched_sectors": {
          "front_left": {
            "observed": true,
            "observation_count": 2,
            "last_visit_time": 1770000010.0,
            "best_entry_state": "window_only",
            "best_target_conditioned_subgoal": "keep_search_target_house",
            "best_target_match_score": 0.42
          },
          "front_center": {
            "observed": true,
            "observation_count": 3,
            "last_visit_time": 1770000012.0,
            "best_entry_state": "blocked_entry",
            "best_target_conditioned_subgoal": "detour_left_to_target_entry",
            "best_target_match_score": 0.82
          }
        },
        "candidate_entries": [
          {
            "candidate_id": "entry_001_02",
            "semantic_class": "door",
            "target_match_score": 0.82,
            "distance_cm": 420.0,
            "opening_width_cm": 110.0,
            "status": "blocked_temporary",
            "attempt_count": 1,
            "last_checked_time": 1770000012.0
          }
        ]
      }
    }
  }
}
```

---

## 9. 与现有模块的接口建议

### 9.1 与 `house_registry` 的关系

- `house_registry` 继续保存房屋级 mission 状态
- `entry_search_memory` 保存门级和搜索级状态

不要把两者混成一个对象。

### 9.2 与 `target-conditioned fusion` 的关系

fusion 应更新：

- `working_memory.top_candidates`
- `working_memory.recent_target_decisions`
- `semantic_memory.candidate_entries`
- `semantic_memory.searched_sectors`

### 9.3 与 teacher / dataset export 的关系

teacher 和 export 不一定直接存全部 memory，但应能从 memory 中读取：

- 当前 house 入口搜索阶段
- 某些扇区是否已搜
- 某候选是否已被排除

### 9.4 与训练特征构建的关系

训练时建议提取：

- `working_memory` -> 当前短时向量
- `semantic_memory` -> 长期稳定向量
- `episodic_memory` -> 可选 retrieval 特征

---

## 10. 第一版实现建议

第一版不要一次全做满。

### 必做

- 顶层 `entry_search_memory.json`
- `semantic_memory.entry_search_status`
- `semantic_memory.searched_sectors`
- `semantic_memory.candidate_entries`
- `working_memory.last_best_entry_id`

### 可后做

- `episodic_memory` 的自动关键帧筛选
- 基于 memory 的 retrieval
- 复杂的 memory summarization

---

## 11. 一句话结论

如果后面要把“目标房屋入口搜索”真正做成可持续的系统，最好的 schema 不是单一状态表，而是：

- **Working Memory**
- **Episodic Memory**
- **Semantic Memory**

三层并存，统一挂在：

- `entry_search_memory[house_id]`

之下。

这份 schema 的意义在于：

- 先把运行时记忆、审核数据、训练前状态统一成同一套结构
- 后面无论接 fusion、teacher、dataset export 还是策略训练，都能按这一套接口继续往下走

