# 多模态入口搜索记忆设计（中文）

## 1. 文档目标

这份文档用于回答两个直接影响后续系统设计的问题：

1. 在多模态融合条件下，应该使用哪种记忆结构最合适？
2. 这些记忆数据应当如何存储、呈现，并接入后续训练？

这里的任务不是普通导航，而是：

- 指定 `target_house_id`
- 在多栋房屋环境中找到 **目标房屋**
- 再找到 **目标房屋的入口**
- 判断入口是否可进入
- 避免反复看同一个区域、重复试同一个门

因此我们不适合只用一种单层记忆，而应该使用：

- **Working Memory**
- **Episodic Memory**
- **Semantic Memory**

组成的 **三层多模态混合记忆**。

---

## 2. 参考论文与设计来源

这份设计不是照搬某一篇论文，而是更接近几篇近年 embodied memory 路线的工程综合版。

最接近的参考包括：

1. **3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning (CVPR 2025)**
   - 强调多视角场景快照、探索过程中的 3D 场景记忆
   - 对我们这里的 `visited_viewpoints` 很有启发

2. **3DLLM-Mem: Long-Term Spatial-Temporal Memory for Embodied 3D Large Language Model (NeurIPS 2025)**
   - 强调 long-term spatial-temporal memory
   - 很适合启发我们把当前帧判断和历史入口搜索联系起来

3. **Human-Inspired Memory Modeling for Embodied Exploration and QA (arXiv 2026)**
   - 明确把记忆拆成 working / episodic / semantic
   - 与我们当前的工程需求最一致

4. **SeGuE: Semantic Guided Exploration**
   - 更偏 next-best-view 与语义引导探索
   - 对 `searched_sectors` 和“下一步去哪里找入口”很有帮助

因此，如果你问“我这边应该用哪种记忆最好”，最稳的答案是：

- **三层多模态混合记忆**

而不是只用：

- 纯文本记忆
- 纯图像历史
- 或单一状态表

---

## 3. 为什么单层记忆不够

### 3.1 只用当前帧

如果只看当前 RGB、depth、YOLO、fusion：

- 会不知道哪些视角已经看过
- 不知道哪个门之前已经排除
- 容易在目标房屋周围来回重复搜索

### 3.2 只用文本摘要

如果只保留“LLM 描述过的文本记忆”：

- 几何细节丢失太多
- 很难恢复门候选的精确状态
- 对训练不友好

### 3.3 只用房屋级状态

现有：

- `UNSEARCHED`
- `IN_PROGRESS`
- `EXPLORED`
- `PERSON_FOUND`

只能表达整栋房子的总体进度，不能表达：

- 哪个方向已经看过
- 哪个门候选被挡住
- 哪个入口可接近

所以需要分层。

---

## 4. 推荐方案：三层多模态混合记忆

### 4.1 Working Memory

作用：

- 保存 **当前决策真正需要的短时上下文**
- 供 fusion / teacher / policy 直接使用

推荐保存：

- 当前帧 `top-K YOLO candidates`
- 当前 `depth ROI` 特征
- 当前 `fusion_result`
- 当前 `target_house_id`
- 最近 `3~5` 步动作
- 最近 `3~5` 帧 target-conditioned 判断
- 最近的 `last_best_entry_id`

这层适合回答：

- 现在看到的是不是目标入口？
- 下一步应该 `approach / detour / keep_search`？

### 4.2 Episodic Memory

作用：

- 记录“我在哪里看过什么”
- 保留关键多模态观察快照

推荐保存：

- `viewpoint snapshot`
- 当时的 `rgb.png`
- `depth_preview.png`
- `fusion_overlay.png`
- `pose / yaw`
- `house_id`
- `sector_id`
- 当时的 `target_conditioned_state`
- 当时最好的 `candidate entry`

这层适合回答：

- 我之前在 House_1 左前角看过一次，只看到窗
- 我在正门前 4 米观察过，这个门当时是 blocked

### 4.3 Semantic Memory

作用：

- 存聚合后的长期稳定知识
- 是后续 planner 和 student 最适合直接读取的一层

推荐保存：

- `entry_search_status`
- `searched_sectors`
- `candidate_entries`
- `last_best_entry_id`
- `candidate status summary`
- `target house search summary`

这层适合回答：

- House_1 的正面和左前角都看过了
- 右前角还没搜索
- 候选门 `entry_001_02` 已经是 `blocked_temporary`

---

## 5. 为什么这套结构最适合多模态融合

因为你当前系统本来就是多模态的：

- `RGB`
- `Depth`
- `YOLO`
- `Fusion`
- `Target house / map prior`

所以记忆也必须是多模态的。

最自然的组织方式不是“按模态拆”，而是：

- **按候选入口组织**
- **按房屋扇区组织**
- **按历史视点组织**

也就是说，你的记忆中心应该是：

- 哪个 `house`
- 哪个 `sector`
- 哪个 `candidate entry`
- 当时的多模态证据是什么

而不是只保存“某帧 RGB”或“某段文本”。

---

## 6. 建议的记忆中心对象

我建议把系统的长期外部入口记忆围绕一个对象展开：

- `entry_search_memory[house_id]`

其内部再包含三部分：

```json
{
  "working_memory": {},
  "episodic_memory": [],
  "semantic_memory": {}
}
```

其中：

- `working_memory`：当前短时信息
- `episodic_memory`：历史快照
- `semantic_memory`：聚合状态

---

## 7. Semantic Memory 最小结构

这一层最重要，因为它直接支撑“不要重复搜索”。

推荐第一版最小结构：

```json
{
  "house_id": "001",
  "entry_search_status": "searching_entry",
  "last_best_entry_id": "entry_001_02",
  "searched_sectors": {},
  "candidate_entries": []
}
```

### 7.1 `entry_search_status`

建议定义：

- `not_started`
- `searching_entry`
- `entry_found`
- `entered_house`
- `entry_search_exhausted`

### 7.2 `searched_sectors`

第一版建议 5 个扇区：

- `front_left`
- `front_center`
- `front_right`
- `left_side`
- `right_side`

每个扇区记录：

- `observed`
- `observation_count`
- `last_visit_time`
- `best_entry_state`
- `best_target_conditioned_subgoal`
- `best_target_match_score`

### 7.3 `candidate_entries`

每个候选门建议记录：

- `candidate_id`
- `semantic_class`
- `target_match_score`
- `distance_cm`
- `opening_width_cm`
- `status`
- `attempt_count`
- `last_checked_time`

其中 `status` 推荐：

- `unverified`
- `non_target`
- `window_rejected`
- `blocked_temporary`
- `blocked_confirmed`
- `approachable`
- `entered`

---

## 8. Episodic Memory 应该怎样呈现

Episodic memory 不建议保存所有帧，而应保存 **关键快照**。

每条快照建议包含：

```json
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
  "rgb_path": "rgb.png",
  "depth_preview_path": "depth_preview.png",
  "fusion_overlay_path": "fusion_overlay.png",
  "target_conditioned_state": "target_house_entry_blocked",
  "target_conditioned_subgoal": "detour_left_to_target_entry",
  "target_conditioned_action_hint": "left",
  "candidate_entry_id": "entry_001_02",
  "timestamp": 1770000012.0
}
```

我建议 episodic memory 保留的是：

- 图片路径
- 当前 pose
- 当前关键判断

而不是把整份运行时大状态全复制一遍。

这样有三个好处：

1. 便于人工审核
2. 便于后续 LLM retrieval
3. 便于训练时做 snapshot-level retrieval or filtering

---

## 9. Working Memory 应该怎样呈现

Working memory 不适合存很久，它更像当前控制器缓存。

推荐结构：

```json
{
  "target_house_id": "001",
  "current_house_id": "001",
  "top_candidates": [
    {
      "candidate_id": "entry_001_02",
      "class_name": "door",
      "confidence": 0.88,
      "target_match_score": 0.82,
      "distance_cm": 420.0,
      "traversable": false
    }
  ],
  "recent_decisions": [
    {
      "target_conditioned_state": "target_house_entry_visible",
      "target_conditioned_subgoal": "keep_search_target_house",
      "target_conditioned_action_hint": "yaw_left"
    }
  ],
  "recent_actions": ["yaw_left", "yaw_left", "hold"]
}
```

Working memory 只负责：

- 当前几步内的连续性
- 减少控制抖动
- 给当前 teacher / student 输入短期上下文

---

## 10. 怎样定义“一个 house 的门已经探索完了”

这个问题不应该只看 house 全局状态，而应看 `entry_search_status`。

我建议分三种“完成”：

### 10.1 成功完成

满足：

- 已确认目标入口
- 且 UAV 已进入房屋内部

状态：

- `entered_house`

### 10.2 搜索穷尽完成

满足：

- 各主要扇区已观察
- 显著候选已验证
- 没有可进入目标入口

状态：

- `entry_search_exhausted`

### 10.3 候选级完成

只代表某个门候选已经处理完，不代表整栋 house 完成。

例如：

- 该候选是窗
- 该候选属于非目标房屋
- 该候选被长期遮挡

此时应更新 `candidate_entries[*].status`，不应提前结束整栋房屋入口搜索。

---

## 11. 怎样避免重复探索

### 11.1 不重复回同一个视点

如果新视点和历史视点：

- 距离差小于 `150~200cm`
- yaw 差小于 `20~30deg`
- 且历史观察结论稳定

则判为“已看过的视点”，优先级下降。

### 11.2 不重复试已排除的门

如果某候选已是：

- `non_target`
- `window_rejected`
- `blocked_confirmed`

则除非视角变化很大，否则不要再把它当主要目标。

### 11.3 优先跟踪最好的目标门

如果某候选已经是：

- `approachable`

则即使画面里又出现其他弱候选，也应优先继续跟踪这个目标门。

---

## 12. 训练时怎样使用这些记忆

### 12.1 存储层

存储时建议保留：

- `JSON`
- `图片路径`
- `summary`

这层主要服务：

- 运行时
- 审核
- 数据清洗

### 12.2 训练层

训练时不直接喂完整 JSON，而是提取成：

- `global_state vector`
- `candidate table`
- `sector memory vector`
- `last_best_entry embedding`
- 可选 `episodic retrieval embedding`

也就是说：

- **存储层：结构化 JSON + 图像**
- **训练层：固定长度张量**

这是最稳的做法。

---

## 13. 第一版实现建议

### 13.1 必做

先实现这三项就够：

- `entry_search_status`
- `searched_sectors`
- `top-3 candidate_entries`

### 13.2 可以暂缓

先不要一开始就做：

- 复杂图数据库
- 全量历史帧索引
- 3D 稠密 scene graph
- 室内外统一长期地图记忆

因为对你当前问题来说，最有收益的是：

- 不重复看已搜扇区
- 不重复试已排除门
- 继续跟踪真正有希望的目标入口

---

## 14. 最推荐的结论

如果你问：

- 我这边是多模态融合，我应该使用哪种记忆最好？

我的答案是：

- **Working + Episodic + Semantic 的三层多模态混合记忆**

如果你再问：

- 这种记忆数据应该怎样呈现？

我的建议是：

1. **运行时与审核**
   - 用 `JSON + 图片路径 + summary`

2. **训练时**
   - 用 `固定长度向量 / 表格特征 / embedding`

3. **长期主记忆对象**
   - 以 `entry_search_memory[house_id]` 为中心组织

一句话总结：

你这条线最适合的不是“单一 memory”，而是：

- **以目标房屋和候选门为中心的三层多模态记忆系统**

这样既能支撑工程落地，也能支撑后面的 teacher、distillation 和策略训练。

