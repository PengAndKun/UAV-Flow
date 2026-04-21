# Entry Search Memory Implementation Plan

## 1. 文档目标

这份文档把 [24_entry_search_memory_schema.md](24_entry_search_memory_schema.md) 的 schema 继续往前推进一步，回答：

- 这套 `entry_search_memory` 应该先接哪一层？
- 每一步需要改哪些模块？
- 先做哪些字段，后做哪些字段？
- 每一步做到什么程度算“通过”？

目标不是一次性把全部 memory 都写完，而是：

- 先做一个最小可用版本
- 确保它能和现有 `target-conditioned fusion / teacher / distillation` 链平滑衔接

---

## 2. 当前系统中的接入位置

这套记忆最终会影响 4 类模块：

1. **运行时状态存储层**
   - 保存和更新 `entry_search_memory`

2. **target-conditioned fusion**
   - 用 memory 抑制重复候选
   - 用 memory 决定是否继续搜索某个扇区

3. **teacher / entry state / dataset export**
   - 把 memory 中的长期上下文变成训练监督的一部分

4. **后续策略训练**
   - 把 memory 压成可训练输入特征

从工程顺序上，建议依次推进，而不是四条线一起改。

---

## 3. 总体实现顺序

推荐顺序：

1. **先做 Memory 文件与读写器**
2. **再接 Semantic Memory**
3. **再接 Working Memory**
4. **再接 Episodic Memory**
5. **然后把 memory 接进 fusion**
6. **最后再接进 teacher / export / training**

也就是说：

- 第一步先让系统“记得住”
- 第二步再让系统“用得上”

---

## 4. 阶段划分

## 阶段 A：Memory Store 基础设施

### 目标

先让系统具备一个独立的：

- `entry_search_memory.json`

文件和读写逻辑。

### 需要做的事

新增一个 memory store 模块，例如：

- `entry_search_memory.py`

负责：

- 读取 memory 文件
- 初始化缺失 house 的 memory
- 更新某个 house 的 memory
- 保存回磁盘

### V1 先支持的顶层字段

只要求：

- `version`
- `updated_at`
- `current_target_house_id`
- `memories`

### 这一步不需要做的事

- 不需要接 fusion
- 不需要接 teacher
- 不需要自动更新 episodic snapshots

### Done 标准

满足以下条件就算通过：

1. 能创建一个新的 `entry_search_memory.json`
2. 能读取 [houses_config.json](/E:/github/UAV-Flow/UAV-Flow-Eval/houses_config.json) 中已有 house，自动补出对应 memory 壳子
3. 能对指定 `house_id` 做读写更新
4. JSON 结构符合 [24_entry_search_memory_schema.md](24_entry_search_memory_schema.md)

---

## 阶段 B：先接 Semantic Memory

### 为什么先做这一层

因为它最稳定、最便宜、对当前系统收益最大。

Working memory 和 episodic memory 偏动态，而 semantic memory 是：

- 长期聚合结果
- 最适合 first-pass 接进 fusion 与 teacher

### V1 先接的字段

优先只做：

- `entry_search_status`
- `last_best_entry_id`
- `searched_sectors`
- `candidate_entries`
- `search_summary`

### 推荐最小写入逻辑

每次跑完一轮 fusion 后：

1. 根据当前 `target_house_id` 找到对应 house memory
2. 用当前判断更新：
   - 当前 sector 是否已观察
   - 当前 best candidate 是否进入 `candidate_entries`
   - 当前 `entry_search_status` 是否变化

### 当前不用做的复杂能力

- 候选聚类
- 跨多轮的复杂合并
- 3D 空间对齐 refinement

### Done 标准

满足以下条件就算通过：

1. 每次 fusion 后，目标 house 的 `searched_sectors` 会变化
2. `candidate_entries` 至少能记录 top-1 或 top-3 候选
3. `entry_search_status` 能从：
   - `not_started -> searching_entry -> entry_found / entry_search_exhausted`
4. 多次运行后 memory 会累积，不会每次被清空

---

## 阶段 C：接 Working Memory

### 目标

加入短时决策连续性，避免动作抖动和状态跳变。

### V1 推荐字段

- `last_best_entry_id`
- `recent_actions`
- `recent_target_decisions`
- `top_candidates`

### 推荐写入方式

每次 fusion / teacher 运行后更新：

- 当前 top candidates
- 当前 target-conditioned 决策
- 最近动作

保留固定窗口：

- 最近 `3~5` 条

### Done 标准

满足以下条件就算通过：

1. 当前 memory 中能看到最近若干条 action
2. 当前 memory 中能看到最近若干条 target-conditioned 判断
3. `last_best_entry_id` 能随最优候选变化而更新

---

## 阶段 D：接 Episodic Memory

### 目标

保存关键视点快照，而不是所有帧。

### V1 推荐策略

只在下面几种情况新增 snapshot：

1. target-conditioned state 发生变化
2. 出现新的高分 candidate
3. 某个 sector 第一次被观察
4. 某候选门状态发生关键变化
   - 例如：
     - `unverified -> blocked_temporary`
     - `blocked_temporary -> approachable`
     - `unverified -> non_target`

### V1 推荐字段

- `snapshot_id`
- `house_id`
- `sector_id`
- `pose`
- `rgb_path`
- `depth_preview_path`
- `fusion_overlay_path`
- `target_conditioned_state`
- `target_conditioned_subgoal`
- `target_conditioned_action_hint`
- `candidate_entry_id`

### Done 标准

满足以下条件就算通过：

1. memory 中能看到有代表性的 snapshot
2. 不会每一帧都新增 snapshot
3. reviewer 或开发者可以回看关键观察历史

---

## 阶段 E：把 Memory 接进 Fusion

### 目标

让 memory 不只是被保存，而是开始参与推理。

### 第一批最值得接的逻辑

#### 1. 重复扇区降权

如果某个 `sector` 已多次观察且没有新候选：

- 降低再次去该扇区的优先级

#### 2. 已排除候选降权

如果某个 `candidate_entry.status` 已是：

- `non_target`
- `window_rejected`
- `blocked_confirmed`

则对相似候选降低优先级。

#### 3. 继续跟踪最优候选

如果 `last_best_entry_id` 仍然有效：

- 不要轻易被新的弱候选打断

### 这一步不要先做的事

- 复杂 memory retrieval
- LLM long-context 推理
- 多 house 竞争式记忆重排

### Done 标准

满足以下条件就算通过：

1. fusion 输出开始受到 memory 影响
2. 系统不再反复围绕同一个无效门打转
3. 系统能在未搜扇区和已搜扇区之间体现优先级差异

---

## 阶段 F：把 Memory 接进 Teacher / Dataset Export

### 目标

让 teacher 和训练数据开始显式利用 memory。

### 推荐先接入的字段

在 teacher 输入里增加：

- 当前 `entry_search_status`
- `sector_observed_mask`
- `last_best_entry_exists`
- `candidate_status_summary`

在 dataset export 中增加：

- `memory_features`
  - `observed_sector_count`
  - `sector_observed_mask`
  - `candidate_entry_count`
  - `approachable_entry_count`
  - `blocked_entry_count`
  - `last_best_entry_exists`

### Done 标准

满足以下条件就算通过：

1. `teacher_output` 可参考 memory 给出更稳定的子任务
2. `entry_state` 或导出样本中已有 memory-derived 特征
3. 训练集已不再只是“当前帧状态”，而是带有短/长时上下文

---

## 阶段 G：接训练器

### 目标

把 memory 真正变成 student 的状态输入。

### 推荐接入方式

训练时不直接喂整份 JSON，而是做 memory feature builder。

建议拆成：

- `working_memory_features`
- `semantic_memory_features`
- `episodic_retrieval_features`（可后做）

### 第一版最值得先接的

优先接：

- `sector_observed_mask`
- `observed_sector_count`
- `candidate_status_counts`
- `last_best_entry_exists`
- `entry_search_status`

episodic retrieval 先放到第二版。

### Done 标准

满足以下条件就算通过：

1. 训练输入中已经包含 memory-derived features
2. 模型开始减少重复搜索类错误
3. `keep_search / approach / detour` 的稳定性提高

---

## 5. 推荐的实际开发顺序

最稳的实际顺序是：

1. `entry_search_memory.py`
2. `memory schema validator`
3. `fusion -> semantic memory writer`
4. `working memory writer`
5. `episodic snapshot writer`
6. `fusion memory-aware scoring`
7. `teacher memory input`
8. `dataset export memory features`
9. `feature_builder memory branch`

也就是说：

- 先写入
- 再使用
- 最后训练

---

## 6. 推荐先改哪些文件

### 第一批

- 新增：
  - `phase2_multimodal_fusion_analysis/entry_search_memory.py`

- 更新：
  - [fusion_entry_analysis.py](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/fusion_entry_analysis.py)

### 第二批

- 更新：
  - [teacher_validator.py](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/teacher_validator.py)
  - [entry_state_builder.py](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/entry_state_builder.py)
  - [distillation_dataset_export.py](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/distillation_dataset_export.py)

### 第三批

- 更新：
  - [feature_builder.py](/E:/github/UAV-Flow/phase2_5_representation_distillation/feature_builder.py)
  - 后续 trainer 输入分支

---

## 7. 第一版最重要的验收标准

如果只能用一句话概括第一版的成功标准，那就是：

**系统开始显式记住“哪些方向已经搜过、哪些门已经试过、当前最值得继续追哪个入口”。**

具体可以拆成：

1. 同一无效门不会被连续高频重复选择
2. 已搜索扇区优先级下降
3. `last_best_entry_id` 能连续跟踪
4. teacher 和 state builder 开始能看到 memory 字段

---

## 8. 一句话结论

`entry_search_memory` 的实现不应该一步到位做成“大而全”的长期场景图，而应该按下面顺序推进：

1. 先做 **Semantic Memory**
2. 再做 **Working Memory**
3. 后做 **Episodic Memory**
4. 然后让 fusion、teacher、dataset、training 逐层接入

这样风险最小、收益最大，也最适合你现在这条多模态融合入口搜索主线。

