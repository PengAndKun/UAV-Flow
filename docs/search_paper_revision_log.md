# Search Paper Revision Log

## 1. 文档目的

本文档用于记录当前 `UAV-Flow` 项目从“分层多模态 UAV 导航系统”向“室内房屋找人具身搜索系统”转向时的方案修改内容、实验修改清单，以及后续最小可发表版本的落地范围。

这份文档不替代：
- [uav_phase_progress_log.md](/E:/github/UAV-Flow/docs/uav_phase_progress_log.md)
- [phase4_entry_plan.md](/E:/github/UAV-Flow/docs/phase4_entry_plan.md)
- [hierarchical_multimodal_uav_plan.md](/E:/github/UAV-Flow/docs/hierarchical_multimodal_uav_plan.md)

它的作用是单独回答两个问题：
- 现有工程方案和新的论文方案到底差在哪里
- 如果只做一个“最小可发表版本”，应该保留什么、修改什么、舍弃什么

---

## 2. 当前方案与新方案的关系

## 2.1 不变的大方向

大方向保持不变，仍然采用：
- 高层稀疏语义规划
- 中层 archive / memory
- 低层轻量 reflex 执行
- 人类监督接管

也就是说，当前工程的主架构仍然是可以复用的，尤其是：
- `uav_control_server.py`
- `planner_server.py`
- `archive_runtime.py`
- `reflex policy` 路线
- `takeover logging`
- `online eval`

## 2.2 真正变化的地方

新的论文问题不再主要研究“导航到目标点”，而是研究：

- 在未知或部分未知房屋中
- 根据语言任务主动搜索人员
- 判断是否有人
- 估计人员位置
- 减少重复搜索
- 在遮挡和复杂几何条件下完成确认

因此，研究问题从：
- `navigation`

转为：
- `language-guided embodied person search`

---

## 3. 与现有工程相比需要修改的核心内容

## 3.1 任务定义的变化

当前工程任务主要是：
- `move right 3 meters`
- `turn right`
- `move forward`

新的论文任务应改为：
- `search the house for people`
- `search the bedroom first`
- `inspect occluded corners`
- `approach and verify suspect person`

结论：
- 当前任务口径偏导航
- 新论文任务口径必须明确转为搜索

## 3.2 planner 输出语义的变化

当前 planner 输出主要是：
- `semantic_subgoal`
- `candidate_waypoints`
- `planner_confidence`

新的 planner / LLM 需要输出：
- 优先搜索区域
- 候选房间或子区域
- 是否继续搜索 / 复查 / 确认
- 搜索优先级解释

结论：
- 现有 planner schema 可保留基本骨架
- 但输出语义要从“导航子目标”升级成“搜索子目标”

## 3.3 archive 含义的变化

当前 archive 更像导航状态记忆，记录：
- 任务标签
- 子目标
- 位姿量化
- 深度摘要
- retrieval 命中

新的 archive 要升级为搜索记忆，至少增加：
- `visited_regions`
- `observed_regions`
- `suspect_regions`
- `confirmation_status`
- `room / region semantic tag`
- `person evidence summary`

结论：
- 当前 archive 不是错
- 但它只是搜索 archive 的前身

## 3.4 低层策略定位的变化

当前低层 `reflex` 已经能做：
- 动作建议
- 半自动执行
- takeover 修正记录

新的低层策略应被重新定义为：
- 面向高层搜索子目标的局部执行器
- 不只是 waypoint follower
- 还要支持靠近确认、遮挡边缘查看、安全复查

## 3.5 感知与结果输出的变化

当前工程已经稳定支持：
- RGB
- depth
- risk
- archive retrieval
- reflex runtime

但新的论文方案必须补上：
- 人体检测 / 可疑区域检测
- 多帧证据融合
- 是否有人
- 人的位置估计
- 证据帧输出

结论：
- 这是当前系统到论文新问题之间最大的能力缺口

---

## 4. 实验修改清单

## 4.1 必须改

- 任务定义改成搜索任务，而不是导航子任务
- 日志 schema 增加：
  - `person_exists`
  - `person_gt_position`
  - `house_id`
  - `room_id`
  - `visited_regions`
  - `suspect_regions`
  - `confirmed_regions`
  - `evidence_frames`
  - `search_result`
- 主评价指标改成：
  - `Search Success Rate`
  - `Localization Error`
  - `Time-to-First-Detection`
  - `Coverage-before-Detection`
  - `Collision Rate`
  - `Takeover Count`
  - `Decision Latency`
  - `Token Usage`
- 新增人体证据融合模块
- baseline 改成搜索相关 baseline，不再只做 `prototype vs mlp`

## 4.2 建议改

- 在 `runtime_interfaces.py` 里定义正式 `mission schema`
- 在 `uav_control_server.py` 里新增 `search_runtime`
- 在 `archive_runtime.py` 中加入 search-conditioned cell 字段
- 在 `uav_control_panel.py` 中增加搜索状态显示
- 在 `online_reflex_eval.py` 基础上新增 search summary
- 将 Phase 3 和 Phase 4 的现有导航实验保留为 system validation

## 4.3 后续再改

- 用真实 LLM mission planner 替换 heuristic planner
- 更强的人体证据融合和遮挡确认逻辑
- 更系统的 safety / replan 学习模块
- 真实机或半实物实验
- 用半自动搜索 episode 做数据飞轮训练

---

## 5. 最小可发表版本

## 5.1 目标

最小可发表版本不追求“一次性做全”，而是保证论文问题、系统能力和实验结果三者一致。

建议的最小可发表版本目标是：

- 在未知室内房屋环境中
- 给定语言搜索任务
- 无人机基于 RGB + depth + pose
- 借助稀疏 planner/LLM + search archive + 轻量 reflex
- 主动搜索是否有人
- 在发现后输出人员位置估计与证据帧

## 5.2 最小系统组成

### 模块 A. Mission Guidance

输入：
- 语言任务
- 当前房屋搜索摘要
- archive 检索结果

输出：
- 下一优先搜索区域
- 候选航点
- 是否重规划

第一版允许：
- 仍然使用规则 planner 或 heuristic planner
- 但输出字段必须符合 mission guidance 口径

### 模块 B. Search Archive

第一版要求 archive 至少记录：
- 当前房间 / 区域
- visited 状态
- suspect 状态
- 当前搜索进度
- 关键局部观测

### 模块 C. Reflex Executor

第一版低层执行可以直接沿用当前：
- `mlp_reflex_policy_v4`
- `assist_step / manual execute`
- safety gating
- takeover logging

换句话说，最小版本不要求立刻重训一个全新的搜索低层策略。

### 模块 D. Person Evidence Fusion

第一版可以做得很轻：
- 单帧人体检测
- depth 辅助位置投影
- 多帧证据累积
- 输出 `person_exists / estimated_position / confidence`

重点是要先把“搜索结果输出”闭环补上。

## 5.3 最小实验范围

### 主实验

建议只做一类核心任务：
- `Search the house for people and return person location if found`

环境配置建议最小覆盖：
- house with person
- house without person
- visible person
- partially occluded person
- different lighting conditions

### baseline 组

最小版本建议只保留 4 组：
- `B1` RGB detection + rule search
- `B2` RGB + depth + rule search
- `B3` planner + reflex without archive search memory
- `Ours` mission guidance + search archive + reflex + evidence fusion

### 指标

最小主指标建议只保留：
- `SSR`
- `LE`
- `TFD`
- `CR`
- `Takeover Count`
- `DL`
- `TU`

### 消融

最小版本只做 4 个消融：
- `-LLM / mission guidance`
- `-archive`
- `-depth`
- `-evidence fusion`

### 系统验证

当前已经完成的这些实验不要浪费，建议下放到系统验证部分：
- reflex runtime 接通
- online eval 工具
- assist_step 行为
- takeover logging
- fixed spawn
- low-level policy offline comparison

---

## 6. 后续修改计划

## 6.1 第一阶段：先改任务与日志，不先改模型

目标：
- 把问题定义从导航切到搜索

要做的文件：
- `UAV-Flow-Eval/runtime_interfaces.py`
- `UAV-Flow-Eval/uav_control_server.py`
- `UAV-Flow-Eval/uav_control_panel.py`

要完成的内容：
- mission schema
- search episode schema
- search runtime state
- capture / bundle / log 中的 search 字段

## 6.2 第二阶段：把 planner 改成 mission guidance adapter

目标：
- planner 不再主要输出导航动作语义
- 而是输出搜索语义

要做的文件：
- `UAV-Flow-Eval/planner_server.py`
- `UAV-Flow-Eval/runtime_interfaces.py`

要完成的内容：
- `mission_input`
- `search_subgoal`
- `candidate_regions`
- `candidate_waypoints`
- `replan_flag`

## 6.3 第三阶段：扩 archive 为 search archive

目标：
- 显式记录搜索记忆而不是纯导航记忆

要做的文件：
- `UAV-Flow-Eval/archive_runtime.py`
- `UAV-Flow-Eval/reflex_dataset_builder.py`

要完成的内容：
- `visited / observed / suspect`
- 区域级别统计
- 搜索进度摘要
- 高价值复查区域检索

## 6.4 第四阶段：补人体证据融合

目标：
- 让系统能正式回答“是否有人、在哪里”

建议新增文件：
- `UAV-Flow-Eval/person_evidence_fusion.py`
- `UAV-Flow-Eval/person_search_runtime.py`

第一版只做：
- detection result cache
- multi-frame fusion
- estimated person position
- evidence frame export

## 6.5 第五阶段：跑新的论文主实验

目标：
- 把现有导航实验迁移成搜索论文实验

要完成的内容：
- house-level benchmark
- person/no-person episodes
- 搜索任务指标
- 主实验表
- 消融实验表
- 泛化实验

---

## 7. 文件级开发清单

### 必须修改

- `UAV-Flow-Eval/runtime_interfaces.py`
- `UAV-Flow-Eval/uav_control_server.py`
- `UAV-Flow-Eval/uav_control_panel.py`
- `UAV-Flow-Eval/planner_server.py`
- `UAV-Flow-Eval/archive_runtime.py`
- `UAV-Flow-Eval/online_reflex_eval.py`

### 建议新增

- `UAV-Flow-Eval/person_evidence_fusion.py`
- `UAV-Flow-Eval/person_search_runtime.py`
- `UAV-Flow-Eval/search_eval.py`
- `UAV-Flow-Eval/search_dataset_builder.py`

### 可以复用现有结果的部分

- `phase3_dataset_export/*`
- `reflex_policy_server.py`
- `takeover logs`
- `capture bundles`
- `fixed spawn pose`
- `reflex executor / assist_step`

---

## 8. 建议的论文结果结构

建议最终论文实验章节分成四块：

### 8.1 Main Results

比较搜索成功率、定位误差、首次发现时间、碰撞率、接管次数。

### 8.2 Ablation

验证：
- mission guidance
- archive
- depth
- evidence fusion

### 8.3 Generalization

验证：
- unseen houses
- unseen occlusion
- unseen lighting
- unseen prompts

### 8.4 System Validation

放当前工程已经做好的：
- runtime stability
- takeover logging
- assist_step
- fixed spawn
- online metrics

---

## 9. 当前推荐执行顺序

如果只按“尽快形成一篇可写论文”的顺序，建议这样推进：

1. 先把 `mission/search schema` 定下来
2. 再把 `uav_control_server.py` 改成能记录 search runtime
3. 再改 `planner_server.py` 的高层输出语义
4. 再扩 `archive_runtime.py`
5. 再补 `person_evidence_fusion.py`
6. 最后集中跑搜索实验

---

## 10. 当前结论

当前系统并没有偏离新论文方向。

更准确地说：
- 当前工程已经做出了一个很强的导航与执行底座
- 新论文需要在这个底座上把“导航问题”升级成“搜索问题”

因此，后续工作的重点不是推翻当前系统，而是：
- 重写任务口径
- 重写实验口径
- 补齐搜索相关状态、记忆与证据输出

一旦这三件事完成，当前已有的 planner / archive / reflex / takeover / online eval 资产都可以继续复用。
