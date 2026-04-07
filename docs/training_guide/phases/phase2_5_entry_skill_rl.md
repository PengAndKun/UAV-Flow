# Phase 2.5：局部门进入与静态避障技能学习

## 目标

在 Phase 2 已经完成 `YOLO26 + depth + fusion` 入口判断之后，新增一个**局部技能层**，专门解决下面这几个问题：

1. 看见目标入口后，如何稳定靠近入口。
2. 前方有静态障碍物时，如何先绕行再重新观察。
3. 入口已经确认可进入时，如何对齐门洞并穿过。
4. 后续如何把这个 skill 蒸馏成轻量 reactive policy，供最终 agent 调用。

这一阶段不负责全局找哪一栋房子，也不负责室内完整寻人；它只负责一个清晰的局部技能：

`Entry-Approach-and-Pass`

---

## 为什么要单独拆出 Phase 2.5

Phase 2 的重点是：

- 看懂当前画面里是不是门/窗/开口。
- 判断几何上是否可穿越。
- 结合前方障碍，输出 `enterable_open_door`、`front_blocked_detour`、`window_visible_keep_search` 等决策。

但仅仅“判断正确”还不够。真正的自动系统还需要一个可以执行的局部策略，解决：

- 门在左侧，应该先左转还是平移？
- 前方 40cm 有柱子，应该先后退还是侧移？
- 门洞可进，但还离 4m，应该怎样逐步靠近？
- 看到的是窗户，不应该贴上去，而应该继续搜索。

所以更合理的系统结构是：

1. **高层 agent**：决定当前目标房屋、是否继续搜门、是否切换到室内搜索。
2. **Phase 2 融合感知模块**：判断当前是否存在候选入口、前方是否被障碍挡住。
3. **Phase 2.5 局部技能策略**：根据融合结果完成靠近、绕障、穿门。
4. **底层执行器**：把策略输出的离散动作发给控制器。

---

## 阶段定位

可以把训练链理解成：

- Phase 2：`看见入口`
- Phase 2.5：`走到入口并通过`
- Phase 3：`跨房屋移动到目标房屋`
- Phase 4：`室内搜索`
- Phase 5：`蒸馏成轻量统一策略`

也就是说，Phase 2.5 是一个**桥接阶段**：

- 它把“感知结果”转成“局部行动能力”
- 它也是后续蒸馏最重要的 teacher 之一

---

## 任务定义

### 核心任务

定义一个局部任务：

`Entry-Approach-and-Pass`

初始状态：

- UAV 已经位于目标房屋附近。
- Phase 2 已经给出一个入口候选，或者给出“前方阻塞需要绕行”的提示。
- 场景暂时限定为静态障碍物，不考虑动态人/车。

目标状态：

- UAV 不碰撞地接近目标入口。
- 在近距离完成门洞对齐。
- 成功穿过入口到达室内安全区域。

失败条件：

- 碰撞。
- 卡在障碍物前持续无进展。
- 长时间围绕窗户或错误开口徘徊。
- 步数预算耗尽。

---

## 输入设计

第一版不建议直接纯 RGB 端到端，而是使用“融合感知 + 局部几何”的状态输入。

### 建议输入

1. 深度图局部观测
- 当前前视深度图，或其下采样版本
- 可选：仅取中心区域和左右区域的深度摘要

2. YOLO26 语义结果
- `door`
- `open door`
- `window`
- `close door`
- 目标框中心、宽高、置信度

3. Phase 2 融合结果
- `final_entry_state`
- `front_obstacle.present`
- `front_obstacle.front_min_depth_cm`
- `front_obstacle.severity`
- `best_entry.entry_distance_cm`
- `best_entry.opening_width_cm`
- `best_entry.traversable`
- `matched_semantic.class_name`
- `match_score`

4. UAV 当前局部状态
- 当前 yaw
- 最近 2 到 4 步动作历史
- 当前与入口中心的图像偏差
- 当前与入口的距离变化趋势

### 推荐表征

推荐使用两部分状态：

- 图像分支：
  - 深度图 / depth feature
- 结构化分支：
  - fusion 标量
  - detection 标量
  - pose / action history

这样训练更稳定，也更适合后期蒸馏。

---

## 动作空间设计

第一版建议使用**离散动作**，而不是连续控制量。

### 推荐动作

- `forward_small`
- `forward_medium`
- `yaw_left_small`
- `yaw_right_small`
- `left_small`
- `right_small`
- `hold`
- 可选：`backward_small`

### 不建议一开始使用连续动作的原因

- 难训。
- 更依赖精细动力学调参。
- 后续蒸馏到板载轻量策略不方便。

离散动作更适合：

- PPO
- 行为克隆初始化
- 数据集蒸馏

---

## 奖励函数设计

Phase 2.5 的奖励要同时体现三件事：

1. 是否朝正确入口靠近。
2. 是否安全避开障碍。
3. 是否高效完成穿门。

### 1. 成功奖励

- 成功进入目标门洞并到达室内安全区：`+10 ~ +20`

### 2. 进度奖励

- 入口距离减小：正奖励
- 入口中心偏差减小：正奖励
- 门洞在图像中心附近：正奖励
- 穿门后进入室内：额外奖励

### 3. 障碍惩罚

- 碰撞：大负奖励
- 前方极近障碍：负奖励
- 靠近高风险区域但无避让：持续负奖励

### 4. 语义约束

- 朝 `open door / door` 靠近：可给正奖励
- 朝 `window` 靠近：惩罚或不奖励
- `front_blocked_detour` 时继续硬顶前进：惩罚

### 5. 效率惩罚

- 每步小负奖励
- 左右抖动、原地反复转向：额外惩罚

---

## 训练课程设计

不建议一步训练完整困难场景，而是按课程学习推进。

### Phase 2.5-A：近距离门洞对齐

目标：

- 门就在前方 2m 到 4m 内
- 几乎无障碍
- 学会把门移到中心并通过

重点：

- 学会“看见门 -> 对齐 -> 前进”

### Phase 2.5-B：静态障碍绕行

目标：

- 门前加入柱子、灌木、栏杆、家具等静态障碍
- 前方可能出现 `front_blocked_detour`

重点：

- 学会“先绕开障碍，再重新对齐入口”

### Phase 2.5-C：门窗混淆

目标：

- 同时出现窗户、门、半开门、假开口

重点：

- 学会不靠近 `window`
- 学会优先靠近真正的 `door/open door`

### Phase 2.5-D：稍远距离入口接近

目标：

- UAV 从更远位置起步
- 门洞较小，障碍更复杂

重点：

- 学会从“看见入口”过渡到“接近并穿过”

---

## 算法建议

### 推荐主算法

第一版建议：

- **PPO + 离散动作**

原因：

- 稳定
- 与仿真环境兼容好
- 容易结合规则和行为克隆

### 推荐训练流程

1. 用规则策略 / 融合决策采集一批可用轨迹
2. 先做行为克隆初始化
3. 再用 PPO 微调

这样会比纯 RL 快很多，也更符合后续蒸馏路线。

---

## 与蒸馏的关系

Phase 2.5 不只是一个临时训练模块，它还承担后续蒸馏 teacher 的角色。

### 可记录到 archive 的轨迹字段

每一步建议额外记录：

- `final_entry_state`
- `front_obstacle.front_min_depth_cm`
- `front_obstacle.severity`
- `best_entry.entry_distance_cm`
- `best_entry.opening_width_cm`
- `best_entry.traversable`
- `matched_semantic.class_name`
- `match_score`
- `action`
- `reward`
- `success_flag`

### 后续蒸馏方向

后面可以从高质量成功轨迹中提取：

- `(observation, action)` 对
- 训练轻量 reactive policy

例如：

- 输入：深度图 + 少量融合标量
- 输出：离散动作

这样最终就能形成：

- 高层 agent 决策任务
- 低层蒸馏策略快速执行穿门与避障

---

## 与最终 agent 的接口

最终系统里，agent 不应该直接控制底层每一步动作，而应该像调用技能一样调用 Phase 2.5。

### 建议接口

输入：

- `target_house_id`
- `current_visible_house_id`
- `phase2_fusion_state`
- `uav_pose`

输出：

- `skill_status`
  - `searching_entry`
  - `approaching_entry`
  - `detouring_obstacle`
  - `crossing_entry`
  - `entry_complete`
  - `entry_failed`
- `suggested_action`
- `progress_score`

这样最终 agent 只需要决定：

- 当前是否调用 `entry_skill`
- 当前是否中断并切回搜索

---

## 核心评价指标

Phase 2.5 建议独立评估，不和全任务混在一起。

### 核心指标

- `entry_pass_success_rate`
- `collision_rate`
- `mean_steps_to_cross_entry`
- `front_block_recovery_rate`
- `wrong_target_approach_rate`
- `window_approach_error_rate`

### 建议报告维度

- 按障碍密度分组
- 按入口类型分组
- 按起始距离分组
- 按是否存在窗户干扰分组

---

## 交付物

- [ ] `phase2_5_entry_skill_env.py`
- [ ] `phase2_5_entry_skill_ppo.py`
- [ ] `phase2_5_entry_skill_bc_init.py`
- [ ] `phase2_5_entry_skill_eval.py`
- [ ] Phase 2.5 成功轨迹 archive
- [ ] 训练曲线与评估报告

---

## 总结

Phase 2 负责“看懂入口”，Phase 2.5 负责“真正进入入口”。

它最合理的定位不是一个全局搜索模型，而是一个局部技能学习模块：

- 看到入口时会靠近
- 前方有障碍时会绕行
- 靠近门洞后会对齐并穿过

这一步既能直接提升系统自动化水平，也能为后续 Phase 5 蒸馏提供高质量 teacher 轨迹。
