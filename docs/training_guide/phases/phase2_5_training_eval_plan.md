# Phase 2.5：训练、验证与 TMM 实验方案

## 1. 这一阶段到底训练什么

Phase 2.5 的目标不要定义得太大。

这一阶段不训练：

- 全局找哪一栋房子
- 全局跨房屋导航
- 室内完整寻人
- 最终统一 agent 的所有行为

这一阶段只训练一个局部技能：

`Entry-Approach-and-Pass`

也就是：

1. 发现目标入口后，如何靠近
2. 前方有静态障碍时，如何绕行
3. 靠近入口后，如何对齐并穿过

这一步训练好了，后面才能稳定接到：

- 高层 agent
- Phase 3 跨房屋导航
- Phase 4 室内搜索
- Phase 5 蒸馏

---

## 2. 推荐的整体训练流程

建议按下面 5 步走，而不是直接端到端硬训。

### Step 2.5-0：固定输入与输出接口

先把训练任务标准化。

#### 输入

- 深度图局部观测
- YOLO26 检测结果
- Phase 2 融合判断结果
- UAV 当前局部状态
- 最近动作历史

#### 输出

离散动作：

- `forward_small`
- `forward_medium`
- `yaw_left_small`
- `yaw_right_small`
- `left_small`
- `right_small`
- `hold`
- 可选：`backward_small`

先用离散动作，不要一开始做连续控制。

---

### Step 2.5-1：先做 teacher 轨迹采集

先不要直接纯 RL。

优先准备两类 teacher 数据：

1. 规则策略轨迹
- 根据融合结果写一个 rule-based entry controller
- 例如：
  - `front_blocked_detour` -> 优先侧移/转向
  - `enterable_open_door` -> 优先对齐门中心并前进
  - `window_visible_keep_search` -> 不靠近窗口，继续搜索

2. 少量人工遥控轨迹
- 采一些“接近门、绕障、穿门”的成功轨迹
- 不需要太多，但要覆盖几种典型情况

这些轨迹会用于：

- 行为克隆初始化
- PPO warm-start
- 后续蒸馏 teacher

---

### Step 2.5-2：行为克隆初始化

这一步的目标不是最终最好，而是让策略一开始就“像样”。

建议：

- 输入：Phase 2.5 状态
- 输出：离散动作
- 损失：交叉熵

训练一个初始化策略：

- `policy_bc_init`

这样后面用 PPO 微调时会快很多，也更稳。

---

### Step 2.5-3：PPO 微调局部技能

这是 Phase 2.5 的核心训练阶段。

推荐：

- 算法：`PPO`
- 动作空间：离散
- 环境：静态门口 + 静态障碍物
- 初始化：从 `policy_bc_init` 开始

这一阶段重点不是“学会识别门”，而是：

- 学会根据融合状态做动作
- 学会绕障
- 学会对齐门洞
- 学会穿门

---

### Step 2.5-4：课程学习

不要一次训练所有复杂情况，建议拆成四个子阶段。

#### A. 近距离门洞对齐

- 门在前方
- 几乎无障碍
- 只训练对齐 + 前进

#### B. 静态障碍绕行

- 门前加入柱子、灌木、栏杆、箱子
- 训练绕障再回到门前

#### C. 门窗混淆

- 画面里同时出现门和窗
- 避免靠近 `window`

#### D. 稍远距离接近

- 起始点更远
- 门洞更小
- 需要先搜索再接近

建议顺序：

`A -> B -> C -> D`

---

### Step 2.5-5：冻结 teacher，准备蒸馏

当 PPO 策略足够稳定后：

- 冻结 teacher policy
- 大规模采轨迹
- 准备给 Phase 5 蒸馏

这时可以导出：

- 成功轨迹
- 失败轨迹
- 状态-动作对
- reward / progress 标签

---

## 3. 状态空间怎样设置

这一块很关键。

我的建议是：**不要让 Phase 2.5 再从原始 RGB 自己重新学语义**。

因为你已经有：

- YOLO26 结果
- 深度分析结果
- 融合决策结果

所以 Phase 2.5 最好的做法是：

- 让语义模块负责“看懂”
- 让 RL 负责“怎么动”

### 推荐状态空间

状态可以拆成三部分。

#### 3.1 深度图分支

输入一张下采样深度图，例如：

- `1 x 64 x 64`
或
- `1 x 96 x 96`

建议只保留：

- 中心区域
- 稍微左右扩展的视野

目的是保留局部几何与障碍形状。

#### 3.2 结构化语义分支

建议把下面这些标量拼起来：

- `door_conf`
- `open_door_conf`
- `window_conf`
- `close_door_conf`
- `best_entry_distance_cm`
- `best_entry_width_cm`
- `best_entry_traversable`
- `front_min_depth_cm`
- `front_obstacle_flag`
- `front_obstacle_severity`
- `matched_semantic_class_id`
- `match_score`
- `entry_center_offset_x`
- `entry_center_offset_y`

#### 3.3 局部控制分支

- `current_yaw`
- `last_action`
- `last_2_or_4_actions`
- `step_budget_ratio`
- `time_since_entry_visible`

### 推荐网络结构

建议 actor-critic 用双分支：

1. 深度图 CNN encoder
2. 结构化状态 MLP encoder

然后 concat 到 shared trunk，再分成：

- actor head
- critic head

---

## 4. Critic / value function 怎么做

如果你用 PPO，critic 本质上就是状态价值函数：

- 输入：和 actor 相同的状态
- 输出：`V(s)`

### 推荐做法

- actor 和 critic 共享前面的 encoder
- 后面分成两个 head

这样：

- 参数少
- 训练更稳
- 更适合你现在的局部技能任务

### 不建议一开始做得太复杂

例如先不要上：

- 大型 transformer
- 世界模型
- 多头复杂 value decomposition

Phase 2.5 的目标是稳定、可解释、好复现。

---

## 5. 奖励函数怎么设计

奖励函数建议按“成功、进度、安全、语义、效率”五层来写。

### 5.1 成功奖励

- 成功穿过目标门：`+15`

### 5.2 进度奖励

- 离目标门中心更近：正奖励
- 门中心更靠近图像中央：正奖励
- 成功从“远距离”进入“近距离对齐区”：额外正奖励

### 5.3 安全惩罚

- 碰撞：`-10`
- 前方近障碍还继续顶：持续负奖励
- 风险太高仍前进：惩罚

### 5.4 语义惩罚

- 靠近 `window`：负奖励
- 把 `window` 当成门去接近：额外负奖励

### 5.5 效率惩罚

- 每步小负奖励
- 左右抖动
- 原地反复转圈

一句话原则：

- 奖励靠近正确入口
- 惩罚撞障碍和错误目标
- 奖励穿门成功

---

## 6. 验证流程应该怎样做

验证不要只看训练 reward，要分成三层。

### 6.1 离线验证

对固定回放样本验证：

- 状态提取是否正确
- 策略输出动作是否合理
- critic 是否收敛

这一步主要查实现错误。

### 6.2 在线仿真验证

在静态仿真环境里跑完整 episode。

建议统计：

- `entry_pass_success_rate`
- `collision_rate`
- `mean_steps_to_cross_entry`
- `front_block_recovery_rate`
- `wrong_target_rate`
- `window_approach_error_rate`

### 6.3 泛化验证

一定要保留一部分：

- 未见过的房屋
- 未见过的障碍布局
- 未见过的光照/贴图

这一步很重要，因为它直接支撑论文里的泛化性。

---

## 7. 训练集 / 验证集怎么拆

对于 RL，这里的 “train/val” 不是普通监督学习那种纯图片拆分，而是按环境和 episode 拆。

### 推荐拆分方式

#### 训练环境

- 70%
- 用于 PPO 和 BC 训练

#### 验证环境

- 15%
- 用于调参

#### 测试环境

- 15%
- 只用于最终汇报

### 更合理的拆法

按“房屋 / 场景模板 / 障碍布局”拆，而不是按帧随机拆。

否则会数据泄漏。

---

## 8. TMM 论文里实验应该怎么设计

这一部分最重要的原则是：

**不要把实验摊得太散。**

TMM 更看重：

- 你的核心创新点是否清晰
- 实验是否能证明这个创新点
- 泛化和鲁棒性是否成立

而不是“把所有能做的都做一遍”。

---

## 9. 你这篇 TMM 的主创新点应该怎样聚焦

我建议聚焦成这三个贡献：

### Contribution 1

一个用于 UAV 房屋入户的多模态入口感知模块：

- YOLO26 语义
- 深度几何
- 融合决策

### Contribution 2

一个局部门进入与静态避障技能策略：

- 能把感知结果转成可执行动作
- 能绕障、对齐、穿门

### Contribution 3

一个从 teacher policy 到 lightweight policy 的蒸馏路线：

- 方便后面部署
- 也形成完整系统闭环

如果你这样聚焦，那么论文的主线会很清楚。

---

## 10. TMM 主实验建议

我建议主实验就做 4 组，不要再无限扩展。

### 实验 1：入口判断效果

比较：

- YOLO-only
- Depth-only
- Fusion (ours)

指标：

- entry classification accuracy
- traversability accuracy
- blocked-entry recognition accuracy

目标：

证明多模态融合比单模态更准。

---

### 实验 2：局部技能效果

比较：

- Rule-based controller
- PPO on depth only
- PPO on fusion state (ours)

指标：

- success rate
- collision rate
- steps to entry

目标：

证明融合状态驱动的局部技能比纯规则和纯深度更好。

---

### 实验 3：泛化实验

在未见过的：

- 房屋
- 障碍布局
- 光照/贴图

上测试：

- success rate
- collision rate

目标：

证明不是只记住训练场景。

---

### 实验 4：蒸馏实验

比较：

- teacher policy
- distilled policy

指标：

- success rate
- collision rate
- inference latency
- parameter count

目标：

证明蒸馏后仍能保留主要性能，且更轻量。

---

## 11. 消融实验建议

消融不要做太多，做 4 个就够。

### Ablation A：去掉深度

- 只用 YOLO 语义

### Ablation B：去掉 YOLO

- 只用深度几何

### Ablation C：去掉 front-obstacle priority

- 验证“前障碍优先”规则是否关键

### Ablation D：去掉 BC 初始化

- PPO 从零开始

这四个消融已经足够说明你的关键设计价值。

---

## 12. 是否需要把 LLM 加进比较

我的建议是：

### 主实验里不要把 LLM 放成主对比

原因：

- 你这个阶段的创新点不是 LLM
- LLM 会把主线带偏
- 审稿人会问更多和大模型无关的额外问题

### 更合理的做法

把 LLM 放成：

- 高层调度器的补充实验
或
- 论文最后的系统 demo / 扩展实验

例如你可以只做一个小实验：

- 没有 LLM：固定目标房屋
- 有 LLM：高层选择目标和子任务，底层仍用你的 skill

这样就够了，不要让 LLM 成为 Phase 2.5 的主实验核心。

一句话：

- **Phase 2.5 主实验不需要 LLM 作为主比较对象**

---

## 13. 现在这个实验够不够

如果你把论文主张限定为：

- 多模态融合入口判断
- 局部门进入与避障技能学习
- 蒸馏到轻量策略

那么：

**够了。**

后面的：

- Phase 3 跨房屋
- Phase 4 室内搜索
- 高层 LLM 调度

都可以作为：

- 系统扩展
- 附加实验
- 补充材料

不一定都要压进 TMM 主结果。

---

## 14. 推荐的论文实验结构

### 主文

1. 方法
2. 入口融合实验
3. 局部技能实验
4. 泛化实验
5. 蒸馏实验

### 补充材料

1. 跨房屋导航
2. 室内搜索
3. LLM 高层调度 demo

这样结构会更聚焦，也更像一篇强方法论文。

---

## 15. 一句话建议

你现在最好的路线是：

1. 把 Phase 2 做成稳定的融合入口判断器
2. 把 Phase 2.5 做成局部门进入与静态避障技能
3. 用 PPO + BC 初始化训练
4. 做 4 个核心实验 + 4 个消融
5. 把 LLM 放到辅助实验，而不是主线

如果这样做，论文主线会清晰很多，而且实验量也不会失控。
