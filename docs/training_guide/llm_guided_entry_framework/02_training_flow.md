# 训练流程设计

## 1. 整体流程

建议训练流程分成 5 步：

1. 训练基础感知模块
2. 构建融合判断器
3. 构建 LLM-guided teacher
4. 训练局部技能策略
5. 蒸馏成轻量执行器

## 2. Step A：基础感知模块

### 2.1 YOLO26 语义检测

目标类别示例：

- `door`
- `open door`
- `close door`
- `window`

输出：

- bbox
- class
- confidence

### 2.2 深度分析模块

输入：

- `depth_cm.png`

输出：

- `front_obstacle.present`
- `front_obstacle.front_min_depth_cm`
- `best_entry.entry_distance_cm`
- `best_entry.opening_width_cm`
- `best_entry.traversable`

## 3. Step B：融合判断器

把 YOLO 和 depth 结果融合成高层状态：

- `enterable_open_door`
- `front_blocked_detour`
- `window_visible_keep_search`
- `no_entry_confirmed`

这一阶段建议优先使用：

- rule-based fusion

原因：

- 可解释
- 易调试
- 适合作为后续 RL 和 LLM 的基础

## 4. Step C：LLM-guided teacher

### 4.1 输入

- RGB
- depth preview
- YOLO 检测摘要
- 深度分析摘要
- 融合判断结果
- 当前位姿

### 4.2 输出

- `subgoal`
- `action_hint`
- `reason`

例如：

- `detour_left`
- `approach_entry`
- `cross_entry`
- `keep_search`

### 4.3 建议角色

LLM 不直接做最终控飞，而是：

- 作为 baseline
- 作为 teacher
- 作为 high-level guide

## 5. Step D：局部技能策略训练

### 5.1 输入状态

推荐三部分：

#### 深度图分支

- 下采样深度图，如 `64x64`

#### 结构化分支

- YOLO 标量
- depth 标量
- fusion 状态
- LLM hint 编码

#### 控制分支

- 当前 yaw
- 最近动作
- 当前步数比例

### 5.2 动作空间

建议离散动作：

- `forward_small`
- `forward_medium`
- `yaw_left_small`
- `yaw_right_small`
- `left_small`
- `right_small`
- `hold`
- `backward_small`

### 5.3 训练顺序

推荐：

1. 规则 teacher + LLM hint 采集轨迹
2. BC 初始化
3. PPO 微调

## 6. Step E：蒸馏

### 6.1 Teacher 来源

teacher 可以是：

- fusion + rule
- fusion + LLM
- PPO policy

### 6.2 Student 目标

训练一个轻量 student：

- 小 CNN + MLP
- 可部署
- 推理快

### 6.3 蒸馏输出

- 最终在线执行器
- 供高层 agent 调用

## 7. 奖励函数建议

### 正奖励

- 成功通过目标门洞
- 朝目标门靠近
- 门中心偏差减小

### 负奖励

- 碰撞
- 贴近窗口
- 高障碍前继续硬冲
- 长时间无进展

## 8. 课程学习建议

建议四阶段：

1. 门前近距离对齐
2. 静态障碍绕行
3. 门窗混淆
4. 远距离入口接近

## 9. 最终输出

这一训练流程最终应该产出三类模型：

1. `fusion model / fusion rules`
2. `LLM-guided teacher`
3. `distilled local entry policy`
