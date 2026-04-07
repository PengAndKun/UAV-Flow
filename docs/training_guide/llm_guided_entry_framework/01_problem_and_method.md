# 问题定义与方法设计

## 1. 论文问题定义

推荐把论文问题定义成：

> 如何利用多模态感知与 LLM 语义推理，指导 UAV 在复杂户外住宅环境中识别可进入入口、决定绕行或接近，并将这种高层推理蒸馏成可实时执行的轻量局部技能策略。

这个问题比“单纯让 PPO 学会穿门”更强，也比“纯 LLM 看图做动作”更工程化、更可部署。

## 2. 系统结构

系统可以拆成四层：

### 2.1 地图与目标层

输入：

- 目标房屋 ID
- 地图定位结果
- 当前 UAV 位姿

作用：

- 判断当前是不是在目标房屋附近
- 避免跑进错误房屋

### 2.2 多模态感知层

输入：

- RGB
- 深度图

模块：

- `YOLO26`：门/窗/开口语义检测
- 深度几何分析：距离、开口宽度、可穿越性、前方障碍
- 融合模块：输出入口状态

输出示例：

- `enterable_open_door`
- `front_blocked_detour`
- `window_visible_keep_search`
- `geometric_opening_needs_confirmation`

### 2.3 LLM/VLM 引导层

输入：

- RGB
- depth preview
- 几何摘要
- YOLO 检测结果
- 当前位姿与历史

输出：

- 当前子任务
  - `search_entry`
  - `approach_entry`
  - `detour_obstacle`
  - `cross_entry`
- 动作提示或 waypoint hint
- 文字解释

### 2.4 局部技能执行层

输入：

- 融合状态
- LLM guidance
- 深度图局部观测

输出：

- 离散动作

例如：

- `forward_small`
- `yaw_left_small`
- `yaw_right_small`
- `left_small`
- `right_small`
- `hold`

## 3. 为什么要引入 LLM

LLM 在这里不只是一个 demo，而是有清晰角色：

### 3.1 作为语义判断器

它擅长处理：

- 这是门还是窗
- 当前是不是在门前
- 现在更该绕行还是继续观察

### 3.2 作为高层 guide

它可以输出：

- 当前子任务
- 下一步局部目标
- 动作提示

### 3.3 作为 teacher

训练时可提供：

- `action_hint`
- `subgoal_hint`
- `reason`

这样比纯 RL 更快、更稳定。

## 4. 为什么最终还要蒸馏

如果完全在线依赖 LLM，会有几个问题：

- 延迟高
- 不稳定
- 成本高
- 实时控飞不合适

所以更合理的最终方案是：

- `LLM` 用于 teacher / guide / baseline
- `distilled policy` 用于最终实时执行

## 5. 与已有阶段的关系

这条方法线可以看成：

- Phase 2：多模态入口感知
- Phase 2.5：局部门进入技能
- Phase 5：蒸馏

也就是说：

- Phase 2 负责“看懂”
- 新方法负责“决定”
- 蒸馏策略负责“快速执行”

## 6. 方法主张

建议论文主张集中在：

1. 仅靠 YOLO 或仅靠深度都不够
2. LLM 可以提升入口语义判断与局部子任务选择
3. 但最终执行仍应由轻量策略完成

一句话：

> We use LLM as a high-level multimodal teacher, not as the final low-level controller.
