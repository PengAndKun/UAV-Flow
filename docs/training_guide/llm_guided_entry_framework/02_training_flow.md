# 训练流程设计

## 1. 整体流程

建议把这一条路线拆成 5 步，而不是直接端到端硬训：

1. 训练基础感知模块
2. 构建融合判断器
3. 构建 `LLM-guided teacher`
4. 训练局部技能策略
5. 蒸馏成轻量执行器

这五步里，真正决定“这篇论文是不是成立”的关键点是：

- Step B：融合判断要稳定
- Step C：LLM teacher 要真的能提供有效 guidance
- Step D：局部策略要能把 guidance 变成动作能力

---

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

### 2.3 你当前完成情况

这一部分你已经基本完成：

- `YOLO26` 训练链已经有了
- `Phase 2 模态 2 深度分析` 已经有了
- 单张图的入口距离、开口宽度、可穿越性、前障碍判断都已经能跑

所以 Step A 基本可以认为是：

- **已完成（可继续调优，但不再是主阻塞）**

---

## 3. Step B：融合判断器

把 YOLO 和 depth 结果融合成高层状态，例如：

- `enterable_open_door`
- `front_blocked_detour`
- `window_visible_keep_search`
- `no_entry_confirmed`

### 3.1 推荐实现方式

当前阶段建议继续优先使用：

- `rule-based fusion`

原因：

- 可解释
- 易调试
- 很适合作为 RL 和 LLM 的基础 teacher

### 3.2 推荐规则优先级

建议保持：

1. 先看 `front_obstacle`
2. 再看语义类别
3. 最后看局部可穿越几何

### 3.3 你当前完成情况

这一部分你也已经基本做出来了：

- `YOLO + depth + fusion` 已经可以输出统一 `fusion_result.json`
- 你已经发现并修正了一个很关键的问题：
  - **前障碍应该优先于门洞可穿越性**

所以 Step B 当前状态可以认为是：

- **已完成第一版**
- 后面主要是继续收规则与统计验证集表现

---

## 4. Step C：LLM-guided teacher

这一部分是你现在最需要补细节的地方。

一句话先说清楚：

- 这一步不是“让 LLM 最终在线控飞”
- 而是“让 LLM 成为高层 guide 和 teacher”

### 4.1 Step C 的目标

给定：

- RGB
- depth preview
- YOLO 检测摘要
- 深度分析摘要
- 融合判断结果
- 当前位姿 / 历史

LLM 需要输出：

- 当前局部子任务 `subgoal`
- 当前动作提示 `action_hint`
- 解释 `reason`
- 可选：一个短期 waypoint hint

例如：

- `keep_search`
- `approach_entry`
- `detour_left`
- `detour_right`
- `cross_entry`
- `backoff_and_reobserve`

### 4.1.1 Step C 第一轮数据先分成哪几类

在 Step C 里，我们不是先采“完整长轨迹”，而是先采和标注一批
**静态决策样本**。每个样本都是一帧或一个很短窗口，用来回答：

- 这是不是一个可进入入口
- 现在应该靠近、绕行，还是继续搜索
- 下一步的粗粒度动作提示是什么

第一轮建议先分成下面 7 类：

1. `enterable_open_door`
   - 语义上明确是 `open door`
   - 深度几何支持通过
   - 没有严重前障碍

2. `enterable_door`
   - 语义上是 `door`
   - 几何上可通过
   - 但不一定能确认门扇已经打开

3. `visible_but_blocked_entry`
   - 能看到门或入口
   - 但当前前方障碍、门前柱子、栏杆、墙角或遮挡使得不能直接穿过

4. `front_blocked_detour`
   - 当前最关键的问题不是“有没有门”，而是“前方太堵”
   - 应优先绕行、侧移或后退重观察

5. `window_visible_keep_search`
   - 看到的是窗户或非入口立面元素
   - 不应接近并尝试进入

6. `geometric_opening_needs_confirmation`
   - 深度图显示像开口
   - 但 YOLO/语义信息不足，不能确认这是门而不是窗或其它空洞

7. `no_entry_confirmed`
   - 当前帧无法确认任何可靠入口
   - 应继续搜索外立面或换视角

如果第一轮样本不够均衡，可以先用 6 类版本：

- 把 `enterable_open_door` 和 `enterable_door` 合并成 `enterable_entry`

这样第一版更容易采够样本，也更容易先把 LLM teacher 的主决策验证清楚。

### 4.1.2 每一类数据分别解决什么问题

这些类别不是随便分的，它们对应的是 Step C 里最核心的几种错误模式：

- `enterable_open_door` / `enterable_door`
  - 验证 teacher 是否能识别“可以接近甚至准备穿过”的入口

- `visible_but_blocked_entry`
  - 验证 teacher 会不会错误地对“看见门”直接给出进入指令

- `front_blocked_detour`
  - 验证 teacher 是否把深度前障碍优先级放在第一位

- `window_visible_keep_search`
  - 验证 teacher 是否会把窗户误判成门

- `geometric_opening_needs_confirmation`
  - 验证 teacher 在语义不确定时是否会保持保守

- `no_entry_confirmed`
  - 验证 teacher 在缺少证据时是否会继续搜索而不是乱给行动建议

### 4.2 Step C 不是在学什么

Step C 不应该直接承担：

- 最终低层动作控制
- 长时间连续闭环控制
- 高频避障执行

这些属于：

- Step D 局部技能策略
- 或最终蒸馏后的 student policy

### 4.3 Step C 推荐输入格式

建议 LLM 的输入分成三部分。

#### A. 原始视觉输入

- RGB
- depth preview

说明：

- RGB 提供语义
- depth preview 提供人可读的几何轮廓

#### B. 结构化摘要

把融合结果压成文字或 JSON，建议包含：

- `front_obstacle.present`
- `front_obstacle.front_min_depth_cm`
- `front_obstacle.severity`
- `best_entry.entry_distance_cm`
- `best_entry.opening_width_cm`
- `best_entry.traversable`
- `semantic.class_name`
- `semantic.confidence`
- `fusion.final_entry_state`
- `fusion.match_score`

#### C. 控制上下文

- 当前 UAV yaw
- 过去 2 到 4 步动作
- 当前 target house id
- 当前是否已经在目标房屋前

### 4.4 Step C 推荐输出 schema

建议你把 LLM 输出强制收成固定 schema，例如：

```json
{
  "subgoal": "approach_entry",
  "action_hint": "yaw_right_small",
  "waypoint_hint": "shift_to_entry_center",
  "risk_level": "low",
  "reason": "Open door is visible and traversable; no severe front obstacle.",
  "confidence": 0.82
}
```

推荐字段：

- `subgoal`
- `action_hint`
- `waypoint_hint`
- `risk_level`
- `reason`
- `confidence`

### 4.5 Step C 实际开发顺序

建议按下面顺序做，不要一步到位：

#### C-1：先做 standalone 验证

先不要接训练环境，先做独立脚本：

- 给一张 RGB
- 给一张 depth preview
- 给一段结构化摘要
- 看 LLM 输出是否稳定

你仓库里这部分已经有基础：

- [vlm_scene_descriptor.py](/E:/github/UAV-Flow/UAV-Flow-Eval/vlm_scene_descriptor.py)
- [anthropic_vlm_scene_descriptor.py](/E:/github/UAV-Flow/UAV-Flow-Eval/anthropic_vlm_scene_descriptor.py)

所以这一步你其实已经有雏形了。

#### C-2：做 prompt 和 schema 固化

你现在最该补的是：

- 固定 Step C 专用 prompt
- 固定输出 schema
- 固定日志保存格式

也就是每次 teacher 推理都保存：

- 输入图片路径
- 输入摘要 JSON
- prompt
- raw reply
- parsed reply

#### C-3：做小规模人工验收集

找一批典型样本，人工标：

- 该不该靠近
- 该不该绕行
- 是门还是窗
- 是不是该继续搜索

然后看 LLM 输出是否符合人工判断。

第一轮人工验收集建议：

- 总量：`80-120` 个样本
- 尽量覆盖上面定义的 7 类
- 每类目标：`10-20` 个样本
- 如果样本不够，优先保证这 4 类：
  - `enterable_open_door`
  - `front_blocked_detour`
  - `window_visible_keep_search`
  - `no_entry_confirmed`

#### C-4：接入 teacher 数据采集

当 Step C 输出稳定后，把它接到 teacher 轨迹采集里。

每步保存：

- observation
- fusion state
- llm guidance
- final executed action

这样后面 BC 和蒸馏都能直接用。

### 4.6 Step C 的三种角色

建议明确区分：

#### 角色 1：Pure LLM baseline

这是实验组，不是最终部署方案。

作用：

- 证明完全靠 LLM 也能给出一部分合理判断
- 同时也暴露它的延迟和不稳定

#### 角色 2：LLM guide

这是更实用的形式。

作用：

- 给局部策略提供 `subgoal/action_hint`
- 不直接控每一步

#### 角色 3：Teacher

这是最重要的角色。

作用：

- 生成高质量轨迹标签
- 后续蒸馏成 student policy

### 4.7 Step C 的 done 标准

我建议你把 Step C 的完成条件定义成：

1. 对固定样本集，LLM 输出 schema 稳定
2. `subgoal/action_hint` 与人工判断大体一致
3. 同一类样本不会频繁反复输出冲突决策
4. 已经可以把结果写入 teacher 轨迹

只要满足这 4 条，就可以认为：

- Step C 第一版完成

### 4.8 你当前到底做到哪了

按你现在仓库状态，我的判断是：

#### 已完成

- 有 standalone VLM/LLM 推理入口
- 有 Anthropic 路线
- 有 prompt log 保存
- 有多模态 scene descriptor 雏形

#### 部分完成

- 已经能“看图 + depth preview + 输出结构化判断”
- 但还没有把它真正固化成 **Phase 2.5 teacher schema**

#### 还没完成

- 还没有形成稳定的：
  - `subgoal`
  - `action_hint`
  - `teacher dataset export`
- 还没有把 LLM guidance 系统性接入 BC / PPO 数据采集

所以结论是：

- **前面的 Step A / B 基本完成了**
- **Step C 有基础，但还没有完成成“可训练 teacher”形态**

---

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

1. `fusion rule + LLM teacher` 采集轨迹
2. BC 初始化
3. PPO 微调

---

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

---

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

---

## 8. 课程学习建议

建议四阶段：

1. 门前近距离对齐
2. 静态障碍绕行
3. 门窗混淆
4. 远距离入口接近

---

## 9. 最终输出

这一训练流程最终应该产出三类模型或模块：

1. `fusion model / fusion rules`
2. `LLM-guided teacher`
3. `distilled local entry policy`
