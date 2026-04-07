# TMM 实验设计建议

## 1. 先说结论

如果你把论文主线限定为：

- 多模态入口感知
- LLM-guided 局部门进入决策
- 蒸馏到轻量执行策略

那么这条路线已经足够支撑一篇 TMM 风格论文。

关键是：

- 不要把问题做得太散
- 不要把所有 phase 都塞进主文
- 重点证明你的方法学创新点

## 2. 论文主创新点建议

建议聚焦成 3 个：

### Contribution 1

一个面向 UAV 入户的多模态入口理解模块：

- YOLO26 语义
- 深度几何
- 融合判断

### Contribution 2

一个 LLM-guided 的局部门进入与静态避障策略框架：

- 不是纯检测
- 不是纯 RL
- 不是纯 LLM

### Contribution 3

一个从 LLM-guided teacher 到轻量策略的蒸馏路线。

## 3. 主实验建议

建议主实验只做 4 组。

### 实验 1：入口判断实验

比较：

- YOLO-only
- Depth-only
- Fusion

指标：

- entrance classification accuracy
- traversability accuracy
- blocked-entry recognition accuracy

目的：

证明多模态融合比单模态更好。

### 实验 2：局部技能实验

比较：

- Rule-based
- RL on depth only
- RL on fusion state
- Fusion + LLM guidance

指标：

- success rate
- collision rate
- mean steps to entry

目的：

证明加入融合状态和 LLM guidance 后，技能策略更稳。

### 实验 3：泛化实验

在未见过的：

- 房屋
- 障碍布局
- 光照 / 贴图

上测试：

- success rate
- collision rate
- wrong-entry rate

目的：

证明方法具有泛化性。

### 实验 4：蒸馏实验

比较：

- Fusion + LLM teacher
- Distilled student

指标：

- success rate
- collision rate
- inference latency
- parameter count

目的：

证明 student 保留核心能力且更适合部署。

## 4. 消融实验建议

建议只做 5 个，足够了。

### Ablation A

去掉 depth，只保留 YOLO。

### Ablation B

去掉 YOLO，只保留 depth。

### Ablation C

去掉 LLM guidance，只保留 fusion rule。

### Ablation D

去掉 front-obstacle priority。

### Ablation E

去掉 BC 初始化，PPO 从零开始。

## 5. 是否要放 Pure LLM baseline

建议：**要放，但不要让它变成主线。**

Pure LLM baseline 可以是：

- 输入：RGB + depth preview + 几何摘要
- 输出：`search / approach / detour / cross` 或 action hint

它的意义是：

- 证明“完全靠大语言模型也能做一部分判断”
- 同时显示它的弱点：
  - 延迟
  - 不稳定
  - 不适合直接低层部署

所以 Pure LLM 更像一个：

- 强 baseline
- teacher 候选

而不是最终部署方案。

## 6. LLM 在论文里怎样放

推荐这样放：

### 方法部分

LLM 是：

- 语义判断器
- 高层局部 guide
- 蒸馏 teacher

### 实验部分

LLM 参与：

- `Pure LLM`
- `Fusion + LLM guidance`
- `Teacher -> Student distillation`

### 不建议

不要把论文主问题写成：

- “大模型直接控制无人机”

这样主线会不够稳，也容易被审稿人质疑实时性和稳定性。

## 7. 这套实验够不够

我的判断是：

**够。**

如果你把主文聚焦在：

- 感知融合
- LLM 引导
- 局部技能学习
- 蒸馏

那么已经足够构成一套完整方法论文。

## 8. 不建议塞进主文的内容

这些可以放到补充材料或系统 demo：

- 完整跨房屋大任务
- 室内完整寻人
- 全流程 end-to-end agent

它们不是不重要，而是会把 TMM 主线冲散。

## 9. 推荐的主文结构

### 主文

1. Introduction
2. Related Work
3. Method
4. Entrance Perception Experiments
5. Local Entry Skill Experiments
6. Generalization Experiments
7. Distillation Experiments
8. Conclusion

### Supplementary

1. Cross-house navigation
2. Full house search demos
3. Additional LLM case studies

## 10. 最后建议

如果你想让这篇论文更像 TMM，最重要的是：

- 聚焦自己的算法创新点
- 不要把系统做得太大太散
- 把实验收成“少而强”的结构

一句话：

> 主文讲“多模态融合 + LLM guidance + 局部技能 + 蒸馏”，其余系统能力做补充。
