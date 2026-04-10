# Pilot Dataset Plan

## 1. 文档目标

这份文档用于定义 `Phase 2.5` 第一批 `pilot dataset` 的采集与验收方案。

目标不是一次性收集最终全部训练数据，而是先构建一个：

- 规模可控
- 标签清晰
- 能快速暴露 teacher / state / export 问题

的小规模样本集。

---

## 2. 为什么要先做 pilot

在当前阶段，最容易出问题的并不是模型结构，而是：

- teacher 输出不稳定
- teacher schema 漂移
- 样本缺字段
- 同类样本口径不一致
- `YOLO + depth + fusion + teacher` 之间存在冲突

如果一开始就大规模采集：

- 错误会成批进入数据集
- 后面清洗成本会很高
- student 训练结果会掩盖数据问题

所以推荐先做一批小样本集，先验证：

1. teacher 能否稳定输出
2. validator 能否稳定分级
3. entry_state_builder 能否稳定构建状态
4. dataset export 能否稳定导出 train/val

---

## 3. Pilot 的定位

pilot 数据集用于回答三个问题：

### 3.1 Teacher 是否可信

teacher 是否会：

- 把窗户当门
- 忽略前障碍
- 给出明显自相矛盾的 `subgoal/action_hint`

### 3.2 状态结构是否够用

student 输入的：

- `global_state`
- `candidates`
- `teacher_targets`

是否已经足够表达局部进入决策问题。

### 3.3 数据链是否跑通

从：

- `fusion_* / labeling`

到：

- `teacher_output.json`
- `teacher_validation.json`
- `entry_state.json`
- `train.jsonl / val.jsonl`

整条链是否已经稳定。

---

## 4. Pilot 的样本单位

第一版仍然建议使用**单帧样本**。

每条样本至少应包含：

- `rgb.png`
- `depth_cm.png`
- `depth_preview.png`
- `yolo_result.json`
- `depth_result.json`
- `fusion_result.json`
- `labeling_summary.txt`
- `teacher_output.json`
- `teacher_validation.json`
- `entry_state.json`

也就是说，pilot 的目标不是长轨迹，而是先把单帧决策样本链跑稳。

---

## 5. Pilot 的类别设计

建议第一轮优先覆盖下面 7 类。

### 5.1 正向入口类

#### `enterable_open_door`

定义：

- YOLO 明确为 `open door`
- depth ROI 支持 `traversable = true`
- 前障碍不严重

目标：

- 验证 teacher 是否能给出 `approach_entry / align_entry / cross_entry`

#### `enterable_door`

定义：

- YOLO 为 `door`
- geometry 可通过
- 但门状态不如 `open door` 明确

目标：

- 验证 teacher 是否保持比 `open_door` 更谨慎

### 5.2 被阻挡类

#### `visible_but_blocked_entry`

定义：

- 门在视野中
- 但当前不能直接通过

目标：

- 验证 teacher 是否不会看到门就直接给进入动作

#### `front_blocked_detour`

定义：

- 当前全局前障碍优先级最高

目标：

- 验证 teacher 是否优先绕行，而不是继续前冲

### 5.3 非入口类

#### `window_visible_keep_search`

定义：

- 当前显著语义目标是 `window`

目标：

- 验证 teacher 是否能稳定区分窗和门

### 5.4 不确定类

#### `geometric_opening_needs_confirmation`

定义：

- depth 看着像开口
- 但语义不足以确认是门

目标：

- 验证 teacher 是否在不确定时保持保守

#### `no_entry_confirmed`

定义：

- 当前没有可靠入口证据

目标：

- 验证 teacher 是否继续搜索，而不是胡乱指定进入目标

---

## 6. 第一轮建议样本数

第一轮 pilot 不需要很多。

建议总量：

- `24 ~ 36` 个样本

推荐分配：

- `enterable_open_door`: `5`
- `enterable_door`: `3`
- `visible_but_blocked_entry`: `4`
- `front_blocked_detour`: `5`
- `window_visible_keep_search`: `5`
- `geometric_opening_needs_confirmation`: `3`
- `no_entry_confirmed`: `3`

这版重点不是均衡到极致，而是先把最关键的错误模式覆盖到。

如果样本不够，优先保证这 4 类：

- `enterable_open_door`
- `front_blocked_detour`
- `window_visible_keep_search`
- `no_entry_confirmed`

---

## 7. 采样原则

### 7.1 不要只采“最好看”的样本

必须包含一些：

- 部分遮挡
- 角度偏斜
- 门在边缘
- 光照偏差
- 门窗相邻

否则 pilot 会过于理想化。

### 7.2 不要大量重复几乎相同的帧

如果只是轻微移动 1 到 2 帧：

- 只保留代表帧

避免：

- 数据数量看起来很多
- 实际信息量很低

### 7.3 每个类别至少覆盖 2 种视角

例如：

- 正视门
- 侧视门

或者：

- 门在图像中心
- 门在图像边缘

---

## 8. 采集来源建议

pilot 样本建议直接从你现有的：

- `phase2_multimodal_fusion_analysis/results/fusion_*/labeling/`

里人工筛选。

推荐流程：

1. 先看 `labeling_summary.txt`
2. 再看：
   - `rgb.png`
   - `depth_overlay.png`
   - `fusion_overlay.png`
3. 判断是否属于某个目标类别
4. 选入 pilot list

这样速度最快，也最贴合你现有工作流。

---

## 9. Pilot 标注要求

每个 pilot 样本至少要补全：

- `gt_entry_state`
- `gt_subgoal`
- `gt_action_hint`
- `review_notes`

建议保存到：

- `annotation_template.json`
或扩展版人工标注 JSON

### 9.1 `gt_entry_state`

必须来自固定集合。

### 9.2 `gt_subgoal`

必须来自固定集合。

### 9.3 `gt_action_hint`

必须使用粗粒度动作，不要写连续控制量。

### 9.4 `review_notes`

建议简短说明：

- 为什么这样标
- 是否有歧义
- 是否属于边界样本

---

## 10. Teacher 验收标准

对 pilot 样本，建议做 teacher 质量统计。

### 10.1 基础通过标准

teacher 至少满足：

- `valid + weak_valid` 占比足够高
- 没有大量 `invalid`

### 10.2 第一轮建议阈值

建议：

- `invalid <= 15%`
- `valid >= 50%`

如果低于这个水平，不建议直接进入 student 训练。

### 10.3 重点检查错误

最需要重点看的错误类型：

- `window -> enterable`
- `front_blocked -> forward`
- `no_entry_confirmed -> cross_entry`
- `target_candidate_id` 无意义漂移

---

## 11. State Builder 验收标准

对 pilot 样本，`entry_state_builder` 应满足：

1. 每条样本都能生成 `entry_state.json`
2. `candidates` 长度固定
3. `teacher_targets` 可嵌入
4. 缺 teacher 的样本也能正确标 `teacher_available = 0`

如果这一步不稳定，不要往下走到导出训练集。

---

## 12. Export 验收标准

对 pilot 样本，导出器应满足：

1. 能稳定生成：
   - `train.jsonl`
   - `val.jsonl`
   - `manifest.json`
2. train/val 划分可追溯
3. 类别分布统计正确
4. 每条样本都能反查回原目录

---

## 13. 推荐的 pilot 开发顺序

建议按这个顺序做：

1. 先人工挑出 `24~36` 个样本
2. 给出 `gt_entry_state / gt_subgoal / gt_action_hint`
3. 跑 teacher
4. 跑 teacher validator
5. 跑 entry_state_builder
6. 跑 dataset export
7. 检查产物是否完整

只有这些都通过后，再进入 BC 初始化训练。

---

## 14. Pilot 完成的标志

可以认为 pilot 完成，当且仅当：

1. 一批样本已经人工标完
2. teacher 输出总体可用
3. validator 已经能稳定筛掉坏样本
4. state builder 已经能稳定生成状态
5. export 已经能生成一个可训练的小数据集

只要这 5 条都成立，就说明：

- 这条 `YOLO + depth + LLM teacher + student state` 路线已经准备好进入代码实现和小规模训练。

---

## 15. 一句话总结

pilot 的目标不是“把数据采很多”，而是：

- 先验证 teacher、state、dataset 三条链都是真的可用。
