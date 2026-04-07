## 论文实验计划（目标期刊：IEEE TMM）

### 论文定位

```
Title (暂定):
  "Multi-Modal Fusion for Autonomous UAV Entry Discovery
   and Indoor Person Search in Unknown Buildings"

核心卖点：
  三模态融合（YOLO + Depth + VLM）用于 UAV 自主入口发现
  → 这是 Phase 2 的核心创新，也是论文最重要的贡献

TMM 匹配度：
  ✓ 多模态融合（RGB + Depth + VLM）—— TMM 核心关注点
  ✓ 视觉感知（YOLO 检测 + depth 分析 + VLM 语义推理）
  ✓ 跨模态融合决策（三模态置信度加权 / 可学习融合）
  ✓ 完整系统（感知-决策-执行流水线）
  ✓ 蒸馏部署（多模态 → 轻量 MLP）
```

### 研究问题定义

```
问题：
  UAV 需要自主进入未知建筑物搜索人员
  → 关键瓶颈是"找到并确认可进入的入口"
  → 单一模态不够：
     · YOLO 只能检测纹理，不知道能不能过去
     · Depth 只知道几何，不理解语义
     · VLM 理解语义，但太慢不能每帧都调

提出方法：
  三模态融合框架：
  YOLO（快速检测候选入口）
  + Depth（几何验证：距离、尺寸、可穿越性）
  + VLM（语义确认：防误检、发现非标准入口）
  → 融合决策模块 → 最终入口判断

创新点（3个）：
  ① 三模态层级融合架构（YOLO快筛 → Depth验证 → VLM确认）
  ② 自适应 VLM 调用策略（不是每帧都调，而是按需调用）
  ③ 多模态感知蒸馏到轻量 MLP 用于实时部署
```

### 论文结构规划

```
1. Introduction（1.5页）
   - 问题：UAV 自主进入建筑物搜索人员
   - 挑战：入口检测需要多模态感知，单模态各有不足
   - 贡献：三模态融合框架 + 蒸馏部署 + 完整搜索系统

2. Related Work（1.5页）
   2.1 UAV Autonomous Navigation and Exploration
   2.2 Object Detection for Architectural Elements（门窗检测）
   2.3 Depth-based Traversability Analysis
   2.4 Vision-Language Models for Embodied AI
   2.5 Multi-Modal Fusion for Robotic Perception（★ 重点定位差异）

3. Problem Formulation（0.5页）
   - 任务定义 + 输入输出 + 环境设置

4. Method（4页，论文核心）
   4.1 System Overview（流水线总图，半页）
   4.2 Spatial Awareness（Phase 1 坐标定位，简写，0.5页）
   4.3 Multi-Modal Entry Discovery（★ Phase 2，主要篇幅，2页）
     4.3.1 YOLO-based Entry Candidate Detection
     4.3.2 Depth-based Geometric Verification
     4.3.3 VLM-based Semantic Confirmation
     4.3.4 Hierarchical Fusion Decision
     4.3.5 Adaptive VLM Calling Strategy
   4.4 Indoor Search and Person Detection（Phase 4，0.5页）
   4.5 Knowledge Distillation for Real-time Deployment（Phase 5，0.5页）

5. Experimental Setup（1页）
   5.1 Simulation Environment
   5.2 Dataset and Training
   5.3 Evaluation Metrics
   5.4 Baselines and Ablations

6. Results（3页，★ 审稿人重点看）
   6.1 Multi-Modal Ablation Study（★★★ 最重要）
   6.2 Fusion Strategy Comparison
   6.3 Baseline Comparison
   6.4 Full Mission Evaluation
   6.5 Generalization Study
   6.6 Distillation Efficiency

7. Discussion（0.5页）
   - VLM 调用延迟分析
   - 失败案例分析
   - 局限性

8. Conclusion（0.5页）
```

---

### 实验设计

#### 实验 1：多模态消融（★★★ 最重要，必须有）

**目的**：证明每个模态都有不可替代的贡献。这是 TMM 审稿人最看重的实验。

**必须呈现的递进关系**：YOLO alone < YOLO+Depth < YOLO+Depth+VLM

| 编号 | 方法 | 模态组合 | 说明 |
|:---:|------|---------|------|
| M1 | YOLO Only | RGB | 只用 YOLO 检测到门就进 |
| M2 | Depth Only | D | 只用 depth 找深度变化区域 |
| M3 | YOLO + Depth | RGB + D | YOLO 检测 + depth 可穿越性验证 |
| M4 | YOLO + VLM | RGB + VLM | YOLO 检测 + VLM 语义确认 |
| M5 | Depth + VLM | D + VLM | depth 几何 + VLM 语义 |
| **M6** | **YOLO + Depth + VLM (Ours)** | **RGB + D + VLM** | **完整三模态融合** |

每种组合跑的实验量：

```
3 个场景 × 5 个起始位置 × 10 次重复 = 150 episodes / 方法
6 种方法 × 150 = 共 900 episodes

每个 episode 记录：
  - entry_success: 是否成功进入正确房屋（0/1）
  - false_entry: 是否误进错误入口（0/1）
  - steps_to_entry: 从开始到进入的步数
  - collision_count: 碰撞次数
  - decision_latency_ms: 平均决策延迟
```

预期结果表格（论文 Table 2）：

```
┌────────┬───────────────┬────────────┬─────────┬──────────┬───────────┐
│ Method │ Entry Success │ False Entry│ Steps   │ Collision│ Latency   │
│        │ Rate (%) ↑    │ Rate (%) ↓ │ ↓       │ Rate ↓   │ (ms) ↓    │
├────────┼───────────────┼────────────┼─────────┼──────────┼───────────┤
│ M1     │ ~65           │ ~20        │ ~180    │ ~0.15    │ ~8        │
│ M2     │ ~40           │ ~10        │ ~250    │ ~0.08    │ ~2        │
│ M3     │ ~82           │ ~8         │ ~120    │ ~0.06    │ ~10       │
│ M4     │ ~75           │ ~5         │ ~140    │ ~0.12    │ ~500      │
│ M5     │ ~55           │ ~5         │ ~200    │ ~0.05    │ ~500      │
│ M6     │ ~92           │ ~3         │ ~95     │ ~0.04    │ ~15*      │
└────────┴───────────────┴────────────┴─────────┴──────────┴───────────┘
* VLM 仅在需要时调用，平均延迟远低于每帧调用
```

#### 实验 2：融合策略对比（建议有）

**目的**：验证融合方法的设计选择。

| 编号 | 融合策略 | 说明 |
|:---:|---------|------|
| F1 | Rule-based (weighted sum) | 手写权重（当前方案 0.4/0.35/0.25） |
| F2 | Learned MLP | 三模态特征拼接 → 3层 MLP → 决策 |
| F3 | Late fusion (concat + linear) | 拼接 → 单层线性 |
| F4 | Cross-attention fusion | 三模态 cross-attention → 决策 |

```
每种融合策略 × 150 episodes = 共 600 episodes

如果 learned fusion (F2/F4) 效果更好：
  → 论文卖点：可学习的多模态融合
如果 rule-based (F1) 效果相当：
  → 论文卖点：简单高效的融合策略，无需额外训练
两种结果都能写
```

#### 实验 3：Baseline 对比（必须有）

**目的**：证明完整系统优于现有方法。

| 编号 | Baseline | 说明 | 来源 |
|:---:|----------|------|------|
| B1 | Random Explore + YOLO | 随机绕飞 + 看到门就进 | 自己实现 |
| B2 | Frontier-based + YOLO | 基于边界的探索 + YOLO 门检测 | 经典方法改编 |
| B3 | CLIP-guided Search | CLIP 语义引导 + 图文匹配找入口 | 改编 CoW/ZSON |
| B4 | LLM-only Planning | 纯 LLM 规划每步动作（无多模态融合） | 自己实现 |
| **Ours** | **Multi-Modal Fusion** | **YOLO+Depth+VLM 融合 + 蒸馏** | — |

```
每种方法 × 150 episodes = 共 750 episodes

对比指标：
  - Entry Success Rate (ESR)
  - Full Mission Success Rate (MSR)
  - Steps to Entry (STE)
  - Collision Rate (CR)
  - Decision Latency (DL)
```

#### 实验 4：完整任务评估（必须有）

**目的**：验证从户外到室内搜索的端到端性能。

```
任务流程：
  UAV 从户外随机位置出发
  → 导航到目标房屋
  → 找入口并进入（Phase 2 核心）
  → 室内搜索人员
  → 报告结果

实验规模：
  3 个房屋 × 2 种人员配置（有人/无人）× 5 个起始位置 × 5 次重复
  = 150 episodes
```

| 指标 | 英文名 | 说明 |
|------|--------|------|
| 任务成功率 | Mission Success Rate (MSR) | 成功进入+正确判断有无人 |
| 入口发现率 | Entry Discovery Rate (EDR) | 找到可进入入口的比例 |
| 人员搜索准确率 | Person Search Accuracy (PSA) | 正确判断有/无人 |
| 平均步数 | Steps to Complete (STC) | 完成任务的效率 |
| 入口步数 | Steps to Entry (STE) | 找到入口的效率 |
| 碰撞率 | Collision Rate (CR) | 安全指标 |
| 决策延迟 | Decision Latency (DL) | 实时性指标 |
| VLM 调用次数 | VLM Calls per Episode | VLM 资源使用 |

#### 实验 5：泛化实验（建议有）

**目的**：验证系统在未见条件下的鲁棒性。

| 泛化维度 | 训练集 | 测试集 | 怎么做 |
|---------|--------|--------|--------|
| 未见房屋 | house_1, house_2, house_3 | house_4, house_5 | UE4 中新建 2 栋不同风格的房屋 |
| 光照变化 | 白天 | 黄昏 + 夜晚 | 修改 UE4 光照设置 |
| 门状态 | 全开 | 半开 + 微开 | 调整门 Actor 的旋转角度 |
| 遮挡 | 无遮挡 | 门前有花盆/灌木 | 在门前放置障碍物 Actor |

```
每个泛化维度 × 50 episodes = 共 200 episodes
记录 ESR 相比训练场景的下降幅度
```

#### 实验 6：蒸馏效率（加分项）

**目的**：验证多模态感知蒸馏到轻量 MLP 后的性能保留率。

| 对比项 | 说明 |
|--------|------|
| Teacher（完整多模态） | YOLO + Depth + VLM，实时性差 |
| Student（蒸馏 MLP） | 28维输入 → MLP → 动作，实时 |
| Performance Ratio | Student / Teacher 的性能比 |

```
目标：Student 保留 Teacher ≥ 85% 的入口发现成功率
     同时推理速度提升 10× 以上
```

---

### 实验优先级与时间规划

```
总时间估算：约 6-8 周

════════════════════════════════════════════════════════════
 P0 — 必须完成（否则投不了）                    约 4 周
════════════════════════════════════════════════════════════

Week 1：Step A-B（YOLO 训练，不需要坐标/depth）
  ├── Step A: 手动飞行截图采集 ~500 张入口图片
  │     · 只需要 RGB 图片，不需要坐标、depth
  │     · 用控制面板手动飞到门/窗附近，截图
  │
  ├── Step B: Roboflow 标注 + YOLO 训练
  │     · 标注 4 类（door_open/closed/window/opening）
  │     · 训练 YOLOv8n，验证 mAP ≥ 0.80, door_open recall ≥ 0.90
  │
  ──── YOLO 训练完成分界线 ────────────────────────────

Week 2：Step C-G（YOLO 上线后，实时运行中验证其他模块）
  ├── Step C: Depth 分析模块
  │     · YOLO 在线检测到入口时，同步获取 depth 帧分析
  │     · 在实时运行中调参（depth_ratio、obstacle 阈值等）
  │
  ├── Step D: VLM 部署与调优
  │     · 部署 LLaVA-1.5-7B（ollama）或 GPT-4o API
  │     · 自适应 VLM 调用策略（AdaptiveVLMCaller）
  │
  ├── Step E: 融合决策模块
  │     · Rule-based 融合（fuse_entry_decision）
  │     · 用实时运行中记录的数据训练 MLP/Attention 融合
  │
  └── Step F-G: 绕飞策略 + 集成测试
        · HouseCirclingExplorer 绕飞 + EntryProcedure 进入
        · 端到端跑通：绕飞 → YOLO 检测 → depth 验证 → VLM 确认 → 进入

Week 3：消融实验 + Baseline 实现
  ├── 实验 1：多模态消融（6 种组合 × 150 episodes）
  │     · 这是最重要的实验，优先跑
  │     · 跑完后确认 YOLO+Depth+VLM 确实最优
  │
  ├── 实验 3：实现 4 个 Baseline
  │     · B1 随机探索 + YOLO（最简单，先实现）
  │     · B2 Frontier + YOLO（改编经典方法）
  │     · B3 CLIP-guided（可简化实现）
  │     · B4 LLM-only（复用已有 planner）
  │
  └── 跑 Baseline 对比实验（5 × 150 episodes）

Week 4：完整任务实验
  ├── 实验 4：端到端完整任务（150 episodes）
  │     · 户外导航 → 入口发现 → 进入 → 室内搜索 → 报告
  │
  └── 整理所有实验数据，画表画图

════════════════════════════════════════════════════════════
 P1 — 强烈建议（显著提升论文质量）              约 2 周
════════════════════════════════════════════════════════════

Week 5：融合策略对比 + 泛化实验
  ├── 实验 2：融合策略对比（4 种策略 × 150 episodes）
  │     · 如果时间紧，可只做 Rule-based vs Learned MLP
  │
  └── 实验 5：泛化实验（4 维度 × 50 episodes）
        · 如果时间紧，优先做"未见房屋"和"门状态"两个维度

Week 6：蒸馏实验 + 论文写作
  ├── 实验 6：蒸馏效率对比
  │     · Teacher vs Student 性能保留率
  │
  └── 开始论文写作
        · 先写 Method 和 Experiments
        · 再写 Introduction 和 Related Work

════════════════════════════════════════════════════════════
 P2 — 加分项（有时间就做）                      约 2 周
════════════════════════════════════════════════════════════

Week 7-8：
  ├── 可学习融合模块（attention-based fusion）
  │     · 如果 Rule-based 效果已经很好，可以不做
  │
  ├── 失败案例可视化分析
  │     · 挑 5-10 个典型失败 episode
  │     · 可视化三模态各自的判断过程
  │
  ├── 域随机化实验（更多泛化维度）
  │
  └── 论文完善 + 投稿
```

---

### 实验基础设施

#### 仿真环境配置

```
引擎：UE4 + AirSim
场景：Neighborhood 场景（3 栋训练房屋 + 2 栋测试房屋）
UAV：四旋翼，前向 RGB + Depth 相机
  · RGB: 640 × 480, FOV = 90°
  · Depth: 640 × 480, 范围 50-5000cm
动作空间：8 个离散动作（forward/backward/left/right/up/down/yaw_left/yaw_right）
步长：约 20cm（移动）/ 15°（旋转）
```

#### 评估指标定义

| 指标 | 缩写 | 计算方式 |
|------|------|---------|
| Entry Success Rate | ESR | 成功进入目标房屋的 episodes / 总 episodes |
| False Entry Rate | FER | 误进非目标入口的 episodes / 总 episodes |
| Mission Success Rate | MSR | 完成全部任务（进入+搜索+正确报告）/ 总 episodes |
| Steps to Entry | STE | 从开始到穿过入口的平均步数 |
| Collision Rate | CR | 碰撞步数 / 总步数 |
| Decision Latency | DL | 融合决策平均耗时（ms） |
| VLM Utilization | VU | 每 episode 平均 VLM 调用次数 |
| Person Search Accuracy | PSA | 正确判断有/无人的比例 |

#### 域随机化方案

```
训练时随机化：
  · 房屋纹理（墙面颜色 3 种、材质 2 种）
  · 光照（白天 3 种角度）
  · 门开启角度（全开 90°/ 半开 45°）
  · 起始位置（距目标房屋 500-2000cm）

测试时额外随机化（泛化实验）：
  · 未见房屋风格
  · 黄昏/夜晚光照
  · 微开 15° / 有遮挡
  · RGB 噪声 + depth 噪声
```

---

### Baseline 实现方案

#### B1: Random Explore + YOLO

```python
# 最简单的 baseline：随机绕飞 + YOLO 看到 door_open 就进
def baseline_random_yolo(yolo_model):
    while not timeout:
        # 随机选动作（偏向 forward 和 yaw）
        action = random.choices(
            ["forward", "yaw_left", "yaw_right"],
            weights=[0.5, 0.25, 0.25]
        )[0]
        execute(action)

        # YOLO 检测
        dets = yolo_model.predict(get_frame())
        door_open = [d for d in dets if d.class == "door_open" and d.conf > 0.5]

        if door_open:
            # 对齐门中心 → forward
            align_and_enter(door_open[0])
```

#### B2: Frontier-based + YOLO

```python
# 基于可视覆盖边界的探索 + YOLO 门检测
def baseline_frontier_yolo(yolo_model, house_bbox):
    visited_angles = set()

    while not timeout:
        # 计算下一个未访问的角度
        current_angle = get_angle_to_house_center()
        next_frontier = find_next_unvisited_angle(visited_angles)

        # 绕飞到该角度
        navigate_to_angle(next_frontier)
        visited_angles.add(discretize(next_frontier))

        # YOLO 检测
        dets = yolo_model.predict(get_frame())
        if has_door_open(dets):
            align_and_enter()
```

#### B3: CLIP-guided Search

```python
# 用 CLIP 图文匹配寻找"open door"
def baseline_clip_guided(clip_model):
    text_query = "an open door of a house"

    while not timeout:
        frame = get_frame()
        similarity = clip_model.similarity(frame, text_query)

        if similarity > threshold:
            # 高相似度 → 靠近
            move_forward()
        else:
            # 低相似度 → 继续探索
            explore_next_direction()
```

#### B4: LLM-only Planning

```python
# 纯 LLM 每步规划（无多模态融合）
def baseline_llm_only(llm, frame):
    prompt = f"""You see this image from a UAV camera.
    Task: Find and enter the building through an open door.
    What action should the UAV take?
    Options: forward, yaw_left, yaw_right, up, down
    """
    action = llm.predict(frame, prompt)
    execute(action)
```

---

### 预期论文表格和图

#### Table 1: Multi-Modal Ablation（最重要的表）

```
方法            | ESR ↑  | FER ↓  | STE ↓  | CR ↓   | DL(ms) ↓
YOLO Only       | 65.3   | 20.1   | 182    | 0.15   | 8
Depth Only      | 40.7   | 10.5   | 248    | 0.08   | 2
YOLO+Depth      | 82.0   | 8.3    | 118    | 0.06   | 10
YOLO+VLM        | 74.7   | 5.2    | 142    | 0.12   | ~500
Depth+VLM       | 55.3   | 4.8    | 198    | 0.05   | ~500
Ours (Y+D+V)    | 92.0   | 3.1    | 95     | 0.04   | 15*
```

#### Table 2: Baseline Comparison

```
方法                 | ESR ↑  | MSR ↑  | STE ↓  | CR ↓
Random+YOLO          | 45.2   | 28.1   | 285    | 0.22
Frontier+YOLO        | 62.8   | 42.5   | 165    | 0.12
CLIP-guided          | 58.3   | 35.7   | 195    | 0.14
LLM-only             | 52.1   | 31.2   | 220    | 0.18
Ours                 | 92.0   | 78.5   | 95     | 0.04
```

#### Figure 要准备的图

```
Fig. 1: 系统总览图（流水线架构）
Fig. 2: Phase 2 三模态融合架构图（★ 核心图）
Fig. 3: 融合决策过程可视化
        （一个 episode 的时序：YOLO/Depth/VLM 各自判断 → 融合结果）
Fig. 4: 消融实验柱状图（6 种模态组合的 ESR 对比）
Fig. 5: Baseline 对比雷达图（ESR/MSR/STE/CR/DL 五个维度）
Fig. 6: 泛化实验折线图（性能随泛化难度的变化）
Fig. 7: 典型成功/失败案例可视化（3-4 个案例）
Fig. 8: 蒸馏前后性能对比（Teacher vs Student）
```

---

### 当前进度与下一步

```
已完成：
  ✓ Phase 0 环境搭建（UE4 + AirSim + 控制接口）
  ✓ Phase 1 地图标记方案确定（手动坐标 + 运行时定位）
  ✓ Phase 2 多模态融合方案详细设计（Step A-G 完整文档）
  ✓ Phase 2 数据采集脚本设计（EntryDataCollector）
  ✓ Phase 2 三模态模块设计（YOLO + Depth + VLM）
  ✓ Phase 2 融合策略设计（Rule-based + MLP + Attention）
  ✓ Phase 2 绕飞策略 + 入口进入流程设计
  ✓ Phase 2 自适应 VLM 调用策略设计
  ✓ YOLO building 检测器训练好（Phase 1 用的，可复用经验）
  ✓ 论文实验方案设计（6 组实验 + 4 个 Baseline）

下一步（按顺序）：
  → ① Step A: 手动飞行截图采集入口图片（~500 张，不需要坐标/depth）
  → ② Step B: Roboflow 标注 + 训练 YOLO 入口检测模型
  → ── YOLO 训练完成后，后续模块在实时运行中采集数据 ──
  → ③ Step C: 实现 depth 分析模块（YOLO 在线推理时同步获取 depth 验证）
  → ④ Step D: 部署 VLM（LLaVA/GPT-4o）+ Prompt 调优
  → ⑤ Step E: 实现融合决策模块（Rule-based + 用实时数据训练 MLP）
  → ⑥ Step F: 实现绕飞策略 + 入口进入流程，UE4 中跑通
  → ⑦ Step G: 集成测试（在线端到端）
  → ⑧ 跑消融实验（实验 1，最重要）
  → ⑨ 实现 Baseline 并跑对比实验（实验 3）
  → ⑩ 跑完整任务实验 + 泛化实验 + 蒸馏实验
  → ⑪ 写论文
```

