# UAV-Flow 完整训练流水线

## 目录

- [概览](#概览)
- [Phase 0：环境准备与数据基础设施](#phase-0环境准备与数据基础设施)
- [Phase 1：房屋感知模块训练](#phase-1房屋感知模块训练)
- [Phase 2：门检测与入口判断模块训练](#phase-2门检测与入口判断模块训练)
- [Phase 3：跨房屋导航策略训练](#phase-3跨房屋导航策略训练)
- [Phase 4：单房屋室内搜索策略训练](#phase-4单房屋室内搜索策略训练)
- [Phase 5：多房屋任务调度蒸馏](#phase-5多房屋任务调度蒸馏)
- [Phase 6：在线微调与迭代](#phase-6在线微调与迭代)
- [训练数据规范](#训练数据规范)
- [评价指标体系](#评价指标体系)
- [硬件与时间估算](#硬件与时间估算)
- [常见问题与调试策略](#常见问题与调试策略)

---

## 概览

### 任务定义

给定自然语言搜索指令（如"搜索这栋房屋并报告是否有人"），UAV 在未知多房屋户外环境中：

1. 识别目标房屋
2. 导航至目标房屋
3. 找到可进入的门
4. 进入房屋
5. 系统性搜索室内各房间
6. 判断是否有人并报告位置

### 训练总链路

```
Phase 0  环境准备 / 数据采集基础设施
  │
  ▼
Phase 1  房屋识别感知（DINOv2/CLIP 特征匹配，不训练）
  │
  ▼
Phase 2  门检测模块（YOLOv8-nano 微调）
  │
  ▼
Phase 3  跨房屋导航（规则策略 + 轨迹采集验证）
  │
  ▼
Phase 4  单房屋室内搜索（BC 行为克隆）
  │
  ▼
Phase 5  多阶段统一蒸馏（MLP / 轻量 CNN）
  │
  ▼
Phase 6  在线微调 / 数据飞轮
```

### 模块依赖关系

```
Phase 1 (房屋识别) ──┐
                     │
Phase 2 (门检测)  ──┤──► Phase 5 (统一蒸馏)
                     │
Phase 3 (跨房屋)  ──┤
                     │
Phase 4 (室内搜索) ─┘
```

Phase 1-4 之间互相独立，可以并行推进。Phase 5 依赖前四个阶段的数据。

---

## Phase 0：环境准备与数据基础设施

### 目标

搭建可重复的数据采集、存储、回放流程，确保后续所有训练数据格式统一。

### 前置条件

- Unreal Engine 4 + AirSim 仿真环境运行正常
- `uav_control_server_basic.py` 可正常启动并响应 `/state`, `/frame`, `/depth_frame`, `/move_relative`
- `uav_control_panel_basic.py` 可连接 server 进行手动遥控

### 数据采集 Bundle 格式

每一步的记录必须包含以下字段，存储为 JSONL 格式（每行一条 JSON）：

```json
{
  "episode_id": "ep_20250325_001",
  "step_index": 42,
  "timestamp": "2025-03-25T14:30:22.123",

  "pose": {
    "x": 2359.9,
    "y": 85.3,
    "z": 225.0,
    "yaw": -1.7
  },

  "action": "forward",
  "action_source": "manual",

  "depth_summary": {
    "front_min_depth": 145.2,
    "front_mean_depth": 302.8,
    "min_depth": 45.1,
    "max_depth": 1200.0
  },

  "task_label": "search house_A for people",
  "target_house_id": "house_A",

  "stage_label": "outdoor_nav",

  "perception": {
    "house_match_score": 0.82,
    "house_match_id": "house_A",
    "door_detected": false,
    "door_class": null,
    "door_confidence": 0.0,
    "door_bbox": null,
    "door_distance_cm": null,
    "person_detected": false,
    "person_confidence": 0.0
  },

  "rgb_path": "captures/ep_20250325_001/frame_00042.jpg",
  "depth_path": "captures/ep_20250325_001/depth_00042.npy"
}
```

### 阶段标签（stage_label）定义

| 标签 | 含义 | 触发条件 |
|------|------|---------|
| `outdoor_nav` | 在户外向目标房屋移动 | 任务开始，或从一栋房屋出来前往下一栋 |
| `house_circling` | 在目标房屋外围绕行找门 | 到达目标房屋附近 |
| `approaching_entry` | 已发现门，正在接近 | 门检测得分 > 0.5 且门属于目标房屋 |
| `entering` | 正在穿越门框进入室内 | 门距 < 250cm 且对齐偏差 < 15% |
| `indoor_search` | 在室内系统性搜索 | 成功进入室内 |
| `suspect_approach` | 接近可疑人员区域 | PEF 证据 > suspect 阈值 |
| `confirming` | 对可疑目标进行多角度确认 | 到达可疑区域附近 |
| `mission_complete` | 任务结束 | 确认有人 / 确认无人 / 步数耗尽 |

### 图像保存规范

- RGB 帧：JPEG 格式，640×480，质量 85%
- Depth 帧：NumPy `.npy` 格式（float32, cm 单位），同时保存伪彩色 JPEG 用于可视化
- 文件命名：`{episode_id}/frame_{step_index:05d}.jpg` 和 `{episode_id}/depth_{step_index:05d}.npy`

### 采集脚本的核心逻辑

利用现有 `uav_control_server_basic.py` 的接口：

1. `GET /state` → 获取 pose + depth_summary
2. `GET /frame` → 获取 RGB 帧
3. `GET /depth_frame` → 获取 depth 可视化帧
4. `POST /move_relative` → 执行动作（手动或脚本）
5. `POST /capture` → 触发一次完整 bundle 保存

每次 `/move_relative` 之后，间隔 300ms 再拉取下一帧（等仿真渲染完成）。

### 交付物

- [ ] 数据采集脚本 `collect_training_data.py`
- [ ] JSONL 解析与验证脚本 `validate_dataset.py`
- [ ] 采集目录结构文档

---

## Phase 1：房屋识别感知模块训练

### 目标

UAV 在飞行途中能判断"当前视野中的建筑是否是目标房屋"。

### 方法选择

**使用 DINOv2 特征匹配（零训练方案）**

不需要训练神经网络。利用 DINOv2 预训练视觉特征做图像检索：
- 每栋房屋预先采集 3-5 张参考视图（正面、侧面、45度角）
- 参考视图通过 DINOv2 编码为特征向量，存入参考向量库
- 推理时，当前 RGB 帧通过同一编码器，与参考库做余弦相似度
- 相似度最高的房屋即为当前匹配结果

### 数据采集

#### 参考图采集

对每栋房屋：

1. 飞到房屋正面，距离约 10-15m，高度约 3-5m → 截帧保存
2. 飞到房屋左侧 45° → 截帧保存
3. 飞到房屋右侧 45° → 截帧保存
4. 飞到房屋正面稍近（5-8m）→ 截帧保存
5. 飞到房屋正面稍高（8m）→ 截帧保存

每栋房屋 5 张参考图。3 栋房屋 = 15 张。

采集方法：用 panel 手动遥控到合适位置，按 Capture 按钮。

#### 测试图采集

对每栋房屋，用 sequence 功能绕飞一圈，每步保存帧：

| 序列 | 动作 | 产出 |
|------|------|------|
| `eeeeeeeeeeee` | 原地右转一整圈（12步×30°=360°） | 12 张/圈 |
| `wwwwweeeeee` | 前进 + 半圈 | 11 张 |
| `rrrwwwweeeeee` | 升高 + 前进 + 半圈 | 13 张 |

每栋房屋约 36 张测试图。3 栋房屋 = 108 张。

#### 负样本

在无房屋可见的区域（空地、树林边）采集 30 张作为 `no_house` 类。

### 验证流程

1. 把参考图过 DINOv2 建参考库
2. 把测试图逐张过 DINOv2 计算与参考库的余弦相似度
3. 统计 top-1 准确率
4. 调整相似度阈值（建议从 0.65 开始，逐步调到 0.75-0.80）

### 验证通过标准

| 指标 | 要求 |
|------|------|
| 目标房屋 top-1 准确率 | ≥ 85% |
| 非目标房屋拒绝率 | ≥ 90%（相似度 < 阈值的比例） |
| 无房屋场景拒绝率 | ≥ 95% |

### 如果 DINOv2 效果不够好

备选升级路径：

1. 换 CLIP ViT-L/14：对语言描述敏感，可以用文本辅助匹配
2. 微调 DINOv2 linear probe：冻结 backbone，只训一个线性分类头（10 分钟可训完）
3. 采集更多参考视图（从 5 张增加到 15 张，覆盖更多角度/光照）

### 交付物

- [ ] 参考图库（每栋 5 张，命名 `ref_{house_id}_{view_id}.jpg`）
- [ ] DINOv2 特征提取脚本 `build_house_reference.py`
- [ ] 匹配验证脚本 `verify_house_matching.py`
- [ ] 匹配准确率报告

---

## Phase 2：门检测与入口判断模块训练

### 目标

检测当前视野中的门，分类其类型（开门/关门/窗户），并判断是否满足进入条件。

### 第一步：YOLOv8-nano 门检测

#### 检测类别

| 类别 ID | 名称 | 含义 |
|--------|------|------|
| 0 | `door_open` | 可进入的开着的门 |
| 1 | `door_closed` | 关闭状态的门 |
| 2 | `window` | 窗户（不可进入，辅助定位） |

#### 数据采集

对每栋房屋的每个门口位置：

| 距离 | 角度范围 | 每个角度步长 | 图片数/门 |
|------|---------|------------|----------|
| 远（8-12m） | 正面 ±60° | 每 15° 一张 | 9 |
| 中（3-6m） | 正面 ±45° | 每 10° 一张 | 10 |
| 近（1-2.5m） | 正面 ±30° | 每 10° 一张 | 7 |

每个门约 26 张。假设 3 栋房屋共 6 个门 = 156 张正样本。

负样本（无门的墙面/角落）：100 张。

总计约 256 张，标注后按 8:2 分训练/验证集。

#### 标注方式

使用 Label Studio 或 Roboflow：

1. 在每张图上画 bounding box
2. 选择类别 `door_open` / `door_closed` / `window`
3. 导出为 YOLO 格式（`labels/{image_name}.txt`，每行 `class cx cy w h`）

#### 训练命令

```bash
pip install ultralytics

yolo train \
  model=yolov8n.pt \
  data=door_dataset.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  project=runs/door_detector \
  name=v1
```

`door_dataset.yaml` 内容：

```yaml
path: ./door_detection_data
train: images/train
val: images/val
names:
  0: door_open
  1: door_closed
  2: window
```

#### 训练验证标准

| 指标 | 要求 |
|------|------|
| mAP@0.5 | ≥ 0.80 |
| door_open recall | ≥ 0.85 |
| door_closed precision | ≥ 0.80 |
| 推理速度（640px） | ≤ 15ms / 帧 |

### 第二步：入口条件判断（纯规则，不训练）

门检测输出 bbox 之后，用 depth 信息做条件判断：

| 条件 | 判断方式 | 输出 |
|------|---------|------|
| 门距 > 250cm | depth 中心区域 percentile_20 | `approach`（继续靠近） |
| 门不在画面中心 | bbox 中心偏离图像中心 > 15% | `reposition`（调整偏航） |
| 门太小 | bbox 宽度 < 图像宽度 25% | `reposition`（可能角度太偏） |
| 以上全部满足 | — | `enter`（可以进入） |
| 未检测到门 | 无 door_open bbox | `explore`（继续探索找门） |

这一步不需要训练，写成确定性规则函数即可。

### 第三步：目标房屋门的验证

**关键问题**：检测到的门可能属于非目标房屋。

解决方案（结合 Phase 1）：

1. YOLO 检测到 `door_open`
2. 取门框 bbox 扩大 2 倍作为房屋外观 ROI
3. 把 ROI 区域过 DINOv2 与目标房屋参考库做匹配
4. 匹配得分 > 0.65 → 认定为目标房屋的门 → 可以进入
5. 匹配得分 < 0.65 → 非目标房屋的门 → 标记为 `wrong_house`，继续探索

### 交付物

- [ ] 门检测标注数据集（256+ 张）
- [ ] 训练好的 YOLOv8-nano 权重 `best.pt`
- [ ] 入口条件判断函数 `check_entry_condition()`
- [ ] 目标房屋门验证函数 `verify_target_house_door()`
- [ ] 检测效果可视化报告

---

## Phase 3：跨房屋导航策略训练

### 目标

UAV 能从当前位置安全飞到指定目标房屋附近（高空巡航）。

### 方法

**以几何规则策略为主，辅以 BC 数据验证。**

跨房屋导航不需要神经网络。这是一个已知目标坐标的 point-to-point 问题，规则策略足够：

#### 规则策略三阶段

```
阶段 A：爬升到巡航高度
  ├─ 当前高度 < 500cm → 执行 up
  └─ 高度 ≥ 500cm → 进入阶段 B

阶段 B：偏航对准 + 前飞
  ├─ 计算当前位置到目标房屋中心的方位角
  ├─ 当前偏航与目标方位角差 > 15° → 执行 yaw_left 或 yaw_right
  ├─ 偏航差 ≤ 15° → 执行 forward
  └─ 水平距离 < 500cm → 进入阶段 C

阶段 C：下降到搜索高度
  ├─ 当前高度 > 280cm → 执行 down
  └─ 高度 ≤ 280cm → 切换到 house_circling 阶段
```

#### 参数配置

| 参数 | 值 | 说明 |
|------|---|------|
| 巡航高度 | 500-600cm | 高于所有房屋屋顶 |
| 偏航对准容差 | 15° | 避免过度修正 |
| 到达判定半径 | 500cm | 进入房屋附近范围 |
| 搜索高度 | 250-300cm | 能看到门口的高度 |
| 每步前进距离 | 30cm | 与 server 一致 |
| 每步偏航角度 | 20° | 与 panel 一致 |

### 数据采集（用于验证和后续蒸馏）

虽然规则策略不需要训练数据，但需要采集轨迹用于：
1. 验证规则策略是否有效
2. 作为 Phase 5 蒸馏的训练数据

#### 采集方式

手动遥控从 A 房屋飞到 B 房屋，全程记录 bundle：

| 采集内容 | 说明 |
|---------|------|
| A→B 轨迹 | 10 次 |
| B→C 轨迹 | 10 次 |
| A→C 轨迹 | 10 次 |
| 总计 | 30 条轨迹 |

每条轨迹约 60-100 步。总数据量约 2400 步。

#### 记录字段

除标准 bundle 外，额外记录：

```json
{
  "stage_label": "outdoor_nav",
  "target_house_id": "house_B",
  "target_house_center": [1200.0, 500.0, 0.0],
  "distance_to_target_cm": 1842.3,
  "bearing_to_target_deg": 45.2,
  "bearing_error_deg": 12.8,
  "current_altitude_cm": 520.0
}
```

### 验证标准

| 指标 | 要求 |
|------|------|
| 到达成功率 | ≥ 95%（30条中≥28条到达目标范围） |
| 平均路径效率 | ≥ 0.70（实际路径 / 直线距离） |
| 碰撞率 | 0%（高空巡航不应有碰撞） |
| 平均飞行步数 | ≤ 120 步 |

### 交付物

- [ ] 规则导航策略 `rule_based_house_navigator.py`
- [ ] 30 条跨房屋轨迹数据
- [ ] 导航成功率验证报告

---

## Phase 4：单房屋室内搜索策略训练

### 目标

UAV 进入房屋后，系统性地搜索各房间，检测人员，报告位置。

### 方法

**行为克隆（Behavior Cloning）训练反射式搜索策略。**

### 子任务分解

室内搜索分为 3 个子行为，分别采集：

| 子行为 | 含义 | 占总数据比例 |
|--------|------|-------------|
| `room_sweep` | 在一个房间内系统扫描 | 50% |
| `room_transition` | 从一个房间移动到下一个房间 | 30% |
| `suspect_confirm` | 靠近可疑目标多角度确认 | 20% |

### 数据采集

#### room_sweep 数据

在每个房间内手动遥控执行"系统性扫描"轨迹：

扫描模式建议：
```
1. 进入房间后，先原地转一圈（eeeeeeeeeeee）获取全景
2. 向房间一侧前进，再转身扫描
3. 向另一侧前进，再转身扫描
4. 检查角落和遮挡区域
```

每个房间采集 5-8 条扫描轨迹（不同起始位置/角度）。
每栋房屋 4-6 个房间 → 每栋 20-48 条。
3 栋房屋 → 60-144 条。

#### room_transition 数据

手动遥控从一个房间穿过走廊/门框到另一个房间：

每栋房屋所有相邻房间对各采集 3-5 次 → 每栋约 15-25 条。
3 栋房屋 → 45-75 条。

#### suspect_confirm 数据

需要在仿真中放置人物模型，然后手动执行：

1. 从 3-5m 外看到人
2. 缓慢接近
3. 绕行观察不同角度
4. 在最终位置悬停确认

每个人物位置采集 3-5 条 → 每栋 4 个位置 × 4 条 → 每栋 16 条。
3 栋房屋 → 48 条。

#### 总量汇总

| 子行为 | 轨迹数 | 平均步数/条 | 总步数 |
|--------|-------|------------|--------|
| room_sweep | 100 | 50 | 5000 |
| room_transition | 60 | 30 | 1800 |
| suspect_confirm | 48 | 40 | 1920 |
| **总计** | **208** | — | **8720** |

### 训练输入特征（20维）

沿用现有 `reflex_policy_model.py` 的 `FEATURE_NAMES`，并扩展：

```
原有特征（来自 reflex_policy_model.py）：
  waypoint_distance_cm           ← 到当前航点距离
  waypoint_forward_error_cm      ← 前后偏差
  waypoint_right_error_cm        ← 左右偏差
  yaw_error_deg                  ← 偏航误差
  vertical_error_cm              ← 高度误差
  progress_to_waypoint_cm        ← 航点接近进度
  risk_score                     ← 碰撞风险
  retrieval_score                ← archive 检索得分
  front_min_depth_cm             ← 前向最近深度
  front_mean_depth_cm            ← 前向平均深度
  planner_confidence             ← 规划器置信度
  planner_sector_id              ← 方向扇区
  archive_visit_count            ← 该 cell 被访问次数
  has_retrieval                  ← 是否有检索命中
  retrieval_matches_subgoal      ← 检索是否匹配子目标

新增特征（室内搜索专用）：
  door_detected                  ← 当前帧是否检测到门（0/1）
  door_distance_cm               ← 检测到的门的距离（无门时=9999）
  person_evidence_score          ← PEF 累积证据分
  room_coverage_ratio            ← 当前房间已扫描比例（0.0-1.0）
  stage_encoding                 ← 当前阶段 one-hot 的第一维
                                   （outdoor=0, entry=1, indoor=2, confirm=3）
```

### 训练方式

使用现有 `train_reflex_policy.py` 的 MLP 模式：

```bash
python train_reflex_policy.py \
  --dataset_jsonl dataset/indoor_search_v1.jsonl \
  --output_path artifacts/indoor_search_mlp_v1.json \
  --model_type mlp \
  --hidden_dim 64 \
  --epochs 300 \
  --learning_rate 0.01 \
  --weight_decay 1e-4 \
  --class_weight_power 0.5 \
  --task_filter "search"
```

`--class_weight_power 0.5` 是关键参数：室内搜索中 `forward` 和 `yaw_right` 出现频率远高于 `up` 和 `hold`，需要平衡类别权重。

### 分阶段训练策略

不建议一次训所有子行为。分三步：

#### 步骤 1：只训 room_sweep
```bash
python train_reflex_policy.py \
  --dataset_jsonl dataset/room_sweep_only.jsonl \
  --output_path artifacts/room_sweep_mlp.json \
  --model_type mlp --hidden_dim 32 --epochs 200
```
验证：在仿真里把 UAV 放进单个房间，看是否能系统性扫描。

#### 步骤 2：加入 room_transition
```bash
python train_reflex_policy.py \
  --dataset_jsonl dataset/sweep_and_transition.jsonl \
  --output_path artifacts/sweep_transition_mlp.json \
  --model_type mlp --hidden_dim 48 --epochs 250
```
验证：在仿真里看 UAV 能否从一个房间走到另一个房间。

#### 步骤 3：加入 suspect_confirm
```bash
python train_reflex_policy.py \
  --dataset_jsonl dataset/full_indoor_search.jsonl \
  --output_path artifacts/indoor_search_mlp_v1.json \
  --model_type mlp --hidden_dim 64 --epochs 300
```
验证：放置人物模型，看 UAV 能否接近并确认。

### 验证标准

| 指标 | 要求 |
|------|------|
| 动作预测准确率（held-out） | ≥ 65% |
| 单房间扫描覆盖率 | ≥ 70%（60秒内） |
| 房间转移成功率 | ≥ 80% |
| 人员发现率（可见人） | ≥ 85% |
| 碰撞率 | ≤ 10% |

### 交付物

- [ ] 室内搜索 JSONL 数据集（8720+ 步）
- [ ] 分阶段训练好的 MLP 权重（3 个版本）
- [ ] 最终合并版权重 `indoor_search_mlp_v1.json`
- [ ] 各阶段验证报告

---

## Phase 5：多房屋任务调度蒸馏

### 目标

把 Phase 1-4 的所有模块蒸馏成一个统一的轻量网络，接受感知特征输入，输出动作 + 阶段切换。

### 与之前各 Phase 的关系

```
Phase 1 的输出 → house_match_score      ──┐
Phase 2 的输出 → door_detected, door_dist ─┤
Phase 3 的轨迹 → outdoor_nav 标签数据     ─┤──► 蒸馏网络输入
Phase 4 的轨迹 → indoor 标签数据          ─┤
server 提供     → pose, depth, risk       ─┘
```

### 蒸馏网络架构

```
输入层（24维特征向量）
   │
   ▼
隐藏层 1 (64 neurons, ReLU)
   │
   ▼
隐藏层 2 (32 neurons, ReLU)
   │
   ├──► 动作头 (8维 softmax → 8个离散动作概率)
   │
   ├──► 阶段头 (4维 softmax → 4个阶段切换概率)
   │
   └──► 安全头 (1维 sigmoid → 碰撞风险 0-1)
```

#### 输入特征定义（24维）

```
跨房屋导航特征（6维）：
  house_match_score              ← Phase 1: 当前帧与目标房屋的匹配得分
  distance_to_target_norm        ← 到目标房屋的归一化距离（/2000cm）
  bearing_error_norm             ← 方位角误差归一化（/180°）
  altitude_norm                  ← 当前高度归一化（/600cm）
  target_reached                 ← 是否已到达目标房屋范围（0/1）
  outdoor_flag                   ← 是否在室外（0/1）

门感知特征（4维）：
  door_detected                  ← 是否检测到开门（0/1）
  door_confidence                ← 检测置信度（0-1）
  door_distance_norm             ← 门距归一化（/500cm）
  door_alignment_norm            ← 门中心偏移归一化（/0.5）

室内搜索特征（6维）：
  front_min_depth_norm           ← 前向最近深度归一化（/300cm）
  front_mean_depth_norm          ← 前向平均深度归一化（/500cm）
  risk_score                     ← 碰撞风险（0-1）
  person_evidence_score_norm     ← 人员证据得分归一化（/5.0）
  room_coverage_ratio            ← 当前房间覆盖率（0-1）
  waypoint_distance_norm         ← 到当前航点距离归一化（/500cm）

导航偏差特征（4维）：
  waypoint_forward_error_norm    ← 前后偏差归一化
  waypoint_right_error_norm      ← 左右偏差归一化
  yaw_error_norm                 ← 偏航误差归一化（/180°）
  vertical_error_norm            ← 高度误差归一化（/100cm）

阶段编码（4维 one-hot）：
  is_outdoor_nav                 ← 1 if stage == outdoor_nav
  is_approaching_entry           ← 1 if stage == approaching_entry
  is_entering                    ← 1 if stage == entering
  is_indoor_search               ← 1 if stage == indoor_search
```

### 训练数据构建

从 Phase 3 和 Phase 4 的采集数据中，按阶段标签统一构建：

| 数据来源 | 阶段标签 | 轨迹数 | 步数 | 占比 |
|---------|---------|-------|------|------|
| Phase 3 跨房屋飞行 | outdoor_nav | 30 | 2400 | 22% |
| Phase 2 进门轨迹 | approaching_entry + entering | 50 | 1500 | 14% |
| Phase 4 room_sweep | indoor_search | 100 | 5000 | 45% |
| Phase 4 room_transition | indoor_search | 60 | 1800 | 16% |
| Phase 4 suspect_confirm | indoor_search | 48 | 300 | 3% |
| **总计** | — | **288** | **11000** | **100%** |

### 训练目标

```
L_total = L_action + 0.3 × L_stage + 0.5 × L_safety

L_action = CrossEntropy(predicted_action, expert_action)
L_stage  = CrossEntropy(predicted_stage, true_stage_label)
L_safety = BCE(predicted_risk, depth_derived_risk_label)
```

其中 `depth_derived_risk_label = 1 if front_min_depth < 80cm else 0`

### 训练超参数

| 参数 | 值 |
|------|---|
| 优化器 | Adam |
| 学习率 | 0.005 |
| 权重衰减 | 1e-4 |
| 批大小 | 128 |
| 训练轮数 | 500 |
| 类别权重 | inverse-sqrt frequency |
| 训练/验证分割 | 80% / 20%（按 episode 分割，不是按步分割） |

### 分步训练策略

```
第1步：只用 outdoor_nav 数据训练
  → 验证动作头能否输出正确的朝目标飞行动作
  → 验证阶段头是否稳定输出 outdoor_nav

第2步：加入 approaching_entry + entering 数据
  → 验证阶段切换是否在门检测得分上升时触发
  → 验证动作头是否从"朝目标飞"切换到"对准门前进"

第3步：加入所有 indoor_search 数据
  → 验证阶段切换在进入门后是否触发
  → 验证扫描行为是否保留
  → 检查类别不平衡问题（forward 可能占 60%+，需要降权）

第4步：全数据联合训练
  → 调整 class_weight_power 使稀有动作（up, down, hold）不被忽略
  → 观察各阶段的动作分布是否合理
```

### 验证标准

| 指标 | 要求 |
|------|------|
| 动作预测整体准确率 | ≥ 60% |
| outdoor_nav 阶段动作准确率 | ≥ 75% |
| 阶段切换准确率 | ≥ 80% |
| 安全头 AUC | ≥ 0.85 |
| 推理延迟 | ≤ 5ms / 步 |

### 交付物

- [ ] 统一蒸馏数据集 `distillation_dataset_v1.jsonl`（11000+ 步）
- [ ] 蒸馏网络权重 `unified_policy_v1.json`
- [ ] 分步训练日志与损失曲线
- [ ] 各阶段验证报告

---

## Phase 6：在线微调与迭代

### 目标

用仿真中的成功 episode 数据持续改进策略。

### 数据飞轮机制

```
运行策略 → 收集 episode → 标记成功/失败
                │
                ├─ 成功 episode → 加入训练集
                │
                └─ 失败 episode → 分析失败原因
                      │
                      ├─ 碰撞 → 补采该场景的安全飞行数据
                      ├─ 找不到门 → 补采该房屋的门搜索数据
                      ├─ 进错房 → 补采该房屋的参考视图
                      └─ 漏检人 → 补采该位置的搜索确认数据
```

### 迭代周期

```
每轮迭代（约 1 天）：
  ├─ 在仿真中运行 20 个 episode
  ├─ 标记成功/失败
  ├─ 成功 episode 加入训练集（预期 60-80% 成功率后约 12-16 条）
  ├─ 失败 episode 按原因归类
  ├─ 对失败原因最多的类别补采 10-15 条数据
  ├─ 重新训练 Phase 5 蒸馏网络（约 30 分钟）
  └─ 验证新模型是否改善
```

### 停止条件

当连续 3 轮迭代的以下指标均满足时，认为训练收敛：

| 指标 | 收敛标准 |
|------|---------|
| 单房屋搜索成功率 | ≥ 75% |
| 多房屋任务完成率 | ≥ 65% |
| 碰撞率 | ≤ 8% |
| 人员发现率（有人时） | ≥ 80% |

---

## 训练数据规范

### 目录结构

```
UAV-Flow/
└── training_data/
    ├── house_references/                ← Phase 1: 参考图
    │   ├── house_A/
    │   │   ├── ref_front.jpg
    │   │   ├── ref_left45.jpg
    │   │   ├── ref_right45.jpg
    │   │   ├── ref_close.jpg
    │   │   └── ref_high.jpg
    │   ├── house_B/
    │   └── house_C/
    │
    ├── door_detection/                  ← Phase 2: 门检测标注
    │   ├── images/
    │   │   ├── train/
    │   │   └── val/
    │   ├── labels/
    │   │   ├── train/
    │   │   └── val/
    │   └── door_dataset.yaml
    │
    ├── trajectories/                    ← Phase 3-4: 轨迹数据
    │   ├── outdoor_nav/
    │   │   ├── ep_001.jsonl
    │   │   └── ...
    │   ├── entry/
    │   │   ├── ep_001.jsonl
    │   │   └── ...
    │   └── indoor_search/
    │       ├── ep_001.jsonl
    │       └── ...
    │
    ├── distillation/                    ← Phase 5: 蒸馏数据
    │   ├── distillation_dataset_v1.jsonl
    │   └── distillation_dataset_v2.jsonl
    │
    └── artifacts/                       ← 训练产物
        ├── house_reference_features.npy
        ├── door_detector_v1/
        │   └── best.pt
        ├── indoor_search_mlp_v1.json
        └── unified_policy_v1.json
```

### 数据版本管理

每次重大数据变更时创建新版本：

```
distillation_dataset_v1.jsonl  ← 初始版本
distillation_dataset_v2.jsonl  ← 加入第一轮迭代数据
distillation_dataset_v3.jsonl  ← 加入第二轮迭代数据
```

旧版本不删除，用于对比实验。

---

## 评价指标体系

### 模块级指标

| 模块 | 指标 | 计算方式 |
|------|------|---------|
| 房屋识别 | top-1 准确率 | 正确匹配 / 总测试帧数 |
| 房屋识别 | 拒绝率 | 非目标帧中 score < 阈值的比例 |
| 门检测 | mAP@0.5 | YOLO 标准评估 |
| 门检测 | door_open recall | 开门被检测到的比例 |
| 跨房屋导航 | 到达成功率 | 到达目标范围 / 总尝试数 |
| 跨房屋导航 | 路径效率 | 直线距离 / 实际飞行路径长度 |
| 室内搜索 | 房间覆盖率 | 已扫描面积 / 房间总面积 |
| 室内搜索 | 人员发现率 | 正确发现 / 实际有人的 episode 数 |

### 系统级指标（论文主实验用）

| 指标 | 缩写 | 说明 |
|------|------|------|
| Search Success Rate | SSR | 正确判断有/无人 + LE ≤ 1m |
| Localization Error | LE | 估计位置与真实位置的欧氏距离 |
| Time-to-First-Detection | TFD | 任务开始到首次有效检测的时间 |
| Collision Rate | CR | 碰撞步数 / 总步数 |
| Decision Latency | DL | 每步平均推理延迟 |
| Token Usage | TU | LLM 调用的 token 总量（仅限使用 LLM 的方法） |
| House Identification Accuracy | HIA | 多房屋场景下目标房屋识别准确率 |
| Entry Success Rate | ESR | 成功进入目标房屋 / 总尝试数 |

---

## 硬件与时间估算

### 硬件需求

| 设备 | 用途 | 最低配置 |
|------|------|---------|
| GPU | YOLOv8 训练 + MLP 训练 | GTX 1080 / RTX 3060 |
| GPU（可选） | DINOv2 特征提取 | 同上（CPU 也可但慢 10 倍） |
| GPU（可选） | VLM 推理（Phase 6+） | RTX 4090（16GB+） |
| CPU | 仿真运行 + 规则策略 | 8核+ |
| 内存 | 仿真 + 数据加载 | 16GB+ |
| 存储 | 图像 + 轨迹数据 | 50GB 可用空间 |

### 时间估算

| Phase | 数据采集 | 标注 | 训练 | 验证 | 总计 |
|-------|---------|------|------|------|------|
| Phase 0 | 2h（写脚本） | — | — | 1h | 3h |
| Phase 1 | 2h | — | 无（零训练） | 1h | 3h |
| Phase 2 | 3h | 3h | 0.5h | 1h | 7.5h |
| Phase 3 | 3h | 0.5h | 无 | 1h | 4.5h |
| Phase 4 | 8h | 2h | 1h | 2h | 13h |
| Phase 5 | 1h（构建数据） | — | 0.5h | 2h | 3.5h |
| Phase 6 | 持续 | — | 持续 | 持续 | 每轮约 4h |
| **总计（到首次完整运行）** | | | | | **约 34.5h（≈ 1 周）** |

---

## 常见问题与调试策略

### Q1：门检测 mAP 太低（< 0.70）

**可能原因**：
- 数据量不够 → 增加到 400 张+
- 开门/关门分界不清 → 检查标注一致性
- 距离变化太大 → 分近/中/远三个子集，检查各子集的 recall

**解决方案**：
- 加入数据增强（mosaic, mixup, HSV 抖动）
- 增加训练轮数到 200
- 换 YOLOv8s（稍大模型）

### Q2：房屋匹配出现误判

**可能原因**：
- 两栋房屋外观太相似 → 需要更多角度的参考图
- 光照变化大 → 采集不同光照条件的参考图

**解决方案**：
- 增加参考图到每栋 10 张
- 在参考图中加入不同光照/时间段的帧
- 如仍不够，对 DINOv2 做 linear probe 微调

### Q3：蒸馏网络阶段切换不准

**可能原因**：
- 阶段之间的转换数据太少 → 阶段边界帧被淹没在大量稳态帧中
- one-hot 编码导致模型对阶段输入过度依赖

**解决方案**：
- 对阶段切换发生的前后 5 步做过采样（2-3 倍重复）
- 阶段头的 loss 权重从 0.3 增加到 0.5
- 检查阶段标签是否标注准确

### Q4：室内搜索老是碰墙

**可能原因**：
- `risk_score` 和 `front_min_depth` 特征没有足够权重
- 安全头的训练数据中碰撞样本太少

**解决方案**：
- 增加 `L_safety` 的权重到 0.8
- 在碰撞多发区域（门框边缘、窄走廊）专门补采数据
- 加入运行时规则：`if front_min_depth < 60cm → override action to hold`

### Q5：forward 动作占比过高，模型几乎只输出 forward

**可能原因**：
- 训练数据中 forward 占 55-65%，模型学到了"无脑前进"的 shortcut

**解决方案**：
- `class_weight_power` 设为 0.5-0.7
- 对 forward 做下采样（每条轨迹中连续 forward > 5 步的部分随机丢弃 30%）
- 对 yaw_left/yaw_right/hold 做上采样

---

## 附录：关键文件对照

| 训练阶段 | 使用的现有文件 | 新建文件 |
|---------|--------------|---------|
| Phase 0 数据采集 | `uav_control_server_basic.py`, `uav_control_panel_basic.py` | `collect_training_data.py` |
| Phase 1 房屋匹配 | — | `build_house_reference.py`, `verify_house_matching.py` |
| Phase 2 门检测 | — | YOLOv8 训练配置 `door_dataset.yaml` |
| Phase 3 跨房屋导航 | — | `rule_based_house_navigator.py` |
| Phase 4 室内搜索 | `reflex_policy_model.py`, `train_reflex_policy.py`, `reflex_dataset_builder.py` | 扩展 `FEATURE_NAMES` |
| Phase 5 蒸馏 | `train_reflex_policy.py`（扩展 MLP 架构） | `build_distillation_dataset.py` |
| Phase 6 迭代 | `online_reflex_eval.py` | `iteration_manager.py` |
