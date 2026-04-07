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

