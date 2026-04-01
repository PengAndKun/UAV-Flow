# UAV-Flow 完整训练流水线

## 目录

- [概览](#概览)
- [Phase 0：环境准备与数据基础设施](#phase-0环境准备与数据基础设施)
- [Phase 1：房屋确认与地图标记](#phase-1房屋确认与地图标记)
- [Phase 2：入口探索](#phase-2入口探索找到目标房屋的可进入入口)
- [Phase 3：跨房屋导航策略训练](#phase-3跨房屋导航策略训练)
- [Phase 4：单房屋室内搜索策略训练](#phase-4单房屋室内搜索策略训练)
- [Phase 5：多房屋任务调度蒸馏](#phase-5多房屋任务调度蒸馏)
- [Phase 6：在线微调与迭代](#phase-6在线微调与迭代)
- [训练数据规范](#训练数据规范)
- [评价指标体系](#评价指标体系)
- [硬件与时间估算](#硬件与时间估算)
- [常见问题与调试策略](#常见问题与调试策略)
- [论文实验计划](#论文实验计划)
- [Planner-Driven 自主探索执行器设计](#planner-driven-自主探索执行器设计)
- [系统升级路线图](#系统升级路线图)

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
Phase 1  地图标记 + 坐标定位（手动标房屋坐标 + 运行时坐标判断）
  │
  ▼
Phase 2  入口探索（★ 重头戏：多模态融合检测可进入入口）
  │       YOLO 检测 + depth 深度分析 + VLM 语义理解
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
Phase 1 (地图标记) ──► Phase 2 (★ 入口探索：多模态融合) ──┐
                                                            │
Phase 3 (跨房屋导航) ──────────────────────────────────┤──► Phase 5 (统一蒸馏)
                                                            │
Phase 4 (室内搜索) ───────────────────────────────────┘
```

Phase 1 很轻量（手动标坐标即可），Phase 2 是核心（多模态融合检测入口）。Phase 3、Phase 4 与 Phase 2 并行推进。Phase 5 依赖前四个阶段的数据。

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

  "task_label": "search house_1 for people",
  "target_house_id": "house_1",

  "stage_label": "outdoor_nav",

  "location": {
    "current_house_id": null,
    "is_at_target_house": false,
    "is_at_wrong_house": false,
    "is_inside": false,
    "distance_to_target_cm": 1842.3,
    "bearing_to_target_deg": 45.2,
    "bearing_error_deg": 12.8,
    "target_approach_progress": 0.23
  },

  "perception": {
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

## Phase 1：地图标记与坐标定位（轻量化）

### 目标

**Phase 1 只做地图层面的事情：手动标记房屋坐标 + 运行时坐标定位。**

Phase 1 不做任何视觉识别——不训练 YOLO building 检测器、不做视觉-地图对齐、不做 VLM 描述。所有视觉感知能力集中在 Phase 2（入口探索）。

Phase 1 结束时系统知道的是：
- 场景中有 N 栋房屋，每栋在 UE4 世界坐标系中的位置和范围
- UAV 每步能通过坐标判断"我在哪栋房屋附近/内部"

### 整体流程

```
Phase 1 分两步：

Step 1. 手动标记房屋坐标（Map Marking）
  在 UE4 编辑器中读取每栋房屋的坐标
  → 手动填写 houses_config.json
  → 耗时约 10-15 分钟

Step 2. 运行时坐标定位（Runtime Localization）
  正式搜索时，每步用 UAV 的 (x, y, z) 坐标判断：
  → 我在哪栋房屋附近？
  → 我在室内还是室外？
  → 我到目标房屋还有多远？

Step 3. 搜索状态管理（Search Status Tracking）
  搜索完一栋 → 标记 explored → 永不重复进入
  → 自动选下一栋 unexplored 房屋
```

---

### Step 1：手动标记房屋坐标——生成 houses_config.json

#### 1.1 操作方法

在 UE4 编辑器中读取每栋房屋的坐标，手动写入配置文件。

```
操作步骤（约 10-15 分钟）：

1. 打开 UE4 编辑器
2. 在场景中找到每栋房屋的 Actor（World Outliner 搜索）
3. 选中房屋 Actor → Details 面板 → Transform → Location
   读取 X, Y 值
4. 估算房屋的宽度和长度（或查看 Actor 的 BoundingBox 组件）
5. 手动填入 houses_config.json

示例：
  house_1: X=2400, Y=100   → center=[2400, 100]
  house_2: X=1200, Y=500   → center=[1200, 500]
  house_3: X=800,  Y=-300  → center=[800, -300]

  每栋房屋约 600cm × 600cm → center ± 300cm = bbox
```

#### 1.2 `houses_config.json` 完整格式

```json
{
  "map_info": {
    "name": "neighborhood_scene_v1",
    "coordinate_system": "unreal_engine_cm",
    "origin_note": "UE4 world origin, units in cm"
  },

  "houses": [
    {
      "house_id": "house_1",
      "center": [2400.0, 100.0],
      "bbox": {
        "x_min": 2100.0, "x_max": 2700.0,
        "y_min": -200.0, "y_max": 400.0
      },
      "approach_radius_cm": 800.0,
      "floor_z_range": [200.0, 350.0],
      "search_status": "unexplored"
    },
    {
      "house_id": "house_2",
      "center": [1200.0, 500.0],
      "bbox": {
        "x_min": 900.0, "x_max": 1500.0,
        "y_min": 250.0, "y_max": 750.0
      },
      "approach_radius_cm": 800.0,
      "floor_z_range": [200.0, 350.0],
      "search_status": "unexplored"
    },
    {
      "house_id": "house_3",
      "center": [800.0, -300.0],
      "bbox": {
        "x_min": 500.0, "x_max": 1100.0,
        "y_min": -600.0, "y_max": 0.0
      },
      "approach_radius_cm": 800.0,
      "floor_z_range": [200.0, 350.0],
      "search_status": "unexplored"
    }
  ],

  "search_altitude_cm": 270.0,
  "global_step_budget": 2000
}
```

**关键字段说明**：

| 字段 | 说明 | 怎么填 |
|------|------|--------|
| `center` | 房屋中心 UE4 坐标 [x, y] | 从编辑器 Transform 读取 |
| `bbox` | 房屋占地范围 | center ± 300cm（大致即可，±100cm 误差无影响） |
| `approach_radius_cm` | 接近范围（进入此范围触发 Phase 2） | 800cm（可调） |
| `floor_z_range` | 室内地面高度范围 | 飞进房屋后读 /state 的 z 值确定 |
| `search_status` | 搜索状态 | 初始填 "unexplored" |

#### 1.3 UAV 位姿来源

UAV 的位置和朝向通过 `/state` 接口获取，**精确已知**，无需额外标定：

```json
{
  "pose": {
    "x": 2359.9,
    "y": 85.3,
    "z": 225.0,
    "yaw": -1.7
  }
}
```

#### 1.4 交付物

- [ ] `houses_config.json`（包含所有房屋的 bbox 坐标）

---

### Step 2：运行时坐标定位——每步知道"我在哪"

纯坐标判断，不需要任何视觉识别。每步从 `/state` 获取 UAV 位姿，和 `houses_config.json` 中的 bbox 做比较。

```
每步判断逻辑（≈ 0ms，确定性）：

  pose (x,y) 落在哪个 house bbox 内 → current_house_id
  pose (x,y) 不在任何 bbox 内 → "outdoor"
  pose.z 是否在 floor_z_range 内 → is_inside
  与 target_house 中心的距离和方位角 → 导航用
  distance < approach_radius → 触发 Phase 2 入口探索
```

#### 2.1 坐标定位函数

```python
def locate_uav(uav_pose, houses_config, target_house_id):
    """Phase 1 运行时定位：纯坐标判断。"""
    x, y, z = uav_pose["x"], uav_pose["y"], uav_pose["z"]

    current_house_id = None
    is_inside = False
    is_at_wrong_house = False

    for house in houses_config["houses"]:
        bbox = house["bbox"]
        if bbox["x_min"] <= x <= bbox["x_max"] and bbox["y_min"] <= y <= bbox["y_max"]:
            current_house_id = house["house_id"]
            floor_range = house.get("floor_z_range", [200, 350])
            is_inside = floor_range[0] <= z <= floor_range[1]
            break

    if current_house_id and current_house_id != target_house_id:
        is_at_wrong_house = True

    # 到目标房屋的距离和方位
    target = next(h for h in houses_config["houses"]
                  if h["house_id"] == target_house_id)
    tc = target["center"]
    dx, dy = tc[0] - x, tc[1] - y
    distance = math.sqrt(dx**2 + dy**2)
    bearing = math.degrees(math.atan2(dy, dx))
    bearing_error = normalize_angle(bearing - uav_pose["yaw"])

    return {
        "current_house_id": current_house_id,
        "is_at_target_house": current_house_id == target_house_id,
        "is_at_wrong_house": is_at_wrong_house,
        "is_inside": is_inside,
        "distance_to_target_cm": distance,
        "bearing_to_target_deg": bearing,
        "bearing_error_deg": bearing_error,
        "target_approach_progress": max(0, 1 - distance / 2000)
    }
```

#### 2.2 "在错误房屋"的处理

```
场景：UAV 飞向 house_1，途中经过 house_2 附近

判断流程：
  1. 坐标判断：当前在 house_2 bbox 内
  2. 目标检查：target_house_id == house_1
  3. 结论：at_wrong_house = true

处理：
  ├─ 不触发任何入口探索（Phase 2 的逻辑不会启动）
  ├─ 日志记录 "at_wrong_house: house_2"
  ├─ 继续 outdoor_nav 飞向 house_1
  └─ 只有到达 house_1 附近后才启动 Phase 2 的入口探索
```

---

### Step 3：搜索状态管理——防止重复探索

#### 3.1 状态流转

```
任务开始
  │
  ▼
所有房屋 = pending
  │
  Phase 1 扫描
  │
  ▼
confirmed 的房屋 → search_status = "unexplored"
未确认的房屋 → 保持 "pending"（等补扫）
  │
  选最近的 unexplored 作为 target
  │
  ▼
unexplored ──(UAV 进入该房屋)──► in_progress
                                      │
                      ┌───────────────┤
                      │               │
                      ▼               ▼
             (搜完+确认无人)    (发现人员)
                      │               │
                      ▼               ▼
                 explored        person_found
                      │               │
                      ▼               ▼
              标记绿色            标记红色
              永不再进入          记录位置
              选下一栋            任务可终止
```

#### 3.2 搜索完成条件

标记 `explored` 需满足全部条件：

| 条件 | 判断方式 | 原因 |
|------|---------|------|
| 曾进入室内 | `is_inside` 持续 ≥ 30 步 | 确保真正进了房屋 |
| 覆盖率达标 | 累积转向角 ≥ 720° | 至少转了两整圈 |
| 无人员证据 | PEF σ < 0.5 | 多帧融合确认无人 |
| 已退出 | `is_inside = false` | 已经出来了 |

发现人员（PEF σ ≥ 3.0）→ 直接 `person_found`。

#### 3.3 搜索状态管理模块

```python
class SearchStatusManager:
    """管理所有房屋的搜索状态。"""

    def __init__(self, houses_config):
        self.houses = {h["house_id"]: h for h in houses_config["houses"]}
        self.indoor_step_counts = {}
        self.yaw_accumulation = {}

    def update(self, house_id, is_inside, yaw_delta, pef_score):
        """每步调用。"""
        house = self.houses[house_id]
        status = house["search_status"]

        if status in ("pending", "confirmed"):
            return

        if status == "unexplored" and is_inside:
            house["search_status"] = "in_progress"
            self.indoor_step_counts[house_id] = 0
            self.yaw_accumulation[house_id] = 0

        if status == "in_progress":
            if is_inside:
                self.indoor_step_counts[house_id] += 1
                self.yaw_accumulation[house_id] += abs(yaw_delta)

            if pef_score >= 3.0:
                house["search_status"] = "person_found"
                return

            if (not is_inside
                and self.indoor_step_counts.get(house_id, 0) >= 30
                and self.yaw_accumulation.get(house_id, 0) >= 720
                and pef_score < 0.5):
                house["search_status"] = "explored"

    def get_next_target(self, current_x, current_y):
        """选最近的 unexplored 房屋。"""
        candidates = [
            h for h in self.houses.values()
            if h["search_status"] == "unexplored"
            and h.get("recon_status") == "confirmed"
        ]
        if not candidates:
            return None

        def dist(h):
            c = h["center"]
            return math.sqrt((c[0] - current_x)**2 + (c[1] - current_y)**2)

        return min(candidates, key=dist)["house_id"]

    def is_mission_complete(self):
        """全部搜完或发现人？"""
        return all(
            h["search_status"] in ("explored", "person_found")
            for h in self.houses.values()
        )
```

#### 3.4 俯视图地图

| 状态 | 颜色 | 边框 | 含义 |
|------|------|------|------|
| `pending` | 深灰色 | 虚线 | 尚未确认 |
| `unexplored` | 浅灰色 | 实线 | 已确认，待搜索 |
| `in_progress` | 黄色 | 粗实线 | 正在搜索中 |
| `explored` | 绿色 | 实线 | 已搜索，无人 |
| `person_found` | 红色 | 粗实线+闪烁 | 发现人员 |

地图上显示：
- 每栋房屋的 VLM 描述
- UAV 位置和朝向箭头
- 到目标房屋的航向线

#### 3.5 多房屋调度

```
1. Phase 1 扫描完成 → 所有 confirmed 房屋变为 unexplored
2. 选最近的 unexplored → target_house_id
3. Phase 3 飞过去
4. Phase 2 找入口 → 进入
5. Phase 4 室内搜索
6. 搜完 → explored / person_found
7. 回到步骤 2

终止：所有房屋搜完 / 步数耗尽 / 发现人员立即报告
```

---

### 蒸馏特征输出

Phase 1 为 Phase 5 蒸馏提供以下特征：

```
坐标定位特征（6维，全部确定性）：
  distance_to_target_norm        ← 到目标中心归一化距离（/2000cm）
  bearing_error_norm             ← 方位角误差归一化（/180°）
  is_at_target_house             ← 在目标房屋范围内（0/1）
  is_at_wrong_house              ← 在非目标房屋范围内（0/1）
  is_inside                      ← 在室内（0/1）
  target_approach_progress       ← 接近进度（0-1）
```

总计 **6 维**，全部由坐标计算得到，无需视觉模型。

---

### 验证流程

| 验证项 | 通过标准 |
|--------|---------|
| 坐标房屋归属准确率 | 100%（坐标判断是确定性的） |
| 错误房屋告警准确率 | 100% |
| 室内/室外切换准确率 | ≥ 98% |
| explored 后不重复进入 | 0 次 |
| 多房屋自动切换 | 所有房屋被访问 |
| 状态流转正确性 | 符合定义规则 |

---

### 交付物

- [ ] `houses_config.json`（手动标记的房屋 bbox 坐标）
- [ ] 坐标定位模块 `house_locator.py`
- [ ] 搜索状态管理模块 `search_status_manager.py`

---

## Phase 2：入口探索——多模态融合检测可进入入口（重头戏）

### 目标

UAV 到达目标房屋附近后，**融合多种感知模态，找到可以进入的入口（门、窗、洞口）**。

这是整个系统的**核心创新点**——不是单纯用一个 YOLO 检测器，而是融合三种模态做出更准确的入口判断。

### Phase 2 需要回答的问题

| 问题 | 回答方式 |
|------|---------|
| 画面中有没有可进入的开口？ | YOLO 检测 `door_open`, `window`, `opening` |
| 这个开口离我多远？够不够大？ | depth 深度分析 |
| 这个开口能不能安全通过？有没有障碍？ | depth 可穿越性判断 |
| 这到底是门还是窗还是墙上的装饰？ | VLM 语义确认 |
| 检测到的入口属于目标房屋吗？ | 坐标验证（Phase 1 提供） |

### 多模态融合架构

```
                        ┌─────────────────────────┐
                        │     每步输入             │
                        │  RGB 帧 + depth 帧      │
                        │  + UAV 位姿              │
                        └────────┬────────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                   ▼
    ┌──────────────────┐ ┌──────────────┐  ┌──────────────────┐
    │  模态 1: YOLO    │ │ 模态 2: Depth│  │ 模态 3: VLM      │
    │  入口检测        │ │ 深度分析      │  │ 语义理解          │
    │                  │ │              │  │                  │
    │  检测类别：      │ │ 分析内容：    │  │ 判断内容：        │
    │  · door_open     │ │ · 入口距离   │  │ · 这是什么类型？  │
    │  · door_closed   │ │ · 开口宽高   │  │ · 能不能通过？    │
    │  · window        │ │ · 可穿越性   │  │ · 安全性如何？    │
    │  · opening       │ │ · 障碍物检测 │  │ · 最佳进入方式？  │
    └────────┬─────────┘ └──────┬───────┘  └────────┬─────────┘
             │                  │                    │
             └──────────────────┼────────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │    融合决策模块        │
                    │                       │
                    │  YOLO 说有门          │
                    │  + depth 说够近且可穿 │
                    │  + VLM 确认是门       │
                    │  = 高置信度入口 ✓     │
                    │                       │
                    │  YOLO 说有门          │
                    │  + depth 说前方有障碍 │
                    │  = 不可穿越 ✗         │
                    │                       │
                    │  YOLO 没检测到        │
                    │  + VLM 说"这面墙有缝隙"│
                    │  + depth 确认有深度   │
                    │  = 可能的入口？→ 靠近 │
                    └───────────────────────┘
```

### 模态 1：YOLO 入口检测

#### 检测类别

```yaml
# phase2_entry_detector.yaml
path: ./entry_detection_data
train: images/train
val: images/val
names:
  0: door_open       # 开着的门（可进入）
  1: door_closed     # 关闭的门（不可进入）
  2: window          # 窗户（可能可进入，需 depth 确认尺寸）
  3: opening         # 其他洞口/缺口（破损墙壁、车库口等）
```

> 相比之前的 3 类，新增了 `opening`（通用洞口），覆盖非标准入口。

#### 标注方法

和之前一样用 Roboflow 拖框，但多了一个类别：

```
标注示例：

正面看到一栋房屋的墙面：
┌────────────────────────────────────┐
│                                    │
│     ┌──────┐    ┌────┐   ┌────┐   │
│     │door  │    │win │   │win │   │
│     │_open │    │dow │   │dow │   │
│     │      │    │    │   │    │   │
│     │      │    └────┘   └────┘   │
│     └──────┘                      │
│                                    │
└────────────────────────────────────┘
→ 标 3 个框：1 × door_open + 2 × window

看到一个有破洞的墙壁：
┌────────────────────────────────────┐
│                                    │
│     墙面         ┌───────┐        │
│                  │opening│        │
│                  │(破洞) │        │
│                  └───────┘        │
│                                    │
└────────────────────────────────────┘
→ 标 1 个框：opening
```

#### 训练数据量

| 类别 | 距离 | 数量 | 标注 |
|------|------|------|------|
| `door_open` | 近（1-3m）+ 中（3-6m）| 每门 40 张 | bbox 框住门框 |
| `door_closed` | 近+中 | 每门 15 张 | bbox 框住门框 |
| `window` | 近+中 | 每栋 15 张 | bbox 框住窗框 |
| `opening` | 近+中 | 20 张 | bbox 框住洞口 |
| 负样本（纯墙面）| 近+中 | 100 张 | 无标注 |

假设 3 栋房屋 × 2 门 = 6 门：
- door_open: 6 × 40 = 240 张
- door_closed: 6 × 15 = 90 张
- window: 3 × 15 = 45 张
- opening: 20 张
- 负样本: 100 张
- **总计约 495 张**，按 8:2 分训练/验证。

#### 训练命令

```bash
yolo train \
  model=yolov8n.pt \
  data=phase2_entry_detector.yaml \
  epochs=120 \
  imgsz=640 \
  batch=16 \
  project=runs/entry_detector \
  name=v1
```

#### YOLO 验证标准

| 指标 | 要求 |
|------|------|
| door_open mAP@0.5 | ≥ 0.85 |
| door_open recall | ≥ 0.90（不能漏检可进入的门） |
| door_closed precision | ≥ 0.80 |
| opening recall | ≥ 0.75 |
| 整体 mAP@0.5 | ≥ 0.80 |
| 推理速度（640px） | ≤ 8ms / 帧 |

---

### 模态 2：Depth 深度分析

YOLO 告诉你"这里有个门"，depth 告诉你"能不能过去"。

#### depth 提供的信息

```python
import numpy as np

def analyze_entry_depth(depth_frame, entry_bbox, frame_width, frame_height):
    """
    对 YOLO 检测到的入口区域做 depth 分析。

    返回入口的距离、尺寸估算、可穿越性。
    """
    x1, y1, x2, y2 = [int(v) for v in entry_bbox]
    entry_region = depth_frame[y1:y2, x1:x2]

    # 1. 入口距离（取 20th percentile 避免噪声）
    valid_depths = entry_region[(entry_region > 50) & (entry_region < 5000)]
    if len(valid_depths) == 0:
        return {"status": "invalid", "reason": "depth 数据无效"}

    distance_cm = float(np.percentile(valid_depths, 20))

    # 2. 入口中心 vs 周围的 depth 差异
    #    如果门/洞口是开的 → 中心 depth >> 边框 depth（看得更远）
    #    如果门是关的 → 中心 depth ≈ 边框 depth（都打在门板上）
    center_h = entry_region.shape[0] // 4
    center_w = entry_region.shape[1] // 4
    center_depth = np.median(entry_region[
        center_h : 3*center_h,
        center_w : 3*center_w
    ])
    edge_depth = np.median(np.concatenate([
        entry_region[:center_h, :].flatten(),
        entry_region[3*center_h:, :].flatten(),
        entry_region[:, :center_w].flatten(),
        entry_region[:, 3*center_w:].flatten()
    ]))

    depth_ratio = center_depth / (edge_depth + 1e-6)
    is_open = depth_ratio > 1.3  # 中心比边缘深 30% 以上 → 开口

    # 3. 可穿越性（门后有没有障碍物）
    has_obstacle = center_depth < 80  # 门后 80cm 内有东西

    # 4. 开口尺寸估算（粗略）
    #    像素宽度 × 距离 / 焦距 ≈ 实际宽度
    pixel_width = x2 - x1
    pixel_height = y2 - y1
    focal_length_px = frame_width / (2 * np.tan(np.radians(45)))  # FOV=90°
    real_width_cm = pixel_width * distance_cm / focal_length_px
    real_height_cm = pixel_height * distance_cm / focal_length_px

    # UAV 宽度约 60cm，需要至少 80cm 宽的开口
    size_sufficient = real_width_cm > 80 and real_height_cm > 60

    return {
        "distance_cm": round(distance_cm, 1),
        "center_depth_cm": round(center_depth, 1),
        "depth_ratio": round(depth_ratio, 2),
        "is_open": is_open,
        "has_obstacle": has_obstacle,
        "estimated_width_cm": round(real_width_cm, 1),
        "estimated_height_cm": round(real_height_cm, 1),
        "size_sufficient": size_sufficient,
        "traversable": is_open and not has_obstacle and size_sufficient
    }
```

#### depth 分析的决策逻辑

```
YOLO 检测到入口（door_open / window / opening）
  │
  ▼
depth 分析：
  ├── distance > 500cm → "too_far"：继续靠近
  ├── distance 250-500cm → "approaching"：正在接近
  ├── distance < 250cm → 进入详细分析：
  │     │
  │     ├── depth_ratio < 1.3 → "closed"：虽然 YOLO 说是 door_open，但 depth 显示是平面（误检？）
  │     ├── has_obstacle → "blocked"：开口后面有障碍物
  │     ├── !size_sufficient → "too_small"：开口太小，UAV 过不去
  │     └── 全部通过 → "traversable" ✓
  │
  ▼
输出 depth 判断结果 → 传给融合决策模块
```

---

### 模态 3：VLM 语义确认

VLM（视觉语言模型）是第三道确认——用自然语言理解画面内容，解决 YOLO 和 depth 无法回答的问题。

#### VLM 的使用场景

| 场景 | YOLO + depth 的不足 | VLM 能做什么 |
|------|---------------------|-------------|
| YOLO 检测到 door_open，但其实是一幅画 | YOLO 只看纹理，不理解语义 | "这是墙上的装饰画，不是真的门" |
| 检测到 window，但不确定能否通过 | depth 只知道有深度 | "窗户打开了，没有纱窗，可以尝试通过" |
| 没检测到任何入口，但墙上有裂缝 | YOLO 没有 "crack" 类别 | "墙壁右侧有一条大裂缝，可能可以通过" |
| 检测到两个 door_open，选哪个 | 两个都满足条件 | "左边是正门，更宽敞；右边是侧门，较窄" |

#### VLM 调用方式

```python
VLM_ENTRY_PROMPT = """Look at this image from a UAV camera approaching a building.

Answer these questions concisely:
1. What potential entry points do you see? (doors, windows, openings, gaps)
2. For each entry point: is it open or closed? Is it large enough for a small drone (60cm wide) to fly through?
3. Are there any obstacles near the entry points?
4. Which entry point would you recommend for a drone to enter, and why?

Be specific about positions (left/center/right of frame).
"""

def vlm_confirm_entry(rgb_frame_path, vlm_model):
    """
    用 VLM 对当前画面做语义入口分析。

    调用时机：YOLO 检测到入口 + depth 确认可穿越 → VLM 做最终确认
    或者：YOLO 没检测到 → 但 VLM 可能发现非标准入口
    """
    response = vlm_model.analyze(rgb_frame_path, VLM_ENTRY_PROMPT)

    return {
        "vlm_description": response,
        "has_entry_recommendation": "recommend" in response.lower(),
        "mentions_obstacle": any(w in response.lower()
                                 for w in ["obstacle", "blocked", "closed"]),
        "mentions_open": any(w in response.lower()
                             for w in ["open", "can fly", "large enough"])
    }
```

#### VLM 的调用频率

VLM 推理速度慢（本地 LLaVA ~500ms，API ~1-2s），不能每步都调用：

| 调用时机 | 原因 |
|---------|------|
| YOLO 检测到 door_open 且 depth 确认可穿越时 | 最终确认，防止误检 |
| 绕飞一圈没找到入口时 | VLM 可能发现 YOLO 漏检的非标准入口 |
| YOLO 检测结果和 depth 矛盾时 | YOLO 说有门但 depth 说是平面 → VLM 仲裁 |
| 首次到达房屋附近时（1 次） | 获取房屋整体外观描述 |

**不调用 VLM 的情况**：
- 正常绕飞没检测到入口 → 不需要 VLM
- YOLO 说 door_closed + depth 确认是平面 → 明确关门，不需要 VLM

---

### 融合决策模块

三个模态的结果汇总到一个决策函数：

```python
def fuse_entry_decision(yolo_result, depth_result, vlm_result=None, uav_pose=None, houses_config=None, target_house_id=None):
    """
    融合三种模态的入口检测结果，做出最终决策。

    决策优先级：
    1. 坐标验证（一票否决：不在目标房屋范围 → 忽略）
    2. depth 可穿越性（一票否决：有障碍 → blocked）
    3. YOLO + VLM 入口确认（综合判断）
    """

    # === 第 0 层：坐标验证 ===
    if uav_pose and houses_config and target_house_id:
        in_target, reason = verify_door_belongs_to_target(
            uav_pose, houses_config, target_house_id
        )
        if not in_target:
            return {
                "decision": "ignore",
                "reason": reason,
                "confidence": 1.0
            }

    # === 第 1 层：距离判断 ===
    if depth_result and depth_result["distance_cm"] > 500:
        return {
            "decision": "approach",
            "reason": f"入口距离 {depth_result['distance_cm']:.0f}cm，继续靠近",
            "confidence": 0.8
        }

    # === 第 2 层：depth 可穿越性（一票否决） ===
    if depth_result and depth_result.get("has_obstacle"):
        return {
            "decision": "blocked",
            "reason": f"入口后方有障碍物 (depth={depth_result['center_depth_cm']:.0f}cm)",
            "confidence": 0.9
        }

    if depth_result and not depth_result.get("size_sufficient"):
        return {
            "decision": "too_small",
            "reason": f"开口太小 ({depth_result['estimated_width_cm']:.0f}×{depth_result['estimated_height_cm']:.0f}cm)",
            "confidence": 0.85
        }

    # === 第 3 层：YOLO + depth + VLM 综合判断 ===
    confidence = 0.0
    reasons = []

    # YOLO 贡献
    if yolo_result:
        cls = yolo_result.get("class", "")
        yolo_conf = yolo_result.get("confidence", 0)
        if cls == "door_open":
            confidence += 0.4 * yolo_conf
            reasons.append(f"YOLO: door_open ({yolo_conf:.2f})")
        elif cls == "opening":
            confidence += 0.3 * yolo_conf
            reasons.append(f"YOLO: opening ({yolo_conf:.2f})")
        elif cls == "window":
            confidence += 0.2 * yolo_conf
            reasons.append(f"YOLO: window ({yolo_conf:.2f})")

    # depth 贡献
    if depth_result and depth_result.get("traversable"):
        confidence += 0.35
        reasons.append(f"depth: 可穿越 (距离{depth_result['distance_cm']:.0f}cm, "
                      f"宽{depth_result['estimated_width_cm']:.0f}cm)")

    # VLM 贡献
    if vlm_result:
        if vlm_result.get("mentions_open"):
            confidence += 0.25
            reasons.append("VLM: 确认入口开放")
        if vlm_result.get("mentions_obstacle"):
            confidence -= 0.15
            reasons.append("VLM: 提到障碍物")

    # 最终决策
    if confidence >= 0.7:
        decision = "enter"
    elif confidence >= 0.4:
        decision = "approach"  # 有可能，再靠近看看
    else:
        decision = "skip"  # 置信度太低，继续绕飞

    return {
        "decision": decision,
        "confidence": round(min(confidence, 1.0), 3),
        "reasons": reasons,
        "yolo": yolo_result,
        "depth": depth_result,
        "vlm": vlm_result
    }
```

#### 融合决策示例

```
场景 1：正常进入
  YOLO: door_open (conf=0.92) → +0.37
  depth: 可穿越, 距离 180cm, 宽 120cm → +0.35
  VLM: "正前方有一扇打开的木门，足够宽" → +0.25
  总分: 0.97 → decision = "enter" ✓

场景 2：看到门但有障碍
  YOLO: door_open (conf=0.85) → +0.34
  depth: 门后 50cm 处有障碍物 → 一票否决
  → decision = "blocked" ✗

场景 3：YOLO 没检测到，但 VLM 发现
  YOLO: 无检测 → +0
  depth: 右侧区域有深度变化 → 不确定
  VLM: "右侧墙壁有一个较大的破洞，看起来可以通过" → +0.25
  总分: 0.25 → decision = "approach"（靠近确认）

场景 4：看到窗户
  YOLO: window (conf=0.88) → +0.18
  depth: 可穿越, 距离 200cm, 宽 90cm → +0.35
  VLM: "窗户完全打开，无纱窗" → +0.25
  总分: 0.78 → decision = "enter" ✓
```

---

### 完整探索流程

```
UAV 到达目标房屋的 approach_radius 范围内
  │
  ▼
切换 stage_label = "house_circling"
加载 Phase 2 入口检测 YOLO 模型
  │
  ▼
绕房屋飞行，每步：
  │
  ├── ① 拍 RGB + depth
  │
  ├── ② YOLO 入口检测（每步都跑，≤ 8ms）
  │     检测到 door_open / window / opening → 候选入口
  │     检测到 door_closed → 记录，跳过
  │     什么都没检测到 → 继续绕飞
  │
  ├── ③ 如果有候选入口：
  │     │
  │     ├── 坐标验证：UAV 在 target_house bbox 内？
  │     │   → 不在 → 忽略（可能是邻居的门）
  │     │
  │     ├── depth 深度分析：
  │     │   → 距离、开口尺寸、可穿越性
  │     │
  │     ├── 如果 YOLO 高置信度 + depth 确认可穿越：
  │     │   → 调用 VLM 做最终确认（仅此时调用，控制频率）
  │     │
  │     └── 融合决策：
  │         → "enter" (≥0.7) → 进入流程
  │         → "approach" (0.4-0.7) → 靠近再判断
  │         → "blocked" → 标记，找下一个
  │         → "skip" (<0.4) → 继续绕飞
  │
  ├── ④ decision = "enter"：
  │     → 调整偏航对齐入口中心
  │     → 切换 stage_label = "entering"
  │     → 前进穿过入口
  │
  └── ⑤ decision = "approach"：
        → 朝候选入口方向靠近
        → 下一步重新检测+分析

绕飞一整圈没找到入口：
  → VLM 分析整圈的帧，看是否有遗漏的非标准入口
  → 如果确实没有 → 标记该房屋暂时无法进入
  → 选下一栋 unexplored 房屋
```

### 蒸馏特征输出

Phase 2 为 Phase 5 蒸馏提供以下特征：

```
入口感知特征（8维）：
  entry_detected              ← 检测到可进入入口（0/1）
  entry_type                  ← 类型编码：door=1, window=0.7, opening=0.5, none=0
  entry_confidence            ← YOLO 检测置信度（0-1）
  entry_distance_norm         ← depth 距离归一化（/500cm）
  entry_alignment_norm        ← 入口中心偏移归一化（/0.5）
  entry_traversable           ← depth 可穿越判断（0/1）
  entry_size_ratio            ← 开口尺寸 / UAV 尺寸（>1 = 够大）
  fusion_confidence           ← 三模态融合置信度（0-1）
```

总计 **8 维**。

---

### 验证流程

| 验证项 | 通过标准 |
|--------|---------|
| YOLO door_open mAP@0.5 | ≥ 0.85 |
| YOLO door_open recall | ≥ 0.90 |
| YOLO opening recall | ≥ 0.75 |
| depth 可穿越性判断准确率 | ≥ 90% |
| 融合决策准确率 | ≥ 85%（对比人工标注的"正确入口"） |
| VLM 入口确认准确率 | ≥ 80% |
| 误进入率（blocked/too_small 的入口还是进了） | ≤ 5% |
| 绕飞一圈找到入口的成功率 | ≥ 90%（有入口的房屋） |

---

### 交付物

- [ ] YOLO 入口检测模型训练数据（~495 张，4 类：door_open/door_closed/window/opening）
- [ ] YOLO 模型权重 `entry_detector_v1/best.pt`
- [ ] depth 分析模块 `entry_depth_analyzer.py`
- [ ] VLM 入口确认模块 `vlm_entry_confirmer.py`
- [ ] 融合决策模块 `entry_fusion_decision.py`
- [ ] 绕房屋探索策略脚本 `house_circling_explorer.py`
- [ ] 检测效果可视化报告（含三模态融合示例）

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
  "target_house_id": "house_2",
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
Phase 1 的输出 → 坐标定位（6维）                   ─┐
Phase 2 的输出 → 入口感知：YOLO+depth+VLM（8维）   ─┤
Phase 3 的轨迹 → outdoor_nav 标签数据              ─┤──► 蒸馏网络输入
Phase 4 的轨迹 → indoor 标签数据                   ─┤
server 提供     → pose, depth, risk                ─┘
```

### 蒸馏网络架构

```
输入层（28维特征向量）
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

#### 输入特征定义（28维）

```
坐标定位特征（6维，确定性，来自 Phase 1）：
  distance_to_target_norm        ← 到目标房屋中心的归一化距离（/2000cm）
  bearing_error_norm             ← 方位角误差归一化（/180°）
  is_at_target_house             ← 是否在目标房屋范围内（0/1）
  is_at_wrong_house              ← 是否在非目标房屋范围内（0/1）
  is_inside                      ← 是否在室内（0/1）
  target_approach_progress       ← 接近目标的进度（0-1）

入口感知特征（8维，来自 Phase 2 多模态融合）：
  entry_detected                 ← 检测到可进入入口（0/1）
  entry_type                     ← 类型编码：door=1, window=0.7, opening=0.5, none=0
  entry_confidence               ← YOLO 检测置信度（0-1）
  entry_distance_norm            ← depth 距离归一化（/500cm）
  entry_alignment_norm           ← 入口中心偏移归一化（/0.5）
  entry_traversable              ← depth 可穿越判断（0/1）
  entry_size_ratio               ← 开口尺寸 / UAV 尺寸
  fusion_confidence              ← 三模态融合置信度（0-1）

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
    │   ├── house_1/
    │   │   ├── ref_front.jpg
    │   │   ├── ref_left45.jpg
    │   │   ├── ref_right45.jpg
    │   │   ├── ref_close.jpg
    │   │   └── ref_high.jpg
    │   ├── house_2/
    │   └── house_3/
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
| 房屋识别 | building 检测率 | 帧中有建筑时成功检测到的比例 |
| 房屋识别 | 方向匹配准确率 | 正确将 building 匹配到地图房屋的比例 |
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
| Phase 1 | 0.5h（标注坐标） | — | 无（坐标判断） | 0.5h | 1h |
| Phase 2 | 3h | 3h | 0.5h | 1h | 7.5h |
| Phase 3 | 3h | 0.5h | 无 | 1h | 4.5h |
| Phase 4 | 8h | 2h | 1h | 2h | 13h |
| Phase 5 | 1h（构建数据） | — | 0.5h | 2h | 3.5h |
| Phase 6 | 持续 | — | 持续 | 持续 | 每轮约 4h |
| **总计（到首次完整运行）** | | | | | **约 32.5h（≈ 1 周）** |

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

## 论文实验计划

### 研究问题定义

当前系统的研究问题从原来的 `navigation`（导航到目标点）正式转向 `language-guided embodied person search`（语言引导的具身人员搜索）。

核心研究目标：
- 在未知或部分未知房屋中
- 根据语言任务主动搜索人员
- 判断是否有人
- 估计人员位置
- 减少重复搜索
- 在遮挡和复杂几何条件下完成确认

### 实验分组

#### Group A：感知与场景理解

目标：验证系统的感知能力

| 子任务 | 指标 |
|--------|------|
| `outside_house` vs `inside_house` vs `threshold_zone` 分类 | 场景状态准确率 |
| `entry_door_visible` 判断 | 门可见性 precision/recall |
| `entry_door_traversable` 判断 | 可穿越性 precision/recall |
| `target_house_match` 目标房屋匹配 | top-1 / top-k 检索准确率 |

#### Group B：入口搜索策略

目标：验证 UAV 能从户外随机位置找到正确入口

任务流程：
```
给定目标房屋参考图
→ 从户外随机位置出发
→ 搜索正确的门
→ 接近并穿越门槛
```

| 指标 | 说明 |
|------|------|
| Entry Success Rate | 成功进入目标房屋的比率 |
| Time-to-Entry | 从开始到进入的时间 |
| Path Length to Entry | 入口路径长度 |
| Steps to Entry | 入口步数 |
| False-Entry Rate | 进入错误房屋的比率 |
| No-Entry Termination Rate | 找不到入口的终止比率 |

#### Group C：完整搜人任务

目标：验证从户外进入到室内搜索的完整任务性能

```
给定目标房屋
→ UAV 从户外出发
→ 进入房屋
→ 搜索人员
→ 确认有/无人并报告位置
```

| 指标 | 说明 |
|------|------|
| Full Mission Success Rate | 完整任务成功率 |
| Person Search Success Rate (SSR) | 搜索成功率 |
| Localization Error (LE) | 人员定位误差 |
| Time-to-First-Detection (TFD) | 首次检测时间 |
| Time-to-Entry | 入口时间 |
| Collision Rate (CR) | 碰撞率 |
| Takeover Count | 接管次数 |
| Token Usage (TU) | LLM 调用 token 数 |
| Decision Latency (DL) | 决策延迟 |

#### Group D：蒸馏实验

目标：验证 archive 轨迹能否监督轻量反应策略

训练变体对比：

| 变体 | 说明 |
|------|------|
| BC only | 纯行为克隆 |
| BC + small RL fine-tuning | BC 加轻量 RL 微调 |
| RGB only vs RGB+depth | 输入模态对比 |
| with vs without semantic archive context | 有/无 archive 上下文 |

蒸馏指标：

| 指标 | 说明 |
|------|------|
| Action Accuracy | 在 held-out 专家轨迹上的动作准确率 |
| Waypoint-Following Success | 航点跟随成功率 |
| Entry-Search Success | 入口搜索成功率 |
| Person-Search Success | 搜人成功率 |
| Inference Latency | 推理延迟 |

### 仿真奖励设计

#### Stage 1：户外探索奖励

奖励新覆盖目标房屋外立面的行为：
- 正奖励：新观察到的外立面面积
- 正奖励：减少与已观察视点的重叠
- 小惩罚：重复访问同一外弧区域

#### Stage 2：入口识别奖励

奖励看到入口相关结构：
- 正奖励：检测到门/窗/开口候选
- 较大奖励：开口属于目标房屋
- 更大奖励：开口被判断为可穿越

#### Stage 3：效率与安全

惩罚项：
- 每步时间惩罚
- 碰撞惩罚
- 不安全近距惩罚
- 严重振荡惩罚
- 反复相反转向惩罚

### 终止条件

#### 户外阶段终止

标记 `no_entry_termination` 的情况：
- UAV 绕目标房屋完整一圈但未发现可穿越入口
- 步数预算耗尽

判断信号：
- 里程计
- 偏航角环绕（yaw wraparound）
- 重访起始扇区

#### 室内任务终止

标记任务完成的情况：
- 人员确认存在（confirmed_present）
- 完成室内全搜索后确认无人（confirmed_absent）
- 安全违规
- 全局步数预算耗尽

### Baseline 对比组

最小版本 4 组 baseline：

| 组别 | 方法 | 说明 |
|------|------|------|
| B1 | RGB detection + rule search | 仅 RGB 检测 + 规则搜索 |
| B2 | RGB + depth + rule search | RGB + 深度 + 规则搜索 |
| B3 | planner + reflex without archive | 规划器 + 反射策略，无 archive 记忆 |
| Ours | mission guidance + search archive + reflex + evidence fusion | 完整系统 |

### 消融实验

| 消融 | 移除内容 | 验证什么 |
|------|---------|---------|
| −LLM / mission guidance | 移除高层语言规划 | 语义搜索引导的贡献 |
| −archive | 移除搜索记忆 | 记忆减少重复探索的贡献 |
| −depth | 移除深度输入 | 深度感知的贡献 |
| −evidence fusion | 移除人体证据融合 | 多帧证据累积的贡献 |

### 泛化实验

验证系统在以下未见条件下的表现：

| 泛化维度 | 说明 |
|---------|------|
| Unseen houses | 从未见过的房屋布局 |
| Unseen occlusion | 从未见过的遮挡模式 |
| Unseen lighting | 从未见过的光照条件 |
| Unseen prompts | 从未见过的语言指令 |

### 域随机化方案

为保证泛化能力，在仿真中随机化以下因素：

| 随机化因素 | 说明 |
|-----------|------|
| 房屋纹理 | 墙面颜色、材质、风格 |
| 光照 / 时间 | 日照角度、室内灯光开关 |
| 家具布局 | 室内物体位置和类型 |
| 门廊物品 | 门口的花盆、鞋架等 |
| 遮挡植被 | 入口附近的树木/灌木 |
| 门的开启程度 | 全开、半开、微开 |
| 相机噪声/模糊 | RGB 帧的噪声和运动模糊 |
| 深度噪声 | depth 传感器噪声 |

### 论文结果结构建议

最终论文实验章节分成四块：

```
8.1 Main Results
  比较 SSR, LE, TFD, CR, Takeover Count

8.2 Ablation
  验证 mission guidance / archive / depth / evidence fusion

8.3 Generalization
  验证 unseen houses / occlusion / lighting / prompts

8.4 System Validation
  放当前工程已做好的：
  - runtime stability
  - takeover logging
  - assist_step
  - fixed spawn
  - online metrics
```

### 最小可发表版本

| 模块 | 内容 | 第一版要求 |
|------|------|----------|
| A. Mission Guidance | 语言任务 + 搜索摘要 → 搜索子目标 | 允许用规则/启发式 planner |
| B. Search Archive | 区域 visited/suspect/进度 | 至少记录区域级状态 |
| C. Reflex Executor | 低层执行 + 安全门控 + 接管日志 | 沿用现有 reflex MLP |
| D. Person Evidence Fusion | 单帧检测 + 多帧融合 + 位置投影 | 输出 person_exists / estimated_position / confidence |

---

## Planner-Driven 自主探索执行器设计

### 设计目标

将当前系统从"LLM 提议高层搜索意图 + 人类驱动运动"转变为：

```
planner/LLM 拥有稀疏探索意图
    │
    ▼
server 自动执行有界探索段
    │
    ▼
人类主要监督并在需要时接管
```

### 当前系统边界

当前 runtime 已支持：
- mission-aware planner 输出
- archive 检索上下文
- 本地 reflex 推理
- safety gating
- takeover 日志
- person evidence 日志

**尚未实现**的部分：
1. planner 没有持久执行循环——只返回语义搜索意图和候选航点
2. 运动仍然由用户触发——`manual` 模式下用户是运动源，`assist_step` 只追加一步
3. 没有 planner 拥有的 episode 状态机——无"当前自主段"概念
4. 航点进度不会触发自主继续——可以请求新计划但不会自行执行
5. 安全和证据未耦合到自主运行控制器——都存在但无统一协调

### 设计原则

1. **LLM 保持稀疏**：不发出逐帧电机控制，只负责高层搜索语义
2. **执行有界**：第一版永远不能无限运行，执行有限搜索段后重新评估
3. **安全优先于自主**：任何安全阻塞、重复进度失败或接管请求停止段
4. **可观测性必需**：每个自主段必须记录原因、进度、停止原因和证据结果
5. **分阶段推出**：先有界段执行器，再连续自动搜索

### 执行器运行时状态

新增 `planner_executor_runtime` 对象：

```json
{
  "mode": "segment",
  "active": true,
  "state": "running",
  "run_id": "run_001",
  "segment_id": "seg_003",
  "mission_id": "mission_search_house_1",
  "trigger": "panel_button",
  "current_plan_id": "plan_007",
  "current_search_subgoal": "search_room",
  "target_waypoint": [2360.0, 85.0, 225.0],
  "step_budget": 20,
  "steps_executed": 12,
  "blocked_count": 0,
  "replan_count": 1,
  "last_action": "forward",
  "last_progress_cm": 18.5,
  "last_stop_reason": null,
  "last_stop_detail": null,
  "started_at": "2025-03-26T14:30:00",
  "updated_at": "2025-03-26T14:30:45"
}
```

#### 状态定义

| 状态 | 含义 |
|------|------|
| `idle` | 空闲，等待启动 |
| `starting` | 正在初始化段 |
| `running` | 自主执行中 |
| `replanning` | 请求新计划中 |
| `blocked` | 被安全门控阻塞 |
| `waiting_confirmation` | 等待人工确认 |
| `completed` | 段正常完成 |
| `aborted` | 段被中止 |
| `takeover` | 人工接管中 |

### 两阶段推出

#### Stage A：计划段执行器（先做这个）

行为：server 执行一个有界自主段

```
获取或复用当前计划
  │
  ▼
用 reflex/局部启发式跟随当前航点
  │
  ▼
在以下条件之一时停止：
  ├─ 航点到达
  ├─ 步数预算耗尽
  ├─ 无进度阈值超过
  ├─ 高风险门控反复触发
  ├─ 人员证据需要确认
  └─ 人工接管开始
```

**API 设计**：

```
POST /execute_plan_segment
  请求: { task_label, step_budget, refresh_plan, trigger }
  响应: { status, planner_executor_runtime, segment_summary, state }

POST /stop_planner_executor
  停止当前段

GET /planner_executor
  查询执行器状态
```

**面板控制**：
- `Execute Plan Segment` 按钮
- `Stop Executor` 按钮
- 执行器状态行

**为什么先做 Stage A**：比连续自主循环容易调试——运行时间有限、日志确定性、故障分析简单、安全风险低。

#### Stage B：连续自动搜索执行器（Stage A 稳定后再做）

行为：后台自主循环

```
while mission_active:
  1. 确保存在任务
  2. 确保有当前计划
  3. 执行一个低层动作
  4. 检查进度/证据/安全
  5. 需要时重新规划
  6. 在完成、接管或失败时停止
```

执行模式（`planner_execute_mode`）：
- `manual` — 完全手动
- `segment` — 有界段执行
- `auto_search` — 连续自动搜索

### 每步执行策略

执行器永远不向 LLM 请求低层动作。每步：

```
1. 读取 current_plan
2. 将语义目标转换为局部执行目标
3. 请求 reflex_runtime
4. 门控 reflex 动作
5. reflex 不可用时回退到局部航点跟随启发式
6. 应用恰好一步移动
7. 重新测量进度
```

动作源优先级：
1. 接受的 reflex 动作（本地策略有信心时）
2. 局部启发式动作（朝活跃航点）
3. hold / stop / replan

### 重规划与停止条件

#### 重规划触发

| 条件 | 说明 |
|------|------|
| 航点到达 | 当前航点已接近 |
| 区域标记为 observed | 当前区域已扫描 |
| 新建 suspect 区域 | 检测到可疑区域 |
| 证据状态变化 | PEF 证据分变化 |
| N 步无进度 | 连续多步距离未减少 |
| N 步偏航误差高 | 连续多步无法对准 |
| archive 建议更好的 cell | 检索发现优先扩展 cell |

#### 硬停止触发

| 条件 | 说明 |
|------|------|
| 接管激活 | 人工按下接管 |
| 重复安全门控阻塞 | 多次被安全阻止 |
| 重复高风险动作 | 连续多步高风险 |
| planner 不可用且无局部回退 | 无法生成动作 |
| 人员确认 + 任务要求报告 | 无需继续探索 |

### 搜索子目标特定执行规则

| 子目标 | 行为特点 |
|--------|---------|
| `search_house` | 更广的扫描行为，更大步数预算，房间/区域转换时重规划 |
| `search_room` | 较小步数预算，紧贴优先区域 |
| `search_frontier` | 偏向探索和覆盖增益，到达前沿或视野打开时停止 |
| `approach_suspect_region` | 减少步数预算，降速/更紧门控，风险上升时快速重规划 |
| `confirm_suspect_region` | 允许观察导向动作，允许短偏航/垂直扫描，证据状态变化时停止 |

### 安全集成

使用现有安全信号：
- `risk_score`
- reflex 置信度阈值
- shield 触发
- takeover 状态
- blocked action 原因

**新增安全计数器**（在 `planner_executor_runtime` 内）：

| 计数器 | 用途 |
|--------|------|
| `consecutive_blocked_steps` | 连续被阻塞步数 |
| `consecutive_no_progress_steps` | 连续无进度步数 |
| `consecutive_high_risk_steps` | 连续高风险步数 |

任一超过阈值 → 段停止，记录清晰原因。

### 证据集成

执行器必须响应 `person_evidence_runtime`：

```
if suspect 出现:
  → 停止段 或 重规划到 approach_suspect_region

if confirmed_present:
  → 停止探索 → 标记 run 成功

if confirmed_absent:
  → 标记区域为 checked → 继续搜索并重规划
```

这是使循环成为真正搜索导向（而非通用导航）的关键。

### 执行器日志

新增日志流 `phase4_executor_logs`，每段记录：

```json
{
  "mission_id": "mission_search_house_1",
  "task_label": "search the house for people",
  "planner_source": "heuristic",
  "planner_model": null,
  "active_plan_id": "plan_007",
  "active_search_subgoal": "search_room",
  "step_budget": 20,
  "executed_step_count": 18,
  "progress_trace": [
    {"step": 1, "action": "forward", "progress_cm": 19.2},
    {"step": 2, "action": "yaw_right", "progress_cm": 0.0}
  ],
  "blocked_reasons": [],
  "replan_reasons": ["waypoint_reached"],
  "takeover_status": false,
  "evidence_before": {"suspect": false, "confidence": 0.12},
  "evidence_after": {"suspect": true, "confidence": 0.68},
  "final_stop_reason": "replan_suspect_detected"
}
```

### 执行器实现计划

| 步骤 | 内容 | 文件 |
|------|------|------|
| Step 1 | 添加运行时状态（不动） | `runtime_interfaces.py`, `uav_control_server.py` |
| Step 2 | 添加 `POST /execute_plan_segment`（有界同步段，默认 `step_budget=5`） | `uav_control_server.py` |
| Step 3 | 添加面板控件 | `uav_control_panel.py` |
| Step 4 | 记录每个段和停止原因的日志 | `phase4_executor_logs/` |
| Step 5 | Stage A 稳定后添加后台 `auto_search` | `uav_control_server.py` |

### 验收标准

Stage A 成功的标准：
- 任务如 `search the house for people` 可以运行一个有界计划段而无需键盘操作
- 段能清晰说明是因为 success / no progress / safety block / replan / takeover 而停止
- planner source/model 在日志中可见
- 证据状态在段前/后都被捕获
- 人类可以随时停止段

---

## 系统升级路线图

### Phase 0-5 工程实现路线

```
Phase 0  稳定手动遥控基线
  已实现: /state, /frame, /capture, /move_relative
    │
    ▼
Phase 1  点云能力
  已实现: depth 重建, 多模式点云输入, 多模态 capture bundle
    │
    ▼
Phase 2  稀疏高层规划
  已实现: planner state schema, /plan, /request_plan, 启发式回退
    │
    ▼
Phase 3  目标条件多模态 archive + reflex 蒸馏
  目标: archive_cell_id, local_policy_action, risk_score
    │
    ▼
Phase 4  完整分层循环
  目标: 稀疏 planner + archive 检索 + reflex + safety head + replan
    │
    ▼
Phase 5  真实硬件流
  目标: 真实点云替换 replay/synthetic, sim-to-real 部署
```

### 当前已实现接口

| 接口 | 功能 | Phase |
|------|------|-------|
| `GET /state` | UAV 运行时状态 | 0 |
| `GET /frame` | RGB 预览 | 0 |
| `GET /depth_frame` | 深度预览 | 0 |
| `POST /move_relative` | 相对运动控制 | 0 |
| `POST /capture` | 多模态 bundle 保存 | 0 |
| `POST /task` | 任务标签设置 | 0 |
| `POST /set_pose` | 位姿设置 | 0 |
| `GET /pointcloud` | 点云数据 | 1 |
| `GET /pointcloud_frame` | 点云可视化 | 1 |
| `GET /plan` | 获取计划 | 2 |
| `POST /plan` | 设置计划 | 2 |
| `POST /request_plan` | 请求新计划 | 2 |

### 运行时已预留字段

Phase 3-4 已在 runtime debug 中预留：

```
current_waypoint        ← 当前航点
local_policy_action     ← 本地策略动作
risk_score              ← 风险分
shield_triggered        ← 盾牌是否触发
archive_cell_id         ← archive cell 索引
```

### 文件级修改清单（按论文方向）

#### 必须修改的文件

| 文件 | 修改内容 |
|------|---------|
| `runtime_interfaces.py` | 添加 mission schema, search episode schema, planner executor runtime |
| `uav_control_server.py` | 添加 search runtime state, execute_plan_segment, 搜索字段 |
| `uav_control_panel.py` | 添加搜索状态显示, 执行器控件 |
| `planner_server.py` | 输出语义从导航子目标升级为搜索子目标 |
| `archive_runtime.py` | 扩展为 search archive（visited/observed/suspect） |
| `online_reflex_eval.py` | 新增 search summary |

#### 建议新增的文件

| 文件 | 用途 |
|------|------|
| `person_evidence_fusion.py` | 人体检测缓存 + 多帧融合 + 位置估计 |
| `person_search_runtime.py` | 搜索任务运行时协调 |
| `search_eval.py` | 搜索任务专用评估脚本 |
| `search_dataset_builder.py` | 搜索任务数据集构建 |

#### 可复用的现有资产

| 资产 | 复用方式 |
|------|---------|
| `phase3_dataset_export/*` | 作为 system validation 的数据 |
| `reflex_policy_server.py` | 直接作为低层执行器 |
| takeover logs | 保留为论文 system validation 部分 |
| capture bundles | 保留为现有实验数据 |
| fixed spawn pose | 保留用于可重复实验 |
| reflex executor / assist_step | 直接沿用 |

### 推荐执行顺序

```
1. 定义 mission / search schema
   ↓
2. 修改 uav_control_server.py 记录 search runtime
   ↓
3. 修改 planner_server.py 高层输出语义
   ↓
4. 扩展 archive_runtime.py 为 search archive
   ↓
5. 补充 person_evidence_fusion.py
   ↓
6. 集中跑搜索实验
```

### 核心结论

当前系统并没有偏离新论文方向。当前工程已经做出了一个很强的导航与执行底座，新论文需要在这个底座上把"导航问题"升级成"搜索问题"。

后续工作的重点不是推翻当前系统，而是：
1. **重写任务口径**：从导航任务 → 搜索任务
2. **重写实验口径**：从导航指标 → 搜索指标
3. **补齐搜索能力**：搜索记忆 + 人体证据融合 + 自主执行器

一旦这三件事完成，当前已有的 planner / archive / reflex / takeover / online eval 资产都可以继续复用。

---

## 附录：关键文件对照

| 训练阶段 | 使用的现有文件 | 新建文件 |
|---------|--------------|---------|
| Phase 0 数据采集 | `uav_control_server_basic.py`, `uav_control_panel_basic.py` | `collect_training_data.py` |
| Phase 1 地图标记+定位 | — | `house_locator.py`, `search_status_manager.py`, `houses_config.json` |
| Phase 2 入口探索 | — | `check_entry_condition.py`, `verify_door_belongs_to_target.py`, `phase2_entry_detector.yaml` |
| Phase 3 跨房屋导航 | — | `rule_based_house_navigator.py` |
| Phase 4 室内搜索 | `reflex_policy_model.py`, `train_reflex_policy.py`, `reflex_dataset_builder.py` | 扩展 `FEATURE_NAMES` |
| Phase 5 蒸馏 | `train_reflex_policy.py`（扩展 MLP 架构） | `build_distillation_dataset.py` |
| Phase 6 迭代 | `online_reflex_eval.py` | `iteration_manager.py` |
