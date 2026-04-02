# UAV-Flow 完整训练流水线

## 目录

- [概览](#概览)
- [Phase 0：环境准备与数据基础设施](#phase-0环境准备与数据基础设施)
- [Phase 1：地图标记与坐标定位](#phase-1地图标记与坐标定位轻量化)
- [Phase 2：入口探索——多模态融合检测可进入入口](#phase-2入口探索多模态融合检测可进入入口核心创新)
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

## Phase 2：入口探索——多模态融合检测可进入入口（核心创新）

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

### Phase 1 → Phase 2 接口

```
Phase 1 提供给 Phase 2 的输入：
┌─────────────────────────────────────────────────┐
│  target_house_id:     "house_1"                 │
│  house_center:        [2400.0, 100.0]           │
│  house_bbox:          x_min~x_max, y_min~y_max  │
│  approach_radius_cm:  800.0                     │
│  floor_z_range:       [200.0, 350.0]            │
│  search_status:       "unexplored"              │
└─────────────────────────────────────────────────┘

触发条件：
  locate_uav() 返回 nearest_house_distance < approach_radius
  → 切换到 Phase 2 入口探索模式
```

```python
# Phase 1 → Phase 2 的切换逻辑
def should_start_phase2(uav_state, houses_config, target_house_id):
    """判断是否应该从 Phase 1 切换到 Phase 2"""
    uav_x = uav_state["x_val"]
    uav_y = uav_state["y_val"]
    
    house = next(h for h in houses_config["houses"] if h["house_id"] == target_house_id)
    cx, cy = house["center"]
    distance = ((uav_x - cx)**2 + (uav_y - cy)**2) ** 0.5
    
    return distance < house["approach_radius_cm"]
```

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

### 实施路线总览

```
Step A：YOLO 训练数据采集        ← 手动飞行截图（不需要坐标/depth）
Step B：YOLO 入口检测训练        ← Roboflow 标注 + 训练 4 类检测器
       ─── YOLO 训练完成分界线 ──────────────────────────
Step C：Depth 分析模块实现       ← YOLO 在线推理时同步获取 depth 验证
Step D：VLM 部署与 Prompt 调优   ← YOLO 检测到入口时触发 VLM 确认
Step E：融合决策模块             ← Rule-based + Learned 融合（用实时数据训练）
Step F：绕飞策略与入口进入       ← 围绕房屋飞行 + 进入流程
Step G：集成测试与端到端验证     ← 在线测试 + 失败恢复
```

---

### Step A：YOLO 训练数据采集（纯图片，不需要坐标）

> **重要**：YOLO 训练只需要 **图片 + 标注框**，不需要 UAV 坐标、不需要 `houses_config.json`、不需要 depth。
> 坐标和 depth 数据在 YOLO 训练好之后，通过**实时运行**自动采集。

#### 采集方式：手动操控 + 截图

```
YOLO 训练数据采集流程：

┌────────────────────────────────────────────────────┐
│                                                    │
│  1. 打开 UE4 + AirSim 仿真环境                     │
│  2. 用控制面板（panel）手动操控 UAV 飞到房屋附近     │
│  3. 调整角度，让门/窗/洞口出现在画面中              │
│  4. 截取 RGB 图片（通过 /frame 接口或面板截图）      │
│  5. 换不同距离、角度、房屋重复                      │
│  6. 上传到 Roboflow 标注                           │
│                                                    │
│  ★ 不需要坐标！不需要 depth！只需要 RGB 图片！       │
│                                                    │
└────────────────────────────────────────────────────┘
```

#### 采集建议

```
每栋房屋的拍摄思路：

              正面
           ────────────────
          │                │
侧面      │                │ 侧面
          │    house_1     │
          │                │
          │                │
           ────────────────
              背面

手动飞到每面墙前面，拍摄包含门/窗的画面：
  · 近距离（~2m）：门/窗占画面 50%+，细节清晰
  · 中距离（~4m）：门/窗占画面 20-30%，看到整面墙
  · 远距离（~6m）：门/窗较小，测试远距离检测能力
  · 不同角度：正对 / 左偏 / 右偏
  · 不同状态：门开 / 门关 / 半开
```

#### 采集参数

| 参数 | 值 | 说明 |
|------|------|------|
| 飞行高度 | 250-350cm (z) | 覆盖门/窗的高度范围 |
| 距墙距离 | 200-600cm | 近/中/远三档 |
| 图片分辨率 | 640×480 或 1280×720 | 与 YOLO 训练 imgsz=640 匹配 |
| 保存格式 | PNG | 只需要 RGB，不需要 depth |

#### 快速截图脚本（可选辅助）

```python
import requests
import os
import time

class SimpleFrameCapture:
    """简单的截图工具——手动飞行，按需截图"""

    def __init__(self, api_base="http://localhost:8080", save_dir="./entry_detection_data/images"):
        self.api_base = api_base
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.frame_count = len(os.listdir(save_dir))  # 续接编号

    def capture(self, tag=""):
        """截取当前画面，tag 用来标记（如 'house1_door_near'）"""
        rgb_resp = requests.get(f"{self.api_base}/frame")
        filename = f"{tag}_{self.frame_count:04d}.png" if tag else f"frame_{self.frame_count:04d}.png"
        filepath = os.path.join(self.save_dir, filename)

        with open(filepath, "wb") as f:
            f.write(rgb_resp.content)

        self.frame_count += 1
        print(f"[{self.frame_count}] Saved: {filepath}")
        return filepath

    def batch_capture(self, tag="", count=5, interval=1.0):
        """连续截取多张（手动移动时用，每隔 interval 秒拍一张）"""
        for i in range(count):
            self.capture(tag=f"{tag}_seq{i}")
            time.sleep(interval)
        print(f"连续截取 {count} 张完成")


# === 使用方法 ===
# 1. 启动 UE4 + AirSim
# 2. 用 panel 手动飞到房屋附近
# 3. 运行：
if __name__ == "__main__":
    cap = SimpleFrameCapture()

    # 飞到 house_1 门前 → 手动截图
    cap.capture(tag="house1_door_open_near")

    # 或者连续拍 5 张（边飞边拍）
    cap.batch_capture(tag="house1_front", count=5, interval=1.0)
```

#### 采集目录结构

```
entry_detection_data/
├── images/                  # 所有 RGB 图片（手动采集）
│   ├── house1_door_open_near_0001.png
│   ├── house1_door_open_mid_0002.png
│   ├── house1_window_left_0003.png
│   ├── house2_door_closed_0004.png
│   ├── house2_opening_0005.png
│   ├── wall_negative_0006.png    # 负样本：纯墙面
│   └── ...
│
└── (标注后 Roboflow 导出 YOLO 格式)
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
```

#### 采集 → 标注 → 训练流程

```
Step A（现在做）              Step B（标注后做）
─────────────                ──────────────────
① 手动飞行截图 ~500 张        ④ 训练 YOLOv8n
② 上传到 Roboflow            ⑤ 验证 mAP
③ 拖框标注 4 个类别            ⑥ 导出 best.pt
   door_open / door_closed
   window / opening

                    ↓ YOLO 训练完成后 ↓

Step C-G（YOLO 训练好后，实时运行时自动采集）
──────────────────────────────────────────────
YOLO 在线推理 → 同时获取 depth 帧 → 验证 depth 模块
                                  → 触发 VLM 调用
                                  → 记录融合决策数据
                                  → 用于 MLP 融合训练
```

> **核心理念**：YOLO 训练和后续模块是**解耦**的。
> - YOLO 只需要图片和标注框，先训练好
> - depth 分析、VLM 确认、融合决策 → 都在 YOLO 训练好后的**实时运行**中使用和验证
> - 实时运行过程中自动记录 depth + VLM 数据，用于 Step E 的 MLP 融合训练

---

### Step B：YOLO 入口检测训练

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

#### 标注方法

用 Roboflow 拖框标注，4 个类别：

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

#### 标注注意事项

```
标注规范：
1. 框住完整的门框/窗框/洞口边缘，不要只框内部
2. 同一个门在不同距离拍的照片都要标注
3. door_open vs door_closed 的判断：
   - 能看到门后空间 → door_open
   - 门板关闭，看不到后面 → door_closed
   - 门半开 → 标为 door_open（能进就算）
4. window vs opening 的判断：
   - 有窗框结构 → window
   - 没有明显框架的洞/缺口 → opening
5. 负样本（纯墙面）不需要标注，YOLO 会自动学习
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

#### 数据增强

```python
# Roboflow 中推荐的增强方式：
augmentation_config = {
    "brightness": {"min": -20, "max": 20},    # 亮度变化（模拟不同光照）
    "blur": {"max": 1.5},                      # 轻微模糊（模拟运动）
    "noise": {"max": 2},                       # 轻微噪声
    "rotation": {"min": -5, "max": 5},         # 小角度旋转（模拟 UAV 姿态）
    "shear": {"min": -5, "max": 5},            # 剪切变换
    "mosaic": True,                            # Mosaic 增强
}

# 不推荐的增强：
# - 大角度旋转（UAV 视角不会翻转）
# - 水平翻转（门的左右信息有意义）
# - 大幅缩放（已有 3 个距离档位覆盖）
```

#### 训练命令

```bash
# 训练 YOLOv8n（轻量版，适合实时推理）
yolo train \
  model=yolov8n.pt \
  data=phase2_entry_detector.yaml \
  epochs=120 \
  imgsz=640 \
  batch=16 \
  project=runs/entry_detector \
  name=v1

# 训练完成后验证
yolo val \
  model=runs/entry_detector/v1/weights/best.pt \
  data=phase2_entry_detector.yaml

# 测试单张图片
yolo predict \
  model=runs/entry_detector/v1/weights/best.pt \
  source=entry_detection_data/images/val/ \
  save=True \
  conf=0.5
```

#### YOLO 验证标准

| 指标 | 要求 | 原因 |
|------|------|------|
| door_open mAP@0.5 | ≥ 0.85 | 主要入口类型，需要高精度 |
| door_open recall | ≥ 0.90 | 不能漏检可进入的门 |
| door_closed precision | ≥ 0.80 | 避免误报为可进入 |
| opening recall | ≥ 0.75 | 非标准入口可接受较低召回 |
| 整体 mAP@0.5 | ≥ 0.80 | 综合指标 |
| 推理速度（640px） | ≤ 8ms / 帧 | 实时性要求 |

#### YOLO 常见问题排查

```
问题 1: door_open recall 低（< 0.85）
  原因：远距离的门太小，YOLO 检测不到
  解决：增加近距离（200cm）的采集比例；降低 conf 阈值到 0.4

问题 2: door_closed 和 door_open 混淆
  原因：半开的门标注不一致
  解决：统一标注规范——能看到门后空间就标 door_open

问题 3: window 和 opening 混淆
  原因：窗框不明显时难以区分
  解决：这两类混淆可接受，因为后续 depth 分析会统一判断可穿越性

问题 4: 负样本上大量误检
  原因：墙面纹理被误认为入口
  解决：增加负样本数量到 150-200 张
```

---

### Step C：Depth 分析模块实现

YOLO 告诉你"这里有个门"，depth 告诉你"能不能过去"。

#### depth 分析核心代码

```python
import numpy as np

def analyze_entry_depth(depth_frame, entry_bbox, frame_width, frame_height):
    """
    对 YOLO 检测到的入口区域做 depth 分析。

    Args:
        depth_frame: np.ndarray, shape=(H, W), 单位 cm
        entry_bbox: [x1, y1, x2, y2] 像素坐标
        frame_width: 图像宽度
        frame_height: 图像高度
    
    Returns:
        dict: 距离、尺寸估算、可穿越性判断
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

#### depth 决策逻辑

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

#### Depth 在线验证与调参

> **注意**：Depth 模块不需要离线数据集。YOLO 训练好后，在 UE4 中实时运行时，
> YOLO 检测到入口 → 同步获取 depth 帧 → 调用 `analyze_entry_depth()` → 观察结果是否合理。

```python
class DepthModuleValidator:
    """
    在实时运行中记录 depth 分析结果，用于调参和统计。

    使用方法：
    1. YOLO 训练好后，用 panel 手动飞到入口附近
    2. YOLO 在线检测到入口 → 自动获取 depth 分析
    3. 人工判断 depth 结果是否正确（通过 UI 显示或日志）
    4. 记录每次的 depth 结果 + 人工判断 → 用于统计准确率和调参
    """

    def __init__(self, log_path="depth_validation_log.jsonl"):
        self.log_path = log_path
        self.results = []

    def validate_and_log(self, depth_result, yolo_result, human_judgment=None):
        """
        记录一次 depth 分析结果。

        Args:
            depth_result: analyze_entry_depth() 的输出
            yolo_result: YOLO 检测结果
            human_judgment: 人工判断（可选），格式：
                {"traversable": True/False, "notes": "门确实开着"}
        """
        import json, time

        record = {
            "timestamp": time.time(),
            "yolo_class": yolo_result.get("class"),
            "yolo_confidence": yolo_result.get("confidence"),
            "depth_distance_cm": depth_result.get("distance_cm"),
            "depth_ratio": depth_result.get("depth_ratio"),
            "depth_traversable": depth_result.get("traversable"),
            "depth_size_sufficient": depth_result.get("size_sufficient"),
            "human_judgment": human_judgment
        }

        self.results.append(record)

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def print_stats(self):
        """打印统计（只统计有人工判断的记录）"""
        judged = [r for r in self.results if r.get("human_judgment")]
        if not judged:
            print("还没有人工标注的记录，继续采集...")
            return

        tp = sum(1 for r in judged
                 if r["depth_traversable"] and r["human_judgment"]["traversable"])
        fp = sum(1 for r in judged
                 if r["depth_traversable"] and not r["human_judgment"]["traversable"])
        fn = sum(1 for r in judged
                 if not r["depth_traversable"] and r["human_judgment"]["traversable"])
        tn = sum(1 for r in judged
                 if not r["depth_traversable"] and not r["human_judgment"]["traversable"])

        total = tp + fp + fn + tn
        print(f"Depth 在线验证统计（{total} 条记录）：")
        print(f"  准确率: {(tp+tn)/total:.2%}")
        print(f"  精确率: {tp/(tp+fp+1e-6):.2%}")
        print(f"  召回率: {tp/(tp+fn+1e-6):.2%}")
        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
```

#### Depth 调参指南

| 参数 | 默认值 | 调整范围 | 调整依据 |
|------|--------|---------|---------|
| `depth_ratio` 阈值 | 1.3 | 1.1-1.5 | 太低→关闭的门也判为开；太高→半开的门误判 |
| `obstacle_threshold` | 80cm | 50-120cm | 太低→正常门框也判为障碍；太高→真实障碍漏检 |
| `min_width_cm` | 80cm | 60-100cm | 取决于 UAV 实际宽度 + 安全余量 |
| `min_height_cm` | 60cm | 40-80cm | 取决于 UAV 实际高度 + 安全余量 |
| `valid_depth_range` | 50-5000cm | 根据场景调 | 过滤 depth 噪声 |
| `distance_percentile` | 20th | 10-30th | 较低值更保守（取更近的距离估计） |

---

### Step D：VLM 部署与 Prompt 调优

VLM（视觉语言模型）是第三道确认——用自然语言理解画面内容，解决 YOLO 和 depth 无法回答的问题。

#### VLM 模型选择

| 模型 | 部署方式 | 显存需求 | 推理速度 | 准确度 | 推荐 |
|------|---------|---------|---------|--------|------|
| LLaVA-1.5-7B | 本地 | ~14GB | ~500ms/张 | 中等 | ★★★ 首选 |
| LLaVA-1.5-13B | 本地 | ~26GB | ~1s/张 | 较高 | ★★ 有大显存时用 |
| GPT-4o | API | 无需 | ~1-2s/张 | 最高 | ★★★ 精度优先 |
| Claude 3.5 Sonnet | API | 无需 | ~1-2s/张 | 最高 | ★★ 备选 |
| Qwen-VL-Chat | 本地 | ~16GB | ~600ms/张 | 中等 | ★ 备选 |

**推荐方案**：
- 训练/调参阶段：GPT-4o（最高精度，建立 ground truth）
- 部署/蒸馏阶段：LLaVA-1.5-7B（本地推理，无 API 依赖）

#### VLM 本地部署（LLaVA）

```bash
# 安装 LLaVA
pip install llava-torch

# 或者使用 ollama（更简单）
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llava:7b

# 验证安装
ollama run llava:7b "Describe this image" --image test.png
```

```python
# LLaVA 本地调用封装
import ollama
import base64

class LocalVLM:
    """本地 VLM 封装（基于 ollama + LLaVA）"""
    
    def __init__(self, model_name="llava:7b"):
        self.model_name = model_name
    
    def analyze(self, image_path, prompt):
        """分析图片并返回文本描述"""
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        response = ollama.chat(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_data]
            }]
        )
        return response["message"]["content"]


class APIVLM:
    """API VLM 封装（GPT-4o）"""
    
    def __init__(self, api_key=None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)
    
    def analyze(self, image_path, prompt):
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{image_data}"
                    }}
                ]
            }],
            max_tokens=300
        )
        return response.choices[0].message.content
```

#### VLM Prompt 设计

```python
# === Prompt V1：通用入口分析 ===
VLM_ENTRY_PROMPT_V1 = """Look at this image from a UAV camera approaching a building.

Answer these questions concisely:
1. What potential entry points do you see? (doors, windows, openings, gaps)
2. For each entry point: is it open or closed? Is it large enough for a small drone (60cm wide) to fly through?
3. Are there any obstacles near the entry points?
4. Which entry point would you recommend for a drone to enter, and why?

Be specific about positions (left/center/right of frame).
"""

# === Prompt V2：带 YOLO 检测结果的确认 Prompt ===
VLM_CONFIRM_PROMPT_V2 = """A drone's camera captured this image while approaching a building.
A detector found a potential {entry_type} at the {position} of the frame.

Please verify:
1. Is there really a {entry_type} at that location? (yes/no)
2. Is it open enough for a 60cm-wide drone to fly through? (yes/no)  
3. Any obstacles blocking the entry? (yes/no)
4. Confidence level of your assessment? (high/medium/low)

Answer in this exact format:
ENTRY_EXISTS: yes/no
PASSABLE: yes/no
OBSTACLE: yes/no
CONFIDENCE: high/medium/low
REASON: <one sentence explanation>
"""

# === Prompt V3：无检测结果时的全局扫描 ===
VLM_SCAN_PROMPT_V3 = """A search-and-rescue drone is looking at a building exterior.
No entry points were detected by the object detector.

Carefully examine the image and identify ANY possible way a small drone (60cm wide) could enter:
- Partially open doors
- Open windows (no glass or screen)
- Holes or gaps in walls
- Damaged sections
- Garage openings
- Ventilation openings

If you find anything, describe its location and size.
If there truly is no entry point, say "NO_ENTRY_FOUND".
"""
```

#### VLM 回答解析器

```python
def parse_vlm_response(response_text, prompt_version="v1"):
    """
    结构化解析 VLM 的回答。
    
    Args:
        response_text: VLM 返回的原始文本
        prompt_version: 使用的 prompt 版本
    
    Returns:
        dict: 结构化的 VLM 判断结果
    """
    text = response_text.lower()
    
    if prompt_version == "v2":
        # V2 prompt 有固定格式，直接解析
        result = {}
        for line in response_text.strip().split("\n"):
            if "ENTRY_EXISTS:" in line:
                result["entry_exists"] = "yes" in line.lower()
            elif "PASSABLE:" in line:
                result["passable"] = "yes" in line.lower()
            elif "OBSTACLE:" in line:
                result["has_obstacle"] = "yes" in line.lower()
            elif "CONFIDENCE:" in line:
                conf_text = line.split(":")[-1].strip().lower()
                result["confidence"] = {"high": 0.9, "medium": 0.6, "low": 0.3}.get(conf_text, 0.5)
            elif "REASON:" in line:
                result["reason"] = line.split(":", 1)[-1].strip()
        
        result["mentions_open"] = result.get("passable", False)
        result["mentions_obstacle"] = result.get("has_obstacle", False)
        return result
    
    else:
        # V1/V3 prompt 用关键词匹配
        return {
            "vlm_description": response_text,
            "has_entry_recommendation": "recommend" in text,
            "mentions_obstacle": any(w in text for w in [
                "obstacle", "blocked", "closed", "barrier",
                "cannot", "too small", "narrow"
            ]),
            "mentions_open": any(w in text for w in [
                "open", "can fly", "large enough", "passable",
                "wide enough", "accessible"
            ]),
            "mentions_no_entry": "no_entry_found" in text or "no entry" in text
        }
```

#### VLM 调用策略（Adaptive Calling）

这是论文的**创新点之一**——自适应 VLM 调用策略，平衡准确率和延迟。

```python
class AdaptiveVLMCaller:
    """
    自适应 VLM 调用策略。
    
    核心思想：不是每帧都调 VLM，而是根据 YOLO+depth 的置信度决定是否需要 VLM。
    
    调用条件矩阵：
    ┌────────────────┬────────────────┬────────────────┐
    │ YOLO 检测      │ depth 判断     │ 是否调用 VLM   │
    ├────────────────┼────────────────┼────────────────┤
    │ door_open ≥0.8 │ traversable    │ ✓ 调用（最终确认）│
    │ door_open ≥0.8 │ closed/blocked │ ✓ 调用（仲裁）   │
    │ door_open <0.8 │ traversable    │ ✓ 调用（补充确认）│
    │ opening ≥0.6   │ traversable    │ ✓ 调用（类型确认）│
    │ window ≥0.7    │ size_sufficient│ ✓ 调用（可行性确认）│
    │ door_closed    │ any            │ ✗ 不调用        │
    │ 无检测          │ N/A            │ 每圈结束调用 1 次 │
    └────────────────┴────────────────┴────────────────┘
    """
    
    def __init__(self, vlm_model, cooldown_steps=5):
        self.vlm = vlm_model
        self.cooldown_steps = cooldown_steps   # 两次 VLM 调用之间的最小步数
        self.last_call_step = -999
        self.call_count = 0
        self.call_history = []
    
    def should_call_vlm(self, yolo_result, depth_result, current_step):
        """判断当前步是否需要调用 VLM"""
        # 冷却期内不调用
        if current_step - self.last_call_step < self.cooldown_steps:
            return False, "cooldown"
        
        # 无 YOLO 检测 → 不调用（除非整圈结束）
        if not yolo_result:
            return False, "no_detection"
        
        cls = yolo_result.get("class", "")
        conf = yolo_result.get("confidence", 0)
        
        # door_closed → 不调用
        if cls == "door_closed":
            return False, "door_closed_skip"
        
        # door_open 高置信度 + depth 可穿越 → 调用（最终确认）
        if cls == "door_open" and conf >= 0.8:
            if depth_result and depth_result.get("traversable"):
                return True, "final_confirm"
            elif depth_result:
                return True, "arbitrate"  # depth 和 YOLO 矛盾，VLM 仲裁
        
        # door_open 低置信度 + depth 可穿越 → 调用（补充确认）
        if cls == "door_open" and conf < 0.8:
            if depth_result and depth_result.get("traversable"):
                return True, "supplement_confirm"
        
        # opening → 调用（确认类型）
        if cls == "opening" and conf >= 0.6:
            if depth_result and depth_result.get("traversable"):
                return True, "type_confirm"
        
        # window + 尺寸足够 → 调用（可行性确认）
        if cls == "window" and conf >= 0.7:
            if depth_result and depth_result.get("size_sufficient"):
                return True, "feasibility_confirm"
        
        return False, "no_trigger"
    
    def call_vlm(self, rgb_path, yolo_result, current_step):
        """执行 VLM 调用"""
        cls = yolo_result.get("class", "") if yolo_result else ""
        
        # 选择合适的 prompt
        if yolo_result:
            bbox = yolo_result.get("bbox", [0,0,0,0])
            cx = (bbox[0] + bbox[2]) / 2
            position = "left" if cx < 213 else ("center" if cx < 427 else "right")
            prompt = VLM_CONFIRM_PROMPT_V2.format(entry_type=cls, position=position)
            version = "v2"
        else:
            prompt = VLM_SCAN_PROMPT_V3
            version = "v3"
        
        # 调用 VLM
        raw_response = self.vlm.analyze(rgb_path, prompt)
        parsed = parse_vlm_response(raw_response, prompt_version=version)
        
        # 记录
        self.last_call_step = current_step
        self.call_count += 1
        self.call_history.append({
            "step": current_step,
            "prompt_version": version,
            "result": parsed
        })
        
        return parsed
    
    def get_stats(self):
        """获取 VLM 调用统计"""
        return {
            "total_calls": self.call_count,
            "call_history": self.call_history
        }
```

**VLM 调用频率分析（论文数据）**：

```
场景                    │ 预计绕飞步数 │ VLM 调用次数 │ 调用率
────────────────────────┼──────────────┼──────────────┼────────
有明显大门的房屋          │ 20-30 步     │ 1-2 次       │ ~5-10%
多个候选入口的房屋        │ 25-35 步     │ 3-5 次       │ ~10-15%
无入口（需全局扫描）      │ 30-40 步     │ 1 次（圈末） │ ~3%

平均调用率：5-10%（vs 每步都调的 100%）
延迟节省：90-95% 的 VLM 推理时间
```

---

### Step E：融合决策模块

三个模态的结果汇总到融合决策模块，做出最终的 enter/approach/skip 判断。

#### 方案 1：Rule-based 融合（基线方案）

```python
def fuse_entry_decision(yolo_result, depth_result, vlm_result=None, 
                         uav_pose=None, houses_config=None, target_house_id=None):
    """
    融合三种模态的入口检测结果，做出最终决策。

    决策层级：
      第 0 层：坐标验证（一票否决：不在目标房屋范围 → 忽略）
      第 1 层：距离判断（太远 → 先靠近）
      第 2 层：depth 可穿越性（一票否决：有障碍 → blocked）
      第 3 层：YOLO + depth + VLM 加权综合判断
    """

    # === 第 0 层：坐标验证 ===
    if uav_pose and houses_config and target_house_id:
        house = next(
            (h for h in houses_config["houses"] if h["house_id"] == target_house_id),
            None
        )
        if house:
            bbox = house["bbox"]
            margin = 200  # 200cm 容差
            uav_x, uav_y = uav_pose["x_val"], uav_pose["y_val"]
            if not (bbox["x_min"] - margin <= uav_x <= bbox["x_max"] + margin and
                    bbox["y_min"] - margin <= uav_y <= bbox["y_max"] + margin):
                return {
                    "decision": "ignore",
                    "reason": f"UAV 不在 {target_house_id} 范围内（可能是邻居的门）",
                    "confidence": 1.0
                }

    # === 第 1 层：距离判断 ===
    if depth_result and depth_result.get("distance_cm", 9999) > 500:
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
            "reason": f"开口太小 ({depth_result['estimated_width_cm']:.0f}×"
                      f"{depth_result['estimated_height_cm']:.0f}cm)",
            "confidence": 0.85
        }

    # === 第 3 层：YOLO + depth + VLM 加权综合判断 ===
    confidence = 0.0
    reasons = []

    # YOLO 贡献（权重按入口类型分级）
    if yolo_result:
        cls = yolo_result.get("class", "")
        yolo_conf = yolo_result.get("confidence", 0)
        yolo_weights = {
            "door_open": 0.4,
            "opening": 0.3,
            "window": 0.2,
        }
        w = yolo_weights.get(cls, 0)
        confidence += w * yolo_conf
        reasons.append(f"YOLO: {cls} ({yolo_conf:.2f})")

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

#### 方案 2：MLP Learned 融合（论文实验方案）

用于 TMM 论文的 **Experiment 2（融合策略对比）**。

```python
import torch
import torch.nn as nn

class MLPFusion(nn.Module):
    """
    可学习的 MLP 融合模块。
    
    输入：8 维特征向量（来自 YOLO + depth + VLM 的结构化输出）
    输出：enter 概率（0-1）
    
    特征定义：
      [0] yolo_detected      : YOLO 检测到入口（0/1）
      [1] yolo_class_score   : 类型加权分 (door=1.0, opening=0.75, window=0.5, none=0)
      [2] yolo_confidence    : YOLO 置信度（0-1）
      [3] depth_traversable  : depth 可穿越判断（0/1）
      [4] depth_ratio        : 中心/边缘 depth 比值（归一化到 0-1）
      [5] depth_size_ratio   : 开口尺寸 / UAV 尺寸（归一化到 0-1）
      [6] vlm_open           : VLM 确认开放（0/1，未调用时=0.5）
      [7] vlm_obstacle       : VLM 提到障碍（0/1，未调用时=0.5）
    """
    
    def __init__(self, input_dim=8, hidden_dims=[32, 16]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)
    
    @staticmethod
    def extract_features(yolo_result, depth_result, vlm_result=None):
        """从三模态结果中提取 8 维特征向量"""
        features = torch.zeros(8)
        
        if yolo_result:
            features[0] = 1.0
            cls_scores = {"door_open": 1.0, "opening": 0.75, "window": 0.5}
            features[1] = cls_scores.get(yolo_result.get("class", ""), 0)
            features[2] = yolo_result.get("confidence", 0)
        
        if depth_result:
            features[3] = 1.0 if depth_result.get("traversable") else 0.0
            features[4] = min(depth_result.get("depth_ratio", 1.0) / 3.0, 1.0)
            if depth_result.get("estimated_width_cm"):
                features[5] = min(depth_result["estimated_width_cm"] / 200.0, 1.0)
        
        if vlm_result:
            features[6] = 1.0 if vlm_result.get("mentions_open") else 0.0
            features[7] = 1.0 if vlm_result.get("mentions_obstacle") else 0.0
        else:
            features[6] = 0.5  # 未调用时用中性值
            features[7] = 0.5
        
        return features


class AttentionFusion(nn.Module):
    """
    Attention-based 融合模块（论文对比方案）。
    
    三个模态各自编码为 d 维向量，通过 cross-attention 融合。
    """
    
    def __init__(self, yolo_dim=3, depth_dim=3, vlm_dim=2, embed_dim=32, num_heads=4):
        super().__init__()
        # 各模态编码器
        self.yolo_encoder = nn.Linear(yolo_dim, embed_dim)
        self.depth_encoder = nn.Linear(depth_dim, embed_dim)
        self.vlm_encoder = nn.Linear(vlm_dim, embed_dim)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, yolo_feat, depth_feat, vlm_feat):
        # 编码
        y = self.yolo_encoder(yolo_feat).unsqueeze(1)   # (B, 1, D)
        d = self.depth_encoder(depth_feat).unsqueeze(1)  # (B, 1, D)
        v = self.vlm_encoder(vlm_feat).unsqueeze(1)      # (B, 1, D)
        
        # 拼接为序列 (B, 3, D)
        tokens = torch.cat([y, d, v], dim=1)
        
        # Self-attention
        attn_out, attn_weights = self.attention(tokens, tokens, tokens)
        
        # 拼接所有 token 并分类
        fused = attn_out.reshape(attn_out.size(0), -1)  # (B, 3*D)
        return self.classifier(fused), attn_weights
```

#### MLP 融合训练流程

```python
def train_mlp_fusion(data_path="fusion_training_data.json", epochs=50):
    """
    训练 MLP 融合模块。
    
    训练数据来源：Phase 2 在线采集时记录的三模态特征 + 人工标注的决策标签。
    """
    import json
    from torch.utils.data import DataLoader, TensorDataset
    
    # 加载训练数据
    with open(data_path, "r") as f:
        data = json.load(f)
    
    features = torch.tensor([d["features"] for d in data], dtype=torch.float32)
    labels = torch.tensor([d["label"] for d in data], dtype=torch.float32).unsqueeze(1)
    # label: 1.0 = 应该进入, 0.0 = 不应该进入
    
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = MLPFusion(input_dim=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_features, batch_labels in loader:
            pred = model(batch_features)
            loss = criterion(pred, batch_labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += ((pred > 0.5) == batch_labels).sum().item()
            total += len(batch_labels)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}, "
                  f"acc={correct/total:.2%}")
    
    torch.save(model.state_dict(), "mlp_fusion_v1.pt")
    return model
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

场景 5：YOLO 和 depth 矛盾
  YOLO: door_open (conf=0.91) → +0.36
  depth: depth_ratio=1.05, 显示为平面 → 不可穿越（一票否决前...）
  VLM 仲裁: "这是一幅门的壁画，不是真门" → -0.15
  → decision = "skip"（VLM 揭示了 YOLO 误检）
```

---

### Step F：绕飞策略与入口进入

#### 绕飞路径点生成算法

```python
import math

class HouseCirclingExplorer:
    """围绕房屋飞行探索入口的策略模块"""
    
    def __init__(self, house_config, step_cm=150):
        """
        Args:
            house_config: 单栋房屋的配置（来自 houses_config.json）
            step_cm: 每步沿切线方向移动的距离（cm）
        """
        self.house_id = house_config["house_id"]
        self.center = house_config["center"]  # [cx, cy]
        self.bbox = house_config["bbox"]
        self.radius = house_config["approach_radius_cm"]
        self.floor_z = house_config["floor_z_range"]
        self.step_cm = step_cm
        
        # 飞行高度 = 楼层中间 + 偏移（对齐门窗高度）
        self.fly_z = (self.floor_z[0] + self.floor_z[1]) / 2 + 50
        
        # 绕飞状态
        self.current_angle = 0.0        # 当前角度（弧度）
        self.total_angle = 0.0          # 累计角度
        self.step_count = 0
        self.candidates = []            # 候选入口列表
        self.best_candidate = None
    
    def generate_waypoints(self, num_points=24):
        """
        生成完整的绕飞路径点（用于预览和规划）。
        
        Args:
            num_points: 一圈的路径点数量（24 = 每 15° 一个点）
        
        Returns:
            list of (x, y, z, yaw_deg)
        """
        cx, cy = self.center
        waypoints = []
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            
            # 路径点位置（在圆周上）
            x = cx + self.radius * math.cos(angle)
            y = cy + self.radius * math.sin(angle)
            z = self.fly_z
            
            # 偏航：始终面朝房屋中心
            yaw_to_center = math.degrees(math.atan2(cy - y, cx - x))
            
            waypoints.append((x, y, z, yaw_to_center))
        
        return waypoints
    
    def get_next_move(self, current_state):
        """
        计算下一步的移动指令。
        
        Args:
            current_state: UAV 当前状态 {"x_val", "y_val", "z_val", "yaw_val"}
        
        Returns:
            dict: {"dx", "dy", "dz", "dyaw", "move_duration"}
        """
        cx, cy = self.center
        ux, uy = current_state["x_val"], current_state["y_val"]
        
        # 计算当前角度
        current_angle = math.atan2(uy - cy, ux - cx)
        
        # 下一个点的角度（逆时针方向 +）
        angle_step = self.step_cm / self.radius  # 弧度
        next_angle = current_angle + angle_step
        
        # 下一个路径点
        next_x = cx + self.radius * math.cos(next_angle)
        next_y = cy + self.radius * math.sin(next_angle)
        next_z = self.fly_z
        
        # 目标偏航（面朝房屋中心）
        target_yaw = math.degrees(math.atan2(cy - next_y, cx - next_x))
        current_yaw = current_state.get("yaw_val", 0)
        dyaw = target_yaw - current_yaw
        if dyaw > 180: dyaw -= 360
        if dyaw < -180: dyaw += 360
        
        # 更新状态
        self.current_angle = next_angle
        self.total_angle += angle_step
        self.step_count += 1
        
        return {
            "dx": next_x - ux,
            "dy": next_y - uy,
            "dz": next_z - current_state["z_val"],
            "dyaw": dyaw,
            "move_duration": 1.5
        }
    
    def is_circle_complete(self):
        """判断是否已经绕飞一整圈"""
        return self.total_angle >= 2 * math.pi * 0.95  # 允许 5% 误差
    
    def add_candidate(self, entry_info, uav_state):
        """记录一个候选入口"""
        self.candidates.append({
            "entry_info": entry_info,
            "uav_state": uav_state,
            "step": self.step_count
        })
    
    def get_best_candidate(self):
        """
        从候选入口中选择最佳的一个。
        
        优先级：
        1. fusion_confidence 最高
        2. 同等 confidence 下，door_open > opening > window
        3. 同类型下，距离最近的
        """
        if not self.candidates:
            return None
        
        type_priority = {"door_open": 3, "opening": 2, "window": 1}
        
        def score(c):
            info = c["entry_info"]
            type_score = type_priority.get(info.get("yolo_class", ""), 0)
            conf_score = info.get("fusion_confidence", 0)
            return (conf_score, type_score)
        
        self.best_candidate = max(self.candidates, key=score)
        return self.best_candidate
```

#### 动态调整策略

```
绕飞过程中的动态调整：

1. 发现候选入口（approach 决策）：
   ┌────────────────────────────────────────────┐
   │  当前绕飞轨迹     →→→                      │
   │                      ↗                     │
   │              候选入口 ⬟                     │
   │                    ↗                        │
   │  调整：减小半径  →→                          │
   │  目标：靠近目标面进行详细检测                  │
   └────────────────────────────────────────────┘
   
   radius_adjustment:
     approach 决策 → radius *= 0.6（靠近到 60% 半径）
     连续 3 步无检测 → radius 恢复原值

2. 发现高置信入口（enter 决策）：
   → 停止绕飞
   → 切换到入口进入流程

3. 绕飞一圈无入口：
   → 降低飞行高度 50cm，再飞一圈
   → 第二圈仍无入口 → 调用 VLM 全局扫描
   → VLM 也无发现 → 标记 "no_entry_found"
```

#### 入口进入流程

当 `fuse_entry_decision()` 返回 `"enter"` 后，执行以下精确进入流程：

```python
class EntryProcedure:
    """入口进入流程控制"""
    
    PHASE_ALIGN = "align"        # 对齐入口中心
    PHASE_APPROACH = "approach"  # 逼近入口
    PHASE_TRAVERSE = "traverse"  # 穿越入口
    PHASE_CONFIRM = "confirm"    # 确认进入成功
    
    def __init__(self, api_base="http://localhost:8080"):
        self.api_base = api_base
        self.phase = self.PHASE_ALIGN
        self.attempt_count = 0
        self.max_attempts = 3
    
    def align_to_entry(self, entry_bbox, frame_width, frame_height):
        """
        Phase 2.1 - 对齐：调整偏航使入口中心对齐图像中心
        
        Returns:
            dict: 需要执行的移动指令，或 None（已对齐）
        """
        # 入口中心
        entry_cx = (entry_bbox[0] + entry_bbox[2]) / 2
        entry_cy = (entry_bbox[1] + entry_bbox[3]) / 2
        
        # 图像中心
        img_cx = frame_width / 2
        img_cy = frame_height / 2
        
        # 水平偏移（归一化到 -1~+1）
        horizontal_offset = (entry_cx - img_cx) / img_cx
        vertical_offset = (entry_cy - img_cy) / img_cy
        
        # 如果偏移小于 5%，认为已对齐
        if abs(horizontal_offset) < 0.05 and abs(vertical_offset) < 0.10:
            self.phase = self.PHASE_APPROACH
            return None  # 已对齐
        
        # 计算调整指令
        move = {"dx": 0, "dy": 0, "dz": 0, "dyaw": 0, "move_duration": 0.5}
        
        # 水平偏移 → 偏航调整（正偏移 → 右转）
        if abs(horizontal_offset) >= 0.05:
            move["dyaw"] = horizontal_offset * 15  # 最大 15° 调整
        
        # 垂直偏移 → 高度调整（正偏移 → 下降）
        if abs(vertical_offset) >= 0.10:
            move["dz"] = vertical_offset * 30  # 最大 30cm 调整
        
        return move
    
    def approach_entry(self, depth_result):
        """
        Phase 2.2 - 逼近：以小步前进靠近入口
        
        每步前进 50cm，并重新检测确认入口仍然存在。
        """
        distance = depth_result.get("distance_cm", 9999)
        
        if distance < 100:
            # 已经足够近，切换到穿越阶段
            self.phase = self.PHASE_TRAVERSE
            return None
        
        # 前进 50cm（或距离的 1/3，取较小值）
        step_size = min(50, distance / 3)
        
        return {
            "dx": step_size, "dy": 0, "dz": 0,
            "dyaw": 0, "move_duration": 1.0
        }
    
    def traverse_entry(self):
        """
        Phase 2.3 - 穿越：匀速前进穿过入口
        
        前进 200cm 穿过门/窗洞口。
        """
        self.phase = self.PHASE_CONFIRM
        
        return {
            "dx": 200, "dy": 0, "dz": 0,
            "dyaw": 0, "move_duration": 3.0  # 慢速穿越
        }
    
    def confirm_entry(self, depth_frame, vlm_model=None, rgb_path=None):
        """
        Phase 2.4 - 确认：判断是否成功进入室内
        
        判断方法：
        1. depth 分布变化：室内 depth 更近（四周墙壁）
        2. VLM 确认（可选）
        """
        # 分析四周 depth
        h, w = depth_frame.shape
        
        # 取四个方向的 depth 均值
        left_depth = np.median(depth_frame[:, :w//4])
        right_depth = np.median(depth_frame[:, 3*w//4:])
        top_depth = np.median(depth_frame[:h//4, :])
        center_depth = np.median(depth_frame[h//4:3*h//4, w//4:3*w//4])
        
        # 室内特征：四周 depth 较近（被墙壁围绕）
        surrounding_depth = np.mean([left_depth, right_depth, top_depth])
        is_indoor = surrounding_depth < 600  # 室内四周 depth < 6m
        
        # VLM 确认
        vlm_confirms = False
        if vlm_model and rgb_path:
            indoor_prompt = ("Is this image taken from inside a building/room? "
                           "Answer YES or NO, then briefly explain.")
            response = vlm_model.analyze(rgb_path, indoor_prompt)
            vlm_confirms = "yes" in response.lower()
        
        return {
            "is_indoor": is_indoor or vlm_confirms,
            "surrounding_depth": round(surrounding_depth, 1),
            "center_depth": round(center_depth, 1),
            "vlm_confirms": vlm_confirms
        }
    
    def execute_step(self, yolo_result, depth_result, depth_frame,
                     frame_width, frame_height, vlm_model=None, rgb_path=None):
        """
        执行进入流程的一步。
        
        Returns:
            dict: {"phase", "move", "status", "details"}
        """
        if self.phase == self.PHASE_ALIGN:
            if not yolo_result:
                self.attempt_count += 1
                if self.attempt_count >= self.max_attempts:
                    return {"phase": self.phase, "status": "failed",
                            "reason": "入口消失，对齐失败"}
                return {"phase": self.phase, "status": "retry",
                        "reason": "入口暂时消失，重新检测"}
            
            move = self.align_to_entry(
                yolo_result["bbox"], frame_width, frame_height
            )
            if move is None:
                return {"phase": self.PHASE_APPROACH, "status": "aligned"}
            return {"phase": self.phase, "move": move, "status": "aligning"}
        
        elif self.phase == self.PHASE_APPROACH:
            # 每步检查入口是否仍在
            if not yolo_result or not depth_result:
                self.attempt_count += 1
                if self.attempt_count >= self.max_attempts:
                    return {"phase": self.phase, "status": "failed",
                            "reason": "逼近过程中入口消失"}
                return {"phase": self.phase, "status": "retry"}
            
            # 检查是否仍然可穿越
            if depth_result.get("has_obstacle"):
                return {"phase": self.phase, "status": "blocked",
                        "reason": "逼近时发现障碍物"}
            
            move = self.approach_entry(depth_result)
            if move is None:
                return {"phase": self.PHASE_TRAVERSE, "status": "close_enough"}
            return {"phase": self.phase, "move": move, "status": "approaching"}
        
        elif self.phase == self.PHASE_TRAVERSE:
            move = self.traverse_entry()
            return {"phase": self.phase, "move": move, "status": "traversing"}
        
        elif self.phase == self.PHASE_CONFIRM:
            result = self.confirm_entry(depth_frame, vlm_model, rgb_path)
            if result["is_indoor"]:
                return {"phase": self.phase, "status": "success",
                        "details": result}
            else:
                return {"phase": self.phase, "status": "uncertain",
                        "details": result,
                        "reason": "不确定是否已进入室内"}
```

#### 完整探索主循环

```python
def phase2_main_loop(house_config, houses_config, api_base="http://localhost:8080"):
    """
    Phase 2 的完整主循环：绕飞 → 检测 → 融合 → 进入。
    
    Args:
        house_config: 目标房屋的配置
        houses_config: 全部房屋配置
    """
    import requests
    
    # 初始化模块
    yolo_model = load_yolo("runs/entry_detector/v1/weights/best.pt")
    vlm = LocalVLM("llava:7b")
    vlm_caller = AdaptiveVLMCaller(vlm, cooldown_steps=5)
    explorer = HouseCirclingExplorer(house_config)
    
    stage = "circling"  # circling → entering → done
    entry_procedure = None
    max_circles = 2
    circle_count = 0
    
    print(f"\n开始探索 {house_config['house_id']}")
    print(f"绕飞半径: {house_config['approach_radius_cm']}cm")
    
    while True:
        # === 获取传感器数据 ===
        state = requests.get(f"{api_base}/state").json()
        rgb_resp = requests.get(f"{api_base}/frame")
        depth_resp = requests.get(f"{api_base}/depth_frame")
        
        # 保存当前帧（供 VLM 使用）
        rgb_path = "/tmp/current_frame.png"
        with open(rgb_path, "wb") as f:
            f.write(rgb_resp.content)
        depth_frame = np.frombuffer(depth_resp.content, dtype=np.float32).reshape(480, 640)
        
        # === YOLO 入口检测（每步都跑，≤8ms）===
        yolo_results = yolo_model.predict(rgb_path, conf=0.5)
        best_yolo = get_best_detection(yolo_results)  # 取最高置信度
        
        if stage == "circling":
            # === 绕飞模式 ===
            
            if best_yolo and best_yolo["class"] != "door_closed":
                # 有候选入口 → depth 分析
                depth_result = analyze_entry_depth(
                    depth_frame, best_yolo["bbox"], 640, 480
                )
                
                # 判断是否需要 VLM
                need_vlm, reason = vlm_caller.should_call_vlm(
                    best_yolo, depth_result, explorer.step_count
                )
                vlm_result = None
                if need_vlm:
                    vlm_result = vlm_caller.call_vlm(
                        rgb_path, best_yolo, explorer.step_count
                    )
                
                # 融合决策
                decision = fuse_entry_decision(
                    best_yolo, depth_result, vlm_result,
                    uav_pose=state,
                    houses_config=houses_config,
                    target_house_id=house_config["house_id"]
                )
                
                print(f"  Step {explorer.step_count}: "
                      f"{best_yolo['class']}({best_yolo['confidence']:.2f}) "
                      f"→ {decision['decision']} ({decision['confidence']:.2f})")
                
                if decision["decision"] == "enter":
                    # 切换到进入模式
                    stage = "entering"
                    entry_procedure = EntryProcedure(api_base)
                    explorer.add_candidate(decision, state)
                    print(f"  >>> 开始进入流程!")
                    continue
                
                elif decision["decision"] == "approach":
                    # 靠近候选入口
                    explorer.add_candidate(decision, state)
                    # 减小绕飞半径
                    explorer.radius *= 0.6
                
            else:
                depth_result = None
            
            # 检查是否绕完一圈
            if explorer.is_circle_complete():
                circle_count += 1
                print(f"  绕飞第 {circle_count} 圈完成")
                
                if explorer.candidates:
                    # 有候选，去最佳候选点
                    best = explorer.get_best_candidate()
                    print(f"  最佳候选: {best['entry_info']}")
                    # 导航到最佳候选位置重新检测
                    # ...（省略导航代码）
                elif circle_count < max_circles:
                    # 降低高度再来一圈
                    explorer.fly_z -= 50
                    explorer.total_angle = 0
                    print(f"  降低高度到 {explorer.fly_z}cm，开始第 {circle_count+1} 圈")
                else:
                    # 两圈都没找到 → VLM 全局扫描
                    vlm_scan = vlm_caller.call_vlm(rgb_path, None, explorer.step_count)
                    if vlm_scan.get("mentions_no_entry"):
                        print(f"  {house_config['house_id']}: 无可进入入口")
                        return {"status": "no_entry", "house_id": house_config["house_id"]}
                    else:
                        print(f"  VLM 发现可能的入口，继续探索...")
            
            # 执行下一步绕飞移动
            move = explorer.get_next_move(state)
            requests.post(f"{api_base}/move_relative", json=move)
        
        elif stage == "entering":
            # === 进入模式 ===
            depth_result = None
            if best_yolo:
                depth_result = analyze_entry_depth(
                    depth_frame, best_yolo["bbox"], 640, 480
                )
            
            result = entry_procedure.execute_step(
                best_yolo, depth_result, depth_frame,
                640, 480, vlm, rgb_path
            )
            
            print(f"  进入流程: {result['phase']} → {result['status']}")
            
            if result["status"] == "success":
                print(f"  >>> 成功进入 {house_config['house_id']}!")
                return {
                    "status": "entered",
                    "house_id": house_config["house_id"],
                    "entry_type": best_yolo["class"] if best_yolo else "unknown",
                    "vlm_stats": vlm_caller.get_stats()
                }
            
            elif result["status"] == "failed":
                print(f"  进入失败: {result.get('reason')}")
                # 后退并返回绕飞
                requests.post(f"{api_base}/move_relative", json={
                    "dx": -200, "dy": 0, "dz": 0, "dyaw": 0, "move_duration": 2.0
                })
                stage = "circling"
                entry_procedure = None
            
            elif "move" in result:
                requests.post(f"{api_base}/move_relative", json=result["move"])
```

---

### Step G：集成测试与端到端验证

#### YOLO 单独验证（YOLO 训练后立即做）

```bash
# YOLO 训练完成后，先单独验证 YOLO 检测效果
# 这一步不需要 depth、VLM，只需要 RGB 图片

# 1. 在验证集上评估
yolo val \
  model=runs/entry_detector/v1/weights/best.pt \
  data=phase2_entry_detector.yaml

# 2. 可视化检测结果（看标注是否合理）
yolo predict \
  model=runs/entry_detector/v1/weights/best.pt \
  source=entry_detection_data/val/images/ \
  save=True \
  conf=0.5

# 3. 确认指标达标后，再上线实时运行（Step C-G）
```

#### 在线实时验证（YOLO 上线后做）

> YOLO 训练好后，在 UE4 中实时运行，所有模块（depth/VLM/融合）在运行过程中同步验证。
> 不需要预先采集离线数据集——数据在运行中自动产生。

```python
class OnlinePipelineValidator:
    """
    在实时运行中记录和统计完整 pipeline 的表现。

    运行方式：
    1. YOLO 训练好后加载模型
    2. 用 panel 手动飞 UAV 或运行绕飞策略
    3. 每步自动记录：YOLO 检测 + depth 分析 + 融合决策
    4. 人工观察并记录 ground truth（这个入口该不该进）
    """

    def __init__(self, log_path="online_validation_log.jsonl"):
        self.log_path = log_path
        self.stats = {
            "total_steps": 0,
            "yolo_detections": 0,
            "depth_traversable": 0,
            "vlm_calls": 0,
            "fusion_enter": 0,
            "fusion_approach": 0,
            "fusion_skip": 0,
            "fusion_blocked": 0,
        }

    def record_step(self, yolo_result, depth_result, vlm_result, fusion_decision):
        """每步自动记录"""
        import json, time

        self.stats["total_steps"] += 1
        if yolo_result:
            self.stats["yolo_detections"] += 1
        if depth_result and depth_result.get("traversable"):
            self.stats["depth_traversable"] += 1
        if vlm_result:
            self.stats["vlm_calls"] += 1
        if fusion_decision:
            key = f"fusion_{fusion_decision['decision']}"
            self.stats[key] = self.stats.get(key, 0) + 1

        record = {
            "timestamp": time.time(),
            "yolo": yolo_result,
            "depth_traversable": depth_result.get("traversable") if depth_result else None,
            "vlm_called": vlm_result is not None,
            "fusion_decision": fusion_decision.get("decision") if fusion_decision else None,
            "fusion_confidence": fusion_decision.get("confidence") if fusion_decision else None,
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\\n")

    def print_report(self):
        """打印在线验证报告"""
        print("=" * 50)
        print("Phase 2 在线验证报告")
        print("=" * 50)
        for k, v in self.stats.items():
            print(f"  {k}: {v}")
        if self.stats["total_steps"] > 0:
            print(f"  YOLO 检测率: {self.stats['yolo_detections']/self.stats['total_steps']:.1%}")
            print(f"  VLM 调用率: {self.stats['vlm_calls']/self.stats['total_steps']:.1%}")
```

#### 在线端到端测试

```
在线测试步骤（需要 UE4 + AirSim 运行）：

测试 1：单栋房屋入口发现
  设置：UAV 从 house_1 的 approach_radius 外开始
  预期：绕飞一圈内发现入口并成功进入
  通过标准：成功率 ≥ 90%（10 次测试中 ≥ 9 次成功）

测试 2：多栋房屋连续探索
  设置：3 栋房屋，UAV 按 Phase 3 导航依次探索
  预期：每栋都能正确判断有无入口
  通过标准：判断准确率 ≥ 85%

测试 3：困难场景
  设置：半开的门、小窗户、装饰画干扰
  预期：VLM 能有效仲裁 YOLO/depth 的误判
  通过标准：VLM 仲裁准确率 ≥ 80%

测试 4：性能测试
  测量：每步的端到端延迟
  通过标准：
    - 无 VLM 步: ≤ 15ms（YOLO 8ms + depth 5ms + 融合 2ms）
    - 有 VLM 步: ≤ 600ms（+ VLM 500ms）
    - VLM 调用率: ≤ 15%
```

#### 失败恢复策略

```
边界情况及处理策略：

┌───────────────────────────────────────────────────────────────┐
│ 情况                    │ 处理                                │
├───────────────────────────────────────────────────────────────┤
│ 1. 绕飞一圈无入口       │ VLM 全局分析 →                      │
│                         │ 尝试第二圈（降低高度 50cm）→          │
│                         │ 仍无 → 标记 "no_entry"               │
├───────────────────────────────────────────────────────────────┤
│ 2. 进入尝试失败         │ 后退 200cm →                         │
│    (碰撞/卡住)          │ 记录失败入口坐标 →                    │
│                         │ 尝试下一个候选入口                    │
├───────────────────────────────────────────────────────────────┤
│ 3. YOLO 大量误检        │ 提高 confidence 阈值到 0.8            │
│    (>5个候选/面)         │ 增加 VLM 调用频率                    │
├───────────────────────────────────────────────────────────────┤
│ 4. VLM 超时/不可用      │ 降级为 YOLO+depth 双模态决策          │
│                         │ 提高 enter 阈值到 0.8                 │
├───────────────────────────────────────────────────────────────┤
│ 5. depth 帧异常         │ 仅用 YOLO 判断 + 保守决策             │
│    (全黑/全白)          │ approach 代替 enter                   │
├───────────────────────────────────────────────────────────────┤
│ 6. 多个可进入入口       │ 优先级排序：                          │
│                         │ door_open > opening > window          │
│                         │ 同类取 fusion_confidence 最高者        │
│                         │ 同分取距离最近者                       │
├───────────────────────────────────────────────────────────────┤
│ 7. 进入后发现是死胡同   │ 后退穿回入口 →                        │
│                         │ 标记该入口 "dead_end" →               │
│                         │ 继续绕飞找下一个                      │
└───────────────────────────────────────────────────────────────┘
```

---

### Phase 2 → Phase 3 接口

```python
# Phase 2 完成后输出给后续阶段的结果
phase2_output = {
    "house_id": "house_1",
    "status": "entered",           # entered / no_entry / failed
    
    # 入口信息
    "entry_type": "door_open",     # door_open / window / opening
    "entry_position": {            # 入口在世界坐标中的位置
        "x": 2680.0,
        "y": 120.0,
        "z": 280.0
    },
    "fusion_confidence": 0.92,
    
    # 探索统计
    "circling_steps": 22,
    "vlm_calls": 2,
    "candidates_found": 3,
    "time_elapsed_sec": 45.0,
    
    # 蒸馏特征快照（最后一步的 8 维特征）
    "distillation_features": {
        "entry_detected": 1,
        "entry_type": 1.0,         # door=1
        "entry_confidence": 0.92,
        "entry_distance_norm": 0.36,
        "entry_alignment_norm": 0.04,
        "entry_traversable": 1,
        "entry_size_ratio": 1.5,
        "fusion_confidence": 0.92
    }
}

# 根据 Phase 2 结果决定下一步：
if phase2_output["status"] == "entered":
    # → 切换到 Phase 4（室内搜索）
    update_search_status(house_id, "in_progress")
    start_phase4_indoor_search(house_id)

elif phase2_output["status"] == "no_entry":
    # → 回到 Phase 3（选择下一栋房屋）
    update_search_status(house_id, "no_entry")
    next_house = select_next_house(houses_config)
    start_phase3_navigation(next_house)

elif phase2_output["status"] == "failed":
    # → 记录失败，回到 Phase 3
    update_search_status(house_id, "entry_failed")
    next_house = select_next_house(houses_config)
    start_phase3_navigation(next_house)
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

### 验证标准总览

| 验证项 | 通过标准 | 测试方式 |
|--------|---------|---------|
| YOLO door_open mAP@0.5 | ≥ 0.85 | yolo val |
| YOLO door_open recall | ≥ 0.90 | yolo val |
| YOLO opening recall | ≥ 0.75 | yolo val |
| depth 可穿越性判断准确率 | ≥ 90% | 离线验证（Step C） |
| depth 距离估计误差 | ≤ 50cm | 离线验证 |
| VLM 入口确认准确率 | ≥ 80% | 离线验证（Step D） |
| VLM 调用率 | ≤ 15% | 在线统计 |
| 融合决策准确率 | ≥ 85% | 对比人工标注 |
| 误进入率 | ≤ 5% | 在线测试 |
| 绕飞一圈找到入口成功率 | ≥ 90% | 在线测试（有入口房屋） |
| 单步延迟（无 VLM） | ≤ 15ms | 在线测试 |
| 单步延迟（有 VLM） | ≤ 600ms | 在线测试 |

---

### 交付物

- [ ] **Step A**: 手动截图工具 `simple_frame_capture.py`（可选辅助）
- [ ] **Step A**: 采集图片 ~500 张（纯 RGB，手动飞行截图）
- [ ] **Step B**: YOLO 标注数据（4 类：door_open/door_closed/window/opening）
- [ ] **Step B**: YOLO 模型权重 `entry_detector_v1/best.pt`
- [ ] **Step C**: depth 分析模块 `entry_depth_analyzer.py`
- [ ] **Step C**: depth 在线验证记录 + 调参报告
- [ ] **Step D**: VLM 部署脚本 + Prompt 配置
- [ ] **Step D**: VLM 入口确认模块 `vlm_entry_confirmer.py`（含回答解析器）
- [ ] **Step D**: 自适应 VLM 调用模块 `adaptive_vlm_caller.py`
- [ ] **Step E**: Rule-based 融合决策模块 `entry_fusion_decision.py`
- [ ] **Step E**: MLP 融合模块 `mlp_fusion.py` + 训练脚本
- [ ] **Step E**: Attention 融合模块 `attention_fusion.py`（论文对比用）
- [ ] **Step F**: 绕飞策略模块 `house_circling_explorer.py`
- [ ] **Step F**: 入口进入控制模块 `entry_procedure.py`
- [ ] **Step G**: 在线验证脚本 `online_pipeline_validator.py`
- [ ] **Step G**: 在线验证报告 + 检测效果可视化（含三模态融合示例）

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
| house_1→house_2 轨迹 | 10 次 |
| house_2→house_3 轨迹 | 10 次 |
| house_1→house_3 轨迹 | 10 次 |
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
| Phase 2 多模态入口探索 | — | `simple_frame_capture.py`, `entry_depth_analyzer.py`, `vlm_entry_confirmer.py`, `adaptive_vlm_caller.py`, `entry_fusion_decision.py`, `mlp_fusion.py`, `attention_fusion.py`, `house_circling_explorer.py`, `entry_procedure.py`, `phase2_entry_detector.yaml` |
| Phase 3 跨房屋导航 | — | `rule_based_house_navigator.py` |
| Phase 4 室内搜索 | `reflex_policy_model.py`, `train_reflex_policy.py`, `reflex_dataset_builder.py` | 扩展 `FEATURE_NAMES` |
| Phase 5 蒸馏 | `train_reflex_policy.py`（扩展 MLP 架构） | `build_distillation_dataset.py` |
| Phase 6 迭代 | `online_reflex_eval.py` | `iteration_manager.py` |
