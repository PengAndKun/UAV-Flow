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
Phase 1  房屋确认（原地扫描 + YOLO building 检测 + 坐标匹配 + VLM 描述）
  │
  ▼
Phase 2  入口探索（飞近目标房屋 + 门/窗检测 + depth 判断可进入性）
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
Phase 1 (房屋确认) ──► Phase 2 (入口探索) ──┐
                                              │
Phase 3 (跨房屋导航) ──────────────────────┤──► Phase 5 (统一蒸馏)
                                              │
Phase 4 (室内搜索) ───────────────────────┘
```

Phase 1 → Phase 2 是串行的（先确认房屋，才能去找入口）。Phase 3、Phase 4 与 Phase 2 并行推进。Phase 5 依赖前四个阶段的数据。

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

## Phase 1：房屋发现、确认与地图标记

### 目标

**Phase 1 要完成两件事：**
1. **发现房屋**——找到场景中有哪些房屋，确定它们在地图（UE4 世界坐标系）上的大致位置
2. **确认并标记**——为每栋房屋绑定视觉特征（参考帧 + VLM 描述），在地图上标记

不检测门、不检测窗、不判断能否进入。Phase 1 结束时系统知道的是：
- 场景中有 N 栋房屋，每栋在地图上的大致坐标范围
- `house_1` 是一栋"白色两层殖民风格房屋"
- `house_2` 是一栋"红砖一层平房"
- 每栋房屋在 RGB 画面中长什么样（参考帧 + VLM 描述）

入口探索（门/窗检测 + depth 判断）是 Phase 2 的事情。

### 核心问题：houses_config.json 从哪来？

之前的方案假设地图上已经有了每栋房屋的 bbox 坐标。但实际情况是：

```
你已有的：
  ✓ UAV 的精确位置和朝向（UE4 的 /state 接口，每步更新）
  ✓ 一张俯视图地图（对齐了 UE4 坐标轴）
  ✓ 已采集的 RGB 图片（各角度拍到了房屋）

你还没有的：
  ✗ 每栋房屋在 UE4 坐标系中的 bbox（center, x_min/max, y_min/max）
  ✗ 房屋的数量、编号

所以第一步是：建立 houses_config.json —— 确定场景中有几栋房屋，每栋在哪
```

### 整体流程

```
Phase 1 分四步：

Step 0. 房屋发现与坐标建立（House Discovery）
  确定场景中有几栋房屋 + 每栋的大致世界坐标
  → 生成 houses_config.json（房屋 bbox 列表）
  方式 A：手动在 UE4 编辑器中读取坐标（最快，10 分钟）
  方式 B：UAV 扫描 + depth 测距自动估算（自动化）
  方式 C：在俯视图上鼠标点击标记（可视化）

Step 1. YOLO 训练 + 原地扫描（Building Detector + Scan）
  标注 building → 训练 YOLO
  → UAV 原地旋转 360° → YOLO 检测 building
  → 方向 + 坐标匹配 → 确定每个 building 对应 houses_config 中哪栋
  → VLM 生成外观描述 → 绑定到地图条目

Step 2. 运行时定位（Runtime Localization）
  正式搜索时，每步用坐标判断"我在哪栋房屋附近"
  + YOLO building 检测做交叉验证

Step 3. 搜索状态管理（Search Status Tracking）
  搜索完一栋 → 标记 explored → 永不重复进入
  → 自动选下一栋 unexplored 房屋
```

---

### Step 0：房屋发现与坐标建立——生成 houses_config.json

#### 0.1 为什么需要这一步

Phase 1 后续的方向匹配算法需要知道每栋房屋在 UE4 世界坐标系中的位置（bbox）。但一开始 `houses_config.json` 是空的——系统不知道场景中有几栋房屋，更不知道它们的坐标。

**Step 0 就是建立这份地图数据。**

#### 0.2 UAV 自身的位置和朝向

这个不需要额外操作——UE4/AirSim 的 `/state` 接口直接返回：

```json
{
  "pose": {
    "x": 2359.9,     ← UE4 世界坐标 X（单位 cm）
    "y": 85.3,        ← UE4 世界坐标 Y（单位 cm）
    "z": 225.0,       ← UE4 世界坐标 Z（高度，单位 cm）
    "yaw": -1.7       ← 偏航角（度）
  }
}
```

UAV 的位置和朝向是**精确已知的**，每步都能获取。

#### 0.3 三种方式获取房屋坐标

##### 方式 A：手动在 UE4 编辑器中读取（推荐先用这个）

**最简单、最快、最精确。**

操作步骤：
1. 打开 UE4 编辑器
2. 在场景中找到每栋房屋的 Actor
3. 读取其 Transform 中的 Location (X, Y)
4. 估算房屋的宽度和长度（或看 Actor 的 BoundingBox）
5. 手动填入 `houses_config.json`

```
在 UE4 编辑器中：
  选中房屋 Actor → Details 面板 → Transform → Location
  house_1: X=2400, Y=100  （读到的值）
  house_2: X=1200, Y=500
  house_3: X=800,  Y=-300

  每栋房屋大约 600cm × 600cm → bbox 向四周扩展 300cm
```

手动填写：

```json
{
  "houses": [
    {
      "house_id": "house_1",
      "center": [2400.0, 100.0],
      "bbox": {
        "x_min": 2100.0, "x_max": 2700.0,
        "y_min": -200.0, "y_max": 400.0
      }
    }
  ]
}
```

**耗时：约 10 分钟。** 第一版直接用这个即可。

##### 方式 B：UAV 扫描 + depth 测距自动估算

如果不想手动读坐标，可以让 UAV 自动发现房屋位置：

```
原理：
  UAV 原地旋转一圈 → YOLO 检测到 building
  → building 在画面中的位置 → 方位角
  → depth 帧中 building 区域的深度 → 距离
  → 方位角 + 距离 + UAV 位置 → 房屋的世界坐标

公式：
  house_x = uav_x + distance × cos(bearing_rad)
  house_y = uav_y + distance × sin(bearing_rad)
```

```python
import numpy as np
import math

def estimate_house_position(uav_pose, bearing_deg, depth_frame, building_bbox):
    """
    根据方位角和 depth 估算房屋的世界坐标。

    Args:
        uav_pose: {"x", "y", "z", "yaw"}
        bearing_deg: building 的绝对方位角
        depth_frame: depth 帧 (numpy array, cm)
        building_bbox: [x1, y1, x2, y2] 像素坐标
    """
    # 取 building bbox 区域的 depth 中位值作为距离估算
    x1, y1, x2, y2 = [int(v) for v in building_bbox]
    building_depth_region = depth_frame[y1:y2, x1:x2]

    # 去掉极端值（天空=很大值，太近=很小值）
    valid_depths = building_depth_region[
        (building_depth_region > 100) & (building_depth_region < 10000)
    ]

    if len(valid_depths) == 0:
        return None

    estimated_distance_cm = np.median(valid_depths)

    # 极坐标 → 世界坐标
    bearing_rad = math.radians(bearing_deg)
    house_x = uav_pose["x"] + estimated_distance_cm * math.cos(bearing_rad)
    house_y = uav_pose["y"] + estimated_distance_cm * math.sin(bearing_rad)

    return {
        "estimated_center": [round(house_x, 1), round(house_y, 1)],
        "estimated_distance_cm": round(estimated_distance_cm, 1),
        "bearing_deg": round(bearing_deg, 1)
    }
```

自动发现流程：

```
UAV 在初始位置原地旋转 360°（12步 × 30°）
  │
  每步：
  ├── 拍 RGB + depth
  ├── YOLO 检测 building
  ├── 对每个 building：
  │     ├── 画面位置 → 方位角
  │     ├── depth 区域 → 距离
  │     └── 方位角 + 距离 → 世界坐标估算
  │
  ▼
汇总所有检测：
  ├── 同一栋房屋在多帧中被检测到（方位角相近）
  ├── 聚类 → 去重 → 每簇 = 一栋房屋
  ├── 取每簇的平均坐标作为 center
  ├── center ± 300cm 作为 bbox
  │
  ▼
生成 houses_config.json
  ├── house_1: center=[2400, 100],  bbox=...
  ├── house_2: center=[1200, 500],  bbox=...
  └── house_3: center=[800, -300],  bbox=...
```

聚类逻辑：

```python
def cluster_building_detections(all_detections, merge_radius_cm=500):
    """
    将多帧中的 building 检测聚类成独立房屋。

    如果两次检测的估算世界坐标距离 < merge_radius → 同一栋房屋。
    """
    clusters = []  # 每个 cluster = 一栋房屋的所有观测

    for det in all_detections:
        if det["estimated_center"] is None:
            continue

        merged = False
        for cluster in clusters:
            # 与该 cluster 的平均位置比较
            avg_x = np.mean([d["estimated_center"][0] for d in cluster])
            avg_y = np.mean([d["estimated_center"][1] for d in cluster])
            dx = det["estimated_center"][0] - avg_x
            dy = det["estimated_center"][1] - avg_y
            dist = math.sqrt(dx**2 + dy**2)

            if dist < merge_radius_cm:
                cluster.append(det)
                merged = True
                break

        if not merged:
            clusters.append([det])

    # 每个 cluster → 一栋房屋
    houses = []
    for i, cluster in enumerate(clusters):
        avg_x = round(np.mean([d["estimated_center"][0] for d in cluster]), 1)
        avg_y = round(np.mean([d["estimated_center"][1] for d in cluster]), 1)

        houses.append({
            "house_id": f"house_{i+1}",
            "center": [avg_x, avg_y],
            "bbox": {
                "x_min": avg_x - 300,
                "x_max": avg_x + 300,
                "y_min": avg_y - 300,
                "y_max": avg_y + 300
            },
            "observation_count": len(cluster),
            "avg_confidence": round(
                np.mean([d["confidence"] for d in cluster]), 2
            )
        })

    return houses
```

##### 方式 C：在俯视图上鼠标点击标记

如果有俯视图地图（截图或渲染），可以做一个简单的标记工具：

```python
import cv2

def mark_houses_on_map(map_image_path, map_origin, map_scale):
    """
    在俯视图上鼠标点击标记房屋位置。

    map_origin: 图片左上角对应的 UE4 坐标 [x, y]
    map_scale:  每像素对应的 cm 数
    """
    img = cv2.imread(map_image_path)
    houses = []
    click_count = [0]

    def on_click(event, px, py, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # 像素 → UE4 坐标
            world_x = map_origin[0] + px * map_scale
            world_y = map_origin[1] + py * map_scale
            click_count[0] += 1

            house = {
                "house_id": f"house_{click_count[0]}",
                "center": [round(world_x, 1), round(world_y, 1)],
                "bbox": {
                    "x_min": round(world_x - 300, 1),
                    "x_max": round(world_x + 300, 1),
                    "y_min": round(world_y - 300, 1),
                    "y_max": round(world_y + 300, 1)
                }
            }
            houses.append(house)

            # 画圆标记
            cv2.circle(img, (px, py), 10, (0, 0, 255), -1)
            cv2.putText(img, f"house_{click_count[0]}", (px+15, py),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow("Map", img)
            print(f"  Marked house_{click_count[0]} at ({world_x:.0f}, {world_y:.0f})")

    cv2.imshow("Map", img)
    cv2.setMouseCallback("Map", on_click)
    print("点击地图上的房屋位置，按 ESC 完成")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return houses
```

#### 0.4 推荐做法

```
第一次做（现在）：
  → 方式 A（手动读 UE4 坐标）
  → 10 分钟搞定，精度最高
  → 快速跑通后续全流程

后续自动化：
  → 方式 B（UAV 扫描 + depth 自动发现）
  → 论文中可以作为"自动环境建图"能力展示
  → 换新场景时不需要手动标注
```

#### 0.5 Step 0 的输出

不管用哪种方式，最终输出是一个 `houses_config.json`：

```json
{
  "map_info": {
    "name": "neighborhood_scene_v1",
    "coordinate_system": "unreal_engine_cm",
    "origin_note": "UE4 world origin, units in cm",
    "fov_deg": 90,
    "discovery_method": "manual_ue4_editor"
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
      "recon_status": "pending",
      "recon_data": null,
      "search_status": "pending"
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
      "recon_status": "pending",
      "recon_data": null,
      "search_status": "pending"
    }
  ],

  "search_altitude_cm": 270.0,
  "global_step_budget": 2000
}
```

> **关键字段说明**：
> - `center`：房屋中心的 UE4 世界坐标 [x, y]
> - `bbox`：中心向四周扩展约 300cm 的矩形范围（精度不需要很高，±100cm 都可以）
> - `floor_z_range`：室内地面的 z 高度范围，用于判断 UAV 是否在室内
> - `recon_status`: "pending" 表示还没被 Phase 1 扫描确认
> - `discovery_method`：记录坐标来源，方便后续复查

#### 0.6 交付物

- [ ] `houses_config.json`（包含所有房屋的 bbox 坐标）
- [ ] 如用方式 B：房屋自动发现脚本 `discover_houses.py`
- [ ] 如用方式 C：俯视图标记工具 `mark_houses_on_map.py`

---

### Step 1：YOLO 训练 + 原地扫描——确认每栋房屋的视觉特征

**前置条件**：Step 0 已完成，`houses_config.json` 中有了每栋房屋的大致坐标。

#### 1.1 这一步要解决什么

Step 0 给了每栋房屋的坐标，但系统还不知道：
- 每个坐标框里的房屋在 RGB 画面中长什么样
- 从当前位置看过去，YOLO 检测到的 building 对应的是哪个 house_id
- 每栋房屋的视觉特征描述（颜色、风格、层数等）

**Step 1 用 YOLO + 方向匹配解决**：检测画面中的 building → 利用 Step 0 的坐标做方向匹配 → 绑定视觉特征。

#### 1.2 这一步不做什么

| 不做 | 原因 |
|------|------|
| ~~检测门是开是关~~ | 那是 Phase 2（入口探索）的任务 |
| ~~检测窗户~~ | Phase 1 只关心"这是不是一栋建筑物" |
| ~~判断能否进入~~ | 需要飞近 + depth，Phase 2 做 |
| ~~爬升到高空~~ | 有坐标就够了，不需要俯瞰 |
| ~~飞到房屋附近~~ | 原地旋转就能覆盖所有方向 |
| ~~使用深度图~~ | Phase 1 纯 RGB，不需要 depth |

---

#### 1.3 YOLO building 检测模型——从标注到训练完整流程

Phase 1 的 YOLO 模型只检测 **一个类别**：`building`。

##### 1.3.1 为什么只要 1 个类别

| 设计选择 | 原因 |
|---------|------|
| 只用 `building`，不区分 `house_1/2/3` | 换场景不用重训，一帧多房屋自然支持，房屋数量变化不影响 |
| 不检测 `door_open/closed/window` | Phase 1 只确认"那个方向有没有建筑物"，门窗是 Phase 2 的事 |
| YOLO 只回答"有/没有建筑物" | "是哪栋房屋"由坐标方向匹配回答 |

##### 1.3.2 标注工具选择

| 工具 | 说明 | 推荐度 |
|------|------|--------|
| **Roboflow** | 在线免费版，上传图片 → 鼠标拖框 → 导出 YOLO 格式，最快 | ⭐⭐⭐ |
| **Label Studio** | 开源本地部署，功能全面 | ⭐⭐ |
| **CVAT** | 开源，专业标注工具 | ⭐⭐ |
| **labelImg** | 极简轻量，纯本地桌面应用 | ⭐ |

##### 1.3.3 标注方法（手动拖框）

**你只需要做一件事：对画面中的每栋建筑物画一个 bbox，类别选 `building`。**

```
标注示例：

情况 A：一帧中只看到一栋房屋
┌──────────────────────────┐
│                          │
│    ┌──────────────┐      │
│    │  building    │      │
│    │              │      │
│    │   ┌──┐       │      │
│    │   │  │       │      │
│    │   └──┘       │      │
│    └──────────────┘      │
│         草地              │
└──────────────────────────┘
→ 标 1 个框：building

情况 B：一帧中同时看到两栋房屋
┌──────────────────────────┐
│                          │
│  ┌─────────┐ ┌────────┐ │
│  │building │ │building│ │
│  │  (近)   │ │ (远)   │ │
│  │         │ │        │ │
│  └─────────┘ └────────┘ │
│         草地              │
└──────────────────────────┘
→ 标 2 个框：都标为 building

情况 C：房屋被部分遮挡（树木等）
→ 仍然标：框住可见部分

情况 D：只有天空/地面/树，没有房屋
→ 不标任何框（作为负样本放进数据集）
```

**标注原则**：
- `building` 类别不区分具体是哪栋房屋——所有房屋都标为 `building`
- bbox 框住建筑物整体轮廓（包括屋顶到地基，门窗都在框内但不需要单独标）
- 一张图上可以有多个 `building` 框
- 被遮挡的房屋也标（框住可见部分）
- 纯天空/地面的图片不标框，直接放入数据集作为负样本

##### 1.3.4 标注数据量

| 内容 | 数量 | 说明 |
|------|------|------|
| 每栋房屋的 360° 旋转拍摄 | 每栋 12 张 | UAV 原地转一圈 |
| 3 栋房屋 | 36 张 | 基础数据 |
| 不同距离/位置补拍 | 每栋 10-15 张 | 远景/中景/侧面 |
| 一帧多房屋的情况 | ~10 张 | 某些角度同时看到 2-3 栋 |
| 负样本（无房屋） | ~10 张 | 纯天空/地面/树木 |
| **总计** | **~90-100 张起步** | 仿真环境外观固定，不需太多 |

> 仿真环境中房屋纹理固定，90 张足够第一版。如果 mAP 不够再补采。后续可混入公开建筑物检测数据集（xView、DOTA、Open Images 的 building 子集）增强泛化。

##### 1.3.5 数据集目录结构

标注完成后导出为 **YOLO 格式**：

```
building_detection_data/
├── images/
│   ├── train/                    ← 80% 的图片（~72 张）
│   │   ├── scan_h1_00.jpg
│   │   ├── scan_h1_01.jpg
│   │   ├── scan_h2_00.jpg
│   │   └── ...
│   └── val/                      ← 20% 的图片（~18 张）
│       ├── scan_h1_03.jpg
│       └── ...
├── labels/
│   ├── train/                    ← 对应标注文件
│   │   ├── scan_h1_00.txt
│   │   ├── scan_h1_01.txt
│   │   └── ...
│   └── val/
│       ├── scan_h1_03.txt
│       └── ...
└── phase1_building_detector.yaml
```

每个 `.txt` 标注文件内容（YOLO 格式，Roboflow 会自动生成）：

```
# class_id  center_x  center_y  width  height  （相对图像尺寸的 0-1 归一化值）
0 0.45 0.40 0.50 0.60
0 0.82 0.35 0.25 0.45
```

> 如果只有一栋房屋就一行，两栋就两行。负样本图片对应的 `.txt` 文件为空。

##### 1.3.6 数据集配置文件

```yaml
# phase1_building_detector.yaml
path: ./building_detection_data
train: images/train
val: images/val

names:
  0: building
```

**就 1 个类别，极简。**

##### 1.3.7 训练命令

```bash
# 安装 ultralytics
pip install ultralytics

# 训练
yolo train \
  model=yolov8n.pt \
  data=phase1_building_detector.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  project=runs/building_detector \
  name=v1
```

训练完成后权重保存在：`runs/building_detector/v1/weights/best.pt`

训练时间预估：
- GPU（GTX 1080 / RTX 3060）：约 15-30 分钟
- CPU：约 1-2 小时（不推荐）

##### 1.3.8 验证训练效果

```bash
# 在验证集上评估
yolo val \
  model=runs/building_detector/v1/weights/best.pt \
  data=phase1_building_detector.yaml

# 对新图片跑预测看效果
yolo predict \
  model=runs/building_detector/v1/weights/best.pt \
  source=test_image.jpg \
  save=True
```

验证标准：

| 指标 | 要求 | 不达标怎么办 |
|------|------|-------------|
| building mAP@0.5 | ≥ 0.85 | 补采数据或增加 epochs |
| building recall | ≥ 0.90（不能漏检） | 补采远距离/遮挡场景数据 |
| 一帧多房屋全检出 | ≥ 85% | 补采多房屋同框的图片 |
| 推理速度（640px） | ≤ 8ms / 帧 | 已用 YOLOv8n，通常满足 |

##### 1.3.9 完整操作步骤总结

```
Step A → 采集图片（你已完成 ✓）
          UAV 在仿真中原地旋转 + 不同位置拍照

Step B → 标注（约 30-40 分钟）
          打开 Roboflow → 上传图片
          → 每张图中的建筑物画 bbox → 类别选 "building"
          → 一张图有多栋就画多个框
          → 标完后导出 YOLO 格式

Step C → 组织数据（约 10 分钟）
          按 8:2 分 train/val
          → 创建 phase1_building_detector.yaml

Step D → 训练（约 15-30 分钟）
          运行 yolo train 命令

Step E → 验证（约 10 分钟）
          运行 yolo val 确认 mAP ≥ 0.85
          → 对几张新图 yolo predict 看效果
          → 不达标则补采数据重新训练

总耗时：约 1-1.5 小时
```

---

#### 1.4 地图与检测对齐——核心算法

YOLO 训练好后，下一步是把检测到的 building 和地图上的房屋框对齐。

##### 1.4.1 对齐的原理

```
已知信息：
  ① UAV 当前位姿 (x, y, z, yaw) ← 从 /state 获取
  ② 地图上每栋房屋的 bbox 坐标  ← houses_config.json
  ③ 相机水平视场角 fov           ← 固定值（如 90°）
  ④ YOLO 检测到的 building bbox  ← 训练好的模型输出

对齐过程：
  building 在画面中的水平位置 (像素)
    → 转换为相对画面中心的偏移比例
    → 转换为角度偏移
    → 加上 UAV 当前 yaw
    → 得到该 building 的绝对方位角
    → 从 UAV 位置沿该方位角发射线
    → 射线穿过地图上哪个 house bbox
    → 该 building = 那栋 house

一句话：画面中 building 的位置 → 方向 → 地图上的房屋
```

##### 1.4.2 图解对齐过程

```
         地图俯视（UE4 坐标系）

         N (y+)
         │
    ┌────┤────┐
    │ house_2 │
    └────┬────┘
         │
         │          ┌──────────┐
    ─────┼──────────│ house_1  │──── E (x+)
         │          └──────────┘
         │
    ┌────┤────┐
    │ house_3 │
    └────┬────┘
         │
    UAV ◈ (朝向 yaw = 0°，即朝东)
    位置 (500, 0)

UAV 拍到的 RGB 画面（640px 宽）：
┌──────────────────────────────────────────┐
│                                          │
│      ┌───────┐              ┌────┐       │
│      │bld #1 │              │bld │       │
│      │       │              │ #2 │       │
│      │       │              │    │       │
│      └───────┘              └────┘       │
│  0px    200px                 500px  640px│
└──────────────────────────────────────────┘

对齐计算：

bld #1 的 bbox 中心 = 200px
  偏移比例 = (200 - 320) / 320 = -0.375（偏左）
  角度偏移 = -0.375 × 45° = -16.9°（FOV=90°的一半=45°）
  绝对方位角 = 0° + (-16.9°) = -16.9°
  从 UAV 发射线 → 穿过 house_1 的 bbox
  → bld #1 = house_1 ✓

bld #2 的 bbox 中心 = 500px
  偏移比例 = (500 - 320) / 320 = +0.5625（偏右）
  角度偏移 = +0.5625 × 45° = +25.3°
  绝对方位角 = 0° + 25.3° = 25.3°
  从 UAV 发射线 → 穿过 house_3 的 bbox
  → bld #2 = house_3 ✓
```

##### 1.4.3 射线-bbox 相交算法

```python
import math

def get_expected_house(uav_x, uav_y, yaw_deg, houses_config):
    """
    根据 UAV 位置和绝对方位角，找该方向上最近的房屋。

    从 UAV 位置沿 yaw 方向发射射线，
    检查射线穿过哪个房屋的 bbox。
    """
    yaw_rad = math.radians(yaw_deg)
    dx = math.cos(yaw_rad)
    dy = math.sin(yaw_rad)

    best_house = None
    best_dist = float('inf')

    for house in houses_config["houses"]:
        bbox = house["bbox"]
        cx = (bbox["x_min"] + bbox["x_max"]) / 2
        cy = (bbox["y_min"] + bbox["y_max"]) / 2

        to_house_x = cx - uav_x
        to_house_y = cy - uav_y

        # 投影到视线方向（正数=在前方）
        proj = to_house_x * dx + to_house_y * dy
        if proj <= 0:
            continue  # 在 UAV 背后，跳过

        # 横向偏移（射线到房屋中心的垂直距离）
        perp = abs(to_house_x * (-dy) + to_house_y * dx)
        dist = math.sqrt(to_house_x**2 + to_house_y**2)

        # 房屋半宽 + 容差
        house_half_width = max(
            bbox["x_max"] - bbox["x_min"],
            bbox["y_max"] - bbox["y_min"]
        ) / 2

        if perp < house_half_width + 200 and dist < best_dist:
            best_dist = dist
            best_house = house["house_id"]

    return best_house, best_dist
```

##### 1.4.4 一帧多 building 匹配

画面中可能同时看到 2-3 栋房屋，每栋在画面不同水平位置，分别匹配：

```python
def match_detections_to_houses(detections, uav_pose, houses_config, frame_width, fov_deg=90):
    """
    将一帧中的所有 building 检测框匹配到地图房屋。

    流程：
    1. 取每个 building bbox 的水平中心
    2. 计算该 building 相对于画面中心的角度偏移
    3. 加上 UAV yaw → 该 building 的绝对方位角
    4. 射线匹配 → 对应哪栋 house
    """
    results = []

    for det in detections:
        if det["class"] != "building":
            continue

        # bbox 水平中心 → 角度偏移
        bbox_cx = (det["bbox"][0] + det["bbox"][2]) / 2
        offset_ratio = (bbox_cx - frame_width / 2) / (frame_width / 2)
        yaw_offset = offset_ratio * (fov_deg / 2)

        # 绝对方位角
        abs_yaw = uav_pose["yaw"] + yaw_offset

        # 射线匹配
        house_id, dist = get_expected_house(
            uav_pose["x"], uav_pose["y"], abs_yaw, houses_config
        )

        results.append({
            "detection": det,
            "matched_house_id": house_id,
            "bearing_deg": abs_yaw,
            "estimated_distance_cm": dist,
            "confidence": det["confidence"]
        })

    return results
```

##### 1.4.5 对齐失败的处理

| 情况 | 原因 | 处理 |
|------|------|------|
| YOLO 检测到 building 但射线没命中任何 bbox | 地图上没有标注该房屋，或 bbox 范围太小 | 记录为 `unmatched_building`，扩大 bbox 容差 |
| 地图上有 bbox 但 YOLO 没检测到 | 房屋被遮挡/距离太远/检测漏了 | 标记为 `pending`，后续飞近补扫 |
| 两个 building 匹配到同一个 house_id | 实际只有一栋房屋但 YOLO 检测出两个框 | 取置信度更高的那个，另一个可能是误检 |
| shooting 射线在两个 bbox 交界处 | UAV 处于两栋房屋的方位角边界 | 取距离更近的那栋 |

---

#### 1.5 完整扫描流程

```
任务开始（UAV 在任意初始位置）
  │
  ▼
读取 UAV 当前 pose (x, y, z, yaw_0)
读取地图 houses_config.json
加载训练好的 YOLO building 检测模型
  │
  ▼
原地旋转一圈（12步 × 30°= 360°），每步：
  │
  │  ┌─────────────────────────────────────────────┐
  │  │ 当前朝向 yaw_i = yaw_0 + i × 30°           │
  │  │                                              │
  │  │ ① 拍 RGB → recon/frame_{i:02d}.jpg          │
  │  │    （不拍 depth——Phase 1 不需要深度）         │
  │  │                                              │
  │  │ ② YOLO 检测 → building 框 ×N                │
  │  │    只检测 building，不检测门窗                │
  │  │                                              │
  │  │ ③ 对每个 building 框做方向-地图对齐：        │
  │  │    bbox 水平中心 → 角度偏移                  │
  │  │    + yaw_i → 绝对方位角                      │
  │  │    → 射线与地图 bbox 相交                    │
  │  │    → matched_house_id                        │
  │  │                                              │
  │  │ ④ 保存该帧记录                               │
  │  └─────────────────────────────────────────────┘
  │
  ▼
旋转完成，汇总结果：
  │
  ├── 对每栋房屋，选 YOLO 置信度最高的帧
  │   → 该帧作为参考帧（best_reference_frame）
  │   → VLM 生成外观描述
  │
  ├── 标记 recon_status = "confirmed"
  │   search_status = "unexplored"
  │
  └── 未被检测到的房屋：
      → recon_status 保持 "pending"
      → 后续飞近时自动补扫
  │
  ▼
在俯视图地图上标记：
  房屋框 + 确认状态 + VLM 描述 + UAV 位置
  │
  ▼
Phase 1 完成 → 系统知道每个方向上是哪栋房屋
               → 每栋房屋有了参考图和文字描述
               → 地图上所有 confirmed 房屋标记为 unexplored
```

#### 1.6 侦察记录数据结构

##### 每帧记录

Phase 1 的每帧记录很简洁——只有 building 信息，没有门窗：

```json
{
  "recon_frame_id": 5,
  "yaw_deg": 150.0,
  "rgb_path": "recon/frame_05.jpg",

  "building_matches": [
    {
      "matched_house_id": "house_2",
      "bearing_deg": 142.3,
      "estimated_distance_cm": 1820.0,
      "confidence": 0.87,
      "bbox_in_frame": [120, 80, 480, 400]
    },
    {
      "matched_house_id": "house_3",
      "bearing_deg": 168.5,
      "estimated_distance_cm": 2450.0,
      "confidence": 0.62,
      "bbox_in_frame": [450, 100, 620, 380]
    }
  ],

  "total_buildings_detected": 2
}
```

> 一帧中看到 2 栋房屋（house_2 和 house_3），每栋独立匹配到地图。

##### 扫描完成后的地图状态

```json
{
  "house_id": "house_1",
  "center": [2400.0, 100.0],
  "bbox": {
    "x_min": 2100.0, "x_max": 2700.0,
    "y_min": -200.0, "y_max": 400.0
  },

  "recon_status": "confirmed",
  "recon_data": {
    "confirmed_at": "2026-03-26T10:15:30",
    "scan_origin": {"x": 2000.0, "y": 300.0, "z": 270.0},

    "best_reference_frame": {
      "path": "recon/frame_01.jpg",
      "yaw_deg": 30.0,
      "confidence": 0.91,
      "distance_cm": 520.0
    },

    "vlm_description": "白色殖民风格两层房屋，有前廊和美国国旗",

    "observed_from_frames": [0, 1, 2],
    "total_observations": 3,
    "avg_confidence": 0.86
  },

  "search_status": "unexplored"
}
```

**注意**：`recon_data` 中没有 `door_candidates`——Phase 1 不检测门。门的信息会在 Phase 2 飞近目标房屋时才获取。

##### 未确认房屋

```json
{
  "house_id": "house_3",
  "recon_status": "pending",
  "recon_data": {
    "scan_attempted_at": "2026-03-26T10:15:30",
    "failure_reason": "not_detected_in_any_frame",
    "note": "距离太远或被遮挡，需飞近补扫"
  },
  "search_status": "pending"
}
```

**补扫机制**：后续飞行中进入 `pending` 房屋的 `approach_radius` 范围内时，自动对该方向拍 RGB + YOLO 检测 + 方向匹配。

---

#### 1.7 VLM 描述生成

对每栋 `confirmed` 房屋的最佳参考帧调用 VLM，生成外观描述：

```python
VLM_PROMPT = """Describe this building in one sentence, focusing on:
- Color and material (e.g., white painted wood, red brick)
- Architectural style (e.g., colonial, ranch, modern)
- Number of floors
- Porch, balcony, or other visible structures
- Any distinctive features (flags, decorations, plants)

Do NOT describe doors or windows in detail — just focus on the
overall appearance that makes this building recognizable.
"""
```

VLM 可选：
- **LLaVA-1.5-7B**（本地部署，RTX 3060 可跑）
- **GPT-4V / GPT-4o**（API 调用，质量更高）
- **Qwen-VL**（本地备选）

VLM 描述的用途：
1. **SSP 规划器上下文**：高层规划时知道每栋房屋的外观
2. **GCMA archive 初始化**：作为 semantic cell 初始描述
3. **人类可读报告**：俯视图地图上标注房屋名字旁的描述文字

---

#### 1.8 完整扫描脚本

```python
import requests
import time
import json
import math
from datetime import datetime

def run_phase1_scan(server_url, houses_config, yolo_model, vlm_model=None):
    """
    Phase 1 原地扫描：确定每个方向上是哪栋房屋。

    只检测 building，不检测门窗。
    不使用 depth，纯 RGB + 坐标。
    """
    # 1. 读取 UAV 位姿
    state = requests.get(f"{server_url}/state").json()
    uav_pose = state["pose"]
    frame_width = 640
    fov_deg = houses_config["map_info"].get("fov_deg", 90)

    all_frames = []
    print(f"[Phase 1] 开始原地扫描，UAV 位置: ({uav_pose['x']:.0f}, {uav_pose['y']:.0f})")

    # 2. 原地旋转 12 步
    for step in range(12):
        # 只拍 RGB（不需要 depth）
        rgb = requests.get(f"{server_url}/frame").content
        rgb_path = f"recon/frame_{step:02d}.jpg"
        save_frame(rgb, rgb_path)

        state = requests.get(f"{server_url}/state").json()
        current_yaw = state["pose"]["yaw"]

        # YOLO 只检测 building
        detections = yolo_model.predict(rgb)
        building_dets = [d for d in detections if d["class"] == "building"]

        # 方向-地图对齐
        building_matches = match_detections_to_houses(
            building_dets, state["pose"], houses_config, frame_width, fov_deg
        )

        frame_record = {
            "recon_frame_id": step,
            "yaw_deg": current_yaw,
            "rgb_path": rgb_path,
            "building_matches": building_matches,
            "total_buildings_detected": len(building_dets)
        }
        all_frames.append(frame_record)

        print(f"  Step {step:2d} | yaw={current_yaw:6.1f}° | "
              f"buildings={len(building_dets)} | "
              f"matched={[m['matched_house_id'] for m in building_matches]}")

        # 旋转 30°
        if step < 11:
            requests.post(f"{server_url}/move_relative",
                         json={"action": "yaw_right"})
            time.sleep(0.3)

    # 3. 汇总：为每栋房屋选最佳参考帧 + VLM 描述
    confirmed_count = 0
    for house in houses_config["houses"]:
        house_id = house["house_id"]

        observations = []
        for frame in all_frames:
            for bm in frame["building_matches"]:
                if bm["matched_house_id"] == house_id:
                    observations.append({
                        "frame_id": frame["recon_frame_id"],
                        "confidence": bm["confidence"],
                        "rgb_path": frame["rgb_path"],
                        "yaw_deg": frame["yaw_deg"],
                        "distance_cm": bm["estimated_distance_cm"]
                    })

        if observations:
            best_obs = max(observations, key=lambda o: o["confidence"])

            vlm_desc = ""
            if vlm_model:
                vlm_desc = vlm_model.describe(best_obs["rgb_path"], VLM_PROMPT)

            house["recon_status"] = "confirmed"
            house["recon_data"] = {
                "confirmed_at": datetime.now().isoformat(),
                "scan_origin": {
                    "x": uav_pose["x"],
                    "y": uav_pose["y"],
                    "z": uav_pose["z"]
                },
                "best_reference_frame": best_obs,
                "vlm_description": vlm_desc,
                "observed_from_frames": [o["frame_id"] for o in observations],
                "total_observations": len(observations),
                "avg_confidence": sum(o["confidence"] for o in observations)
                                  / len(observations)
            }
            house["search_status"] = "unexplored"
            confirmed_count += 1
            print(f"  ✓ {house_id} confirmed | "
                  f"best_conf={best_obs['confidence']:.2f} | "
                  f"seen_in={len(observations)} frames")
        else:
            house["recon_status"] = "pending"
            house["recon_data"] = {
                "scan_attempted_at": datetime.now().isoformat(),
                "failure_reason": "not_detected_in_any_frame"
            }
            print(f"  ✗ {house_id} NOT detected — marked as pending")

    print(f"\n[Phase 1] 扫描完成: {confirmed_count}/{len(houses_config['houses'])} 房屋已确认")

    # 4. 保存更新后的配置
    save_json(houses_config, "houses_config_after_recon.json")
    save_json(all_frames, "recon_frames_log.json")

    return houses_config
```

---

### Step 2：运行时定位——每步知道"我在哪栋房屋附近"

Phase 1 扫描完成后，进入正式搜索阶段。每一步需要回答"我在哪栋房屋附近"。

#### 2.1 两层定位机制

```
每步判断"我在哪栋房屋"：

第1层 — 坐标判断（≈ 0ms，确定性，最可靠）
  ├─ pose (x,y) 落在哪个 house bbox 内 → current_house_id
  ├─ pose (x,y) 不在任何 bbox 内 → "outdoor"
  ├─ pose.z 是否在 floor_z_range 内 → is_inside
  └─ 与 target_house 中心的距离和方位角 → 导航用

第2层 — YOLO building 检测 + 方向匹配（≤ 8ms，可选交叉验证）
  ├─ 当前帧是否看到 building → building_detected
  ├─ 方向匹配到哪个 house_id → visual_house_id
  └─ 与坐标判断对比 → 一致则高置信度，不一致则以坐标为准
```

**注意**：运行时定位只判断"在哪栋房屋附近"，不判断门窗。门窗检测是 Phase 2 在 `house_circling` 阶段才开始做的。

#### 2.2 坐标定位函数

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

#### 2.3 地图配置文件

`houses_config.json` 由 Step 0 生成（参见 Step 0.5 节）。运行时定位使用其中的 `bbox` 字段判断 UAV 在哪栋房屋范围内。

关键字段回顾：

```json
{
  "houses": [
    {
      "house_id": "house_1",
      "center": [2400.0, 100.0],
      "bbox": { "x_min": 2100.0, "x_max": 2700.0, "y_min": -200.0, "y_max": 400.0 },
      "approach_radius_cm": 800.0,
      "floor_z_range": [200.0, 350.0],
      "recon_status": "confirmed",
      "search_status": "unexplored"
    }
  ],

  "search_altitude_cm": 270.0,
  "global_step_budget": 2000
}
```

#### 2.4 "在错误房屋"的处理

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
坐标定位特征（6维，确定性）：
  distance_to_target_norm        ← 到目标中心归一化距离（/2000cm）
  bearing_error_norm             ← 方位角误差归一化（/180°）
  is_at_target_house             ← 在目标房屋范围内（0/1）
  is_at_wrong_house              ← 在非目标房屋范围内（0/1）
  is_inside                      ← 在室内（0/1）
  target_approach_progress       ← 接近进度（0-1）

YOLO 视觉特征（3维，概率性）：
  yolo_building_detected         ← 检测到 building（0/1）
  yolo_building_confidence       ← building 最高置信度（0-1）
  yolo_building_count            ← building 数量（归一化 /5）

方向匹配特征（2维，确定性）：
  direction_match_confirmed      ← 方向匹配确认看到目标房屋（0/1）
  nearest_building_bearing_error ← 最近 building 与目标方位角误差（归一化 /180°）
```

总计 **11 维**（6 确定性 + 3 概率性 + 2 方向匹配）。

> Phase 1 不输出门相关特征。门特征（`door_detected`, `door_confidence` 等）由 Phase 2 提供。

---

### 验证流程

#### YOLO 模型验证

| 验证项 | 通过标准 | 不达标处理 |
|--------|---------|-----------|
| building mAP@0.5 | ≥ 0.85 | 补采数据或增加 epochs |
| building recall | ≥ 0.90 | 补采远距离/遮挡场景 |
| 一帧多房屋全检出 | ≥ 85% | 补采多房屋同框图片 |
| 推理速度 | ≤ 8ms | 已用 YOLOv8n，通常满足 |

#### 扫描对齐验证

| 验证项 | 通过标准 |
|--------|---------|
| 所有房屋被确认 | 100%（如有 pending 需说明原因） |
| 方向匹配准确率 | ≥ 90% building 正确匹配到 house_id |
| 一帧多房屋匹配 | ≥ 85% 全部正确 |
| VLM 描述质量 | 包含颜色+风格+层数 |
| 补扫触发 | pending 房屋飞近后自动变 confirmed |

#### 运行时定位验证

| 验证项 | 通过标准 |
|--------|---------|
| 坐标房屋归属准确率 | 100% |
| 方向匹配交叉验证一致率 | ≥ 85% |
| 错误房屋告警准确率 | 100% |
| 室内/室外切换准确率 | ≥ 98% |

#### 搜索状态验证

| 验证项 | 通过标准 |
|--------|---------|
| explored 后不重复进入 | 0 次 |
| 多房屋自动切换 | 所有房屋被访问 |
| person_found 准确率 | ≥ 95% |
| 状态流转正确性 | 符合定义规则 |

---

### 交付物

- [ ] 地图配置文件 `houses_config.json`（bbox + 高度范围 + 初始状态）
- [ ] YOLO building 检测标注数据（90-100 张，1 类：building）
- [ ] YOLO building 检测模型权重 `building_detector_v1/best.pt`
- [ ] 数据集配置 `phase1_building_detector.yaml`
- [ ] 方向-房屋匹配模块 `direction_house_matcher.py`
- [ ] 原地扫描脚本 `phase1_scan.py`
- [ ] 坐标定位模块 `house_locator.py`
- [ ] 搜索状态管理模块 `search_status_manager.py`
- [ ] 俯视地图渲染模块
- [ ] 扫描结果报告（每帧记录 + 每栋房屋汇总 + VLM 描述）

---

## Phase 2：入口探索——找到目标房屋的可进入入口

### 目标

UAV 到达目标房屋附近后，**在房屋周围探索，找到可以进入的入口**。

Phase 2 的触发条件：Phase 1 确认了目标房屋 + Phase 3 飞到了目标房屋附近。

Phase 2 需要回答：
1. 这栋房屋有没有门？门在哪个方向？
2. 门是开着的还是关着的？
3. 门前有没有障碍物？能不能通过？（depth 判断）
4. 现在检测到的门确实属于目标房屋吗？（坐标确认）

### 与 Phase 1 的区别

| | Phase 1（房屋确认） | Phase 2（入口探索） |
|---|---|---|
| **时机** | 任务开始时，原地扫描 | 到达目标房屋附近后 |
| **检测对象** | 只检测 `building` | 检测 `door_open`, `door_closed`, `window` |
| **是否用 depth** | 不用 | 用（判断门距、障碍物） |
| **UAV 行为** | 原地旋转，不飞行 | 绕房屋飞行，找门 |
| **YOLO 模型** | Phase 1 building 检测器 | Phase 2 入口检测器（独立模型） |

### Phase 2 入口检测 YOLO 模型

Phase 2 使用**独立的 YOLO 模型**，专门检测入口相关类别：

```yaml
# phase2_entry_detector.yaml
path: ./entry_detection_data
train: images/train
val: images/val
names:
  0: door_open       # 开着的门（可进入）
  1: door_closed     # 关闭的门（不可进入）
  2: window          # 窗户（辅助定位，不可进入）
```

**为什么用独立模型而不复用 Phase 1 的**：
- Phase 1 的 building 检测器针对远/中距离的整栋建筑物轮廓优化
- Phase 2 的入口检测器针对近距离的门框、窗框等小目标优化
- 训练数据分布不同：Phase 1 是远景全貌，Phase 2 是近景细节
- 分开训练，各自效果最优

#### 训练数据采集

UAV 到达每栋房屋附近后，绕房屋飞行采集近距离图片：

| 类别 | 距离 | 角度 | 数量 | 标注 |
|------|------|------|------|------|
| `door_open` | 近（1-3m）| 正面 ±45° | 每门 25 张 | bbox 框住门框 |
| `door_open` | 中（3-6m）| 正面 ±60° | 每门 15 张 | bbox 框住门框 |
| `door_closed` | 近+中 | 多角度 | 每门 15 张 | bbox 框住门框 |
| `window` | 近+中 | 多角度 | 每栋 15 张 | bbox 框住窗框 |
| 负样本（无门墙面）| 近+中 | — | 100 张 | 无标注 |

假设 3 栋房屋 × 2 门 = 6 门：
- door_open: 6 × 40 = 240 张
- door_closed: 6 × 15 = 90 张
- window: 3 × 15 = 45 张
- 负样本: 100 张

总计约 **475 张**，按 8:2 分训练/验证。

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

#### 训练验证标准

| 指标 | 要求 |
|------|------|
| door_open mAP@0.5 | ≥ 0.85 |
| door_open recall | ≥ 0.90（不能漏检开门） |
| door_closed precision | ≥ 0.80 |
| 整体 mAP@0.5 | ≥ 0.80 |
| 推理速度（640px） | ≤ 8ms / 帧 |

### 入口探索流程

Phase 2 在 `house_circling` 阶段执行：

```
UAV 到达目标房屋的 approach_radius 范围内
  │
  ▼
切换 stage_label = "house_circling"
加载 Phase 2 入口检测模型
  │
  ▼
绕房屋飞行，每步：
  │
  ├── ① 拍 RGB + depth
  │
  ├── ② YOLO 入口检测：
  │     检测到 door_open → 候选入口
  │     检测到 door_closed → 记录，但不是入口
  │     检测到 window → 记录，辅助定位
  │     什么都没检测到 → 继续绕飞
  │
  ├── ③ 如果检测到 door_open：
  │     │
  │     ├── 坐标验证：UAV 当前在 target_house bbox 内？
  │     │   → 不在 → 忽略（可能看到邻居房屋的门）
  │     │   → 在 → 继续判断
  │     │
  │     ├── depth 判断门距：
  │     │   → depth 中心区域 > 250cm → 继续靠近
  │     │   → depth 中心区域 ≤ 250cm → 已经够近
  │     │
  │     ├── depth 判断可通过性：
  │     │   → 门框区域 depth 连续（无障碍物）→ 可穿越
  │     │   → 门框区域有近距障碍 → 不可穿越
  │     │
  │     └── 对齐判断：
  │         → door bbox 中心偏离画面中心 > 15% → 调整偏航
  │         → door bbox 太小（< 25% 画面宽度）→ 继续靠近
  │         → 全部满足 → 可以进入 ✓
  │
  └── ④ 进入条件全部满足：
        → 切换 stage_label = "approaching_entry" → "entering"
        → 前进穿过门框

绕飞一整圈没找到入口：
  → 标记该房屋暂时无法进入
  → 尝试下一栋 unexplored 房屋
```

### 入口条件判断（纯规则，不训练）

```python
def check_entry_condition(door_det, depth_frame, frame_width, frame_height):
    """
    判断检测到的门是否满足进入条件。
    结合 YOLO 检测结果和 depth 信息。

    返回: ("enter" | "approach" | "reposition" | "blocked", 原因)
    """
    bbox = door_det["bbox"]  # [x1, y1, x2, y2]
    bbox_cx = (bbox[0] + bbox[2]) / 2
    bbox_cy = (bbox[1] + bbox[3]) / 2
    bbox_w = bbox[2] - bbox[0]

    # 1. 门距判断（depth）
    # 取门框区域的 depth 中位值
    door_region = depth_frame[
        int(bbox[1]):int(bbox[3]),
        int(bbox[0]):int(bbox[2])
    ]
    door_distance_cm = np.percentile(door_region, 20)

    if door_distance_cm > 250:
        return "approach", f"门距 {door_distance_cm:.0f}cm，继续靠近"

    # 2. 对齐判断
    cx_offset = abs(bbox_cx - frame_width / 2) / (frame_width / 2)
    if cx_offset > 0.15:
        return "reposition", f"门偏离中心 {cx_offset:.1%}，调整偏航"

    # 3. 门框大小判断
    bbox_w_ratio = bbox_w / frame_width
    if bbox_w_ratio < 0.25:
        return "approach", f"门框太小（{bbox_w_ratio:.1%}），继续靠近"

    # 4. 可穿越性判断（depth 连续性）
    # 门框中心区域的 depth 应该大于门框周围（说明门后是空间）
    door_center_depth = np.median(door_region[
        door_region.shape[0]//4 : 3*door_region.shape[0]//4,
        door_region.shape[1]//4 : 3*door_region.shape[1]//4
    ])
    if door_center_depth < 80:  # 门后 80cm 内有障碍
        return "blocked", f"门后有障碍物（depth {door_center_depth:.0f}cm）"

    return "enter", "所有条件满足，可以进入"
```

### 目标房屋门的验证

检测到的门可能属于邻居房屋。验证方法：

```python
def verify_door_belongs_to_target(uav_pose, houses_config, target_house_id):
    """
    坐标判断：UAV 当前是否在目标房屋范围内。
    Phase 2 只在目标房屋范围内时才响应 door 检测。
    """
    x, y = uav_pose["x"], uav_pose["y"]
    target = next(h for h in houses_config["houses"]
                  if h["house_id"] == target_house_id)
    bbox = target["bbox"]

    in_target = (bbox["x_min"] <= x <= bbox["x_max"]
                 and bbox["y_min"] <= y <= bbox["y_max"])

    if not in_target:
        return False, "UAV 不在目标房屋范围内，忽略门检测"
    return True, "确认在目标房屋范围内"
```

### 蒸馏特征输出

Phase 2 为 Phase 5 蒸馏提供以下特征：

```
门感知特征（5维）：
  door_detected              ← 检测到 door_open（0/1）
  door_confidence            ← 检测置信度（0-1）
  door_distance_norm         ← depth 测量的门距归一化（/500cm）
  door_alignment_norm        ← 门中心偏移归一化（/0.5）
  door_traversable           ← 可穿越判断（0/1）
```

### 交付物

- [ ] Phase 2 入口检测 YOLO 模型训练数据（475 张，3 类：door_open/door_closed/window）
- [ ] Phase 2 YOLO 模型权重 `entry_detector_v1/best.pt`
- [ ] 入口条件判断函数 `check_entry_condition()`
- [ ] 目标房屋门验证函数 `verify_door_belongs_to_target()`
- [ ] 绕房屋探索入口的策略脚本
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
Phase 1 的输出 → building 检测 + 方向匹配（11维）──┐
Phase 2 的输出 → door/depth 感知（5维）            ─┤
Phase 3 的轨迹 → outdoor_nav 标签数据              ─┤──► 蒸馏网络输入
Phase 4 的轨迹 → indoor 标签数据                   ─┤
server 提供     → pose, depth, risk                ─┘
```

### 蒸馏网络架构

```
输入层（30维特征向量）
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

#### 输入特征定义（30维）

```
坐标定位特征（6维，确定性，来自 Phase 1）：
  distance_to_target_norm        ← 到目标房屋中心的归一化距离（/2000cm）
  bearing_error_norm             ← 方位角误差归一化（/180°）
  is_at_target_house             ← 是否在目标房屋范围内（0/1）
  is_at_wrong_house              ← 是否在非目标房屋范围内（0/1）
  is_inside                      ← 是否在室内（0/1）
  target_approach_progress       ← 接近目标的进度（0-1）

YOLO building 特征（3维，概率性，来自 Phase 1）：
  yolo_building_detected         ← 当前帧检测到 building（0/1）
  yolo_building_confidence       ← building 最高置信度（0-1）
  yolo_building_count            ← building 数量（归一化 /5）

方向匹配特征（2维，确定性，来自 Phase 1）：
  direction_match_confirmed      ← 方向匹配确认看到目标房屋（0/1）
  nearest_building_bearing_error ← 最近 building 与目标方位角误差（归一化 /180°）

门感知特征（5维，来自 Phase 2）：
  door_detected                  ← 检测到 door_open（0/1）
  door_confidence                ← 检测置信度（0-1）
  door_distance_norm             ← depth 门距归一化（/500cm）
  door_alignment_norm            ← 门中心偏移归一化（/0.5）
  door_traversable               ← 可穿越判断（0/1）

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
| Phase 1 房屋确认 | — | `phase1_scan.py`, `direction_house_matcher.py`, `house_locator.py`, `search_status_manager.py`, `phase1_building_detector.yaml` |
| Phase 2 入口探索 | — | `check_entry_condition.py`, `verify_door_belongs_to_target.py`, `phase2_entry_detector.yaml` |
| Phase 3 跨房屋导航 | — | `rule_based_house_navigator.py` |
| Phase 4 室内搜索 | `reflex_policy_model.py`, `train_reflex_policy.py`, `reflex_dataset_builder.py` | 扩展 `FEATURE_NAMES` |
| Phase 5 蒸馏 | `train_reflex_policy.py`（扩展 MLP 架构） | `build_distillation_dataset.py` |
| Phase 6 迭代 | `online_reflex_eval.py` | `iteration_manager.py` |
