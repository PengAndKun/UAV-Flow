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

