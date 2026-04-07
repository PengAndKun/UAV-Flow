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

