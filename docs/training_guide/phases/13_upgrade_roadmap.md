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

