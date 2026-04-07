# UAV-Flow 训练流水线文档（分阶段）

> 源文件：[`full_training_pipeline.md`](../full_training_pipeline.md)（完整版，未删除）

## 文档结构

| 文件 | 内容 | 行数 |
|------|------|------|
| [`00_overview.md`](00_overview.md) | 任务定义、训练链路、模块依赖 | 69 |
| [`phase0_environment.md`](phase0_environment.md) | 环境准备与数据基础设施 | 107 |
| [`phase1_map_marking.md`](phase1_map_marking.md) | 地图标记与坐标定位（轻量化） | 396 |
| [`phase2_entry_discovery.md`](phase2_entry_discovery.md) | **入口探索——多模态融合（核心创新）** | 2020 |
| [`phase3_navigation.md`](phase3_navigation.md) | 跨房屋导航策略训练 | 91 |
| [`phase4_indoor_search.md`](phase4_indoor_search.md) | 单房屋室内搜索策略训练（BC） | 164 |
| [`phase5_distillation.md`](phase5_distillation.md) | 多房屋任务调度蒸馏（28维 MLP） | 151 |
| [`phase6_online_finetune.md`](phase6_online_finetune.md) | 在线微调与数据飞轮 | 45 |
| [`07_data_spec.md`](07_data_spec.md) | 训练数据规范 | 61 |
| [`08_metrics.md`](08_metrics.md) | 评价指标体系 | 28 |
| [`09_hardware.md`](09_hardware.md) | 硬件与时间估算 | 26 |
| [`10_troubleshooting.md`](10_troubleshooting.md) | 常见问题与调试策略 | 57 |
| [`11_tmm_experiment.md`](11_tmm_experiment.md) | 论文实验计划（IEEE TMM） | 543 |
| [`12_planner_executor.md`](12_planner_executor.md) | Planner-Driven 自主探索执行器 | 281 |
| [`13_upgrade_roadmap.md`](13_upgrade_roadmap.md) | 系统升级路线图 | 118 |
| [`14_appendix.md`](14_appendix.md) | 附录：关键文件对照 | 12 |

## Phase 2 内部结构（核心文档）

Phase 2 是最大的文档（2020 行），内部按 Step A-G 组织：

```
Step A：YOLO 训练数据采集（手动截图，不需要坐标/depth）
Step B：YOLO 入口检测训练（Roboflow 标注 + YOLOv8n）
── YOLO 训练完成分界线 ──
Step C：Depth 分析模块（在线验证）
Step D：VLM 部署与 Prompt 调优（LLaVA / GPT-4o）
Step E：融合决策模块（Rule-based + MLP + Attention）
Step F：绕飞策略与入口进入
Step G：集成测试与端到端验证
```

## 阅读建议

- 快速了解全局 → `00_overview.md`
- 开始实施 → 按 Phase 0 → 1 → 2 → ... 顺序
- 写论文 → `11_tmm_experiment.md` + `phase2_entry_discovery.md`
- 当前重点 → `phase2_entry_discovery.md`（核心创新）
