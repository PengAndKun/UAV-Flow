# UAV-Flow 训练阶段文档索引

> 源文档：[`../full_training_pipeline.md`](../full_training_pipeline.md)

## 说明

这个目录将完整训练流水线拆成按阶段阅读和执行的独立文档，便于：

- 按阶段推进实现
- 单独维护某一阶段的方法设计
- 论文写作时引用对应模块
- 后续继续插入中间阶段而不破坏总流程

## 阶段文档

| 文件 | 主题 | 说明 |
|---|---|---|
| [`00_overview.md`](00_overview.md) | 总览 | 任务定义、模块关系、整体流程 |
| [`phase0_environment.md`](phase0_environment.md) | Phase 0 | 环境准备与基础数据设施 |
| [`phase1_map_marking.md`](phase1_map_marking.md) | Phase 1 | 地图标定、房屋坐标与定位 |
| [`phase2_entry_discovery.md`](phase2_entry_discovery.md) | Phase 2 | 多模态入口发现与融合判断 |
| [`phase2_5_entry_skill_rl.md`](phase2_5_entry_skill_rl.md) | Phase 2.5 | 局部门进入、静态避障、技能学习 |
| [`phase3_navigation.md`](phase3_navigation.md) | Phase 3 | 跨房屋导航 |
| [`phase4_indoor_search.md`](phase4_indoor_search.md) | Phase 4 | 室内搜索与确认 |
| [`phase5_distillation.md`](phase5_distillation.md) | Phase 5 | 轨迹蒸馏与轻量策略 |
| [`phase6_online_finetune.md`](phase6_online_finetune.md) | Phase 6 | 在线微调与迭代 |

## 支撑文档

| 文件 | 主题 |
|---|---|
| [`07_data_spec.md`](07_data_spec.md) | 数据格式规范 |
| [`08_metrics.md`](08_metrics.md) | 评价指标 |
| [`09_hardware.md`](09_hardware.md) | 硬件与时间估算 |
| [`10_troubleshooting.md`](10_troubleshooting.md) | 常见问题与调试 |
| [`11_tmm_experiment.md`](11_tmm_experiment.md) | 论文实验计划 |
| [`12_planner_executor.md`](12_planner_executor.md) | Planner-Driven 执行器设计 |
| [`13_upgrade_roadmap.md`](13_upgrade_roadmap.md) | 升级路线图 |
| [`14_appendix.md`](14_appendix.md) | 附录 |

## 当前推荐阅读顺序

1. [`00_overview.md`](00_overview.md)
2. [`phase0_environment.md`](phase0_environment.md)
3. [`phase1_map_marking.md`](phase1_map_marking.md)
4. [`phase2_entry_discovery.md`](phase2_entry_discovery.md)
5. [`phase2_5_entry_skill_rl.md`](phase2_5_entry_skill_rl.md)
6. [`phase3_navigation.md`](phase3_navigation.md)
7. [`phase4_indoor_search.md`](phase4_indoor_search.md)
8. [`phase5_distillation.md`](phase5_distillation.md)
9. [`phase6_online_finetune.md`](phase6_online_finetune.md)

## Phase 2.5 的定位

新增的 Phase 2.5 位于：

- Phase 2 感知融合之后
- Phase 3 / Phase 4 执行动作之前

它不负责“找哪一栋房子”，也不负责“室内完整寻人”，而是单独学习一个局部技能：

- 发现可进入入口后，如何接近入口
- 前方有静态障碍时，如何绕行
- 对齐门洞后，如何稳定进入

它也是后续 Phase 5 蒸馏的重要 teacher 来源。
