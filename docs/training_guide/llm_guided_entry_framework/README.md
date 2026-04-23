# LLM-Guided Entry Framework

## 目录定位

这个目录用于整理一条新的论文路线：

- 多模态入口感知
- LLM / VLM 高层语义判断与动作引导
- 局部门进入与静态避障技能学习
- 从 LLM-guided teacher 到轻量策略的蒸馏

它和 [`phases`](../phases/README.md) 目录的区别是：

- `phases/` 更偏完整流水线分阶段说明
- 本目录更偏一条可直接写论文的方法主线

## 建议阅读顺序

1. [`01_problem_and_method.md`](01_problem_and_method.md)
2. [`02_training_flow.md`](02_training_flow.md)
3. [`03_experiment_plan.md`](03_experiment_plan.md)
4. [`04_stepc_labeling_standard.md`](04_stepc_labeling_standard.md)
5. [`05_pure_llm_baseline_test.md`](05_pure_llm_baseline_test.md)
6. [`06_agent_state_schema.md`](06_agent_state_schema.md)
7. [`07_teacher_schema_spec.md`](07_teacher_schema_spec.md)
8. [`08_entry_state_builder_spec.md`](08_entry_state_builder_spec.md)
9. [`09_distillation_dataset_export_spec.md`](09_distillation_dataset_export_spec.md)
10. [`10_implementation_sequence.md`](10_implementation_sequence.md)
11. [`11_teacher_validator_spec.md`](11_teacher_validator_spec.md)
12. [`12_pilot_dataset_plan.md`](12_pilot_dataset_plan.md)
13. [`13_representation_distillation_plan.md`](13_representation_distillation_plan.md)
14. [`14_target_conditioned_entry_fusion.md`](14_target_conditioned_entry_fusion.md)
15. [`15_target_conditioned_fusion_schema.md`](15_target_conditioned_fusion_schema.md)
16. [`16_target_match_score_design.md`](16_target_match_score_design.md)
17. [`17_target_conditioned_teacher_spec.md`](17_target_conditioned_teacher_spec.md)
18. [`18_target_conditioned_entry_state_builder_spec.md`](18_target_conditioned_entry_state_builder_spec.md)
19. [`19_target_conditioned_dataset_export_spec.md`](19_target_conditioned_dataset_export_spec.md)
20. [`20_representation_distillation_trainer_spec.md`](20_representation_distillation_trainer_spec.md)
21. [`21_representation_distillation_implementation_plan.md`](21_representation_distillation_implementation_plan.md)
22. [`22_v3_followup_data_collection_plan.md`](22_v3_followup_data_collection_plan.md)
23. [`22_v3_followup_data_collection_plan_zh.md`](22_v3_followup_data_collection_plan_zh.md)
24. [`23_entry_search_memory_design_zh.md`](23_entry_search_memory_design_zh.md)
25. [`24_entry_search_memory_schema.md`](24_entry_search_memory_schema.md)
26. [`25_entry_search_memory_implementation_plan.md`](25_entry_search_memory_implementation_plan.md)
27. [`26_memory_aware_data_collection_spec_zh.md`](26_memory_aware_data_collection_spec_zh.md)
28. [`27_memory_snapshot_export_spec.md`](27_memory_snapshot_export_spec.md)
29. [`28_memory_architecture_diagram_zh.md`](28_memory_architecture_diagram_zh.md)
30. [`29_v5_training_improvement_plan_zh.md`](29_v5_training_improvement_plan_zh.md)
31. [`30_memory_function_test_checklist_zh.md`](30_memory_function_test_checklist_zh.md)

## 这条路线的核心思想

不是让 LLM 直接长期在线控飞，而是：

1. 用 `YOLO26 + depth` 做稳定的底层几何与语义感知
2. 用 `LLM/VLM` 做更高层的入口语义判断与局部子任务引导
3. 用 `RL / BC` 学习局部进入技能
4. 最后蒸馏成轻量、可实时执行的策略

## 目标问题

给定目标房屋，UAV 需要在户外场景中：

1. 识别当前看到的是不是目标房屋
2. 识别可进入入口，而不是窗户或假开口
3. 判断当前应该靠近、绕行还是继续搜索
4. 安全进入目标房屋

## 论文主线建议

推荐聚焦成 3 个贡献：

1. 多模态入口感知与可穿越性判断
2. LLM-guided 局部进入决策与技能学习
3. 从 teacher policy 到轻量执行策略的蒸馏
