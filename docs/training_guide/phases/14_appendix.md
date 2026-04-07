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

