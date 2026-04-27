# 36. V6 数据补采与再训练计划

## 1. 文档目的

V5 实验已经证明两点：

```text
1. memory-aware representation 已经能稳定识别 no-entry full-coverage。
2. memory ablation 后 no-entry 判断显著下降，说明模型确实使用了结构化记忆。
```

因此，下一步不是进入 action / RL 训练，而是补齐 V5 中仍然薄弱的目标入口状态，让 `z_entry` 表示对更多入口搜索情况稳定。

V6 的目标是：

```text
补齐 target_house_entry_approachable
补齐 target_house_entry_blocked
补齐 target_house_geometric_opening_needs_confirmation
补齐 target_house_entry_visible
补齐目标入口 / 非目标入口 / window 同屏干扰
```

然后重新跑：

```text
V6 data collection
-> LLM teacher
-> dataset export
-> memory_aware_v6 training
-> embedding analysis
-> formal memory ablation
-> V5 vs V6 对比
```

---

## 2. 当前 V5 薄弱点

当前 V5 数据分布：

| 类别 | V5 数量 | 问题 |
|---|---:|---|
| `target_house_not_in_view` | 44 | 足够 |
| `non_target_house_entry_visible` | 42 | 基本够用 |
| `target_house_no_entry_after_full_coverage` | 27 | 已证明有效 |
| `target_house_entry_blocked` | 9 | 偏少 |
| `target_house_entry_approachable` | 6 | 明显偏少 |
| `target_house_entry_visible` | 1 | 严重不足 |
| `target_house_geometric_opening_needs_confirmation` | 1 | 严重不足 |

V5 表示模型已经很好地学会了：

```text
not_in_view
non_target_entry
no_entry_after_full_coverage
```

但还需要加强：

```text
目标入口出现但尚未稳定可接近
目标入口被遮挡或路径不可通
疑似几何开口但需要继续确认
目标入口可接近并准备 approach
```

---

## 3. V6 数据目标

V6 补采后，建议总数据规模达到：

```text
总样本数 >= 250
新增样本数 >= 120
```

目标类别数量：

| 类别 | V5 数量 | V6 目标总数 | 建议新增 |
|---|---:|---:|---:|
| `target_house_entry_approachable` | 6 | >= 30 | >= 24 |
| `target_house_entry_blocked` | 9 | >= 30 | >= 21 |
| `target_house_geometric_opening_needs_confirmation` | 1 | >= 15 | >= 14 |
| `target_house_entry_visible` | 1 | >= 15 | >= 14 |
| `target_house_no_entry_after_full_coverage` | 27 | >= 35 | >= 8 |
| `non_target_house_entry_visible` | 42 | >= 50 | >= 8 |

最低通过线：

```text
approachable >= 20
blocked >= 20
geometric_opening >= 10
entry_visible >= 10
```

理想通过线：

```text
approachable >= 30
blocked >= 30
geometric_opening >= 15
entry_visible >= 15
```

---

## 4. 推荐采集 Episode 清单

### 4.1 House 1 正门可接近

Episode 名称：

```text
search_house_1_front_approach_v6
```

目的：

```text
补 target_house_entry_approachable
补 approach_target_entry
补 candidate_0 目标入口稳定关联
```

采集方式：

```text
从远处正对 house 1
缓慢靠近正门
保持门在画面中心附近
至少采到 20 个 capture
最后 5-8 帧应该稳定为 approachable
```

合格标准：

```text
candidate_entry_count >= 1
has_reliable_entry = true
best_entry_status = approachable
target_conditioned_state = target_house_entry_approachable
target_conditioned_subgoal = approach_target_entry
```

---

### 4.2 House 1 左侧绕行确认

Episode 名称：

```text
search_house_1_left_detour_to_entry_v6
```

目的：

```text
补 blocked -> detour -> approachable 的时序过程
```

采集方式：

```text
先从偏左或被遮挡角度看目标入口
让门部分可见但路径不直接可通
随后向左侧或侧前方移动
最后确认入口可接近
```

预期状态演化：

```text
target_house_entry_blocked
-> target_house_entry_visible
-> target_house_entry_approachable
```

合格标准：

```text
前半段至少 5 帧 blocked
中间至少 3 帧 entry_visible 或 geometric_opening
后半段至少 5 帧 approachable
```

---

### 4.3 House 1 右侧绕行确认

Episode 名称：

```text
search_house_1_right_detour_to_entry_v6
```

目的：

```text
补 detour_right_to_target_entry
补 blocked / visible / approachable 过渡
```

采集方式：

```text
与 left detour 类似，但从右侧绕行
保证视角变化明显
不要一开始就直接正对入口
```

合格标准：

```text
detour_right_to_target_entry >= 3
target_house_entry_blocked >= 5
target_house_entry_approachable >= 5
```

---

### 4.4 House 1 门前遮挡

Episode 名称：

```text
search_house_1_blocked_entry_v6
```

目的：

```text
补 target_house_entry_blocked
补 depth blocked evidence
```

采集方式：

```text
让入口可见，但无人机前方或入口路径被物体/墙体/角度遮挡
不要马上绕过去
在 blocked 状态多停留几个 capture
```

合格标准：

```text
target_house_entry_blocked >= 10
target_conditioned_subgoal = detour_left_to_target_entry 或 detour_right_to_target_entry
front_obstacle_present 或 depth blocked evidence 明显
```

---

### 4.5 几何开口待确认

Episode 名称：

```text
search_house_1_geometric_opening_uncertain_v6
```

目的：

```text
补 target_house_geometric_opening_needs_confirmation
```

采集方式：

```text
远距离观察疑似门洞
RGB/YOLO 不够确定
Depth 有开口迹象但还不能直接进入
不要太快靠近确认
围绕不确定视角多采几帧
```

合格标准：

```text
target_house_geometric_opening_needs_confirmation >= 8
target_conditioned_subgoal = keep_search_target_house
confidence 不需要特别高，但 teacher label 应该 PASS
```

---

### 4.6 目标入口与非目标入口同屏

Episode 名称：

```text
search_house_1_target_vs_non_target_same_view_v6
```

目的：

```text
补 target / non-target entry association
补 candidate id 稳定性
补 ignore_non_target_entry 与 approach_target_entry 区分
```

采集方式：

```text
让画面里同时出现目标 house 附近入口和其他 house 的入口/窗户
从不同角度采集
不要让模型只看到单一入口
```

合格标准：

```text
同一 episode 中同时出现：
non_target_house_entry_visible
target_house_entry_visible 或 target_house_entry_approachable
candidate_entry_count >= 2
```

---

### 4.7 Window 干扰

Episode 名称：

```text
search_house_1_window_distractor_v6
```

目的：

```text
补 window / fake opening 干扰
提升入口归属和过滤能力
```

采集方式：

```text
让窗户或类似门洞占据画面显著位置
同时目标入口可能不在画面中或只部分出现
从多个角度观察
```

合格标准：

```text
non_target_house_entry_visible 或 keep_search_target_house
candidate_entry_count >= 1
rejected_entry_count >= 1
best_entry_status 不应错误稳定为 approachable
```

---

### 4.8 House 2 无入口完整搜索复采

Episode 名称：

```text
search_house_2_full_no_entry_repeat_v6
```

目的：

```text
增强 no-entry 泛化
确认 V5 学到的 no-entry 不是只记住一次轨迹
```

采集方式：

```text
重新绕 house 2 完整搜索一圈
每个 sector 至少观察一次
不要过早结束
最后阶段才触发 complete_no_entry_search
```

合格标准：

```text
front_left / front_center / front_right / left_side / right_side 尽量覆盖
full_coverage_ready = true
has_reliable_entry = false
target_house_no_entry_after_full_coverage >= 8
```

---

## 5. 每个 Episode 的采集数量建议

| Episode | 建议 capture 数 |
|---|---:|
| `search_house_1_front_approach_v6` | 20-30 |
| `search_house_1_left_detour_to_entry_v6` | 25-35 |
| `search_house_1_right_detour_to_entry_v6` | 25-35 |
| `search_house_1_blocked_entry_v6` | 20-30 |
| `search_house_1_geometric_opening_uncertain_v6` | 20-30 |
| `search_house_1_target_vs_non_target_same_view_v6` | 25-40 |
| `search_house_1_window_distractor_v6` | 20-30 |
| `search_house_2_full_no_entry_repeat_v6` | 30-50 |

总新增建议：

```text
新增 episode >= 8
新增 capture >= 180
```

如果采集时间有限，最低建议：

```text
新增 episode >= 5
新增 capture >= 100
```

优先级：

```text
P0: front_approach / blocked_entry / left_detour / right_detour
P1: geometric_opening / target_vs_non_target_same_view
P2: window_distractor / house_2_no_entry_repeat
```

---

## 6. 采集前启动命令

Server：

```powershell
python E:\github\UAV-Flow\UAV-Flow-Eval\uav_control_server_basic.py `
  --env_bin_win E:\github\UAV-Flow\UnrealEnv\UE4_ExampleScene_Win\UE4_ExampleScene\Binaries\Win64\UE4_ExampleScene.exe `
  --viewport_mode free `
  --preview_mode first_person `
  --fixed_spawn_pose_file E:\github\UAV-Flow\uav_fixed_spawn_pose.json `
  --capture_dir E:\github\UAV-Flow\captures_remote
```

Panel：

```powershell
python E:\github\UAV-Flow\UAV-Flow-Eval\uav_control_panel_basic.py `
  --host 127.0.0.1 `
  --port 5020 `
  --timeout_s 8 `
  --state_interval_ms 1500 `
  --preview_interval_ms 1500 `
  --depth_interval_ms 1800
```

采集设置建议：

```text
Memory Collection:
Episode Label = 对应 episode 名称
House ID = 当前目标 house
Auto Mode = step
Step Interval = 3
Reset Store On Start = checked
```

如果转向变化很大：

```text
q / e yaw 动作会触发 capture
触发后 step counter 应重新计数
```

---

## 7. 每个 Episode 的人工检查标准

采集过程中打开：

```text
Open Memory Window
```

重点看：

```text
target_house_id 是否正确
current_house_id 是否合理
entry_search_status 是否随过程变化
candidate_entry_count 是否增加
best_entry_id 是否稳定
best_entry_status 是否符合画面
full_coverage_ready 是否只在后期出现
```

采集结束后，每个 episode 应该有：

```text
entry_search_memory_snapshot_start.json
entry_search_memory_snapshot_stop.json
memory_fusion_captures/
```

每个 capture 应该有：

```text
labeling/sample_metadata.json
labeling/temporal_context.json
labeling/state.json
labeling/yolo_result.json
labeling/depth_result.json
labeling/fusion_result.json
labeling/fusion_overlay.png
labeling/entry_search_memory_snapshot_before.json
labeling/entry_search_memory_snapshot_after.json
```

---

## 8. 采集后第一轮验证

建议先看 session 目录：

```text
E:\github\UAV-Flow\captures_remote\memory_collection_sessions
```

需要确认：

```text
新增 episode 数量正确
每个 episode capture 数量符合预期
没有大量 missing_prompt / missing_fusion / missing_snapshot
```

如果需要先肉眼检查：

```text
fusion_overlay.png
entry_search_memory_snapshot_after.json
```

重点看：

```text
目标房屋是否对齐
入口候选是否合理
window 是否被 reject
no-entry 是否只在完整搜索后出现
```

---

## 9. LLM Teacher 标注

对新增 session 跑：

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_memory_aware_llm_teacher_batch.py `
  --session_dir E:\github\UAV-Flow\captures_remote\memory_collection_sessions\<episode_dir> `
  --model gpt-5.5
```

如果已经跑过部分样本，应使用已有跳过逻辑，避免重复访问。

标注后需要检查：

```text
llm_teacher_label.json
llm_teacher_label_validated.json
```

每个新增 capture 应该尽量达到：

```text
status = PASS
target_conditioned_state 非空
target_conditioned_subgoal 非空
target_conditioned_action_hint 非空
```

---

## 10. V6 Dataset Export

建议导出名称：

```text
phase2_5_memory_aware_dataset_v6
```

命令：

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\distillation_dataset_export.py `
  --results_root E:\github\UAV-Flow\captures_remote\memory_collection_sessions `
  --export_name phase2_5_memory_aware_dataset_v6 `
  --llm_teacher_mode require `
  --min_teacher_confidence 0.55
```

导出后检查：

```text
manifest.json
quality_report.json
train.jsonl
val.jsonl
```

重点看：

```text
total_exported
target_conditioned_state_counts
target_conditioned_subgoal_counts
memory_available_count
llm_teacher_available_count
```

V6 导出通过标准：

```text
total_exported >= 250
memory_available_count = total_exported
llm_teacher_available_count = total_exported
target_house_entry_approachable >= 20
target_house_entry_blocked >= 20
target_house_geometric_opening_needs_confirmation >= 10
target_house_entry_visible >= 10
```

---

## 11. V6 训练命令

建议 run name：

```text
memory_aware_v6_202604xx
```

训练：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python E:\github\UAV-Flow\phase2_5_representation_distillation\train_representation_distillation.py `
  --config_path E:\github\UAV-Flow\phase2_5_representation_distillation\configs\base_config.json `
  --export_dir E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\<v6_export_dir> `
  --run_name memory_aware_v6_202604xx `
  --device cpu `
  --epochs 80 `
  --stage1_epochs 5 `
  --stage2_epochs 12 `
  --batch_size 16 `
  --seed 202604280
```

评估：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:MPLBACKEND='Agg'
python E:\github\UAV-Flow\phase2_5_representation_distillation\evaluate_representation_distillation.py `
  --checkpoint_path E:\github\UAV-Flow\phase2_5_representation_distillation\runs\memory_aware_v6_202604xx\checkpoints\best.pt `
  --export_dir E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\<v6_export_dir> `
  --device cpu `
  --run_name memory_aware_v6_202604xx_eval
```

---

## 12. V6 Embedding Analysis

导出 embedding：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python E:\github\UAV-Flow\phase2_5_representation_distillation\export_representation_embeddings.py `
  --checkpoint_path E:\github\UAV-Flow\phase2_5_representation_distillation\runs\memory_aware_v6_202604xx\checkpoints\best.pt `
  --export_dir E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\<v6_export_dir> `
  --run_name memory_aware_v6_202604xx `
  --split all `
  --device cpu `
  --batch_size 32
```

分析：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:MPLBACKEND='Agg'
python E:\github\UAV-Flow\phase2_5_representation_distillation\analyze_representation_embeddings.py `
  --embedding_dir E:\github\UAV-Flow\phase2_5_representation_distillation\embeddings\memory_aware_v6_202604xx `
  --nearest_k 5
```

重点看：

```text
no_entry_separation_margin
temporal_warning_count
state_centroid_distance.csv
embedding_projection.png
nearest_neighbors.jsonl
```

---

## 13. V6 Formal Ablation

如果 V6 单次训练通过，再跑正式消融：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python E:\github\UAV-Flow\phase2_5_representation_distillation\run_formal_ablation_experiments.py `
  --base_export_dir E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\<v6_export_dir> `
  --experiment_name memory_aware_v6_formal_ablation_10seed_10split `
  --num_repeats 10 `
  --start_seed 202604280 `
  --experiment_modes seed_retrain random_split `
  --inference_repeats 1
```

V6 正式消融通过标准：

```text
none state accuracy mean >= 0.90
none subgoal accuracy mean >= 0.85
none no-entry margin mean >= 0.70
zero_memory no-entry margin delta mean <= -0.50
zero_candidates no-entry margin delta 接近 0
```

---

## 14. V5 vs V6 对比指标

最终对比表应包含：

| 指标 | V5 | V6 | 期望变化 |
|---|---:|---:|---|
| val state acc | 当前值 | 新值 | 不下降或提升 |
| val subgoal acc | 当前值 | 新值 | 不下降或提升 |
| no-entry margin | 当前值 | 新值 | 保持高 |
| approachable support | 6 | >= 30 | 明显增加 |
| blocked support | 9 | >= 30 | 明显增加 |
| geometric opening support | 1 | >= 15 | 明显增加 |
| formal zero_memory margin delta | 已显著下降 | 仍显著下降 | 证明 memory 稳定有效 |

V6 最关键的不是让所有指标都比 V5 高，而是：

```text
在类别更完整、场景更复杂的情况下，memory-aware representation 仍然稳定。
```

---

## 15. 当前下一步

现在最直接的任务是：

```text
开始采集 V6 memory sessions
```

建议先采 P0：

```text
1. search_house_1_front_approach_v6
2. search_house_1_blocked_entry_v6
3. search_house_1_left_detour_to_entry_v6
4. search_house_1_right_detour_to_entry_v6
```

采完这 4 个后，先不要急着继续采所有 P1/P2。

先做一次快速检查：

```text
新增样本是否真的补到了 approachable / blocked
memory snapshot 是否稳定
LLM teacher 是否能 PASS
```

如果 P0 质量合格，再继续采：

```text
geometric_opening_uncertain
target_vs_non_target_same_view
window_distractor
house_2_full_no_entry_repeat
```

