# 30. Memory 功能测试 Checklist

## 1. 文档目的

这份文档用于把当前 `memory-aware collection + memory-aware fusion` 的测试流程收成一页式 checklist。

目标不是验证“模型最终效果有多好”，而是先验证下面三件事是否真的成立：

1. `memory` 采集链是否打通
2. `memory` 是否真正被写入样本与 snapshot
3. `memory` 是否已经开始影响 fusion 决策

---

## 2. 测试总览

建议按下面顺序测：

1. 基础采集链测试
2. 低收益 sector 测试
3. blocked 门重复测试
4. 非目标门历史抑制测试
5. sequence / phase1 scan 时序测试

---

## 2.1 最短测试路线

如果你现在只想先确认“memory 功能到底通没通”，建议先按下面这条最短路线做：

### 步骤 A：启动

1. 启动：
   - [uav_control_server_basic.py](/E:/github/UAV-Flow/UAV-Flow-Eval/uav_control_server_basic.py)
   - [uav_control_panel_basic.py](/E:/github/UAV-Flow/UAV-Flow-Eval/uav_control_panel_basic.py)
2. 推荐直接使用下面这组命令启动：

```powershell
python E:\github\UAV-Flow\UAV-Flow-Eval\uav_control_server_basic.py `
  --env_bin_win E:\github\UAV-Flow\UnrealEnv\UE4_ExampleScene_Win\UE4_ExampleScene\Binaries\Win64\UE4_ExampleScene.exe `
  --viewport_mode free `
  --preview_mode first_person `
  --fixed_spawn_pose_file E:\github\UAV-Flow\uav_fixed_spawn_pose.json `
  --capture_dir E:\github\UAV-Flow\captures_remote
```

```powershell
python E:\github\UAV-Flow\UAV-Flow-Eval\uav_control_panel_basic.py `
  --host 127.0.0.1 `
  --port 5020 `
  --timeout_s 8 `
  --state_interval_ms 1500 `
  --preview_interval_ms 1500 `
  --depth_interval_ms 1800
```

3. 如果你的 Unreal 可执行文件路径或采集目录不同，只需要替换：
   - `--env_bin_win`
   - `--capture_dir`
4. 在 panel 中打开：
   - `Open Memory Window`

### 步骤 B：开始一个 episode

1. 选择 `House ID`
2. 填一个 `Task Label`
3. 在 `Memory Collection` 里输入一个 `Episode Label`
4. 点击：
   - `Start Episode`

### 步骤 C：执行最小采集

1. 做 2 到 3 步移动
2. 点击：
   - `Capture`
3. 点击：
   - `Snapshot Now`

### 步骤 D：停止

1. 点击：
   - `Stop Episode`

### 步骤 E：立刻检查

先检查下面三件事：

1. `Memory Collection Inspector` 里：
   - `episode_id` 非空
   - `step_index` 大于 0
   - `snapshot_count` 大于 0

2. `capture_dir` 中新增的 `*_bundle.json` 里：
   - `memory_episode_id`
   - `memory_step_index`
   - `memory_snapshot_before_path`
   - `memory_snapshot_after_path`

3. [entry_search_memory.json](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/entry_search_memory.json) 里：
   - `current_target_house_id`
   - 对应 `house_id` 的 memory 已更新

如果这三项都成立，就说明：

- memory 采集链已经打通
- 现在可以继续测更深入的 `sector / blocked / sequence / phase1 scan`

---

## 3. 测试前准备

### 3.1 启动服务

启动：

- [uav_control_server_basic.py](/E:/github/UAV-Flow/UAV-Flow-Eval/uav_control_server_basic.py)
- [uav_control_panel_basic.py](/E:/github/UAV-Flow/UAV-Flow-Eval/uav_control_panel_basic.py)

推荐命令：

```powershell
python E:\github\UAV-Flow\UAV-Flow-Eval\uav_control_server_basic.py `
  --env_bin_win E:\github\UAV-Flow\UnrealEnv\UE4_ExampleScene_Win\UE4_ExampleScene\Binaries\Win64\UE4_ExampleScene.exe `
  --viewport_mode free `
  --preview_mode first_person `
  --fixed_spawn_pose_file E:\github\UAV-Flow\uav_fixed_spawn_pose.json `
  --capture_dir E:\github\UAV-Flow\captures_remote
```

```powershell
python E:\github\UAV-Flow\UAV-Flow-Eval\uav_control_panel_basic.py `
  --host 127.0.0.1 `
  --port 5020 `
  --timeout_s 8 `
  --state_interval_ms 1500 `
  --preview_interval_ms 1500 `
  --depth_interval_ms 1800
```

如果你已经有自己常用的启动命令，也可以继续用，只要保证：

- server 和 panel 端口一致
- `--capture_dir` 指向你当前用于采集的目录
- server 能正常加载 Unreal 环境

### 3.2 建议窗口

建议同时打开：

1. 主控制面板
2. `Memory Collection Inspector`
3. 如有需要，再打开：
   - RGB Preview
   - Depth Preview

### 3.3 关键观察文件

重点看这些：

- [entry_search_memory.json](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/entry_search_memory.json)
- `capture_dir` 下的 `*_bundle.json`
- `capture_dir` 下的：
  - `*_entry_search_memory_snapshot_before.json`
  - `*_entry_search_memory_snapshot_after.json`
- `phase1_scan_manifest.json`
- 各样本目录中的：
  - `fusion_result.json`

---

## 4. 基础采集链测试

### 4.1 操作步骤

1. 在 panel 中选择一个 `House ID`
2. 设置 `Task Label`
3. 在 `Memory Collection` 区域点击：
   - `Start Episode`
4. 执行几步移动
5. 点击：
   - `Capture`
6. 点击：
   - `Snapshot Now`
7. 点击：
   - `Stop Episode`

### 4.2 期望现象

你应该看到：

1. `Memory Collection Inspector` 中：
   - `active` 从 `false -> true -> false`
   - `episode_id` 非空
   - `step_index` 增长
   - `snapshot_count` 增长

2. `*_bundle.json` 中出现：
   - `memory_collection_active`
   - `memory_episode_id`
   - `memory_step_index`
   - `memory_snapshot_before_path`
   - `memory_snapshot_after_path`

3. 采集目录中出现：
   - `*_entry_search_memory_snapshot_before.json`
   - `*_entry_search_memory_snapshot_after.json`

### 4.3 判定标准

通过标准：

- `episode_id` 正常生成
- `step_index` 不是一直为 0
- capture 结果中带有 before/after memory snapshot

---

## 5. 低收益 Sector 测试

### 5.1 测试目标

验证系统是否会记住：

- 某个 `sector` 已经反复看过
- 而且收益很低
- 从而触发 `sector_penalty` 或高层换边策略

### 5.2 操作步骤

1. `Start Episode`
2. 选择一个目标房屋
3. 从相似角度连续观察同一房屋侧面
4. 尽量让画面里主要是：
   - 墙面
   - 窗户
   - 弱门线索
5. 连续多次：
   - 小幅转向
   - Capture
   - 或运行同一类 fusion 样本

### 5.3 期望现象

在 memory 中：

- `searched_sectors.<sector>.observation_count` 增长
- `best_target_match_score` 仍然偏低

在 fusion 结果中，后续应逐渐看到：

- `memory_guidance.sector_penalty_applied = true`
或
- `memory_decision_guidance.override_reason = low_yield_sector_shift`

### 5.4 重点观察字段

看：

- [entry_search_memory.json](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/entry_search_memory.json)
  - `semantic_memory.searched_sectors`
- `fusion_result.json`
  - `memory_guidance`
  - `memory_decision_guidance`

---

## 6. Blocked 门重复测试

### 6.1 测试目标

验证系统是否会记住：

- 某个目标门已经连续多次被确认 `blocked`
- 不再无限死盯同一个 blocked 门

### 6.2 操作步骤

1. `Start Episode`
2. 选一个目标房屋
3. 让 UAV 多次从近似视角观察同一扇被障碍挡住的门
4. 多次执行：
   - Capture
   - 或 sequence 小移动后再次观察

### 6.3 期望现象

在 memory 中：

- `candidate_entries[*].status = blocked_temporary` 或 `blocked_confirmed`
- `attempt_count` 增长
- `last_best_entry_id` 稳定

在 fusion 结果中，后续应逐渐看到：

- `memory_history_boost`
- `memory_tracking_boost`
- 或高层：
  - `memory_decision_guidance.override_reason = persistent_blocked_shift`

### 6.4 判定标准

通过标准：

- 同一扇门不是每次都被当全新候选
- 系统会先持续跟踪，再在必要时换 sector

---

## 7. 非目标门历史抑制测试

### 7.1 测试目标

验证系统是否会记住：

- 某个门不是目标房屋入口
- 下一次看到相似门时会自动降权

### 7.2 操作步骤

1. 目标设为 `house_1`
2. 让画面里多出现 `house_2` 或其它房屋的门
3. 连续几次观察相似的非目标门

### 7.3 期望现象

在 memory 中：

- 对应候选会变成：
  - `non_target`
  - 或 `window_rejected`

在 fusion 中：

- `memory_history_penalty > 0`
- `candidate_total_score < candidate_total_score_raw`

### 7.4 判定标准

通过标准：

- 非目标门不会每次都重新占据高优先级

---

## 8. Sequence 时序测试

### 8.1 测试目标

验证：

- sequence 不再只是“发一串 move”
- 而是会留下清晰的 memory 边界

### 8.2 操作步骤

1. `Start Episode`
2. 在 `Sequence Control` 中输入一段：
   - 例如 `wwqdd`
3. 执行 sequence

### 8.3 期望现象

系统应生成 sequence 边界 snapshot：

- `sequence_start`
- `sequence_end`

如果中途停止或失败，应出现：

- `sequence_stop`
或
- `sequence_failed`

### 8.4 判定标准

通过标准：

- memory timeline 里能看出 sequence 的起止

---

## 9. Phase1 Scan 时序测试

### 9.1 测试目标

验证：

- `Phase1 Scan x12` 是否已经真正接入 episode-relative step
- 是否保存 scan 级 start/end memory snapshot

### 9.2 操作步骤

1. `Start Episode`
2. 点击：
   - `Phase1 Scan x12`
3. 扫描完成后查看 scan 目录

### 9.3 期望现象

在 `phase1_scan_manifest.json` 中应看到：

- `memory_episode_id`
- `memory_step_index_start`
- `memory_step_index_end`
- `memory_snapshot_start_path`
- `memory_snapshot_end_path`

每个 step bundle 中应带有：

- `memory_episode_id`
- `memory_step_index`

### 9.4 判定标准

通过标准：

- scan 内每一步不再只是局部 `0..11` 视角样本
- 而是能映射回 episode 内整体时序

---

## 10. 测试通过后，你应该看到的整体效果

如果 memory 真的工作了，你会逐渐看到三类变化：

1. `step` 在累积
   - 系统不再把每条样本都当孤立帧

2. `sector` 在累积
   - 哪些区域已经低收益，系统会逐渐记住

3. `candidate` 有历史
   - 同一个门会被持续跟踪
   - 坏候选会被持续抑制

一句话说：

系统会从：

- “每一帧都重新判断”

逐步变成：

- “我记得这栋房、这个门、这个角度之前发生过什么”

---

## 11. 推荐测试顺序

如果只做一轮最小验证，推荐顺序：

1. 基础采集链测试
2. Phase1 Scan 时序测试
3. Sequence 时序测试
4. 低收益 sector 测试
5. blocked 门重复测试
6. 非目标门历史抑制测试

---

## 12. 一句话总结

当前 memory 测试不是去看一个抽象 embedding，而是去看这三类具体现象：

1. 重复区域会不会被识别成低收益
2. 同一扇门会不会被持续跟踪
3. 错误或无效入口会不会被记住并降权

如果这三类现象都出现了，就说明：

- memory 不是“只存文件”
- 而是已经开始真实参与你的入口搜索决策
