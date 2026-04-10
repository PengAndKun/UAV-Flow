# Distillation Dataset Export Specification

## 1. 文档目标

这份文档定义 `distillation_dataset_export` 的职责：

- 从现有 `fusion` 结果包和 teacher 结果中导出训练样本
- 生成 BC / PPO / 蒸馏统一可用的数据集

目标是先把“导出规则”固定，再去写代码。

---

## 2. 数据来源

推荐从下面这些文件夹中读取：

- `phase2_multimodal_fusion_analysis/results/fusion_*/`

每个 run 下优先读：

- `labeling/`
- `fusion/fusion_result.json`
- `inputs/state.json`
- `inputs/camera_info.json`

如果 teacher 已经跑好，再额外读：

- `labeling/teacher_output.json`
- 或 `labeling/anthropic_*_scene_result.json` 经转换后的 teacher 输出

---

## 3. 导出目标

建议一份导出数据至少支持三种用途：

1. `BC initialization`
2. `teacher-student distillation`
3. `RL offline warm start`

所以导出格式要同时包含：

- 状态
- teacher 标签
- 动作标签
- 元数据

---

## 4. 样本单位

第一版建议以**单帧样本**为主，而不是长序列。

每个样本对应：

- 一个 `fusion_xxx/labeling/` 样本包
- 一个 `entry_state.json`
- 一个 `teacher_output.json`

如果后面要做顺序模型，再在第二版增加序列导出。

---

## 5. 导出结构

建议导出目录结构：

```text
phase2_5_distillation_dataset/
├─ manifest.json
├─ train.jsonl
├─ val.jsonl
├─ train_ids.txt
├─ val_ids.txt
└─ samples/
   ├─ sample_000001/
   │  ├─ entry_state.json
   │  ├─ teacher_output.json
   │  └─ metadata.json
   └─ ...
```

---

## 6. 单条样本字段

建议 JSONL 中每条记录包含：

```json
{
  "sample_id": "fusion_20260409_101400",
  "split": "train",
  "entry_state_path": "samples/sample_000001/entry_state.json",
  "teacher_output_path": "samples/sample_000001/teacher_output.json",
  "metadata_path": "samples/sample_000001/metadata.json",
  "teacher_available": 1,
  "target_candidate_id": 0,
  "entry_state": "enterable_open_door",
  "subgoal": "approach_entry",
  "action_hint": "forward"
}
```

---

## 7. 样本过滤规则

不是所有 fusion 样本都应直接进蒸馏集。

### 7.1 必须满足

- `fusion_result.json` 存在
- `entry_state.json` 构建成功
- 必要图片和状态文件存在

### 7.2 teacher 样本额外要求

如果要进入蒸馏训练集，建议要求：

- teacher schema 合法
- `validation.status != invalid`
- `confidence >= threshold`

建议初始阈值：

- `confidence >= 0.55`

### 7.3 可选过滤

可剔除：

- 同一位置高度重复的近重复样本
- 明显坏图
- 无有效候选且无 teacher 指导的样本

---

## 8. 训练/验证划分原则

### 8.1 不按帧随机拆

严禁：

- 同一局部场景的几乎重复帧同时进入 train 和 val

### 8.2 推荐按 run 或场景拆

建议：

- 按 `fusion_xxx` run 级别拆
- 更进一步可按 `house_id` 或场景区域拆

### 8.3 推荐比例

第一版建议：

- `train = 80%`
- `val = 20%`

---

## 9. 类别平衡建议

导出时建议统计各类数量：

- `enterable_open_door`
- `enterable_door`
- `visible_but_blocked_entry`
- `front_blocked_detour`
- `window_visible_keep_search`
- `geometric_opening_needs_confirmation`
- `no_entry_confirmed`

如果极不平衡，建议：

- 保留全部正类
- 对过多的易样本做下采样

尤其要注意：

- `window_visible_keep_search`
- `front_blocked_detour`

这两类对鲁棒性很重要。

---

## 10. 导出后的训练接口

导出后的每条样本应该能直接被训练代码读取成：

- `global_state_tensor`
- `candidate_tensor`
- `teacher_target_tensor`

也就是说，导出器应该尽量把：

- 文件组织
- 字段映射
- 标签编码

都提前处理好。

---

## 11. 推荐保存的附加统计

建议导出时自动写：

### `manifest.json`

包含：

- 总样本数
- train/val 数量
- 各 `entry_state` 数量
- 各 `subgoal` 数量
- teacher 可用率

### `quality_report.json`

包含：

- teacher invalid 数量
- 缺文件数量
- 重复样本数量
- 被过滤样本数量和原因

---

## 12. 第一版最小落地建议

第一版先做：

1. 单帧样本导出
2. JSONL + 样本目录
3. train/val 划分
4. teacher 合法性过滤

先不要一上来就做：

- 长序列导出
- 复杂 replay buffer
- 多阶段 curriculum packing

---

## 13. 和前两份文档的关系

- [06_agent_state_schema.md](06_agent_state_schema.md)
  - 定义每条样本的状态结构
- [07_teacher_schema_spec.md](07_teacher_schema_spec.md)
  - 定义 teacher 输出结构
- 本文档
  - 定义怎样把前两者批量导出成训练集

---

## 14. 一句话总结

`distillation_dataset_export` 的目标不是做训练，
而是把当前零散结果包整理成：

- 结构固定
- 标签稳定
- 可直接喂给 BC / PPO / 蒸馏训练的数据集
