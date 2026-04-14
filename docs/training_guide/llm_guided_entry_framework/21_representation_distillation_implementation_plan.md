# Representation Distillation Implementation Plan

## 1. 文档目标

本文件回答一个很实际的问题：

- `representation distillation trainer` 到底应该先写哪些文件
- 每个模块之间怎么连
- 每一步做到什么算通过

它是 [20_representation_distillation_trainer_spec.md](20_representation_distillation_trainer_spec.md) 的工程落地版。

一句话说：

- `20` 负责定义“训练器应该长什么样”
- `21` 负责定义“我们应该怎么把它写出来”

---

## 2. 总体实现顺序

建议严格按下面顺序推进：

1. `dataset reader`
2. `feature vector builder`
3. `label encoder`
4. `student model`
5. `loss builder`
6. `trainer loop`
7. `evaluator`

不要反过来先写 trainer 主循环。

原因很简单：

- 如果样本读取没稳定，后面所有训练都会乱
- 如果标签编码没固定，模型 head 会反复改
- 如果 evaluator 没先想清楚，训练完也不知道结果对不对

---

## 3. 推荐文件拆分

第一版建议放在新目录：

- `E:\github\UAV-Flow\phase2_5_representation_distillation`

目录建议如下：

```text
phase2_5_representation_distillation/
├─ README.md
├─ train_representation_distillation.py
├─ evaluate_representation_distillation.py
├─ dataset.py
├─ feature_builder.py
├─ label_schema.py
├─ model.py
├─ losses.py
├─ trainer.py
├─ evaluator.py
├─ utils_io.py
└─ configs/
   └─ base_config.json
```

---

## 4. Step 1: `dataset.py`

## 4.1 作用

负责读取：

- `train.jsonl`
- `val.jsonl`

并返回统一样本对象。

## 4.2 输入

来自：

- [train.jsonl](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/exports/phase2_5_distillation_dataset_20260414_163838/train.jsonl)
- [val.jsonl](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/exports/phase2_5_distillation_dataset_20260414_163838/val.jsonl)

## 4.3 输出

每条样本读取后，至少保留：

- `sample_id`
- `entry_state_path`
- `teacher_output_path`
- `teacher_validation_path`
- `metadata_path`
- 普通 teacher 标签
- target-conditioned 标签

## 4.4 验收标准

- 能稳定读取 train/val
- 遇到缺文件能给出清晰报错
- 样本总数和 manifest 对得上

---

## 5. Step 2: `feature_builder.py`

## 5.1 作用

把 `entry_state.json` 转成真正能喂给网络的张量化结构。

## 5.2 建议输出

分三块：

- `global_tensor`
- `candidate_tensor`
- `mask_tensor`

### `global_tensor`

来源：

- `entry_state.global_state`

### `candidate_tensor`

来源：

- `entry_state.candidates`

形状建议：

- `[K, D_candidate]`

其中：

- `K = 3`

### `mask_tensor`

用于表示：

- 某个 candidate 是否有效
- 某个 target-conditioned 标签是否存在

## 5.3 关键点

- 所有离散字段先在 `label_schema.py` 里统一编码
- 所有归一化规则固定，不要散落在训练代码里

## 5.4 验收标准

- 单样本可以成功构造成张量
- 所有样本维度一致
- 缺失 candidate 能自动补零

---

## 6. Step 3: `label_schema.py`

## 6.1 作用

统一定义所有类别标签和编码映射。

## 6.2 必须固定的 label space

至少包括：

- `entry_state`
- `subgoal`
- `action_hint`
- `target_conditioned_state`
- `target_conditioned_subgoal`
- `target_conditioned_action_hint`
- `target_conditioned_target_candidate_id`

## 6.3 推荐做法

提供这些函数：

- `encode_entry_state(...)`
- `encode_subgoal(...)`
- `encode_action_hint(...)`
- `encode_target_state(...)`
- `encode_target_subgoal(...)`
- `encode_target_action(...)`
- `encode_target_candidate_id(...)`

同时保留：

- `id_to_label`
- `label_to_id`

## 6.4 验收标准

- 同一种标签在所有模块里编码一致
- 不允许 trainer 内部自己写硬编码映射

---

## 7. Step 4: `model.py`

## 7.1 作用

定义 student 网络。

## 7.2 第一版建议结构

- `GlobalEncoder`
- `CandidateEncoder`
- `FusionMLP`
- 多个分类 head

## 7.3 建议类拆分

- `GlobalEncoder`
- `CandidateEncoder`
- `EntryRepresentationStudent`

其中主模型输出：

- `z_entry`
- `entry_state_logits`
- `subgoal_logits`
- `action_hint_logits`
- `target_state_logits`
- `target_subgoal_logits`
- `target_action_logits`
- `target_candidate_logits`

## 7.4 验收标准

- 前向传播不报错
- 输入 batch 后所有 logits 维度正确
- 能在 CPU 上先跑一个 fake batch

---

## 8. Step 5: `losses.py`

## 8.1 作用

集中管理所有 loss，避免 trainer 中到处散写。

## 8.2 第一版内容

建议实现：

- `build_class_weights(...)`
- `compute_multi_head_loss(...)`

## 8.3 输入

- 模型输出 logits
- 编码后的标签
- mask

## 8.4 输出

- `total_loss`
- `loss_dict`

例如：

- `loss_entry_state`
- `loss_subgoal`
- `loss_action_hint`
- `loss_target_state`
- `loss_target_subgoal`
- `loss_target_action`
- `loss_target_candidate`

## 8.5 验收标准

- 单 batch 能稳定算 loss
- 缺失 target-conditioned 标签时不会炸
- loss_dict 里的每一项都能打印和记录

---

## 9. Step 6: `trainer.py`

## 9.1 作用

实现标准训练循环。

## 9.2 建议职责

- 构造 optimizer
- 构造 dataloader
- 跑 train epoch
- 跑 val epoch
- 保存 best / last checkpoint
- 记录 metrics

## 9.3 第一版不要做太复杂

先不要加：

- 分布式训练
- AMP 混合精度
- EMA
- scheduler 大量花样

第一版只需要：

- `AdamW`
- `early stopping`
- `best checkpoint`

## 9.4 验收标准

- 能稳定训练 1~2 epoch
- metrics 会下降
- checkpoint 会保存

---

## 10. Step 7: `evaluator.py`

## 10.1 作用

把验证集指标统一算出来。

## 10.2 推荐输出

- overall accuracy
- per-class precision / recall / F1
- confusion matrix

重点看：

- `target_conditioned_state`
- `target_conditioned_subgoal`
- `target_conditioned_action_hint`

## 10.3 重点类

必须单独盯：

- `target_house_entry_approachable`
- `target_house_entry_blocked`
- `target_house_entry_visible`
- `non_target_house_entry_visible`
- `target_house_not_in_view`

## 10.4 验收标准

- 每个 head 都有清晰指标
- confusion matrix 能输出到文件

---

## 11. 推荐开发顺序

建议真正动手时按下面顺序提交：

### Milestone 1

- `label_schema.py`
- `dataset.py`

通过标准：

- 能把 train/val 正确读出来

### Milestone 2

- `feature_builder.py`

通过标准：

- 能把一条样本转成固定维度输入

### Milestone 3

- `model.py`
- `losses.py`

通过标准：

- fake batch 跑通 forward + loss

### Milestone 4

- `trainer.py`
- `train_representation_distillation.py`

通过标准：

- 真数据能训练 1 个 epoch

### Milestone 5

- `evaluator.py`
- `evaluate_representation_distillation.py`

通过标准：

- 能输出验证指标和 confusion matrix

---

## 12. 第一版最小实验

第一版实验建议只做：

### 任务

- `entry_state`
- `target_conditioned_state`

### 再加一层

等状态头稳定后，再加：

- `target_conditioned_subgoal`

### 最后再加

- `action_hint`
- `target_conditioned_action_hint`
- `target_conditioned_target_candidate_id`

也就是说：

- **先把状态学稳**
- **再加决策**
- **最后加动作**

---

## 13. 风险点

当前最容易出问题的是：

1. 类别不平衡  
2. target-conditioned 标签偏少  
3. `target_candidate_id` 噪声较大  
4. `target_house_not_in_view` 数量明显更多

所以第一版一定要：

- 报 per-class F1
- 报 class-weight 配置
- 不要只看 overall accuracy

---

## 14. 完成标准

我建议 trainer 模块完成的标志是：

1. 能读最新 export 数据
2. 能训练并保存 best checkpoint
3. 能输出 target-conditioned 关键指标
4. 至少在验证集上：
   - `target_conditioned_state` 有可解释结果
   - confusion matrix 看起来不是完全塌缩到单一类

到这一步，我们就可以继续进入：

- 更完整的表示蒸馏实验
- BC 初始化
- PPO 微调
