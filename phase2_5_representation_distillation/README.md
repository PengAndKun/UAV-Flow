# Phase 2.5 Representation Distillation

This directory contains the first implementation blocks for the Phase 2.5
representation distillation pipeline.

Recommended config entry:

- `configs/base_config.json`

Current modules:

- `label_schema.py`
  - fixed label spaces and encoding/decoding helpers
- `dataset.py`
  - export-dataset reader for `train.jsonl` / `val.jsonl`
- `feature_builder.py`
  - converts `entry_state.json` samples into fixed-length numeric training features
- `model.py`
  - multi-branch student network for distilled entry representation learning
- `losses.py`
  - multi-head classification losses for ordinary and target-conditioned supervision
- `trainer.py`
  - dataloader, class-weight building, epoch loops, checkpoint saving
- `train_representation_distillation.py`
  - command-line entry for first-pass student training
- `evaluator.py`
  - validation metrics, per-class reports, and confusion matrix export
- `evaluate_representation_distillation.py`
  - command-line entry for checkpoint evaluation

The implementation order follows:

1. label schema
2. dataset reader
3. feature builder
4. student model
5. loss builder
6. trainer
7. evaluator

Basic training usage:

```powershell
python E:\github\UAV-Flow\phase2_5_representation_distillation\train_representation_distillation.py
```

Use another config file:

```powershell
python E:\github\UAV-Flow\phase2_5_representation_distillation\train_representation_distillation.py `
  --config_path E:\github\UAV-Flow\phase2_5_representation_distillation\configs\base_config.json
```

Command-line arguments still override the config file.

Trainer updates:

- staged curriculum is enabled by default
  - `stage1_epochs`: train only `entry_state` + `target_conditioned_state`
  - `stage2_epochs`: add `subgoal` + `target_conditioned_subgoal`
  - remaining epochs: train all heads
- `best.pt` is selected by a target-priority validation metric instead of raw total loss

Example with explicit curriculum overrides:

```powershell
python E:\github\UAV-Flow\phase2_5_representation_distillation\train_representation_distillation.py `
  --config_path E:\github\UAV-Flow\phase2_5_representation_distillation\configs\base_config.json `
  --stage1_epochs 5 `
  --stage2_epochs 12 `
  --epochs 20 `
  --run_name pilot_distill_v2
```
