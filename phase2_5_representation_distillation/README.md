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
  - current version also folds `memory_context.memory_features` into the global feature branch
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
- `export_representation_embeddings.py`
  - exports the distilled `z_entry` vectors, predictions, probabilities, and sample metadata
- `analyze_representation_embeddings.py`
  - analyzes exported embeddings with centroid distances, nearest neighbors, no-entry separation, and episode consistency checks
- `evaluate_representation_ablation.py`
  - evaluates the same checkpoint with selected input features ablated, e.g. zeroed memory features

The implementation order follows:

1. label schema
2. dataset reader
3. feature builder
4. student model
5. loss builder
6. trainer
7. evaluator
8. embedding exporter
9. embedding analyzer
10. ablation evaluator

Basic training usage:

```powershell
python E:\github\UAV-Flow\phase2_5_representation_distillation\train_representation_distillation.py
```

Memory-aware training note:

- exported samples from
  [phase2_5_distillation_dataset_20260421_223847](/E:/github/UAV-Flow/phase2_multimodal_fusion_analysis/exports/phase2_5_distillation_dataset_20260421_223847)
  already include `memory_features`
- `feature_builder.py` appends these memory-aware features to `global_features`
- the student model input dimension is inferred automatically, so no manual model-dimension edit is required

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

Export memory-aware representations:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python E:\github\UAV-Flow\phase2_5_representation_distillation\export_representation_embeddings.py `
  --checkpoint_path E:\github\UAV-Flow\phase2_5_representation_distillation\runs\memory_aware_v5_pilot_20260427\checkpoints\best.pt `
  --export_dir E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\phase2_5_memory_aware_dataset_v3_20260427_20260427_141237 `
  --run_name memory_aware_v5_pilot_20260427 `
  --split all `
  --device cpu `
  --batch_size 32
```

Analyze exported representations:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
$env:MPLBACKEND='Agg'
python E:\github\UAV-Flow\phase2_5_representation_distillation\analyze_representation_embeddings.py `
  --embedding_dir E:\github\UAV-Flow\phase2_5_representation_distillation\embeddings\memory_aware_v5_pilot_20260427 `
  --nearest_k 5
```

Run memory ablation:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
python E:\github\UAV-Flow\phase2_5_representation_distillation\evaluate_representation_ablation.py `
  --checkpoint_path E:\github\UAV-Flow\phase2_5_representation_distillation\runs\memory_aware_v5_pilot_20260427\checkpoints\best.pt `
  --export_dir E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\phase2_5_memory_aware_dataset_v3_20260427_20260427_141237 `
  --run_name memory_aware_v5_pilot_20260427_val_memory_ablation `
  --split val `
  --ablations none zero_memory zero_candidates zero_memory_and_candidates `
  --device cpu `
  --batch_size 32 `
  --repeats 10
```
