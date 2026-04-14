# Phase 2 Multimodal Fusion Analysis

This folder stores the Phase 2 multimodal fusion workflow used by
`UAV-Flow-Eval/uav_control_panel_basic.py`.

The fusion pipeline combines:

- Phase 2 YOLO26 semantic detection from `phase2_door_or_window_yolo26_training`
- Phase 2 depth-only geometry analysis from `phase2_modality2_depth_analysis`

The panel fetches the current synchronized observation:

- `/frame` for RGB
- `/depth_raw` for raw depth centimeters
- `/camera_info`
- `/state`

Then it writes each run to:

- `phase2_multimodal_fusion_analysis/results/fusion_YYYYMMDD_HHMMSS[_label]/`

Each run contains:

- `inputs/`
- `yolo/`
- `depth/`
- `fusion/`
- `labeling/`

Main result files:

- `fusion/fusion_result.json`
- `fusion/fusion_summary.txt`
- `fusion/fusion_overlay.png`

The `labeling/` folder is the human-labeling package exported for Step C.
It contains:

- `rgb.png`
- `depth_cm.png`
- `depth_preview.png`
- `yolo_annotated.jpg`
- `depth_overlay.png`
- `fusion_overlay.png`
- `camera_info.json`
- `state.json`
- `state_excerpt.json`
- `pose_history_summary.json`
- `pose_history_summary.txt`
- `yolo_result.json`
- `yolo_summary.txt`
- `depth_result.json`
- `depth_summary.txt`
- `fusion_result.json`
- `fusion_summary.txt`
- `labeling_manifest.json`
- `annotation_template.json`
- `labeling_summary.txt`

Recommended labeling workflow:

1. Review `rgb.png` and `depth_preview.png`
2. Check `yolo_annotated.jpg` and `depth_overlay.png`
3. Read `labeling_summary.txt`
4. Fill `annotation_template.json`

Batch `Pure LLM baseline` script:

- `run_batch_pure_llm_baseline.py`

This script iterates through `results/fusion_*/labeling/`, runs the standalone
Anthropic descriptor on:

- `rgb.png`
- `depth_preview.png`

and writes, inside each `labeling/` folder:

- `anthropic_<model>_scene_result.json`
- `anthropic_<model>_vs_labeling_compare.json`

It also writes a batch summary JSON under `results/`.

Default behavior for `run_batch_pure_llm_baseline.py`:

- scan all `results/fusion_*/labeling/`
- skip folders that already have both:
  - `anthropic_<model>_scene_result.json`
  - `anthropic_<model>_vs_labeling_compare.json`

If you want to revisit existing folders and call the API again, use:

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_batch_pure_llm_baseline.py --rerun_existing
```

`--force` remains supported as an alias of `--rerun_existing`.

Teacher normalization and validation:

- `teacher_validator.py`
- `run_teacher_validator.py`

This validator reads each `labeling/` package, picks the newest:

- `anthropic*_scene_result.json`

then:

- normalizes it into `teacher_output.json`
- validates it against `fusion_result.json`, `yolo_result.json`, `depth_result.json`
- writes `teacher_validation.json`
- preserves the original generic teacher fields:
  - `entry_state`
  - `subgoal`
  - `action_hint`
- and now also writes target-conditioned teacher targets:
  - `target_conditioned_state`
  - `target_conditioned_subgoal`
  - `target_conditioned_action_hint`
  - `target_conditioned_reason`

Batch usage example:

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_teacher_validator.py
```

Single run example:

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_teacher_validator.py --only_dir fusion_20260408_145511
```

Entry state builder:

- `entry_state_builder.py`
- `run_entry_state_builder.py`

This builder reads each `labeling/` package and writes:

- `entry_state.json`

The file contains:

- `global_state`
- `candidates` (fixed top-K, current implementation uses `K=3`)
- `teacher_targets`
- `metadata`

The current builder now preserves target-conditioned signals as well:

- `global_state.target_house_id / target_house_in_fov / target_house_expected_side`
- `candidates[*].candidate_target_match_score / candidate_total_score / candidate_is_target_house_entry`
- `teacher_targets.target_conditioned_state / target_conditioned_subgoal / target_conditioned_action_hint`

Batch usage example:

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_entry_state_builder.py
```

Single run example:

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_entry_state_builder.py --only_dir fusion_20260408_145511
```

Distillation dataset export:

- `distillation_dataset_export.py`
- `run_distillation_dataset_export.py`

This exporter reads each `labeling/` package that already has:

- `entry_state.json`
- `teacher_output.json`
- `teacher_validation.json`

and writes a train/val export package under:

- `phase2_multimodal_fusion_analysis/exports/`

Each export package contains:

- `manifest.json`
- `quality_report.json`
- `train.jsonl`
- `val.jsonl`
- `train_ids.txt`
- `val_ids.txt`
- `samples/`

The exported JSONL records and manifest now include both:

- the original teacher supervision
  - `entry_state`
  - `subgoal`
  - `action_hint`
- and the target-conditioned supervision
  - `target_conditioned_state`
  - `target_conditioned_subgoal`
  - `target_conditioned_action_hint`
  - `target_conditioned_target_candidate_id`

Batch usage example:

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\run_distillation_dataset_export.py
```

Refresh existing `results/` folders with a newly trained YOLO model:

- `refresh_results_with_new_yolo.py`

This script is for the case where you retrain the Phase 2 YOLO model and want to
replace the old semantic analysis inside existing:

- `phase2_multimodal_fusion_analysis/results/fusion_*/`

It will, by default:

1. Re-run YOLO + fusion on each run directory using the new weights
2. Overwrite old `yolo/`, `fusion/`, and `labeling/` semantic/fusion result files
3. Refresh existing `anthropic*_vs_labeling_compare.json` files using the new `labeling_summary.txt`
4. Refresh `teacher_output.json` / `teacher_validation.json`
5. Refresh `entry_state.json`

Batch usage example:

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\refresh_results_with_new_yolo.py
```

Reprocess only one run directory:

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\refresh_results_with_new_yolo.py --only_dir fusion_20260408_145511
```

Use a specific YOLO checkpoint:

```powershell
python E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\refresh_results_with_new_yolo.py `
  --weights E:\github\UAV-Flow\phase2_door_or_window_yolo26_training\models\phase2_entry_detector\your_run\weights\best.pt
```

Optional skip flags:

- `--skip_compare_refresh`
- `--skip_teacher_refresh`
- `--skip_entry_state_refresh`

The script writes a batch summary JSON under:

- `phase2_multimodal_fusion_analysis/results/refresh_results_with_new_yolo_summary_*.json`
