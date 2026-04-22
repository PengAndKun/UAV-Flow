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
- if `target_house_review.json` exists, it now prefers the reviewed house id and suppresses stale target-conditioned labels when the reviewed house changes the original target assignment
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
- `memory_context`
- `teacher_targets`
- `metadata`

The current builder now preserves target-conditioned signals as well:

- `global_state.target_house_id / target_house_in_fov / target_house_expected_side`
- `candidates[*].candidate_target_match_score / candidate_total_score / candidate_is_target_house_entry`
- `memory_context.memory_features`
  - `observed_sector_count`
  - `entry_search_status`
  - `current_sector_observation_count`
  - `current_sector_low_yield_flag`
  - `last_best_entry_status / last_best_entry_attempt_count`
  - `previous_action / previous_subgoal`
- `teacher_targets.target_conditioned_state / target_conditioned_subgoal / target_conditioned_action_hint`
- when `target_house_review.json` changed or filled the target house id, the builder now prefers the reviewed target id and clears stale target-conditioned candidate-match features
- it also tries to read:
  - `entry_search_memory_snapshot_before.json`
  - `entry_search_memory_snapshot_after.json`
  - `entry_search_memory_snapshot.json`
  and falls back to the memory summary embedded in `fusion_result.json`

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
- plus first-pass memory-aware fields
  - `memory_available`
  - `memory_source`
  - `memory_snapshot_before_path`
  - `memory_snapshot_after_path`
  - `memory_features`

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

Entry search memory store:

- `entry_search_memory.py`

This module provides the first-pass runtime storage for target-house entry
search memory. It currently focuses on the infrastructure layer:

- create / load / save `entry_search_memory.json`
- initialize one memory object per house from `UAV-Flow-Eval/houses_config.json`
- maintain:
  - `working_memory`
  - `episodic_memory`
  - `semantic_memory`
- update:
  - current target house id
  - recent actions
  - recent target-conditioned decisions
  - top candidates
  - searched sectors
  - candidate entry states
  - episodic snapshots

Default storage path:

- `phase2_multimodal_fusion_analysis/entry_search_memory.json`

Current integration status:

- `fusion_entry_analysis.py` now updates the first-pass `semantic_memory`
  after each fusion run
- the written fusion result now includes:
  - `fusion.entry_search_memory.memory_path`
  - `fusion.entry_search_memory.house_id`
  - `fusion.entry_search_memory.sector_id`
  - `fusion.entry_search_memory.entry_search_status`
- `fusion_entry_analysis.py` also starts using `semantic_memory` for first-pass
  decision adjustment:
  - repeated low-yield sectors can slightly reduce weak candidate priority
  - previously rejected candidates can be down-weighted
  - the previous `last_best_entry_id` can receive a small tracking boost
- the written fusion result now also includes `fusion.memory_guidance`
- the written fusion result now also includes `fusion.memory_decision_guidance`
  so higher-level target-conditioned decisions can:
  - shift away from repeated low-yield sectors
  - stop repeatedly retrying a persistently blocked target entry candidate
- `panel_summary` and `labeling_summary.txt` also expose a compact memory
  summary so it is easier to verify whether the runtime memory is changing

Default usage sketch:

```python
from phase2_multimodal_fusion_analysis import EntrySearchMemoryStore

store = EntrySearchMemoryStore()
store.load()
store.ensure_from_houses_config()
store.set_current_target_house("001")
store.set_entry_search_status("001", "searching_entry")
store.update_sector("001", "front_center", best_entry_state="blocked_entry")
store.save()
```
