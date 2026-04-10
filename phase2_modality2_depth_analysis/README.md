# Phase 2 Modality 2 Depth Analysis

This folder contains a standalone Phase 2 "modality 2" tool for analyzing a
single depth image and estimating:

- front obstacle status
- entry distance
- opening width
- traversability

It is designed to match the Phase 2 "depth analysis" stage described in:

- [full_training_pipeline.md](/E:/github/UAV-Flow/docs/training_guide/full_training_pipeline.md)

## Files

- [run_phase2_depth_entry_analysis.py](/E:/github/UAV-Flow/phase2_modality2_depth_analysis/run_phase2_depth_entry_analysis.py)

## Input

The script supports:

- `*.png` depth images in centimeters
- `*.npy` depth arrays

Recommended input:

- `*_depth_cm.png`

## Output

Each run creates a timestamped folder under:

- [outputs](/E:/github/UAV-Flow/phase2_modality2_depth_analysis/outputs)

The folder contains:

- `analysis_overlay.png`
- `analysis_result.json`
- `analysis_summary.txt`

## Example

```powershell
python E:\github\UAV-Flow\phase2_modality2_depth_analysis\run_phase2_depth_entry_analysis.py `
  --depth_path E:\github\UAV-Flow\captures_remote\capture_20260403_133952_house_1_depth_cm.png
```

## Key Result Fields

- `front_obstacle.present`
- `front_obstacle.front_min_depth_cm`
- `entry_assessment.candidate_count`
- `entry_assessment.best_candidate.entry_distance_cm`
- `entry_assessment.best_candidate.opening_width_cm`
- `entry_assessment.best_candidate.traversable`

## Defaults

- horizontal FOV: `90 deg`
- max valid depth: `1200 cm`
- front obstacle threshold: `140 cm`
- minimum traversable opening width: `90 cm`
- minimum clearance depth: `160 cm`
