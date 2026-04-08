# Step C Labeling Standard

## 1. Purpose

This document defines the minimum labeling package and the recommended label taxonomy
for Step C (`LLM-guided teacher`) in the entry framework.

The goal is to make each Phase 2 fusion run directly reusable as:

- a human acceptance sample
- a prompt-debug sample
- a future teacher-distillation sample

---

## 2. Required Files Per Sample

Each sample should be stored as one fusion run directory and must include a `labeling/`
subfolder containing at least:

- `rgb.png`
- `depth_cm.png`
- `depth_preview.png`
- `yolo_annotated.jpg`
- `depth_overlay.png`
- `fusion_overlay.png`
- `camera_info.json`
- `state.json`
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

These files serve different roles:

- `rgb.png`: human semantic inspection
- `depth_cm.png`: raw geometry source for later re-analysis
- `depth_preview.png`: human-readable depth view
- `yolo_annotated.jpg`: semantic detection evidence
- `depth_overlay.png`: geometric opening / obstacle evidence
- `fusion_overlay.png`: combined decision visualization
- `state.json` and `pose_history_summary.*`: current pose, task context, and history
- `*_result.json`: machine-readable outputs
- `*_summary.txt`: quick human-readable summaries

---

## 3. Minimum Review Images

For manual review, the annotator should always inspect these 5 images first:

1. `rgb.png`
2. `depth_preview.png`
3. `yolo_annotated.jpg`
4. `depth_overlay.png`
5. `fusion_overlay.png`

Recommended use:

- `rgb.png`: determine whether the semantic target is a door, window, wall opening, or clutter
- `depth_preview.png`: understand coarse front geometry
- `yolo_annotated.jpg`: check whether semantic detections are correct
- `depth_overlay.png`: check whether the proposed opening is really traversable
- `fusion_overlay.png`: verify whether the final decision matches the evidence

---

## 4. Primary Label Types

The primary label is `gt_entry_state`.

Recommended label set:

- `enterable_open_door`
- `enterable_door`
- `visible_but_blocked_entry`
- `front_blocked_detour`
- `window_visible_keep_search`
- `geometric_opening_needs_confirmation`
- `no_entry_confirmed`

Definitions:

- `enterable_open_door`
  - A clearly open door is visible and the opening is traversable.
- `enterable_door`
  - A door-like entry is visible and traversable, but not necessarily explicitly open.
- `visible_but_blocked_entry`
  - A likely door/entry is visible, but current geometry does not support safe crossing.
- `front_blocked_detour`
  - A severe front obstacle is present; bypass behavior should take priority.
- `window_visible_keep_search`
  - A window or non-entry facade element is visible; do not enter.
- `geometric_opening_needs_confirmation`
  - Depth suggests a traversable opening, but semantic evidence is weak or ambiguous.
- `no_entry_confirmed`
  - No reliable entry candidate is currently confirmed.

---

## 5. Secondary Labels

The secondary label is `gt_subgoal`.

Recommended label set:

- `approach_entry`
- `cross_entry`
- `detour_left`
- `detour_right`
- `keep_search`
- `backoff_and_reobserve`

Interpretation:

- `approach_entry`: move closer to the candidate entry
- `cross_entry`: pass through the entry now
- `detour_left`: bypass obstacle by shifting/turning left
- `detour_right`: bypass obstacle by shifting/turning right
- `keep_search`: continue searching the facade for a valid entry
- `backoff_and_reobserve`: move back or stabilize, then inspect again

---

## 6. Tertiary Labels

The tertiary label is `gt_action_hint`.

Recommended label set:

- `forward`
- `yaw_left`
- `yaw_right`
- `left`
- `right`
- `hold`
- `backward`

This label is not the final low-level controller output. It is only the coarse action
hint used for teacher evaluation and later distillation.

---

## 7. Suggested Dataset Split For Human Benchmark

For the first small human-labeled benchmark, recommend:

- `80-120` samples total
- balanced as much as possible across the primary classes

Recommended first-round target:

- `enterable_open_door`: 15-20
- `enterable_door`: 10-15
- `visible_but_blocked_entry`: 10-15
- `front_blocked_detour`: 15-20
- `window_visible_keep_search`: 15-20
- `geometric_opening_needs_confirmation`: 10-15
- `no_entry_confirmed`: 10-15

If balancing all 7 classes is difficult, the first round may merge:

- `enterable_open_door` + `enterable_door`

and keep a 6-class benchmark.

---

## 8. Annotation Rules

Use these priority rules during human labeling:

1. Severe front obstacle overrides entry desirability
2. `window` should not be labeled as an enterable door
3. `cross_entry` should only be used when the UAV is already close enough and the path is clear
4. If semantics and geometry disagree, prefer the more conservative label
5. If the image is genuinely ambiguous, use `geometric_opening_needs_confirmation`

---

## 9. Annotation Workflow

For each sample:

1. Open `labeling_summary.txt`
2. Inspect the 5 review images
3. Check `state.json` or `pose_history_summary.json` if context is needed
4. Fill `annotation_template.json`
5. Save the completed label file next to the sample

Recommended annotation fields:

```json
{
  "gt_entry_state": "",
  "gt_subgoal": "",
  "gt_action_hint": "",
  "reviewer": "",
  "notes": ""
}
```

---

## 10. Why This Standard Matters

This labeling package provides:

- a clean benchmark for comparing `Fusion-only`, `Pure LLM`, and `Fusion + LLM`
- a reproducible teacher-evaluation dataset
- an easier path to export teacher trajectories for BC / distillation

It should be treated as the standard input-output unit for Step C validation.
