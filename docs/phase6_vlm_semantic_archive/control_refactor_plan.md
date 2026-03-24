# Phase 6 Control Refactor Plan

## Main Principle

Do not continue stuffing all new logic directly into [uav_control_server.py](/E:/github/UAV-Flow/UAV-Flow-Eval/uav_control_server.py).

That file should remain the integration hub, but the new Phase 6 logic should be extracted into dedicated controller/runtime modules.

The recommended rule is:

- keep HTTP endpoints, top-level orchestration, and state packaging in `uav_control_server.py`
- move new mission logic, semantic memory, and retrieval logic into dedicated files

## Proposed New Files

### 1. `UAV-Flow-Eval/vlm_scene_descriptor.py`

Purpose:

- send RGB or RGB+depth-conditioned prompts to a VLM
- produce concise scene descriptions
- optionally produce structured tags:
  - `scene_state`
  - `entry_visible`
  - `entry_traversable`
  - `room_type`
  - `occlusion_summary`
  - `unexplored_direction`

Expected output:

```python
{
    "description": "outside porch view of target house; open front door visible; interior appears reachable",
    "tags": {...},
    "model_name": "...",
    "latency_ms": ...
}
```

### 2. `UAV-Flow-Eval/semantic_archive_runtime.py`

Purpose:

- maintain text-based archive entries
- compute/store sentence embeddings
- retrieve similar semantic states
- support task-conditioned archive lookup

Recommended archive entry schema:

```python
{
    "archive_id": "...",
    "task_label": "...",
    "stage_label": "...",
    "scene_description": "...",
    "scene_embedding": [...],
    "pose": {...},
    "doorway_summary": {...},
    "target_house_match": {...},
    "outcome": {...},
    "action_history": [...],
    "reward_summary": {...}
}
```

### 3. `UAV-Flow-Eval/reference_house_matcher.py`

Purpose:

- given a target-house reference image and current frame
- compute whether the UAV is viewing the target house
- score visible door/window candidates against the task target

Candidate backbones:

- NetVLAD
- DINOv2
- CLIP retrieval baseline

### 4. `UAV-Flow-Eval/phase6_mission_controller.py`

Purpose:

- maintain the explicit stage machine for:
  - outside localization
  - target house confirmation
  - entry search
  - door approach
  - threshold crossing
  - indoor person search
  - suspect verification

This should become the new mission brain for `search the house for people`.

Recommended top-level stages:

- `outside_localization`
- `target_house_verification`
- `entry_search`
- `approach_entry`
- `cross_entry`
- `indoor_room_search`
- `suspect_verification`
- `mission_report`

### 5. `UAV-Flow-Eval/phase6_waypoint_planner.py`

Purpose:

- convert phase controller state and semantic archive retrievals into continuous waypoint proposals
- output a short waypoint queue, not just one waypoint

Recommended output:

```python
{
    "stage": "approach_entry",
    "waypoints": [...],
    "reasoning": "...",
    "confidence": 0.0
}
```

### 6. `UAV-Flow-Eval/archive_distillation_export.py`

Purpose:

- export high-score archive trajectories
- build BC/RL training datasets
- attach stage labels and semantic archive context

## What Should Stay In `uav_control_server.py`

[uav_control_server.py](/E:/github/UAV-Flow/UAV-Flow-Eval/uav_control_server.py) should still handle:

- HTTP endpoints
- simulator interaction
- frame/depth capture
- global runtime state packaging
- control execution
- logging and capture bundling

But it should delegate new Phase 6 logic to:

- `vlm_scene_descriptor.py`
- `semantic_archive_runtime.py`
- `reference_house_matcher.py`
- `phase6_mission_controller.py`
- `phase6_waypoint_planner.py`

## High-Priority Changes Inside `uav_control_server.py`

### A. Add new runtime blocks

Need to add:

- `vlm_scene_runtime`
- `semantic_archive_runtime`
- `reference_match_runtime`
- `phase6_mission_runtime`
- `waypoint_queue_runtime`

These should be included in:

- `/state`
- capture bundle
- planner/action request payloads

### B. Add new request/update flow

Recommended online pipeline:

1. refresh RGB + depth
2. run VLM scene descriptor
3. run reference-house matcher
4. update phase6 mission controller
5. update semantic archive
6. generate waypoint queue
7. execute or export the resulting step

### C. Add new endpoints

Recommended server endpoints:

- `POST /request_vlm_scene`
- `POST /request_reference_match`
- `POST /request_phase6_waypoints`
- `POST /execute_waypoint_queue_segment`
- `GET /phase6_runtime`

## Control Philosophy Change

Current philosophy:

- planner returns one plan
- lower layer executes locally

Phase 6 philosophy:

- VLM interprets the scene
- mission controller decides the stage
- semantic archive supplies task-relevant memory
- waypoint planner returns a short sequence
- low-level controller executes the sequence

This is a stronger fit for:

- outside-to-inside house entry
- structured person search
- archive-driven distillation

## Suggested Code Order

1. add `phase6_mission_controller.py`
2. add `vlm_scene_descriptor.py`
3. add `reference_house_matcher.py`
4. add `semantic_archive_runtime.py`
5. add `phase6_waypoint_planner.py`
6. integrate all of them into [uav_control_server.py](/E:/github/UAV-Flow/UAV-Flow-Eval/uav_control_server.py)
7. add dataset export with `archive_distillation_export.py`

