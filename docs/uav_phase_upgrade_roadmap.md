# UAV Phase Upgrade Roadmap

## Current Status

The current codebase now has a working Phase 0 baseline and the first
foundational pieces of Phase 1 and Phase 2:

- `UAV-Flow-Eval/uav_control_server.py`
  - UAV runtime backend
  - RGB preview and capture
  - point-cloud packet interface and point-cloud preview interface
  - structured planner state interface
- `UAV-Flow-Eval/uav_control_panel.py`
  - remote control panel
  - RGB preview window
  - point-cloud preview window
  - task label and multimodal capture
  - planner debug display and plan request button
- `UAV-Flow-Eval/runtime_interfaces.py`
  - shared payload schema helpers for point cloud / planner / capture / runtime debug

## Phase 0

Goal:

- Stable manual teleoperation baseline
- RGB observation preview
- action / pose / timestamp logging

Implemented baseline:

- `/state`
- `/frame`
- `/capture`
- `/move_relative`

## Phase 1

Goal:

- Add point-cloud capability without breaking the current control chain
- Support real point-cloud later, but keep replay/synthetic compatibility now
- Build aligned RGB + point cloud + pose + action capture bundles

Implemented in this repo:

- point-cloud input modes:
  - `none`
  - `depth` (single-view depth back-projected reconstruction)
  - `synthetic`
  - `replay`
  - `external`
- new server interfaces:
  - `GET /pointcloud`
  - `GET /pointcloud_frame`
  - `POST /task`
- multimodal capture bundle:
  - RGB image
  - point-cloud packet JSON
  - bundle metadata JSON
- sample replay files:
  - `UAV-Flow-Eval/examples/sample_pointcloud_replay.jsonl`
  - `UAV-Flow-Eval/examples/sample_radar_replay.jsonl` (legacy-compatible alias sample)

## Phase 2

Goal:

- Replace step-by-step action prediction with sparse high-level planning
- Keep `uav_control_server.py` as executor, not semantic planner
- Expose planner state in a structured way

Implemented in this repo:

- structured planner state schema
- planner state embedded into `/state`
- new planner interfaces:
  - `GET /plan`
  - `POST /plan`
  - `POST /request_plan`
- planner debug shown in panel:
  - semantic subgoal
  - sector id
  - confidence
  - first waypoint
- fallback planner behavior:
  - if no external planner is configured, server synthesizes a heuristic waypoint

## Phase 3

Next target:

- Add goal-conditioned multimodal archive
- Distill a lightweight reflex navigator
- start filling:
  - `archive_cell_id`
  - `local_policy_action`
  - `risk_score`

Current code already reserves runtime debug fields for this phase.

## Phase 4

Next target:

- Full hierarchical loop:
  - sparse planner
  - archive retriever
  - reflex navigator
  - safety head
  - replan manager

Current code already has the right runtime slots for:

- `current_waypoint`
- `local_policy_action`
- `risk_score`
- `shield_triggered`
- `archive_cell_id`

## Phase 5

Next target:

- Replace replay/synthetic point cloud with real hardware stream
- connect to sim-to-real deployment stack
- keep panel as monitor/debug tool

## Suggested Commands

### 1. Depth-reconstructed point cloud + local fallback planner

```bash
python UAV-Flow-Eval/uav_control_server.py ^
  --env_bin_win E:\github\UAV-Flow\UnrealEnv\UE4_ExampleScene_Win\UE4_ExampleScene\Binaries\Win64\UE4_ExampleScene.exe ^
  --viewport_mode free ^
  --preview_mode first_person ^
  --pointcloud_mode depth ^
  --default_task_label "search the target area"
```

```bash
python UAV-Flow-Eval/uav_control_panel.py --host 127.0.0.1 --port 5020
```

### 2. Replay point cloud

```bash
python UAV-Flow-Eval/uav_control_server.py ^
  --env_bin_win E:\github\UAV-Flow\UnrealEnv\UE4_ExampleScene_Win\UE4_ExampleScene\Binaries\Win64\UE4_ExampleScene.exe ^
  --viewport_mode free ^
  --preview_mode first_person ^
  --pointcloud_mode replay ^
  --pointcloud_replay_path E:\github\UAV-Flow\UAV-Flow-Eval\examples\sample_pointcloud_replay.jsonl
```

### 3. External planner hook

```bash
python UAV-Flow-Eval/uav_control_server.py ^
  --env_bin_win E:\github\UAV-Flow\UnrealEnv\UE4_ExampleScene_Win\UE4_ExampleScene\Binaries\Win64\UE4_ExampleScene.exe ^
  --pointcloud_mode depth ^
  --planner_url http://127.0.0.1:6001 ^
  --planner_endpoint /plan
```

If the planner is unavailable, the server falls back to a local heuristic plan so
the runtime loop remains usable for experiments.
