# Standalone LLM/VLM Module

## Goal

Before routing requests through the full controller stack, validate that the
model can directly consume:

- current RGB image
- aligned depth visualization
- optional target-house reference image

and return a structured scene interpretation plus waypoint hints.

The standalone module is:

- [vlm_scene_descriptor.py](/E:/github/UAV-Flow/UAV-Flow-Eval/vlm_scene_descriptor.py)

## What It Does

The script:

1. loads an RGB image
2. loads a depth image
3. optionally loads a target-house reference image
4. creates a composite image panel
5. sends the composite image to the configured API/model
6. requests structured JSON output
7. prints the parsed result and saves the composite image
8. saves the exact system/user prompt to a prompt-log folder

This bypasses:

- `uav_control_panel.py`
- `uav_control_server.py`
- planner/action orchestration

so it is the cleanest way to verify whether the model really connects and
understands the scene.

## Expected Output

The module asks for:

- `scene_state`
- `active_stage`
- `target_house_visible`
- `entry_door_visible`
- `entry_door_traversable`
- `scene_description`
- `likely_unexplored_regions`
- `next_waypoints`
- `confidence`

## Example: Gemini Lite

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
python E:\github\UAV-Flow\UAV-Flow-Eval\vlm_scene_descriptor.py `
  --api_style google_genai_sdk `
  --base_url google-genai-sdk `
  --model gemini-3.1-flash-lite-preview `
  --api_key_env GEMINI_API_KEY `
  --task_label "search the house for people" `
  --rgb_path E:\github\UAV-Flow\captures_remote\capture_rgb.png `
  --depth_path E:\github\UAV-Flow\captures_remote\capture_depth.png `
  --output_json E:\github\UAV-Flow\captures_remote\vlm_scene_result.json
```

## Example: Gemini Flash

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
python E:\github\UAV-Flow\UAV-Flow-Eval\vlm_scene_descriptor.py `
  --api_style google_genai_sdk `
  --base_url google-genai-sdk `
  --model gemini-3-flash-preview `
  --api_key_env GEMINI_API_KEY `
  --task_label "search the house for people" `
  --rgb_path E:\github\UAV-Flow\captures_remote\capture_rgb.png `
  --depth_path E:\github\UAV-Flow\captures_remote\capture_depth.png `
  --output_json E:\github\UAV-Flow\captures_remote\vlm_scene_result_flash.json
```

## Optional Target-House Reference

If you want the model to reason about whether the visible doorway belongs to
the target house, add:

```powershell
  --reference_path E:\github\UAV-Flow\captures_remote\target_house_reference.png
```

This produces a three-panel composite:

- RGB
- depth
- target reference

## Why This Module Matters

This standalone module is the first clean building block for the new Phase 6
pipeline:

- `RGB + depth -> VLM language description -> semantic archive -> planner`

It also helps determine whether failures come from:

- model/API connectivity
- prompt/schema quality
- controller orchestration
- runtime timeouts

rather than mixing all of them together.

## Prompt Logging

Each run now also saves the exact prompt into:

- `E:\github\UAV-Flow\phase6_prompt_logs`

The output JSON will include:

- `prompt_log_path`

You can override the folder with:

```powershell
  --prompt_log_dir E:\github\UAV-Flow\phase6_prompt_logs
```
