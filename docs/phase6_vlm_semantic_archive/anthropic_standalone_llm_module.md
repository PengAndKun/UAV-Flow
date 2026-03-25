# Anthropic Standalone LLM/VLM Module

## Goal

Validate Anthropic-compatible multimodal scene interpretation without routing
through:

- `uav_control_panel.py`
- `uav_control_server.py`
- planner/action orchestration

The standalone script is:

- [anthropic_vlm_scene_descriptor.py](/E:/github/UAV-Flow/UAV-Flow-Eval/anthropic_vlm_scene_descriptor.py)

It uses:

- `import anthropic`
- `ANTHROPIC_BASE_URL`
- `ANTHROPIC_AUTH_TOKEN`

## Supported Models

Current intended models:

- `qwen3-coder-next`
- `claude-sonnet-4-6`

## What It Does

The script:

1. loads current RGB image
2. loads aligned depth image
3. optionally loads target-house reference image
4. builds a composite image panel
5. sends the image to an Anthropic-compatible backend
6. requests one structured JSON object
7. saves parsed output and the composite image
8. saves the exact system/user prompt to a prompt-log folder

## Output Fields

The model is asked to return:

- `scene_state`
- `active_stage`
- `target_house_visible`
- `entry_door_visible`
- `entry_door_traversable`
- `scene_description`
- `likely_unexplored_regions`
- `next_waypoints`
- `confidence`

## Example: qwen3-coder-next

```powershell
$env:ANTHROPIC_BASE_URL="http://YOUR_BASE_URL"
$env:ANTHROPIC_AUTH_TOKEN="YOUR_TOKEN"
python E:\github\UAV-Flow\UAV-Flow-Eval\anthropic_vlm_scene_descriptor.py `
  --model qwen3-coder-next `
  --task_label "search the house for people" `
  --rgb_path E:\github\UAV-Flow\captures_remote\capture_rgb.png `
  --depth_path E:\github\UAV-Flow\captures_remote\capture_depth.png `
  --output_json E:\github\UAV-Flow\captures_remote\anthropic_qwen_scene_result.json
```

## Example: claude-sonnet-4-6

```powershell
$env:ANTHROPIC_BASE_URL="http://YOUR_BASE_URL"
$env:ANTHROPIC_AUTH_TOKEN="YOUR_TOKEN"
python E:\github\UAV-Flow\UAV-Flow-Eval\anthropic_vlm_scene_descriptor.py `
  --model claude-sonnet-4-6 `
  --task_label "search the house for people" `
  --rgb_path E:\github\UAV-Flow\captures_remote\capture_rgb.png `
  --depth_path E:\github\UAV-Flow\captures_remote\capture_depth.png `
  --output_json E:\github\UAV-Flow\captures_remote\anthropic_claude_scene_result.json
```

## Optional Target-House Reference

To let the model reason about whether the visible doorway belongs to the target
house, add:

```powershell
  --reference_path E:\github\UAV-Flow\captures_remote\target_house_reference.png
```

## Why This Module Matters

This module isolates:

- Anthropic API connectivity
- multimodal scene understanding quality
- RGB+depth reasoning quality
- structured JSON reliability

before the logic is mixed back into the controller stack.

## Prompt Logging

Each run now also saves the exact prompt into:

- `E:\github\UAV-Flow\phase6_prompt_logs`

The output JSON will include:

- `prompt_log_path`

You can override the folder with:

```powershell
  --prompt_log_dir E:\github\UAV-Flow\phase6_prompt_logs
```
