# Phase 6: VLM Semantic Archive Search Plan

## Goal

This phase reshapes the project from:

- RGB/depth -> heuristic doorway/runtime summaries -> LLM planner

into:

- RGB/depth -> VLM scene description -> sentence embedding semantic archive -> task-conditioned strategic planner -> waypoint/action execution

The core change is that the archive is no longer only geometric/state-bin memory. It becomes a human-readable semantic memory that the high-level planner can consume directly.

## New Core Ideas

### 1. VLM-generated language description as the intermediate representation

For each key frame or state, run a VLM to produce a short scene description, for example:

- `front porch outside target house; open front door visible; dark interior behind doorway`
- `living room with sofa occlusion; right hallway still unexplored`
- `bedroom doorway visible; bed-side corner partially occluded`

Then:

1. Store the raw description text in the archive.
2. Encode the description with a sentence embedding model.
3. Use that embedding for archive retrieval and similarity search.

Why this helps:

- archive entries become human-readable
- LLM strategic planning can read archive descriptions directly
- less need for ad hoc cross-modal bridges between low-level numeric states and high-level language planning

### 2. Simulation-first data generation

Training data should come primarily from the simulator.

Environment setup:

- place house assets in outdoor neighborhoods
- optionally procedurally generate multiple adjacent houses
- start the UAV from random outdoor viewpoints and distances
- vary:
  - lighting
  - textures
  - time of day
  - furniture layouts
  - doorway states
  - clutter and occlusion

This supports large-scale automatic data generation without relying on real flight data.

### 3. Task-conditioned house entry search

The mission `search the house for people` must first be decomposed into entry-finding stages.

Recommended stage sequence:

1. determine whether the UAV is outside, at the threshold, or already inside
2. if outside, search for an entry door of the target house
3. approach the door
4. cross the threshold
5. switch to indoor structured person search
6. verify suspects and report final result

### 4. Target-house confirmation by image retrieval

The system should not treat all visible doors as equivalent.

At mission start, provide a reference image for the target house exterior.

During entry search:

- compute visual similarity between current view and target-house reference
- only reward or prioritize entry candidates that belong to the target house

Candidate backbone options:

- NetVLAD
- DINOv2
- CLIP-style global image retrieval

This can be presented as:

- task-conditioned entry recognition
- different from generic doorway detection

### 5. Distillation from archive to reactive policy

The semantic archive stores successful trajectories:

- observations
- scene descriptions
- actions
- stage labels
- rewards/outcomes

High-score trajectories become supervision for a lightweight reactive policy.

Training path:

1. behavior cloning initialization
2. optional small-scale online RL fine-tuning in simulation

Reactive policy input:

- RGB
- depth/radar
- optional local semantic hint
- current stage label

Reactive policy output:

- discrete actions such as
  - forward
  - left
  - right
  - yaw_left
  - yaw_right
  - up
  - hold

## Why This Phase Is Needed

The current system already has:

- LLM high-level planning
- scene-waypoint prompting
- language memory summaries
- doorway heuristics

But it still lacks:

- reliable semantic scene understanding directly grounded in RGB+depth
- a semantic archive retrieval layer based on textual state descriptions
- target-house-conditioned entry search
- a clean distillation data path from successful semantic search episodes to a reactive controller

This phase addresses all four.

## Planned Deliverables

1. VLM scene description module
2. sentence-embedding semantic archive
3. target-house visual retrieval module
4. phase-aware mission controller
5. archive export pipeline for behavior cloning
6. comprehensive simulation evaluation plan

## Recommended Document Order

- [control_refactor_plan.md](/E:/github/UAV-Flow/docs/phase6_vlm_semantic_archive/control_refactor_plan.md)
- [experiment_plan.md](/E:/github/UAV-Flow/docs/phase6_vlm_semantic_archive/experiment_plan.md)

