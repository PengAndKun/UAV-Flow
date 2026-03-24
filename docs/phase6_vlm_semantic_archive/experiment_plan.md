# Phase 6 Experiment Plan

## Research Goal

Evaluate whether a semantic-archive-based embodied search system can:

1. localize itself as outside/inside/threshold
2. identify the target house from reference imagery
3. find and enter the correct doorway
4. search indoors for people
5. distill successful behavior into a reactive policy

## Core Experimental Story

Phase 6 should support a stronger paper narrative:

- a VLM produces language descriptions as semantic state
- sentence embeddings index semantic archive entries
- task-conditioned image retrieval verifies the target house
- stage-aware waypoint planning guides outdoor-to-indoor entry
- successful trajectories are distilled into a lightweight reactive policy

## Experiment Groups

### Group A. Perception and scene understanding

Goal:

- verify outside/inside classification
- verify entry-door visibility
- verify traversability judgment
- verify target-house matching

Subtasks:

- `outside_house vs inside_house vs threshold_zone`
- `entry_door_visible`
- `entry_door_traversable`
- `target_house_match`

Metrics:

- scene-state accuracy
- door visibility precision/recall
- traversable-entry precision/recall
- target-house retrieval top-1 / top-k accuracy

### Group B. Entry-search policy

Goal:

- verify the UAV can move from random outdoor start positions to the correct entry

Task:

- given a target house reference image
- start outside at random position
- search for the correct door
- approach and cross the threshold

Metrics:

- entry success rate
- time-to-entry
- path length to entry
- steps to entry
- false-entry rate
- no-entry termination rate

### Group C. Full person-search mission

Goal:

- verify full mission performance after entering the house

Task:

- target house given
- UAV starts outside
- enter house
- search for person
- confirm presence/absence and report location

Metrics:

- full mission success rate
- person search success rate
- localization error
- time-to-first-detection
- time-to-entry
- collision rate
- takeover count
- token usage
- decision latency

### Group D. Distillation experiments

Goal:

- verify archive trajectories can supervise a lightweight reactive policy

Training variants:

- BC only
- BC + small RL fine-tuning
- RGB only vs RGB+depth
- with vs without semantic archive context

Metrics:

- action accuracy on held-out expert trajectories
- waypoint-following success
- entry-search success
- person-search success
- inference latency

## Reward Design For Simulation

### Stage 1. Outdoor exploration reward

Reward new coverage on the exterior of the target house.

Example terms:

- positive reward for newly observed facade area
- positive reward for reducing overlap with already observed viewpoints
- small penalty for revisiting the same exterior arc

### Stage 2. Entry recognition reward

Reward seeing entry-relevant structures:

- positive reward for detecting door/window/opening candidates
- larger reward if the opening belongs to the target house
- larger reward again if the opening is judged traversable

### Stage 3. Efficiency and safety

Penalize:

- each step
- collisions
- unsafe proximity
- severe oscillation
- repeated opposite turns

## Target-House Verification Plan

At mission start:

- provide a reference image for the target house

During flight:

- compute similarity between current observation and target reference
- only treat a doorway as mission-valid if:
  - it is associated with the target house
  - similarity exceeds threshold

Suggested methods:

- NetVLAD baseline
- DINOv2 global retrieval
- CLIP image-image retrieval baseline

## Termination Conditions

### Outdoor stage termination

Mark `no_entry_termination` if:

- the UAV completes a full loop around the target house without a traversable entry
- or step budget is exhausted

Signals:

- odometry
- yaw wraparound
- revisit of starting sector

### Indoor mission termination

Mark mission finished if:

- person confirmed present
- person confirmed absent after full indoor search
- safety violation
- global step budget exhausted

## Domain Randomization Plan

To guarantee generalization, randomize:

- house textures
- lighting / time of day
- furniture layout
- porch objects
- occluding vegetation
- doorway openness
- camera noise / blur
- depth noise

## Minimum Publishable Version

To keep Phase 6 scoped, the minimum publishable version should include:

1. VLM scene description runtime
2. semantic archive with sentence embeddings
3. reference-house verification baseline
4. stage-aware waypoint generation
5. outdoor-to-indoor entry experiment
6. one archive-to-BC reactive policy experiment

## Immediate Implementation Plan

### Step 1

Implement:

- `phase6_mission_controller.py`
- `vlm_scene_descriptor.py`

### Step 2

Implement:

- `reference_house_matcher.py`
- `semantic_archive_runtime.py`

### Step 3

Implement:

- `phase6_waypoint_planner.py`
- new `/request_phase6_waypoints` integration in [uav_control_server.py](/E:/github/UAV-Flow/UAV-Flow-Eval/uav_control_server.py)

### Step 4

Add:

- dataset export pipeline
- offline BC training entrypoint

### Step 5

Run experiments in this order:

1. scene-state accuracy
2. doorway + target-house verification
3. entry-search success
4. full person-search success
5. distillation baseline

