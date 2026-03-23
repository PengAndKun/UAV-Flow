# Phase 5 Revision: Multimodal Scene Interpretation And Continuous Waypoints

## Why This Revision Is Needed

The previous Phase 5 direction emphasized:
- geometry-first doorway candidates
- structured mission stages
- planner/action prompts consuming compact doorway summaries

That direction is still useful as an engineering baseline, but it is not the right main story for the paper.

The user clarified the intended method more precisely:

1. the UAV should first understand the scene from RGB + depth together
2. it should determine whether it is outside the house or already inside
3. if outside, it should identify the real entry door and whether the opening is traversable
4. it should then generate a short sequence of navigation waypoints
5. the UAV should move according to those waypoints
6. the waypoint-following part is exactly the component we ultimately want to train

So the main Phase 5 method should become:

- multimodal scene interpretation
- staged task decomposition
- continuous waypoint generation
- waypoint execution and future waypoint-conditioned learning

This is different from treating doorway detection as the main decision-maker.

---

## Revised Core Idea

For a task like:

`search the house for people`

the system should not immediately start generic house search.

It should first solve the scene interpretation problem:

1. where am I relative to the house?
2. is there a visible entry door?
3. is that door traversable?
4. what are the next several waypoint targets?

The multimodal model should directly consume:
- RGB image
- depth image
- current pose
- task text
- current mission stage
- memory summary

and return:
- scene classification
- active phase/stage
- doorway understanding
- a short sequence of waypoints
- rationale

---

## Correct Phase 5 Task Decomposition

The task `search the house for people` should be decomposed into the following ordered stages.

### Stage 0: Scene Interpretation

Goal:
- infer whether the UAV is outside the house, inside the house, or at an ambiguous transition point

Inputs:
- RGB
- depth
- pose
- task text
- memory summary

Outputs:
- `scene_state = outside_house | inside_house | threshold_zone | unknown`
- `entry_door_visible = true/false`
- `entry_door_traversable = true/false`
- `confidence`
- textual rationale

### Stage 1: Entry Search

Only active when:
- `scene_state == outside_house`

Goal:
- find the true entrance instead of treating the facade as generic search space

Outputs:
- candidate entry region
- candidate entry center
- entry confidence

### Stage 2: Entry Approach

Goal:
- generate approach waypoints that align the UAV with the doorway

Outputs:
- 2 to 5 short-horizon waypoints
- expected heading updates
- stop condition

### Stage 3: Threshold Crossing

Goal:
- pass through the doorway safely

Outputs:
- transition waypoints
- expected new scene state after crossing

### Stage 4: Indoor Stabilization

Goal:
- once inside, produce a short stabilization waypoint sequence before broad search

Outputs:
- indoor stabilization waypoint(s)
- initial room-search heading

### Stage 5: Structured Indoor Search

Goal:
- perform room-wise or frontier-wise indoor person search

Outputs:
- search waypoints
- candidate search regions
- revisit cues

### Stage 6: Suspect Verification

Goal:
- if human evidence appears, stop generic search and verify the target

Outputs:
- verification waypoints
- final presence/absence decision support

### Stage 7: Final Report

Goal:
- produce the final result for the mission

Outputs:
- whether a person exists
- estimated position
- evidence summary

---

## New Main Interface

Instead of using the multimodal model only as a sparse high-level planner, Phase 5 should introduce a new interface:

`multimodal_scene_planner`

Its job is:
- understand the current scene
- identify the active stage
- generate a short continuous waypoint sequence

### Input

The model input should include:
- `task_label`
- `rgb_image`
- `depth_image`
- `pose`
- `language_memory_summary`
- `search_runtime`
- `person_evidence_runtime`
- `phase5_manual_summary`

### Output

The model output should be structured JSON like this:

```json
{
  "scene_state": "outside_house",
  "active_stage": "approach_entry",
  "entry_door_visible": true,
  "entry_door_traversable": true,
  "planner_confidence": 0.88,
  "waypoints": [
    {
      "label": "entry_align",
      "x_hint": 0.50,
      "y_hint": 0.58,
      "z_offset_cm": 0,
      "yaw_hint_deg": 0
    },
    {
      "label": "entry_threshold",
      "x_hint": 0.50,
      "y_hint": 0.68,
      "z_offset_cm": 0,
      "yaw_hint_deg": 0
    },
    {
      "label": "indoor_stabilize",
      "x_hint": 0.50,
      "y_hint": 0.80,
      "z_offset_cm": 0,
      "yaw_hint_deg": 0
    }
  ],
  "reasoning": "The UAV is outside the house and facing an open front door. The doorway region is deep and traversable in the depth image."
}
```

The waypoint sequence does not need to be world-coordinate perfect at first.
Image-normalized hints are acceptable for the first paper version.

---

## Why RGB And Depth Must Be Used Together

The user's example is correct:

- RGB can tell us that the UAV is facing a house entrance or door-like structure
- depth can tell us that the opening is actually deep and traversable

If we only use RGB:
- a closed door, dark panel, or window can look similar

If we only use depth:
- a generic dark opening or shadowed recess can be mistaken for an entry

So the correct recognition unit is:

- semantic appearance from RGB
- geometry and traversability from depth

This multimodal combination should be the main Phase 5 recognition logic.

---

## Relationship To Training

This revision also clarifies what part of the system should eventually be trained.

The intended future stack is:

1. teacher:
   - multimodal LLM/VLM
   - outputs scene interpretation + continuous waypoint sequence

2. student:
   - local UAV executor / low-level policy
   - learns to move according to the generated waypoint sequence

So the waypoint sequence is not just an execution artifact.
It is also a future training target.

That makes Phase 5 much cleaner in paper form:

- perception and reasoning produce waypoint supervision
- control and learning consume waypoint supervision

---

## What Should Change In The Current Codebase

The current codebase already has:
- planner server
- pure LLM action path
- doorway runtime
- language memory
- Phase 5 mission manual

But the main method should now pivot to a new path.

### Keep

These modules should remain:
- `language_search_memory.py`
- `phase5_mission_manual.py`
- `doorway_detection.py`
- `planner_server.py`
- `uav_control_server.py`
- `uav_control_panel.py`

### Reposition

These modules should be treated as support, not the final main logic:
- `doorway_detection.py`
  - support signal
  - not the sole entry decision-maker
- `llm_action_adapter.py`
  - baseline / comparison route
  - not the main paper method

### Add

The next new modules should be:
- `multimodal_scene_waypoint_adapter.py`
- `validate_multimodal_scene_waypoints.py`

And new runtime objects:
- `scene_interpretation_runtime`
- `waypoint_sequence_runtime`

---

## Recommended Immediate Development Order

### Step 1

Write the new multimodal prompt and JSON schema.

This step should define:
- scene-state labels
- stage labels
- waypoint sequence schema
- rationale schema

### Step 2

Create a standalone adapter:
- RGB + depth + context in
- structured scene interpretation + waypoint list out

Do not tie it to low-level control yet.

### Step 3

Add a viewer in the panel for:
- raw multimodal reply
- parsed scene interpretation
- parsed waypoint sequence

### Step 4

Use the returned waypoint sequence as the target for a bounded executor:
- `execute_waypoint_sequence`

### Step 5

Only after that, use the resulting trajectory pairs as training data for the waypoint-following model.

---

## Paper Framing

The paper should describe the method as:

- a staged multimodal scene understanding and waypoint generation framework
- not merely doorway heuristics
- not merely step-wise LLM action selection

The method contribution should be:

1. multimodal scene interpretation for indoor-search entry reasoning
2. stage-aware mission decomposition
3. continuous waypoint generation from RGB + depth
4. future training supervision for waypoint-conditioned execution

---

## Current Decision

The project should proceed with the following rule:

- doorway heuristics remain available as auxiliary signals
- but the main Phase 5 design is now:
  - `RGB + depth -> scene interpretation -> staged reasoning -> continuous waypoints`

This is the plan that should guide the next implementation step.
