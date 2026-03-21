# Planner-Driven Exploration Executor Design

## Purpose

This design defines the next execution layer that turns the current stack from:
- "LLM/planner proposes high-level search intent"
- "human still drives most motion"

into:
- "planner/LLM owns sparse exploration intent"
- "server executes bounded exploration segments automatically"
- "human mainly supervises and takes over when needed"

The goal is to make house-search episodes feel genuinely planner-driven without
forcing the LLM to emit low-level actions.

---

## Current Boundary

The current runtime already supports:
- mission-aware planner output
- archive retrieval context
- local reflex inference
- safety gating
- takeover logging
- person-evidence logging

However, the system is still **not** truly LLM-driven exploration yet.

### Why Not Yet

1. Planner output is high-level only.
   - The planner returns semantic search intent and candidate waypoints.
   - It does not own a persistent execution loop.

2. Motion is still user-triggered.
   - In `manual` mode, the user remains the motion source.
   - In `assist_step` mode, the system only appends one reflex action after a manual move.

3. There is no planner-owned episode state machine.
   - No notion of "current autonomous segment".
   - No "continue / hold / replan / stop" loop owned by the server.

4. Waypoint progress does not yet trigger autonomous continuation.
   - The server can request a new plan.
   - But it does not keep executing toward that plan by itself.

5. Safety and evidence are not yet coupled to an autonomous run controller.
   - Risk, reflex gating, takeover, and evidence all exist.
   - They are not yet coordinated by one autonomous executor.

---

## Design Goal

Add a **Planner-Driven Exploration Executor** that:
- consumes the active planner output
- converts it into bounded low-level execution
- monitors progress and safety
- requests replans when needed
- stops on evidence or takeover

The planner still decides **what to search next**.
The executor decides **how long to keep following that decision before stopping or replanning**.

---

## Design Principles

1. LLM stays sparse.
   - It must not emit per-frame motor control.
   - It remains responsible for high-level search semantics only.

2. Execution is bounded.
   - The first version should never run forever.
   - It should execute a limited search segment, then re-evaluate.

3. Safety beats autonomy.
   - Any safety block, repeated failed progress, or takeover request stops the segment.

4. Observability is required.
   - Every autonomous segment must be logged with cause, progress, stop reason, and evidence outcome.

5. Stage rollout is mandatory.
   - Start with a bounded segment executor.
   - Only then move to continuous auto-search.

---

## New Runtime Concept

Add a new runtime object:
- `planner_executor_runtime`

Suggested fields:
- `mode`
- `active`
- `state`
- `run_id`
- `segment_id`
- `mission_id`
- `trigger`
- `current_plan_id`
- `current_search_subgoal`
- `target_waypoint`
- `step_budget`
- `steps_executed`
- `blocked_count`
- `replan_count`
- `last_action`
- `last_progress_cm`
- `last_stop_reason`
- `last_stop_detail`
- `started_at`
- `updated_at`

### Suggested States

- `idle`
- `starting`
- `running`
- `replanning`
- `blocked`
- `waiting_confirmation`
- `completed`
- `aborted`
- `takeover`

---

## Two-Stage Executor Rollout

## Stage A: Plan Segment Executor

This is the recommended next implementation.

### Behavior

The server executes a bounded autonomous segment:
- request or reuse the current plan
- follow the current waypoint using reflex/local heuristic
- stop after one of the following:
  - waypoint reached
  - step budget exhausted
  - no-progress threshold exceeded
  - high-risk gate triggered repeatedly
  - person evidence requires confirmation
  - human takeover starts

### Why This First

It is much easier to debug than a continuous autonomous loop:
- bounded runtime
- deterministic logs
- easier failure analysis
- lower safety risk

### Suggested API

New endpoints:
- `POST /execute_plan_segment`
- `POST /stop_planner_executor`
- `GET /planner_executor`

Suggested request for `POST /execute_plan_segment`:
- `task_label`
- `step_budget`
- `refresh_plan`
- `trigger`

Suggested response:
- `status`
- `planner_executor_runtime`
- `segment_summary`
- `state`

### Suggested Panel Actions

Add buttons:
- `Execute Plan Segment`
- `Stop Executor`

This gives us a low-risk way to test "LLM drives exploration" without requiring a background loop yet.

---

## Stage B: Continuous Auto Search Executor

After segment mode is stable, add a background autonomous loop.

### Behavior

The executor thread repeatedly:
1. ensures a mission exists
2. ensures there is a current plan
3. executes one low-level action
4. checks progress/evidence/safety
5. replans when needed
6. stops on completion, takeover, or failure

### New Modes

Suggested `planner_execute_mode` values:
- `manual`
- `segment`
- `auto_search`

### Why This Is Second

Continuous autonomy is much harder to inspect:
- more race conditions
- more chance of stale plan usage
- harder to attribute failures

So Stage B should only happen after Stage A logs look healthy.

---

## Execution Policy

The executor should never ask the LLM for low-level actions.

Instead, per step:
1. Read `current_plan`.
2. Convert the current semantic target into a local execution target.
3. Request `reflex_runtime`.
4. Gate the reflex action.
5. If reflex is not usable, fall back to local waypoint-following heuristic.
6. Apply exactly one movement step.
7. Re-measure progress.

### Per-Step Sources

Priority order:
1. accepted reflex action from local policy
2. local heuristic action toward the active waypoint
3. hold / stop / replan

This keeps the system layered:
- planner decides semantic intent
- reflex decides local action when confident
- heuristic keeps the segment from stalling when reflex is weak

---

## Stop And Replan Conditions

The executor should stop or replan on explicit criteria.

### Replan Conditions

- waypoint reached
- current region marked observed
- suspect region newly created
- evidence status changed
- no progress for `N` steps
- yaw error remains high for `N` steps
- archive suggests a better expansion cell

### Hard Stop Conditions

- takeover active
- repeated safety gate blocks
- repeated high-risk actions
- planner unavailable and no local fallback allowed
- person confirmed and task requires report instead of further exploration

---

## Search-Subgoal-Specific Execution Rules

### `search_house`
- prefer broader sweep behavior
- allow larger step budget
- replan on room/region transition

### `search_room`
- lower step budget than house-level search
- stay tighter to the priority region

### `search_frontier`
- bias toward exploration and coverage gain
- stop once frontier is reached or visibility opens up

### `approach_suspect_region`
- reduce step budget
- lower speed / tighter gating
- replan quickly if risk rises

### `confirm_suspect_region`
- permit observation-oriented actions
- allow short yaw/vertical scan patterns
- stop once evidence state changes

---

## Safety Integration

The executor must use existing safety signals:
- `risk_score`
- reflex confidence threshold
- shield trigger
- takeover state
- blocked action reason

### New Safety Counters

Suggested counters inside `planner_executor_runtime`:
- `consecutive_blocked_steps`
- `consecutive_no_progress_steps`
- `consecutive_high_risk_steps`

If any exceeds threshold, the segment stops with a clear reason.

---

## Evidence Integration

The executor must react to `person_evidence_runtime`.

### Required behaviors

- if `suspect` appears:
  - either stop the segment
  - or replan into `approach_suspect_region`

- if `confirmed_present`:
  - stop exploration
  - mark run as success

- if `confirmed_absent`:
  - mark region as checked
  - continue search with replan

This is what makes the loop genuinely search-oriented rather than generic navigation.

---

## Logging Requirements

Add a new log stream:
- `phase4_executor_logs`

Each segment should record:
- mission/task label
- planner source/model
- active plan id
- active search subgoal
- step budget
- executed step count
- progress trace
- blocked reasons
- replan reasons
- takeover status
- evidence status before/after
- final stop reason

This should also be included in:
- `/state`
- capture bundle metadata

---

## Minimal Implementation Plan

## Step 1

Add runtime state only:
- `planner_executor_runtime`
- `/planner_executor`

No automatic movement yet.

## Step 2

Add `POST /execute_plan_segment`
- executes a bounded synchronous segment
- default `step_budget=5`

## Step 3

Add panel controls:
- `Execute Plan Segment`
- `Stop Executor`
- executor status line

## Step 4

Log every segment and stop reason.

## Step 5

Only after segment mode is stable:
- add background `auto_search`

---

## Recommended Next Coding Scope

The next safe implementation slice is:

1. extend `runtime_interfaces.py`
   - add `planner_executor_runtime` schema helpers

2. extend `uav_control_server.py`
   - add executor runtime state
   - add `/planner_executor`
   - add `/execute_plan_segment`
   - add bounded segment execution loop

3. extend `uav_control_panel.py`
   - add executor status line
   - add segment execution controls

4. log to:
   - `phase4_executor_logs`

This is the smallest implementation that makes the system genuinely more autonomous while remaining debuggable.

---

## Acceptance Criteria For Stage A

Stage A should be considered successful when:
- a task like `search the house for people` can run one bounded plan segment without keyboard motion
- the segment clearly states whether it stopped due to:
  - success
  - no progress
  - safety block
  - replan
  - takeover
- planner source/model is visible in the logs
- evidence state is captured before/after the segment
- the human can stop the segment at any time

---

## Short Conclusion

The correct next step is **not** "let the LLM emit actions".
The correct next step is:

- keep LLM as sparse semantic planner
- add a planner-owned bounded exploration executor
- let reflex/local heuristic supply low-level movement
- stop or replan on safety, evidence, or progress failure

That is the smallest design that truly turns the current system into LLM-driven embodied search.
