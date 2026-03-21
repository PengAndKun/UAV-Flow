## Phase 4.5 Development Plan

Title:
- `LLM High-Level Search Planner Adapter`

Primary goal:
- replace the current heuristic planner with a sparse, structured LLM planner for house-level person-search missions

Scope boundary:
- the LLM owns high-level search intent only
- the LLM does not own:
  - per-step motor control
  - local obstacle avoidance
  - low-level yaw/translation micro-actions
- low-level execution remains in:
  - `reflex_policy_server.py`
  - `uav_control_server.py`
  - safety / takeover gates

---

## 1. What Phase 4.5 Should Deliver

Phase 4.5 should make the planner layer capable of:
- reading mission/search context from the current runtime
- producing structured search guidance using an LLM
- staying sparse in cadence and cost
- remaining safe through schema validation and fallback to heuristic planning

Required high-level outputs:
- `mission_type`
- `search_subgoal`
- `priority_region`
- `candidate_regions`
- `candidate_waypoints`
- `confirm_target`
- `planner_confidence`
- `should_replan`
- `explanation`

Success condition:
- live runs use LLM-generated semantic guidance for person-search tasks
- all LLM outputs are normalized into the existing plan schema
- failures or malformed outputs fall back cleanly to heuristic planning

---

## 2. API Inputs Needed From The User

Before Phase 4.5 live experiments, the following must be provided:
- `api_key`
- `base_url`
- `model_name`
- whether the endpoint supports image input
- token/cost constraints
- desired planner cadence policy

Recommended first-pass planner cadence:
- call the LLM only when one of the following is true:
  - task starts
  - current region changes
  - suspect evidence appears
  - confirmation is requested
  - planner step interval `K` is reached

Recommended first-pass cost policy:
- start with text-only planning
- summarize visual/depth/search state into structured text
- add image input later only if text-only guidance is too weak

---

## 3. Current Codebase Starting Point

Already available:
- mission/search schema in:
  - `UAV-Flow-Eval/runtime_interfaces.py`
- planner service entrypoint in:
  - `UAV-Flow-Eval/planner_server.py`
- mission/search runtime propagation in:
  - `UAV-Flow-Eval/uav_control_server.py`
- mission-guidance validator in:
  - `UAV-Flow-Eval/validate_mission_guidance.py`
- evidence/result runtime in:
  - `UAV-Flow-Eval/uav_control_server.py`
  - `UAV-Flow-Eval/uav_control_panel.py`

Current limitation:
- `planner_server.py` still builds plans from local heuristic logic only
- there is no LLM adapter, prompt builder, or response parser yet

---

## 4. File-Level Development Plan

### 4.5.1 Add a Planner Client Layer

New file:
- `UAV-Flow-Eval/llm_planner_client.py`

Purpose:
- isolate all LLM API calls behind a small client

Responsibilities:
- build HTTP requests
- attach auth headers
- send planner prompt payloads
- return raw model output plus usage metadata
- surface timeout / parse / auth failures cleanly

Suggested output shape:
- `text`
- `raw_response`
- `latency_ms`
- `usage`
- `model_name`

### 4.5.2 Add Prompt Building And Response Parsing

New file:
- `UAV-Flow-Eval/llm_planner_adapter.py`

Purpose:
- convert runtime state into an LLM prompt
- parse the LLM response into the shared planner schema

Responsibilities:
- build compact mission/search prompt text from:
  - `task_label`
  - `mission`
  - `search_runtime`
  - `person_evidence_runtime`
  - `search_result`
  - local pose/depth summary
- require structured JSON output from the model
- validate and coerce the result through existing plan helpers
- reject malformed output and trigger fallback

Expected internal functions:
- `build_llm_planner_prompt(...)`
- `call_llm_planner(...)`
- `parse_llm_planner_response(...)`
- `build_llm_plan(...)`

### 4.5.3 Upgrade Planner Server Into Dual Mode

Modify:
- `UAV-Flow-Eval/planner_server.py`

Add:
- planner mode switch:
  - `heuristic`
  - `llm`
  - `hybrid`

Recommended args:
- `--planner_mode`
- `--llm_api_key_env`
- `--llm_base_url`
- `--llm_model`
- `--llm_timeout_s`
- `--llm_max_retries`
- `--llm_force_json`
- `--llm_include_images`
- `--fallback_to_heuristic`

Behavior:
- `heuristic`: current behavior
- `llm`: use LLM planner first, fallback if enabled
- `hybrid`: use heuristic for generic navigation, LLM for search/verify missions

### 4.5.4 Extend Runtime Metadata For Evaluation

Modify:
- `UAV-Flow-Eval/runtime_interfaces.py`
- `UAV-Flow-Eval/uav_control_server.py`

Add planner/runtime metadata:
- `planner_source = heuristic | llm | fallback_heuristic`
- `planner_latency_ms`
- `planner_usage`
- `planner_model_name`
- `planner_prompt_version`
- `llm_fallback_used`

This data should appear in:
- `/state`
- capture bundles
- runtime debug

### 4.5.5 Add Planner-Specific Validation

New file:
- `UAV-Flow-Eval/validate_llm_planner.py`

Purpose:
- run offline tests on search-task prompts without Unreal

Validation targets:
- correct `mission_type`
- correct `search_subgoal`
- correct `priority_region`
- stable JSON schema
- safe fallback behavior on malformed outputs

### 4.5.6 Add Experiment Logging Support

Modify:
- `UAV-Flow-Eval/uav_control_server.py`
- optionally `UAV-Flow-Eval/online_reflex_eval.py`

Add logging for:
- LLM plan count
- fallback count
- planner token usage
- planner latency distribution
- mission-level planner decisions over time

---

## 5. Recommended Implementation Order

### Milestone A: Offline Adapter Only

Goal:
- make the LLM planner work without touching live flight yet

Tasks:
- build `llm_planner_client.py`
- build `llm_planner_adapter.py`
- add `validate_llm_planner.py`
- test with fixed prompts and mock runtime payloads

Exit check:
- one valid plan JSON returned for each of:
  - `search the house for people`
  - `search the bedroom first`
  - `approach and verify the suspect region`

### Milestone B: Planner Server Dual Mode

Goal:
- let `planner_server.py` switch between heuristic and LLM planning

Tasks:
- add mode flags
- add fallback path
- add `/health` and `/schema` metadata for planner mode/model info

Exit check:
- planner server returns valid normalized plan payloads in:
  - `heuristic`
  - `llm`
  - `hybrid`

### Milestone C: Live Runtime Integration

Goal:
- use LLM high-level guidance in real live runs

Tasks:
- wire planner metadata into `/state`
- wire planner metadata into capture bundles
- run live tasks from the control panel

Exit check:
- panel, `/state`, and capture bundles agree on:
  - planner source
  - mission type
  - search subgoal
  - candidate regions

### Milestone D: First Comparison Experiment

Goal:
- compare current heuristic planner against the new LLM planner

Comparison:
- `heuristic planner + current runtime`
- `LLM planner + current runtime`

Metrics:
- mission completion quality
- takeover count
- planner latency
- planner token usage
- search-result correctness

---

## 6. Minimum Experiment Plan For 4.5

### Task Set

Use these first:
- `search the house for people`
- `search the bedroom first`
- `search the living room for a survivor`
- `approach and verify the suspect region near the bedroom door`
- `revisit the bathroom and confirm whether a person is there`

### Experiment 1: Schema And Stability

Purpose:
- verify the LLM reliably outputs valid structured plans

Check:
- parse success rate
- fallback rate
- average planner latency

### Experiment 2: Live Guidance Alignment

Purpose:
- verify the LLM guidance matches the mission semantics during live runs

Check:
- `Mission ...` line
- `Plan ...` line
- capture bundle metadata
- search log consistency

### Experiment 3: Heuristic vs LLM Planner

Purpose:
- show whether LLM guidance improves search-task behavior

Compare:
- heuristic planner
- LLM planner

Track:
- takeover count
- evidence progression
- search-result consistency
- latency and token cost

---

## 7. Safety And Fallback Rules

Must keep:
- existing reflex execution gates
- takeover logging
- assist-step/manual execution modes
- fallback to heuristic if:
  - HTTP failure
  - timeout
  - malformed JSON
  - missing required planner fields

The LLM planner must never directly emit:
- motor commands
- low-level action labels
- unsafe bypass flags

---

## 8. Immediate Next Step

The best next implementation action is:
- create `llm_planner_client.py`
- create `llm_planner_adapter.py`
- extend `planner_server.py` with `--planner_mode heuristic|llm|hybrid`

This is the smallest slice that turns Phase 4.5 from planning into executable code.

---

## 9. Current Implementation Status

Completed in the first working 4.5 slice:
- `llm_planner_client.py`
- `llm_planner_adapter.py`
- `planner_server.py` planner-mode support:
  - `heuristic`
  - `llm`
  - `hybrid`
- runtime metadata propagation through:
  - `runtime_interfaces.py`
  - `uav_control_server.py`
  - `uav_control_panel.py`
- offline validator:
  - `validate_llm_planner.py`

Current client compatibility:
- OpenAI-compatible `chat/completions`
- OpenAI-compatible `responses`
- configurable endpoint path
- configurable auth header / auth scheme
- optional image input support

Offline validation result:
- `case_count = 5`
- `pass_count = 5`
- `fail_count = 0`
- malformed JSON handling passed

Meaning:
- the local 4.5 planner scaffold is no longer the blocker
- the next real milestone is live API integration and heuristic-vs-LLM comparison

### Live Backend Status

Current live backend status after smoke testing:
- `google_gemini` is usable now
- `anthropic_messages` is implemented locally but currently blocked by provider/model availability for the supplied token

Recommended first live backend:
- `gemini-3.1-flash-lite-preview`

Reason:
- simple JSON request succeeded
- end-to-end `build_llm_plan(...)` smoke test succeeded
- runtime metadata and token usage were returned correctly

### Planner Switching UX Status

The live experiment surface is now split into two layers:
- `planner_server.py` owns the actual planner mode / route / backend config
- `uav_control_panel.py` can switch that config through `uav_control_server.py`

Implemented pieces:
- planner config endpoint in planner server:
  - `GET /config`
  - `POST /config`
- planner config proxy in control server:
  - `GET /planner_config`
  - `POST /planner_config`
- panel-side switching UI:
  - preset-based switching for baseline and LLM backends
  - manual editing for mode / route / API style / model / base URL / env name / fallback

This means Phase 4.5 experiments can now be run as:
- pure heuristic baseline
- pure Gemini Lite
- pure Gemini Flash
- search hybrid
- anthropic-compatible variants when provider-side access becomes available
