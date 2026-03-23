# Phase 5 Recognition Mission Plan

## Goal

Phase 5 shifts the project from:
- “LLM plans how to search”

to:
- “the UAV first recognizes its search situation, then follows a staged mission manual to reach and find the target.”

The key idea is that `search the house for people` should not be treated as one flat instruction.
It should be decomposed into a recognition-first mission manual:

1. determine whether the UAV is outside or already inside the house
2. if outside, find an entry doorway
3. approach and cross the doorway safely
4. once inside, perform structured room/space search
5. when suspect evidence appears, switch to verification
6. finally report whether a person exists and where they are likely located

---

## Why Phase 5 Is Needed

Current Phase 4.6 capabilities are already useful:
- high-level LLM search planning
- pure LLM action selection
- language search memory
- person evidence logging

However, the current system still lacks a strong explicit recognition layer for:
- indoor vs outdoor state
- doorway / entry affordance
- “can I enter here?” reasoning

That gap is exactly why the UAV may stop near a house facade or door-like structure without truly recognizing:
- this is the entrance
- it is traversable
- entering is now the correct subtask

So Phase 5 should formalize:
- pre-action recognition
- staged task decomposition
- skill/manual generation before execution

---

## Core Phase 5 Pipeline

### Stage 1: Localize Context

Question:
- Am I currently outside the house, inside the house, or in an ambiguous transition area?

Signals:
- mission text
- search runtime
- language memory summary
- doorway candidates
- depth geometry

Outputs:
- `location_state = outside_house | inside_house | unknown`
- rationale notes for why that state was chosen

### Stage 2: Find Entry Door

Only active when:
- `location_state == outside_house`

Question:
- Where is the most likely traversable front door / doorway / entrance opening?

Signals:
- depth opening geometry
- doorway candidate detector
- facade search memory
- LLM region prioritization

Outputs:
- doorway candidates
- candidate ranking
- failed facade observations

### Stage 3: Approach Entry Door

Question:
- How do I align with the best doorway candidate safely?

Signals:
- doorway centerline
- opening width
- local collision risk
- free-space continuity

Outputs:
- aligned entry approach
- doorway traversability update

### Stage 4: Cross Entry Transition

Question:
- Can I safely pass through the doorway and transition to interior search?

Signals:
- near-range depth
- doorway boundary continuity
- post-crossing free space

Outputs:
- `location_state -> inside_house`
- entry success/failure event

### Stage 5: Structured House Search

Once indoors, the UAV should not just wander.
It should:
- search room-by-room or frontier-by-frontier
- use archive + language memory
- reduce repeated sweeps
- record visited/suspect/confirmed regions

Outputs:
- visited region updates
- room summaries
- suspect region candidates

### Stage 6: Verify Target

When suspect evidence appears:
- stop generic search
- approach the suspect region
- confirm or reject target presence

Outputs:
- `confirmed_present` or `confirmed_absent`
- estimated target position
- supporting evidence

### Stage 7: Report Result

Final state should explicitly answer:
- Is there a person?
- Where is the estimated person location?
- What evidence supports that answer?

---

## Recommended Representation

Phase 5 should not rely on language alone.

Use two complementary memory layers:

1. Structured spatial/search memory
- archive runtime
- search runtime
- doorway candidates
- visited/suspect/confirmed region counters

2. Language mission memory
- short region summaries
- why a doorway is promising
- why a region should be revisited
- final mission summary

This means:
- control logic remains stable and structured
- LLM still receives compact high-level explanations

---

## New Runtime Concepts

Phase 5 should introduce or expand the following runtime objects:

- `environment_context_runtime`
  - indoor/outdoor/unknown classification
  - rationale

- `doorway_runtime`
  - doorway candidates
  - best candidate
  - traversability confidence

- `mission_manual_runtime`
  - staged task manual
  - active stage
  - stage transitions

- `language_memory_runtime`
  - already exists
  - should now also summarize:
    - entry search
    - doorway observations
    - room-to-room search progress

---

## Implementation Order

### Step A

Build the pre-mission manual generator.

Output:
- staged manual
- active stage
- environment hypothesis

### Step B

Add doorway detection / doorway candidate runtime.

Start with geometry-first detection:
- opening width
- depth discontinuity
- traversable central gap

Then optionally add semantic confirmation later.

### Step C

Feed:
- `environment_context_runtime`
- `doorway_runtime`
- `mission_manual_runtime`

into planner prompts and action prompts.

### Step D

Use the active stage to constrain behavior:
- outside house -> find entry
- doorway found -> approach/cross
- inside house -> search
- suspect present -> verify

---

## Immediate Code Direction

The first concrete Phase 5 artifact should be:
- a mission-manual builder

That builder already fits the current architecture because:
- planner already consumes mission/search summaries
- language memory already exists
- action planner already consumes compact execution context

So the next integration step is:
- generate the Phase 5 mission manual before search execution
- expose it in runtime
- add it to planner/action prompts

---

## Expected Experiment Impact

Phase 5 should improve:
- entry behavior when the UAV starts outside
- doorway reasoning
- reduction of “stuck at facade / not entering” failures
- staged transition quality between search phases

This is especially important for the thesis because it makes the task formulation more realistic:
- not just “search”
- but “recognize current context, decide entry/search/verification stage, then act”
