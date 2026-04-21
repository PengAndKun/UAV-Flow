# V3 Follow-up Data Collection Plan

## Purpose

This document converts the `pilot_distill_v3_reviewed` evaluation results into a concrete next-round data collection plan.

The goal is not to collect more samples in a generic way. The goal is to **fill the specific target-conditioned weaknesses that remain after review-cleaned training**.

## Current conclusion from V3

`v3` confirms that reviewed target-house labels are useful:

- `target_conditioned_state` improved and is now the strongest part of the model.
- `target_conditioned_subgoal` also improved.
- `target_conditioned_action_hint` and `target_conditioned_target_candidate_id` are still weak.

This means the model is learning:

- whether the current visible entry belongs to the target house,
- whether it should ignore a non-target entry,
- whether the target entry is blocked or merely visible.

But it is still weak at:

- deciding the fine-grained action direction,
- choosing left versus right detour consistently,
- recognizing the rare `approach_target_entry` case,
- recognizing the rare `target_house_entry_approachable` case.

## What not to prioritize

Do **not** spend the next round mainly collecting more of these classes:

- `target_house_not_in_view`
- `ignore_non_target_entry`
- `reorient_to_target_house`
- `yaw_left`
- `yaw_right`

These already dominate the reviewed export set and will make the distribution even more imbalanced.

## Highest-priority classes to collect

### Priority 1: `target_house_entry_approachable`

This is the most important missing target-conditioned state.

Current problem:

- It is almost absent in training.
- The model still cannot reliably separate:
  - target entry is visible but not ready,
  - target entry is ready to approach.

Recommended new samples:

- `+20` to `+30`

Required scene conditions:

- target house is correct,
- target entry is clearly visible,
- path is not blocked,
- door is not yet at crossing distance,
- door is near the image center or only mildly off-center.

Desired labels:

- `target_conditioned_state = target_house_entry_approachable`
- `target_conditioned_subgoal = approach_target_entry`
- `target_conditioned_action_hint = forward`

### Priority 2: `approach_target_entry`

This is the key target-conditioned subgoal that still has too little support.

Current problem:

- The model often confuses it with:
  - `detour_right_to_target_entry`
  - `ignore_non_target_entry`
  - or generic non-target behavior

Recommended new samples:

- `+25`

Required scene conditions:

- target house already identified correctly,
- target entry belongs to target house,
- no strong obstacle in the immediate corridor,
- UAV should clearly move forward rather than rotate in place.

Desired labels:

- `target_conditioned_subgoal = approach_target_entry`
- `target_conditioned_action_hint = forward`

### Priority 3: `detour_left_to_target_entry`

Current problem:

- left/right target detour is still unbalanced,
- the model is better at some right-detour patterns than left-detour patterns.

Recommended new samples:

- `+15` left-detour samples
- `+8` right-detour samples for balance check

Required scene conditions:

- target entry is correct,
- obstacle is real and near the corridor,
- obstacle geometry clearly makes one direction preferable.

Desired labels:

- `target_conditioned_state = target_house_entry_blocked`
- `target_conditioned_subgoal = detour_left_to_target_entry` or `detour_right_to_target_entry`
- `target_conditioned_action_hint = left` or `right`

### Priority 4: `hold`

Current problem:

- The model still does not understand the near-target stabilize-and-confirm behavior.

Recommended new samples:

- `+8` to `+12`

Required scene conditions:

- target entry is already near,
- it is not yet time to cross,
- a short stable observation is more appropriate than moving.

Desired labels:

- `target_conditioned_action_hint = hold`

## Collection templates

## Template A: Approach-ready target entry

Use this template to collect `target_house_entry_approachable` and `approach_target_entry`.

Setup:

- choose a target house explicitly,
- place UAV facing the correct entrance,
- keep door visible,
- avoid strong clutter in the approach corridor.

Distance bands:

- far: `6m to 8m`
- mid: `4m to 6m`
- near: `2.5m to 4m`

For each band, collect:

- door centered,
- door slightly left,
- door slightly right.

Expected action:

- `forward`

Do not include:

- another house door dominating the frame,
- very large occluders in front of the target entry,
- situations where the UAV should actually rotate instead of move forward.

## Template B: Search on target facade

Use this template to collect `target_house_entry_visible` and `keep_search_target_house`.

Setup:

- target house is already in front view,
- but the target entry is not yet reliably confirmed,
- facade, columns, windows, porch edges, or side wall are visible.

Recommended motion when collecting:

- slow horizontal observation,
- gentle yaw,
- no strong approach yet.

Expected actions:

- `yaw_left`
- `yaw_right`

Do not include:

- obvious non-target house doors that dominate the frame,
- scenes where the correct action should be forward.

## Template C: Blocked target entry detour

Use this template to collect `detour_left_to_target_entry` and `detour_right_to_target_entry`.

Setup:

- target entry is correct,
- obstacle blocks the direct path,
- the obstacle should clearly create a left/right preference.

Good blocking objects:

- porch columns,
- bushes,
- railings,
- lamp posts,
- furniture near the entrance.

Best practice:

- create paired samples
- keep the same door and camera distance
- only change obstacle side or UAV offset

This makes the left/right supervision much cleaner.

## Template D: Near-target stabilize

Use this template to collect `hold`.

Setup:

- target entry already near,
- target house confirmed,
- no need to rotate aggressively,
- no need to push forward immediately.

Expected action:

- `hold`

This template is useful for later crossing-stage decisions as well.

## Collection rules

### Rule 1: One sample should reflect one dominant decision

Avoid mixing:

- reorientation,
- target search,
- detour,
- and direct approach

in a way that makes the correct label ambiguous.

### Rule 2: Avoid near-duplicate frames

Do not save many nearly identical consecutive frames just to increase count.

Preferred:

- slight pose changes,
- slight offset changes,
- slight distance changes,
- different obstruction conditions.

### Rule 3: Keep target identity explicit

Before collecting each episode, verify:

- selected `target_house_id`
- current house context
- intended entry belongs to the target house

This is especially important now that reviewed target-house labels are part of the training chain.

### Rule 4: Use balanced left/right collection

Whenever collecting detour samples:

- do not collect only one side
- always aim for a left/right pair if possible

### Rule 5: Separate target-visible from target-approachable

This boundary is one of the main remaining weaknesses.

If the frame still requires target-facade search:

- label it as `target_house_entry_visible`

If the frame clearly supports moving forward to the target entry:

- label it as `target_house_entry_approachable`

## Suggested next-round targets

Recommended minimum next-round additions:

- `target_house_entry_approachable / approach_target_entry / forward`: `+25`
- `target_house_entry_visible / keep_search_target_house`: `+15`
- `detour_left_to_target_entry / left`: `+15`
- `detour_right_to_target_entry / right`: `+8`
- `hold`: `+10`

This is enough for a focused `v4` without trying to rebalance the entire dataset at once.

## After collection

After the next collection round, run the same preprocessing chain again:

1. refresh fusion results if needed
2. review target house id if needed
3. run teacher validator
4. run entry state builder
5. run dataset export
6. compare the new target-conditioned class distribution against `v3`

The goal of the next round is:

- not just more data,
- but better support for target-conditioned approach and fine-grained action decisions.
