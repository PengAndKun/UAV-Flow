"""
Phase 5 recognition-mission manual generation helpers.

This module formalizes a pre-mission "skill manual" for house-search tasks so
the UAV can reason in stages before taking concrete actions. The manual does
not execute anything by itself; it provides a structured staged decomposition
that later planner/executor layers can consume.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from runtime_interfaces import now_timestamp


def _coerce_depth_stats(depth_stats: Optional[Dict[str, Any]]) -> Dict[str, float]:
    raw = depth_stats if isinstance(depth_stats, dict) else {}
    return {
        "min_depth_cm": float(raw.get("min_depth_cm", raw.get("min_depth", 0.0)) or 0.0),
        "max_depth_cm": float(raw.get("max_depth_cm", raw.get("max_depth", 0.0)) or 0.0),
        "mean_depth_cm": float(raw.get("mean_depth_cm", raw.get("mean_depth", 0.0)) or 0.0),
    }


def infer_house_search_environment(
    *,
    task_label: str,
    mission: Optional[Dict[str, Any]] = None,
    search_runtime: Optional[Dict[str, Any]] = None,
    language_memory_runtime: Optional[Dict[str, Any]] = None,
    doorway_runtime: Optional[Dict[str, Any]] = None,
    depth_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a lightweight environment-context hypothesis for Phase 5 search tasks.

    This is intentionally conservative. It combines explicit mission text with a
    few weak scene heuristics so later modules can decide whether to:
    - search immediately indoors
    - look for an entry doorway first
    - keep gathering evidence because the indoor/outdoor state is uncertain
    """

    mission_payload = mission if isinstance(mission, dict) else {}
    search_payload = search_runtime if isinstance(search_runtime, dict) else {}
    memory_payload = language_memory_runtime if isinstance(language_memory_runtime, dict) else {}
    doorway_payload = doorway_runtime if isinstance(doorway_runtime, dict) else {}
    depth_payload = _coerce_depth_stats(depth_stats)

    task_text = str(task_label or mission_payload.get("task_label", "")).strip().lower()
    mission_type = str(mission_payload.get("mission_type", "semantic_navigation") or "semantic_navigation")
    focus_region_label = str(search_payload.get("priority_region", {}).get("region_label", "")).strip().lower()
    memory_summary = str(memory_payload.get("global_summary", "")).strip().lower()
    best_doorway = doorway_payload.get("best_candidate", {}) if isinstance(doorway_payload.get("best_candidate"), dict) else {}
    doorway_candidate_count = int(doorway_payload.get("candidate_count", 0) or 0)

    outside_score = 0
    inside_score = 0
    rationale: List[str] = []

    if mission_type in ("person_search", "room_search", "target_verification"):
        rationale.append("Mission already targets indoor/room-aware person search.")

    if any(token in task_text for token in ("house", "room", "bedroom", "hallway", "door", "doorway", "entry")):
        rationale.append("Task text references house or room structure.")

    if any(token in task_text for token in ("enter", "entry", "door", "doorway", "front door")):
        outside_score += 1
        rationale.append("Task text suggests an entry sequence may be required.")

    if "doorway" in focus_region_label or "door" in focus_region_label:
        outside_score += 1
        rationale.append("Current focus region references a doorway/entry area.")

    if "unobserved" in memory_summary and "entire house" in memory_summary:
        outside_score += 1
        rationale.append("Language memory still frames the house as globally unobserved.")

    traversable_candidates = int(doorway_payload.get("traversable_candidate_count", 0) or 0)
    if doorway_candidate_count > 0:
        outside_score += 1
        rationale.append("Doorway detector reports doorway-like facade openings.")
    if traversable_candidates > 0:
        outside_score += 2
        rationale.append("Doorway detector reports traversable doorway candidates.")
    if bool(best_doorway.get("traversable", False)):
        outside_score += 1
        rationale.append("Best doorway candidate looks directly enterable from the current view.")
    if float(best_doorway.get("depth_gain_cm", 0.0) or 0.0) >= 180.0:
        outside_score += 1
        rationale.append("Best doorway candidate has strong depth separation from the facade.")

    min_depth = float(depth_payload.get("min_depth_cm", 0.0))
    max_depth = float(depth_payload.get("max_depth_cm", 0.0))
    if max_depth > 900.0 and min_depth < 120.0:
        outside_score += 1
        rationale.append("Depth suggests a nearby boundary with a deep opening beyond it.")
    elif max_depth > 500.0 and min_depth > 120.0 and doorway_candidate_count == 0:
        inside_score += 1
        rationale.append("Depth suggests the UAV is already in an open interior navigable space.")

    if int(search_payload.get("visited_region_count", 0)) >= 8:
        inside_score += 1
        rationale.append("Search runtime already shows multiple visited search regions.")

    if int(search_payload.get("suspect_region_count", 0)) > 0:
        inside_score += 1
        rationale.append("Search runtime already contains suspect regions, implying interior exploration.")

    if outside_score > inside_score:
        location_state = "outside_house"
    elif inside_score > outside_score:
        location_state = "inside_house"
    else:
        location_state = "unknown"
        rationale.append("Indoor/outdoor hypothesis is still ambiguous.")

    return {
        "schema_version": "phase5.environment_context.v1",
        "location_state": location_state,
        "inside_score": int(inside_score),
        "outside_score": int(outside_score),
        "doorway_candidate_count": doorway_candidate_count,
        "traversable_doorway_count": traversable_candidates,
        "rationale": rationale,
    }


def build_phase5_stage(
    *,
    stage_id: str,
    stage_name: str,
    objective: str,
    activation_reason: str,
    planner_focus: str,
    perception_focus: List[str],
    memory_outputs: List[str],
    exit_conditions: List[str],
    status: str = "pending",
) -> Dict[str, Any]:
    return {
        "stage_id": str(stage_id),
        "stage_name": str(stage_name),
        "objective": str(objective),
        "activation_reason": str(activation_reason),
        "planner_focus": str(planner_focus),
        "perception_focus": [str(item) for item in perception_focus],
        "memory_outputs": [str(item) for item in memory_outputs],
        "exit_conditions": [str(item) for item in exit_conditions],
        "status": str(status or "pending"),
    }


def build_phase5_mission_manual(
    *,
    task_label: str,
    mission: Optional[Dict[str, Any]] = None,
    search_runtime: Optional[Dict[str, Any]] = None,
    person_evidence_runtime: Optional[Dict[str, Any]] = None,
    language_memory_runtime: Optional[Dict[str, Any]] = None,
    doorway_runtime: Optional[Dict[str, Any]] = None,
    depth_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the Phase 5 staged task manual for indoor person-search.

    Core idea:
    - before acting, synthesize a staged skills/manual view
    - decompose house-search into environment-recognition and execution stages
    - keep the active stage explicit so later planner/policy modules can follow it
    """

    mission_payload = mission if isinstance(mission, dict) else {}
    search_payload = search_runtime if isinstance(search_runtime, dict) else {}
    evidence_payload = person_evidence_runtime if isinstance(person_evidence_runtime, dict) else {}

    env_context = infer_house_search_environment(
        task_label=task_label,
        mission=mission_payload,
        search_runtime=search_payload,
        language_memory_runtime=language_memory_runtime,
        doorway_runtime=doorway_runtime,
        depth_stats=depth_stats,
    )

    location_state = str(env_context.get("location_state", "unknown") or "unknown")
    traversable_doorway_count = int(env_context.get("traversable_doorway_count", 0) or 0)
    stage_list: List[Dict[str, Any]] = []

    stage_list.append(
        build_phase5_stage(
            stage_id="phase5_stage_01_localize_context",
            stage_name="Localize Context",
            objective="Determine whether the UAV currently starts outside the house, inside the house, or in an ambiguous transition area.",
            activation_reason="Every house-search mission should begin with environment-state recognition before navigation commitment.",
            planner_focus="classify indoor_vs_outdoor and choose the next mission branch",
            perception_focus=["depth free-space structure", "doorway candidates", "language memory summary", "search-runtime region counts"],
            memory_outputs=["environment_context.location_state", "location rationale notes", "initial mission branch"],
            exit_conditions=["location_state == inside_house", "location_state == outside_house", "location_state == unknown but confidence stabilized"],
            status="active",
        )
    )

    if location_state == "outside_house":
        stage_list.extend(
            [
                build_phase5_stage(
                    stage_id="phase5_stage_02_find_entry_door",
                    stage_name="Find Entry Door",
                    objective="Search for a traversable front doorway or entrance opening that leads into the house.",
                    activation_reason="The UAV appears to be outside and must first identify a valid entry point.",
                    planner_focus="prioritize doorway candidates and unexplored entry-facing facade sectors",
                    perception_focus=["doorway detection", "depth opening geometry", "frontier candidates", "semantic doorway cues"],
                    memory_outputs=["doorway candidate notes", "entry candidate ranking", "failed facade observations"],
                    exit_conditions=["doorway candidate confidence exceeds threshold", "no valid entry candidate after full facade sweep"],
                ),
                build_phase5_stage(
                    stage_id="phase5_stage_03_approach_entry",
                    stage_name="Approach Entry Door",
                    objective="Approach the best entry candidate while maintaining collision-safe alignment.",
                    activation_reason="A candidate entry doorway has been found and should be approached before crossing.",
                    planner_focus="stabilize entry alignment and reduce approach distance",
                    perception_focus=["doorway centerline", "collision risk", "depth opening width", "approach clearance"],
                    memory_outputs=["entry alignment state", "failed approach attempts", "doorway traversability update"],
                    exit_conditions=["entry distance below threshold", "approach blocked", "candidate rejected"],
                ),
                build_phase5_stage(
                    stage_id="phase5_stage_04_cross_entry",
                    stage_name="Cross Entry Transition",
                    objective="Pass through the detected doorway and transition into the interior search state.",
                    activation_reason="The UAV is aligned with a traversable entry and should cross into the house.",
                    planner_focus="short safe crossing through doorway",
                    perception_focus=["opening continuity", "door frame boundaries", "near-collision risk", "post-crossing free space"],
                    memory_outputs=["entry success/failure event", "indoor transition timestamp", "new interior anchor region"],
                    exit_conditions=["location_state becomes inside_house", "crossing blocked", "replan required"],
                ),
            ]
        )

    stage_list.extend(
        [
            build_phase5_stage(
                stage_id="phase5_stage_05_house_search",
                stage_name="Structured House Search",
                objective="Explore rooms and connected spaces systematically to maximize search coverage while avoiding repeated sweeps.",
                activation_reason="Once indoors, the UAV should switch from entry logic to room/space search logic.",
                planner_focus="search_house or search_room with coverage-first ordering",
                perception_focus=["room openings", "depth corridors", "person evidence", "archive retrieval", "language memory region notes"],
                memory_outputs=["visited region updates", "room summaries", "frontier notes", "candidate suspect regions"],
                exit_conditions=["suspect evidence appears", "house coverage threshold reached", "mission timeout"],
            ),
            build_phase5_stage(
                stage_id="phase5_stage_06_verify_target",
                stage_name="Verify Target",
                objective="Approach, revisit, or reject suspect regions until the system can confirm whether a person is present.",
                activation_reason="Suspect evidence requires a dedicated confirmation phase rather than generic room sweeping.",
                planner_focus="approach_suspect_region / confirm_suspect_region / revisit_suspect_region",
                perception_focus=["suspect region geometry", "multi-frame evidence", "close-range safety", "occlusion boundaries"],
                memory_outputs=["confirmation evidence", "estimated target position", "verified negative regions"],
                exit_conditions=["confirmed_present", "confirmed_absent", "manual review required"],
            ),
            build_phase5_stage(
                stage_id="phase5_stage_07_report_result",
                stage_name="Report Result",
                objective="Summarize whether a person exists, where they are estimated to be, and what evidence supports the conclusion.",
                activation_reason="Every mission should finish with an explicit search result rather than only a trajectory trace.",
                planner_focus="stabilize final report state and capture evidence summary",
                perception_focus=["search_result summary", "person evidence state", "language memory notes"],
                memory_outputs=["final report", "mission summary", "supporting evidence references"],
                exit_conditions=["result_status != unknown", "manual termination"],
            ),
        ]
    )

    active_stage_id = stage_list[0]["stage_id"]
    if location_state == "outside_house":
        if traversable_doorway_count > 0:
            active_stage_id = "phase5_stage_03_approach_entry"
        else:
            active_stage_id = "phase5_stage_02_find_entry_door"
    if location_state == "inside_house":
        active_stage_id = "phase5_stage_05_house_search"
    if str(evidence_payload.get("evidence_status", "idle") or "idle") in ("suspect", "confirmed_present", "confirmed_absent"):
        active_stage_id = "phase5_stage_06_verify_target"
    if str(evidence_payload.get("evidence_status", "idle") or "idle") == "confirmed_present":
        active_stage_id = "phase5_stage_07_report_result"
    if str(evidence_payload.get("evidence_status", "idle") or "idle") == "confirmed_absent":
        active_stage_id = "phase5_stage_07_report_result"

    for stage in stage_list:
        stage["status"] = "active" if stage["stage_id"] == active_stage_id else "pending"

    return {
        "schema_version": "phase5.mission_manual.v1",
        "generated_at": now_timestamp(),
        "task_label": str(task_label or ""),
        "mission_type": str(mission_payload.get("mission_type", "semantic_navigation") or "semantic_navigation"),
        "environment_context": env_context,
        "active_stage_id": active_stage_id,
        "stages": stage_list,
    }
