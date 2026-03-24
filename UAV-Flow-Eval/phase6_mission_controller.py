"""
Phase 6 mission controller.

This module upgrades the current "search the house for people" flow into a
more explicit stage machine that matches the new paper direction:

RGB/depth/VLM scene understanding -> stage-aware mission runtime ->
targeted waypoint/action generation.

The controller intentionally stays lightweight for now. It consumes the
existing Phase 5/scene-waypoint/doorway/langmem outputs and synthesizes a
clear Phase 6 runtime block that later modules can build on.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from runtime_interfaces import build_phase6_mission_runtime_state


PHASE6_STAGE_NAMES: Dict[str, str] = {
    "outside_localization": "Outside Localization",
    "target_house_verification": "Target House Verification",
    "entry_search": "Entry Search",
    "approach_entry": "Approach Entry",
    "cross_entry": "Cross Entry",
    "indoor_room_search": "Indoor Room Search",
    "suspect_verification": "Suspect Verification",
    "mission_report": "Mission Report",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _truthy_task_for_house_search(task_label: str, mission_type: str) -> bool:
    text = str(task_label or "").strip().lower()
    if mission_type in ("person_search", "room_search", "target_verification"):
        return True
    return any(token in text for token in ("house", "room", "person", "entry", "door"))


def _build_stage_queue(active_stage_id: str) -> List[str]:
    ordering = [
        "outside_localization",
        "target_house_verification",
        "entry_search",
        "approach_entry",
        "cross_entry",
        "indoor_room_search",
        "suspect_verification",
        "mission_report",
    ]
    if active_stage_id not in ordering:
        return ordering
    start_index = ordering.index(active_stage_id)
    return ordering[start_index:]


def build_phase6_mission_runtime(
    *,
    task_label: str,
    mission: Optional[Dict[str, Any]] = None,
    search_runtime: Optional[Dict[str, Any]] = None,
    doorway_runtime: Optional[Dict[str, Any]] = None,
    vlm_scene_runtime: Optional[Dict[str, Any]] = None,
    reference_match_runtime: Optional[Dict[str, Any]] = None,
    phase5_mission_manual: Optional[Dict[str, Any]] = None,
    scene_waypoint_runtime: Optional[Dict[str, Any]] = None,
    language_memory_runtime: Optional[Dict[str, Any]] = None,
    person_evidence_runtime: Optional[Dict[str, Any]] = None,
    search_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    mission_payload = _as_dict(mission)
    search_payload = _as_dict(search_runtime)
    doorway_payload = _as_dict(doorway_runtime)
    vlm_payload = _as_dict(vlm_scene_runtime)
    reference_payload = _as_dict(reference_match_runtime)
    phase5_payload = _as_dict(phase5_mission_manual)
    scene_payload = _as_dict(scene_waypoint_runtime)
    language_payload = _as_dict(language_memory_runtime)
    evidence_payload = _as_dict(person_evidence_runtime)
    result_payload = _as_dict(search_result)

    mission_type = str(mission_payload.get("mission_type", "semantic_navigation") or "semantic_navigation")
    task_text = str(task_label or mission_payload.get("task_label", "")).strip()
    house_search_like = _truthy_task_for_house_search(task_text, mission_type)

    phase5_env = _as_dict(phase5_payload.get("environment_context"))
    best_doorway = _as_dict(doorway_payload.get("best_candidate"))

    scene_state = str(
        vlm_payload.get("scene_state", "")
        or scene_payload.get("scene_state", "")
        or ""
    )
    if not scene_state:
        scene_state = str(phase5_env.get("location_state", "unknown") or "unknown")
    location_state = scene_state or "unknown"

    doorway_candidate_count = int(doorway_payload.get("candidate_count", 0) or 0)
    traversable_doorway_count = int(doorway_payload.get("traversable_candidate_count", 0) or 0)
    entry_visible = bool(scene_payload.get("entry_door_visible", False) or doorway_candidate_count > 0)
    if bool(vlm_payload.get("entry_visible", False)):
        entry_visible = True
    entry_traversable = bool(
        vlm_payload.get("entry_traversable", False)
        or
        scene_payload.get("entry_door_traversable", False)
        or traversable_doorway_count > 0
        or bool(best_doorway.get("traversable", False))
    )

    evidence_status = str(evidence_payload.get("evidence_status", "idle") or "idle")
    result_status = str(result_payload.get("result_status", "unknown") or "unknown")
    scene_stage = str(scene_payload.get("active_stage", "scene_interpretation") or "scene_interpretation")
    visited_region_count = int(search_payload.get("visited_region_count", 0) or 0)

    target_house_reference_required = house_search_like and location_state in ("outside_house", "threshold_zone", "unknown")
    target_house_match_state = str(reference_payload.get("match_state", "unavailable") or "unavailable")
    target_house_match_confidence = float(reference_payload.get("match_confidence", 0.0) or 0.0)
    if target_house_reference_required:
        if target_house_match_state == "unavailable":
            target_house_match_state = "not_configured"

    stage_reason = ""
    active_stage_id = "outside_localization"
    recommended_next_goal = "stabilize_scene_understanding"
    recommended_control_mode = "planner_waypoint"

    if result_status in ("person_detected", "no_person_confirmed"):
        active_stage_id = "mission_report"
        stage_reason = "Mission already has a terminal search result."
        recommended_next_goal = "report_result"
        recommended_control_mode = "hold_and_report"
    elif evidence_status == "suspect" or mission_type == "target_verification":
        active_stage_id = "suspect_verification"
        stage_reason = "Active suspect evidence requires close verification."
        recommended_next_goal = "verify_suspect"
        recommended_control_mode = "planner_waypoint"
    elif scene_state == "outside_house":
        if target_house_match_state in ("pending", "not_configured", "matched", "candidate", "not_matched", "unknown"):
            if entry_visible and entry_traversable:
                active_stage_id = "approach_entry"
                stage_reason = "A traversable entry is visible while the UAV is still outside."
                recommended_next_goal = "approach_entry"
                if target_house_match_state == "matched":
                    stage_reason = "A traversable entry is visible and the current facade matches the target house."
            elif entry_visible:
                active_stage_id = "entry_search"
                stage_reason = "Door-like structures are visible but traversability is still uncertain."
                recommended_next_goal = "rank_entry_candidates"
            else:
                active_stage_id = "entry_search"
                stage_reason = "The UAV is outside and still needs to locate a valid entry."
                recommended_next_goal = "find_entry_door"
        else:
            active_stage_id = "target_house_verification"
            stage_reason = "The UAV is outside and should verify the target house before committing to an entry."
            recommended_next_goal = "verify_target_house"
    elif scene_state == "threshold_zone":
        if entry_traversable or scene_stage in ("cross_entry", "indoor_stabilize"):
            active_stage_id = "cross_entry"
            stage_reason = "The UAV is aligned with a threshold and should cross into the interior."
            recommended_next_goal = "cross_threshold"
            recommended_control_mode = "planner_waypoint"
        else:
            active_stage_id = "approach_entry"
            stage_reason = "The UAV is near an entry threshold but still needs alignment."
            recommended_next_goal = "refine_entry_alignment"
    elif scene_state == "inside_house":
        active_stage_id = "indoor_room_search"
        stage_reason = "The UAV is already inside and should perform structured room search."
        recommended_next_goal = "search_interior_rooms"
        recommended_control_mode = "planner_waypoint"
    else:
        active_stage_id = "outside_localization"
        stage_reason = "Scene localization is still ambiguous; collect more evidence before committing."
        recommended_next_goal = "localize_outside_vs_inside"
        recommended_control_mode = "planner_waypoint"

    waypoint_queue_labels = [
        str(item.get("label", ""))
        for item in _as_list(scene_payload.get("waypoint_queue"))
        if isinstance(item, dict) and str(item.get("label", ""))
    ]
    if not waypoint_queue_labels:
        waypoint_queue_labels = [
            str(item.get("label", ""))
            for item in _as_list(scene_payload.get("waypoints"))
            if isinstance(item, dict) and str(item.get("label", ""))
        ]

    focus_summary = str(language_payload.get("current_focus_summary", "") or "").strip()
    doorway_label = str(best_doorway.get("label", "") or doorway_payload.get("focus_label", "") or "")
    if not doorway_label:
        doorway_label = "entry_doorway" if entry_visible else ""
    summary_parts = [
        f"scene={scene_state or 'unknown'}",
        f"stage={active_stage_id}",
        f"entry={int(entry_visible)}/{int(entry_traversable)}",
        f"door={doorway_label or 'none'}",
        f"visited={visited_region_count}",
        f"goal={recommended_next_goal}",
    ]
    summary_parts.append(f"target={target_house_match_state}:{target_house_match_confidence:.2f}")
    if focus_summary:
        summary_parts.append(f"focus={focus_summary}")

    return build_phase6_mission_runtime_state(
        mission_id=str(mission_payload.get("mission_id", "")),
        mission_type=mission_type,
        task_label=task_text,
        scene_state=scene_state or "unknown",
        location_state=location_state or "unknown",
        active_stage_id=active_stage_id,
        active_stage_name=PHASE6_STAGE_NAMES.get(active_stage_id, active_stage_id.replace("_", " ").title()),
        stage_reason=stage_reason,
        target_house_match_state=target_house_match_state,
        target_house_match_confidence=target_house_match_confidence,
        target_house_reference_required=target_house_reference_required,
        entry_visible=entry_visible,
        entry_traversable=entry_traversable,
        entry_candidate_count=doorway_candidate_count,
        traversable_entry_count=traversable_doorway_count,
        entry_label=doorway_label,
        waypoint_queue_labels=waypoint_queue_labels,
        recommended_next_goal=recommended_next_goal,
        recommended_control_mode=recommended_control_mode,
        stage_queue=_build_stage_queue(active_stage_id),
        summary="; ".join(summary_parts),
    )
