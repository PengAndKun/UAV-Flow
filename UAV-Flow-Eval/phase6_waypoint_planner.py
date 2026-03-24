"""
Phase 6 waypoint planner.

This module converts Phase 6 mission-stage understanding into a lightweight
queue of semantic waypoints. It is intentionally simple: it reuses the
existing scene-waypoint outputs when available, and augments them with
doorway- and target-house-aware waypoint hints for entry search.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from runtime_interfaces import build_phase6_waypoint_runtime_state


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _copy_waypoint(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    label = str(raw.get("label", "") or raw.get("name", "") or "").strip()
    if not label:
        label = "waypoint"
    waypoint = {
        "label": label,
        "x_hint": float(raw.get("x_hint", raw.get("center_x_norm", 0.5)) or 0.5),
        "y_hint": float(raw.get("y_hint", raw.get("center_y_norm", 0.5)) or 0.5),
        "z_offset_cm": float(raw.get("z_offset_cm", 0.0) or 0.0),
        "yaw_hint_deg": float(raw.get("yaw_hint_deg", 0.0) or 0.0),
        "reason": str(raw.get("reason", raw.get("rationale", "")) or ""),
        "source": str(raw.get("source", "scene_waypoint") or "scene_waypoint"),
    }
    confidence = raw.get("confidence")
    if confidence is not None:
        waypoint["confidence"] = float(confidence or 0.0)
    return waypoint


def _doorway_waypoints(best_doorway: Dict[str, Any], *, stage_id: str) -> List[Dict[str, Any]]:
    if not best_doorway:
        return []
    center_x = float(best_doorway.get("center_x_norm", 0.5) or 0.5)
    center_y = float(best_doorway.get("center_y_norm", 0.58) or 0.58)
    label = str(best_doorway.get("label", "entry_doorway") or "entry_doorway")
    traversable = bool(best_doorway.get("traversable", False))
    doorway_reason = str(best_doorway.get("rationale", "") or "")

    queue: List[Dict[str, Any]] = [
        {
            "label": "doorway_center_align",
            "x_hint": center_x,
            "y_hint": max(0.40, min(0.65, center_y - 0.05)),
            "z_offset_cm": 0.0,
            "yaw_hint_deg": 0.0,
            "reason": f"Align with {label}. {doorway_reason}".strip(),
            "source": "doorway_runtime",
            "confidence": float(best_doorway.get("confidence", 0.0) or 0.0),
        }
    ]
    if stage_id in ("approach_entry", "cross_entry") or traversable:
        queue.append(
            {
                "label": "doorway_threshold_pass",
                "x_hint": center_x,
                "y_hint": min(0.82, max(0.60, center_y + 0.08)),
                "z_offset_cm": 0.0,
                "yaw_hint_deg": 0.0,
                "reason": f"Pass through {label} threshold.",
                "source": "doorway_runtime",
                "confidence": float(best_doorway.get("confidence", 0.0) or 0.0),
            }
        )
        queue.append(
            {
                "label": "indoor_stabilize_after_entry",
                "x_hint": center_x,
                "y_hint": min(0.90, max(0.68, center_y + 0.16)),
                "z_offset_cm": 0.0,
                "yaw_hint_deg": 0.0,
                "reason": "Stabilize just inside the entry before indoor search.",
                "source": "doorway_runtime",
                "confidence": float(best_doorway.get("confidence", 0.0) or 0.0),
            }
        )
    return queue


def build_phase6_waypoint_runtime(
    *,
    task_label: str,
    phase6_mission_runtime: Optional[Dict[str, Any]] = None,
    scene_waypoint_runtime: Optional[Dict[str, Any]] = None,
    doorway_runtime: Optional[Dict[str, Any]] = None,
    reference_match_runtime: Optional[Dict[str, Any]] = None,
    semantic_archive_runtime: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    phase6_payload = _as_dict(phase6_mission_runtime)
    scene_payload = _as_dict(scene_waypoint_runtime)
    doorway_payload = _as_dict(doorway_runtime)
    reference_payload = _as_dict(reference_match_runtime)
    semantic_payload = _as_dict(semantic_archive_runtime)

    stage_id = str(phase6_payload.get("active_stage_id", "outside_localization") or "outside_localization")
    scene_state = str(phase6_payload.get("scene_state", "unknown") or "unknown")
    active_goal = str(phase6_payload.get("recommended_next_goal", "") or "")
    control_mode = str(phase6_payload.get("recommended_control_mode", "planner_waypoint") or "planner_waypoint")
    candidate_count = int(doorway_payload.get("candidate_count", 0) or 0)
    traversable_entry_count = int(doorway_payload.get("traversable_candidate_count", 0) or 0)
    match_state = str(reference_payload.get("match_state", "unavailable") or "unavailable")
    match_conf = float(reference_payload.get("match_confidence", 0.0) or 0.0)

    queue: List[Dict[str, Any]] = []
    for item in _as_list(scene_payload.get("waypoints")):
        normalized = _copy_waypoint(item)
        if normalized is not None:
            queue.append(normalized)

    best_doorway = _as_dict(doorway_payload.get("best_candidate"))
    doorway_queue = _doorway_waypoints(best_doorway, stage_id=stage_id)

    if stage_id in ("entry_search", "approach_entry", "cross_entry"):
        queue = doorway_queue + queue
    elif not queue and doorway_queue:
        queue = doorway_queue

    if stage_id == "target_house_verification" and queue:
        queue[0]["reason"] = (
            "Verify the facade/entry belongs to the target house before committing. "
            + str(queue[0].get("reason", "") or "")
        ).strip()

    if stage_id == "indoor_room_search" and not queue:
        queue.append(
            {
                "label": "indoor_forward_probe",
                "x_hint": 0.5,
                "y_hint": 0.68,
                "z_offset_cm": 0.0,
                "yaw_hint_deg": 0.0,
                "reason": "Probe deeper into the interior for structured room search.",
                "source": "phase6_waypoint_planner",
                "confidence": 0.45,
            }
        )

    semantic_matches = _as_list(semantic_payload.get("top_matches"))
    current_entry = _as_dict(semantic_payload.get("current_entry"))
    semantic_summary = str(
        current_entry.get("semantic_text", "")
        or current_entry.get("scene_description", "")
        or semantic_payload.get("summary", "")
        or ""
    ).strip()
    if semantic_summary and queue:
        queue[0]["reason"] = f"{queue[0].get('reason', '')} Archive hint: {semantic_summary}".strip()

    selected_waypoint = dict(queue[0]) if queue else {}
    status = "ok" if queue else "idle"
    summary = (
        f"stage={stage_id} scene={scene_state} goal={active_goal or 'none'} "
        f"queue={len(queue)} door_cand={candidate_count} traversable={traversable_entry_count} "
        f"target={match_state}:{match_conf:.2f} sem_matches={len(semantic_matches)}"
    )

    return build_phase6_waypoint_runtime_state(
        status=status,
        source="phase6_waypoint_planner",
        scene_state=scene_state,
        active_stage_id=stage_id,
        active_goal=active_goal,
        control_mode=control_mode,
        waypoint_queue=queue,
        selected_waypoint=selected_waypoint,
        candidate_count=candidate_count,
        traversable_entry_count=traversable_entry_count,
        target_house_match_state=match_state,
        target_house_match_confidence=match_conf,
        summary=summary,
    )
