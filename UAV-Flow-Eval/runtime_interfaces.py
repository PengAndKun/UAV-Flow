"""
Shared runtime schemas for the staged multimodal UAV control stack.

These helpers keep the Phase 0/1/2 server and panel payloads aligned so later
planner/archive/reflex modules can reuse the same shapes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def now_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


def build_sensor_pose(
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    yaw: float = 0.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    frame_id: str = "uav_body",
) -> Dict[str, Any]:
    """Build a consistent sensor pose dictionary."""
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll),
        "frame_id": frame_id,
    }


def build_pointcloud_packet(
    *,
    timestamp: Optional[str] = None,
    frame_id: str,
    sensor_pose: Optional[Dict[str, Any]] = None,
    pointcloud_points: Optional[List[List[float]]] = None,
    pointcloud_stats: Optional[Dict[str, Any]] = None,
    source_mode: str = "none",
    sequence_id: int = 0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a normalized point-cloud packet."""
    points = pointcloud_points or []
    stats = {
        "point_count": len(points),
        "width": len(points),
        "height": 1,
        "is_dense": True,
    }
    if pointcloud_stats:
        stats.update(pointcloud_stats)
    return {
        "timestamp": timestamp or now_timestamp(),
        "frame_id": frame_id,
        "sensor_pose": sensor_pose or build_sensor_pose(frame_id=frame_id),
        "pointcloud_points": points,
        "pointcloud_stats": stats,
        "source_mode": source_mode,
        "sequence_id": int(sequence_id),
        "metadata": metadata or {},
        # Legacy aliases kept for backward compatibility with earlier Phase 1 code.
        "radar_points": points,
        "range_bins": [],
    }


def build_waypoint(
    x: float,
    y: float,
    z: float,
    yaw: float,
    radius: float = 50.0,
    semantic_label: str = "",
) -> Dict[str, Any]:
    """Build a structured waypoint payload."""
    return {
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "yaw": float(yaw),
        "radius": float(radius),
        "semantic_label": semantic_label,
    }


def build_search_region(
    *,
    region_id: str = "",
    region_label: str = "",
    region_type: str = "area",
    room_type: str = "",
    priority: int = 0,
    status: str = "unobserved",
    rationale: str = "",
) -> Dict[str, Any]:
    """Build a normalized search-region descriptor for mission guidance."""
    return {
        "region_id": str(region_id or ""),
        "region_label": str(region_label or ""),
        "region_type": str(region_type or "area"),
        "room_type": str(room_type or ""),
        "priority": int(priority),
        "status": str(status or "unobserved"),
        "rationale": str(rationale or ""),
    }


def build_doorway_candidate(
    *,
    candidate_id: str = "",
    label: str = "",
    bbox: Optional[Dict[str, Any]] = None,
    center_x_norm: float = 0.0,
    center_y_norm: float = 0.0,
    width_ratio: float = 0.0,
    height_ratio: float = 0.0,
    opening_depth_cm: float = 0.0,
    surrounding_depth_cm: float = 0.0,
    clearance_depth_cm: float = 0.0,
    depth_gain_cm: float = 0.0,
    rgb_door_score: float = 0.0,
    depth_opening_score: float = 0.0,
    confidence: float = 0.0,
    traversable: bool = False,
    rationale: str = "",
) -> Dict[str, Any]:
    """Build a normalized doorway-candidate descriptor."""
    return {
        "candidate_id": str(candidate_id or ""),
        "label": str(label or ""),
        "bbox": bbox or {},
        "center_x_norm": float(center_x_norm),
        "center_y_norm": float(center_y_norm),
        "width_ratio": float(width_ratio),
        "height_ratio": float(height_ratio),
        "opening_depth_cm": float(opening_depth_cm),
        "surrounding_depth_cm": float(surrounding_depth_cm),
        "clearance_depth_cm": float(clearance_depth_cm),
        "depth_gain_cm": float(depth_gain_cm),
        "rgb_door_score": float(rgb_door_score),
        "depth_opening_score": float(depth_opening_score),
        "confidence": float(confidence),
        "traversable": bool(traversable),
        "rationale": str(rationale or ""),
    }


def build_doorway_runtime_state(
    *,
    frame_id: str = "",
    status: str = "idle",
    detector_name: str = "rgb_depth_doorway_heuristic",
    available: bool = False,
    candidate_count: int = 0,
    traversable_candidate_count: int = 0,
    best_candidate: Optional[Dict[str, Any]] = None,
    candidates: Optional[List[Dict[str, Any]]] = None,
    focus_label: str = "",
    summary: str = "",
    last_updated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a normalized doorway-runtime state."""
    return {
        "schema_version": "phase5.doorway_runtime.v1",
        "frame_id": str(frame_id or ""),
        "status": str(status or "idle"),
        "detector_name": str(detector_name or "rgb_depth_doorway_heuristic"),
        "available": bool(available),
        "candidate_count": int(candidate_count),
        "traversable_candidate_count": int(traversable_candidate_count),
        "best_candidate": best_candidate or {},
        "candidates": candidates or [],
        "focus_label": str(focus_label or ""),
        "summary": str(summary or ""),
        "last_updated_at": last_updated_at or now_timestamp(),
    }


def coerce_search_region_payload(raw: Any) -> Optional[Dict[str, Any]]:
    """Normalize an arbitrary region-like object into the shared search-region schema."""
    if not isinstance(raw, dict):
        return None
    return build_search_region(
        region_id=str(raw.get("region_id", raw.get("id", ""))),
        region_label=str(raw.get("region_label", raw.get("label", ""))),
        region_type=str(raw.get("region_type", "area")),
        room_type=str(raw.get("room_type", "")),
        priority=int(raw.get("priority", 0) or 0),
        status=str(raw.get("status", "unobserved")),
        rationale=str(raw.get("rationale", raw.get("reason", ""))),
    )


def build_mission_state(
    *,
    mission_id: str = "",
    created_at: Optional[str] = None,
    task_label: str = "",
    mission_text: str = "",
    mission_type: str = "semantic_navigation",
    target_type: str = "waypoint",
    search_scope: str = "local",
    priority_regions: Optional[List[Dict[str, Any]]] = None,
    confirm_target: bool = False,
    success_criteria: Optional[List[str]] = None,
    constraints: Optional[List[str]] = None,
    status: str = "idle",
) -> Dict[str, Any]:
    """Build a normalized mission descriptor for Phase 4 search-style guidance."""
    return {
        "schema_version": "phase4.mission.v1",
        "mission_id": str(mission_id or ""),
        "created_at": created_at or now_timestamp(),
        "task_label": str(task_label or ""),
        "mission_text": str(mission_text or task_label or ""),
        "mission_type": str(mission_type or "semantic_navigation"),
        "target_type": str(target_type or "waypoint"),
        "search_scope": str(search_scope or "local"),
        "priority_regions": priority_regions or [],
        "confirm_target": bool(confirm_target),
        "success_criteria": success_criteria or [],
        "constraints": constraints or [],
        "status": str(status or "idle"),
    }


def build_search_runtime_state(
    *,
    mission_id: str = "",
    mission_type: str = "semantic_navigation",
    mission_status: str = "idle",
    current_search_subgoal: str = "idle",
    priority_region: Optional[Dict[str, Any]] = None,
    candidate_regions: Optional[List[Dict[str, Any]]] = None,
    visited_region_count: int = 0,
    suspect_region_count: int = 0,
    confirmed_region_count: int = 0,
    evidence_count: int = 0,
    detection_state: str = "unknown",
    search_status: str = "",
    confirm_target: bool = False,
    estimated_person_position: Optional[Dict[str, Any]] = None,
    last_reasoning: str = "",
    replan_count: int = 0,
) -> Dict[str, Any]:
    """Build a normalized search-runtime state for mission-oriented evaluation."""
    return {
        "schema_version": "phase4.search_runtime.v1",
        "mission_id": str(mission_id or ""),
        "mission_type": str(mission_type or "semantic_navigation"),
        "mission_status": str(mission_status or "idle"),
        "current_search_subgoal": str(current_search_subgoal or "idle"),
        "priority_region": priority_region or {},
        "candidate_regions": candidate_regions or [],
        "visited_region_count": int(visited_region_count),
        "suspect_region_count": int(suspect_region_count),
        "confirmed_region_count": int(confirmed_region_count),
        "evidence_count": int(evidence_count),
        "detection_state": str(detection_state or "unknown"),
        "search_status": str(search_status or detection_state or "unknown"),
        "confirm_target": bool(confirm_target),
        "estimated_person_position": estimated_person_position or {},
        "last_reasoning": str(last_reasoning or ""),
        "replan_count": int(replan_count),
    }


def build_person_evidence_runtime_state(
    *,
    mission_id: str = "",
    mission_type: str = "semantic_navigation",
    evidence_status: str = "idle",
    suspect_count: int = 0,
    confirm_present_count: int = 0,
    confirm_absent_count: int = 0,
    evidence_event_count: int = 0,
    confidence: float = 0.0,
    suspect_region: Optional[Dict[str, Any]] = None,
    estimated_person_position: Optional[Dict[str, Any]] = None,
    evidence_capture_ids: Optional[List[str]] = None,
    last_event_type: str = "",
    last_reason: str = "",
    last_note: str = "",
    last_updated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a normalized runtime view of fused person-search evidence."""
    return {
        "schema_version": "phase4.person_evidence.v1",
        "mission_id": str(mission_id or ""),
        "mission_type": str(mission_type or "semantic_navigation"),
        "evidence_status": str(evidence_status or "idle"),
        "suspect_count": int(suspect_count),
        "confirm_present_count": int(confirm_present_count),
        "confirm_absent_count": int(confirm_absent_count),
        "evidence_event_count": int(evidence_event_count),
        "confidence": float(confidence),
        "suspect_region": suspect_region or {},
        "estimated_person_position": estimated_person_position or {},
        "evidence_capture_ids": evidence_capture_ids or [],
        "last_event_type": str(last_event_type or ""),
        "last_reason": str(last_reason or ""),
        "last_note": str(last_note or ""),
        "last_updated_at": last_updated_at or now_timestamp(),
    }


def build_search_result_state(
    *,
    mission_id: str = "",
    mission_type: str = "semantic_navigation",
    result_status: str = "unknown",
    person_exists: Optional[bool] = None,
    estimated_person_position: Optional[Dict[str, Any]] = None,
    confidence: float = 0.0,
    supporting_capture_ids: Optional[List[str]] = None,
    summary: str = "",
    last_updated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a normalized search-result state for person-search episodes."""
    return {
        "schema_version": "phase4.search_result.v1",
        "mission_id": str(mission_id or ""),
        "mission_type": str(mission_type or "semantic_navigation"),
        "result_status": str(result_status or "unknown"),
        "person_exists": person_exists,
        "estimated_person_position": estimated_person_position or {},
        "confidence": float(confidence),
        "supporting_capture_ids": supporting_capture_ids or [],
        "summary": str(summary or ""),
        "last_updated_at": last_updated_at or now_timestamp(),
    }


def build_language_search_memory_state(
    *,
    mission_id: str = "",
    mission_type: str = "semantic_navigation",
    task_label: str = "",
    global_summary: str = "",
    current_focus_region: Optional[Dict[str, Any]] = None,
    current_focus_summary: str = "",
    region_notes: Optional[List[Dict[str, Any]]] = None,
    recent_notes: Optional[List[Dict[str, Any]]] = None,
    note_count: int = 0,
    region_note_count: int = 0,
    last_updated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a normalized language-first search-memory runtime snapshot."""
    return {
        "schema_version": "phase4.language_search_memory.v1",
        "mission_id": str(mission_id or ""),
        "mission_type": str(mission_type or "semantic_navigation"),
        "task_label": str(task_label or ""),
        "global_summary": str(global_summary or ""),
        "current_focus_region": current_focus_region or {},
        "current_focus_summary": str(current_focus_summary or ""),
        "region_notes": region_notes or [],
        "recent_notes": recent_notes or [],
        "note_count": int(note_count),
        "region_note_count": int(region_note_count),
        "last_updated_at": last_updated_at or now_timestamp(),
    }


def build_planner_executor_runtime_state(
    *,
    mode: str = "manual",
    active: bool = False,
    state: str = "idle",
    run_id: str = "",
    mission_id: str = "",
    trigger: str = "",
    current_plan_id: str = "",
    current_search_subgoal: str = "idle",
    target_waypoint: Optional[Dict[str, Any]] = None,
    step_budget: int = 0,
    refresh_plan: bool = False,
    plan_refresh_interval_steps: int = 0,
    continuous_mode: bool = False,
    hold_retry_budget: int = 0,
    hold_retry_count: int = 0,
    steps_executed: int = 0,
    blocked_count: int = 0,
    replan_count: int = 0,
    last_action: str = "idle",
    last_progress_cm: float = 0.0,
    last_stop_reason: str = "",
    last_stop_detail: str = "",
    started_at: Optional[str] = None,
    updated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a normalized runtime state for planner-driven exploration execution."""
    return {
        "schema_version": "phase4.planner_executor.v1",
        "mode": str(mode or "manual"),
        "active": bool(active),
        "state": str(state or "idle"),
        "run_id": str(run_id or ""),
        "mission_id": str(mission_id or ""),
        "trigger": str(trigger or ""),
        "current_plan_id": str(current_plan_id or ""),
        "current_search_subgoal": str(current_search_subgoal or "idle"),
        "target_waypoint": target_waypoint or {},
        "step_budget": int(step_budget),
        "refresh_plan": bool(refresh_plan),
        "plan_refresh_interval_steps": int(plan_refresh_interval_steps),
        "continuous_mode": bool(continuous_mode),
        "hold_retry_budget": int(hold_retry_budget),
        "hold_retry_count": int(hold_retry_count),
        "steps_executed": int(steps_executed),
        "blocked_count": int(blocked_count),
        "replan_count": int(replan_count),
        "last_action": str(last_action or "idle"),
        "last_progress_cm": float(last_progress_cm),
        "last_stop_reason": str(last_stop_reason or ""),
        "last_stop_detail": str(last_stop_detail or ""),
        "started_at": started_at or now_timestamp(),
        "updated_at": updated_at or now_timestamp(),
    }


def build_llm_action_runtime_state(
    *,
    action_id: str = "",
    mode: str = "llm_action_only",
    policy_name: str = "",
    source: str = "none",
    status: str = "idle",
    suggested_action: str = "hold",
    should_execute: bool = False,
    confidence: float = 0.0,
    rationale: str = "",
    stop_condition: str = "hold_position",
    should_request_plan: bool = False,
    last_trigger: str = "",
    last_latency_ms: float = 0.0,
    risk_score: float = 0.0,
    model_name: str = "",
    api_style: str = "",
    route_mode: str = "",
    usage: Optional[Dict[str, Any]] = None,
    attempt_count: int = 0,
    fallback_used: bool = False,
    fallback_reason: str = "",
    upstream_error: str = "",
    raw_text: str = "",
    parsed_payload: Optional[Dict[str, Any]] = None,
    system_prompt_excerpt: str = "",
    user_prompt_excerpt: str = "",
    updated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a normalized runtime state for pure LLM action prediction."""
    return {
        "schema_version": "phase4.llm_action.v1",
        "action_id": str(action_id or ""),
        "mode": str(mode or "llm_action_only"),
        "policy_name": str(policy_name or ""),
        "source": str(source or "none"),
        "status": str(status or "idle"),
        "suggested_action": str(suggested_action or "hold"),
        "should_execute": bool(should_execute),
        "confidence": float(confidence),
        "policy_confidence": float(confidence),
        "rationale": str(rationale or ""),
        "stop_condition": str(stop_condition or "hold_position"),
        "should_request_plan": bool(should_request_plan),
        "last_trigger": str(last_trigger or ""),
        "last_latency_ms": float(last_latency_ms),
        "risk_score": float(risk_score),
        "model_name": str(model_name or ""),
        "api_style": str(api_style or ""),
        "route_mode": str(route_mode or ""),
        "usage": usage or {},
        "attempt_count": int(attempt_count),
        "fallback_used": bool(fallback_used),
        "fallback_reason": str(fallback_reason or ""),
        "upstream_error": str(upstream_error or ""),
        "raw_text": str(raw_text or ""),
        "parsed_payload": parsed_payload or {},
        "system_prompt_excerpt": str(system_prompt_excerpt or ""),
        "user_prompt_excerpt": str(user_prompt_excerpt or ""),
        "updated_at": updated_at or now_timestamp(),
    }


def build_scene_waypoint_runtime_state(
    *,
    request_id: str = "",
    mode: str = "scene_waypoint_only",
    policy_name: str = "",
    source: str = "none",
    status: str = "idle",
    scene_state: str = "unknown",
    active_stage: str = "scene_interpretation",
    entry_door_visible: bool = False,
    entry_door_traversable: bool = False,
    planner_confidence: float = 0.0,
    should_request_plan: bool = False,
    reasoning: str = "",
    waypoints: Optional[List[Dict[str, Any]]] = None,
    last_trigger: str = "",
    last_latency_ms: float = 0.0,
    model_name: str = "",
    api_style: str = "",
    route_mode: str = "",
    usage: Optional[Dict[str, Any]] = None,
    attempt_count: int = 0,
    fallback_used: bool = False,
    fallback_reason: str = "",
    upstream_error: str = "",
    raw_text: str = "",
    parsed_payload: Optional[Dict[str, Any]] = None,
    system_prompt_excerpt: str = "",
    user_prompt_excerpt: str = "",
    updated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a normalized runtime state for multimodal scene interpretation and waypoint generation."""
    return {
        "schema_version": "phase5.scene_waypoint_runtime.v1",
        "request_id": str(request_id or ""),
        "mode": str(mode or "scene_waypoint_only"),
        "policy_name": str(policy_name or ""),
        "source": str(source or "none"),
        "status": str(status or "idle"),
        "scene_state": str(scene_state or "unknown"),
        "active_stage": str(active_stage or "scene_interpretation"),
        "entry_door_visible": bool(entry_door_visible),
        "entry_door_traversable": bool(entry_door_traversable),
        "planner_confidence": float(planner_confidence),
        "should_request_plan": bool(should_request_plan),
        "reasoning": str(reasoning or ""),
        "waypoints": waypoints or [],
        "last_trigger": str(last_trigger or ""),
        "last_latency_ms": float(last_latency_ms),
        "model_name": str(model_name or ""),
        "api_style": str(api_style or ""),
        "route_mode": str(route_mode or ""),
        "usage": usage or {},
        "attempt_count": int(attempt_count),
        "fallback_used": bool(fallback_used),
        "fallback_reason": str(fallback_reason or ""),
        "upstream_error": str(upstream_error or ""),
        "raw_text": str(raw_text or ""),
        "parsed_payload": parsed_payload or {},
        "system_prompt_excerpt": str(system_prompt_excerpt or ""),
        "user_prompt_excerpt": str(user_prompt_excerpt or ""),
        "updated_at": updated_at or now_timestamp(),
    }


def build_phase6_mission_runtime_state(
    *,
    mission_id: str = "",
    mission_type: str = "semantic_navigation",
    task_label: str = "",
    scene_state: str = "unknown",
    location_state: str = "unknown",
    active_stage_id: str = "outside_localization",
    active_stage_name: str = "Outside Localization",
    stage_reason: str = "",
    target_house_match_state: str = "unavailable",
    target_house_match_confidence: float = 0.0,
    target_house_reference_required: bool = False,
    entry_visible: bool = False,
    entry_traversable: bool = False,
    entry_candidate_count: int = 0,
    traversable_entry_count: int = 0,
    entry_label: str = "",
    waypoint_queue_labels: Optional[List[str]] = None,
    recommended_next_goal: str = "",
    recommended_control_mode: str = "planner_waypoint",
    stage_queue: Optional[List[str]] = None,
    summary: str = "",
    updated_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a normalized Phase 6 mission-runtime state."""
    return {
        "schema_version": "phase6.mission_runtime.v1",
        "mission_id": str(mission_id or ""),
        "mission_type": str(mission_type or "semantic_navigation"),
        "task_label": str(task_label or ""),
        "scene_state": str(scene_state or "unknown"),
        "location_state": str(location_state or "unknown"),
        "active_stage_id": str(active_stage_id or "outside_localization"),
        "active_stage_name": str(active_stage_name or "Outside Localization"),
        "stage_reason": str(stage_reason or ""),
        "target_house_match_state": str(target_house_match_state or "unavailable"),
        "target_house_match_confidence": float(target_house_match_confidence),
        "target_house_reference_required": bool(target_house_reference_required),
        "entry_visible": bool(entry_visible),
        "entry_traversable": bool(entry_traversable),
        "entry_candidate_count": int(entry_candidate_count),
        "traversable_entry_count": int(traversable_entry_count),
        "entry_label": str(entry_label or ""),
        "waypoint_queue_labels": [str(item) for item in (waypoint_queue_labels or [])],
        "recommended_next_goal": str(recommended_next_goal or ""),
        "recommended_control_mode": str(recommended_control_mode or "planner_waypoint"),
        "stage_queue": [str(item) for item in (stage_queue or [])],
        "summary": str(summary or ""),
        "updated_at": updated_at or now_timestamp(),
    }


def coerce_llm_action_runtime_payload(
    raw: Any,
    *,
    default_policy_name: str = "",
    default_source: str = "external",
) -> Dict[str, Any]:
    """Normalize arbitrary LLM action output into the shared runtime schema."""
    payload = raw if isinstance(raw, dict) else {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    parsed_payload = payload.get("parsed_payload") if isinstance(payload.get("parsed_payload"), dict) else {}
    return build_llm_action_runtime_state(
        action_id=str(payload.get("action_id", "")),
        mode=str(payload.get("mode", "llm_action_only")),
        policy_name=str(payload.get("policy_name", default_policy_name)),
        source=str(payload.get("source", default_source)),
        status=str(payload.get("status", "idle")),
        suggested_action=str(payload.get("suggested_action", payload.get("action", "hold"))),
        should_execute=bool(payload.get("should_execute", False)),
        confidence=float(payload.get("confidence", payload.get("policy_confidence", 0.0)) or 0.0),
        rationale=str(payload.get("rationale", "")),
        stop_condition=str(payload.get("stop_condition", "hold_position")),
        should_request_plan=bool(payload.get("should_request_plan", False)),
        last_trigger=str(payload.get("last_trigger", "")),
        last_latency_ms=float(payload.get("last_latency_ms", payload.get("latency_ms", 0.0)) or 0.0),
        risk_score=float(payload.get("risk_score", 0.0) or 0.0),
        model_name=str(payload.get("model_name", "")),
        api_style=str(payload.get("api_style", "")),
        route_mode=str(payload.get("route_mode", "")),
        usage=usage,
        attempt_count=int(payload.get("attempt_count", 0) or 0),
        fallback_used=bool(payload.get("fallback_used", False)),
        fallback_reason=str(payload.get("fallback_reason", "")),
        upstream_error=str(payload.get("upstream_error", "")),
        raw_text=str(payload.get("raw_text", "")),
        parsed_payload=parsed_payload,
        system_prompt_excerpt=str(payload.get("system_prompt_excerpt", "")),
        user_prompt_excerpt=str(payload.get("user_prompt_excerpt", "")),
        updated_at=str(payload.get("updated_at", now_timestamp())),
    )


def coerce_scene_waypoint_runtime_payload(
    raw: Any,
    *,
    default_policy_name: str = "",
    default_source: str = "external",
) -> Dict[str, Any]:
    """Normalize arbitrary multimodal scene-waypoint output into the shared runtime schema."""
    payload = raw if isinstance(raw, dict) else {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    parsed_payload = payload.get("parsed_payload") if isinstance(payload.get("parsed_payload"), dict) else {}
    return build_scene_waypoint_runtime_state(
        request_id=str(payload.get("request_id", "")),
        mode=str(payload.get("mode", "scene_waypoint_only")),
        policy_name=str(payload.get("policy_name", default_policy_name)),
        source=str(payload.get("source", default_source)),
        status=str(payload.get("status", "idle")),
        scene_state=str(payload.get("scene_state", "unknown")),
        active_stage=str(payload.get("active_stage", "scene_interpretation")),
        entry_door_visible=bool(payload.get("entry_door_visible", False)),
        entry_door_traversable=bool(payload.get("entry_door_traversable", False)),
        planner_confidence=float(payload.get("planner_confidence", 0.0) or 0.0),
        should_request_plan=bool(payload.get("should_request_plan", False)),
        reasoning=str(payload.get("reasoning", "")),
        waypoints=payload.get("waypoints") if isinstance(payload.get("waypoints"), list) else [],
        last_trigger=str(payload.get("last_trigger", "")),
        last_latency_ms=float(payload.get("last_latency_ms", payload.get("latency_ms", 0.0)) or 0.0),
        model_name=str(payload.get("model_name", "")),
        api_style=str(payload.get("api_style", "")),
        route_mode=str(payload.get("route_mode", "")),
        usage=usage,
        attempt_count=int(payload.get("attempt_count", 0) or 0),
        fallback_used=bool(payload.get("fallback_used", False)),
        fallback_reason=str(payload.get("fallback_reason", "")),
        upstream_error=str(payload.get("upstream_error", "")),
        raw_text=str(payload.get("raw_text", "")),
        parsed_payload=parsed_payload,
        system_prompt_excerpt=str(payload.get("system_prompt_excerpt", "")),
        user_prompt_excerpt=str(payload.get("user_prompt_excerpt", "")),
        updated_at=str(payload.get("updated_at", now_timestamp())),
    )


def coerce_waypoint_payload(raw: Any, *, default_radius: float = 50.0) -> Optional[Dict[str, Any]]:
    """Normalize an arbitrary waypoint-like object into the shared waypoint schema."""
    if not isinstance(raw, dict):
        return None
    return build_waypoint(
        raw.get("x", 0.0),
        raw.get("y", 0.0),
        raw.get("z", 0.0),
        raw.get("yaw", 0.0),
        raw.get("radius", default_radius),
        str(raw.get("semantic_label", "")),
    )


def build_plan_state(
    *,
    plan_id: str = "",
    planner_name: str = "manual",
    generated_at: Optional[str] = None,
    sector_id: Optional[int] = None,
    candidate_waypoints: Optional[List[Dict[str, Any]]] = None,
    semantic_subgoal: str = "idle",
    planner_confidence: float = 0.0,
    should_replan: bool = False,
    mission_type: str = "semantic_navigation",
    search_subgoal: str = "",
    priority_region: Optional[Dict[str, Any]] = None,
    candidate_regions: Optional[List[Dict[str, Any]]] = None,
    confirm_target: bool = False,
    explanation: str = "",
    debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a normalized high-level planner state."""
    return {
        "plan_id": plan_id,
        "planner_name": planner_name,
        "schema_version": "phase2.plan.v1",
        "generated_at": generated_at or now_timestamp(),
        "sector_id": sector_id,
        "candidate_waypoints": candidate_waypoints or [],
        "semantic_subgoal": semantic_subgoal,
        "planner_confidence": float(planner_confidence),
        "should_replan": bool(should_replan),
        "mission_type": str(mission_type or "semantic_navigation"),
        "search_subgoal": str(search_subgoal or semantic_subgoal or "idle"),
        "priority_region": priority_region or {},
        "candidate_regions": candidate_regions or [],
        "confirm_target": bool(confirm_target),
        "explanation": str(explanation or ""),
        "debug": debug or {},
    }


def build_plan_request(
    *,
    task_label: str,
    instruction: str,
    frame_id: str,
    timestamp: str,
    pose: Dict[str, Any],
    depth: Optional[Dict[str, Any]] = None,
    camera_info: Optional[Dict[str, Any]] = None,
    image_b64: str = "",
    planner_name: str = "",
    trigger: str = "manual",
    step_index: int = 0,
    mission: Optional[Dict[str, Any]] = None,
    search_runtime: Optional[Dict[str, Any]] = None,
    doorway_runtime: Optional[Dict[str, Any]] = None,
    phase5_mission_manual: Optional[Dict[str, Any]] = None,
    phase6_mission_runtime: Optional[Dict[str, Any]] = None,
    person_evidence_runtime: Optional[Dict[str, Any]] = None,
    search_result: Optional[Dict[str, Any]] = None,
    language_memory_runtime: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the shared Phase 2 planner request payload."""
    return {
        "schema_version": "phase2.plan_request.v1",
        "planner_name": planner_name,
        "task_label": task_label,
        "instruction": instruction,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "trigger": trigger,
        "step_index": int(step_index),
        "pose": pose,
        "depth": depth or {},
        "camera_info": camera_info or {},
        "image_b64": image_b64,
        "mission": mission or {},
        "search_runtime": search_runtime or {},
        "doorway_runtime": doorway_runtime or {},
        "phase5_mission_manual": phase5_mission_manual or {},
        "phase6_mission_runtime": phase6_mission_runtime or {},
        "person_evidence_runtime": person_evidence_runtime or {},
        "search_result": search_result or {},
        "language_memory_runtime": language_memory_runtime or {},
        "context": context or {},
    }


def build_llm_action_request(
    *,
    task_label: str,
    instruction: str,
    frame_id: str,
    timestamp: str,
    pose: Dict[str, Any],
    depth: Optional[Dict[str, Any]] = None,
    camera_info: Optional[Dict[str, Any]] = None,
    image_b64: str = "",
    planner_name: str = "",
    trigger: str = "manual",
    step_index: int = 0,
    mission: Optional[Dict[str, Any]] = None,
    search_runtime: Optional[Dict[str, Any]] = None,
    doorway_runtime: Optional[Dict[str, Any]] = None,
    phase5_mission_manual: Optional[Dict[str, Any]] = None,
    phase6_mission_runtime: Optional[Dict[str, Any]] = None,
    person_evidence_runtime: Optional[Dict[str, Any]] = None,
    search_result: Optional[Dict[str, Any]] = None,
    language_memory_runtime: Optional[Dict[str, Any]] = None,
    current_plan: Optional[Dict[str, Any]] = None,
    reflex_runtime: Optional[Dict[str, Any]] = None,
    runtime_debug: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the shared Phase 4 pure-LLM action request payload."""
    return {
        "schema_version": "phase4.llm_action_request.v1",
        "planner_name": planner_name,
        "task_label": task_label,
        "instruction": instruction,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "trigger": trigger,
        "step_index": int(step_index),
        "pose": pose,
        "depth": depth or {},
        "camera_info": camera_info or {},
        "image_b64": image_b64,
        "mission": mission or {},
        "search_runtime": search_runtime or {},
        "doorway_runtime": doorway_runtime or {},
        "phase5_mission_manual": phase5_mission_manual or {},
        "phase6_mission_runtime": phase6_mission_runtime or {},
        "person_evidence_runtime": person_evidence_runtime or {},
        "search_result": search_result or {},
        "language_memory_runtime": language_memory_runtime or {},
        "current_plan": current_plan or {},
        "reflex_runtime": reflex_runtime or {},
        "runtime_debug": runtime_debug or {},
        "context": context or {},
    }


def build_scene_waypoint_request(
    *,
    task_label: str,
    instruction: str,
    frame_id: str,
    timestamp: str,
    pose: Dict[str, Any],
    depth: Optional[Dict[str, Any]] = None,
    camera_info: Optional[Dict[str, Any]] = None,
    image_b64: str = "",
    planner_name: str = "",
    trigger: str = "manual",
    step_index: int = 0,
    mission: Optional[Dict[str, Any]] = None,
    search_runtime: Optional[Dict[str, Any]] = None,
    doorway_runtime: Optional[Dict[str, Any]] = None,
    phase5_mission_manual: Optional[Dict[str, Any]] = None,
    phase6_mission_runtime: Optional[Dict[str, Any]] = None,
    person_evidence_runtime: Optional[Dict[str, Any]] = None,
    search_result: Optional[Dict[str, Any]] = None,
    language_memory_runtime: Optional[Dict[str, Any]] = None,
    current_plan: Optional[Dict[str, Any]] = None,
    runtime_debug: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the shared Phase 5 scene-waypoint request payload."""
    return {
        "schema_version": "phase5.scene_waypoint_request.v1",
        "planner_name": planner_name,
        "task_label": task_label,
        "instruction": instruction,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "trigger": trigger,
        "step_index": int(step_index),
        "pose": pose,
        "depth": depth or {},
        "camera_info": camera_info or {},
        "image_b64": image_b64,
        "mission": mission or {},
        "search_runtime": search_runtime or {},
        "doorway_runtime": doorway_runtime or {},
        "phase5_mission_manual": phase5_mission_manual or {},
        "phase6_mission_runtime": phase6_mission_runtime or {},
        "person_evidence_runtime": person_evidence_runtime or {},
        "search_result": search_result or {},
        "language_memory_runtime": language_memory_runtime or {},
        "current_plan": current_plan or {},
        "runtime_debug": runtime_debug or {},
        "context": context or {},
    }


def coerce_plan_payload(
    raw: Any,
    *,
    default_plan_id: str,
    default_planner_name: str,
    default_semantic_subgoal: str = "idle",
    default_radius: float = 50.0,
) -> Dict[str, Any]:
    """Normalize arbitrary planner output into the shared plan schema."""
    payload = raw if isinstance(raw, dict) else {}
    candidate_waypoints: List[Dict[str, Any]] = []
    for waypoint in payload.get("candidate_waypoints") or []:
        normalized = coerce_waypoint_payload(waypoint, default_radius=default_radius)
        if normalized is not None:
            candidate_waypoints.append(normalized)
    candidate_regions: List[Dict[str, Any]] = []
    for region in payload.get("candidate_regions") or []:
        normalized_region = coerce_search_region_payload(region)
        if normalized_region is not None:
            candidate_regions.append(normalized_region)
    priority_region = coerce_search_region_payload(payload.get("priority_region")) or {}
    return build_plan_state(
        plan_id=str(payload.get("plan_id", default_plan_id)),
        planner_name=str(payload.get("planner_name", default_planner_name)),
        generated_at=str(payload.get("generated_at", now_timestamp())),
        sector_id=payload.get("sector_id"),
        candidate_waypoints=candidate_waypoints,
        semantic_subgoal=str(payload.get("semantic_subgoal", default_semantic_subgoal)),
        planner_confidence=float(payload.get("planner_confidence", 0.0)),
        should_replan=bool(payload.get("should_replan", False)),
        mission_type=str(payload.get("mission_type", "semantic_navigation")),
        search_subgoal=str(payload.get("search_subgoal", payload.get("semantic_subgoal", default_semantic_subgoal))),
        priority_region=priority_region,
        candidate_regions=candidate_regions,
        confirm_target=bool(payload.get("confirm_target", False)),
        explanation=str(payload.get("explanation", "")),
        debug=payload.get("debug") if isinstance(payload.get("debug"), dict) else {},
    )


def build_runtime_debug_state(
    *,
    current_waypoint: Optional[Dict[str, Any]] = None,
    local_policy_action: Optional[Dict[str, Any]] = None,
    risk_score: float = 0.0,
    shield_triggered: bool = False,
    archive_cell_id: str = "",
) -> Dict[str, Any]:
    """Build placeholders for later Phase 3/4 runtime signals."""
    return {
        "current_waypoint": current_waypoint,
        "local_policy_action": local_policy_action or {},
        "risk_score": float(risk_score),
        "shield_triggered": bool(shield_triggered),
        "archive_cell_id": archive_cell_id,
    }


def build_reflex_runtime_state(
    *,
    mode: str = "heuristic_stub",
    policy_name: str = "",
    source: str = "local_heuristic",
    status: str = "idle",
    suggested_action: str = "idle",
    should_execute: bool = False,
    last_trigger: str = "",
    last_latency_ms: float = 0.0,
    waypoint_distance_cm: float = 0.0,
    yaw_error_deg: float = 0.0,
    vertical_error_cm: float = 0.0,
    progress_to_waypoint_cm: float = 0.0,
    retrieval_cell_id: str = "",
    retrieval_score: float = 0.0,
    retrieval_semantic_subgoal: str = "",
    risk_score: float = 0.0,
    shield_triggered: bool = False,
    policy_confidence: float = 0.0,
    prototype_distance: float = 0.0,
    model_type: str = "",
) -> Dict[str, Any]:
    """Build a normalized Phase 3 reflex runtime state."""
    return {
        "mode": mode,
        "policy_name": str(policy_name or ""),
        "source": str(source or "local_heuristic"),
        "status": status,
        "suggested_action": suggested_action,
        "should_execute": bool(should_execute),
        "last_trigger": str(last_trigger or ""),
        "last_latency_ms": float(last_latency_ms),
        "waypoint_distance_cm": float(waypoint_distance_cm),
        "yaw_error_deg": float(yaw_error_deg),
        "vertical_error_cm": float(vertical_error_cm),
        "progress_to_waypoint_cm": float(progress_to_waypoint_cm),
        "retrieval_cell_id": str(retrieval_cell_id or ""),
        "retrieval_score": float(retrieval_score),
        "retrieval_semantic_subgoal": str(retrieval_semantic_subgoal or ""),
        "risk_score": float(risk_score),
        "shield_triggered": bool(shield_triggered),
        "policy_confidence": float(policy_confidence),
        "confidence": float(policy_confidence),
        "prototype_distance": float(prototype_distance),
        "model_type": str(model_type or ""),
    }


def build_reflex_request(
    *,
    policy_name: str,
    frame_id: str,
    timestamp: str,
    task_label: str,
    pose: Dict[str, Any],
    depth: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
    current_waypoint: Optional[Dict[str, Any]] = None,
    archive: Optional[Dict[str, Any]] = None,
    runtime_debug: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the shared Phase 3 reflex policy request payload."""
    return {
        "schema_version": "phase3.reflex_request.v1",
        "policy_name": policy_name,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "task_label": task_label,
        "pose": pose,
        "depth": depth or {},
        "plan": plan or {},
        "current_waypoint": current_waypoint or {},
        "archive": archive or {},
        "runtime_debug": runtime_debug or {},
        "context": context or {},
    }


def coerce_reflex_runtime_payload(
    raw: Any,
    *,
    default_mode: str = "heuristic_stub",
    default_policy_name: str = "",
    default_source: str = "local_heuristic",
) -> Dict[str, Any]:
    """Normalize arbitrary reflex policy output into the shared runtime schema."""
    payload = raw if isinstance(raw, dict) else {}
    return build_reflex_runtime_state(
        mode=str(payload.get("mode", default_mode)),
        policy_name=str(payload.get("policy_name", default_policy_name)),
        source=str(payload.get("source", default_source)),
        status=str(payload.get("status", "idle")),
        suggested_action=str(payload.get("suggested_action", "idle")),
        should_execute=bool(payload.get("should_execute", False)),
        last_trigger=str(payload.get("last_trigger", "")),
        last_latency_ms=float(payload.get("last_latency_ms", 0.0)),
        waypoint_distance_cm=float(payload.get("waypoint_distance_cm", 0.0)),
        yaw_error_deg=float(payload.get("yaw_error_deg", 0.0)),
        vertical_error_cm=float(payload.get("vertical_error_cm", 0.0)),
        progress_to_waypoint_cm=float(payload.get("progress_to_waypoint_cm", 0.0)),
        retrieval_cell_id=str(payload.get("retrieval_cell_id", "")),
        retrieval_score=float(payload.get("retrieval_score", 0.0)),
        retrieval_semantic_subgoal=str(payload.get("retrieval_semantic_subgoal", "")),
        risk_score=float(payload.get("risk_score", 0.0)),
        shield_triggered=bool(payload.get("shield_triggered", False)),
        policy_confidence=float(payload.get("policy_confidence", 0.0)),
        prototype_distance=float(payload.get("prototype_distance", 0.0)),
        model_type=str(payload.get("model_type", "")),
    )


def build_reflex_sample(
    *,
    capture_id: str,
    task_label: str,
    action_label: str,
    pose: Dict[str, Any],
    current_waypoint: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
    archive: Optional[Dict[str, Any]] = None,
    reflex_runtime: Optional[Dict[str, Any]] = None,
    runtime_debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a compact reflex-training/replay sample from runtime metadata."""
    reflex = reflex_runtime or {}
    debug = runtime_debug or {}
    plan_payload = plan or {}
    archive_payload = archive or {}
    active_retrieval = archive_payload.get("active_retrieval") if isinstance(archive_payload.get("active_retrieval"), dict) else {}
    return {
        "schema_version": "phase3.reflex_sample.v1",
        "capture_id": capture_id,
        "task_label": task_label,
        "executed_action": action_label,
        "pose": pose,
        "target_waypoint": current_waypoint if isinstance(current_waypoint, dict) else {},
        "planner_name": str(plan_payload.get("planner_name", "")),
        "planner_subgoal": str(plan_payload.get("semantic_subgoal", "")),
        "archive_cell_id": str(debug.get("archive_cell_id", archive_payload.get("current_cell_id", ""))),
        "retrieval_cell_id": str(reflex.get("retrieval_cell_id", active_retrieval.get("cell_id", ""))),
        "retrieval_score": float(reflex.get("retrieval_score", active_retrieval.get("retrieval_score", 0.0) or 0.0)),
        "retrieval_semantic_subgoal": str(
            reflex.get("retrieval_semantic_subgoal", active_retrieval.get("semantic_subgoal", ""))
        ),
        "policy_mode": str(reflex.get("mode", "")),
        "policy_name": str(reflex.get("policy_name", "")),
        "policy_source": str(reflex.get("source", "")),
        "policy_confidence": float(reflex.get("policy_confidence", 0.0)),
        "suggested_action": str(reflex.get("suggested_action", "idle")),
        "should_execute": bool(reflex.get("should_execute", False)),
        "waypoint_distance_cm": float(reflex.get("waypoint_distance_cm", 0.0)),
        "yaw_error_deg": float(reflex.get("yaw_error_deg", 0.0)),
        "vertical_error_cm": float(reflex.get("vertical_error_cm", 0.0)),
        "progress_to_waypoint_cm": float(reflex.get("progress_to_waypoint_cm", 0.0)),
        "risk_score": float(reflex.get("risk_score", debug.get("risk_score", 0.0))),
        "shield_triggered": bool(reflex.get("shield_triggered", debug.get("shield_triggered", False))),
        "local_policy_action": debug.get("local_policy_action", {}),
        "current_waypoint": debug.get("current_waypoint", current_waypoint if isinstance(current_waypoint, dict) else {}),
    }


def build_capture_bundle(
    *,
    capture_id: str,
    capture_time: str,
    env_id: str,
    task_label: str,
    action_label: str,
    rgb_image_path: str,
    depth_image_path: Optional[str] = None,
    depth_preview_path: Optional[str] = None,
    camera_info_path: Optional[str] = None,
    pointcloud_packet_path: str,
    metadata_path: str,
    pose: Dict[str, Any],
    depth: Optional[Dict[str, Any]] = None,
    camera_info: Optional[Dict[str, Any]] = None,
    pointcloud: Dict[str, Any],
    plan: Dict[str, Any],
    runtime_debug: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a normalized multimodal capture bundle."""
    return {
        "capture_id": capture_id,
        "capture_time": capture_time,
        "env_id": env_id,
        "task_label": task_label,
        "action_label": action_label,
        "rgb_image_path": rgb_image_path,
        "depth_image_path": depth_image_path,
        "depth_preview_path": depth_preview_path,
        "camera_info_path": camera_info_path,
        "pointcloud_packet_path": pointcloud_packet_path,
        "metadata_path": metadata_path,
        "pose": pose,
        "depth": depth or {},
        "camera_info": camera_info or {},
        "pointcloud": pointcloud,
        "plan": plan,
        "runtime_debug": runtime_debug,
        # Legacy aliases kept for backward compatibility with earlier Phase 1 code.
        "radar_packet_path": pointcloud_packet_path,
        "radar": pointcloud,
    }


def build_radar_packet(**kwargs: Any) -> Dict[str, Any]:
    """Backward-compatible alias for older radar terminology."""
    if "radar_points" in kwargs and "pointcloud_points" not in kwargs:
        kwargs["pointcloud_points"] = kwargs.pop("radar_points")
    if "range_bins" in kwargs:
        kwargs.pop("range_bins")
    return build_pointcloud_packet(**kwargs)
