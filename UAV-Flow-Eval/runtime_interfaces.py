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
    person_evidence_runtime: Optional[Dict[str, Any]] = None,
    search_result: Optional[Dict[str, Any]] = None,
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
        "person_evidence_runtime": person_evidence_runtime or {},
        "search_result": search_result or {},
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
