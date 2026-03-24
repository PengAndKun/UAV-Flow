"""
Lightweight VLM-style scene descriptor for Phase 6.

This is intentionally a local heuristic bootstrapper rather than a full VLM.
It generates human-readable scene text plus structured tags so the runtime can
already behave like a semantic archive pipeline while we wire in stronger
models later.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from runtime_interfaces import build_vlm_scene_runtime_state


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def _task_mentions_house(task_label: str) -> bool:
    text = str(task_label or "").strip().lower()
    return any(token in text for token in ("house", "home", "building", "room", "person", "people"))


def _estimate_brightness(rgb_frame: Optional[np.ndarray]) -> float:
    if rgb_frame is None or not isinstance(rgb_frame, np.ndarray) or rgb_frame.size == 0:
        return 0.0
    gray = np.mean(rgb_frame.astype(np.float32), axis=2)
    return float(np.mean(gray) / 255.0)


def _estimate_occlusion_tags(rgb_frame: Optional[np.ndarray]) -> List[str]:
    if rgb_frame is None or not isinstance(rgb_frame, np.ndarray) or rgb_frame.size == 0:
        return []
    h, w = rgb_frame.shape[:2]
    third = max(1, w // 3)
    left = float(np.mean(rgb_frame[:, :third]))
    center = float(np.mean(rgb_frame[:, third : 2 * third]))
    right = float(np.mean(rgb_frame[:, 2 * third :]))
    tags: List[str] = []
    if left + 8 < center:
        tags.append("left_side_more_occluded")
    if right + 8 < center:
        tags.append("right_side_more_occluded")
    if center + 8 < min(left, right):
        tags.append("center_corridor_like_opening")
    return tags


def _summarize_direction_from_doorway(best_candidate: Dict[str, Any]) -> str:
    center_x = float(best_candidate.get("center_x_norm", 0.5) or 0.5)
    if center_x < 0.4:
        return "left"
    if center_x > 0.6:
        return "right"
    return "forward"


def build_vlm_scene_runtime(
    *,
    task_label: str,
    mission: Optional[Dict[str, Any]] = None,
    search_runtime: Optional[Dict[str, Any]] = None,
    doorway_runtime: Optional[Dict[str, Any]] = None,
    scene_waypoint_runtime: Optional[Dict[str, Any]] = None,
    phase5_mission_manual: Optional[Dict[str, Any]] = None,
    phase6_mission_runtime: Optional[Dict[str, Any]] = None,
    person_evidence_runtime: Optional[Dict[str, Any]] = None,
    search_result: Optional[Dict[str, Any]] = None,
    reference_match_runtime: Optional[Dict[str, Any]] = None,
    current_plan: Optional[Dict[str, Any]] = None,
    depth_summary: Optional[Dict[str, Any]] = None,
    rgb_frame: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    mission_payload = _as_dict(mission)
    search_payload = _as_dict(search_runtime)
    doorway_payload = _as_dict(doorway_runtime)
    scene_payload = _as_dict(scene_waypoint_runtime)
    phase5_payload = _as_dict(phase5_mission_manual)
    phase6_payload = _as_dict(phase6_mission_runtime)
    evidence_payload = _as_dict(person_evidence_runtime)
    result_payload = _as_dict(search_result)
    reference_payload = _as_dict(reference_match_runtime)
    plan_payload = _as_dict(current_plan)
    depth_payload = _as_dict(depth_summary)

    best_door = _as_dict(doorway_payload.get("best_candidate"))
    phase5_env = _as_dict(phase5_payload.get("environment_context"))

    scene_state = str(scene_payload.get("scene_state", "") or phase6_payload.get("scene_state", "") or phase5_env.get("location_state", "unknown") or "unknown")
    entry_visible = bool(
        scene_payload.get("entry_door_visible", False)
        or phase6_payload.get("entry_visible", False)
        or int(doorway_payload.get("candidate_count", 0) or 0) > 0
    )
    entry_traversable = bool(
        scene_payload.get("entry_door_traversable", False)
        or phase6_payload.get("entry_traversable", False)
        or int(doorway_payload.get("traversable_candidate_count", 0) or 0) > 0
        or bool(best_door.get("traversable", False))
    )
    room_type_hint = str(search_payload.get("priority_region", {}).get("room_type", "") if isinstance(search_payload.get("priority_region"), dict) else "")
    if not room_type_hint and scene_state == "inside_house":
        room_type_hint = "interior_space"

    brightness = _estimate_brightness(rgb_frame)
    occlusion_tags = _estimate_occlusion_tags(rgb_frame)
    scene_tags: List[str] = []
    if _task_mentions_house(task_label):
        scene_tags.append("house_search_task")
    if scene_state:
        scene_tags.append(scene_state)
    if entry_visible:
        scene_tags.append("entry_visible")
    if entry_traversable:
        scene_tags.append("entry_traversable")
    if room_type_hint:
        scene_tags.append(f"room_hint:{room_type_hint}")
    scene_tags.extend(occlusion_tags)
    match_state = str(reference_payload.get("match_state", "unknown") or "unknown")
    if match_state and match_state not in ("unknown", "unavailable"):
        scene_tags.append(f"target_match:{match_state}")

    unexplored_direction = "forward"
    if entry_visible and best_door:
        unexplored_direction = _summarize_direction_from_doorway(best_door)
    elif "left_side_more_occluded" in occlusion_tags and "right_side_more_occluded" not in occlusion_tags:
        unexplored_direction = "right"
    elif "right_side_more_occluded" in occlusion_tags and "left_side_more_occluded" not in occlusion_tags:
        unexplored_direction = "left"

    evidence_status = str(evidence_payload.get("evidence_status", "idle") or "idle")
    result_status = str(result_payload.get("result_status", "unknown") or "unknown")
    semantic_subgoal = str(plan_payload.get("semantic_subgoal", "") or search_payload.get("current_search_subgoal", "") or "idle")
    depth_min = float(depth_payload.get("min_depth", 0.0) or 0.0)
    depth_max = float(depth_payload.get("max_depth", 0.0) or 0.0)

    description_parts: List[str] = []
    if scene_state == "outside_house":
        description_parts.append("The UAV appears outside the house facade.")
    elif scene_state == "inside_house":
        description_parts.append("The UAV appears inside the house interior.")
    elif scene_state == "threshold_zone":
        description_parts.append("The UAV is aligned near a doorway threshold.")
    else:
        description_parts.append("The UAV scene is ambiguous and still being localized.")

    if entry_visible and entry_traversable:
        description_parts.append("A traversable entry doorway is visible ahead.")
    elif entry_visible:
        description_parts.append("A door-like opening is visible but traversability is still uncertain.")
    else:
        description_parts.append("No reliable entry opening is confirmed yet.")

    if room_type_hint:
        description_parts.append(f"Current room hint: {room_type_hint}.")
    if semantic_subgoal and semantic_subgoal != "idle":
        description_parts.append(f"Current search subgoal is {semantic_subgoal}.")
    if evidence_status not in ("idle", "searching"):
        description_parts.append(f"Evidence state is {evidence_status}.")
    if result_status not in ("unknown", "not_applicable"):
        description_parts.append(f"Mission result state is {result_status}.")
    if match_state not in ("unknown", "unavailable"):
        description_parts.append(f"Target house match state is {match_state}.")
    if depth_max > 0.0:
        description_parts.append(f"Depth spans approximately {depth_min:.0f}cm to {depth_max:.0f}cm.")
    if occlusion_tags:
        description_parts.append(f"Occlusion cues: {', '.join(occlusion_tags)}.")

    scene_description = " ".join(description_parts)
    semantic_text = (
        f"scene={scene_state}; entry_visible={int(entry_visible)}; entry_traversable={int(entry_traversable)}; "
        f"goal={semantic_subgoal}; direction={unexplored_direction}; match={match_state}; "
        f"evidence={evidence_status}; result={result_status}"
    )
    occlusion_summary = ", ".join(occlusion_tags) if occlusion_tags else "no strong occlusion cue"
    confidence = 0.35
    confidence += 0.2 if scene_state in ("outside_house", "inside_house", "threshold_zone") else 0.0
    confidence += 0.2 if entry_visible else 0.0
    confidence += 0.15 if entry_traversable else 0.0
    confidence += 0.05 if brightness > 0.1 else 0.0

    return build_vlm_scene_runtime_state(
        mission_id=str(mission_payload.get("mission_id", "")),
        mission_type=str(mission_payload.get("mission_type", "semantic_navigation") or "semantic_navigation"),
        task_label=str(task_label or ""),
        status="ok",
        source="local_vlm_heuristic",
        model_name="vlm_scene_descriptor_v0",
        scene_description=scene_description,
        semantic_text=semantic_text,
        scene_tags=scene_tags,
        scene_state=scene_state or "unknown",
        entry_visible=entry_visible,
        entry_traversable=entry_traversable,
        room_type_hint=room_type_hint,
        occlusion_summary=occlusion_summary,
        unexplored_direction=unexplored_direction,
        confidence=_clamp(confidence, 0.0, 1.0),
    )
