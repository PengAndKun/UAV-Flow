"""
Prompt building and structured response parsing for the LLM planner layer.

This adapter keeps the LLM responsible for high-level search intent while
reusing heuristic seed waypoints for low-level geometric stability.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, List, Optional

from llm_planner_client import LLMPlannerClient
from runtime_interfaces import build_plan_state, build_search_region, build_waypoint, now_timestamp


class LLMPlannerAdapterError(RuntimeError):
    """Raised when the LLM planner adapter cannot produce a valid plan."""


ALLOWED_MISSION_TYPES = {
    "semantic_navigation",
    "person_search",
    "room_search",
    "target_verification",
}

ALLOWED_SEARCH_SUBGOALS = {
    "idle",
    "search_house",
    "search_room",
    "search_frontier",
    "find_entry_door",
    "approach_entry_door",
    "traverse_entry_door",
    "advance_to_waypoint",
    "approach_suspect_region",
    "confirm_suspect_region",
    "revisit_suspect_region",
    "reorient_for_navigation",
    "ascend_and_observe",
    "descend_and_observe",
}

ALLOWED_WAYPOINT_STRATEGIES = {
    "use_seed_waypoints",
    "shorter_approach",
    "broader_sweep",
    "align_with_entry",
    "pass_through_opening",
    "reorient_left",
    "reorient_right",
    "ascend_and_observe",
    "descend_and_observe",
}

LLM_PLAN_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "mission_type": {"type": "string", "enum": sorted(ALLOWED_MISSION_TYPES)},
        "search_subgoal": {"type": "string", "enum": sorted(ALLOWED_SEARCH_SUBGOALS)},
        "priority_region_label": {"type": "string"},
        "candidate_region_labels": {
            "type": "array",
            "items": {"type": "string"},
        },
        "confirm_target": {"type": "boolean"},
        "should_replan": {"type": "boolean"},
        "planner_confidence": {"type": "number"},
        "semantic_subgoal": {"type": "string"},
        "waypoint_strategy": {"type": "string", "enum": sorted(ALLOWED_WAYPOINT_STRATEGIES)},
        "explanation": {"type": "string"},
    },
    "required": [
        "mission_type",
        "search_subgoal",
        "priority_region_label",
        "candidate_region_labels",
        "confirm_target",
        "should_replan",
        "planner_confidence",
        "semantic_subgoal",
        "waypoint_strategy",
        "explanation",
    ],
    "additionalProperties": False,
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _normalize_angle_deg(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def _strip_fenced_json(text: str) -> str:
    raw = str(text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
            raw = "\n".join(lines[1:-1]).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    return raw


def _sanitize_json_candidate(text: str) -> str:
    raw = str(text or "").replace("\ufeff", "").strip()
    if not raw:
        return raw
    parts: List[str] = []
    in_string = False
    escaped = False
    for char in raw:
        if escaped:
            parts.append(char)
            escaped = False
            continue
        if char == "\\":
            parts.append(char)
            escaped = True
            continue
        if char == '"':
            parts.append(char)
            in_string = not in_string
            continue
        if in_string:
            if char == "\n":
                parts.append("\\n")
                continue
            if char == "\r":
                parts.append("\\r")
                continue
            if char == "\t":
                parts.append("\\t")
                continue
        parts.append(char)
    sanitized = "".join(parts)
    sanitized = re.sub(r",\s*([}\]])", r"\1", sanitized)
    if sanitized.count('"') % 2 == 1:
        sanitized += '"'
    bracket_delta = sanitized.count("[") - sanitized.count("]")
    if bracket_delta > 0:
        sanitized += "]" * bracket_delta
    brace_delta = sanitized.count("{") - sanitized.count("}")
    if brace_delta > 0:
        sanitized += "}" * brace_delta
    return sanitized


def _extract_partial_llm_fields(text: str) -> Dict[str, Any]:
    raw = str(text or "")
    extracted: Dict[str, Any] = {}

    def capture_string(key: str) -> Optional[str]:
        patterns = [
            rf'"{re.escape(key)}"\s*:\s*"([^"\r\n}}]*)"',
            rf'"{re.escape(key)}"\s*:\s*"([^"\r\n]*)',
        ]
        for pattern in patterns:
            match = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
            if match:
                return str(match.group(1)).strip()
        return None

    def capture_bool(key: str) -> Optional[bool]:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*(true|false)', raw, flags=re.IGNORECASE)
        if match:
            return str(match.group(1)).lower() == "true"
        return None

    def capture_number(key: str) -> Optional[float]:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*(-?\d+(?:\.\d+)?)', raw, flags=re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
        return None

    def capture_string_array(key: str) -> List[str]:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*\[(.*?)\]', raw, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return []
        values = re.findall(r'"([^"\r\n]+)"', match.group(1))
        return [str(value).strip() for value in values if str(value).strip()]

    for key in ("mission_type", "search_subgoal", "priority_region_label", "semantic_subgoal", "waypoint_strategy", "explanation"):
        value = capture_string(key)
        if value:
            extracted[key] = value
    for key in ("confirm_target", "should_replan"):
        value = capture_bool(key)
        if value is not None:
            extracted[key] = value
    confidence = capture_number("planner_confidence")
    if confidence is not None:
        extracted["planner_confidence"] = confidence
    labels = capture_string_array("candidate_region_labels")
    if labels:
        extracted["candidate_region_labels"] = labels
    return extracted


def parse_llm_json_response(text: str) -> Dict[str, Any]:
    candidate = _strip_fenced_json(text)
    if not candidate:
        raise LLMPlannerAdapterError("LLM planner returned empty text.")
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        repaired_candidate = _sanitize_json_candidate(candidate)
        if repaired_candidate != candidate:
            try:
                payload = json.loads(repaired_candidate)
            except json.JSONDecodeError:
                payload = {}
        else:
            payload = {}
        if not payload:
            payload = _extract_partial_llm_fields(candidate)
        if not payload:
            raise LLMPlannerAdapterError(f"LLM planner returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMPlannerAdapterError("LLM planner JSON response must be an object.")
    return payload


def _summarize_region(region: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "region_label": str(region.get("region_label", "")),
        "region_type": str(region.get("region_type", "")),
        "room_type": str(region.get("room_type", "")),
        "priority": int(region.get("priority", 0) or 0),
        "status": str(region.get("status", "")),
        "rationale": str(region.get("rationale", "")),
    }


def _build_region_candidates(request_payload: Dict[str, Any], heuristic_seed: Dict[str, Any]) -> List[Dict[str, Any]]:
    mission = request_payload.get("mission") if isinstance(request_payload.get("mission"), dict) else {}
    search_runtime = request_payload.get("search_runtime") if isinstance(request_payload.get("search_runtime"), dict) else {}
    source_lists = [
        heuristic_seed.get("candidate_regions"),
        search_runtime.get("candidate_regions"),
        mission.get("priority_regions"),
    ]
    regions: List[Dict[str, Any]] = []
    for source in source_lists:
        if not isinstance(source, list):
            continue
        for item in source:
            if not isinstance(item, dict):
                continue
            label = str(item.get("region_label", "")).strip().lower()
            if label and all(str(region.get("region_label", "")).strip().lower() != label for region in regions):
                regions.append(item)
    return regions


def _compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"))


def _truncate_debug_text(text: str, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


def build_llm_planner_prompt(
    request_payload: Dict[str, Any],
    heuristic_seed: Dict[str, Any],
) -> Dict[str, str]:
    task_label = str(request_payload.get("task_label", "")).strip() or "idle"
    mission = request_payload.get("mission") if isinstance(request_payload.get("mission"), dict) else {}
    search_runtime = request_payload.get("search_runtime") if isinstance(request_payload.get("search_runtime"), dict) else {}
    person_evidence_runtime = (
        request_payload.get("person_evidence_runtime")
        if isinstance(request_payload.get("person_evidence_runtime"), dict)
        else {}
    )
    search_result = request_payload.get("search_result") if isinstance(request_payload.get("search_result"), dict) else {}
    language_memory_runtime = (
        request_payload.get("language_memory_runtime")
        if isinstance(request_payload.get("language_memory_runtime"), dict)
        else {}
    )
    doorway_runtime = request_payload.get("doorway_runtime") if isinstance(request_payload.get("doorway_runtime"), dict) else {}
    phase5_mission_manual = (
        request_payload.get("phase5_mission_manual")
        if isinstance(request_payload.get("phase5_mission_manual"), dict)
        else {}
    )
    pose = request_payload.get("pose") if isinstance(request_payload.get("pose"), dict) else {}
    depth = request_payload.get("depth") if isinstance(request_payload.get("depth"), dict) else {}
    context = request_payload.get("context") if isinstance(request_payload.get("context"), dict) else {}
    archive_context = context.get("archive") if isinstance(context.get("archive"), dict) else {}
    reflex_runtime = context.get("reflex_runtime") if isinstance(context.get("reflex_runtime"), dict) else {}
    available_regions = [_summarize_region(region) for region in _build_region_candidates(request_payload, heuristic_seed)]
    heuristic_summary = {
        "mission_type": heuristic_seed.get("mission_type", "semantic_navigation"),
        "search_subgoal": heuristic_seed.get("search_subgoal", heuristic_seed.get("semantic_subgoal", "idle")),
        "priority_region": _summarize_region(heuristic_seed.get("priority_region", {}) if isinstance(heuristic_seed.get("priority_region"), dict) else {}),
        "candidate_regions": available_regions,
        "planner_confidence": float(heuristic_seed.get("planner_confidence", 0.0)),
        "semantic_subgoal": str(heuristic_seed.get("semantic_subgoal", "idle")),
    }
    pose_summary = {
        "x": round(float(pose.get("x", 0.0)), 1),
        "y": round(float(pose.get("y", 0.0)), 1),
        "z": round(float(pose.get("z", 0.0)), 1),
        "yaw": round(float(pose.get("yaw", 0.0)), 1),
    }
    depth_summary = {
        "min_depth_cm": round(float(depth.get("min_depth", 0.0)), 1),
        "max_depth_cm": round(float(depth.get("max_depth", 0.0)), 1),
    }
    search_summary = {
        "mission_status": str(search_runtime.get("mission_status", "idle")),
        "current_search_subgoal": str(search_runtime.get("current_search_subgoal", "idle")),
        "detection_state": str(search_runtime.get("detection_state", "unknown")),
        "visited_region_count": int(search_runtime.get("visited_region_count", 0)),
        "suspect_region_count": int(search_runtime.get("suspect_region_count", 0)),
        "confirmed_region_count": int(search_runtime.get("confirmed_region_count", 0)),
        "evidence_count": int(search_runtime.get("evidence_count", 0)),
    }
    archive_summary = {
        "current_cell_id": str(archive_context.get("current_cell_id", "")),
        "recent_cell_ids": archive_context.get("recent_cell_ids", [])[:6] if isinstance(archive_context.get("recent_cell_ids"), list) else [],
        "top_cells": archive_context.get("top_cells", [])[:3] if isinstance(archive_context.get("top_cells"), list) else [],
    }
    reflex_summary = {
        "mode": str(reflex_runtime.get("mode", "")),
        "suggested_action": str(reflex_runtime.get("suggested_action", "")),
        "policy_confidence": float(reflex_runtime.get("policy_confidence", 0.0)),
        "risk_score": float(context.get("risk_score", 0.0) or 0.0),
    }
    evidence_summary = {
        "evidence_status": str(person_evidence_runtime.get("evidence_status", "idle")),
        "suspect_count": int(person_evidence_runtime.get("suspect_count", 0)),
        "confirm_present_count": int(person_evidence_runtime.get("confirm_present_count", 0)),
        "confirm_absent_count": int(person_evidence_runtime.get("confirm_absent_count", 0)),
        "confidence": float(person_evidence_runtime.get("confidence", 0.0)),
        "last_event_type": str(person_evidence_runtime.get("last_event_type", "")),
        "last_reason": str(person_evidence_runtime.get("last_reason", "")),
        "suspect_region": _summarize_region(
            person_evidence_runtime.get("suspect_region", {})
            if isinstance(person_evidence_runtime.get("suspect_region"), dict)
            else {}
        ),
    }
    search_result_summary = {
        "result_status": str(search_result.get("result_status", "unknown")),
        "person_exists": search_result.get("person_exists"),
        "confidence": float(search_result.get("confidence", 0.0)),
        "estimated_person_position": search_result.get("estimated_person_position", {})
        if isinstance(search_result.get("estimated_person_position"), dict)
        else {},
        "summary": str(search_result.get("summary", "")),
    }
    language_memory_summary = {
        "global_summary": str(language_memory_runtime.get("global_summary", "")),
        "current_focus_region": _summarize_region(
            language_memory_runtime.get("current_focus_region", {})
            if isinstance(language_memory_runtime.get("current_focus_region"), dict)
            else {}
        ),
        "current_focus_summary": str(language_memory_runtime.get("current_focus_summary", "")),
        "note_count": int(language_memory_runtime.get("note_count", 0)),
        "region_note_count": int(language_memory_runtime.get("region_note_count", 0)),
        "recent_notes": [
            {
                "note_type": str(note.get("note_type", "")),
                "region_label": str(note.get("region_label", "")),
                "text": str(note.get("text", "")),
            }
            for note in (language_memory_runtime.get("recent_notes") or [])[:4]
            if isinstance(note, dict)
        ],
        "region_notes": [
            {
                "region_label": str(note.get("region_label", "")),
                "status": str(note.get("status", "")),
                "summary": str(note.get("summary", "")),
            }
            for note in (language_memory_runtime.get("region_notes") or [])[:4]
            if isinstance(note, dict)
        ],
    }
    best_doorway = doorway_runtime.get("best_candidate", {}) if isinstance(doorway_runtime.get("best_candidate"), dict) else {}
    doorway_summary = {
        "status": str(doorway_runtime.get("status", "idle")),
        "candidate_count": int(doorway_runtime.get("candidate_count", 0)),
        "traversable_candidate_count": int(doorway_runtime.get("traversable_candidate_count", 0)),
        "summary": str(doorway_runtime.get("summary", "")),
        "best_candidate": {
            "label": str(best_doorway.get("label", "")),
            "traversable": bool(best_doorway.get("traversable", False)),
            "confidence": float(best_doorway.get("confidence", 0.0)),
            "depth_gain_cm": float(best_doorway.get("depth_gain_cm", 0.0)),
            "clearance_depth_cm": float(best_doorway.get("clearance_depth_cm", 0.0)),
            "width_ratio": float(best_doorway.get("width_ratio", 0.0)),
            "height_ratio": float(best_doorway.get("height_ratio", 0.0)),
        },
    }
    phase5_environment = (
        phase5_mission_manual.get("environment_context", {})
        if isinstance(phase5_mission_manual.get("environment_context"), dict)
        else {}
    )
    phase5_stages = phase5_mission_manual.get("stages", []) if isinstance(phase5_mission_manual.get("stages"), list) else []
    active_stage_id = str(phase5_mission_manual.get("active_stage_id", "") or "")
    active_stage = next(
        (
            stage for stage in phase5_stages
            if isinstance(stage, dict) and str(stage.get("stage_id", "")) == active_stage_id
        ),
        {},
    )
    phase5_summary = {
        "active_stage_id": active_stage_id,
        "active_stage_name": str(active_stage.get("stage_name", "")),
        "active_objective": str(active_stage.get("objective", "")),
        "planner_focus": str(active_stage.get("planner_focus", "")),
        "location_state": str(phase5_environment.get("location_state", "unknown")),
        "inside_score": int(phase5_environment.get("inside_score", 0)),
        "outside_score": int(phase5_environment.get("outside_score", 0)),
        "doorway_candidate_count": int(phase5_environment.get("doorway_candidate_count", 0)),
        "rationale": [
            str(item)
            for item in (phase5_environment.get("rationale") or [])[:3]
        ],
    }

    system_prompt = (
        "You are a high-level UAV house-search planner. "
        "You must output one JSON object only. "
        "Do not output markdown, code fences, or explanations outside JSON. "
        "You are not allowed to output low-level motor commands. "
        "Use the heuristic seed plan as a geometric fallback, but improve mission/search intent. "
        "Choose only from these mission types: semantic_navigation, person_search, room_search, target_verification. "
        "Choose only from these search_subgoal values: "
        "search_house, search_room, search_frontier, find_entry_door, approach_entry_door, traverse_entry_door, advance_to_waypoint, approach_suspect_region, "
        "confirm_suspect_region, revisit_suspect_region, reorient_for_navigation, ascend_and_observe, descend_and_observe. "
        "Choose only from these waypoint_strategy values: use_seed_waypoints, shorter_approach, broader_sweep, "
        "align_with_entry, pass_through_opening, reorient_left, reorient_right, ascend_and_observe, descend_and_observe. "
        "All string values must be single-line plain strings without embedded newlines."
    )
    user_prompt = (
        "Return JSON with keys: "
        "mission_type, search_subgoal, priority_region_label, candidate_region_labels, confirm_target, "
        "should_replan, planner_confidence, semantic_subgoal, waypoint_strategy, explanation.\n"
        f"task_label={task_label}\n"
        f"mission={_compact_json(mission)}\n"
        f"search_runtime={_compact_json(search_summary)}\n"
        f"pose={_compact_json(pose_summary)}\n"
        f"depth={_compact_json(depth_summary)}\n"
        f"archive={_compact_json(archive_summary)}\n"
        f"reflex={_compact_json(reflex_summary)}\n"
        f"person_evidence={_compact_json(evidence_summary)}\n"
        f"search_result={_compact_json(search_result_summary)}\n"
        f"language_memory={_compact_json(language_memory_summary)}\n"
        f"doorway_runtime={_compact_json(doorway_summary)}\n"
        f"phase5_manual={_compact_json(phase5_summary)}\n"
        f"heuristic_seed={_compact_json(heuristic_summary)}\n"
        "Rules:\n"
        "- prioritize person-search semantics over generic navigation semantics when the task mentions people, survivors, suspects, rooms, or verification.\n"
        "- follow the active Phase 5 stage when it is available; treat it as the current mission manual.\n"
        "- if the UAV appears to be outside and doorway_runtime reports a traversable entry candidate, prefer entry-oriented subgoals before generic interior search.\n"
        "- use find_entry_door / approach_entry_door / traverse_entry_door when doorway reasoning is central to progress.\n"
        "- prefer candidate_region_labels from the provided region list.\n"
        "- use language_memory to avoid revisiting already described regions unless verification or revisit is justified.\n"
        "- use waypoint_strategy to adjust the seed geometry instead of inventing raw motor actions.\n"
        "- keep planner_confidence between 0.0 and 1.0.\n"
        "- keep explanation short and single-line.\n"
    )
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "image_b64": str(request_payload.get("image_b64", "") or ""),
    }


def _normalize_region_label(label: str) -> str:
    return str(label or "").strip().lower().replace("_", " ")


def _select_regions_by_label(
    labels: List[str],
    available_regions: List[Dict[str, Any]],
    default_regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    chosen: List[Dict[str, Any]] = []
    normalized_lookup = {
        _normalize_region_label(str(region.get("region_label", ""))): region
        for region in available_regions
        if isinstance(region, dict) and str(region.get("region_label", "")).strip()
    }
    for label in labels:
        normalized = _normalize_region_label(label)
        if not normalized:
            continue
        region = normalized_lookup.get(normalized)
        if region is None:
            region = build_search_region(
                region_id=normalized.replace(" ", "_"),
                region_label=str(label).strip(),
                region_type="area",
                priority=max(1, 4 - len(chosen)),
                status="unobserved",
                rationale="Region synthesized from LLM planner output.",
            )
        if all(_normalize_region_label(str(item.get("region_label", ""))) != normalized for item in chosen):
            chosen.append(region)
    if chosen:
        return chosen
    return default_regions


def _apply_waypoint_strategy(
    *,
    seed_waypoints: List[Dict[str, Any]],
    current_pose: Dict[str, Any],
    strategy: str,
    waypoint_radius_cm: float,
    semantic_label: str,
) -> List[Dict[str, Any]]:
    if not seed_waypoints:
        current_x = float(current_pose.get("x", 0.0))
        current_y = float(current_pose.get("y", 0.0))
        current_z = float(current_pose.get("z", 0.0))
        current_yaw = float(current_pose.get("yaw", 0.0))
        distance_cm = 200.0
        target_yaw = current_yaw
        if strategy == "reorient_left":
            target_yaw = _normalize_angle_deg(current_yaw - 45.0)
        elif strategy == "reorient_right":
            target_yaw = _normalize_angle_deg(current_yaw + 45.0)
        theta = math.radians(target_yaw)
        return [
            build_waypoint(
                x=current_x + distance_cm * math.cos(theta),
                y=current_y + distance_cm * math.sin(theta),
                z=current_z,
                yaw=target_yaw,
                radius=waypoint_radius_cm,
                semantic_label=semantic_label,
            )
        ]

    first_seed = seed_waypoints[0] if isinstance(seed_waypoints[0], dict) else {}
    current_x = float(current_pose.get("x", 0.0))
    current_y = float(current_pose.get("y", 0.0))
    current_z = float(current_pose.get("z", 0.0))
    base_dx = float(first_seed.get("x", current_x)) - current_x
    base_dy = float(first_seed.get("y", current_y)) - current_y
    base_dz = float(first_seed.get("z", current_z)) - current_z
    base_yaw = float(first_seed.get("yaw", current_pose.get("yaw", 0.0)))
    scale = 1.0
    yaw_delta = 0.0
    z_delta = 0.0
    if strategy == "shorter_approach":
        scale = 0.6
    elif strategy == "broader_sweep":
        scale = 1.25
    elif strategy == "reorient_left":
        yaw_delta = -45.0
    elif strategy == "reorient_right":
        yaw_delta = 45.0
    elif strategy == "ascend_and_observe":
        z_delta = 60.0
    elif strategy == "descend_and_observe":
        z_delta = -40.0

    if yaw_delta != 0.0:
        radius = math.hypot(base_dx, base_dy)
        source_yaw = math.degrees(math.atan2(base_dy, base_dx)) if radius > 1e-6 else float(current_pose.get("yaw", 0.0))
        rotated_yaw = _normalize_angle_deg(source_yaw + yaw_delta)
        theta = math.radians(rotated_yaw)
        base_dx = radius * math.cos(theta)
        base_dy = radius * math.sin(theta)
        base_yaw = rotated_yaw

    waypoints: List[Dict[str, Any]] = []
    for index, seed in enumerate(seed_waypoints[:2]):
        if not isinstance(seed, dict):
            continue
        blend = 1.0 if index == 0 else 0.6
        x = current_x + base_dx * scale * blend
        y = current_y + base_dy * scale * blend
        z = current_z + base_dz * scale * blend + z_delta
        yaw = base_yaw
        label = semantic_label if index == 0 else "staging_waypoint"
        waypoints.append(
            build_waypoint(
                x=x,
                y=y,
                z=z,
                yaw=yaw,
                radius=float(seed.get("radius", waypoint_radius_cm) or waypoint_radius_cm),
                semantic_label=label,
            )
        )
    return waypoints or seed_waypoints


def build_llm_plan(
    *,
    request_payload: Dict[str, Any],
    heuristic_seed: Dict[str, Any],
    client: LLMPlannerClient,
    planner_name: str,
    waypoint_radius_cm: float,
) -> Dict[str, Any]:
    prompt = build_llm_planner_prompt(request_payload, heuristic_seed)
    response = client.generate(
        system_prompt=prompt["system_prompt"],
        user_prompt=prompt["user_prompt"],
        image_b64=prompt["image_b64"],
        json_schema=LLM_PLAN_RESPONSE_SCHEMA,
    )
    payload = parse_llm_json_response(response["text"])

    mission_type = str(payload.get("mission_type", heuristic_seed.get("mission_type", "semantic_navigation"))).strip()
    if mission_type not in ALLOWED_MISSION_TYPES:
        mission_type = str(heuristic_seed.get("mission_type", "semantic_navigation"))

    search_subgoal = str(payload.get("search_subgoal", heuristic_seed.get("search_subgoal", heuristic_seed.get("semantic_subgoal", "idle")))).strip()
    if search_subgoal not in ALLOWED_SEARCH_SUBGOALS:
        search_subgoal = str(heuristic_seed.get("search_subgoal", heuristic_seed.get("semantic_subgoal", "idle")))

    semantic_subgoal = str(payload.get("semantic_subgoal", heuristic_seed.get("semantic_subgoal", search_subgoal))).strip() or search_subgoal
    planner_confidence = _clamp(float(payload.get("planner_confidence", heuristic_seed.get("planner_confidence", 0.55)) or 0.55), 0.0, 1.0)
    should_replan = bool(payload.get("should_replan", False))
    confirm_target = bool(payload.get("confirm_target", heuristic_seed.get("confirm_target", False)))
    explanation = str(payload.get("explanation", "")).strip() or "LLM planner returned a structured search plan."
    waypoint_strategy = str(payload.get("waypoint_strategy", "use_seed_waypoints")).strip()
    if waypoint_strategy not in ALLOWED_WAYPOINT_STRATEGIES:
        waypoint_strategy = "use_seed_waypoints"

    available_regions = _build_region_candidates(request_payload, heuristic_seed)
    default_regions = heuristic_seed.get("candidate_regions") if isinstance(heuristic_seed.get("candidate_regions"), list) else []
    candidate_labels = payload.get("candidate_region_labels") if isinstance(payload.get("candidate_region_labels"), list) else []
    candidate_regions = _select_regions_by_label([str(label) for label in candidate_labels], available_regions, default_regions)
    priority_region_label = str(payload.get("priority_region_label", "")).strip()
    priority_region_candidates = _select_regions_by_label(
        [priority_region_label] if priority_region_label else [],
        available_regions,
        candidate_regions[:1] if candidate_regions else default_regions[:1],
    )
    priority_region = priority_region_candidates[0] if priority_region_candidates else {}
    if not candidate_regions and priority_region:
        candidate_regions = [priority_region]

    seed_waypoints = heuristic_seed.get("candidate_waypoints") if isinstance(heuristic_seed.get("candidate_waypoints"), list) else []
    pose = request_payload.get("pose") if isinstance(request_payload.get("pose"), dict) else {}
    candidate_waypoints = _apply_waypoint_strategy(
        seed_waypoints=seed_waypoints,
        current_pose=pose,
        strategy=waypoint_strategy,
        waypoint_radius_cm=waypoint_radius_cm,
        semantic_label=semantic_subgoal,
    )

    debug = {
        "source": "llm_planner",
        "api_style": response.get("api_style", ""),
        "endpoint_path": response.get("endpoint_path", ""),
        "model_name": response.get("model_name", ""),
        "usage": response.get("usage", {}),
        "latency_ms": float(response.get("latency_ms", 0.0)),
        "attempt_count": int(response.get("attempt_count", 1)),
        "waypoint_strategy": waypoint_strategy,
        "raw_text": response.get("text", ""),
        "response_text_preview": _truncate_debug_text(response.get("text", ""), 1200),
        "parsed_payload": payload,
        "system_prompt_excerpt": _truncate_debug_text(prompt["system_prompt"], 1200),
        "user_prompt_excerpt": _truncate_debug_text(prompt["user_prompt"], 2400),
    }
    return build_plan_state(
        plan_id=f"llm_plan_{request_payload.get('frame_id', 'unknown')}",
        planner_name=planner_name,
        generated_at=now_timestamp(),
        sector_id=heuristic_seed.get("sector_id"),
        candidate_waypoints=candidate_waypoints,
        semantic_subgoal=semantic_subgoal,
        planner_confidence=planner_confidence,
        should_replan=should_replan,
        mission_type=mission_type,
        search_subgoal=search_subgoal,
        priority_region=priority_region,
        candidate_regions=candidate_regions,
        confirm_target=confirm_target,
        explanation=explanation,
        debug=debug,
    )
