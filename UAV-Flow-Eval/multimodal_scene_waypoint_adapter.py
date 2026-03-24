"""
Multimodal scene interpretation and continuous waypoint prompting.

This adapter is the Phase 5 bridge from RGB + depth observations to:
- scene-state classification (outside/inside/threshold)
- staged mission progress (find entry, approach, cross entry, indoor search)
- a short sequence of normalized waypoint hints for downstream execution/training
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from llm_planner_client import LLMPlannerClient
from runtime_interfaces import build_scene_waypoint_runtime_state


class MultimodalSceneWaypointAdapterError(RuntimeError):
    """Raised when the multimodal scene-waypoint adapter cannot produce a valid result."""


ALLOWED_SCENE_STATES = {
    "outside_house",
    "inside_house",
    "threshold_zone",
    "unknown",
}

ALLOWED_STAGE_IDS = {
    "scene_interpretation",
    "find_entry_door",
    "approach_entry",
    "cross_entry",
    "indoor_stabilize",
    "house_search",
    "verify_target",
    "report_result",
}

SCENE_WAYPOINT_RESPONSE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "scene_state": {"type": "string", "enum": sorted(ALLOWED_SCENE_STATES)},
        "active_stage": {"type": "string", "enum": sorted(ALLOWED_STAGE_IDS)},
        "entry_door_visible": {"type": "boolean"},
        "entry_door_traversable": {"type": "boolean"},
        "planner_confidence": {"type": "number"},
        "should_request_plan": {"type": "boolean"},
        "reasoning": {"type": "string"},
        "waypoints": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "x_hint": {"type": "number"},
                    "y_hint": {"type": "number"},
                    "z_offset_cm": {"type": "number"},
                    "yaw_hint_deg": {"type": "number"},
                },
                "required": ["label", "x_hint", "y_hint", "z_offset_cm", "yaw_hint_deg"],
                "additionalProperties": False,
            },
            "minItems": 1,
            "maxItems": 5,
        },
    },
    "required": [
        "scene_state",
        "active_stage",
        "entry_door_visible",
        "entry_door_traversable",
        "planner_confidence",
        "should_request_plan",
        "reasoning",
        "waypoints",
    ],
    "additionalProperties": False,
}


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"))


def _truncate_debug_text(text: str, limit: int) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return f"{value[: limit - 3]}..."


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


def _extract_partial_fields(text: str) -> Dict[str, Any]:
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

    for key in ("scene_state", "active_stage", "reasoning"):
        value = capture_string(key)
        if value:
            extracted[key] = value
    for key in ("entry_door_visible", "entry_door_traversable", "should_request_plan"):
        value = capture_bool(key)
        if value is not None:
            extracted[key] = value
    confidence = capture_number("planner_confidence")
    if confidence is not None:
        extracted["planner_confidence"] = confidence
    return extracted


def _coerce_waypoints(raw_waypoints: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_waypoints, list):
        return []
    normalized: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_waypoints[:5]):
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "label": str(item.get("label", f"waypoint_{idx + 1}") or f"waypoint_{idx + 1}"),
                "x_hint": round(_clamp(float(item.get("x_hint", 0.5) or 0.5), 0.0, 1.0), 4),
                "y_hint": round(_clamp(float(item.get("y_hint", 0.5) or 0.5), 0.0, 1.0), 4),
                "z_offset_cm": round(float(item.get("z_offset_cm", 0.0) or 0.0), 1),
                "yaw_hint_deg": round(float(item.get("yaw_hint_deg", 0.0) or 0.0), 1),
            }
        )
    return normalized


def parse_scene_waypoint_json_response(text: str) -> Dict[str, Any]:
    candidate = _strip_fenced_json(text)
    if not candidate:
        raise MultimodalSceneWaypointAdapterError("Scene waypoint model returned empty text.")
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
            payload = _extract_partial_fields(candidate)
        if not payload:
            raise MultimodalSceneWaypointAdapterError(f"Scene waypoint model returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise MultimodalSceneWaypointAdapterError("Scene waypoint JSON response must be an object.")
    return payload


def _build_request_summaries(request_payload: Dict[str, Any]) -> Dict[str, Any]:
    mission = request_payload.get("mission") if isinstance(request_payload.get("mission"), dict) else {}
    search_runtime = request_payload.get("search_runtime") if isinstance(request_payload.get("search_runtime"), dict) else {}
    doorway_runtime = request_payload.get("doorway_runtime") if isinstance(request_payload.get("doorway_runtime"), dict) else {}
    phase5_manual = request_payload.get("phase5_mission_manual") if isinstance(request_payload.get("phase5_mission_manual"), dict) else {}
    language_memory_runtime = request_payload.get("language_memory_runtime") if isinstance(request_payload.get("language_memory_runtime"), dict) else {}
    search_result = request_payload.get("search_result") if isinstance(request_payload.get("search_result"), dict) else {}
    person_evidence_runtime = request_payload.get("person_evidence_runtime") if isinstance(request_payload.get("person_evidence_runtime"), dict) else {}
    current_plan = request_payload.get("current_plan") if isinstance(request_payload.get("current_plan"), dict) else {}
    pose = request_payload.get("pose") if isinstance(request_payload.get("pose"), dict) else {}
    depth = request_payload.get("depth") if isinstance(request_payload.get("depth"), dict) else {}
    runtime_debug = request_payload.get("runtime_debug") if isinstance(request_payload.get("runtime_debug"), dict) else {}
    context = request_payload.get("context") if isinstance(request_payload.get("context"), dict) else {}

    best_candidate = doorway_runtime.get("best_candidate") if isinstance(doorway_runtime.get("best_candidate"), dict) else {}
    phase5_environment = phase5_manual.get("environment_context") if isinstance(phase5_manual.get("environment_context"), dict) else {}
    active_stage_id = str(phase5_manual.get("active_stage_id", "") or "")
    stages = phase5_manual.get("stages") if isinstance(phase5_manual.get("stages"), list) else []
    active_stage = next(
        (stage for stage in stages if isinstance(stage, dict) and str(stage.get("stage_id", "")) == active_stage_id),
        {},
    )

    return {
        "task_label": str(request_payload.get("task_label", "") or "idle"),
        "mission": {
            "mission_type": str(mission.get("mission_type", "semantic_navigation")),
            "search_scope": str(mission.get("search_scope", "local")),
            "confirm_target": bool(mission.get("confirm_target", False)),
        },
        "search_runtime": {
            "current_search_subgoal": str(search_runtime.get("current_search_subgoal", "idle")),
            "detection_state": str(search_runtime.get("detection_state", "unknown")),
            "visited_region_count": int(search_runtime.get("visited_region_count", 0)),
            "suspect_region_count": int(search_runtime.get("suspect_region_count", 0)),
            "confirmed_region_count": int(search_runtime.get("confirmed_region_count", 0)),
        },
        "doorway_runtime": {
            "status": str(doorway_runtime.get("status", "idle")),
            "candidate_count": int(doorway_runtime.get("candidate_count", 0)),
            "traversable_candidate_count": int(doorway_runtime.get("traversable_candidate_count", 0)),
            "summary": str(doorway_runtime.get("summary", "")),
            "best_candidate": {
                "label": str(best_candidate.get("label", "")),
                "traversable": bool(best_candidate.get("traversable", False)),
                "confidence": float(best_candidate.get("confidence", 0.0)),
                "center_x_norm": float(best_candidate.get("center_x_norm", 0.0)),
                "center_y_norm": float(best_candidate.get("center_y_norm", 0.0)),
                "width_ratio": float(best_candidate.get("width_ratio", 0.0)),
                "height_ratio": float(best_candidate.get("height_ratio", 0.0)),
                "depth_gain_cm": float(best_candidate.get("depth_gain_cm", 0.0)),
                "clearance_depth_cm": float(best_candidate.get("clearance_depth_cm", 0.0)),
            },
        },
        "phase5_manual": {
            "active_stage_id": active_stage_id,
            "active_stage_name": str(active_stage.get("stage_name", "")),
            "active_objective": str(active_stage.get("objective", "")),
            "location_state": str(phase5_environment.get("location_state", "unknown")),
            "inside_score": int(phase5_environment.get("inside_score", 0)),
            "outside_score": int(phase5_environment.get("outside_score", 0)),
            "rationale": [str(item) for item in (phase5_environment.get("rationale") or [])[:4]],
        },
        "language_memory": {
            "global_summary": str(language_memory_runtime.get("global_summary", "")),
            "current_focus_summary": str(language_memory_runtime.get("current_focus_summary", "")),
        },
        "search_result": {
            "result_status": str(search_result.get("result_status", "unknown")),
            "person_exists": search_result.get("person_exists"),
            "confidence": float(search_result.get("confidence", 0.0)),
        },
        "person_evidence": {
            "evidence_status": str(person_evidence_runtime.get("evidence_status", "idle")),
            "suspect_count": int(person_evidence_runtime.get("suspect_count", 0)),
            "confirm_present_count": int(person_evidence_runtime.get("confirm_present_count", 0)),
            "confirm_absent_count": int(person_evidence_runtime.get("confirm_absent_count", 0)),
        },
        "current_plan": {
            "semantic_subgoal": str(current_plan.get("semantic_subgoal", "")),
            "search_subgoal": str(current_plan.get("search_subgoal", "")),
            "explanation": str(current_plan.get("explanation", "")),
        },
        "pose": {
            "x": round(float(pose.get("x", 0.0)), 1),
            "y": round(float(pose.get("y", 0.0)), 1),
            "z": round(float(pose.get("z", 0.0)), 1),
            "yaw": round(float(pose.get("yaw", 0.0)), 1),
        },
        "depth": {
            "min_depth_cm": round(float(depth.get("min_depth", 0.0)), 1),
            "max_depth_cm": round(float(depth.get("max_depth", 0.0)), 1),
            "front_min_depth_cm": round(float(depth.get("front_min_depth", 0.0)), 1),
            "front_mean_depth_cm": round(float(depth.get("front_mean_depth", 0.0)), 1),
        },
        "runtime_debug": {
            "risk_score": round(float(runtime_debug.get("risk_score", 0.0)), 3),
            "current_waypoint": runtime_debug.get("current_waypoint", {})
            if isinstance(runtime_debug.get("current_waypoint"), dict)
            else {},
        },
        "context": {
            "archive": context.get("archive", {}) if isinstance(context.get("archive"), dict) else {},
        },
    }


def build_scene_waypoint_prompt(request_payload: Dict[str, Any]) -> Dict[str, Any]:
    summaries = _build_request_summaries(request_payload)
    system_prompt = (
        "You are a multimodal UAV scene interpreter and waypoint planner for house-entry search. "
        "You MUST use both the RGB image and the depth image context when available. "
        "First determine whether the UAV is outside the house, inside the house, at a threshold/doorway, or unknown. "
        "Then decide the current stage among scene_interpretation, find_entry_door, approach_entry, cross_entry, indoor_stabilize, house_search, verify_target, report_result. "
        "If a visible door/opening is likely traversable, mark entry_door_visible=true and entry_door_traversable=true. "
        "Return a short waypoint sequence of 1 to 5 normalized image-plane waypoint hints to guide movement. "
        "x_hint and y_hint are normalized in [0,1], where y increases toward the lower part of the image. "
        "The waypoint sequence should help the UAV approach a visible entry, pass through it, and stabilize indoors before broader search. "
        "Output exactly one JSON object only. Do not output markdown, code fences, or extra commentary. "
        "All string values must be single-line plain strings without embedded newlines."
    )
    user_prompt = (
        "Return JSON with keys: scene_state, active_stage, entry_door_visible, entry_door_traversable, planner_confidence, should_request_plan, reasoning, waypoints.\n"
        f"task_label={summaries['task_label']}\n"
        f"mission={_compact_json(summaries['mission'])}\n"
        f"search_runtime={_compact_json(summaries['search_runtime'])}\n"
        f"doorway_runtime={_compact_json(summaries['doorway_runtime'])}\n"
        f"phase5_manual={_compact_json(summaries['phase5_manual'])}\n"
        f"language_memory={_compact_json(summaries['language_memory'])}\n"
        f"search_result={_compact_json(summaries['search_result'])}\n"
        f"person_evidence={_compact_json(summaries['person_evidence'])}\n"
        f"current_plan={_compact_json(summaries['current_plan'])}\n"
        f"pose={_compact_json(summaries['pose'])}\n"
        f"depth={_compact_json(summaries['depth'])}\n"
        f"runtime_debug={_compact_json(summaries['runtime_debug'])}\n"
        f"context={_compact_json(summaries['context'])}\n"
        "Rules:\n"
        "- scene_state must be one of outside_house, inside_house, threshold_zone, unknown.\n"
        "- active_stage must match the immediate next objective, not the final mission.\n"
        "- planner_confidence must be between 0.0 and 1.0.\n"
        "- waypoints should be short-term hints that can be executed sequentially.\n"
        "- If the UAV appears outside and an open/traversable front door is visible, choose approach_entry or cross_entry rather than house_search.\n"
        "- If the UAV is clearly indoors, choose indoor_stabilize or house_search.\n"
        "- reasoning should be concise and reference the scene, doorway, and depth affordance.\n"
    )
    return {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "image_b64": str(request_payload.get("image_b64", "") or ""),
    }


def build_heuristic_scene_waypoint_runtime(
    *,
    request_payload: Dict[str, Any],
    policy_name: str = "local_scene_waypoint_fallback",
    source: str = "local_heuristic",
    fallback_used: bool = False,
    fallback_reason: str = "",
    model_name: str = "",
    api_style: str = "",
    route_mode: str = "",
) -> Dict[str, Any]:
    doorway_runtime = request_payload.get("doorway_runtime") if isinstance(request_payload.get("doorway_runtime"), dict) else {}
    phase5_manual = request_payload.get("phase5_mission_manual") if isinstance(request_payload.get("phase5_mission_manual"), dict) else {}
    phase5_env = phase5_manual.get("environment_context") if isinstance(phase5_manual.get("environment_context"), dict) else {}
    best_candidate = doorway_runtime.get("best_candidate") if isinstance(doorway_runtime.get("best_candidate"), dict) else {}
    scene_state = str(phase5_env.get("location_state", "unknown") or "unknown")
    active_stage = str(phase5_manual.get("active_stage_name", "") or phase5_manual.get("active_stage_id", "") or "scene_interpretation")
    if active_stage.startswith("phase5_stage_"):
        active_stage = "scene_interpretation"
    entry_visible = bool(doorway_runtime.get("candidate_count", 0) > 0)
    entry_traversable = bool(doorway_runtime.get("traversable_candidate_count", 0) > 0 or best_candidate.get("traversable", False))
    if scene_state == "outside_house" and entry_visible and entry_traversable:
        active_stage = "approach_entry"
    elif scene_state == "outside_house" and entry_visible:
        active_stage = "find_entry_door"
    elif scene_state == "inside_house":
        active_stage = "house_search"
    x_hint = float(best_candidate.get("center_x_norm", 0.5) or 0.5)
    y_hint = float(best_candidate.get("center_y_norm", 0.62) or 0.62)
    if scene_state == "inside_house":
        y_hint = 0.68
    waypoints = [
        {
            "label": "heuristic_focus",
            "x_hint": round(_clamp(x_hint, 0.0, 1.0), 4),
            "y_hint": round(_clamp(y_hint, 0.0, 1.0), 4),
            "z_offset_cm": 0.0,
            "yaw_hint_deg": 0.0,
        }
    ]
    return build_scene_waypoint_runtime_state(
        request_id=f"scene_waypoint_{request_payload.get('frame_id', 'unknown')}",
        mode="scene_waypoint_only",
        policy_name=policy_name,
        source=source,
        status="fallback" if fallback_used else "ok",
        scene_state=scene_state if scene_state in ALLOWED_SCENE_STATES else "unknown",
        active_stage=active_stage if active_stage in ALLOWED_STAGE_IDS else "scene_interpretation",
        entry_door_visible=entry_visible,
        entry_door_traversable=entry_traversable,
        planner_confidence=0.55 if entry_visible else 0.35,
        should_request_plan=True,
        reasoning=str(doorway_runtime.get("summary", "") or phase5_manual.get("summary", "") or "Heuristic multimodal fallback summary."),
        waypoints=waypoints,
        last_trigger=str(request_payload.get("trigger", "manual_request") or "manual_request"),
        last_latency_ms=0.0,
        model_name=model_name,
        api_style=api_style,
        route_mode=route_mode,
        usage={},
        attempt_count=0,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        upstream_error=fallback_reason,
        raw_text="",
        parsed_payload={
            "scene_state": scene_state,
            "active_stage": active_stage,
            "entry_door_visible": entry_visible,
            "entry_door_traversable": entry_traversable,
            "planner_confidence": 0.55 if entry_visible else 0.35,
            "should_request_plan": True,
            "reasoning": str(doorway_runtime.get("summary", "") or ""),
            "waypoints": waypoints,
        },
    )


def build_multimodal_scene_waypoints(
    *,
    request_payload: Dict[str, Any],
    client: LLMPlannerClient,
    policy_name: str = "external_scene_waypoint_planner",
) -> Dict[str, Any]:
    prompt = build_scene_waypoint_prompt(request_payload)
    response = client.generate(
        system_prompt=prompt["system_prompt"],
        user_prompt=prompt["user_prompt"],
        image_b64=prompt["image_b64"],
        json_schema=SCENE_WAYPOINT_RESPONSE_SCHEMA,
    )
    payload = parse_scene_waypoint_json_response(response["text"])
    scene_state = str(payload.get("scene_state", "unknown") or "unknown").strip()
    if scene_state not in ALLOWED_SCENE_STATES:
        scene_state = "unknown"
    active_stage = str(payload.get("active_stage", "scene_interpretation") or "scene_interpretation").strip()
    if active_stage not in ALLOWED_STAGE_IDS:
        active_stage = "scene_interpretation"
    waypoints = _coerce_waypoints(payload.get("waypoints"))
    if not waypoints:
        waypoints = [{"label": "hold_focus", "x_hint": 0.5, "y_hint": 0.62, "z_offset_cm": 0.0, "yaw_hint_deg": 0.0}]
    reasoning = str(payload.get("reasoning", "") or "").replace("\n", " ").strip()
    return build_scene_waypoint_runtime_state(
        request_id=f"scene_waypoint_{request_payload.get('frame_id', 'unknown')}",
        mode="scene_waypoint_only",
        policy_name=policy_name,
        source="multimodal_scene_llm",
        status="ok",
        scene_state=scene_state,
        active_stage=active_stage,
        entry_door_visible=bool(payload.get("entry_door_visible", False)),
        entry_door_traversable=bool(payload.get("entry_door_traversable", False)),
        planner_confidence=_clamp(payload.get("planner_confidence", 0.0), 0.0, 1.0),
        should_request_plan=bool(payload.get("should_request_plan", True)),
        reasoning=reasoning,
        waypoints=waypoints,
        last_trigger=str(request_payload.get("trigger", "manual_request") or "manual_request"),
        last_latency_ms=float(response.get("latency_ms", 0.0) or 0.0),
        model_name=str(response.get("model_name", "") or ""),
        api_style=str(response.get("api_style", "") or ""),
        route_mode="llm_only",
        usage=response.get("usage", {}) if isinstance(response.get("usage"), dict) else {},
        attempt_count=int(response.get("attempt_count", 0) or 0),
        fallback_used=False,
        fallback_reason="",
        upstream_error="",
        raw_text=str(response.get("text", "") or ""),
        parsed_payload={
            "scene_state": scene_state,
            "active_stage": active_stage,
            "entry_door_visible": bool(payload.get("entry_door_visible", False)),
            "entry_door_traversable": bool(payload.get("entry_door_traversable", False)),
            "planner_confidence": _clamp(payload.get("planner_confidence", 0.0), 0.0, 1.0),
            "should_request_plan": bool(payload.get("should_request_plan", True)),
            "reasoning": reasoning,
            "waypoints": waypoints,
        },
        system_prompt_excerpt=_truncate_debug_text(prompt["system_prompt"], 1800),
        user_prompt_excerpt=_truncate_debug_text(prompt["user_prompt"], 4000),
    )
