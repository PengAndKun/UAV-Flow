"""
Simple Phase 2 planner service for UAV-Flow-Eval.

This server accepts structured planner requests from `uav_control_server.py`
and returns sparse high-level waypoint plans. It is intentionally lightweight:
the goal is to provide a stable external planner endpoint before swapping in a
larger model-based planner.
"""

import argparse
import json
import logging
import os
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib import error
from urllib.parse import urlparse

import numpy as np

from llm_action_adapter import LLMActionAdapterError, build_llm_action
from llm_planner_adapter import LLMPlannerAdapterError, build_llm_plan
from llm_planner_client import LLMPlannerClient, LLMPlannerClientError, LLMPlannerConfig
from runtime_interfaces import (
    build_llm_action_runtime_state,
    build_plan_request,
    build_plan_state,
    build_search_region,
    build_waypoint,
    coerce_plan_payload,
    now_timestamp,
)

logger = logging.getLogger(__name__)

ROOM_REGION_HINTS = [
    (("bedroom", "bed room"), "bedroom", "bedroom", "Bedrooms often contain likely resting or hiding targets."),
    (("kitchen",), "kitchen", "kitchen", "Kitchens are common traversed spaces worth checking quickly."),
    (("bathroom", "restroom"), "bathroom", "bathroom", "Bathrooms are compact rooms that need explicit confirmation."),
    (("living room", "livingroom", "lounge"), "living_room", "living_room", "Living rooms contain large furniture and occluded corners."),
    (("hallway", "corridor"), "hallway", "hallway", "Hallways connect multiple rooms and are useful for coverage expansion."),
    (("stairs", "stair", "staircase"), "stairs", "stairs", "Stair areas are transition zones that often require reorientation."),
    (("door", "doorway", "entry"), "doorway", "doorway", "Doorways are useful search pivots for entering new rooms."),
    (("corner", "occluded corner"), "corner", "", "Occluded corners are high-value regions for search confirmation."),
]


def should_use_llm_mode(
    *,
    route_mode: str,
    mission_type: str,
) -> Tuple[bool, str]:
    """Decide whether the current request should use the LLM planner path."""
    if route_mode == "llm_only":
        return True, "explicit_llm_only"
    if route_mode == "search_hybrid":
        if mission_type in ("person_search", "room_search", "target_verification"):
            return True, "search_mission_routed_to_llm"
        return False, "non_search_mission_routed_to_heuristic"
    return False, "explicit_heuristic_only"


def resolve_planner_route_mode(planner_mode: str, explicit_route_mode: str) -> str:
    route_mode = str(explicit_route_mode or "auto").strip().lower()
    if route_mode and route_mode != "auto":
        return route_mode
    normalized_mode = str(planner_mode or "heuristic").strip().lower()
    if normalized_mode == "llm":
        return "llm_only"
    if normalized_mode == "hybrid":
        return "search_hybrid"
    return "heuristic_only"


def build_planner_debug_metadata(
    *,
    source: str,
    planner_mode: str,
    route_mode: str,
    route_reason: str,
    fallback_used: bool,
    fallback_reason: str = "",
    llm_model_name: str = "",
    llm_api_style: str = "",
    llm_usage: Optional[Dict[str, Any]] = None,
    llm_latency_ms: float = 0.0,
) -> Dict[str, Any]:
    return {
        "source": str(source or ""),
        "planner_mode": str(planner_mode or "heuristic"),
        "route_mode": str(route_mode or "heuristic_only"),
        "route_reason": str(route_reason or ""),
        "fallback_used": bool(fallback_used),
        "fallback_reason": str(fallback_reason or ""),
        "llm_model_name": str(llm_model_name or ""),
        "llm_api_style": str(llm_api_style or ""),
        "llm_usage": llm_usage or {},
        "llm_latency_ms": float(llm_latency_ms),
    }


def build_fallback_plan_from_seed(
    heuristic_seed: Dict[str, Any],
    *,
    planner_name: str,
    planner_mode: str,
    route_mode: str,
    route_reason: str,
    fallback_reason: str,
    llm_model_name: str,
    llm_api_style: str,
) -> Dict[str, Any]:
    debug = heuristic_seed.get("debug") if isinstance(heuristic_seed.get("debug"), dict) else {}
    merged_debug = dict(debug)
    merged_debug.update(
        build_planner_debug_metadata(
            source="heuristic_fallback",
            planner_mode=planner_mode,
            route_mode=route_mode,
            route_reason=route_reason,
            fallback_used=True,
            fallback_reason=fallback_reason,
            llm_model_name=llm_model_name,
            llm_api_style=llm_api_style,
        )
    )
    fallback_plan = dict(heuristic_seed)
    fallback_plan["planner_name"] = planner_name
    fallback_plan["debug"] = merged_debug
    return fallback_plan


def infer_heuristic_action_from_request(request_payload: Dict[str, Any]) -> Tuple[str, str]:
    runtime_debug = request_payload.get("runtime_debug") if isinstance(request_payload.get("runtime_debug"), dict) else {}
    current_plan = request_payload.get("current_plan") if isinstance(request_payload.get("current_plan"), dict) else {}
    reflex_runtime = request_payload.get("reflex_runtime") if isinstance(request_payload.get("reflex_runtime"), dict) else {}
    risk_score = float(runtime_debug.get("risk_score", 0.0) or 0.0)
    reflex_suggested = str(reflex_runtime.get("suggested_action", "") or "").strip().lower()
    semantic_subgoal = str(current_plan.get("semantic_subgoal", "") or "").strip().lower()
    search_subgoal = str(current_plan.get("search_subgoal", "") or "").strip().lower()
    combined = f"{semantic_subgoal} {search_subgoal}"

    if risk_score >= 0.85:
        return "yaw_left", "high risk fallback: prefer reorientation"
    if reflex_suggested in {"forward", "backward", "left", "right", "up", "down", "yaw_left", "yaw_right"}:
        return reflex_suggested, "reuse local reflex suggestion as heuristic action fallback"
    if any(token in combined for token in ("turn_left", "yaw_left", "scan_left")):
        return "yaw_left", "heuristic action follows left-turn plan cue"
    if any(token in combined for token in ("turn_right", "yaw_right", "scan_right")):
        return "yaw_right", "heuristic action follows right-turn plan cue"
    if any(token in combined for token in ("ascend", "move_up", "upward")):
        return "up", "heuristic action follows upward search cue"
    if any(token in combined for token in ("descend", "move_down", "downward")):
        return "down", "heuristic action follows downward search cue"
    if any(token in combined for token in ("backward", "reverse")):
        return "backward", "heuristic action follows backward search cue"
    if any(token in combined for token in ("strafe_left", "move_left", "left_room")):
        return "left", "heuristic action follows leftward search cue"
    if any(token in combined for token in ("strafe_right", "move_right", "right_room")):
        return "right", "heuristic action follows rightward search cue"
    if any(token in combined for token in ("hold", "stop", "pause", "confirm")):
        return "hold", "heuristic action holds for confirmation"
    return "forward", "default heuristic action moves forward toward exploration frontier"


def build_heuristic_action_response(
    request_payload: Dict[str, Any],
    *,
    policy_name: str,
    route_mode: str,
    route_reason: str,
    source: str,
    fallback_used: bool,
    fallback_reason: str = "",
    llm_model_name: str = "",
    llm_api_style: str = "",
) -> Dict[str, Any]:
    action_name, rationale = infer_heuristic_action_from_request(request_payload)
    return build_llm_action_runtime_state(
        action_id=f"heuristic_action_{request_payload.get('frame_id', 'unknown')}",
        mode="heuristic_action",
        policy_name=policy_name,
        source=source,
        status="ok" if not fallback_used else "fallback",
        suggested_action=action_name,
        should_execute=action_name not in {"hold", "hold_position", "idle"},
        confidence=0.35 if not fallback_used else 0.25,
        rationale=rationale,
        stop_condition="continue_search",
        should_request_plan=False,
        last_trigger=str(request_payload.get("trigger", "manual_request") or "manual_request"),
        last_latency_ms=0.0,
        risk_score=float(((request_payload.get("runtime_debug") or {}).get("risk_score", 0.0)) or 0.0),
        model_name=str(llm_model_name or ""),
        api_style=str(llm_api_style or ""),
        route_mode=str(route_mode or ""),
        usage={},
        attempt_count=0,
        fallback_used=bool(fallback_used),
        fallback_reason=str(fallback_reason or route_reason or ""),
        upstream_error="",
        raw_text="",
        parsed_payload={
            "action": action_name,
            "confidence": 0.35 if not fallback_used else 0.25,
            "rationale": rationale,
            "stop_condition": "continue_search",
            "should_request_plan": False,
            "route_reason": str(route_reason or ""),
        },
    )


def build_llm_client_from_args(args: argparse.Namespace) -> Optional[LLMPlannerClient]:
    api_style = str(args.llm_api_style or "openai_chat")
    api_key = str(args.llm_api_key or "").strip()
    env_candidates: List[str] = []
    if str(args.llm_api_key_env or "").strip():
        env_candidates.append(str(args.llm_api_key_env).strip())
    if api_style in ("openai_chat", "openai_responses"):
        env_candidates.extend(["OPENAI_API_KEY"])
    elif api_style == "anthropic_messages":
        env_candidates.extend(["ANTHROPIC_API_KEY"])
    elif api_style in ("google_gemini", "google_genai_sdk"):
        env_candidates.extend(["GEMINI_API_KEY", "GOOGLE_API_KEY"])
    if not api_key:
        seen_env_names = set()
        for env_name in env_candidates:
            if not env_name or env_name in seen_env_names:
                continue
            seen_env_names.add(env_name)
            env_value = str(os.getenv(env_name, "")).strip()
            if env_value:
                api_key = env_value
                break
    if args.planner_mode == "heuristic":
        return None
    if not api_key:
        logger.warning(
            "LLM planner API key is empty. api_style=%s llm_api_key_env=%s env_candidates=%s",
            api_style,
            str(args.llm_api_key_env or ""),
            ",".join(env_candidates),
        )
    base_url = str(args.llm_base_url or "").strip()
    if not base_url:
        if api_style == "anthropic_messages":
            base_url = "https://api.anthropic.com"
        elif api_style == "google_gemini":
            base_url = "https://generativelanguage.googleapis.com"
        elif api_style == "google_genai_sdk":
            base_url = "google-genai-sdk"
    if not base_url:
        logger.info("Planner mode=%s without llm_base_url: LLM planner disabled until configured.", args.planner_mode)
        return None
    if not str(args.llm_model or "").strip():
        logger.info("Planner mode=%s without llm_model: LLM planner disabled until configured.", args.planner_mode)
        return None
    config = LLMPlannerConfig(
        base_url=base_url,
        api_key=api_key,
        model_name=str(args.llm_model).strip(),
        api_style=api_style,
        endpoint_path=str(args.llm_endpoint_path or "").strip(),
        timeout_s=float(args.llm_timeout_s),
        max_retries=max(1, int(args.llm_max_retries)),
        temperature=float(args.llm_temperature),
        max_output_tokens=max(128, int(args.llm_max_output_tokens)),
        include_images=str(args.llm_input_mode or "text_image") == "text_image",
        force_json=not bool(args.llm_disable_force_json),
        auth_header=str(args.llm_auth_header or "Authorization"),
        auth_scheme=str(args.llm_auth_scheme or "Bearer"),
        anthropic_version=str(args.llm_anthropic_version or "2023-06-01"),
    )
    return LLMPlannerClient(config)


def build_planner_config_payload(args: argparse.Namespace, llm_client: Optional[LLMPlannerClient]) -> Dict[str, Any]:
    route_mode = resolve_planner_route_mode(args.planner_mode, str(args.planner_route_mode or "auto"))
    active_api_key = ""
    if llm_client is not None and getattr(llm_client, "config", None) is not None:
        active_api_key = str(getattr(llm_client.config, "api_key", "") or "")
    if not active_api_key:
        active_api_key = str(args.llm_api_key or "")
    return {
        "status": "ok",
        "planner_name": str(args.planner_name or ""),
        "planner_mode": str(args.planner_mode or "heuristic"),
        "planner_route_mode": route_mode,
        "planner_route_mode_raw": str(args.planner_route_mode or "auto"),
        "llm_enabled": llm_client is not None,
        "llm_base_url": str(args.llm_base_url or ""),
        "llm_model": str(args.llm_model or ""),
        "llm_api_style": str(args.llm_api_style or ""),
        "llm_input_mode": str(args.llm_input_mode or "text_image"),
        "fallback_to_heuristic": bool(args.fallback_to_heuristic),
        "llm_api_key_env": str(args.llm_api_key_env or ""),
        "llm_api_key_configured": bool(str(active_api_key).strip()),
        "llm_timeout_s": float(args.llm_timeout_s),
        "llm_max_retries": int(args.llm_max_retries),
        "llm_temperature": float(args.llm_temperature),
        "llm_max_output_tokens": int(args.llm_max_output_tokens),
    }


def infer_mission_type(task_label: str, request_payload: Dict[str, Any]) -> str:
    text = task_label.lower()
    mission = request_payload.get("mission") if isinstance(request_payload.get("mission"), dict) else {}
    if mission.get("mission_type"):
        return str(mission.get("mission_type"))
    if any(keyword in text for keyword in ["confirm", "verify", "approach", "closer inspection"]):
        return "target_verification"
    if any(keyword in text for keyword in ["person", "people", "human", "survivor", "victim"]):
        return "person_search"
    if any(keyword in text for keyword in ["search", "find", "inspect", "look for"]):
        return "room_search"
    return "semantic_navigation"


def infer_candidate_regions(task_label: str, mission_type: str) -> List[Dict[str, Any]]:
    text = task_label.lower()
    candidate_regions: List[Tuple[int, Dict[str, Any]]] = []
    if any(keyword in text for keyword in ["house", "building", "entire home", "whole house", "whole home"]):
        candidate_regions.append(
            (
                text.find("house") if "house" in text else 0,
                build_search_region(
                    region_id="entire_house",
                    region_label="entire house",
                    region_type="house",
                    room_type="house",
                    priority=4,
                    status="unobserved",
                    rationale="Global mission scope requires broad house-level search coverage.",
                ),
            )
        )
    for keywords, region_label, room_type, rationale in ROOM_REGION_HINTS:
        matches = [text.find(keyword) for keyword in keywords if keyword in text]
        if matches:
            candidate_regions.append(
                (
                    min(matches),
                    build_search_region(
                        region_id=f"{region_label}_{len(candidate_regions) + 1}",
                        region_label=region_label.replace("_", " "),
                        region_type="room" if room_type else "area",
                        room_type=room_type,
                        priority=0,
                        status="suspect" if mission_type in ("person_search", "target_verification") else "unobserved",
                        rationale=rationale,
                    ),
                )
            )
    if any(keyword in text for keyword in ["suspect", "possible person", "possible target"]) and not any(
        region.get("region_label") == "suspect region" for _, region in candidate_regions
    ):
        candidate_regions.append(
            (
                max(0, text.find("suspect")),
                build_search_region(
                    region_id="suspect_region",
                    region_label="suspect region",
                    region_type="area",
                    priority=0,
                    status="suspect",
                    rationale="The task explicitly mentions a suspect region that should be revisited or confirmed.",
                ),
            )
        )
    candidate_regions.sort(key=lambda item: item[0])
    normalized_regions: List[Dict[str, Any]] = []
    for index, (_position, region) in enumerate(candidate_regions):
        region["priority"] = max(1, 4 - index)
        normalized_regions.append(region)
    if normalized_regions:
        return normalized_regions
    if mission_type in ("person_search", "room_search", "target_verification"):
        return [
            build_search_region(
                region_id="forward_search_sector",
                region_label="forward search sector",
                region_type="sector",
                priority=2,
                status="unobserved",
                rationale="Default search region derived from the current forward-facing observation.",
            )
        ]
    return []


def infer_search_subgoal(task_label: str, mission_type: str, semantic_subgoal: str, candidate_regions: List[Dict[str, Any]]) -> str:
    text = task_label.lower()
    if mission_type == "target_verification":
        if any(keyword in text for keyword in ["approach", "closer", "move closer", "go closer"]):
            return "approach_suspect_region"
        return "confirm_suspect_region"
    if mission_type in ("person_search", "room_search"):
        if any(keyword in text for keyword in ["revisit", "recheck", "again"]):
            return "revisit_suspect_region"
        if any(keyword in text for keyword in ["confirm", "verify"]):
            return "confirm_suspect_region"
        if any(keyword in text for keyword in ["house", "building", "whole house", "entire house"]):
            return "search_house"
        if any(keyword in text for keyword in ["cover", "sweep", "scan"]):
            return "search_frontier"
        if candidate_regions and candidate_regions[0].get("region_type") == "room":
            return "search_room"
        return "search_frontier"
    if semantic_subgoal in ("turn_left", "turn_right"):
        return "reorient_for_navigation"
    if semantic_subgoal.startswith("move_"):
        return "advance_to_waypoint"
    return semantic_subgoal or "idle"


def build_search_explanation(
    *,
    mission_type: str,
    matched_keywords: List[str],
    candidate_regions: List[Dict[str, Any]],
    semantic_subgoal: str,
) -> str:
    region_labels = [str(region.get("region_label", "")) for region in candidate_regions if region.get("region_label")]
    if mission_type in ("person_search", "room_search", "target_verification"):
        if region_labels:
            return (
                f"Mission guidance prioritizes {', '.join(region_labels[:3])} "
                f"while preserving navigation subgoal={semantic_subgoal}."
            )
        return f"Mission guidance falls back to forward search frontier with subgoal={semantic_subgoal}."
    return f"Heuristic navigation matched={','.join(matched_keywords)} subgoal={semantic_subgoal}."


def normalize_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def extract_distance_cm(text: str, default_cm: float) -> float:
    """Extract a rough distance from task text, defaulting to centimeters."""
    match = re.search(r"(\d+(?:\.\d+)?)\s*(meter|meters|m|cm)?", text.lower())
    if not match:
        return float(default_cm)
    value = float(match.group(1))
    unit = (match.group(2) or "cm").lower()
    if unit in {"meter", "meters", "m"}:
        return value * 100.0
    return value


def infer_direction(task_label: str) -> Tuple[float, str, List[str]]:
    """Infer a relative yaw offset and semantic subgoal from instruction text."""
    text = task_label.lower()
    matched: List[str] = []

    if any(keyword in text for keyword in ["left", "turn left"]):
        matched.append("left")
        return -90.0, "turn_left", matched
    if any(keyword in text for keyword in ["right", "turn right"]):
        matched.append("right")
        return 90.0, "turn_right", matched
    if any(keyword in text for keyword in ["back", "backward", "behind"]):
        matched.append("backward")
        return 180.0, "move_backward", matched

    if any(keyword in text for keyword in ["inspect", "search", "find", "look for"]):
        matched.append("search")
        return 0.0, "forward_search", matched
    if any(keyword in text for keyword in ["follow", "go", "forward", "ahead", "move"]):
        matched.append("forward")
        return 0.0, "move_forward", matched

    matched.append("default_forward")
    return 0.0, "move_forward", matched


def build_heuristic_plan(request_payload: Dict[str, Any], planner_name: str, waypoint_radius_cm: float) -> Dict[str, Any]:
    """Generate a structured sparse plan from a simple heuristic policy."""
    task_label = str(request_payload.get("task_label", "")).strip() or "idle"
    pose = request_payload.get("pose", {}) if isinstance(request_payload.get("pose"), dict) else {}
    depth = request_payload.get("depth", {}) if isinstance(request_payload.get("depth"), dict) else {}
    current_x = float(pose.get("x", 0.0))
    current_y = float(pose.get("y", 0.0))
    current_z = float(pose.get("z", 0.0))
    current_yaw = float(pose.get("yaw", 0.0))
    step_index = int(request_payload.get("step_index", 0))

    offset_deg, semantic_subgoal, matched_keywords = infer_direction(task_label)
    mission_type = infer_mission_type(task_label, request_payload)
    command_yaw = normalize_angle_deg(current_yaw + offset_deg)
    requested_distance_cm = extract_distance_cm(task_label, 300.0)
    visible_min_depth = float(depth.get("min_depth", requested_distance_cm))
    safe_distance_cm = max(120.0, min(requested_distance_cm, visible_min_depth - 30.0 if visible_min_depth > 0 else requested_distance_cm))

    if any(keyword in task_label.lower() for keyword in ["up", "ascend", "rise", "higher"]):
        semantic_subgoal = "ascend_and_observe"
        target_z = current_z + 80.0
    elif any(keyword in task_label.lower() for keyword in ["down", "descend", "lower"]):
        semantic_subgoal = "descend_and_observe"
        target_z = current_z - 60.0
    else:
        target_z = current_z

    theta = np.radians(command_yaw)
    primary_waypoint = build_waypoint(
        x=current_x + safe_distance_cm * float(np.cos(theta)),
        y=current_y + safe_distance_cm * float(np.sin(theta)),
        z=target_z,
        yaw=command_yaw,
        radius=waypoint_radius_cm,
        semantic_label=semantic_subgoal,
    )
    staging_waypoint = build_waypoint(
        x=current_x + (safe_distance_cm * 0.5) * float(np.cos(theta)),
        y=current_y + (safe_distance_cm * 0.5) * float(np.sin(theta)),
        z=(current_z + target_z) / 2.0,
        yaw=command_yaw,
        radius=waypoint_radius_cm,
        semantic_label="staging_waypoint",
    )

    sector_count = 8
    sector_id = int(round(((command_yaw % 360.0) / 360.0) * sector_count)) % sector_count
    confidence = 0.55 if matched_keywords and matched_keywords[0] != "default_forward" else 0.4
    candidate_regions = infer_candidate_regions(task_label, mission_type)
    priority_region = candidate_regions[0] if candidate_regions else {}
    search_subgoal = infer_search_subgoal(task_label, mission_type, semantic_subgoal, candidate_regions)
    confirm_target = mission_type == "target_verification" or any(
        keyword in task_label.lower() for keyword in ["confirm", "verify", "approach", "closer inspection"]
    )
    explanation = build_search_explanation(
        mission_type=mission_type,
        matched_keywords=matched_keywords,
        candidate_regions=candidate_regions,
        semantic_subgoal=semantic_subgoal,
    )

    return build_plan_state(
        plan_id=f"external_plan_{request_payload.get('frame_id', 'unknown')}",
        planner_name=planner_name,
        generated_at=now_timestamp(),
        sector_id=sector_id,
        candidate_waypoints=[primary_waypoint, staging_waypoint],
        semantic_subgoal=semantic_subgoal,
        planner_confidence=confidence,
        should_replan=False,
        mission_type=mission_type,
        search_subgoal=search_subgoal,
        priority_region=priority_region,
        candidate_regions=candidate_regions,
        confirm_target=confirm_target,
        explanation=explanation,
        debug={
            "source": "external_heuristic_planner",
            "matched_keywords": matched_keywords,
            "requested_distance_cm": requested_distance_cm,
            "safe_distance_cm": safe_distance_cm,
            "step_index": step_index,
        },
    )


def make_handler(args: argparse.Namespace, llm_client: Optional[LLMPlannerClient]):
    class PlannerRequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            payload = json.loads(raw.decode("utf-8"))
            return payload if isinstance(payload, dict) else {}

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/health"):
                self._send_json(
                    {
                        "status": "ok",
                        "service": "planner_server",
                        "planner_name": args.planner_name,
                        "planner_mode": args.planner_mode,
                        "llm_enabled": llm_client is not None,
                        "llm_model": str(args.llm_model or ""),
                        "llm_api_style": str(args.llm_api_style or ""),
                        "planner_route_mode": resolve_planner_route_mode(args.planner_mode, str(args.planner_route_mode or "auto")),
                    }
                )
            elif parsed.path == "/config":
                self._send_json(build_planner_config_payload(args, llm_client))
            elif parsed.path == "/schema":
                example_request = build_plan_request(
                    task_label="move forward 3 meters",
                    instruction="move forward 3 meters",
                    frame_id="frame_000001",
                    timestamp=now_timestamp(),
                    pose={"x": 0.0, "y": 0.0, "z": 120.0, "yaw": 0.0},
                    depth={"min_depth": 250.0, "max_depth": 1200.0},
                    camera_info={"frame_id": "PX4/CameraDepth_optical"},
                    planner_name=args.planner_name,
                    trigger="manual_request",
                    step_index=0,
                )
                example_plan = build_heuristic_plan(example_request, args.planner_name, args.default_waypoint_radius_cm)
                self._send_json(
                    {
                        "status": "ok",
                        "planner_mode": args.planner_mode,
                        "llm_enabled": llm_client is not None,
                        "request_example": example_request,
                        "plan_example": example_plan,
                    }
                )
            else:
                self._send_json({"status": "error", "message": "Not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            nonlocal llm_client
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/config":
                    update_payload = self._read_json_body()
                    if not isinstance(update_payload, dict):
                        raise ValueError("Planner config payload must be a JSON object.")
                    if "planner_name" in update_payload:
                        args.planner_name = str(update_payload.get("planner_name", args.planner_name) or args.planner_name)
                    if "planner_mode" in update_payload:
                        planner_mode = str(update_payload.get("planner_mode", args.planner_mode) or args.planner_mode)
                        if planner_mode not in {"heuristic", "llm", "hybrid"}:
                            raise ValueError(f"Unsupported planner_mode: {planner_mode}")
                        args.planner_mode = planner_mode
                    if "planner_route_mode" in update_payload:
                        route_mode = str(update_payload.get("planner_route_mode", args.planner_route_mode) or args.planner_route_mode)
                        if route_mode not in {"auto", "heuristic_only", "llm_only", "search_hybrid"}:
                            raise ValueError(f"Unsupported planner_route_mode: {route_mode}")
                        args.planner_route_mode = route_mode
                    if "llm_model" in update_payload:
                        args.llm_model = str(update_payload.get("llm_model", args.llm_model) or "")
                    if "llm_api_style" in update_payload:
                        api_style = str(update_payload.get("llm_api_style", args.llm_api_style) or args.llm_api_style)
                        if api_style not in {"openai_chat", "openai_responses", "anthropic_messages", "google_gemini", "google_genai_sdk"}:
                            raise ValueError(f"Unsupported llm_api_style: {api_style}")
                        args.llm_api_style = api_style
                    if "llm_base_url" in update_payload:
                        args.llm_base_url = str(update_payload.get("llm_base_url", args.llm_base_url) or "")
                    if "llm_input_mode" in update_payload:
                        input_mode = str(update_payload.get("llm_input_mode", args.llm_input_mode) or args.llm_input_mode)
                        if input_mode not in {"text", "text_image"}:
                            raise ValueError(f"Unsupported llm_input_mode: {input_mode}")
                        args.llm_input_mode = input_mode
                    if "fallback_to_heuristic" in update_payload:
                        args.fallback_to_heuristic = bool(update_payload.get("fallback_to_heuristic", args.fallback_to_heuristic))
                    if "llm_api_key_env" in update_payload:
                        args.llm_api_key_env = str(update_payload.get("llm_api_key_env", args.llm_api_key_env) or "")
                    if "llm_api_key" in update_payload:
                        args.llm_api_key = str(update_payload.get("llm_api_key", args.llm_api_key) or "")
                    if "llm_timeout_s" in update_payload:
                        args.llm_timeout_s = float(update_payload.get("llm_timeout_s", args.llm_timeout_s))
                    if "llm_max_retries" in update_payload:
                        args.llm_max_retries = max(1, int(update_payload.get("llm_max_retries", args.llm_max_retries)))
                    if "llm_temperature" in update_payload:
                        args.llm_temperature = float(update_payload.get("llm_temperature", args.llm_temperature))
                    if "llm_max_output_tokens" in update_payload:
                        args.llm_max_output_tokens = max(128, int(update_payload.get("llm_max_output_tokens", args.llm_max_output_tokens)))
                    llm_client = build_llm_client_from_args(args)
                    self._send_json(build_planner_config_payload(args, llm_client))
                    return
                if parsed.path == args.action_endpoint:
                    request_payload = self._read_json_body()
                    mission = request_payload.get("mission") if isinstance(request_payload.get("mission"), dict) else {}
                    mission_type = str(mission.get("mission_type", "semantic_navigation") or "semantic_navigation")
                    route_mode = resolve_planner_route_mode(args.planner_mode, str(args.planner_route_mode or "auto"))
                    use_llm_mode, route_reason = should_use_llm_mode(route_mode=route_mode, mission_type=mission_type)
                    if use_llm_mode:
                        if llm_client is None:
                            if args.fallback_to_heuristic:
                                logger.warning("LLM action requested but client is unavailable, using heuristic action fallback.")
                                llm_action = build_heuristic_action_response(
                                    request_payload,
                                    policy_name=args.planner_name,
                                    route_mode=route_mode,
                                    route_reason=route_reason,
                                    source="heuristic_action_fallback",
                                    fallback_used=True,
                                    fallback_reason="llm_client_unavailable",
                                    llm_model_name=str(args.llm_model or ""),
                                    llm_api_style=str(args.llm_api_style or ""),
                                )
                            else:
                                raise RuntimeError("LLM action mode requested but no LLM client is configured.")
                        else:
                            try:
                                llm_action = build_llm_action(
                                    request_payload=request_payload,
                                    client=llm_client,
                                    policy_name=args.planner_name,
                                )
                            except (LLMActionAdapterError, LLMPlannerClientError, error.URLError, TimeoutError) as exc:
                                if args.fallback_to_heuristic:
                                    logger.warning("LLM action request failed, using heuristic action fallback: %s", exc)
                                    llm_action = build_heuristic_action_response(
                                        request_payload,
                                        policy_name=args.planner_name,
                                        route_mode=route_mode,
                                        route_reason=route_reason,
                                        source="heuristic_action_fallback",
                                        fallback_used=True,
                                        fallback_reason=str(exc),
                                        llm_model_name=str(args.llm_model or ""),
                                        llm_api_style=str(args.llm_api_style or ""),
                                    )
                                else:
                                    raise
                    else:
                        llm_action = build_heuristic_action_response(
                            request_payload,
                            policy_name=args.planner_name,
                            route_mode=route_mode,
                            route_reason=route_reason,
                            source="local_heuristic",
                            fallback_used=False,
                            llm_model_name="",
                            llm_api_style="",
                        )
                    self._send_json({"status": "ok", "llm_action_runtime": llm_action})
                    return
                if parsed.path != args.endpoint:
                    self._send_json({"status": "error", "message": "Not found"}, 404)
                    return
                request_payload = self._read_json_body()
                route_mode = resolve_planner_route_mode(args.planner_mode, str(args.planner_route_mode or "auto"))
                heuristic_seed = build_heuristic_plan(
                    request_payload,
                    planner_name=args.planner_name if args.planner_mode == "heuristic" else "heuristic_seed",
                    waypoint_radius_cm=args.default_waypoint_radius_cm,
                )
                mission_type = str(heuristic_seed.get("mission_type", "semantic_navigation"))
                use_llm_mode, route_reason = should_use_llm_mode(route_mode=route_mode, mission_type=mission_type)
                if use_llm_mode:
                    if llm_client is None:
                        if args.fallback_to_heuristic:
                            logger.warning("LLM planner requested but client is unavailable, using heuristic fallback.")
                            plan = build_fallback_plan_from_seed(
                                heuristic_seed,
                                planner_name=args.planner_name,
                                planner_mode=args.planner_mode,
                                route_mode=route_mode,
                                route_reason=route_reason,
                                fallback_reason="llm_client_unavailable",
                                llm_model_name=str(args.llm_model or ""),
                                llm_api_style=str(args.llm_api_style or ""),
                            )
                        else:
                            raise RuntimeError("LLM planner mode requested but no LLM client is configured.")
                    else:
                        try:
                            plan = build_llm_plan(
                                request_payload=request_payload,
                                heuristic_seed=heuristic_seed,
                                client=llm_client,
                                planner_name=args.planner_name,
                                waypoint_radius_cm=args.default_waypoint_radius_cm,
                            )
                            plan_debug = plan.get("debug") if isinstance(plan.get("debug"), dict) else {}
                            plan_debug.update(
                                {
                                    "planner_mode": str(args.planner_mode or "heuristic"),
                                    "route_mode": route_mode,
                                    "route_reason": route_reason,
                                }
                            )
                            plan["debug"] = plan_debug
                        except (LLMPlannerAdapterError, LLMPlannerClientError, error.URLError, TimeoutError) as exc:
                            if args.fallback_to_heuristic:
                                logger.warning("LLM planner request failed, using heuristic fallback: %s", exc)
                                plan = build_fallback_plan_from_seed(
                                    heuristic_seed,
                                    planner_name=args.planner_name,
                                    planner_mode=args.planner_mode,
                                    route_mode=route_mode,
                                    route_reason=route_reason,
                                    fallback_reason=str(exc),
                                    llm_model_name=str(args.llm_model or ""),
                                    llm_api_style=str(args.llm_api_style or ""),
                                )
                            else:
                                raise
                else:
                    seed_debug = heuristic_seed.get("debug") if isinstance(heuristic_seed.get("debug"), dict) else {}
                    seed_debug.update(
                        build_planner_debug_metadata(
                            source="heuristic",
                            planner_mode=args.planner_mode,
                            route_mode=route_mode,
                            route_reason=route_reason,
                            fallback_used=False,
                            llm_model_name="",
                            llm_api_style="",
                        )
                    )
                    heuristic_seed["planner_name"] = args.planner_name
                    heuristic_seed["debug"] = seed_debug
                    plan = heuristic_seed
                normalized_plan = coerce_plan_payload(
                    plan,
                    default_plan_id=f"external_plan_{request_payload.get('frame_id', 'unknown')}",
                    default_planner_name=args.planner_name,
                    default_semantic_subgoal="move_forward",
                    default_radius=args.default_waypoint_radius_cm,
                )
                self._send_json({"status": "ok", "plan": normalized_plan})
            except Exception as exc:
                logger.exception("Planner request failed")
                self._send_json({"status": "error", "message": str(exc)}, 500)

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("HTTP %s - %s", self.address_string(), fmt % args)

    return PlannerRequestHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Phase 2 planner service for UAV-Flow-Eval")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=5021, help="Planner HTTP port")
    parser.add_argument("--endpoint", default="/plan", help="Planner endpoint path")
    parser.add_argument("--action_endpoint", default="/action", help="Pure LLM action endpoint path")
    parser.add_argument("--planner_name", default="external_heuristic_planner", help="Planner name returned in plan payloads")
    parser.add_argument("--planner_mode", default="heuristic", choices=["heuristic", "llm", "hybrid"], help="Planner execution mode")
    parser.add_argument(
        "--planner_route_mode",
        default="auto",
        choices=["auto", "heuristic_only", "llm_only", "search_hybrid"],
        help="Explicit planner routing mode; use this to disable hybrid auto-switching during experiments",
    )
    parser.add_argument("--default_waypoint_radius_cm", type=float, default=60.0, help="Default waypoint radius")
    parser.add_argument("--llm_base_url", default="", help="Base URL for the LLM planner endpoint")
    parser.add_argument("--llm_api_key", default="", help="Explicit API key for the LLM planner")
    parser.add_argument("--llm_api_key_env", default="", help="Environment variable name used to load the API key if --llm_api_key is empty")
    parser.add_argument("--llm_model", default="", help="Model name used by the LLM planner")
    parser.add_argument(
        "--llm_api_style",
        default="openai_chat",
        choices=["openai_chat", "openai_responses", "anthropic_messages", "google_gemini", "google_genai_sdk"],
        help="Request/response style used for the LLM planner API",
    )
    parser.add_argument("--llm_endpoint_path", default="", help="Override the LLM endpoint path instead of using the default for the selected style")
    parser.add_argument("--llm_auth_header", default="Authorization", help="HTTP header used for planner API auth")
    parser.add_argument("--llm_auth_scheme", default="Bearer", help="Auth scheme prefix for the planner API header; set empty string to send the raw key")
    parser.add_argument("--llm_anthropic_version", default="2023-06-01", help="Anthropic API version header value when using anthropic_messages style")
    parser.add_argument("--llm_timeout_s", type=float, default=30.0, help="Timeout for a single LLM planner request")
    parser.add_argument("--llm_max_retries", type=int, default=1, help="Retry count for LLM planner requests")
    parser.add_argument("--llm_temperature", type=float, default=0.1, help="Sampling temperature used for LLM planner calls")
    parser.add_argument("--llm_max_output_tokens", type=int, default=800, help="Maximum output tokens used for the LLM planner call")
    parser.add_argument("--llm_input_mode", default="text_image", choices=["text", "text_image"], help="Whether to send only structured text or text plus the current image")
    parser.add_argument("--llm_disable_force_json", action="store_true", help="Disable explicit JSON response-format enforcement")
    parser.add_argument("--no_fallback_to_heuristic", dest="fallback_to_heuristic", action="store_false", help="Fail planner requests instead of falling back to the heuristic seed")
    parser.set_defaults(fallback_to_heuristic=True)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )
    llm_client = build_llm_client_from_args(args)
    handler = make_handler(args, llm_client)
    server = HTTPServer((args.host, args.port), handler)
    logger.info(
        "Planner server listening on http://%s:%s%s mode=%s llm_enabled=%s model=%s",
        args.host,
        args.port,
        args.endpoint,
        args.planner_mode,
        llm_client is not None,
        str(args.llm_model or ""),
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
