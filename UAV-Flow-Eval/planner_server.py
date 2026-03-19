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
import re
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

import numpy as np

from runtime_interfaces import (
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


def make_handler(args: argparse.Namespace):
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
                self._send_json({"status": "ok", "service": "planner_server", "planner_name": args.planner_name})
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
                self._send_json({"status": "ok", "request_example": example_request, "plan_example": example_plan})
            else:
                self._send_json({"status": "error", "message": "Not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path != args.endpoint:
                    self._send_json({"status": "error", "message": "Not found"}, 404)
                    return
                request_payload = self._read_json_body()
                plan = build_heuristic_plan(
                    request_payload,
                    planner_name=args.planner_name,
                    waypoint_radius_cm=args.default_waypoint_radius_cm,
                )
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
    parser.add_argument("--planner_name", default="external_heuristic_planner", help="Planner name returned in plan payloads")
    parser.add_argument("--default_waypoint_radius_cm", type=float, default=60.0, help="Default waypoint radius")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )
    handler = make_handler(args)
    server = HTTPServer((args.host, args.port), handler)
    logger.info("Planner server listening on http://%s:%s%s", args.host, args.port, args.endpoint)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
