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

from runtime_interfaces import build_plan_state, build_plan_request, build_waypoint, coerce_plan_payload, now_timestamp

logger = logging.getLogger(__name__)


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

    return build_plan_state(
        plan_id=f"external_plan_{request_payload.get('frame_id', 'unknown')}",
        planner_name=planner_name,
        generated_at=now_timestamp(),
        sector_id=sector_id,
        candidate_waypoints=[primary_waypoint, staging_waypoint],
        semantic_subgoal=semantic_subgoal,
        planner_confidence=confidence,
        should_replan=False,
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
