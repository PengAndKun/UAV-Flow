"""
Simple Phase 3 reflex policy service for UAV-Flow-Eval.

This keeps the runtime interface stable before swapping in a learned local
policy. It accepts a structured reflex request and returns a normalized
reflex-runtime payload.
"""

import argparse
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import numpy as np

from reflex_policy_model import build_model_reflex_runtime, load_artifact
from runtime_interfaces import (
    build_reflex_request,
    build_reflex_runtime_state,
    coerce_reflex_runtime_payload,
    now_timestamp,
)

logger = logging.getLogger(__name__)


def normalize_angle_deg(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def build_heuristic_reflex(request_payload: Dict[str, Any], policy_name: str) -> Dict[str, Any]:
    pose = request_payload.get("pose", {}) if isinstance(request_payload.get("pose"), dict) else {}
    current_waypoint = request_payload.get("current_waypoint", {}) if isinstance(request_payload.get("current_waypoint"), dict) else {}
    archive = request_payload.get("archive", {}) if isinstance(request_payload.get("archive"), dict) else {}
    runtime_debug = request_payload.get("runtime_debug", {}) if isinstance(request_payload.get("runtime_debug"), dict) else {}

    suggested_action = "hold_position"
    status = "idle"
    should_execute = False
    waypoint_distance_cm = 0.0
    yaw_error_deg = 0.0
    vertical_error_cm = 0.0
    progress_to_waypoint_cm = 0.0

    retrieval = archive.get("active_retrieval") if isinstance(archive.get("active_retrieval"), dict) else {}
    risk_score = float(runtime_debug.get("risk_score", 0.0))
    shield_triggered = bool(runtime_debug.get("shield_triggered", False))

    if current_waypoint:
        px = float(pose.get("x", 0.0))
        py = float(pose.get("y", 0.0))
        pz = float(pose.get("z", 0.0))
        pyaw = float(pose.get("yaw", 0.0))
        dx = float(current_waypoint.get("x", px)) - px
        dy = float(current_waypoint.get("y", py)) - py
        dz = float(current_waypoint.get("z", pz)) - pz
        horizontal_distance = float(np.hypot(dx, dy))
        waypoint_distance_cm = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        vertical_error_cm = dz
        desired_yaw = normalize_angle_deg(np.degrees(np.arctan2(dy, dx))) if horizontal_distance > 1e-6 else pyaw
        yaw_error_deg = normalize_angle_deg(desired_yaw - pyaw)
        previous_distance = float(runtime_debug.get("previous_waypoint_distance_cm", waypoint_distance_cm))
        progress_to_waypoint_cm = previous_distance - waypoint_distance_cm
        status = "tracking_waypoint"

        if shield_triggered or risk_score >= 0.85:
            suggested_action = "shield_hold"
            status = "shield_hold"
        elif abs(yaw_error_deg) > 12.0:
            suggested_action = "yaw_left" if yaw_error_deg < 0.0 else "yaw_right"
            should_execute = True
        elif abs(vertical_error_cm) > 12.0:
            suggested_action = "down" if vertical_error_cm < 0.0 else "up"
            should_execute = True
        elif waypoint_distance_cm > 30.0:
            suggested_action = "forward"
            should_execute = True
        else:
            suggested_action = "hold_position"
            status = "waypoint_arrived"
    elif shield_triggered or risk_score >= 0.85:
        suggested_action = "shield_hold"
        status = "shield_hold"
    else:
        suggested_action = "scan_hover"
        status = "idle"

    return build_reflex_runtime_state(
        mode="external_policy_stub",
        policy_name=policy_name,
        source="external",
        status=status,
        suggested_action=suggested_action,
        should_execute=should_execute,
        last_trigger=str(request_payload.get("context", {}).get("trigger", "")),
        last_latency_ms=0.0,
        waypoint_distance_cm=waypoint_distance_cm,
        yaw_error_deg=yaw_error_deg,
        vertical_error_cm=vertical_error_cm,
        progress_to_waypoint_cm=progress_to_waypoint_cm,
        retrieval_cell_id=str(retrieval.get("cell_id", "")),
        retrieval_score=float(retrieval.get("retrieval_score", 0.0) or 0.0),
        retrieval_semantic_subgoal=str(retrieval.get("semantic_subgoal", "")),
        risk_score=risk_score,
        shield_triggered=shield_triggered,
    )


def make_handler(args: argparse.Namespace, model_artifact: Optional[Dict[str, Any]]):
    class ReflexRequestHandler(BaseHTTPRequestHandler):
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
                        "service": "reflex_policy_server",
                        "policy_name": args.policy_name,
                        "policy_mode": str(model_artifact.get("model_type", "prototype_model")) if model_artifact else "heuristic_stub",
                        "model_artifact_path": args.model_artifact or "",
                    }
                )
            elif parsed.path == "/schema":
                example_request = build_reflex_request(
                    policy_name=args.policy_name,
                    frame_id="frame_000001",
                    timestamp=now_timestamp(),
                    task_label="move right 3 meters",
                    pose={"x": 0.0, "y": 0.0, "z": 120.0, "yaw": 0.0},
                    depth={"min_depth": 120.0, "max_depth": 900.0, "front_min_depth": 180.0, "front_mean_depth": 320.0},
                    plan={"planner_name": "external_heuristic_planner", "semantic_subgoal": "move_forward", "planner_confidence": 0.65},
                    current_waypoint={"x": 200.0, "y": 0.0, "z": 120.0, "yaw": 0.0, "radius": 60.0, "semantic_label": "move_forward"},
                    archive={"active_retrieval": {"cell_id": "cell_demo", "retrieval_score": 2.3, "semantic_subgoal": "move_forward"}},
                    runtime_debug={"risk_score": 0.1, "shield_triggered": False},
                    context={"trigger": "manual_request"},
                )
                example_reflex = (
                    build_model_reflex_runtime(request_payload=example_request, artifact=model_artifact)
                    if model_artifact
                    else build_heuristic_reflex(example_request, args.policy_name)
                )
                self._send_json(
                    {
                        "status": "ok",
                        "request_example": example_request,
                        "reflex_example": example_reflex,
                        "policy_mode": str(model_artifact.get("model_type", "prototype_model")) if model_artifact else "heuristic_stub",
                    }
                )
            else:
                self._send_json({"status": "error", "message": "Not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path != args.endpoint:
                    self._send_json({"status": "error", "message": "Not found"}, 404)
                    return
                request_payload = self._read_json_body()
                reflex = (
                    build_model_reflex_runtime(request_payload=request_payload, artifact=model_artifact)
                    if model_artifact
                    else build_heuristic_reflex(request_payload, args.policy_name)
                )
                normalized = coerce_reflex_runtime_payload(
                    reflex,
                    default_mode="prototype_policy" if model_artifact else "external_policy_stub",
                    default_policy_name=args.policy_name,
                    default_source="external_model" if model_artifact else "external",
                )
                self._send_json({"status": "ok", "reflex_runtime": normalized})
            except Exception as exc:
                logger.exception("Reflex policy request failed")
                self._send_json({"status": "error", "message": str(exc)}, 500)

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("HTTP %s - %s", self.address_string(), fmt % args)

    return ReflexRequestHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple Phase 3 reflex policy service for UAV-Flow-Eval")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=5022, help="Reflex policy HTTP port")
    parser.add_argument("--endpoint", default="/reflex_policy", help="Reflex policy endpoint path")
    parser.add_argument("--policy_name", default="external_reflex_stub", help="Policy name returned in reflex payloads")
    parser.add_argument("--model_artifact", default="", help="Optional trained artifact JSON; when provided, use prototype-model inference")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )
    model_artifact = load_artifact(args.model_artifact) if args.model_artifact else None
    if model_artifact:
        logger.info(
            "Loaded reflex model artifact from %s policy_name=%s samples=%s",
            args.model_artifact,
            model_artifact.get("policy_name", ""),
            model_artifact.get("training_summary", {}).get("sample_count", 0),
        )
    handler = make_handler(args, model_artifact)
    server = HTTPServer((args.host, args.port), handler)
    logger.info("Reflex policy server listening on http://%s:%s%s", args.host, args.port, args.endpoint)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
