"""
Rule-based UAV control server compatible with the OpenVLA-UAV HTTP API.

This server exposes `/predict` and `/reset` so it can be used directly by
`batch_run_act_all.py`. It does not load a large model; instead it converts
common natural-language motion commands into deterministic UAV motion.
"""

import argparse
import base64
import json
import logging
import math
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def normalize_angle_deg(angle_deg: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def angle_delta_deg(target_deg: float, current_deg: float) -> float:
    """Shortest signed angle delta from current to target."""
    return normalize_angle_deg(target_deg - current_deg)


def move_towards_point(
    current_xyz: Tuple[float, float, float],
    target_xyz: Tuple[float, float, float],
    max_step_cm: float,
) -> Tuple[float, float, float]:
    """Move from current_xyz toward target_xyz by up to max_step_cm."""
    dx = target_xyz[0] - current_xyz[0]
    dy = target_xyz[1] - current_xyz[1]
    dz = target_xyz[2] - current_xyz[2]
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)
    if distance <= max_step_cm or distance <= 1e-6:
        return target_xyz
    scale = max_step_cm / distance
    return (
        current_xyz[0] + dx * scale,
        current_xyz[1] + dy * scale,
        current_xyz[2] + dz * scale,
    )


def step_towards_angle(current_deg: float, target_deg: float, max_step_deg: float) -> float:
    """Rotate from current_deg toward target_deg by up to max_step_deg."""
    delta = angle_delta_deg(target_deg, current_deg)
    if abs(delta) <= max_step_deg:
        return normalize_angle_deg(target_deg)
    return normalize_angle_deg(current_deg + math.copysign(max_step_deg, delta))


class RuleBasedUAVActionAgent:
    """A lightweight policy server that maps text instructions to UAV actions."""

    def __init__(
        self,
        host: str,
        port: int,
        position_step_cm: float,
        yaw_step_deg: float,
        default_turn_deg: float,
    ) -> None:
        self.host = host
        self.port = port
        self.position_step_cm = position_step_cm
        self.yaw_step_deg = yaw_step_deg
        self.default_turn_deg = default_turn_deg
    
    def handle_predict_payload(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        try:
            self.decode_image(data.get("image"))
            instruction = str(data.get("instr", "")).strip()
            proprio = self.normalize_proprio(data.get("proprio"))
            target_local = self.normalize_target_local(data.get("target_local"))

            action, action_ori, debug = self.plan_action(
                instruction=instruction,
                proprio=proprio,
                target_local=target_local,
            )
            return (
                {
                    "status": "success",
                    "action": [action],
                    "action_ori": [action_ori],
                    "message": "Rule-based action generated successfully",
                    "debug": debug,
                },
                200,
            )
        except Exception as exc:
            logger.exception("Rule-based prediction failed")
            return {"status": "error", "message": str(exc)}, 500

    @staticmethod
    def handle_reset_payload() -> Tuple[Dict[str, Any], int]:
        return {"status": "success", "message": "Rule-based controller is stateless"}, 200

    @staticmethod
    def handle_health_payload() -> Tuple[Dict[str, Any], int]:
        return {"status": "ok"}, 200

    def run(self) -> None:
        logger.info("Starting rule-based UAV server on %s:%s", self.host, self.port)

        agent = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path != "/health":
                    self.send_json({"status": "error", "message": "Not found"}, 404)
                    return
                payload, status = agent.handle_health_payload()
                self.send_json(payload, status)

            def do_POST(self) -> None:  # noqa: N802
                content_length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
                try:
                    data = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                except json.JSONDecodeError:
                    self.send_json({"status": "error", "message": "Invalid JSON body"}, 400)
                    return

                if self.path == "/predict":
                    payload, status = agent.handle_predict_payload(data)
                elif self.path == "/reset":
                    payload, status = agent.handle_reset_payload()
                else:
                    payload, status = {"status": "error", "message": "Not found"}, 404
                self.send_json(payload, status)

            def log_message(self, fmt: str, *args: Any) -> None:
                logger.debug("HTTP %s - %s", self.address_string(), fmt % args)

            def send_json(self, payload: Dict[str, Any], status: int) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        with ThreadingHTTPServer((self.host, self.port), RequestHandler) as server:
            server.serve_forever()

    @staticmethod
    def decode_image(image_b64: Optional[str]) -> Optional[bytes]:
        """Decode the input image to keep the same request contract as OpenVLA-UAV."""
        if not image_b64:
            return None
        return base64.b64decode(image_b64)

    @staticmethod
    def normalize_proprio(proprio_raw: Any) -> List[float]:
        """Normalize the proprio payload to [x, y, z, yaw_deg]."""
        if not isinstance(proprio_raw, list):
            return [0.0, 0.0, 0.0, 0.0]
        values = [float(v) for v in proprio_raw[:4]]
        while len(values) < 4:
            values.append(0.0)
        values[3] = normalize_angle_deg(values[3])
        return values

    @staticmethod
    def normalize_target_local(target_local: Any) -> Optional[List[float]]:
        """Normalize optional target-local coordinates."""
        if not isinstance(target_local, list) or len(target_local) < 3:
            return None
        return [float(target_local[0]), float(target_local[1]), float(target_local[2])]

    @staticmethod
    def extract_distance_cm(text: str, pattern: str) -> Optional[float]:
        match = re.search(pattern, text)
        if not match:
            return None
        return float(match.group(1)) * 100.0

    @staticmethod
    def extract_angle_deg(text: str, pattern: str) -> Optional[float]:
        match = re.search(pattern, text)
        if not match:
            return None
        return float(match.group(1))

    @staticmethod
    def yaw_to_target_deg(
        current_xy: Tuple[float, float],
        target_xy: Tuple[float, float],
    ) -> float:
        """Return the yaw that points from current_xy to target_xy."""
        dx = target_xy[0] - current_xy[0]
        dy = target_xy[1] - current_xy[1]
        return normalize_angle_deg(math.degrees(math.atan2(dy, dx)))

    def solve_face_target(
        self,
        current_pose: List[float],
        target_local: List[float],
    ) -> Optional[Dict[str, Any]]:
        target_yaw_deg = self.yaw_to_target_deg(
            (current_pose[0], current_pose[1]),
            (target_local[0], target_local[1]),
        )
        return {
            "target_pos": (current_pose[0], current_pose[1], current_pose[2]),
            "target_yaw_deg": target_yaw_deg,
            "command": "face_target",
        }

    def solve_navigate_away_from_target(
        self,
        current_pose: List[float],
        target_local: List[float],
        radius_cm: float,
    ) -> Dict[str, Any]:
        vec_x = current_pose[0] - target_local[0]
        vec_y = current_pose[1] - target_local[1]
        vec_norm = math.hypot(vec_x, vec_y)
        if vec_norm <= 1e-6:
            vec_x, vec_y, vec_norm = -1.0, 0.0, 1.0
        desired_x = target_local[0] + radius_cm * vec_x / vec_norm
        desired_y = target_local[1] + radius_cm * vec_y / vec_norm
        desired_z = current_pose[2]
        target_yaw_deg = self.yaw_to_target_deg((desired_x, desired_y), (target_local[0], target_local[1]))
        return {
            "target_pos": (desired_x, desired_y, desired_z),
            "target_yaw_deg": target_yaw_deg,
            "command": "navigate_away_from_target",
        }

    def solve_orbit_target(
        self,
        current_pose: List[float],
        target_local: List[float],
        radius_cm: float,
        clockwise: bool,
    ) -> Dict[str, Any]:
        rel_x = current_pose[0] - target_local[0]
        rel_y = current_pose[1] - target_local[1]
        current_radius = math.hypot(rel_x, rel_y)
        if current_radius <= 1e-6:
            rel_x, rel_y, current_radius = -radius_cm, 0.0, radius_cm

        angle = math.atan2(rel_y, rel_x)
        orbit_step_cm = min(self.position_step_cm, 10.0)
        step_angle = orbit_step_cm / max(radius_cm, 1e-6)
        signed_step = -step_angle if clockwise else step_angle

        if abs(current_radius - radius_cm) > self.position_step_cm:
            desired_x = target_local[0] + radius_cm * rel_x / current_radius
            desired_y = target_local[1] + radius_cm * rel_y / current_radius
        else:
            desired_x = target_local[0] + radius_cm * math.cos(angle + signed_step)
            desired_y = target_local[1] + radius_cm * math.sin(angle + signed_step)

        desired_z = current_pose[2]
        target_yaw_deg = self.yaw_to_target_deg((desired_x, desired_y), (target_local[0], target_local[1]))
        return {
            "target_pos": (desired_x, desired_y, desired_z),
            "target_yaw_deg": target_yaw_deg,
            "command": "orbit_target_clockwise" if clockwise else "orbit_target_counterclockwise",
        }

    def solve_motion_command(
        self,
        instruction: str,
        current_pose: List[float],
        target_local: Optional[List[float]],
    ) -> Dict[str, Any]:
        text = instruction.lower().strip().rstrip(".")

        if target_local is not None and (
            "turn to the direction of" in text
            or "face toward" in text
            or "face towards" in text
        ):
            return self.solve_face_target(current_pose, target_local)

        if target_local is not None:
            nav_match = re.search(r"navigate to a point ([0-9.]+)\s*meters? away from", text)
            if nav_match:
                return self.solve_navigate_away_from_target(current_pose, target_local, float(nav_match.group(1)) * 100.0)

            orbit_patterns = [
                r"(orbit|circle around).*?radius of\s*([0-9.]+)\s*meters?",
                r"(orbit|circle around).*?at a\s*([0-9.]+)\s*-\s*meter radius",
                r"(orbit|circle around).*?at a\s*([0-9.]+)\s*meter radius",
            ]
            for pattern in orbit_patterns:
                orbit_match = re.search(pattern, text)
                if orbit_match:
                    radius_cm = float(orbit_match.group(2)) * 100.0
                    clockwise = any(token in text for token in ("clockwise", "right direction"))
                    return self.solve_orbit_target(current_pose, target_local, radius_cm, clockwise)

        right_turn = re.search(r"(turn|rotate)\s+(?:right|clockwise)(?:\s+by)?\s*([0-9.]+)?\s*degrees?", text)
        rotate_right = re.search(r"rotate\s+([0-9.]+)\s*degrees?\s+to\s+the\s+right", text)
        if right_turn or rotate_right or "turn right" in text or "turn clockwise" in text:
            if right_turn and right_turn.group(2):
                turn_deg = float(right_turn.group(2))
            elif rotate_right:
                turn_deg = float(rotate_right.group(1))
            else:
                turn_deg = self.default_turn_deg
            return {
                "target_pos": (current_pose[0], current_pose[1], current_pose[2]),
                "target_yaw_deg": normalize_angle_deg(current_pose[3] + turn_deg),
                "command": "turn_right",
            }

        left_turn = re.search(r"(turn|rotate)\s+(?:left|counterclockwise)(?:\s+by)?\s*([0-9.]+)?\s*degrees?", text)
        rotate_left = re.search(r"rotate\s+([0-9.]+)\s*degrees?\s+to\s+the\s+left", text)
        if left_turn or rotate_left or "turn left" in text or "turn counterclockwise" in text:
            if left_turn and left_turn.group(2):
                turn_deg = float(left_turn.group(2))
            elif rotate_left:
                turn_deg = float(rotate_left.group(1))
            else:
                turn_deg = self.default_turn_deg
            return {
                "target_pos": (current_pose[0], current_pose[1], current_pose[2]),
                "target_yaw_deg": normalize_angle_deg(current_pose[3] - turn_deg),
                "command": "turn_left",
            }

        altitude_with_angle = re.search(
            r"(climb|ascend) to an altitude of ([0-9.]+)\s*meters?.*?([0-9.]+)\s*(?:-\s*)?degrees?",
            text,
        )
        if altitude_with_angle:
            target_z = float(altitude_with_angle.group(2)) * 100.0
            angle_deg = float(altitude_with_angle.group(3))
            planar_x = target_z / max(math.tan(math.radians(angle_deg)), 1e-6)
            return {
                "target_pos": (planar_x, 0.0, target_z),
                "target_yaw_deg": current_pose[3],
                "command": "climb_to_altitude_with_angle",
            }

        altitude_only = re.search(r"(ascend|climb) to an altitude of ([0-9.]+)\s*meters?", text)
        if altitude_only:
            target_z = float(altitude_only.group(2)) * 100.0
            return {
                "target_pos": (0.0, 0.0, target_z),
                "target_yaw_deg": current_pose[3],
                "command": "ascend_to_altitude",
            }

        descend_by_angle = re.search(
            r"(lower altitude by|descend)\s+([0-9.]+)\s*meters?.*?([0-9.]+)\s*(?:-\s*)?degrees?",
            text,
        )
        if descend_by_angle and "altitude by" in text:
            drop_cm = float(descend_by_angle.group(2)) * 100.0
            angle_deg = float(descend_by_angle.group(3))
            planar_x = drop_cm / max(math.tan(math.radians(angle_deg)), 1e-6)
            return {
                "target_pos": (planar_x, 0.0, -drop_cm),
                "target_yaw_deg": current_pose[3],
                "command": "descend_by_altitude_with_angle",
            }

        ascend_distance_angle = re.search(r"ascend\s+([0-9.]+)\s*meters?.*?([0-9.]+)\s*(?:-\s*)?degrees?", text)
        if ascend_distance_angle and "altitude" not in text:
            distance_cm = float(ascend_distance_angle.group(1)) * 100.0
            angle_deg = float(ascend_distance_angle.group(2))
            target_x = distance_cm * math.cos(math.radians(angle_deg))
            target_z = distance_cm * math.sin(math.radians(angle_deg))
            return {
                "target_pos": (target_x, 0.0, target_z),
                "target_yaw_deg": current_pose[3],
                "command": "ascend_by_distance_and_angle",
            }

        descend_distance_angle = re.search(r"descend\s+([0-9.]+)\s*meters?.*?([0-9.]+)\s*(?:-\s*)?degrees?", text)
        if descend_distance_angle:
            distance_cm = float(descend_distance_angle.group(1)) * 100.0
            angle_deg = float(descend_distance_angle.group(2))
            target_x = distance_cm * math.cos(math.radians(angle_deg))
            target_z = -distance_cm * math.sin(math.radians(angle_deg))
            return {
                "target_pos": (target_x, 0.0, target_z),
                "target_yaw_deg": current_pose[3],
                "command": "descend_by_distance_and_angle",
            }

        forward_match = re.search(r"move forward\s+([0-9.]+)\s*meters?", text)
        if forward_match:
            return {
                "target_pos": (float(forward_match.group(1)) * 100.0, 0.0, 0.0),
                "target_yaw_deg": current_pose[3],
                "command": "move_forward",
            }

        backward_match = re.search(r"move backward\s+([0-9.]+)\s*meters?", text)
        if backward_match:
            return {
                "target_pos": (-float(backward_match.group(1)) * 100.0, 0.0, 0.0),
                "target_yaw_deg": current_pose[3],
                "command": "move_backward",
            }

        lateral_match = re.search(
            r"move\s+([0-9.]+)\s*meters?\s+(?:to\s+the\s+)?(left|right)(?:\s+at\s+a?\s*([0-9.]+)\s*(?:-\s*)?degrees?(?:\s+angle)?)?",
            text,
        )
        if lateral_match:
            distance_cm = float(lateral_match.group(1)) * 100.0
            direction = lateral_match.group(2)
            angle_deg = float(lateral_match.group(3)) if lateral_match.group(3) else 90.0
            sign_y = 1.0 if direction == "right" else -1.0
            return {
                "target_pos": (
                    distance_cm * math.cos(math.radians(angle_deg)),
                    sign_y * distance_cm * math.sin(math.radians(angle_deg)),
                    0.0,
                ),
                "target_yaw_deg": current_pose[3],
                "command": f"move_{direction}",
            }

        angled_lateral_match = re.search(
            r"move\s+([0-9.]+)\s*meters?\s+at\s+([0-9.]+)\s*(?:-\s*)?degrees?\s+to\s+the\s+(left|right)",
            text,
        )
        if angled_lateral_match:
            distance_cm = float(angled_lateral_match.group(1)) * 100.0
            angle_deg = float(angled_lateral_match.group(2))
            direction = angled_lateral_match.group(3)
            sign_y = 1.0 if direction == "right" else -1.0
            return {
                "target_pos": (
                    distance_cm * math.cos(math.radians(angle_deg)),
                    sign_y * distance_cm * math.sin(math.radians(angle_deg)),
                    0.0,
                ),
                "target_yaw_deg": current_pose[3],
                "command": f"move_{direction}_by_angle",
            }

        shorthand_lateral = re.search(
            r"move\s+([0-9.]+)\s*meters?\s+([0-9.]+)\s*(?:-\s*)?degrees?\s+(leftward|rightward)",
            text,
        )
        if shorthand_lateral:
            distance_cm = float(shorthand_lateral.group(1)) * 100.0
            angle_deg = float(shorthand_lateral.group(2))
            direction = shorthand_lateral.group(3)
            sign_y = 1.0 if direction == "rightward" else -1.0
            return {
                "target_pos": (
                    distance_cm * math.cos(math.radians(angle_deg)),
                    sign_y * distance_cm * math.sin(math.radians(angle_deg)),
                    0.0,
                ),
                "target_yaw_deg": current_pose[3],
                "command": f"move_{direction}",
            }

        return {
            "target_pos": (current_pose[0], current_pose[1], current_pose[2]),
            "target_yaw_deg": current_pose[3],
            "command": "hold_position",
        }

    def plan_action(
        self,
        instruction: str,
        proprio: List[float],
        target_local: Optional[List[float]],
    ) -> Tuple[List[float], List[float], Dict[str, Any]]:
        current_x, current_y, current_z, current_yaw_deg = proprio
        plan = self.solve_motion_command(instruction, proprio, target_local)
        target_pos = plan["target_pos"]
        target_yaw_deg = normalize_angle_deg(float(plan["target_yaw_deg"]))

        next_x, next_y, next_z = move_towards_point(
            (current_x, current_y, current_z),
            target_pos,
            self.position_step_cm,
        )
        next_yaw_deg = step_towards_angle(current_yaw_deg, target_yaw_deg, self.yaw_step_deg)

        delta_x = next_x - current_x
        delta_y = next_y - current_y
        delta_z = next_z - current_z
        delta_yaw_deg = angle_delta_deg(next_yaw_deg, current_yaw_deg)

        action = [next_x, next_y, next_z, math.radians(next_yaw_deg)]
        action_ori = [delta_x, delta_y, delta_z, math.radians(delta_yaw_deg)]
        debug = {
            "instruction": instruction,
            "command": plan["command"],
            "target_pos": [float(v) for v in target_pos],
            "target_yaw_deg": target_yaw_deg,
            "current_pose": [current_x, current_y, current_z, current_yaw_deg],
            "target_local": target_local,
        }
        logger.info("command=%s current=%s target=%s next=%s", plan["command"], proprio, target_pos, action)
        return action, action_ori, debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based UAV control server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=5007, help="Port to bind")
    parser.add_argument("--position_step_cm", type=float, default=20.0, help="Max local-frame translation per step")
    parser.add_argument("--yaw_step_deg", type=float, default=3.0, help="Max yaw change per step in degrees")
    parser.add_argument("--default_turn_deg", type=float, default=30.0, help="Default turn amount for turn-left/turn-right without an explicit angle")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    agent = RuleBasedUAVActionAgent(
        host=args.host,
        port=args.port,
        position_step_cm=args.position_step_cm,
        yaw_step_deg=args.yaw_step_deg,
        default_turn_deg=args.default_turn_deg,
    )
    agent.run()


if __name__ == "__main__":
    main()
