"""
Launch and hold the Unreal UAV environment, then expose simple HTTP controls.

This file is the "game side" process:
- launches the Unreal environment
- keeps the UAV alive in the scene
- exposes move/screenshot/frame/state endpoints

Use `uav_control_panel.py` as the separate controller/UI process.
"""

import argparse
import json
import logging
import os
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import cv2
import gym
import gym_unrealcv
import numpy as np

from gym_unrealcv.envs.wrappers import augmentation, configUE, time_dilation

from batch_run_act_all import (
    configure_player_viewport,
    create_obj_if_needed,
    get_follow_preview_cam_id,
    get_policy_cam_id,
    get_third_person_preview_image,
    maybe_override_env_binary,
    set_cam,
    set_free_view_near_pose,
    validate_env_binary_exists,
)

logger = logging.getLogger(__name__)


def normalize_angle_deg(angle_deg: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (angle_deg + 180.0) % 360.0 - 180.0


def build_obj_info(task_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build object placement info from a task JSON."""
    if "obj_id" not in task_data or "use_obj" not in task_data:
        return None
    if "target_pos" in task_data and isinstance(task_data["target_pos"], list) and len(task_data["target_pos"]) == 6:
        obj_pos = task_data["target_pos"][:3]
        obj_rot = task_data["target_pos"][3:]
    else:
        obj_pos = task_data.get("obj_pos")
        obj_rot = task_data.get("obj_rot", [0, 0, 0])
    if obj_pos is None:
        return None
    return {
        "use_obj": task_data["use_obj"],
        "obj_id": task_data["obj_id"],
        "obj_pos": obj_pos,
        "obj_rot": obj_rot,
    }


class UAVControlBackend:
    """Own the Unreal environment and provide thread-safe control methods."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.RLock()
        self.last_raw_frame: Optional[np.ndarray] = None
        self.last_capture: Optional[Dict[str, Any]] = None
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.command_task_yaw_deg: float = 0.0
        self.last_action: str = "idle"

        os.makedirs(self.args.capture_dir, exist_ok=True)

        maybe_override_env_binary(self.args.env_id, self.args.env_bin_win)
        validate_env_binary_exists(self.args.env_id)

        self.env = gym.make(self.args.env_id)
        if int(self.args.time_dilation) > 0:
            self.env = time_dilation.TimeDilationWrapper(self.env, int(self.args.time_dilation))
        self.env.unwrapped.agents_category = ["drone"]
        self.env = configUE.ConfigUEWrapper(self.env, resolution=(self.args.window_width, self.args.window_height))
        # Track environments expect tracker+target during reset, so keep the
        # internal population at 2 and hide any extra UAVs after startup.
        self.env = augmentation.RandomPopulationWrapper(self.env, 2, 2, random_target=False)
        self.env.seed(int(self.args.seed))
        self.env.reset()

        self.player_name = self.env.unwrapped.player_list[0]
        self.env.unwrapped.unrealcv.set_phy(self.player_name, 0)
        logger.info("Active player list after reset: %s", self.env.unwrapped.player_list)
        self.hide_non_primary_agents()

        configure_player_viewport(
            self.env,
            self.args.viewport_mode,
            (self.args.viewport_offset_x, self.args.viewport_offset_y, self.args.viewport_offset_z),
            (self.args.viewport_roll, self.args.viewport_pitch, self.args.viewport_yaw),
        )

        self.policy_cam_id = get_policy_cam_id(self.env)
        self.preview_cam_id = get_follow_preview_cam_id(self.env, self.policy_cam_id)
        self.free_view_offset = (
            self.args.free_view_offset_x,
            self.args.free_view_offset_y,
            self.args.free_view_offset_z,
        )
        self.free_view_rotation = (
            self.args.free_view_roll,
            self.args.free_view_pitch,
            self.args.free_view_yaw,
        )
        self.preview_offset = (
            self.args.preview_offset_x,
            self.args.preview_offset_y,
            self.args.preview_offset_z,
        )
        self.preview_rotation = (
            self.args.preview_roll,
            self.args.preview_pitch,
            self.args.preview_yaw,
        )

        self.apply_initial_task_or_spawn()
        self.command_task_yaw_deg = self.get_task_pose()[3]
        self.position_free_view_once()
        self.refresh_preview()

    def apply_initial_task_or_spawn(self) -> None:
        """Optionally position the drone and target from a task JSON or CLI pose."""
        if self.args.task_json:
            with open(self.args.task_json, "r", encoding="utf-8") as f:
                task_data = json.load(f)
            obj_info = build_obj_info(task_data)
            create_obj_if_needed(self.env, obj_info)
            initial_pos = task_data.get("initial_pos")
            if isinstance(initial_pos, list) and len(initial_pos) >= 5:
                self.set_task_pose(initial_pos[0:3], float(initial_pos[4]))
                logger.info("Loaded initial pose from task JSON: %s", initial_pos[:5])
                return

        if self.args.spawn_x is not None and self.args.spawn_y is not None and self.args.spawn_z is not None:
            spawn_yaw = self.args.spawn_yaw if self.args.spawn_yaw is not None else self.get_task_pose()[3]
            self.set_task_pose(
                [self.args.spawn_x, self.args.spawn_y, self.args.spawn_z],
                spawn_yaw,
            )
            logger.info(
                "Applied manual spawn pose: x=%.3f y=%.3f z=%.3f yaw=%.3f",
                self.args.spawn_x,
                self.args.spawn_y,
                self.args.spawn_z,
                spawn_yaw,
            )

    def hide_non_primary_agents(self) -> None:
        """Hide any extra UAV agents required internally by the Track env."""
        extra_players = list(self.env.unwrapped.player_list[1:])
        for idx, player_name in enumerate(extra_players, start=1):
            try:
                hide_pos = [0.0, 0.0, -10000.0 - 100.0 * idx]
                self.env.unwrapped.unrealcv.set_phy(player_name, 0)
                self.env.unwrapped.unrealcv.set_obj_location(player_name, hide_pos)
                self.env.unwrapped.unrealcv.set_obj_rotation(player_name, [0.0, 0.0, 0.0])
                logger.info("Hid extra UAV agent %s at %s", player_name, hide_pos)
            except Exception as exc:
                logger.warning("Failed to hide extra UAV agent %s: %s", player_name, exc)

    def get_env_yaw_deg(self) -> float:
        """Read actor yaw from Unreal."""
        rotation = self.env.unwrapped.unrealcv.get_obj_rotation(self.player_name)
        if isinstance(rotation, (list, tuple)) and len(rotation) > 1:
            return normalize_angle_deg(float(rotation[1]))
        return 0.0

    def get_task_pose(self) -> List[float]:
        """Return [x, y, z, yaw] in the task-JSON yaw convention."""
        location = self.env.unwrapped.unrealcv.get_obj_location(self.player_name)
        task_yaw = normalize_angle_deg(self.get_env_yaw_deg() + 180.0)
        return [float(location[0]), float(location[1]), float(location[2]), task_yaw]

    def set_task_pose(self, position: List[float], yaw_deg: float) -> None:
        """Set pose using the task-JSON yaw convention."""
        self.env.unwrapped.unrealcv.set_obj_location(self.player_name, list(position))
        self.env.unwrapped.unrealcv.set_rotation(self.player_name, float(yaw_deg) - 180.0)

    def set_env_yaw_deg(self, yaw_deg: float) -> None:
        """Set pose yaw in the raw Unreal/actor convention."""
        self.env.unwrapped.unrealcv.set_rotation(self.player_name, float(normalize_angle_deg(yaw_deg)))

    def get_control_yaw_deg(self) -> float:
        """Return the yaw used by relative keyboard movement."""
        if self.args.movement_yaw_mode == "task":
            return self.command_task_yaw_deg
        if self.args.movement_yaw_mode == "camera" and self.args.preview_mode == "first_person":
            try:
                set_cam(self.env, self.policy_cam_id)
                cam_rotation = self.env.unwrapped.unrealcv.get_cam_rotation(self.policy_cam_id)
                if isinstance(cam_rotation, (list, tuple)) and len(cam_rotation) > 1:
                    return normalize_angle_deg(float(cam_rotation[1]))
            except Exception as exc:
                logger.debug("Failed to read control camera yaw, fallback to UAV yaw: %s", exc)
        return self.get_env_yaw_deg()

    def position_free_view_once(self) -> None:
        """Optionally place the native Unreal free view near the UAV once at startup."""
        if self.args.viewport_mode != "free" or not self.args.follow_free_view:
            return
        pose = self.get_task_pose()
        focus_pose = [pose[0], pose[1], pose[2], 0.0, pose[3]]
        set_free_view_near_pose(self.env, focus_pose, self.free_view_offset, self.free_view_rotation)

    def get_preview_frame(self) -> np.ndarray:
        """Capture the current UAV preview frame."""
        if self.args.preview_mode == "third_person":
            return get_third_person_preview_image(
                self.env,
                self.preview_cam_id,
                self.preview_offset,
                self.preview_rotation,
            )
        set_cam(self.env, self.policy_cam_id)
        return self.env.unwrapped.unrealcv.get_image(self.policy_cam_id, "lit")

    def refresh_preview(self) -> np.ndarray:
        """Refresh and cache the latest frame."""
        frame = self.get_preview_frame()
        self.last_raw_frame = frame
        return frame

    def get_state(self) -> Dict[str, Any]:
        """Return current controller state."""
        task_pose = self.get_task_pose()
        env_yaw = self.get_env_yaw_deg()
        control_yaw = self.get_control_yaw_deg()
        return {
            "status": "ok",
            "env_id": self.args.env_id,
            "player_name": self.player_name,
            "player_count": 1,
            "internal_player_count": len(self.env.unwrapped.player_list),
            "movement_yaw_mode": self.args.movement_yaw_mode,
            "last_action": self.last_action,
            "pose": {
                "x": task_pose[0],
                "y": task_pose[1],
                "z": task_pose[2],
                "yaw": control_yaw,
                "command_yaw": self.command_task_yaw_deg,
                "task_yaw": task_pose[3],
                "uav_yaw": env_yaw,
            },
            "preview_mode": self.args.preview_mode,
            "last_capture": self.last_capture,
        }

    def move_relative(
        self,
        forward_cm: float = 0.0,
        right_cm: float = 0.0,
        up_cm: float = 0.0,
        yaw_delta_deg: float = 0.0,
        action_name: str = "custom",
    ) -> Dict[str, Any]:
        """Move the UAV relative to a stable control yaw."""
        with self.lock:
            x, y, z, _actual_task_yaw = self.get_task_pose()
            move_yaw_deg = self.get_control_yaw_deg()
            theta = np.radians(move_yaw_deg)
            delta_x = float(forward_cm * np.cos(theta) - right_cm * np.sin(theta))
            delta_y = float(forward_cm * np.sin(theta) + right_cm * np.cos(theta))
            new_pos = [x + delta_x, y + delta_y, z + up_cm]
            self.command_task_yaw_deg = normalize_angle_deg(self.command_task_yaw_deg + yaw_delta_deg)
            self.set_task_pose(new_pos, self.command_task_yaw_deg)
            self.refresh_preview()
            self.last_action = action_name
            actual_task_yaw_after = self.get_task_pose()[3]
            actual_uav_yaw_after = self.get_env_yaw_deg()
            logger.info(
                "Remote action=%s local=(fwd=%.1f,right=%.1f,up=%.1f,yaw=%.1f) "
                "world=(dx=%.1f,dy=%.1f,dz=%.1f) pos=(%.1f, %.1f, %.1f) "
                "move_yaw=%.1f command_yaw=%.1f actual_task_yaw=%.1f actual_uav_yaw=%.1f",
                action_name,
                forward_cm,
                right_cm,
                up_cm,
                yaw_delta_deg,
                delta_x,
                delta_y,
                up_cm,
                new_pos[0],
                new_pos[1],
                new_pos[2],
                move_yaw_deg,
                self.command_task_yaw_deg,
                actual_task_yaw_after,
                actual_uav_yaw_after,
            )
            return self.get_state()

    def get_frame_jpeg(self) -> bytes:
        """Return the current preview frame encoded as JPEG."""
        with self.lock:
            frame = self.refresh_preview()
            encode_ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.frame_jpeg_quality)],
            )
            if not encode_ok:
                raise RuntimeError("Failed to encode preview frame")
            return encoded.tobytes()

    def capture_frame(self, label: Optional[str] = None) -> Dict[str, Any]:
        """Save the latest raw frame and metadata to disk."""
        with self.lock:
            if self.last_raw_frame is None:
                self.refresh_preview()
            if self.last_raw_frame is None:
                raise RuntimeError("No frame available for capture")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_{label}" if label else ""
            image_path = os.path.join(self.args.capture_dir, f"capture_{timestamp}{suffix}.png")
            meta_path = os.path.join(self.args.capture_dir, f"capture_{timestamp}{suffix}.json")
            cv2.imwrite(image_path, self.last_raw_frame)
            metadata = {
                "env_id": self.args.env_id,
                "task_json": self.args.task_json,
                "preview_mode": self.args.preview_mode,
                "pose": self.get_state()["pose"],
                "capture_time": timestamp,
                "label": label,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            self.last_capture = {
                "image_path": image_path,
                "meta_path": meta_path,
                "capture_time": timestamp,
            }
            logger.info("Captured preview image: %s", image_path)
            return {
                "status": "ok",
                "image_path": image_path,
                "meta_path": meta_path,
                "pose": metadata["pose"],
            }

    def shutdown(self) -> Dict[str, Any]:
        """Close the environment and stop the HTTP server."""
        logger.info("Shutdown requested")
        if self.httpd is not None:
            threading.Thread(target=self.httpd.shutdown, daemon=True).start()
        return {"status": "ok", "message": "Shutdown requested"}

    def close(self) -> None:
        """Close Unreal resources."""
        try:
            self.env.close()
        except Exception as exc:
            logger.warning("Failed to close environment cleanly: %s", exc)


def make_handler(backend: UAVControlBackend):
    """Build an HTTP handler bound to a backend instance."""

    class ControlRequestHandler(BaseHTTPRequestHandler):
        def _read_json(self) -> Dict[str, Any]:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length) if content_length > 0 else b"{}"
            if not raw_body:
                return {}
            return json.loads(raw_body.decode("utf-8"))

        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/health":
                    self._send_json({"status": "ok"})
                elif parsed.path == "/state":
                    self._send_json(backend.get_state())
                elif parsed.path == "/frame":
                    body = backend.get_frame_jpeg()
                    self._send_bytes(body, "image/jpeg")
                else:
                    self._send_json({"status": "error", "message": "Not found"}, 404)
            except Exception as exc:
                logger.exception("GET %s failed", parsed.path)
                self._send_json({"status": "error", "message": str(exc)}, 500)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                data = self._read_json()
                if parsed.path == "/move_relative":
                    payload = backend.move_relative(
                        forward_cm=float(data.get("forward_cm", 0.0)),
                        right_cm=float(data.get("right_cm", 0.0)),
                        up_cm=float(data.get("up_cm", 0.0)),
                        yaw_delta_deg=float(data.get("yaw_delta_deg", 0.0)),
                        action_name=str(data.get("action_name", "custom")),
                    )
                    self._send_json(payload)
                elif parsed.path == "/capture":
                    self._send_json(backend.capture_frame(label=data.get("label")))
                elif parsed.path == "/shutdown":
                    self._send_json(backend.shutdown())
                else:
                    self._send_json({"status": "error", "message": "Not found"}, 404)
            except Exception as exc:
                logger.exception("POST %s failed", parsed.path)
                self._send_json({"status": "error", "message": str(exc)}, 500)

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("HTTP %s - %s", self.address_string(), fmt % args)

    return ControlRequestHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Unreal and expose UAV control endpoints")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=5020, help="HTTP port for control")
    parser.add_argument("--env_id", default="UnrealTrack-SuburbNeighborhood_Day-ContinuousColor-v0", help="Gym environment id")
    parser.add_argument("--env_bin_win", default=None, help="Override env_bin_win for the chosen environment")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--time_dilation", type=int, default=10, help="Simulator time dilation")
    parser.add_argument("--window_width", type=int, default=640, help="UnrealCV capture width")
    parser.add_argument("--window_height", type=int, default=480, help="UnrealCV capture height")
    parser.add_argument("--viewport_mode", default="free", choices=["first_person", "third_person", "free"], help="Native Unreal viewport mode")
    parser.add_argument("--viewport_offset_x", type=float, default=-220.0, help="Third-person viewport X offset")
    parser.add_argument("--viewport_offset_y", type=float, default=0.0, help="Third-person viewport Y offset")
    parser.add_argument("--viewport_offset_z", type=float, default=90.0, help="Third-person viewport Z offset")
    parser.add_argument("--viewport_roll", type=float, default=0.0, help="Third-person viewport roll")
    parser.add_argument("--viewport_pitch", type=float, default=-12.0, help="Third-person viewport pitch")
    parser.add_argument("--viewport_yaw", type=float, default=0.0, help="Third-person viewport yaw")
    parser.add_argument("--free_view_offset_x", type=float, default=-220.0, help="Free-view follow X offset")
    parser.add_argument("--free_view_offset_y", type=float, default=140.0, help="Free-view follow Y offset")
    parser.add_argument("--free_view_offset_z", type=float, default=50.0, help="Free-view follow Z offset")
    parser.add_argument("--free_view_roll", type=float, default=0.0, help="Free-view roll")
    parser.add_argument("--free_view_pitch", type=float, default=0.0, help="Free-view pitch offset")
    parser.add_argument("--free_view_yaw", type=float, default=0.0, help="Free-view yaw offset")
    parser.add_argument("--follow_free_view", action="store_true", help="Place the native Unreal free view near the UAV once at startup")
    parser.add_argument("--preview_mode", default="first_person", choices=["first_person", "third_person"], help="Frame/capture mode")
    parser.add_argument("--preview_offset_x", type=float, default=-260.0, help="Third-person preview X offset")
    parser.add_argument("--preview_offset_y", type=float, default=0.0, help="Third-person preview Y offset")
    parser.add_argument("--preview_offset_z", type=float, default=120.0, help="Third-person preview Z offset")
    parser.add_argument("--preview_roll", type=float, default=0.0, help="Third-person preview roll")
    parser.add_argument("--preview_pitch", type=float, default=-12.0, help="Third-person preview pitch")
    parser.add_argument("--preview_yaw", type=float, default=0.0, help="Third-person preview yaw offset")
    parser.add_argument(
        "--movement_yaw_mode",
        default="task",
        choices=["task", "uav", "camera"],
        help="Yaw frame used by WASD translation: task=stable server command yaw, uav=live actor yaw, camera=live preview yaw",
    )
    parser.add_argument("--frame_jpeg_quality", type=int, default=90, help="JPEG quality used by /frame")
    parser.add_argument("--capture_dir", default="./captures_remote", help="Directory used for server-side captures")
    parser.add_argument("--task_json", default=None, help="Optional task JSON used to place the UAV and target")
    parser.add_argument("--spawn_x", type=float, default=None, help="Optional UAV spawn x")
    parser.add_argument("--spawn_y", type=float, default=None, help="Optional UAV spawn y")
    parser.add_argument("--spawn_z", type=float, default=None, help="Optional UAV spawn z")
    parser.add_argument("--spawn_yaw", type=float, default=None, help="Optional UAV spawn yaw in task-json convention")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )

    backend = UAVControlBackend(args)
    handler = make_handler(backend)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    backend.httpd = server
    logger.info("UAV control server listening on http://%s:%s", args.host, args.port)

    try:
        server.serve_forever()
    finally:
        backend.close()


if __name__ == "__main__":
    main()
