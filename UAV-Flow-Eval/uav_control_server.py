"""
Launch and hold the Unreal UAV environment, then expose simple HTTP controls.
"""

import argparse
import base64
import json
import logging
import os
import re
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib import error, request
from urllib.parse import urlparse

import cv2
import gym
import gym_unrealcv
import numpy as np

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
from gym_unrealcv.envs.wrappers import augmentation, configUE, time_dilation
from lesson4.depth_planar_pipeline import coerce_depth_planar_image, generate_camera_info
from runtime_interfaces import build_plan_state, build_runtime_debug_state, build_waypoint, now_timestamp

logger = logging.getLogger(__name__)


def normalize_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def build_obj_info(task_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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


def encode_image_b64(frame: np.ndarray, jpeg_quality: int = 90) -> str:
    encode_ok, encoded = cv2.imencode(
        ".jpg",
        frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
    )
    if not encode_ok:
        raise RuntimeError("Failed to encode frame for planner payload")
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def render_depth_preview(
    depth_image: np.ndarray,
    width: int,
    height: int,
    *,
    min_depth_cm: Optional[float] = None,
    max_depth_cm: Optional[float] = None,
    source_mode: str = "depth",
) -> np.ndarray:
    depth = coerce_depth_planar_image(depth_image)
    finite = depth[np.isfinite(depth)]
    canvas = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    canvas[:] = (12, 12, 18)

    if finite.size:
        preview_min = float(np.min(finite)) if min_depth_cm is None else float(min_depth_cm)
        preview_max = float(np.max(finite)) if max_depth_cm is None else float(max_depth_cm)
        if preview_max <= preview_min:
            preview_max = preview_min + 1.0
        valid_mask = np.isfinite(depth) & (depth >= preview_min) & (depth <= preview_max)
        clipped = np.clip(depth, preview_min, preview_max)
        normalized = 1.0 - ((clipped - preview_min) / (preview_max - preview_min))
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        preview_u8 = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)
        canvas = cv2.applyColorMap(preview_u8, cv2.COLORMAP_TURBO)
        canvas[~valid_mask] = (16, 16, 20)
    else:
        preview_min = 0.0
        preview_max = 0.0

    if width > 0 and height > 0 and (canvas.shape[1] != width or canvas.shape[0] != height):
        canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_NEAREST)

    cv2.putText(canvas, f"Depth mode: {source_mode}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"range(cm): {preview_min:.1f} -> {preview_max:.1f}", (12, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "near=warm  far=cool", (12, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    return canvas


class UAVControlBackend:
    """Own the Unreal environment and provide thread-safe control methods."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.RLock()
        self.last_raw_frame: Optional[np.ndarray] = None
        self.last_depth_frame: Optional[np.ndarray] = None
        self.last_depth_summary: Dict[str, Any] = {
            "frame_id": "depth-init",
            "available": False,
            "min_depth": 0.0,
            "max_depth": 0.0,
            "image_width": 0,
            "image_height": 0,
            "fov_deg": 0.0,
            "pipeline": "lesson4_depth_image_proc_style",
            "camera_info": {},
        }
        self.last_capture: Optional[Dict[str, Any]] = None
        self.httpd: Optional[HTTPServer] = None
        self.command_task_yaw_deg: float = 0.0
        self.last_action: str = "idle"
        self.frame_index: int = 0
        self.last_observation_time: str = now_timestamp()
        self.last_observation_id: str = "frame_000000"
        self.current_task_label: str = args.default_task_label
        self.current_plan: Dict[str, Any] = build_plan_state(
            planner_name=args.planner_name,
            semantic_subgoal="idle",
        )
        self.runtime_debug: Dict[str, Any] = build_runtime_debug_state()

        os.makedirs(self.args.capture_dir, exist_ok=True)

        maybe_override_env_binary(self.args.env_id, self.args.env_bin_win)
        validate_env_binary_exists(self.args.env_id)

        self.env = gym.make(self.args.env_id)
        if int(self.args.time_dilation) > 0:
            self.env = time_dilation.TimeDilationWrapper(self.env, int(self.args.time_dilation))
        self.env.unwrapped.agents_category = ["drone"]
        self.env = configUE.ConfigUEWrapper(self.env, resolution=(self.args.window_width, self.args.window_height))
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
        self.refresh_observations()

    def apply_initial_task_or_spawn(self) -> None:
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
        rotation = self.env.unwrapped.unrealcv.get_obj_rotation(self.player_name)
        if isinstance(rotation, (list, tuple)) and len(rotation) > 1:
            return normalize_angle_deg(float(rotation[1]))
        return 0.0

    def _coerce_fov_deg(self, raw_fov: Any) -> float:
        if isinstance(raw_fov, (int, float)):
            value = float(raw_fov)
            if value > 1e-3:
                return value
        if isinstance(raw_fov, str):
            stripped = raw_fov.strip()
            try:
                value = float(stripped)
                if value > 1e-3:
                    return value
            except ValueError:
                matches = re.findall(r"[-+]?\d*\.?\d+", stripped)
                if len(matches) == 1:
                    value = float(matches[0])
                    if value > 1e-3:
                        return value
        fallback = float(self.last_depth_summary.get("fov_deg", self.args.default_depth_fov_deg))
        logger.warning("Unexpected get_cam_fov response %r, fallback to %.2f deg", raw_fov, fallback)
        return fallback

    def get_task_pose(self) -> List[float]:
        location = self.env.unwrapped.unrealcv.get_obj_location(self.player_name)
        task_yaw = normalize_angle_deg(self.get_env_yaw_deg() + 180.0)
        return [float(location[0]), float(location[1]), float(location[2]), task_yaw]

    def set_task_pose(self, position: List[float], yaw_deg: float) -> None:
        self.env.unwrapped.unrealcv.set_obj_location(self.player_name, list(position))
        self.env.unwrapped.unrealcv.set_rotation(self.player_name, float(yaw_deg) - 180.0)

    def get_control_yaw_deg(self) -> float:
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
        if self.args.viewport_mode != "free" or not self.args.follow_free_view:
            return
        pose = self.get_task_pose()
        focus_pose = [pose[0], pose[1], pose[2], 0.0, pose[3]]
        set_free_view_near_pose(self.env, focus_pose, self.free_view_offset, self.free_view_rotation)

    def get_preview_frame(self) -> np.ndarray:
        if self.args.preview_mode == "third_person":
            return get_third_person_preview_image(
                self.env,
                self.preview_cam_id,
                self.preview_offset,
                self.preview_rotation,
            )
        set_cam(self.env, self.policy_cam_id)
        return self.env.unwrapped.unrealcv.get_image(self.policy_cam_id, "lit")

    def get_depth_observation(self) -> Tuple[np.ndarray, float]:
        set_cam(self.env, self.policy_cam_id)
        depth_image = coerce_depth_planar_image(self.env.unwrapped.unrealcv.get_depth(self.policy_cam_id))
        fov_deg = self._coerce_fov_deg(self.env.unwrapped.unrealcv.get_cam_fov(self.policy_cam_id))
        return depth_image, fov_deg

    def invalidate_cached_observations(self) -> None:
        self.last_raw_frame = None
        self.last_depth_frame = None

    def refresh_preview_only(self) -> np.ndarray:
        self.frame_index += 1
        self.last_observation_time = now_timestamp()
        self.last_observation_id = f"frame_{self.frame_index:06d}"
        frame = self.get_preview_frame()
        self.last_raw_frame = frame
        return frame

    def refresh_depth_only(self) -> Tuple[np.ndarray, float]:
        if self.last_raw_frame is None:
            self.refresh_preview_only()
        depth_image, depth_fov_deg = self.get_depth_observation()
        self.last_depth_frame = depth_image
        camera_info = generate_camera_info(
            int(depth_image.shape[1]),
            int(depth_image.shape[0]),
            float(depth_fov_deg),
            self.args.depth_camera_frame_id,
        )
        finite_depth = depth_image[np.isfinite(depth_image)]
        configured_min_depth = float(self.args.depth_min_cm)
        configured_max_depth = float(self.args.depth_max_cm)
        valid_depth = finite_depth[(finite_depth >= configured_min_depth) & (finite_depth <= configured_max_depth)]
        self.last_depth_summary = {
            "frame_id": self.last_observation_id,
            "available": bool(finite_depth.size),
            "min_depth": float(np.min(valid_depth)) if valid_depth.size else configured_min_depth,
            "max_depth": float(np.max(valid_depth)) if valid_depth.size else configured_max_depth,
            "raw_min_depth": float(np.min(finite_depth)) if finite_depth.size else 0.0,
            "raw_max_depth": float(np.max(finite_depth)) if finite_depth.size else 0.0,
            "image_width": int(depth_image.shape[1]),
            "image_height": int(depth_image.shape[0]),
            "fov_deg": float(depth_fov_deg),
            "source_mode": "lesson4_depth_planar",
            "pipeline": "lesson4_depth_image_proc_style",
            "camera_id": int(self.policy_cam_id),
            "camera_info": camera_info,
        }
        return depth_image, depth_fov_deg

    def refresh_observations(self) -> np.ndarray:
        frame = self.refresh_preview_only()
        self.refresh_depth_only()
        return frame

    def refresh_preview(self) -> np.ndarray:
        return self.refresh_observations()

    def get_depth_frame_jpeg(self) -> bytes:
        with self.lock:
            if self.last_depth_frame is None:
                self.refresh_depth_only()
            if self.last_depth_frame is None:
                raise RuntimeError("No depth frame available")
            image = render_depth_preview(
                self.last_depth_frame,
                self.args.depth_preview_width,
                self.args.depth_preview_height,
                min_depth_cm=float(self.args.depth_min_cm),
                max_depth_cm=float(self.args.depth_max_cm),
                source_mode=self.last_depth_summary.get("source_mode", "policy_depth"),
            )
            encode_ok, encoded = cv2.imencode(
                ".jpg",
                image,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.frame_jpeg_quality)],
            )
            if not encode_ok:
                raise RuntimeError("Failed to encode depth preview frame")
            return encoded.tobytes()

    def get_depth_raw_png(self) -> bytes:
        with self.lock:
            if self.last_depth_frame is None:
                self.refresh_depth_only()
            if self.last_depth_frame is None:
                raise RuntimeError("No depth frame available")
            depth_u16 = np.clip(
                np.nan_to_num(self.last_depth_frame, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
                65535.0,
            ).astype(np.uint16)
            encode_ok, encoded = cv2.imencode(".png", depth_u16)
            if not encode_ok:
                raise RuntimeError("Failed to encode raw depth frame")
            return encoded.tobytes()

    def get_camera_info(self) -> Dict[str, Any]:
        with self.lock:
            if self.last_depth_frame is None:
                self.refresh_depth_only()
            return self.last_depth_summary.get("camera_info", {})

    def set_task_label(self, task_label: str) -> Dict[str, Any]:
        self.current_task_label = str(task_label or "").strip()
        return {"status": "ok", "task_label": self.current_task_label}

    def set_plan_state(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        candidate_waypoints_raw = payload.get("candidate_waypoints") or []
        candidate_waypoints: List[Dict[str, Any]] = []
        for waypoint in candidate_waypoints_raw:
            if not isinstance(waypoint, dict):
                continue
            candidate_waypoints.append(
                build_waypoint(
                    waypoint.get("x", 0.0),
                    waypoint.get("y", 0.0),
                    waypoint.get("z", 0.0),
                    waypoint.get("yaw", 0.0),
                    waypoint.get("radius", 50.0),
                    str(waypoint.get("semantic_label", "")),
                )
            )
        self.current_plan = build_plan_state(
            plan_id=str(payload.get("plan_id", f"plan_{self.last_observation_id}")),
            planner_name=str(payload.get("planner_name", self.args.planner_name)),
            generated_at=str(payload.get("generated_at", now_timestamp())),
            sector_id=payload.get("sector_id"),
            candidate_waypoints=candidate_waypoints,
            semantic_subgoal=str(payload.get("semantic_subgoal", "idle")),
            planner_confidence=float(payload.get("planner_confidence", 0.0)),
            should_replan=bool(payload.get("should_replan", False)),
            debug=payload.get("debug") if isinstance(payload.get("debug"), dict) else {},
        )
        self.runtime_debug["current_waypoint"] = candidate_waypoints[0] if candidate_waypoints else None
        return self.current_plan

    def request_plan(self, task_label: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            if task_label is not None:
                self.set_task_label(task_label)
            frame = self.refresh_observations()
            pose = self.get_task_pose()
            task_label_value = self.current_task_label or "idle"
            plan_payload: Optional[Dict[str, Any]] = None

            if self.args.planner_url:
                planner_request = {
                    "task_label": task_label_value,
                    "instruction": task_label_value,
                    "frame_id": self.last_observation_id,
                    "timestamp": self.last_observation_time,
                    "pose": {
                        "x": pose[0],
                        "y": pose[1],
                        "z": pose[2],
                        "yaw": pose[3],
                    },
                    "depth": self.last_depth_summary,
                    "camera_info": self.last_depth_summary.get("camera_info", {}),
                    "image_b64": encode_image_b64(frame, self.args.frame_jpeg_quality),
                }
                req = request.Request(
                    f"{self.args.planner_url.rstrip('/')}{self.args.planner_endpoint}",
                    data=json.dumps(planner_request).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                try:
                    with request.urlopen(req, timeout=self.args.planner_timeout_s) as resp:
                        raw = json.loads(resp.read().decode("utf-8"))
                    if isinstance(raw, dict):
                        plan_payload = raw.get("plan") if isinstance(raw.get("plan"), dict) else raw
                except (error.URLError, TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
                    logger.warning("Planner request failed, falling back to heuristic plan: %s", exc)

            if plan_payload is None:
                move_yaw = self.get_control_yaw_deg()
                theta = np.radians(move_yaw)
                waypoint = build_waypoint(
                    x=pose[0] + self.args.default_plan_distance_cm * float(np.cos(theta)),
                    y=pose[1] + self.args.default_plan_distance_cm * float(np.sin(theta)),
                    z=pose[2],
                    yaw=move_yaw,
                    radius=self.args.default_waypoint_radius_cm,
                    semantic_label=task_label_value or "forward_search",
                )
                plan_payload = build_plan_state(
                    plan_id=f"plan_{self.last_observation_id}",
                    planner_name="heuristic_fallback",
                    sector_id=int(round(((move_yaw % 360.0) / 360.0) * self.args.default_sector_count)) % self.args.default_sector_count,
                    candidate_waypoints=[waypoint],
                    semantic_subgoal=task_label_value or "move_forward",
                    planner_confidence=0.35,
                    should_replan=False,
                    debug={
                        "source": "local_heuristic",
                        "planner_interval_steps": self.args.planner_interval_steps,
                    },
                )

            return self.set_plan_state(plan_payload)

    def update_runtime_debug(
        self,
        *,
        current_waypoint: Optional[Dict[str, Any]] = None,
        local_policy_action: Optional[Dict[str, Any]] = None,
        risk_score: Optional[float] = None,
        shield_triggered: Optional[bool] = None,
        archive_cell_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if current_waypoint is not None:
            self.runtime_debug["current_waypoint"] = current_waypoint
        if local_policy_action is not None:
            self.runtime_debug["local_policy_action"] = local_policy_action
        if risk_score is not None:
            self.runtime_debug["risk_score"] = float(risk_score)
        if shield_triggered is not None:
            self.runtime_debug["shield_triggered"] = bool(shield_triggered)
        if archive_cell_id is not None:
            self.runtime_debug["archive_cell_id"] = str(archive_cell_id)
        return self.runtime_debug

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
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
                "task_label": self.current_task_label,
                "observation": {
                    "frame_id": self.last_observation_id,
                    "timestamp": self.last_observation_time,
                },
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
                "depth": self.last_depth_summary,
                "camera_info": self.last_depth_summary.get("camera_info", {}),
                "plan": self.current_plan,
                "runtime_debug": self.runtime_debug,
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
        with self.lock:
            x, y, z, _actual_task_yaw = self.get_task_pose()
            move_yaw_deg = self.get_control_yaw_deg()
            theta = np.radians(move_yaw_deg)
            delta_x = float(forward_cm * np.cos(theta) - right_cm * np.sin(theta))
            delta_y = float(forward_cm * np.sin(theta) + right_cm * np.cos(theta))
            new_pos = [x + delta_x, y + delta_y, z + up_cm]
            self.command_task_yaw_deg = normalize_angle_deg(self.command_task_yaw_deg + yaw_delta_deg)
            self.set_task_pose(new_pos, self.command_task_yaw_deg)
            self.invalidate_cached_observations()
            self.last_action = action_name
            self.runtime_debug["local_policy_action"] = {
                "action_name": action_name,
                "forward_cm": float(forward_cm),
                "right_cm": float(right_cm),
                "up_cm": float(up_cm),
                "yaw_delta_deg": float(yaw_delta_deg),
            }
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
        with self.lock:
            if self.last_raw_frame is None:
                frame = self.refresh_preview_only()
            else:
                frame = self.last_raw_frame
            encode_ok, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.frame_jpeg_quality)],
            )
            if not encode_ok:
                raise RuntimeError("Failed to encode preview frame")
            return encoded.tobytes()

    def capture_frame(self, label: Optional[str] = None, task_label: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            if self.last_raw_frame is None or self.last_depth_frame is None:
                self.refresh_observations()
            if self.last_raw_frame is None:
                raise RuntimeError("No frame available for capture")
            if task_label is not None:
                self.set_task_label(task_label)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_{label}" if label else ""
            capture_id = f"capture_{timestamp}{suffix}"
            image_path = os.path.join(self.args.capture_dir, f"{capture_id}_rgb.png")
            depth_path = os.path.join(self.args.capture_dir, f"{capture_id}_depth_cm.png")
            depth_preview_path = os.path.join(self.args.capture_dir, f"{capture_id}_depth_preview.png")
            camera_info_path = os.path.join(self.args.capture_dir, f"{capture_id}_camera_info.json")
            meta_path = os.path.join(self.args.capture_dir, f"{capture_id}_bundle.json")

            cv2.imwrite(image_path, self.last_raw_frame)
            if self.last_depth_frame is not None:
                depth_to_save = np.clip(
                    np.nan_to_num(self.last_depth_frame, nan=0.0, posinf=0.0, neginf=0.0),
                    0.0,
                    65535.0,
                )
                cv2.imwrite(depth_path, depth_to_save.astype(np.uint16))
                depth_preview = render_depth_preview(
                    self.last_depth_frame,
                    int(self.last_depth_summary.get("image_width", self.args.depth_preview_width)) or self.args.depth_preview_width,
                    int(self.last_depth_summary.get("image_height", self.args.depth_preview_height)) or self.args.depth_preview_height,
                    min_depth_cm=float(self.args.depth_min_cm),
                    max_depth_cm=float(self.args.depth_max_cm),
                    source_mode=self.last_depth_summary.get("source_mode", "policy_depth"),
                )
                cv2.imwrite(depth_preview_path, depth_preview)

            camera_info = self.last_depth_summary.get("camera_info", {})
            with open(camera_info_path, "w", encoding="utf-8") as f:
                json.dump(camera_info, f, indent=2)

            state = self.get_state()
            metadata = {
                "capture_id": capture_id,
                "capture_time": timestamp,
                "env_id": self.args.env_id,
                "task_label": self.current_task_label,
                "action_label": self.last_action,
                "rgb_image_path": image_path,
                "depth_image_path": depth_path if self.last_depth_frame is not None else None,
                "depth_preview_path": depth_preview_path if self.last_depth_frame is not None else None,
                "camera_info_path": camera_info_path,
                "metadata_path": meta_path,
                "pose": state["pose"],
                "depth": self.last_depth_summary,
                "camera_info": camera_info,
                "plan": self.current_plan,
                "runtime_debug": self.runtime_debug,
                "preview_mode": self.args.preview_mode,
                "task_json": self.args.task_json,
                "label": label,
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            self.last_capture = {
                "capture_id": capture_id,
                "image_path": image_path,
                "depth_path": depth_path if self.last_depth_frame is not None else None,
                "depth_preview_path": depth_preview_path if self.last_depth_frame is not None else None,
                "camera_info_path": camera_info_path,
                "meta_path": meta_path,
                "capture_time": timestamp,
            }
            logger.info("Captured RGB/depth bundle: %s", meta_path)
            return {
                "status": "ok",
                "capture_id": capture_id,
                "image_path": image_path,
                "depth_path": depth_path if self.last_depth_frame is not None else None,
                "depth_preview_path": depth_preview_path if self.last_depth_frame is not None else None,
                "camera_info_path": camera_info_path,
                "meta_path": meta_path,
                "task_label": self.current_task_label,
                "pose": metadata["pose"],
                "depth": self.last_depth_summary,
                "camera_info": camera_info,
                "plan": self.current_plan,
            }

    def shutdown(self) -> Dict[str, Any]:
        logger.info("Shutdown requested")
        if self.httpd is not None:
            threading.Thread(target=self.httpd.shutdown, daemon=True).start()
        return {"status": "ok", "message": "Shutdown requested"}

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def make_handler(backend: UAVControlBackend):
    class ControlRequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
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
            try:
                if parsed.path in ("/", "/health"):
                    self._send_json({"status": "ok"})
                elif parsed.path == "/state":
                    self._send_json(backend.get_state())
                elif parsed.path == "/frame":
                    self._send_bytes(backend.get_frame_jpeg(), "image/jpeg")
                elif parsed.path in ("/depth", "/depth_frame"):
                    self._send_bytes(backend.get_depth_frame_jpeg(), "image/jpeg")
                elif parsed.path in ("/depth_raw", "/depth_raw.png"):
                    self._send_bytes(backend.get_depth_raw_png(), "image/png")
                elif parsed.path == "/camera_info":
                    self._send_json({"status": "ok", "camera_info": backend.get_camera_info()})
                elif parsed.path == "/plan":
                    self._send_json({"status": "ok", "plan": backend.current_plan})
                else:
                    self._send_json({"status": "error", "message": "Not found"}, 404)
            except Exception as exc:
                logger.exception("GET %s failed", parsed.path)
                self._send_json({"status": "error", "message": str(exc)}, 500)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                data = self._read_json_body()
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
                    self._send_json(
                        backend.capture_frame(
                            label=data.get("label"),
                            task_label=data.get("task_label"),
                        )
                    )
                elif parsed.path == "/task":
                    self._send_json(backend.set_task_label(str(data.get("task_label", ""))))
                elif parsed.path == "/plan":
                    plan = backend.set_plan_state(data if isinstance(data, dict) else {})
                    self._send_json({"status": "ok", "plan": plan})
                elif parsed.path == "/request_plan":
                    plan = backend.request_plan(task_label=data.get("task_label"))
                    self._send_json({"status": "ok", "plan": plan})
                elif parsed.path == "/runtime_debug":
                    debug_state = backend.update_runtime_debug(
                        current_waypoint=data.get("current_waypoint") if isinstance(data.get("current_waypoint"), dict) else None,
                        local_policy_action=data.get("local_policy_action") if isinstance(data.get("local_policy_action"), dict) else None,
                        risk_score=data.get("risk_score"),
                        shield_triggered=data.get("shield_triggered"),
                        archive_cell_id=data.get("archive_cell_id"),
                    )
                    self._send_json({"status": "ok", "runtime_debug": debug_state})
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
    parser.add_argument("--movement_yaw_mode", default="task", choices=["task", "uav", "camera"], help="Yaw frame used by WASD translation")
    parser.add_argument("--frame_jpeg_quality", type=int, default=90, help="JPEG quality used by /frame and /depth_frame")
    parser.add_argument("--default_depth_fov_deg", type=float, default=90.0, help="Fallback FOV when UnrealCV returns malformed camera FOV")
    parser.add_argument("--depth_camera_frame_id", default="PX4/CameraDepth_optical", help="Frame id used in generated depth camera_info")
    parser.add_argument("--depth_min_cm", type=float, default=20.0, help="Minimum depth kept in summaries/previews")
    parser.add_argument("--depth_max_cm", type=float, default=1200.0, help="Maximum depth kept in summaries/previews")
    parser.add_argument("--depth_preview_width", type=int, default=480, help="Depth preview render width")
    parser.add_argument("--depth_preview_height", type=int, default=360, help="Depth preview render height")
    parser.add_argument("--capture_dir", default="./captures_remote", help="Directory used for server-side captures")
    parser.add_argument("--default_task_label", default="", help="Default task label used by capture/planner endpoints")
    parser.add_argument("--planner_name", default="phase2-planner", help="Planner name stored in /plan state")
    parser.add_argument("--planner_url", default=None, help="Optional external planner base URL")
    parser.add_argument("--planner_endpoint", default="/plan", help="Planner endpoint path used with planner_url")
    parser.add_argument("--planner_timeout_s", type=float, default=5.0, help="Timeout for planner requests")
    parser.add_argument("--planner_interval_steps", type=int, default=5, help="Target replan interval for debug/metadata")
    parser.add_argument("--default_plan_distance_cm", type=float, default=300.0, help="Fallback heuristic waypoint distance")
    parser.add_argument("--default_waypoint_radius_cm", type=float, default=60.0, help="Fallback waypoint acceptance radius")
    parser.add_argument("--default_sector_count", type=int, default=8, help="Fallback discrete sector count")
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
    server = HTTPServer((args.host, args.port), handler)
    backend.httpd = server
    logger.info("UAV control server listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    finally:
        backend.close()


if __name__ == "__main__":
    main()
