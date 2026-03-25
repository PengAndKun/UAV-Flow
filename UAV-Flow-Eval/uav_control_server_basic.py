"""
Low-latency basic UAV control server.

Only keeps:
- environment startup / spawn
- basic movement enable + move_relative
- task label
- RGB/depth endpoints
- capture
- camera info / minimal state
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
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
from house_registry import HouseRegistry, HouseStatus

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

logger = logging.getLogger(__name__)


def now_timestamp() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


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
    return {"use_obj": task_data["use_obj"], "obj_id": task_data["obj_id"], "obj_pos": obj_pos, "obj_rot": obj_rot}


def render_depth_preview(depth_image: np.ndarray, width: int, height: int, *, min_depth_cm: float, max_depth_cm: float, source_mode: str) -> np.ndarray:
    depth = coerce_depth_planar_image(depth_image)
    finite = depth[np.isfinite(depth)]
    canvas = np.zeros((depth.shape[0], depth.shape[1], 3), dtype=np.uint8)
    canvas[:] = (12, 12, 18)
    if finite.size:
        preview_min = float(np.min(finite)) if min_depth_cm <= 0 else float(min_depth_cm)
        preview_max = float(np.max(finite)) if max_depth_cm <= preview_min else float(max_depth_cm)
        if preview_max <= preview_min:
            preview_max = preview_min + 1.0
        valid_mask = np.isfinite(depth) & (depth >= preview_min) & (depth <= preview_max)
        clipped = np.clip(depth, preview_min, preview_max)
        normalized = 1.0 - ((clipped - preview_min) / (preview_max - preview_min))
        preview_u8 = np.clip(np.nan_to_num(normalized) * 255.0, 0.0, 255.0).astype(np.uint8)
        canvas = cv2.applyColorMap(preview_u8, cv2.COLORMAP_TURBO)
        canvas[~valid_mask] = (16, 16, 20)
    if width > 0 and height > 0 and (canvas.shape[1] != width or canvas.shape[0] != height):
        canvas = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_NEAREST)
    cv2.putText(canvas, f"Depth mode: {source_mode}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


class BasicUAVControlBackend:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lock = threading.RLock()
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.last_raw_frame: Optional[np.ndarray] = None
        self.last_depth_frame: Optional[np.ndarray] = None
        self.last_capture: Optional[Dict[str, Any]] = None
        self.last_action = "idle"
        self.frame_index = 0
        self.last_observation_time = now_timestamp()
        self.last_observation_id = "frame_000000"
        self.current_task_label = str(args.default_task_label or "")
        self.movement_enabled = bool(args.start_with_basic_movement)
        self.fixed_spawn_pose_path = os.path.abspath(args.fixed_spawn_pose_file) if args.fixed_spawn_pose_file else ""
        self.command_task_yaw_deg = 0.0
        # --- House registry ---
        self.house_registry = HouseRegistry(args.houses_config)
        self.last_depth_summary: Dict[str, Any] = {
            "frame_id": self.last_observation_id,
            "available": False,
            "min_depth": 0.0,
            "max_depth": 0.0,
            "front_min_depth": 0.0,
            "front_mean_depth": 0.0,
            "image_width": 0,
            "image_height": 0,
            "fov_deg": float(args.default_depth_fov_deg),
            "source_mode": "lesson4_depth_planar",
            "camera_info": {},
        }

        os.makedirs(args.capture_dir, exist_ok=True)
        maybe_override_env_binary(args.env_id, args.env_bin_win)
        validate_env_binary_exists(args.env_id)
        self.env = gym.make(args.env_id)
        if int(args.time_dilation) > 0:
            self.env = time_dilation.TimeDilationWrapper(self.env, int(args.time_dilation))
        self.env.unwrapped.agents_category = ["drone"]
        self.env = configUE.ConfigUEWrapper(self.env, resolution=(args.window_width, args.window_height))
        self.env = augmentation.RandomPopulationWrapper(self.env, 2, 2, random_target=False)
        self.env.seed(int(args.seed))
        self.env.reset()

        self.player_name = self.env.unwrapped.player_list[0]
        self.env.unwrapped.unrealcv.set_phy(self.player_name, 0)
        self.hide_non_primary_agents()
        configure_player_viewport(
            self.env,
            args.viewport_mode,
            (args.viewport_offset_x, args.viewport_offset_y, args.viewport_offset_z),
            (args.viewport_roll, args.viewport_pitch, args.viewport_yaw),
        )
        self.policy_cam_id = get_policy_cam_id(self.env)
        self.preview_cam_id = get_follow_preview_cam_id(self.env, self.policy_cam_id)
        self.free_view_offset = (args.free_view_offset_x, args.free_view_offset_y, args.free_view_offset_z)
        self.free_view_rotation = (args.free_view_roll, args.free_view_pitch, args.free_view_yaw)
        self.preview_offset = (args.preview_offset_x, args.preview_offset_y, args.preview_offset_z)
        self.preview_rotation = (args.preview_roll, args.preview_pitch, args.preview_yaw)

        self.apply_initial_task_or_spawn()
        self.command_task_yaw_deg = float(self.get_task_pose()[3])
        self.position_free_view_once()
        self.refresh_observations()

    def hide_non_primary_agents(self) -> None:
        for idx, player_name in enumerate(self.env.unwrapped.player_list[1:], start=1):
            try:
                hide_pos = [0.0, 0.0, -10000.0 - 100.0 * idx]
                self.env.unwrapped.unrealcv.set_phy(player_name, 0)
                self.env.unwrapped.unrealcv.set_obj_location(player_name, hide_pos)
                self.env.unwrapped.unrealcv.set_obj_rotation(player_name, [0.0, 0.0, 0.0])
            except Exception as exc:
                logger.warning("Failed to hide extra UAV agent %s: %s", player_name, exc)

    def load_fixed_spawn_pose(self) -> Optional[Dict[str, float]]:
        if not self.fixed_spawn_pose_path or not os.path.exists(self.fixed_spawn_pose_path):
            return None
        try:
            with open(self.fixed_spawn_pose_path, "r", encoding="utf-8") as pose_file:
                payload = json.load(pose_file)
            if not isinstance(payload, dict):
                return None
            return {k: float(payload[k]) for k in ("x", "y", "z", "yaw")}
        except Exception as exc:
            logger.warning("Failed to load fixed spawn pose from %s: %s", self.fixed_spawn_pose_path, exc)
            return None

    def save_fixed_spawn_pose(self, pose: Dict[str, float]) -> None:
        if not self.fixed_spawn_pose_path:
            return
        with open(self.fixed_spawn_pose_path, "w", encoding="utf-8") as pose_file:
            json.dump({k: float(pose[k]) for k in ("x", "y", "z", "yaw")}, pose_file, indent=2)

    def get_env_yaw_deg(self) -> float:
        rotation = self.env.unwrapped.unrealcv.get_obj_rotation(self.player_name)
        if isinstance(rotation, (list, tuple)) and len(rotation) > 1:
            return normalize_angle_deg(float(rotation[1]))
        return 0.0

    def get_task_pose(self) -> List[float]:
        # In the basic-only controller, the displayed/input yaw is the real UAV yaw.
        location = self.env.unwrapped.unrealcv.get_obj_location(self.player_name)
        return [float(location[0]), float(location[1]), float(location[2]), float(self.get_env_yaw_deg())]

    def set_task_pose(self, position: List[float], yaw_deg: float) -> None:
        self.set_task_position_only(position)
        self.set_task_yaw_absolute(float(yaw_deg))

    def set_task_position_only(self, position: List[float]) -> None:
        self.env.unwrapped.unrealcv.set_obj_location(self.player_name, list(position))

    def set_task_yaw_absolute(self, yaw_deg: float, tolerance_deg: float = 1.5, attempts: int = 6) -> float:
        target_task_yaw = normalize_angle_deg(float(yaw_deg))
        current_task_yaw = self.get_task_pose()[3]
        for _ in range(max(1, int(attempts))):
            # This blueprint helper is the stable drone-facing API in the existing teleop path.
            self.env.unwrapped.unrealcv.set_rotation(self.player_name, target_task_yaw - 180.0)
            time.sleep(0.08)
            current_task_yaw = self.get_task_pose()[3]
            if abs(normalize_angle_deg(target_task_yaw - current_task_yaw)) <= float(tolerance_deg):
                break
        return float(current_task_yaw)

    def apply_initial_task_or_spawn(self) -> None:
        if self.args.task_json:
            with open(self.args.task_json, "r", encoding="utf-8") as f:
                task_data = json.load(f)
            create_obj_if_needed(self.env, build_obj_info(task_data))
            initial_pos = task_data.get("initial_pos")
            if isinstance(initial_pos, list) and len(initial_pos) >= 5:
                self.set_task_pose(initial_pos[:3], float(initial_pos[4]))
                self.current_task_label = str(task_data.get("instr", self.current_task_label) or self.current_task_label)
                return
        if self.args.spawn_x is not None and self.args.spawn_y is not None and self.args.spawn_z is not None:
            spawn_pose = {
                "x": float(self.args.spawn_x),
                "y": float(self.args.spawn_y),
                "z": float(self.args.spawn_z),
                "yaw": float(self.args.spawn_yaw if self.args.spawn_yaw is not None else self.get_task_pose()[3]),
            }
        else:
            spawn_pose = self.load_fixed_spawn_pose()
        if spawn_pose is not None:
            self.set_task_pose([spawn_pose["x"], spawn_pose["y"], spawn_pose["z"]], spawn_pose["yaw"])
            self.save_fixed_spawn_pose(spawn_pose)
        elif self.fixed_spawn_pose_path:
            pose = self.get_task_pose()
            self.save_fixed_spawn_pose({"x": pose[0], "y": pose[1], "z": pose[2], "yaw": pose[3]})

    def position_free_view_once(self) -> None:
        if self.args.viewport_mode != "free" or not self.args.follow_free_view:
            return
        pose = self.get_task_pose()
        set_free_view_near_pose(self.env, [pose[0], pose[1], pose[2], 0.0, pose[3]], self.free_view_offset, self.free_view_rotation)

    def _coerce_fov_deg(self, raw_fov: Any) -> float:
        if isinstance(raw_fov, (int, float)):
            value = float(raw_fov)
            if value > 1e-3:
                return value
        if isinstance(raw_fov, str):
            try:
                return float(raw_fov.strip())
            except ValueError:
                matches = re.findall(r"[-+]?\\d*\\.?\\d+", raw_fov)
                if matches:
                    return float(matches[0])
        return float(self.last_depth_summary.get("fov_deg", self.args.default_depth_fov_deg))

    def get_preview_frame(self) -> np.ndarray:
        if self.args.preview_mode == "third_person":
            return get_third_person_preview_image(self.env, self.preview_cam_id, self.preview_offset, self.preview_rotation)
        set_cam(self.env, self.policy_cam_id)
        return self.env.unwrapped.unrealcv.get_image(self.policy_cam_id, "lit")

    def get_depth_observation(self) -> Tuple[np.ndarray, float]:
        set_cam(self.env, self.policy_cam_id)
        return coerce_depth_planar_image(self.env.unwrapped.unrealcv.get_depth(self.policy_cam_id)), self._coerce_fov_deg(self.env.unwrapped.unrealcv.get_cam_fov(self.policy_cam_id))

    def refresh_preview_only(self) -> None:
        self.frame_index += 1
        self.last_observation_time = now_timestamp()
        self.last_observation_id = f"frame_{self.frame_index:06d}"
        self.last_raw_frame = self.get_preview_frame()

    def refresh_depth_only(self) -> None:
        if self.last_raw_frame is None:
            self.refresh_preview_only()
        depth_image, depth_fov_deg = self.get_depth_observation()
        self.last_depth_frame = depth_image
        finite_depth = depth_image[np.isfinite(depth_image)]
        valid_depth = finite_depth[(finite_depth >= float(self.args.depth_min_cm)) & (finite_depth <= float(self.args.depth_max_cm))]
        h, w = depth_image.shape[:2]
        patch = depth_image[int(h * 0.55):int(h * 0.9), int(w * 0.4):int(w * 0.6)]
        patch_valid = patch[np.isfinite(patch)]
        self.last_depth_summary = {
            "frame_id": self.last_observation_id,
            "available": bool(finite_depth.size),
            "min_depth": float(np.min(valid_depth)) if valid_depth.size else float(self.args.depth_min_cm),
            "max_depth": float(np.max(valid_depth)) if valid_depth.size else float(self.args.depth_max_cm),
            "front_min_depth": float(np.min(patch_valid)) if patch_valid.size else 0.0,
            "front_mean_depth": float(np.mean(patch_valid)) if patch_valid.size else 0.0,
            "image_width": int(w),
            "image_height": int(h),
            "fov_deg": float(depth_fov_deg),
            "source_mode": "lesson4_depth_planar",
            "camera_info": generate_camera_info(int(w), int(h), float(depth_fov_deg), self.args.depth_camera_frame_id),
        }

    def refresh_observations(self) -> None:
        self.refresh_preview_only()
        self.refresh_depth_only()

    def _build_pose_state(self) -> Dict[str, Any]:
        pose = self.get_task_pose()
        return {
            "x": float(pose[0]),
            "y": float(pose[1]),
            "z": float(pose[2]),
            "task_yaw": float(pose[3]),
            "uav_yaw": float(pose[3]),
            "command_yaw": float(self.command_task_yaw_deg),
        }

    def get_state(self, *, status: str = "ok", message: str = "") -> Dict[str, Any]:
        return {
            "status": status,
            "message": message,
            "env_id": self.args.env_id,
            "player_name": self.player_name,
            "task_label": self.current_task_label,
            "movement_enabled": self.movement_enabled,
            "last_action": self.last_action,
            "last_action_origin": "basic_movement",
            "observation_id": self.last_observation_id,
            "observation_time": self.last_observation_time,
            "pose": self._build_pose_state(),
            "depth": dict(self.last_depth_summary),
            "camera_info": dict(self.last_depth_summary.get("camera_info", {})),
            "last_capture": dict(self.last_capture) if isinstance(self.last_capture, dict) else None,
            "house_registry": self.house_registry.get_status_summary(),
        }

    # ------------------------------------------------------------------
    # House registry API helpers
    # ------------------------------------------------------------------
    def get_house_registry(self) -> Dict[str, Any]:
        return {"status": "ok", "registry": self.house_registry.to_dict()}

    def select_target_house(self, house_id: str) -> Dict[str, Any]:
        ok = self.house_registry.set_target(house_id)
        if not ok:
            return {"status": "error", "message": f"House '{house_id}' not found."}
        self.house_registry.save_to_file(self.args.houses_config)
        house = self.house_registry.get_house(house_id)
        return {
            "status": "ok",
            "message": f"Target set to '{house_id}'.",
            "target_house": house.to_dict() if house else None,
            "registry_summary": self.house_registry.get_status_summary(),
        }

    def mark_house_explored(self, house_id: str, *, person_found: bool = False,
                             person_location: Optional[Dict] = None, notes: str = "") -> Dict[str, Any]:
        ok = self.house_registry.mark_explored(
            house_id, person_found=person_found,
            person_location=person_location, notes=notes,
        )
        if not ok:
            return {"status": "error", "message": f"House '{house_id}' not found."}
        self.house_registry.save_to_file(self.args.houses_config)
        return {
            "status": "ok",
            "message": f"House '{house_id}' marked {'PERSON_FOUND' if person_found else 'EXPLORED'}.",
            "registry_summary": self.house_registry.get_status_summary(),
        }

    def navigate_step_to_house(self, house_id: str) -> Dict[str, Any]:
        """Execute ONE movement step toward the target house at cruise altitude."""
        house = self.house_registry.get_house(house_id)
        if house is None:
            return {"status": "error", "message": f"House '{house_id}' not found."}
        with self.lock:
            if not self.movement_enabled:
                return self.get_state(status="disabled", message="Enable movement first.")
            x, y, z, yaw = self.get_task_pose()
            tx, ty, tz_cruise = house.center_x, house.center_y, house.approach_z
            # Step 1: climb to cruise altitude
            if z < tz_cruise - 30:
                return self.move_relative(up_cm=30.0, action_name="nav_climb")
            # Step 2: yaw toward target
            dx, dy = tx - x, ty - y
            dist = (dx**2 + dy**2) ** 0.5
            if dist < 100:
                return self.get_state(message=f"Arrived near house '{house_id}'.")
            target_yaw = float(np.degrees(np.arctan2(dy, dx)))
            yaw_err = float(normalize_angle_deg(target_yaw - yaw))
            if abs(yaw_err) > 20:
                step = 20.0 if yaw_err > 0 else -20.0
                return self.move_relative(yaw_delta_deg=step, action_name="nav_yaw")
            # Step 3: fly forward
            step_cm = min(50.0, dist)
            return self.move_relative(forward_cm=step_cm, action_name="nav_forward")

    def set_movement_enabled(self, enabled: bool) -> Dict[str, Any]:
        with self.lock:
            self.movement_enabled = bool(enabled)
            if self.movement_enabled:
                self.command_task_yaw_deg = float(self.get_task_pose()[3])
            return self.get_state(
                message=(
                    f"Basic movement {'enabled' if self.movement_enabled else 'disabled'}."
                    + (" Synced heading." if self.movement_enabled else "")
                )
            )

    def move_relative(
        self,
        *,
        forward_cm: float = 0.0,
        right_cm: float = 0.0,
        up_cm: float = 0.0,
        yaw_delta_deg: float = 0.0,
        action_name: str = "custom",
    ) -> Dict[str, Any]:
        with self.lock:
            if not self.movement_enabled:
                return self.get_state(status="disabled", message="Basic movement is disabled. Enable it first.")
            x, y, z, task_yaw_deg = self.get_task_pose()
            actual_uav_yaw_before = float(task_yaw_deg)
            move_yaw_deg = float(actual_uav_yaw_before)
            theta = np.radians(move_yaw_deg)
            delta_x = float(forward_cm * np.cos(theta) - right_cm * np.sin(theta))
            delta_y = float(forward_cm * np.sin(theta) + right_cm * np.cos(theta))
            new_pos = [x + delta_x, y + delta_y, z + float(up_cm)]
            yaw_changed = abs(float(yaw_delta_deg)) > 1e-6
            if abs(float(forward_cm)) > 1e-6 or abs(float(right_cm)) > 1e-6 or abs(float(up_cm)) > 1e-6:
                self.set_task_position_only(new_pos)
            if yaw_changed:
                target_task_yaw = normalize_angle_deg(task_yaw_deg + float(yaw_delta_deg))
                self.command_task_yaw_deg = float(target_task_yaw)
                self.set_task_yaw_absolute(target_task_yaw)
            self.last_action = str(action_name or "custom")
            self.refresh_observations()
            self.command_task_yaw_deg = float(self.get_task_pose()[3])
            actual_task_yaw_after = self.get_task_pose()[3]
            actual_uav_yaw_after = self.get_task_pose()[3]
            logger.info(
                "Basic move action=%s local=(fwd=%.1f,right=%.1f,up=%.1f,yaw=%.1f) world=(dx=%.1f,dy=%.1f,dz=%.1f) "
                "pos=(%.1f, %.1f, %.1f) move_yaw=%.1f command_yaw=%.1f actual_task_yaw=%.1f actual_uav_yaw=%.1f",
                self.last_action,
                float(forward_cm),
                float(right_cm),
                float(up_cm),
                float(yaw_delta_deg),
                delta_x,
                delta_y,
                float(up_cm),
                new_pos[0],
                new_pos[1],
                new_pos[2],
                move_yaw_deg,
                float(self.command_task_yaw_deg),
                float(actual_task_yaw_after),
                float(actual_uav_yaw_after),
            )
            return self.get_state(message=f"Executed {self.last_action}.")

    def set_task_label(self, task_label: str) -> Dict[str, Any]:
        with self.lock:
            self.current_task_label = str(task_label or "").strip()
            return self.get_state(message="Task label updated.")

    def set_manual_pose(self, pose_payload: Dict[str, Any]) -> Dict[str, Any]:
        with self.lock:
            x = float(pose_payload["x"])
            y = float(pose_payload["y"])
            z = float(pose_payload["z"])
            target_task_yaw = normalize_angle_deg(float(pose_payload["yaw"]))
            self.set_task_position_only([x, y, z])
            self.command_task_yaw_deg = float(target_task_yaw)
            self.set_task_yaw_absolute(target_task_yaw, tolerance_deg=1.0, attempts=8)
            self.last_action = "set_pose"
            self.refresh_observations()
            self.command_task_yaw_deg = float(self.get_task_pose()[3])
            logger.info(
                "Manual pose set -> pos=(%.1f, %.1f, %.1f) target_task_yaw=%.1f actual_task_yaw=%.1f actual_uav_yaw=%.1f",
                x,
                y,
                z,
                target_task_yaw,
                float(self.get_task_pose()[3]),
                float(self.get_task_pose()[3]),
            )
            return self.get_state(message="Manual pose applied.")

    def get_frame_jpeg(self) -> bytes:
        with self.lock:
            if self.last_raw_frame is None:
                self.refresh_preview_only()
            if self.last_raw_frame is None:
                raise RuntimeError("No RGB frame available")
            encode_ok, encoded = cv2.imencode(".jpg", self.last_raw_frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.frame_jpeg_quality)])
            if not encode_ok:
                raise RuntimeError("Failed to encode RGB frame")
            return encoded.tobytes()

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
                source_mode=str(self.last_depth_summary.get("source_mode", "lesson4_depth_planar")),
            )
            encode_ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.frame_jpeg_quality)])
            if not encode_ok:
                raise RuntimeError("Failed to encode depth preview frame")
            return encoded.tobytes()

    def get_depth_raw_png(self) -> bytes:
        with self.lock:
            if self.last_depth_frame is None:
                self.refresh_depth_only()
            if self.last_depth_frame is None:
                raise RuntimeError("No depth frame available")
            depth_u16 = np.clip(np.nan_to_num(self.last_depth_frame, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 65535.0).astype(np.uint16)
            encode_ok, encoded = cv2.imencode(".png", depth_u16)
            if not encode_ok:
                raise RuntimeError("Failed to encode raw depth frame")
            return encoded.tobytes()

    def get_camera_info(self) -> Dict[str, Any]:
        with self.lock:
            if not self.last_depth_summary.get("camera_info"):
                self.refresh_depth_only()
            return dict(self.last_depth_summary.get("camera_info", {}))

    def capture_frame(self, label: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            if self.last_raw_frame is None or self.last_depth_frame is None:
                self.refresh_observations()
            if self.last_raw_frame is None:
                raise RuntimeError("No frame available for capture")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = f"_{label}" if label else ""
            capture_id = f"capture_{timestamp}{suffix}"
            rgb_path = os.path.join(self.args.capture_dir, f"{capture_id}_rgb.png")
            depth_path = os.path.join(self.args.capture_dir, f"{capture_id}_depth_cm.png")
            depth_preview_path = os.path.join(self.args.capture_dir, f"{capture_id}_depth_preview.png")
            camera_info_path = os.path.join(self.args.capture_dir, f"{capture_id}_camera_info.json")
            bundle_path = os.path.join(self.args.capture_dir, f"{capture_id}_bundle.json")

            cv2.imwrite(rgb_path, self.last_raw_frame)
            if self.last_depth_frame is not None:
                depth_to_save = np.clip(np.nan_to_num(self.last_depth_frame, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 65535.0)
                cv2.imwrite(depth_path, depth_to_save.astype(np.uint16))
                depth_preview = render_depth_preview(
                    self.last_depth_frame,
                    int(self.last_depth_summary.get("image_width", self.args.depth_preview_width)) or self.args.depth_preview_width,
                    int(self.last_depth_summary.get("image_height", self.args.depth_preview_height)) or self.args.depth_preview_height,
                    min_depth_cm=float(self.args.depth_min_cm),
                    max_depth_cm=float(self.args.depth_max_cm),
                    source_mode=str(self.last_depth_summary.get("source_mode", "lesson4_depth_planar")),
                )
                cv2.imwrite(depth_preview_path, depth_preview)
            camera_info = self.get_camera_info()
            with open(camera_info_path, "w", encoding="utf-8") as f:
                json.dump(camera_info, f, indent=2)
            bundle = {
                "capture_id": capture_id,
                "capture_time": timestamp,
                "env_id": self.args.env_id,
                "task_label": self.current_task_label,
                "last_action": self.last_action,
                "movement_enabled": self.movement_enabled,
                "pose": self._build_pose_state(),
                "depth": dict(self.last_depth_summary),
                "camera_info": camera_info,
                "rgb_image_path": rgb_path,
                "depth_image_path": depth_path if self.last_depth_frame is not None else None,
                "depth_preview_path": depth_preview_path if self.last_depth_frame is not None else None,
                "camera_info_path": camera_info_path,
                "bundle_path": bundle_path,
            }
            with open(bundle_path, "w", encoding="utf-8") as f:
                json.dump(bundle, f, indent=2)
            self.last_capture = bundle
            return {"status": "ok", "capture": bundle, "state": self.get_state(message=f"Captured {capture_id}.")}

    def shutdown_async(self) -> None:
        httpd = self.httpd
        if httpd is None:
            return

        def _shutdown() -> None:
            try:
                httpd.shutdown()
            except Exception as exc:
                logger.warning("Failed to shutdown HTTP server cleanly: %s", exc)

        threading.Thread(target=_shutdown, daemon=True).start()

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass


def make_handler(backend: BasicUAVControlBackend):
    class BasicControlRequestHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError) as exc:
                logger.warning("Client disconnected before JSON response was sent: %s", exc)

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            try:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError) as exc:
                logger.warning("Client disconnected before binary response was sent: %s", exc)

        def _read_json_body(self) -> Dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            payload = json.loads(raw.decode("utf-8"))
            return payload if isinstance(payload, dict) else {}

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("HTTP %s - %s", self.address_string(), fmt % args)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                if parsed.path in ("/", "/health"):
                    self._send_json({"status": "ok"})
                elif parsed.path == "/state":
                    self._send_json(backend.get_state())
                elif parsed.path == "/house_registry":
                    self._send_json(backend.get_house_registry())
                elif parsed.path == "/frame":
                    self._send_bytes(backend.get_frame_jpeg(), "image/jpeg")
                elif parsed.path in ("/depth", "/depth_frame"):
                    self._send_bytes(backend.get_depth_frame_jpeg(), "image/jpeg")
                elif parsed.path in ("/depth_raw", "/depth_raw.png"):
                    self._send_bytes(backend.get_depth_raw_png(), "image/png")
                elif parsed.path == "/camera_info":
                    self._send_json({"status": "ok", "camera_info": backend.get_camera_info()})
                else:
                    self._send_json({"status": "error", "message": "Not found"}, 404)
            except Exception as exc:
                logger.exception("GET %s failed", parsed.path)
                self._send_json({"status": "error", "message": str(exc)}, 500)

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            try:
                data = self._read_json_body()
                if parsed.path == "/basic_movement_enable":
                    self._send_json(backend.set_movement_enabled(bool(data.get("enabled", False))))
                elif parsed.path == "/move_relative":
                    self._send_json(
                        backend.move_relative(
                            forward_cm=float(data.get("forward_cm", 0.0)),
                            right_cm=float(data.get("right_cm", 0.0)),
                            up_cm=float(data.get("up_cm", 0.0)),
                            yaw_delta_deg=float(data.get("yaw_delta_deg", 0.0)),
                            action_name=str(data.get("action_name", "custom")),
                        )
                    )
                elif parsed.path == "/capture":
                    self._send_json(backend.capture_frame(label=data.get("label")))
                elif parsed.path == "/task":
                    self._send_json(backend.set_task_label(str(data.get("task_label", ""))))
                elif parsed.path == "/set_pose":
                    required_keys = {"x", "y", "z", "yaw"}
                    if not isinstance(data, dict) or not required_keys.issubset(set(data.keys())):
                        self._send_json({"status": "error", "message": "Expected JSON with keys: x, y, z, yaw"}, 400)
                    else:
                        self._send_json(backend.set_manual_pose(data))
                elif parsed.path == "/refresh":
                    with backend.lock:
                        backend.refresh_observations()
                        self._send_json(backend.get_state(message="Observations refreshed."))
                elif parsed.path == "/select_target_house":
                    house_id = str(data.get("house_id", ""))
                    if not house_id:
                        self._send_json({"status": "error", "message": "house_id required."}, 400)
                    else:
                        self._send_json(backend.select_target_house(house_id))
                elif parsed.path == "/mark_house_explored":
                    house_id = str(data.get("house_id", ""))
                    if not house_id:
                        self._send_json({"status": "error", "message": "house_id required."}, 400)
                    else:
                        self._send_json(backend.mark_house_explored(
                            house_id,
                            person_found=bool(data.get("person_found", False)),
                            person_location=data.get("person_location"),
                            notes=str(data.get("notes", "")),
                        ))
                elif parsed.path == "/navigate_step_to_house":
                    house_id = str(data.get("house_id", ""))
                    target = backend.house_registry.get_target_house()
                    if not house_id and target:
                        house_id = target.id
                    if not house_id:
                        self._send_json({"status": "error", "message": "No house_id and no target set."}, 400)
                    else:
                        self._send_json(backend.navigate_step_to_house(house_id))
                elif parsed.path == "/shutdown":
                    backend.shutdown_async()
                    self._send_json({"status": "ok", "message": "Shutdown requested."})
                else:
                    self._send_json({"status": "error", "message": "Not found"}, 404)
            except Exception as exc:
                logger.exception("POST %s failed", parsed.path)
                self._send_json({"status": "error", "message": str(exc)}, 500)

    return BasicControlRequestHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-latency basic UAV control server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5020)
    parser.add_argument("--env_id", default="UnrealTrack-SuburbNeighborhood_Day-ContinuousColor-v0")
    parser.add_argument("--env_bin_win", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time_dilation", type=int, default=10)
    parser.add_argument("--window_width", type=int, default=640)
    parser.add_argument("--window_height", type=int, default=480)
    parser.add_argument("--viewport_mode", default="free", choices=["first_person", "third_person", "free"])
    parser.add_argument("--viewport_offset_x", type=float, default=-220.0)
    parser.add_argument("--viewport_offset_y", type=float, default=0.0)
    parser.add_argument("--viewport_offset_z", type=float, default=90.0)
    parser.add_argument("--viewport_roll", type=float, default=0.0)
    parser.add_argument("--viewport_pitch", type=float, default=-12.0)
    parser.add_argument("--viewport_yaw", type=float, default=0.0)
    parser.add_argument("--free_view_offset_x", type=float, default=-220.0)
    parser.add_argument("--free_view_offset_y", type=float, default=140.0)
    parser.add_argument("--free_view_offset_z", type=float, default=50.0)
    parser.add_argument("--free_view_roll", type=float, default=0.0)
    parser.add_argument("--free_view_pitch", type=float, default=0.0)
    parser.add_argument("--free_view_yaw", type=float, default=0.0)
    parser.add_argument("--follow_free_view", action="store_true")
    parser.add_argument("--preview_mode", default="first_person", choices=["first_person", "third_person"])
    parser.add_argument("--preview_offset_x", type=float, default=-260.0)
    parser.add_argument("--preview_offset_y", type=float, default=0.0)
    parser.add_argument("--preview_offset_z", type=float, default=120.0)
    parser.add_argument("--preview_roll", type=float, default=0.0)
    parser.add_argument("--preview_pitch", type=float, default=-12.0)
    parser.add_argument("--preview_yaw", type=float, default=0.0)
    parser.add_argument("--default_task_label", default="")
    parser.add_argument("--capture_dir", default="./captures_remote")
    parser.add_argument("--task_json", default=None)
    parser.add_argument("--spawn_x", type=float, default=None)
    parser.add_argument("--spawn_y", type=float, default=None)
    parser.add_argument("--spawn_z", type=float, default=None)
    parser.add_argument("--spawn_yaw", type=float, default=None)
    parser.add_argument("--fixed_spawn_pose_file", default="")
    parser.add_argument("--start_with_basic_movement", action="store_true")
    parser.add_argument("--frame_jpeg_quality", type=int, default=85)
    parser.add_argument("--depth_preview_width", type=int, default=640)
    parser.add_argument("--depth_preview_height", type=int, default=480)
    parser.add_argument("--depth_min_cm", type=float, default=20.0)
    parser.add_argument("--depth_max_cm", type=float, default=1200.0)
    parser.add_argument("--default_depth_fov_deg", type=float, default=90.0)
    parser.add_argument("--depth_camera_frame_id", default="uav_depth_camera")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--houses_config", default="./houses_config.json",
                        help="Path to houses_config.json defining house positions and search state.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )
    backend = BasicUAVControlBackend(args)
    handler = make_handler(backend)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    backend.httpd = server
    logger.info("Basic UAV control server listening on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        backend.close()


if __name__ == "__main__":
    main()
