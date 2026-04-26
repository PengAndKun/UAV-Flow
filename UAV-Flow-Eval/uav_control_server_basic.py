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
import ctypes
import json
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from ctypes import wintypes

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
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(EVAL_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
DEFAULT_HOUSES_CONFIG = os.path.join(EVAL_DIR, "houses_config.json")

from phase2_multimodal_fusion_analysis import (  # noqa: E402
    DEFAULT_ENTRY_SEARCH_MEMORY_PATH,
    EntrySearchMemoryStore,
    run_phase2_fusion_analysis,
)


def now_timestamp() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def sanitize_fragment(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"[^0-9A-Za-z_\-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text[:80]


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
        if not os.path.isabs(self.args.houses_config):
            self.args.houses_config = os.path.abspath(os.path.join(EVAL_DIR, self.args.houses_config))
        self.lock = threading.RLock()
        self.unrealcv_lock = threading.RLock()
        self.httpd: Optional[ThreadingHTTPServer] = None
        self.last_raw_frame: Optional[np.ndarray] = None
        self.last_depth_frame: Optional[np.ndarray] = None
        self.last_capture: Optional[Dict[str, Any]] = None
        self.last_action = "idle"
        self.frame_index = 0
        self.last_observation_time = now_timestamp()
        self.last_observation_id = "frame_000000"
        self.last_door_open_state: Optional[bool] = None
        self.last_door_control_time = ""
        self.last_door_target_name = ""
        self.last_door_control_response = ""
        self.last_door_control_command = ""
        self.last_door_control_ok: Optional[bool] = None
        self.last_door_control_method = ""
        self.last_door_window_title = ""
        self.current_task_label = str(args.default_task_label or "")
        self.movement_enabled = bool(args.start_with_basic_movement)
        self.fixed_spawn_pose_path = os.path.abspath(args.fixed_spawn_pose_file) if args.fixed_spawn_pose_file else ""
        self.command_task_yaw_deg = 0.0
        self.memory_collection_root = os.path.abspath(
            args.memory_collection_root or os.path.join(args.capture_dir, "memory_collection_sessions")
        )
        self.memory_store_path = os.path.abspath(args.entry_search_memory_path or DEFAULT_ENTRY_SEARCH_MEMORY_PATH)
        self.memory_store = EntrySearchMemoryStore(self.memory_store_path)
        self.memory_collection_active = False
        self.memory_collection_episode_id = ""
        self.memory_collection_episode_label = ""
        self.memory_collection_started_at = ""
        self.memory_collection_dir = ""
        self.memory_collection_step_index = 0
        self.memory_collection_snapshot_count = 0
        self.last_memory_snapshot_before_path = ""
        self.last_memory_snapshot_after_path = ""
        self.last_memory_snapshot_time = ""
        self.memory_capture_root = ""
        self.last_memory_capture_run_dir = ""
        self.last_memory_capture_label = ""
        self.last_memory_capture_source = ""
        self.last_memory_capture_time = ""
        # --- House registry ---
        self.house_registry = HouseRegistry(args.houses_config)
        try:
            logger.info(
                "Loaded houses_config=%s houses=%d target=%s",
                self.args.houses_config,
                len(self.house_registry.get_all_houses()),
                self.house_registry.to_dict().get("current_target_id", ""),
            )
        except Exception:
            logger.exception("Failed to summarize house registry after load")
        self.overhead_cam_id = 0
        self.last_overhead_frame: Optional[np.ndarray] = None
        self.last_overhead_refresh_time = ""
        self.last_overhead_info: Dict[str, Any] = {}
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
        os.makedirs(self.memory_collection_root, exist_ok=True)
        self.memory_store.load()
        self.memory_store.ensure_from_houses_config(self.args.houses_config)
        self.memory_store.save()
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
        try:
            logger.info("Interactive doors discovered: count=%d", len(self._interactive_door_names()))
        except Exception:
            logger.exception("Failed to summarize interactive doors after initialization")

    def _interactive_door_names(self) -> List[str]:
        env_cfg = getattr(self.env.unwrapped, "env_configs", {}) if hasattr(self.env, "unwrapped") else {}
        raw = env_cfg.get("interactive_door", []) if isinstance(env_cfg, dict) else []
        if not isinstance(raw, list):
            return []
        names: List[str] = []
        for item in raw:
            name = str(item or "").strip()
            if name:
                names.append(name)
        return names

    def _nearest_interactive_door(self) -> Tuple[Optional[str], Optional[List[float]], Optional[float]]:
        door_names = self._interactive_door_names()
        if not door_names:
            return None, None, None
        x, y, _, _ = self.get_task_pose()
        best_name: Optional[str] = None
        best_loc: Optional[List[float]] = None
        best_dist: Optional[float] = None
        for door_name in door_names:
            try:
                with self.unrealcv_lock:
                    loc = self.env.unwrapped.unrealcv.get_obj_location(door_name)
                if not isinstance(loc, (list, tuple)) or len(loc) < 3:
                    continue
                dx = float(loc[0]) - float(x)
                dy = float(loc[1]) - float(y)
                dist = float(np.hypot(dx, dy))
                if best_dist is None or dist < best_dist:
                    best_name = door_name
                    best_loc = [float(loc[0]), float(loc[1]), float(loc[2])]
                    best_dist = dist
            except Exception as exc:
                logger.debug("Failed to query interactive door %s: %s", door_name, exc)
                continue
        return best_name, best_loc, best_dist

    def _looks_like_error_response(self, raw_response: str) -> bool:
        text = str(raw_response or "").strip().lower()
        if not text:
            return False
        return text.startswith("error") or " argument invalid" in text or "failed" in text

    def _interaction_window_tokens(self) -> List[str]:
        tokens: List[str] = []
        for raw in (
            os.path.splitext(os.path.basename(str(self.args.env_bin_win or "")))[0],
            str(self.args.env_id or ""),
            str(getattr(self.env.unwrapped, "env_name", "") or ""),
        ):
            token = str(raw or "").strip()
            if token and token not in tokens:
                tokens.append(token)
        return tokens

    def _find_interaction_window(self) -> Tuple[Optional[int], str]:
        if os.name != "nt":
            return None, ""
        user32 = ctypes.windll.user32
        tokens = [token.lower() for token in self._interaction_window_tokens()]
        if not tokens:
            return None, ""
        matches: List[Tuple[int, str]] = []

        @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        def enum_windows_proc(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            title = str(buffer.value or "").strip()
            if not title:
                return True
            lowered = title.lower()
            if any(token in lowered for token in tokens):
                matches.append((int(hwnd), title))
            return True

        user32.EnumWindows(enum_windows_proc, 0)
        if not matches:
            return None, ""
        hwnd, title = matches[0]
        return hwnd, title

    def _send_interact_e_to_window(self) -> Tuple[bool, str, str]:
        if os.name != "nt":
            return False, "", "Keyboard interaction is only supported on Windows."
        hwnd, title = self._find_interaction_window()
        if not hwnd:
            return False, "", "Could not find Unreal window for interaction."
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

        user32.ShowWindow(hwnd, 5)
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        user32.SetActiveWindow(hwnd)
        time.sleep(0.05)

        ULONG_PTR = wintypes.WPARAM
        INPUT_KEYBOARD = 1
        KEYEVENTF_KEYUP = 0x0002
        WM_KEYDOWN = 0x0100
        WM_KEYUP = 0x0101
        VK_E = 0x45

        class KEYBDINPUT(ctypes.Structure):
            _fields_ = [
                ("wVk", wintypes.WORD),
                ("wScan", wintypes.WORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class _INPUTUNION(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT)]

        class INPUT(ctypes.Structure):
            _fields_ = [("type", wintypes.DWORD), ("union", _INPUTUNION)]

        inputs = (INPUT * 2)()
        inputs[0] = INPUT(type=INPUT_KEYBOARD, union=_INPUTUNION(ki=KEYBDINPUT(wVk=VK_E, wScan=0, dwFlags=0, time=0, dwExtraInfo=0)))
        inputs[1] = INPUT(type=INPUT_KEYBOARD, union=_INPUTUNION(ki=KEYBDINPUT(wVk=VK_E, wScan=0, dwFlags=KEYEVENTF_KEYUP, time=0, dwExtraInfo=0)))

        kernel32.SetLastError(0)
        sent = int(user32.SendInput(len(inputs), inputs, ctypes.sizeof(INPUT)))
        if sent == len(inputs):
            return True, title, "SendInput:E"

        sendinput_error = int(ctypes.get_last_error())
        down_ok = int(user32.PostMessageW(hwnd, WM_KEYDOWN, VK_E, 0))
        up_ok = int(user32.PostMessageW(hwnd, WM_KEYUP, VK_E, 0))
        if down_ok and up_ok:
            return True, title, f"PostMessage:E fallback sendinput={sent} lasterr={sendinput_error}"
        return False, title, (
            f"SendInput returned {sent} lasterr={sendinput_error}; "
            f"PostMessage down={down_ok} up={up_ok}"
        )

    def hide_non_primary_agents(self) -> None:
        for idx, player_name in enumerate(self.env.unwrapped.player_list[1:], start=1):
            try:
                hide_pos = [0.0, 0.0, -10000.0 - 100.0 * idx]
                with self.unrealcv_lock:
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
        with self.unrealcv_lock:
            rotation = self.env.unwrapped.unrealcv.get_obj_rotation(self.player_name)
            if isinstance(rotation, (list, tuple)) and len(rotation) > 1:
                return normalize_angle_deg(float(rotation[1]))
        return 0.0

    def get_task_pose(self) -> List[float]:
        # In the basic-only controller, the displayed/input yaw is the real UAV yaw.
        with self.unrealcv_lock:
            location = self.env.unwrapped.unrealcv.get_obj_location(self.player_name)
            rotation = self.env.unwrapped.unrealcv.get_obj_rotation(self.player_name)
        yaw_deg = 0.0
        if isinstance(rotation, (list, tuple)) and len(rotation) > 1:
            yaw_deg = normalize_angle_deg(float(rotation[1]))
        return [float(location[0]), float(location[1]), float(location[2]), float(yaw_deg)]

    def set_task_pose(self, position: List[float], yaw_deg: float) -> None:
        self.set_task_position_only(position)
        self.set_task_yaw_absolute(float(yaw_deg))

    def set_task_position_only(self, position: List[float]) -> None:
        with self.unrealcv_lock:
            self.env.unwrapped.unrealcv.set_obj_location(self.player_name, list(position))

    def set_task_yaw_absolute(self, yaw_deg: float, tolerance_deg: float = 1.5, attempts: int = 6) -> float:
        target_task_yaw = normalize_angle_deg(float(yaw_deg))
        current_task_yaw = self.get_task_pose()[3]
        for _ in range(max(1, int(attempts))):
            # This blueprint helper is the stable drone-facing API in the existing teleop path.
            with self.unrealcv_lock:
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
        with self.unrealcv_lock:
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
        with self.unrealcv_lock:
            if self.args.preview_mode == "third_person":
                return get_third_person_preview_image(self.env, self.preview_cam_id, self.preview_offset, self.preview_rotation)
            set_cam(self.env, self.policy_cam_id)
            return self.env.unwrapped.unrealcv.get_image(self.policy_cam_id, "lit")

    def get_depth_observation(self) -> Tuple[np.ndarray, float]:
        with self.unrealcv_lock:
            set_cam(self.env, self.policy_cam_id)
            depth = self.env.unwrapped.unrealcv.get_depth(self.policy_cam_id)
            fov = self.env.unwrapped.unrealcv.get_cam_fov(self.policy_cam_id)
        return coerce_depth_planar_image(depth), self._coerce_fov_deg(fov)

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

    def _get_overhead_map_config(self) -> Dict[str, Any]:
        registry = self.house_registry.to_dict()
        world_bounds = registry.get("world_bounds", {}) if isinstance(registry, dict) else {}
        overhead_map = registry.get("overhead_map", {}) if isinstance(registry, dict) else {}

        min_x = float(world_bounds.get("min_x", 0.0))
        min_y = float(world_bounds.get("min_y", 0.0))
        max_x = float(world_bounds.get("max_x", 0.0))
        max_y = float(world_bounds.get("max_y", 0.0))
        if max_x <= min_x or max_y <= min_y:
            houses = self.house_registry.get_all_houses()
            if houses:
                min_x = min(h.center_x - h.radius_cm for h in houses) - 400.0
                min_y = min(h.center_y - h.radius_cm for h in houses) - 400.0
                max_x = max(h.center_x + h.radius_cm for h in houses) + 400.0
                max_y = max(h.center_y + h.radius_cm for h in houses) + 400.0
            else:
                min_x, min_y, max_x, max_y = 1000.0, -500.0, 5000.0, 3000.0

        span_x = max(1.0, max_x - min_x)
        span_y = max(1.0, max_y - min_y)
        center_x = float(overhead_map.get("center_x", (min_x + max_x) * 0.5))
        center_y = float(overhead_map.get("center_y", (min_y + max_y) * 0.5))
        height_z = float(overhead_map.get("height_z", max(span_x, span_y) * 1.25 + 400.0))
        yaw_deg = float(overhead_map.get("yaw_deg", 0.0))
        return {
            "world_bounds": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
            "center_x": center_x,
            "center_y": center_y,
            "height_z": height_z,
            "yaw_deg": yaw_deg,
            "image_path": str(overhead_map.get("image_path", "")),
        }

    def refresh_overhead_map_only(self) -> None:
        cfg = self._get_overhead_map_config()
        cam_loc = [float(cfg["center_x"]), float(cfg["center_y"]), float(cfg["height_z"])]
        cam_rot = [-90.0, float(cfg["yaw_deg"]), 0.0]
        with self.unrealcv_lock:
            self.env.unwrapped.unrealcv.set_cam_location(self.overhead_cam_id, cam_loc)
            self.env.unwrapped.unrealcv.set_cam_rotation(self.overhead_cam_id, cam_rot)
        time.sleep(0.06)
        with self.unrealcv_lock:
            self.last_overhead_frame = self.env.unwrapped.unrealcv.get_image(self.overhead_cam_id, "lit")
        self.last_overhead_refresh_time = now_timestamp()
        saved_image_path = ""
        image_path = str(cfg.get("image_path", "") or "").strip()
        if self.last_overhead_frame is not None and image_path:
            try:
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                if cv2.imwrite(image_path, self.last_overhead_frame):
                    saved_image_path = image_path
            except Exception as exc:
                logger.warning("Failed to save overhead map image to %s: %s", image_path, exc)
        self.last_overhead_info = {
            "cam_id": int(self.overhead_cam_id),
            "camera_location": cam_loc,
            "camera_rotation": cam_rot,
            "refresh_time": self.last_overhead_refresh_time,
            "saved_image_path": saved_image_path,
            "image_width": int(self.last_overhead_frame.shape[1]) if self.last_overhead_frame is not None else 0,
            "image_height": int(self.last_overhead_frame.shape[0]) if self.last_overhead_frame is not None else 0,
            **cfg,
        }

    def get_overhead_map_info(self) -> Dict[str, Any]:
        if not self.last_overhead_info:
            self.refresh_overhead_map_only()
        return dict(self.last_overhead_info)

    def set_overhead_calibration(
        self,
        *,
        anchors: List[Dict[str, Any]],
        image_width: int,
        image_height: int,
        image_path: str = "",
    ) -> Dict[str, Any]:
        if len(anchors) < 3:
            return {"status": "error", "message": "At least 3 anchors are required for calibration."}

        world_mat = []
        image_x = []
        image_y = []
        clean_anchors: List[Dict[str, Any]] = []
        for idx, anchor in enumerate(anchors, start=1):
            wx = float(anchor["world_x"])
            wy = float(anchor["world_y"])
            ix = float(anchor["image_x"])
            iy = float(anchor["image_y"])
            world_mat.append([wx, wy, 1.0])
            image_x.append(ix)
            image_y.append(iy)
            clean_anchors.append({
                "index": idx,
                "label": f"P{idx}",
                "world_x": wx,
                "world_y": wy,
                "image_x": ix,
                "image_y": iy,
            })

        A = np.asarray(world_mat, dtype=np.float64)
        bx = np.asarray(image_x, dtype=np.float64)
        by = np.asarray(image_y, dtype=np.float64)
        coeff_x, _, _, _ = np.linalg.lstsq(A, bx, rcond=None)
        coeff_y, _, _, _ = np.linalg.lstsq(A, by, rcond=None)
        affine = np.asarray([
            [float(coeff_x[0]), float(coeff_x[1]), float(coeff_x[2])],
            [float(coeff_y[0]), float(coeff_y[1]), float(coeff_y[2])],
        ], dtype=np.float64)

        pred_x = A @ coeff_x
        pred_y = A @ coeff_y
        errors = np.sqrt((pred_x - bx) ** 2 + (pred_y - by) ** 2)
        rmse_px = float(np.sqrt(np.mean(errors ** 2))) if errors.size else 0.0

        self.house_registry.set_overhead_calibration(
            anchors=clean_anchors,
            affine_world_to_image=affine.tolist(),
            image_width=int(image_width),
            image_height=int(image_height),
            rmse_px=rmse_px,
        )
        if image_path:
            self.house_registry.update_overhead_map({"image_path": str(image_path)})
        self.house_registry.save_to_file(self.args.houses_config)
        return {
            "status": "ok",
            "message": f"Overhead calibration saved with {len(clean_anchors)} anchors (rmse={rmse_px:.2f}px).",
            "calibration": self.house_registry.to_dict().get("overhead_map", {}).get("calibration", {}),
        }

    def clear_overhead_calibration(self) -> Dict[str, Any]:
        self.house_registry.clear_overhead_calibration()
        self.house_registry.save_to_file(self.args.houses_config)
        return {"status": "ok", "message": "Overhead calibration cleared."}

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

    def _build_door_control_state(self) -> Dict[str, Any]:
        nearest_name, nearest_loc, nearest_dist = self._nearest_interactive_door()
        available_count = len(self._interactive_door_names())
        return {
            "supported": bool(available_count),
            "mode": "player_blueprint_action",
            "control_player_name": str(self.player_name or ""),
            "available_count": int(available_count),
            "nearest_door_name": str(nearest_name or ""),
            "nearest_door_distance_cm": None if nearest_dist is None else float(nearest_dist),
            "nearest_door_location": nearest_loc,
            "last_requested_open": None if self.last_door_open_state is None else bool(self.last_door_open_state),
            "last_requested_label": (
                "unknown"
                if self.last_door_open_state is None
                else ("open" if self.last_door_open_state else "closed")
            ),
            "last_target_name": str(self.last_door_target_name or ""),
            "last_command": str(self.last_door_control_command or ""),
            "last_response": str(self.last_door_control_response or ""),
            "last_ok": None if self.last_door_control_ok is None else bool(self.last_door_control_ok),
            "last_method": str(self.last_door_control_method or ""),
            "last_window_title": str(self.last_door_window_title or ""),
            "last_control_time": str(self.last_door_control_time or ""),
        }

    def _sync_memory_runtime_context(self) -> Dict[str, Any]:
        mission = self.get_house_mission_state()
        target_house_id = str(mission.get("target_house_id", "") or "").strip()
        current_house_id = str(mission.get("current_house_id", "") or "").strip()
        self.memory_store.ensure_from_houses_config(self.args.houses_config)
        if target_house_id:
            self.memory_store.set_current_target_house(target_house_id)
        touched_ids: List[str] = []
        for house_id in (target_house_id, current_house_id):
            if house_id and house_id not in touched_ids:
                touched_ids.append(house_id)
        for house_id in touched_ids:
            patch = {
                "target_house_id": target_house_id,
                "current_house_id": current_house_id,
            }
            self.memory_store.update_working_memory(house_id, patch)
        return mission

    def _record_memory_action(self, action_name: str, *, increment_step: bool = True) -> None:
        mission = self._sync_memory_runtime_context()
        target_house_id = str(mission.get("target_house_id", "") or "").strip()
        current_house_id = str(mission.get("current_house_id", "") or "").strip()
        touched_ids: List[str] = []
        for house_id in (target_house_id, current_house_id):
            if house_id and house_id not in touched_ids:
                touched_ids.append(house_id)
        for house_id in touched_ids:
            self.memory_store.append_recent_action(house_id, action_name)
        if self.memory_collection_active and increment_step:
            self.memory_collection_step_index = int(self.memory_collection_step_index) + 1
        self.memory_store.save()

    def _build_memory_collection_state(self) -> Dict[str, Any]:
        mission = self.get_house_mission_state()
        current_target_id = str(self.memory_store.to_dict().get("current_target_house_id", "") or "")
        return {
            "active": bool(self.memory_collection_active),
            "episode_id": str(self.memory_collection_episode_id or ""),
            "episode_label": str(self.memory_collection_episode_label or ""),
            "started_at": str(self.memory_collection_started_at or ""),
            "collection_dir": str(self.memory_collection_dir or ""),
            "capture_root": str(self.memory_capture_root or ""),
            "step_index": int(self.memory_collection_step_index),
            "snapshot_count": int(self.memory_collection_snapshot_count),
            "memory_store_path": str(self.memory_store_path),
            "current_target_house_id": current_target_id,
            "current_house_id": str(mission.get("current_house_id", "") or ""),
            "last_snapshot_before_path": str(self.last_memory_snapshot_before_path or ""),
            "last_snapshot_after_path": str(self.last_memory_snapshot_after_path or ""),
            "last_snapshot_time": str(self.last_memory_snapshot_time or ""),
            "last_capture_run_dir": str(self.last_memory_capture_run_dir or ""),
            "last_capture_label": str(self.last_memory_capture_label or ""),
            "last_capture_source": str(self.last_memory_capture_source or ""),
            "last_capture_time": str(self.last_memory_capture_time or ""),
        }

    def _reset_memory_store(self) -> None:
        self.memory_store = EntrySearchMemoryStore(self.memory_store_path)
        self.memory_store.ensure_from_houses_config(self.args.houses_config)
        self._sync_memory_runtime_context()
        self.memory_store.save()

    def _snapshot_payload(
        self,
        snapshot_type: str,
        *,
        note: str = "",
        extra: Optional[Dict[str, Any]] = None,
        step_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        mission = self._sync_memory_runtime_context()
        payload = {
            "snapshot_type": str(snapshot_type or "manual"),
            "snapshot_time": now_timestamp(),
            "episode_id": str(self.memory_collection_episode_id or ""),
            "episode_label": str(self.memory_collection_episode_label or ""),
            "step_index": int(self.memory_collection_step_index if step_index is None else step_index),
            "task_label": str(self.current_task_label or ""),
            "last_action": str(self.last_action or ""),
            "pose": self._build_pose_state(),
            "house_mission": mission,
            "memory": self.memory_store.to_dict(),
            "note": str(note or ""),
        }
        if isinstance(extra, dict) and extra:
            payload["extra"] = extra
        return payload

    def _save_memory_snapshot_file(
        self,
        output_path: str,
        snapshot_type: str,
        *,
        note: str = "",
        extra: Optional[Dict[str, Any]] = None,
        step_index: Optional[int] = None,
    ) -> str:
        payload = self._snapshot_payload(snapshot_type, note=note, extra=extra, step_index=step_index)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        self.last_memory_snapshot_time = str(payload.get("snapshot_time", "") or "")
        self.memory_collection_snapshot_count = int(self.memory_collection_snapshot_count) + 1
        return output_path

    def start_memory_collection(self, *, episode_label: str = "", reset_store: bool = True) -> Dict[str, Any]:
        with self.lock:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            label_suffix = f"_{str(episode_label).strip()}" if str(episode_label or "").strip() else ""
            episode_id = f"memory_episode_{timestamp}{label_suffix}"
            collection_dir = os.path.join(self.memory_collection_root, episode_id)
            os.makedirs(collection_dir, exist_ok=True)
            self.memory_collection_active = True
            self.memory_collection_episode_id = episode_id
            self.memory_collection_episode_label = str(episode_label or "").strip()
            self.memory_collection_started_at = now_timestamp()
            self.memory_collection_dir = collection_dir
            self.memory_collection_step_index = 0
            self.memory_collection_snapshot_count = 0
            self.last_memory_snapshot_before_path = ""
            self.last_memory_snapshot_after_path = ""
            self.last_memory_snapshot_time = ""
            self.memory_capture_root = os.path.join(collection_dir, "memory_fusion_captures")
            os.makedirs(self.memory_capture_root, exist_ok=True)
            self.last_memory_capture_run_dir = ""
            self.last_memory_capture_label = ""
            self.last_memory_capture_source = ""
            self.last_memory_capture_time = ""
            if reset_store:
                self._reset_memory_store()
            start_snapshot_path = os.path.join(collection_dir, "entry_search_memory_snapshot_start.json")
            self.last_memory_snapshot_before_path = self._save_memory_snapshot_file(
                start_snapshot_path,
                "episode_start",
                note="Memory collection started.",
            )
            logger.info(
                "Memory collection started episode=%s dir=%s reset_store=%s",
                episode_id,
                collection_dir,
                reset_store,
            )
            return self.get_state(message=f"Memory collection started: {episode_id}")

    def stop_memory_collection(self) -> Dict[str, Any]:
        with self.lock:
            if self.memory_collection_active and self.memory_collection_dir:
                stop_snapshot_path = os.path.join(self.memory_collection_dir, "entry_search_memory_snapshot_stop.json")
                self.last_memory_snapshot_after_path = self._save_memory_snapshot_file(
                    stop_snapshot_path,
                    "episode_stop",
                    note="Memory collection stopped.",
                )
            episode_id = self.memory_collection_episode_id
            self.memory_collection_active = False
            self.memory_collection_episode_label = self.memory_collection_episode_label
            logger.info("Memory collection stopped episode=%s", episode_id)
            return self.get_state(message=f"Memory collection stopped: {episode_id or 'none'}")

    def reset_memory_collection(self) -> Dict[str, Any]:
        with self.lock:
            self._reset_memory_store()
            self.memory_collection_step_index = 0
            self.memory_collection_snapshot_count = 0
            self.last_memory_snapshot_before_path = ""
            self.last_memory_snapshot_after_path = ""
            self.last_memory_snapshot_time = ""
            self.last_memory_capture_run_dir = ""
            self.last_memory_capture_label = ""
            self.last_memory_capture_source = ""
            self.last_memory_capture_time = ""
            logger.info("Memory store reset path=%s", self.memory_store_path)
            return self.get_state(message="Memory store reset.")

    def save_memory_snapshot(self, *, note: str = "", snapshot_type: str = "manual") -> Dict[str, Any]:
        with self.lock:
            if self.memory_collection_active and self.memory_collection_dir:
                snapshot_dir = self.memory_collection_dir
            else:
                snapshot_dir = self.memory_collection_root
            snapshot_name = f"entry_search_memory_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            snapshot_path = os.path.join(snapshot_dir, snapshot_name)
            self.last_memory_snapshot_after_path = self._save_memory_snapshot_file(
                snapshot_path,
                snapshot_type,
                note=note,
            )
            logger.info("Memory snapshot saved type=%s path=%s", snapshot_type, snapshot_path)
            return self.get_state(message=f"Memory snapshot saved: {snapshot_name}")

    def capture_memory_fusion_sample(
        self,
        *,
        label: str = "",
        capture_source: str = "manual",
        note: str = "",
    ) -> Dict[str, Any]:
        with self.lock:
            if not self.memory_collection_active or not self.memory_collection_dir:
                return self.get_state(status="error", message="Start a memory collection episode first.")

            self._sync_memory_runtime_context()
            if self.last_raw_frame is None or self.last_depth_frame is None:
                self.refresh_observations()
            if self.last_raw_frame is None or self.last_depth_frame is None:
                raise RuntimeError("No synchronized RGB/depth frame available for memory capture.")

            step_index = int(self.memory_collection_step_index)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_label = sanitize_fragment(label)
            safe_source = sanitize_fragment(capture_source) or "manual"
            label_suffix = f"_{safe_label}" if safe_label else ""
            run_name = f"memory_capture_{timestamp}_step{step_index:04d}_{safe_source}{label_suffix}"
            capture_root = self.memory_capture_root or os.path.join(self.memory_collection_dir, "memory_fusion_captures")
            os.makedirs(capture_root, exist_ok=True)
            run_dir = os.path.join(capture_root, run_name)
            labeling_dir = os.path.join(run_dir, "labeling")
            os.makedirs(labeling_dir, exist_ok=True)

            step_note = str(note or "").strip()
            state = self.get_state()
            camera_info = self.get_camera_info()
            rgb_bgr = np.ascontiguousarray(self.last_raw_frame.copy())
            depth_raw = np.clip(
                np.nan_to_num(self.last_depth_frame, nan=0.0, posinf=0.0, neginf=0.0),
                0.0,
                65535.0,
            ).astype(np.uint16)

            extra = {
                "capture_source": safe_source,
                "step_index": step_index,
                "requested_label": str(label or ""),
                "note": step_note,
                "run_dir": run_dir,
            }
            before_snapshot_path = os.path.join(labeling_dir, "entry_search_memory_snapshot_before.json")
            self.last_memory_snapshot_before_path = self._save_memory_snapshot_file(
                before_snapshot_path,
                "before_memory_capture",
                note="Snapshot before memory capture analyze.",
                extra=extra,
                step_index=step_index,
            )

            result = run_phase2_fusion_analysis(
                rgb_bgr=rgb_bgr,
                depth_raw=depth_raw,
                existing_run_dir=Path(run_dir),
                label="",
                camera_info=camera_info if isinstance(camera_info, dict) else {},
                state=state if isinstance(state, dict) else {},
            )

            # Reload the shared memory store after fusion so the "after" snapshot
            # reflects the same updated memory that fusion_result.json embeds.
            self.memory_store.load()
            self._sync_memory_runtime_context()
            self.memory_store.save()

            mission_after = self.get_house_mission_state()
            fusion_payload = result.get("fusion", {}) if isinstance(result.get("fusion"), dict) else {}
            target_context = (
                fusion_payload.get("target_context", {})
                if isinstance(fusion_payload.get("target_context"), dict)
                else {}
            )
            after_snapshot_path = os.path.join(labeling_dir, "entry_search_memory_snapshot_after.json")
            self.last_memory_snapshot_after_path = self._save_memory_snapshot_file(
                after_snapshot_path,
                "after_memory_capture",
                note="Snapshot after memory capture analyze.",
                extra=extra,
                step_index=step_index,
            )

            target_house_id = str(
                mission_after.get("target_house_id")
                or target_context.get("target_house_id")
                or ""
            ).strip()
            current_house_id = str(
                mission_after.get("current_house_id")
                or target_context.get("current_house_id")
                or ""
            ).strip()
            sample_metadata = {
                "episode_id": str(self.memory_collection_episode_id or ""),
                "memory_episode_id": str(self.memory_collection_episode_id or ""),
                "episode_label": str(self.memory_collection_episode_label or ""),
                "step_index": step_index,
                "memory_step_index": step_index,
                "capture_source": safe_source,
                "capture_label": str(label or ""),
                "requested_label": str(label or ""),
                "note": step_note,
                "collection_dir": str(self.memory_collection_dir or ""),
                "capture_root": str(capture_root),
                "capture_run_dir": str(run_dir),
                "labeling_dir": str(labeling_dir),
                "memory_store_path": str(self.memory_store_path),
                "memory_snapshot_before_path": str(self.last_memory_snapshot_before_path or ""),
                "memory_snapshot_after_path": str(self.last_memory_snapshot_after_path or ""),
                "target_house_id": target_house_id,
                "current_house_id": current_house_id,
            }
            temporal_context = {
                "episode_id": str(self.memory_collection_episode_id or ""),
                "step_index": step_index,
                "previous_action": str(self.last_action or ""),
                "capture_source": safe_source,
                "capture_time": now_timestamp(),
                "current_target_house_id": target_house_id,
                "current_house_id": current_house_id,
                "memory_snapshot_before_path": str(self.last_memory_snapshot_before_path or ""),
                "memory_snapshot_after_path": str(self.last_memory_snapshot_after_path or ""),
                "memory_store_path": str(self.memory_store_path),
            }
            with open(os.path.join(labeling_dir, "sample_metadata.json"), "w", encoding="utf-8") as fh:
                json.dump(sample_metadata, fh, indent=2, ensure_ascii=False)
            with open(os.path.join(labeling_dir, "temporal_context.json"), "w", encoding="utf-8") as fh:
                json.dump(temporal_context, fh, indent=2, ensure_ascii=False)

            self.last_memory_capture_run_dir = str(run_dir)
            self.last_memory_capture_label = str(label or "")
            self.last_memory_capture_source = str(safe_source or "")
            self.last_memory_capture_time = now_timestamp()
            self.last_capture = {
                "capture_id": run_name,
                "capture_time": self.last_memory_capture_time,
                "capture_source": safe_source,
                "run_dir": str(run_dir),
                "labeling_dir": str(labeling_dir),
                "fusion_overlay_path": result.get("fusion_overlay_path"),
                "bundle_path": os.path.join(labeling_dir, "labeling_manifest.json"),
                "step_index": step_index,
            }
            logger.info(
                "Memory capture analyze done -> run=%s source=%s step=%d label=%s",
                run_name,
                safe_source,
                step_index,
                str(label or ""),
            )
            return {
                "status": "ok",
                "message": f"Memory capture analyze saved: {run_name}",
                "state": self.get_state(message=f"Memory capture analyze saved: {run_name}"),
                "result": result,
                "run_dir": str(run_dir),
                "labeling_dir": str(labeling_dir),
                "memory_snapshot_before_path": self.last_memory_snapshot_before_path,
                "memory_snapshot_after_path": self.last_memory_snapshot_after_path,
            }

    def get_memory_state(self) -> Dict[str, Any]:
        with self.lock:
            mission = self._sync_memory_runtime_context()
            target_house_id = str(mission.get("target_house_id", "") or "").strip()
            current_house_id = str(mission.get("current_house_id", "") or "").strip()
            return {
                "status": "ok",
                "memory_collection": self._build_memory_collection_state(),
                "house_mission": mission,
                "memory_store_path": str(self.memory_store_path),
                "memory_store": self.memory_store.to_dict(),
                "target_house_memory": self.memory_store.get_house_memory(target_house_id) if target_house_id else None,
                "current_house_memory": self.memory_store.get_house_memory(current_house_id) if current_house_id else None,
            }

    def _capture_memory_artifacts(
        self,
        *,
        snapshot_dir: str,
        snapshot_prefix: str,
        capture_id: str,
        capture_time: str,
        rgb_path: str,
        depth_path: Optional[str],
        depth_preview_path: Optional[str],
        bundle_path: str,
        step_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        effective_step_index = int(self.memory_collection_step_index if step_index is None else step_index)
        if self.memory_collection_active and step_index is not None:
            self.memory_collection_step_index = max(int(self.memory_collection_step_index), int(step_index))
        result: Dict[str, Any] = {
            "memory_collection_active": bool(self.memory_collection_active),
            "memory_store_path": str(self.memory_store_path),
            "memory_episode_id": str(self.memory_collection_episode_id or ""),
            "memory_episode_label": str(self.memory_collection_episode_label or ""),
            "memory_step_index": effective_step_index,
            "memory_snapshot_before_path": None,
            "memory_snapshot_after_path": None,
        }
        if not self.memory_collection_active:
            return result
        before_path = os.path.join(snapshot_dir, f"{snapshot_prefix}_entry_search_memory_snapshot_before.json")
        after_path = os.path.join(snapshot_dir, f"{snapshot_prefix}_entry_search_memory_snapshot_after.json")
        extra = {
            "capture_id": capture_id,
            "capture_time": capture_time,
            "bundle_path": bundle_path,
            "rgb_image_path": rgb_path,
            "depth_image_path": depth_path,
            "depth_preview_path": depth_preview_path,
        }
        self.last_memory_snapshot_before_path = self._save_memory_snapshot_file(
            before_path,
            "before_capture",
            note="Snapshot before saving capture bundle.",
            extra=extra,
            step_index=effective_step_index,
        )
        mission = self.get_house_mission_state()
        target_house_id = str(mission.get("target_house_id", "") or "").strip()
        current_house_id = str(mission.get("current_house_id", "") or "").strip()
        episodic_snapshot = {
            "snapshot_id": f"{capture_id}_after",
            "episode_id": str(self.memory_collection_episode_id or ""),
            "step_index": effective_step_index,
            "capture_id": capture_id,
            "capture_time": capture_time,
            "task_label": str(self.current_task_label or ""),
            "last_action": str(self.last_action or ""),
            "pose": self._build_pose_state(),
            "rgb_image_path": rgb_path,
            "depth_image_path": depth_path,
            "depth_preview_path": depth_preview_path,
            "bundle_path": bundle_path,
            "current_house_id": current_house_id,
            "target_house_id": target_house_id,
        }
        touched_ids: List[str] = []
        for house_id in (target_house_id, current_house_id):
            if house_id and house_id not in touched_ids:
                touched_ids.append(house_id)
        for house_id in touched_ids:
            self.memory_store.append_episodic_snapshot(house_id, episodic_snapshot)
        self.memory_store.save()
        self.last_memory_snapshot_after_path = self._save_memory_snapshot_file(
            after_path,
            "after_capture",
            note="Snapshot after saving capture bundle and episodic update.",
            extra=extra,
            step_index=effective_step_index,
        )
        result["memory_snapshot_before_path"] = self.last_memory_snapshot_before_path
        result["memory_snapshot_after_path"] = self.last_memory_snapshot_after_path
        return result

    def get_state(self, *, status: str = "ok", message: str = "") -> Dict[str, Any]:
        try:
            self._sync_memory_runtime_context()
        except Exception:
            logger.exception("Failed to sync memory runtime context while building state")
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
            "house_mission": self.get_house_mission_state(),
            "overhead_map": dict(self.last_overhead_info),
            "door_control": self._build_door_control_state(),
            "memory_collection": self._build_memory_collection_state(),
        }

    # ------------------------------------------------------------------
    # House registry API helpers
    # ------------------------------------------------------------------
    def get_house_mission_state(self) -> Dict[str, Any]:
        x, y, z, yaw = self.get_task_pose()
        current_house = self.house_registry.get_containing_house(x, y)
        nearest_house = self.house_registry.get_nearest_house(x, y)
        nearest_unsearched = self.house_registry.get_nearest_unsearched(x, y)
        target_house = self.house_registry.get_target_house()
        distance_to_target = target_house.distance_to(x, y) if target_house is not None else None
        inside_target = (
            target_house is not None
            and distance_to_target is not None
            and distance_to_target <= float(target_house.radius_cm)
        )
        return {
            "current_house_id": current_house.id if current_house else "",
            "current_house_name": current_house.name if current_house else "",
            "current_house_status": current_house.status.value if current_house else "",
            "nearest_house_id": nearest_house.id if nearest_house else "",
            "nearest_house_name": nearest_house.name if nearest_house else "",
            "nearest_unsearched_house_id": nearest_unsearched.id if nearest_unsearched else "",
            "nearest_unsearched_house_name": nearest_unsearched.name if nearest_unsearched else "",
            "target_house_id": target_house.id if target_house else "",
            "target_house_name": target_house.name if target_house else "",
            "target_house_status": target_house.status.value if target_house else "",
            "distance_to_target_cm": float(distance_to_target) if distance_to_target is not None else None,
            "inside_current_house": current_house is not None,
            "inside_target_house": bool(inside_target),
            "uav_x": float(x),
            "uav_y": float(y),
            "uav_z": float(z),
            "uav_yaw": float(yaw),
        }

    def get_house_registry(self) -> Dict[str, Any]:
        return {"status": "ok", "registry": self.house_registry.to_dict()}

    def reload_house_registry(self) -> Dict[str, Any]:
        try:
            self.house_registry.load_from_file(self.args.houses_config)
            return {
                "status": "ok",
                "message": "House registry reloaded.",
                "registry": self.house_registry.to_dict(),
                "house_mission": self.get_house_mission_state(),
            }
        except Exception as exc:
            return {"status": "error", "message": f"Failed to reload house registry: {exc}"}

    def select_target_house(self, house_id: str) -> Dict[str, Any]:
        ok = self.house_registry.set_target(house_id)
        if not ok:
            return {"status": "error", "message": f"House '{house_id}' not found."}
        self.house_registry.save_to_file(self.args.houses_config)
        self.memory_store.ensure_from_houses_config(self.args.houses_config)
        self.memory_store.set_current_target_house(house_id)
        self.memory_store.save()
        house = self.house_registry.get_house(house_id)
        return {
            "status": "ok",
            "message": f"Target set to '{house_id}'.",
            "target_house": house.to_dict() if house else None,
            "registry_summary": self.house_registry.get_status_summary(),
            "house_mission": self.get_house_mission_state(),
        }

    def select_nearest_unsearched_house(self) -> Dict[str, Any]:
        x, y, _, _ = self.get_task_pose()
        house = self.house_registry.get_nearest_unsearched(x, y)
        if house is None:
            return {
                "status": "error",
                "message": "No unsearched house available.",
                "registry_summary": self.house_registry.get_status_summary(),
                "house_mission": self.get_house_mission_state(),
            }
        return self.select_target_house(house.id)

    def mark_house_explored(self, house_id: str, *, person_found: bool = False,
                             person_location: Optional[Dict] = None, notes: str = "") -> Dict[str, Any]:
        ok = self.house_registry.mark_explored(
            house_id, person_found=person_found,
            person_location=person_location, notes=notes,
        )
        if not ok:
            return {"status": "error", "message": f"House '{house_id}' not found."}
        self.house_registry.save_to_file(self.args.houses_config)
        memory = self.memory_store.get_house_memory(house_id, ensure=True)
        if isinstance(memory, dict):
            memory["house_status"] = "PERSON_FOUND" if person_found else "EXPLORED"
        self.memory_store.save()
        return {
            "status": "ok",
            "message": f"House '{house_id}' marked {'PERSON_FOUND' if person_found else 'EXPLORED'}.",
            "registry_summary": self.house_registry.get_status_summary(),
            "house_mission": self.get_house_mission_state(),
        }

    def mark_current_house_explored(self, *, person_found: bool = False,
                                    person_location: Optional[Dict] = None, notes: str = "") -> Dict[str, Any]:
        x, y, _, _ = self.get_task_pose()
        house = self.house_registry.get_containing_house(x, y)
        if house is None:
            return {
                "status": "error",
                "message": "UAV is not inside any configured house boundary.",
                "registry_summary": self.house_registry.get_status_summary(),
                "house_mission": self.get_house_mission_state(),
            }
        return self.mark_house_explored(
            house.id,
            person_found=person_found,
            person_location=person_location,
            notes=notes,
        )

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

    def set_open_door_state(self, is_open: bool) -> Dict[str, Any]:
        with self.lock:
            door_name, door_loc, door_distance = self._nearest_interactive_door()
            if not door_name:
                logger.warning("Door control requested open=%s but no interactive door is available.", is_open)
                self.last_door_control_ok = False
                self.last_door_control_response = "No interactive door found."
                self.last_door_control_command = ""
                return self.get_state(status="error", message="No interactive door found in current environment configuration.")
            try:
                command = self.env.unwrapped.unrealcv.set_open_door(self.player_name, 1 if is_open else 0, return_cmd=True)
                raw_response = ""
                logger.info(
                    "Door control request -> action=%s player=%s nearest=%s loc=%s distance_cm=%s cmd=%s",
                    "open" if is_open else "close",
                    self.player_name,
                    door_name,
                    door_loc,
                    f"{door_distance:.1f}" if door_distance is not None else "n/a",
                    command,
                )
                with self.unrealcv_lock:
                    raw_response = str(self.env.unwrapped.unrealcv.client.request(command))
                self.last_door_open_state = bool(is_open)
                self.last_door_control_time = now_timestamp()
                self.last_door_target_name = str(door_name)
                self.last_door_control_command = str(command)
                self.last_door_control_response = raw_response
                self.last_door_control_ok = not self._looks_like_error_response(raw_response)
                self.last_door_control_method = "player_blueprint_set_open_door"
                self.last_door_window_title = ""
                self.last_action = "door_open" if is_open else "door_close"
                distance_note = "" if door_distance is None else f" nearest={door_distance:.1f}cm"
                logger.info(
                    "Door control completed -> action=%s player=%s nearest=%s loc=%s distance_cm=%s ok=%s response=%s",
                    "open" if is_open else "close",
                    self.player_name,
                    door_name,
                    door_loc,
                    f"{door_distance:.1f}" if door_distance is not None else "n/a",
                    self.last_door_control_ok,
                    raw_response,
                )
                if not self.last_door_control_ok:
                    return self.get_state(status="error", message=f"Door command rejected for player {self.player_name} near {door_name}.{distance_note} response={raw_response}")
                return self.get_state(message=f"Requested door {'open' if is_open else 'close'} via player {self.player_name} near {door_name}.{distance_note}")
            except Exception as exc:
                self.last_door_control_time = now_timestamp()
                self.last_door_target_name = str(door_name)
                self.last_door_control_command = str(locals().get("command", ""))
                self.last_door_control_response = str(exc)
                self.last_door_control_ok = False
                self.last_door_control_method = "player_blueprint_set_open_door"
                self.last_door_window_title = ""
                logger.warning("Door control failed for player=%s nearest=%s open=%s: %s", self.player_name, door_name, is_open, exc)
                return self.get_state(status="error", message=f"Door control failed: {exc}")

    def toggle_open_door_state(self) -> Dict[str, Any]:
        next_state = True if self.last_door_open_state is None else (not bool(self.last_door_open_state))
        return self.set_open_door_state(next_state)

    def interact_open_door_with_key(self) -> Dict[str, Any]:
        with self.lock:
            door_name, door_loc, door_distance = self._nearest_interactive_door()
            ok, window_title, response = self._send_interact_e_to_window()
            self.last_door_control_time = now_timestamp()
            self.last_door_target_name = str(door_name or "")
            self.last_door_control_command = "SendInput:E"
            self.last_door_control_response = str(response)
            self.last_door_control_ok = bool(ok)
            self.last_door_control_method = "keyboard_interact_e"
            self.last_door_window_title = str(window_title or "")
            self.last_action = "door_interact_e"
            logger.info(
                "Door interact(E) -> target=%s loc=%s distance_cm=%s window=%s ok=%s response=%s",
                door_name or "",
                door_loc,
                f"{door_distance:.1f}" if door_distance is not None else "n/a",
                window_title or "",
                ok,
                response,
            )
            if not ok:
                return self.get_state(status="error", message=f"Door interact(E) failed: {response}")
            self._record_memory_action("door_interact_e")
            return self.get_state(message=f"Door interact(E) sent to window '{window_title}'")

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
            self._record_memory_action(self.last_action)
            return self.get_state(message=f"Executed {self.last_action}.")

    def set_task_label(self, task_label: str) -> Dict[str, Any]:
        with self.lock:
            self.current_task_label = str(task_label or "").strip()
            self._sync_memory_runtime_context()
            self.memory_store.save()
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
            self._record_memory_action("set_pose")
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

    def get_overhead_map_jpeg(self) -> bytes:
        with self.lock:
            if self.last_overhead_frame is None:
                self.refresh_overhead_map_only()
            if self.last_overhead_frame is None:
                raise RuntimeError("No overhead map frame available")
            encode_ok, encoded = cv2.imencode(".jpg", self.last_overhead_frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.frame_jpeg_quality)])
            if not encode_ok:
                raise RuntimeError("Failed to encode overhead map frame")
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
            memory_meta = self._capture_memory_artifacts(
                snapshot_dir=self.args.capture_dir,
                snapshot_prefix=capture_id,
                capture_id=capture_id,
                capture_time=timestamp,
                rgb_path=rgb_path,
                depth_path=depth_path if self.last_depth_frame is not None else None,
                depth_preview_path=depth_preview_path if self.last_depth_frame is not None else None,
                bundle_path=bundle_path,
            )
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
                **memory_meta,
            }
            with open(bundle_path, "w", encoding="utf-8") as f:
                json.dump(bundle, f, indent=2)
            self.last_capture = bundle
            return {"status": "ok", "capture": bundle, "state": self.get_state(message=f"Captured {capture_id}.")}

    def _save_capture_bundle_to_directory(
        self,
        output_dir: str,
        *,
        capture_id: str,
        capture_time: str,
        step_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        rgb_dir = os.path.join(output_dir, "rgb")
        depth_dir = os.path.join(output_dir, "depth_cm")
        depth_preview_dir = os.path.join(output_dir, "depth_preview")
        meta_dir = os.path.join(output_dir, "meta")
        for path in (rgb_dir, depth_dir, depth_preview_dir, meta_dir):
            os.makedirs(path, exist_ok=True)

        step_suffix = f"step_{int(step_index):02d}" if step_index is not None else capture_id
        rgb_path = os.path.join(rgb_dir, f"{step_suffix}_rgb.png")
        depth_path = os.path.join(depth_dir, f"{step_suffix}_depth_cm.png")
        depth_preview_path = os.path.join(depth_preview_dir, f"{step_suffix}_depth_preview.png")
        camera_info_path = os.path.join(meta_dir, f"{step_suffix}_camera_info.json")
        bundle_path = os.path.join(meta_dir, f"{step_suffix}_bundle.json")

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
        memory_meta = self._capture_memory_artifacts(
            snapshot_dir=meta_dir,
            snapshot_prefix=step_suffix,
            capture_id=capture_id,
            capture_time=capture_time,
            rgb_path=rgb_path,
            depth_path=depth_path if self.last_depth_frame is not None else None,
            depth_preview_path=depth_preview_path if self.last_depth_frame is not None else None,
            bundle_path=bundle_path,
            step_index=step_index,
        )
        bundle = {
            "capture_id": capture_id,
            "capture_time": capture_time,
            "step_index": int(step_index) if step_index is not None else None,
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
            **memory_meta,
        }
        with open(bundle_path, "w", encoding="utf-8") as f:
            json.dump(bundle, f, indent=2)
        return bundle

    def capture_phase1_spin_scan(
        self,
        *,
        label: Optional[str] = None,
        num_steps: int = 12,
        settle_time_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        with self.lock:
            steps = max(1, int(num_steps))
            settle_time_s = max(0.0, float(0.20 if settle_time_s is None else settle_time_s))
            scan_root = os.path.join(self.args.capture_dir, "phase1_spin_scans")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            label_suffix = f"_{str(label).strip()}" if str(label or "").strip() else ""
            scan_id = f"phase1_spin_scan_{timestamp}{label_suffix}"
            scan_dir = os.path.join(scan_root, scan_id)
            os.makedirs(scan_dir, exist_ok=True)
            base_step_index = int(self.memory_collection_step_index)

            start_pose = self.get_task_pose()
            start_yaw = float(start_pose[3])
            step_angle_deg = 360.0 / float(steps)
            captures: List[Dict[str, Any]] = []
            memory_snapshot_start_path = ""
            memory_snapshot_end_path = ""

            if self.memory_collection_active:
                memory_snapshot_start_path = self._save_memory_snapshot_file(
                    os.path.join(scan_dir, "phase1_scan_memory_snapshot_start.json"),
                    "phase1_scan_start",
                    note="Phase1 spin scan started.",
                    extra={"scan_id": scan_id, "num_steps": steps},
                    step_index=base_step_index,
                )

            logger.info(
                "Phase1 spin scan started id=%s steps=%d start_pose=(%.1f, %.1f, %.1f, %.1f)",
                scan_id,
                steps,
                float(start_pose[0]),
                float(start_pose[1]),
                float(start_pose[2]),
                start_yaw,
            )

            for step_index in range(steps):
                target_yaw = normalize_angle_deg(start_yaw + step_index * step_angle_deg)
                self.command_task_yaw_deg = float(target_yaw)
                self.set_task_yaw_absolute(target_yaw, tolerance_deg=1.0, attempts=8)
                # Let the view settle briefly after rotation so RGB/depth are
                # captured after the scene finishes updating.
                time.sleep(settle_time_s)
                self.refresh_observations()
                capture_time = datetime.now().isoformat(timespec="milliseconds")
                capture_id = f"{scan_id}_step_{step_index:02d}"
                global_step_index = base_step_index + step_index
                bundle = self._save_capture_bundle_to_directory(
                    scan_dir,
                    capture_id=capture_id,
                    capture_time=capture_time,
                    step_index=global_step_index,
                )
                captures.append(bundle)

            # Restore the original heading after the scan.
            self.command_task_yaw_deg = float(start_yaw)
            self.set_task_yaw_absolute(start_yaw, tolerance_deg=1.0, attempts=8)
            self.refresh_observations()
            self.last_action = "phase1_spin_scan"
            self.memory_collection_step_index = base_step_index + steps
            self._record_memory_action("phase1_spin_scan", increment_step=False)
            if self.memory_collection_active:
                memory_snapshot_end_path = self._save_memory_snapshot_file(
                    os.path.join(scan_dir, "phase1_scan_memory_snapshot_end.json"),
                    "phase1_scan_end",
                    note="Phase1 spin scan completed.",
                    extra={"scan_id": scan_id, "num_steps": steps},
                    step_index=self.memory_collection_step_index,
                )

            manifest = {
                "scan_id": scan_id,
                "scan_time": timestamp,
                "scan_dir": scan_dir,
                "task_label": self.current_task_label,
                "num_steps": steps,
                "step_angle_deg": step_angle_deg,
                "settle_time_s": settle_time_s,
                "start_pose": {
                    "x": float(start_pose[0]),
                    "y": float(start_pose[1]),
                    "z": float(start_pose[2]),
                    "yaw": start_yaw,
                },
                "end_pose": self._build_pose_state(),
                "memory_collection_active": bool(self.memory_collection_active),
                "memory_episode_id": str(self.memory_collection_episode_id or ""),
                "memory_step_index_start": base_step_index,
                "memory_step_index_end": int(self.memory_collection_step_index),
                "memory_snapshot_start_path": memory_snapshot_start_path or None,
                "memory_snapshot_end_path": memory_snapshot_end_path or None,
                "captures": captures,
            }
            manifest_path = os.path.join(scan_dir, "phase1_scan_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            self.last_capture = {
                "capture_id": scan_id,
                "capture_time": timestamp,
                "bundle_path": manifest_path,
                "scan_dir": scan_dir,
                "num_steps": steps,
            }

            logger.info("Phase1 spin scan completed id=%s dir=%s steps=%d", scan_id, scan_dir, steps)
            return {
                "status": "ok",
                "scan": manifest,
                "manifest_path": manifest_path,
                "state": self.get_state(message=f"Phase1 spin scan saved to {scan_dir}."),
            }

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
                elif parsed.path == "/memory_state":
                    self._send_json(backend.get_memory_state())
                elif parsed.path == "/house_registry":
                    self._send_json(backend.get_house_registry())
                elif parsed.path == "/house_mission":
                    self._send_json({"status": "ok", "house_mission": backend.get_house_mission_state()})
                elif parsed.path == "/overhead_map_info":
                    self._send_json({"status": "ok", "overhead_map": backend.get_overhead_map_info()})
                elif parsed.path == "/overhead_map_frame":
                    self._send_bytes(backend.get_overhead_map_jpeg(), "image/jpeg")
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
                elif parsed.path == "/capture_phase1_spin_scan":
                    self._send_json(
                        backend.capture_phase1_spin_scan(
                            label=data.get("label"),
                            num_steps=int(data.get("num_steps", 12)),
                            settle_time_s=float(data.get("settle_time_s", 0.20)),
                        )
                    )
                elif parsed.path == "/memory_collection_start":
                    self._send_json(
                        backend.start_memory_collection(
                            episode_label=str(data.get("episode_label", "") or ""),
                            reset_store=bool(data.get("reset_store", True)),
                        )
                    )
                elif parsed.path == "/memory_collection_stop":
                    self._send_json(backend.stop_memory_collection())
                elif parsed.path == "/memory_collection_reset":
                    self._send_json(backend.reset_memory_collection())
                elif parsed.path == "/memory_snapshot":
                    self._send_json(
                        backend.save_memory_snapshot(
                            note=str(data.get("note", "") or ""),
                            snapshot_type=str(data.get("snapshot_type", "manual") or "manual"),
                        )
                    )
                elif parsed.path == "/memory_capture_analyze":
                    self._send_json(
                        backend.capture_memory_fusion_sample(
                            label=str(data.get("label", "") or ""),
                            capture_source=str(data.get("capture_source", "manual") or "manual"),
                            note=str(data.get("note", "") or ""),
                        )
                    )
                elif parsed.path == "/task":
                    self._send_json(backend.set_task_label(str(data.get("task_label", ""))))
                elif parsed.path == "/door_open":
                    self._send_json(backend.set_open_door_state(True))
                elif parsed.path == "/door_close":
                    self._send_json(backend.set_open_door_state(False))
                elif parsed.path == "/door_toggle":
                    self._send_json(backend.toggle_open_door_state())
                elif parsed.path == "/door_interact_e":
                    self._send_json(backend.interact_open_door_with_key())
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
                elif parsed.path == "/refresh_overhead_map":
                    with backend.lock:
                        backend.refresh_overhead_map_only()
                        self._send_json({"status": "ok", "overhead_map": backend.get_overhead_map_info()})
                elif parsed.path == "/set_overhead_calibration":
                    anchors = data.get("anchors", [])
                    image_width = int(data.get("image_width", 0))
                    image_height = int(data.get("image_height", 0))
                    image_path = str(data.get("image_path", "") or "")
                    self._send_json(backend.set_overhead_calibration(
                        anchors=anchors if isinstance(anchors, list) else [],
                        image_width=image_width,
                        image_height=image_height,
                        image_path=image_path,
                    ))
                elif parsed.path == "/clear_overhead_calibration":
                    self._send_json(backend.clear_overhead_calibration())
                elif parsed.path == "/reload_house_registry":
                    self._send_json(backend.reload_house_registry())
                elif parsed.path == "/select_target_house":
                    house_id = str(data.get("house_id", ""))
                    if not house_id:
                        self._send_json({"status": "error", "message": "house_id required."}, 400)
                    else:
                        self._send_json(backend.select_target_house(house_id))
                elif parsed.path == "/select_nearest_unsearched_house":
                    self._send_json(backend.select_nearest_unsearched_house())
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
                elif parsed.path == "/mark_current_house_explored":
                    self._send_json(backend.mark_current_house_explored(
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
    parser.add_argument("--memory_collection_root", default="",
                        help="Optional directory for memory collection sessions. Defaults to <capture_dir>/memory_collection_sessions.")
    parser.add_argument("--entry_search_memory_path", default=DEFAULT_ENTRY_SEARCH_MEMORY_PATH,
                        help="Path to the shared entry_search_memory.json file used during collection.")
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
    parser.add_argument("--houses_config", default=DEFAULT_HOUSES_CONFIG,
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
