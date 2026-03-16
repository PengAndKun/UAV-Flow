"""
Manual UAV teleoperation with keyboard control and image capture.

Features:
- keyboard/manual control of the UAV inside Unreal
- a small Tk control panel with movement buttons
- a live OpenCV preview window
- a Capture button that saves the current image and pose metadata
- an auto-capture mode to verify image access without opening the UI
"""

import argparse
import json
import logging
import os
import time
import tkinter as tk
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gym
import gym_unrealcv
import numpy as np
import sys

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

PREVIEW_WINDOW_NAME = "UAV Manual Preview"


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


class ManualUAVTeleop:
    """Manual teleop loop with a control panel and capture support."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root: Optional[tk.Tk] = None
        self.status_var: Optional[tk.StringVar] = None
        self.last_frame: Optional[np.ndarray] = None
        self.last_raw_frame: Optional[np.ndarray] = None
        self.last_capture_path: Optional[str] = None
        self.preview_running = False

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
        self.position_free_view_once()
        self.refresh_preview()

    def apply_initial_task_or_spawn(self) -> None:
        """Optionally position the drone and target from a task JSON or CLI spawn pose."""
        if self.args.task_json:
            with open(self.args.task_json, "r", encoding="utf-8") as f:
                task_data = json.load(f)
            obj_info = build_obj_info(task_data)
            create_obj_if_needed(self.env, obj_info)
            initial_pos = task_data.get("initial_pos")
            if isinstance(initial_pos, list) and len(initial_pos) >= 5:
                self.set_dataset_pose(initial_pos[0:3], float(initial_pos[4]))
                logger.info("Loaded initial pose from task JSON: %s", initial_pos[:5])
                return

        if self.args.spawn_x is not None and self.args.spawn_y is not None and self.args.spawn_z is not None:
            spawn_yaw = self.args.spawn_yaw if self.args.spawn_yaw is not None else self.get_dataset_pose()[3]
            self.set_dataset_pose(
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
        """Read the Unreal actor yaw directly from the environment."""
        rotation = self.env.unwrapped.unrealcv.get_obj_rotation(self.player_name)
        if isinstance(rotation, (list, tuple)) and len(rotation) > 1:
            return float(rotation[1])
        return 0.0

    def get_dataset_pose(self) -> List[float]:
        """Return [x, y, z, yaw] in the same yaw convention used by task JSONs."""
        location = self.env.unwrapped.unrealcv.get_obj_location(self.player_name)
        dataset_yaw = normalize_angle_deg(self.get_env_yaw_deg() + 180.0)
        return [float(location[0]), float(location[1]), float(location[2]), dataset_yaw]

    def set_dataset_pose(self, position: List[float], yaw_deg: float) -> None:
        """Set pose using the task-JSON yaw convention."""
        self.env.unwrapped.unrealcv.set_obj_location(self.player_name, list(position))
        self.env.unwrapped.unrealcv.set_rotation(self.player_name, float(yaw_deg) - 180.0)

    def position_free_view_once(self) -> None:
        """Optionally place the native Unreal free view near the UAV once at startup."""
        if self.args.viewport_mode != "free" or not self.args.follow_free_view:
            return
        pose = self.get_dataset_pose()
        focus_pose = [pose[0], pose[1], pose[2], 0.0, pose[3]]
        set_free_view_near_pose(self.env, focus_pose, self.free_view_offset, self.free_view_rotation)

    def get_preview_frame(self) -> np.ndarray:
        """Capture the current preview image."""
        if self.args.preview_mode == "third_person":
            return get_third_person_preview_image(
                self.env,
                self.preview_cam_id,
                self.preview_offset,
                self.preview_rotation,
            )
        set_cam(self.env, self.policy_cam_id)
        return self.env.unwrapped.unrealcv.get_image(self.policy_cam_id, "lit")

    def refresh_preview(self) -> None:
        """Update the OpenCV preview window and pose label."""
        raw_frame = self.get_preview_frame()
        self.last_raw_frame = raw_frame
        frame = raw_frame
        if self.args.preview_width > 0 and self.args.preview_height > 0:
            frame = cv2.resize(frame, (self.args.preview_width, self.args.preview_height))
        self.last_frame = frame
        if not self.args.hide_preview_window:
            cv2.namedWindow(PREVIEW_WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(PREVIEW_WINDOW_NAME, frame)
            cv2.waitKey(1)

        pose = self.get_dataset_pose()
        status = (
            f"Pose x={pose[0]:.1f} y={pose[1]:.1f} z={pose[2]:.1f} yaw={pose[3]:.1f} | "
            f"step={self.args.move_step_cm:.1f}cm yaw_step={self.args.yaw_step_deg:.1f}deg"
        )
        if self.status_var is not None:
            self.status_var.set(status)

    def move_relative(self, forward_cm: float = 0.0, right_cm: float = 0.0, up_cm: float = 0.0, yaw_delta_deg: float = 0.0) -> None:
        """Move the UAV relative to its current heading."""
        x, y, z, yaw_deg = self.get_dataset_pose()
        theta = np.radians(yaw_deg)
        delta_x = float(forward_cm * np.cos(theta) - right_cm * np.sin(theta))
        delta_y = float(forward_cm * np.sin(theta) + right_cm * np.cos(theta))
        new_pos = [x + delta_x, y + delta_y, z + up_cm]
        new_yaw = normalize_angle_deg(yaw_deg + yaw_delta_deg)
        self.set_dataset_pose(new_pos, new_yaw)
        logger.info(
            "Manual move -> pos=(%.1f, %.1f, %.1f) yaw=%.1f",
            new_pos[0],
            new_pos[1],
            new_pos[2],
            new_yaw,
        )
        self.refresh_preview()

    def capture_current_frame(self) -> str:
        """Save the latest preview frame and a small metadata JSON."""
        if self.last_raw_frame is None:
            self.refresh_preview()
        if self.last_raw_frame is None:
            raise RuntimeError("No frame available for capture")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = os.path.join(self.args.capture_dir, f"capture_{timestamp}.png")
        meta_path = os.path.join(self.args.capture_dir, f"capture_{timestamp}.json")
        cv2.imwrite(image_path, self.last_raw_frame)
        metadata = {
            "env_id": self.args.env_id,
            "task_json": self.args.task_json,
            "preview_mode": self.args.preview_mode,
            "pose": self.get_dataset_pose(),
            "capture_time": timestamp,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        self.last_capture_path = image_path
        logger.info("Captured preview image: %s", image_path)
        if self.status_var is not None:
            self.status_var.set(f"Captured image: {image_path}")
        return image_path

    def schedule_preview_refresh(self) -> None:
        """Periodic preview refresh for the Tk UI."""
        if self.root is None:
            return
        self.refresh_preview()
        self.root.after(self.args.preview_interval_ms, self.schedule_preview_refresh)

    def on_close(self) -> None:
        """Close UI and Unreal resources."""
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            self.env.close()
        except Exception:
            pass
        if self.root is not None:
            self.root.destroy()

    def bind_keys(self) -> None:
        """Bind keyboard shortcuts to discrete UAV actions."""
        if self.root is None:
            return

        bindings = {
            "w": lambda: self.move_relative(forward_cm=self.args.move_step_cm),
            "s": lambda: self.move_relative(forward_cm=-self.args.move_step_cm),
            "a": lambda: self.move_relative(right_cm=-self.args.move_step_cm),
            "d": lambda: self.move_relative(right_cm=self.args.move_step_cm),
            "r": lambda: self.move_relative(up_cm=self.args.vertical_step_cm),
            "f": lambda: self.move_relative(up_cm=-self.args.vertical_step_cm),
            "q": lambda: self.move_relative(yaw_delta_deg=-self.args.yaw_step_deg),
            "e": lambda: self.move_relative(yaw_delta_deg=self.args.yaw_step_deg),
            "c": self.capture_current_frame,
            "v": self.refresh_preview,
        }
        for key, callback in bindings.items():
            self.root.bind(f"<KeyPress-{key}>", lambda _event, cb=callback: cb())
            self.root.bind(f"<KeyPress-{key.upper()}>", lambda _event, cb=callback: cb())

    def build_ui(self) -> None:
        """Build the Tk control panel."""
        self.root = tk.Tk()
        self.root.title("UAV Manual Teleop")
        self.root.geometry("520x260")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.status_var = tk.StringVar(value="Initializing...")

        header = tk.Label(
            self.root,
            text="Keyboard: W/S/A/D move, R/F up-down, Q/E yaw, C capture",
            anchor="w",
            justify="left",
        )
        header.pack(fill="x", padx=12, pady=(10, 6))

        status = tk.Label(self.root, textvariable=self.status_var, anchor="w", justify="left")
        status.pack(fill="x", padx=12, pady=(0, 10))

        control_frame = tk.Frame(self.root)
        control_frame.pack(fill="x", padx=12, pady=4)

        buttons: List[Tuple[str, Any]] = [
            ("Forward (W)", lambda: self.move_relative(forward_cm=self.args.move_step_cm)),
            ("Backward (S)", lambda: self.move_relative(forward_cm=-self.args.move_step_cm)),
            ("Left (A)", lambda: self.move_relative(right_cm=-self.args.move_step_cm)),
            ("Right (D)", lambda: self.move_relative(right_cm=self.args.move_step_cm)),
            ("Up (R)", lambda: self.move_relative(up_cm=self.args.vertical_step_cm)),
            ("Down (F)", lambda: self.move_relative(up_cm=-self.args.vertical_step_cm)),
            ("Yaw Left (Q)", lambda: self.move_relative(yaw_delta_deg=-self.args.yaw_step_deg)),
            ("Yaw Right (E)", lambda: self.move_relative(yaw_delta_deg=self.args.yaw_step_deg)),
            ("Capture (C)", self.capture_current_frame),
            ("Refresh (V)", self.refresh_preview),
        ]

        for idx, (label, callback) in enumerate(buttons):
            btn = tk.Button(control_frame, text=label, command=callback, width=20)
            row = idx // 2
            col = idx % 2
            btn.grid(row=row, column=col, padx=6, pady=6, sticky="ew")

        control_frame.grid_columnconfigure(0, weight=1)
        control_frame.grid_columnconfigure(1, weight=1)

        hint = tk.Label(
            self.root,
            text=f"Preview window: {PREVIEW_WINDOW_NAME}\nCapture folder: {self.args.capture_dir}",
            anchor="w",
            justify="left",
        )
        hint.pack(fill="x", padx=12, pady=(6, 10))

        self.bind_keys()

    def run(self) -> None:
        """Run auto-capture or the full control panel."""
        if self.args.auto_capture_once:
            time.sleep(self.args.auto_capture_wait_s)
            self.refresh_preview()
            path = self.capture_current_frame()
            print(path)
            if self.args.hold_after_capture_s > 0:
                time.sleep(self.args.hold_after_capture_s)
            if self.args.force_exit_after_auto_capture:
                sys.stdout.flush()
                os._exit(0)
            self.on_close()
            return

        self.build_ui()
        self.schedule_preview_refresh()
        assert self.root is not None
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual UAV teleoperation with capture")
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
    parser.add_argument("--preview_mode", default="first_person", choices=["first_person", "third_person"], help="Preview window mode")
    parser.add_argument("--preview_width", type=int, default=960, help="Preview window width")
    parser.add_argument("--preview_height", type=int, default=540, help="Preview window height")
    parser.add_argument("--preview_offset_x", type=float, default=-260.0, help="Third-person preview X offset")
    parser.add_argument("--preview_offset_y", type=float, default=0.0, help="Third-person preview Y offset")
    parser.add_argument("--preview_offset_z", type=float, default=120.0, help="Third-person preview Z offset")
    parser.add_argument("--preview_roll", type=float, default=0.0, help="Third-person preview roll")
    parser.add_argument("--preview_pitch", type=float, default=-12.0, help="Third-person preview pitch")
    parser.add_argument("--preview_yaw", type=float, default=0.0, help="Third-person preview yaw offset")
    parser.add_argument("--preview_interval_ms", type=int, default=120, help="Preview refresh interval in ms")
    parser.add_argument("--move_step_cm", type=float, default=20.0, help="Forward/left-right translation step in cm")
    parser.add_argument("--vertical_step_cm", type=float, default=20.0, help="Up/down translation step in cm")
    parser.add_argument("--yaw_step_deg", type=float, default=5.0, help="Yaw step in degrees")
    parser.add_argument("--capture_dir", default="./captures_manual", help="Directory used for captured preview images")
    parser.add_argument("--task_json", default=None, help="Optional task JSON used to place the UAV and any target object")
    parser.add_argument("--spawn_x", type=float, default=None, help="Optional UAV spawn x")
    parser.add_argument("--spawn_y", type=float, default=None, help="Optional UAV spawn y")
    parser.add_argument("--spawn_z", type=float, default=None, help="Optional UAV spawn z")
    parser.add_argument("--spawn_yaw", type=float, default=None, help="Optional UAV spawn yaw in task-json convention")
    parser.add_argument("--auto_capture_once", action="store_true", help="Capture one frame immediately and exit")
    parser.add_argument("--auto_capture_wait_s", type=float, default=2.0, help="Wait time before the auto capture")
    parser.add_argument("--hold_after_capture_s", type=float, default=1.0, help="Hold time after auto capture before exit")
    parser.add_argument("--hide_preview_window", action="store_true", help="Do not show the OpenCV preview window")
    parser.add_argument("--force_exit_after_auto_capture", action="store_true", help="Force-exit after auto capture to avoid simulator close hangs")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )
    teleop = ManualUAVTeleop(args)
    teleop.run()


if __name__ == "__main__":
    main()
