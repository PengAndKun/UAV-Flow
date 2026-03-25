from __future__ import annotations

import argparse
import json
import logging
import os
import threading
import time
import tkinter as tk
from tkinter import filedialog
from typing import Any, Dict, List, Optional
from urllib import request

import cv2
import numpy as np
from PIL import Image, ImageTk

from map_overhead_widget import OverheadMapWidget

logger = logging.getLogger(__name__)

MOVE_COMMANDS: Dict[str, Dict[str, Any]] = {
    "w": {"forward_cm": 20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "forward"},
    "s": {"forward_cm": -20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "backward"},
    "a": {"forward_cm": 0.0, "right_cm": -20.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "left"},
    "d": {"forward_cm": 0.0, "right_cm": 20.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "right"},
    "r": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 20.0, "yaw_delta_deg": 0.0, "action_name": "up"},
    "f": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": -20.0, "yaw_delta_deg": 0.0, "action_name": "down"},
    "q": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": -30.0, "action_name": "yaw_left"},
    "e": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 30.0, "action_name": "yaw_right"},
    "x": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "hold"},
}


class Client:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def get_json(self, path: str) -> Dict[str, Any]:
        req = request.Request(f"{self.base_url}{path}", method="GET")
        with request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def post_json(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = json.dumps(payload or {}).encode("utf-8")
        req = request.Request(f"{self.base_url}{path}", data=body, headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_image(self, path: str) -> np.ndarray:
        req = request.Request(f"{self.base_url}{path}", method="GET")
        with request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read()
        image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to decode image from {path}")
        return image


class Panel:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.client = Client(f"http://{args.host}:{args.port}", args.timeout_s)
        self.root = tk.Tk()
        self.root.title("UAV Basic Controller")
        self.root.geometry("1240x920")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.status_var = tk.StringVar(value="Ready")
        self.pose_var = tk.StringVar(value="Pose: waiting...")
        self.depth_var = tk.StringVar(value="Depth: waiting...")
        self.control_var = tk.StringVar(value="Movement: waiting...")
        self.capture_var = tk.StringVar(value="Capture: waiting...")
        self.mission_var = tk.StringVar(value="Mission: idle")
        self.current_house_var = tk.StringVar(value="Current house: none")
        self.target_house_var = tk.StringVar(value="Target: none")
        self.target_dist_var = tk.StringVar(value="Target distance: n/a")
        self.map_status_var = tk.StringVar(value="Map: closed")
        self.map_pose_var = tk.StringVar(value="Map pose: n/a")
        self.calib_var = tk.StringVar(value="Calibration: none")
        self.anchor_world_x_var = tk.StringVar(value="")
        self.anchor_world_y_var = tk.StringVar(value="")

        self.task_label_var = tk.StringVar(value="")
        self.capture_label_var = tk.StringVar(value="")
        self.sequence_var = tk.StringVar(value="")
        self.sequence_delay_var = tk.StringVar(value="260")
        self.auto_state_var = tk.BooleanVar(value=True)
        self.auto_rgb_var = tk.BooleanVar(value=False)
        self.auto_depth_var = tk.BooleanVar(value=False)
        self.show_houses_var = tk.BooleanVar(value=False)
        self.show_route_var = tk.BooleanVar(value=False)

        self.preview_window: Optional[tk.Toplevel] = None
        self.preview_label: Optional[tk.Label] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.depth_window: Optional[tk.Toplevel] = None
        self.depth_label: Optional[tk.Label] = None
        self.depth_photo: Optional[ImageTk.PhotoImage] = None
        self.map_window: Optional[tk.Toplevel] = None
        self.map_widget: Optional[OverheadMapWidget] = None
        self.open_map_window: Optional[tk.Toplevel] = None
        self.open_map_widget: Optional[OverheadMapWidget] = None
        self.pose_text: Optional[tk.Text] = None
        self.movement_toggle_button: Optional[tk.Button] = None
        self.calibration_anchors: List[Dict[str, float]] = []
        self.pending_anchor_world: Optional[Dict[str, float]] = None
        self.pending_image_anchor: Optional[Dict[str, float]] = None
        self.loaded_map_image_path: str = ""
        self.loaded_map_image: Optional[np.ndarray] = None

        self.manual_request_inflight = False
        self.move_request_inflight = False
        self.state_refresh_inflight = False
        self.preview_refresh_inflight = False
        self.depth_refresh_inflight = False
        self.map_refresh_inflight = False
        self.map_background_refresh_inflight = False
        self.background_pause_until = 0.0
        self.sequence_thread: Optional[threading.Thread] = None
        self.sequence_stop_event = threading.Event()
        self.movement_enabled_state = False

        self.eval_dir = os.path.dirname(os.path.abspath(__file__))
        self.houses_config_path = os.path.join(self.eval_dir, "houses_config.json")
        self.default_map_path = os.path.join(self.eval_dir, "map", "qq.png")

        self.build_ui()
        for delay, fn in ((200, self.schedule_state_refresh), (350, self.schedule_preview_refresh), (500, self.schedule_depth_refresh), (1200, self.schedule_map_refresh)):
            self.root.after(delay, fn)

    def build_ui(self) -> None:
        root = self.root
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        status = tk.LabelFrame(root, text="Runtime Status")
        status.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=8, pady=8)
        for idx, var in enumerate((self.status_var, self.pose_var, self.depth_var, self.control_var, self.capture_var, self.mission_var, self.current_house_var, self.target_house_var, self.target_dist_var, self.map_status_var, self.map_pose_var, self.calib_var)):
            tk.Label(status, textvariable=var, anchor="w", justify="left", font=("Consolas", 11)).grid(row=idx, column=0, sticky="ew", padx=6, pady=2)
        status.grid_columnconfigure(0, weight=1)

        left = tk.Frame(root); left.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 8)); left.grid_columnconfigure(0, weight=1)
        right = tk.Frame(root); right.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=(0, 8)); right.grid_columnconfigure(0, weight=1)

        task = tk.LabelFrame(left, text="Task And Capture"); task.grid(row=0, column=0, sticky="ew", pady=(0, 8)); task.grid_columnconfigure(1, weight=1)
        tk.Label(task, text="Task Label").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(task, textvariable=self.task_label_var).grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        tk.Button(task, text="Set Task", command=self.on_set_task).grid(row=0, column=2, padx=6, pady=6)
        tk.Label(task, text="Capture Label").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(task, textvariable=self.capture_label_var).grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        tk.Button(task, text="Capture", command=self.on_capture).grid(row=1, column=2, padx=6, pady=6)
        tk.Label(task, text="Init Pose JSON").grid(row=2, column=0, sticky="nw", padx=6, pady=6)
        self.pose_text = tk.Text(task, width=42, height=6)
        self.pose_text.grid(row=2, column=1, sticky="ew", padx=6, pady=6)
        self.pose_text.insert("1.0", json.dumps({"x": 2359.9, "y": 85.3, "z": 225.0, "yaw": -1.7}, indent=2))
        tk.Button(task, text="Set Pose", command=self.on_set_pose).grid(row=2, column=2, padx=6, pady=6, sticky="n")

        move = tk.LabelFrame(left, text="Basic Movement"); move.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        self.movement_toggle_button = tk.Button(move, text="Enable Basic Movement", command=self.on_toggle_movement, width=24)
        self.movement_toggle_button.grid(row=0, column=0, columnspan=3, pady=(6, 10))
        for label, symbol, row, col in (
            ("Yaw Left (Q)", "q", 1, 0), ("Forward (W)", "w", 1, 1), ("Yaw Right (E)", "e", 1, 2),
            ("Left (A)", "a", 2, 0), ("Hold (X)", "x", 2, 1), ("Right (D)", "d", 2, 2),
            ("Up (R)", "r", 3, 0), ("Backward (S)", "s", 3, 1), ("Down (F)", "f", 3, 2),
        ):
            tk.Button(move, text=label, command=lambda s=symbol: self.send_move_symbol(s), width=18).grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
        for col in range(3): move.grid_columnconfigure(col, weight=1)

        seq = tk.LabelFrame(left, text="Sequence Control"); seq.grid(row=2, column=0, sticky="ew", pady=(0, 8)); seq.grid_columnconfigure(1, weight=1)
        tk.Label(seq, text="Symbols").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(seq, textvariable=self.sequence_var).grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        tk.Button(seq, text="Execute Sequence", command=self.on_execute_sequence).grid(row=0, column=2, padx=6, pady=6)
        tk.Label(seq, text="Delay ms").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(seq, textvariable=self.sequence_delay_var, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=6)
        tk.Button(seq, text="Stop Sequence", command=self.on_stop_sequence).grid(row=1, column=2, padx=6, pady=6)
        tk.Label(seq, text="Use w/s/a/d/r/f/q/e/x. Example: wwwqdd", anchor="w").grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 6))

        preview = tk.LabelFrame(left, text="Preview Windows"); preview.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        tk.Button(preview, text="Toggle RGB", command=self.toggle_preview_window, width=16).grid(row=0, column=0, padx=6, pady=6)
        tk.Button(preview, text="Refresh RGB", command=self.refresh_preview_window, width=16).grid(row=0, column=1, padx=6, pady=6)
        tk.Checkbutton(preview, text="Auto RGB", variable=self.auto_rgb_var).grid(row=0, column=2, padx=6, pady=6)
        tk.Button(preview, text="Toggle Depth", command=self.toggle_depth_window, width=16).grid(row=1, column=0, padx=6, pady=6)
        tk.Button(preview, text="Refresh Depth", command=self.refresh_depth_window, width=16).grid(row=1, column=1, padx=6, pady=6)
        tk.Checkbutton(preview, text="Auto Depth", variable=self.auto_depth_var).grid(row=1, column=2, padx=6, pady=6)

        mission = tk.LabelFrame(right, text="House Mission"); mission.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        for col in range(2): mission.grid_columnconfigure(col, weight=1)
        tk.Button(mission, text="Open", command=self.toggle_open_map_window).grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        tk.Button(mission, text="Setting Map", command=self.toggle_map_window).grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        tk.Button(mission, text="Auto-Select Nearest", command=self.on_auto_select_house).grid(row=1, column=0, padx=6, pady=6, sticky="ew")
        tk.Button(mission, text="Nav Step -> House", command=self.on_navigate_step_to_house).grid(row=1, column=1, padx=6, pady=6, sticky="ew")
        tk.Button(mission, text="Mark Target Explored", command=self.on_mark_explored).grid(row=2, column=0, padx=6, pady=6, sticky="ew")
        tk.Button(mission, text="Mark Current Explored", command=self.on_mark_current_explored).grid(row=2, column=1, padx=6, pady=6, sticky="ew")
        tk.Button(mission, text="Mark Person Found", command=self.on_mark_person_found).grid(row=3, column=0, columnspan=2, padx=6, pady=6, sticky="ew")

        refresh = tk.LabelFrame(right, text="Refresh"); refresh.grid(row=1, column=0, sticky="ew")
        tk.Button(refresh, text="Refresh State", command=self.refresh_state_once).grid(row=0, column=0, padx=6, pady=6)
        tk.Checkbutton(refresh, text="Auto State", variable=self.auto_state_var).grid(row=0, column=1, padx=6, pady=6)
        tk.Label(refresh, text=f"State interval: {self.args.state_interval_ms} ms").grid(row=0, column=2, padx=6, pady=6)

    def safe(self, func, *args, label: str = "Request", **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.warning("%s failed: %s", label, exc)
            self.status_var.set(f"{label} failed: {exc}")
            return None

    def pause(self, seconds: float) -> None:
        self.background_pause_until = max(self.background_pause_until, time.time() + seconds)

    def bg_ok(self) -> bool:
        return (not self.manual_request_inflight) and (not self.move_request_inflight) and time.time() >= self.background_pause_until

    def _read_local_houses_config(self) -> Dict[str, Any]:
        try:
            with open(self.houses_config_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            return raw if isinstance(raw, dict) else {}
        except Exception as exc:
            logger.warning("Failed to read local houses_config.json: %s", exc)
            return {}

    def _get_local_overhead_config(self) -> Dict[str, Any]:
        raw = self._read_local_houses_config()
        overhead = raw.get("overhead_map", {})
        return overhead if isinstance(overhead, dict) else {}

    def _resolve_map_image_path(self, path: str) -> str:
        candidate = str(path or "").strip()
        if not candidate:
            candidate = self.default_map_path
        if not os.path.isabs(candidate):
            candidate = os.path.abspath(os.path.join(self.eval_dir, candidate))
        return candidate

    def _ensure_default_map_loaded(self, *, force: bool = False) -> None:
        overhead_cfg = self._get_local_overhead_config()
        image_path = self._resolve_map_image_path(str(overhead_cfg.get("image_path", "") or ""))
        if not force and self.loaded_map_image is not None and self.loaded_map_image_path == image_path:
            return
        image = cv2.imread(image_path)
        if image is not None:
            self.loaded_map_image = image
            self.loaded_map_image_path = image_path
            self.map_status_var.set(f"Map: loaded {os.path.basename(image_path)}")

    def _world_to_image_point(self, world_x: float, world_y: float, calibration: Dict[str, Any]) -> Optional[tuple[float, float]]:
        affine = calibration.get("affine_world_to_image") if isinstance(calibration, dict) else None
        if not isinstance(affine, list) or len(affine) != 2:
            return None
        try:
            matrix = np.asarray(affine, dtype=np.float32)
            image_x = float(matrix[0, 0]) * float(world_x) + float(matrix[0, 1]) * float(world_y) + float(matrix[0, 2])
            image_y = float(matrix[1, 0]) * float(world_x) + float(matrix[1, 1]) * float(world_y) + float(matrix[1, 2])
            return image_x, image_y
        except Exception:
            return None

    def _restore_saved_points_from_registry(self, reg: Dict[str, Any]) -> None:
        if self.calibration_anchors:
            return
        overhead_cfg = reg.get("overhead_map", {}) if isinstance(reg.get("overhead_map", {}), dict) else {}
        calibration = overhead_cfg.get("calibration", {}) if isinstance(overhead_cfg.get("calibration", {}), dict) else {}
        saved_anchors = calibration.get("anchors", []) if isinstance(calibration.get("anchors", []), list) else []
        restored: List[Dict[str, float]] = []
        for anchor in saved_anchors[:5]:
            try:
                restored.append(
                    {
                        "world_x": float(anchor["world_x"]),
                        "world_y": float(anchor["world_y"]),
                        "image_x": float(anchor["image_x"]),
                        "image_y": float(anchor["image_y"]),
                    }
                )
            except Exception:
                continue
        if restored:
            self.calibration_anchors = restored

    def apply_state(self, state: Dict[str, Any]) -> None:
        pose, depth = state.get("pose", {}), state.get("depth", {})
        mission = state.get("house_mission", {}) if isinstance(state.get("house_mission"), dict) else {}
        self.pose_var.set(f"Pose x={float(pose.get('x',0)):.1f} y={float(pose.get('y',0)):.1f} z={float(pose.get('z',0)):.1f} yaw={float(pose.get('task_yaw',0)):.1f} action={state.get('last_action','idle')} task={state.get('task_label','')}")
        self.depth_var.set(f"Depth frame={depth.get('frame_id','')} min={float(depth.get('min_depth',0)):.1f} max={float(depth.get('max_depth',0)):.1f} front_min={float(depth.get('front_min_depth',0)):.1f}")
        self.movement_enabled_state = bool(state.get("movement_enabled", False))
        self.control_var.set(f"Movement enabled={1 if self.movement_enabled_state else 0} origin={state.get('last_action_origin','n/a')}")
        self.capture_var.set(f"Capture last={(state.get('last_capture') or {}).get('capture_id','none')}")
        self.mission_var.set(f"Mission: current={mission.get('current_house_name') or 'none'} target={mission.get('target_house_name') or 'none'}")
        self.current_house_var.set(f"Current house: {mission.get('current_house_name') or 'none'} [{mission.get('current_house_status') or '-'}]")
        self.target_house_var.set(f"Target: {mission.get('target_house_name') or 'none'} [{mission.get('target_house_status') or '-'}]  next={mission.get('nearest_unsearched_house_name') or 'none'}")
        dist = mission.get("distance_to_target_cm")
        self.target_dist_var.set("Target distance: n/a" if dist is None else f"Target distance: {float(dist):.1f} cm")
        if self.movement_toggle_button is not None:
            self.movement_toggle_button.configure(text="Disable Basic Movement" if self.movement_enabled_state else "Enable Basic Movement")

    def refresh_state_once(self) -> None:
        if self.state_refresh_inflight: return
        def worker():
            self.state_refresh_inflight = True
            try:
                state = self.safe(self.client.get_json, "/state", label="Refresh State")
                if isinstance(state, dict): self.root.after(0, lambda: self.apply_state(state))
            finally:
                self.state_refresh_inflight = False
        threading.Thread(target=worker, daemon=True).start()

    def schedule_state_refresh(self) -> None:
        if self.auto_state_var.get() and self.bg_ok(): self.refresh_state_once()
        self.root.after(self.args.state_interval_ms, self.schedule_state_refresh)

    def _sync_preview_pair(self) -> None:
        self.safe(self.client.post_json, "/refresh", {}, label="Sync Observation")
        rgb = self.safe(self.client.get_image, "/frame", label="Refresh RGB")
        depth = self.safe(self.client.get_image, "/depth_frame", label="Refresh Depth")
        if isinstance(rgb, np.ndarray):
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)))
            self.root.after(0, lambda p=photo: self.apply_preview_photo(p, rgb=True))
        if isinstance(depth, np.ndarray):
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)))
            self.root.after(0, lambda p=photo: self.apply_preview_photo(p, rgb=False))

    def refresh_preview_window(self) -> None:
        if not self.preview_window or not self.preview_label or self.preview_refresh_inflight: return
        def worker():
            self.preview_refresh_inflight = True
            try: self._sync_preview_pair()
            finally: self.preview_refresh_inflight = False
        threading.Thread(target=worker, daemon=True).start()

    def refresh_depth_window(self) -> None:
        if not self.depth_window or not self.depth_label or self.depth_refresh_inflight: return
        def worker():
            self.depth_refresh_inflight = True
            try: self._sync_preview_pair()
            finally: self.depth_refresh_inflight = False
        threading.Thread(target=worker, daemon=True).start()

    def schedule_preview_refresh(self) -> None:
        if self.auto_rgb_var.get() and self.preview_window and self.bg_ok() and not self.preview_refresh_inflight: self.refresh_preview_window()
        self.root.after(self.args.preview_interval_ms, self.schedule_preview_refresh)

    def schedule_depth_refresh(self) -> None:
        if self.auto_depth_var.get() and self.depth_window and self.bg_ok() and not self.depth_refresh_inflight: self.refresh_depth_window()
        self.root.after(self.args.depth_interval_ms, self.schedule_depth_refresh)

    def refresh_map_async(self, *, with_background: bool = False) -> None:
        if self.map_refresh_inflight: return
        def worker():
            self.map_refresh_inflight = True
            try:
                state = self.safe(self.client.get_json, "/state", label="Map state")
                registry = self.safe(self.client.get_json, "/house_registry", label="Map registry")
                if isinstance(state, dict) and isinstance(registry, dict): self.root.after(0, lambda: self.apply_map_data(state, registry))
                if with_background and self.map_widget is not None: self._refresh_map_background_worker()
            finally:
                self.map_refresh_inflight = False
        threading.Thread(target=worker, daemon=True).start()

    def _refresh_map_background_worker(self) -> None:
        if self.map_background_refresh_inflight or not self.map_widget: return
        self.map_background_refresh_inflight = True
        try:
            self.safe(self.client.post_json, "/refresh_overhead_map", {}, label="Refresh overhead map")
            image = self.safe(self.client.get_image, "/overhead_map_frame", label="Load overhead map")
            if isinstance(image, np.ndarray):
                self.root.after(0, lambda img=image: self.map_widget.set_background_image(img))
                self.root.after(0, lambda: self.map_status_var.set("Map: background refreshed"))
        finally:
            self.map_background_refresh_inflight = False

    def refresh_map_background_async(self) -> None:
        if not self.map_widget:
            self.status_var.set("Open Setting Map first.")
            return
        threading.Thread(target=self._refresh_map_background_worker, daemon=True).start()

    def schedule_map_refresh(self) -> None:
        has_setting = self.map_widget and self.map_window and self.map_window.winfo_exists()
        has_open = self.open_map_widget and self.open_map_window and self.open_map_window.winfo_exists()
        if has_setting or has_open:
            self.refresh_map_async()
        self.root.after(1500, self.schedule_map_refresh)

    def apply_map_data(self, state: Dict[str, Any], registry: Dict[str, Any]) -> None:
        reg = registry.get("registry", {})
        summary = reg.get("status_summary", {})
        mission = state.get("house_mission", {}) if isinstance(state.get("house_mission"), dict) else {}
        overhead = state.get("overhead_map", {}) if isinstance(state.get("overhead_map"), dict) else {}
        overhead_cfg = reg.get("overhead_map", {}) if isinstance(reg.get("overhead_map", {}), dict) else {}
        calibration = reg.get("overhead_map", {}).get("calibration", {}) if isinstance(reg.get("overhead_map", {}), dict) else {}
        self.mission_var.set(f"Mission: unsearched={summary.get('UNSEARCHED',0)} in_progress={summary.get('IN_PROGRESS',0)} explored={summary.get('EXPLORED',0)} found={summary.get('PERSON_FOUND',0)}")
        self.map_status_var.set(
            f"Map: current={mission.get('current_house_name') or 'none'} "
            f"target={mission.get('target_house_name') or 'none'} "
            f"bg={overhead.get('refresh_time','-')}"
        )
        self._restore_saved_points_from_registry(reg)
        if self.pending_image_anchor is not None:
            self.calib_var.set(
                f"Calibration: clicked image=({self.pending_image_anchor['image_x']:.1f}, {self.pending_image_anchor['image_y']:.1f}) "
                f"anchors={len(self.calibration_anchors)}/5"
            )
        elif calibration:
            self.calib_var.set(
                f"Calibration: anchors={len(calibration.get('anchors', []))} "
                f"rmse={float(calibration.get('rmse_px', 0.0)):.2f}px"
            )
        elif self.pending_anchor_world is not None:
            self.calib_var.set(
                f"Calibration: pending world=({self.pending_anchor_world['world_x']:.1f}, "
                f"{self.pending_anchor_world['world_y']:.1f}) click map point next"
            )
        else:
            self.calib_var.set(f"Calibration: unsolved local_anchors={len(self.calibration_anchors)}")
        self._ensure_default_map_loaded()
        pose = state.get("pose", {})
        pose_x = float(pose.get("x", 0))
        pose_y = float(pose.get("y", 0))
        pose_yaw = float(pose.get("task_yaw", 0))
        image_size = None
        if self.loaded_map_image is not None:
            image_size = (int(self.loaded_map_image.shape[1]), int(self.loaded_map_image.shape[0]))
        elif overhead.get("image_width") and overhead.get("image_height"):
            image_size = (int(overhead.get("image_width", 0)), int(overhead.get("image_height", 0)))
        affine = calibration.get("affine_world_to_image") if isinstance(calibration, dict) else None
        anchors = calibration.get("anchors", []) if isinstance(calibration, dict) else []
        if not anchors and self.calibration_anchors:
            anchors = [
                {"label": f"P{i+1}", **anchor}
                for i, anchor in enumerate(self.calibration_anchors)
            ]
        if self.pending_image_anchor is not None:
            anchors = list(anchors)
            anchors.append({
                "label": f"P{len(self.calibration_anchors) + 1}?",
                "image_x": float(self.pending_image_anchor["image_x"]),
                "image_y": float(self.pending_image_anchor["image_y"]),
            })
        image_point = self._world_to_image_point(pose_x, pose_y, calibration)
        if image_point is None:
            self.map_pose_var.set(f"Map pose: world=({pose_x:.1f}, {pose_y:.1f}) yaw={pose_yaw:.1f} image=n/a")
        else:
            self.map_pose_var.set(
                f"Map pose: world=({pose_x:.1f}, {pose_y:.1f}) image=({image_point[0]:.1f}, {image_point[1]:.1f}) yaw={pose_yaw:.1f}"
            )

        for widget in (self.map_widget, self.open_map_widget):
            if widget is None:
                continue
            if self.loaded_map_image is not None:
                widget.set_background_image(self.loaded_map_image)
            widget.set_calibration(affine, image_size, anchors)
            widget.update_uav(pose_x, pose_y, pose_yaw)
            widget.update_houses([])
            widget.set_route_target(None)

        if not self.map_widget or not self.show_houses_var.get():
            return
        current_id, target_id = mission.get("current_house_id", ""), reg.get("current_target_id", "")
        houses = []
        target_xy = None
        for h in reg.get("houses", []):
            name = str(h.get("name", ""))
            if h.get("id", "") == current_id: name = f"{name} (UAV)"
            cx = float(h.get("center_x", 0))
            cy = float(h.get("center_y", 0))
            hid = h.get("id", "")
            is_target = hid == target_id
            if is_target:
                target_xy = (cx, cy)
            houses.append({
                "id": hid,
                "name": name,
                "center_x": cx,
                "center_y": cy,
                "radius_cm": float(h.get("radius_cm", 600)),
                "status": h.get("status", "UNSEARCHED"),
                "is_target": is_target,
                "is_current": hid == current_id,
            })
        self.map_widget.update_houses(houses)
        self.map_widget.set_route_target(target_xy if self.show_route_var.get() else None)

    def call_async(self, desc: str, fn) -> None:
        if self.manual_request_inflight:
            self.status_var.set(f"{desc} skipped while another request is running.")
            return
        def worker():
            self.manual_request_inflight = True; self.pause(1.0); self.root.after(0, lambda: self.status_var.set(f"{desc}..."))
            try: fn()
            finally: self.manual_request_inflight = False; self.pause(0.4)
        threading.Thread(target=worker, daemon=True).start()

    def on_set_task(self) -> None: self.call_async("Setting task", lambda: self._apply_response(self.safe(self.client.post_json, "/task", {"task_label": self.task_label_var.get().strip()}, label="Set Task")))
    def on_capture(self) -> None: self.call_async("Capturing", lambda: self._apply_response((self.safe(self.client.post_json, "/capture", {"label": self.capture_label_var.get().strip()}, label="Capture") or {}).get("state")))
    def on_set_pose(self) -> None:
        try: payload = json.loads(self.pose_text.get("1.0", "end").strip()) if self.pose_text else {}
        except json.JSONDecodeError as exc: self.status_var.set(f"Invalid pose JSON: {exc}"); return
        self.call_async("Setting pose", lambda: self._apply_response(self.safe(self.client.post_json, "/set_pose", payload, label="Set Pose")))
    def on_toggle_movement(self) -> None: self.call_async("Toggling movement", lambda: self._apply_response(self.safe(self.client.post_json, "/basic_movement_enable", {"enabled": not self.movement_enabled_state}, label="Toggle Movement")))

    def _apply_response(self, response: Any) -> None:
        if isinstance(response, dict):
            self.root.after(0, lambda: self.apply_state(response))
            self.refresh_map_async()

    def _execute_move(self, symbol: str, *, from_sequence: bool = False) -> bool:
        payload = MOVE_COMMANDS.get(symbol.lower())
        if payload is None: return False
        self.move_request_inflight = True; self.pause(1.0)
        try:
            resp = self.safe(self.client.post_json, "/move_relative", payload, label=f"Move {symbol}")
            if isinstance(resp, dict):
                self.root.after(0, lambda: self.apply_state(resp)); self.refresh_map_async()
                if from_sequence: self.root.after(0, lambda s=symbol: self.status_var.set(f"Sequence sent: {s}"))
                return True
            return False
        finally:
            self.move_request_inflight = False; self.pause(0.4)

    def send_move_symbol(self, symbol: str) -> None:
        if self.move_request_inflight: self.status_var.set(f"Move {symbol} ignored while another move is in flight."); return
        threading.Thread(target=lambda: self._execute_move(symbol), daemon=True).start()

    def on_execute_sequence(self) -> None:
        symbols = [s for s in self.sequence_var.get().strip().lower() if s in MOVE_COMMANDS]
        if not symbols: self.status_var.set("No valid sequence symbols."); return
        try: delay_s = max(0.05, float(self.sequence_delay_var.get().strip()) / 1000.0)
        except ValueError: self.status_var.set("Invalid sequence delay."); return
        if self.sequence_thread and self.sequence_thread.is_alive(): self.status_var.set("Sequence already running."); return
        def worker():
            self.sequence_stop_event.clear(); total = len(symbols)
            for i, s in enumerate(symbols, start=1):
                if self.sequence_stop_event.is_set(): self.root.after(0, lambda i=i, t=total: self.status_var.set(f"Sequence stopped at step {i-1}/{t}.")); return
                while self.move_request_inflight and not self.sequence_stop_event.is_set(): time.sleep(0.02)
                self.root.after(0, lambda i=i, t=total, s=s: self.status_var.set(f"Sequence step {i}/{t}: {s}"))
                if not self._execute_move(s, from_sequence=True): self.root.after(0, lambda i=i, t=total: self.status_var.set(f"Sequence failed at step {i}/{t}.")); return
                time.sleep(delay_s)
            self.root.after(0, lambda: self.status_var.set(f"Sequence completed: {total}/{total} steps"))
        self.sequence_thread = threading.Thread(target=worker, daemon=True); self.sequence_thread.start()

    def on_stop_sequence(self) -> None: self.sequence_stop_event.set(); self.status_var.set("Stopping sequence...")

    def toggle_preview_window(self) -> None:
        if self.preview_window and self.preview_window.winfo_exists():
            self.preview_window.destroy(); self.preview_window = None; self.preview_label = None; self.preview_photo = None; return
        self.preview_window = tk.Toplevel(self.root); self.preview_window.title("UAV RGB Preview"); self.preview_label = tk.Label(self.preview_window); self.preview_label.pack(fill="both", expand=True); self.refresh_preview_window()

    def toggle_depth_window(self) -> None:
        if self.depth_window and self.depth_window.winfo_exists():
            self.depth_window.destroy(); self.depth_window = None; self.depth_label = None; self.depth_photo = None; return
        self.depth_window = tk.Toplevel(self.root); self.depth_window.title("UAV Depth Preview"); self.depth_label = tk.Label(self.depth_window); self.depth_label.pack(fill="both", expand=True); self.refresh_depth_window()

    def apply_preview_photo(self, photo: ImageTk.PhotoImage, *, rgb: bool) -> None:
        if rgb:
            self.preview_photo = photo
            if self.preview_label is not None: self.preview_label.configure(image=photo)
        else:
            self.depth_photo = photo
            if self.depth_label is not None: self.depth_label.configure(image=photo)

    def toggle_open_map_window(self) -> None:
        if self.open_map_window and self.open_map_window.winfo_exists():
            self.open_map_window.destroy()
            self.open_map_window = None
            self.open_map_widget = None
            self.map_status_var.set("Map: closed")
            return
        self._ensure_default_map_loaded(force=True)
        reg = self.safe(self.client.get_json, "/house_registry", label="Load map bounds") or {}
        wb = reg.get("registry", {}).get("world_bounds", {})
        bounds = (
            float(wb.get("min_x", 1000.0)),
            float(wb.get("min_y", -500.0)),
            float(wb.get("max_x", 5000.0)),
            float(wb.get("max_y", 3000.0)),
        )
        self.open_map_window = tk.Toplevel(self.root)
        self.open_map_window.title("Overhead Map - UAV Pose")
        self.open_map_window.resizable(False, False)
        toolbar = tk.Frame(self.open_map_window)
        toolbar.pack(fill="x", padx=8, pady=(8, 0))
        tk.Label(toolbar, textvariable=self.map_pose_var, anchor="w").pack(side="left", padx=(0, 8))
        tk.Label(toolbar, textvariable=self.map_status_var, anchor="w").pack(side="left")
        self.open_map_widget = OverheadMapWidget(self.open_map_window, world_bounds=bounds, canvas_w=760, canvas_h=560)
        self.open_map_widget.canvas.pack(padx=8, pady=8)
        self.refresh_map_async()

    def toggle_map_window(self) -> None:
        if self.map_window and self.map_window.winfo_exists():
            self.map_window.destroy(); self.map_window = None; self.map_widget = None; self.map_status_var.set("Map: closed"); return
        self._ensure_default_map_loaded(force=True)
        self.map_window = tk.Toplevel(self.root); self.map_window.title("Setting Map - Alignment"); self.map_window.resizable(False, False)
        toolbar = tk.Frame(self.map_window); toolbar.pack(fill="x", padx=8, pady=(8, 0))
        tk.Button(toolbar, text="Capture Fixed Map", command=self.refresh_map_background_async).pack(side="left")
        tk.Button(toolbar, text="Load Local Image", command=self.on_load_map_image).pack(side="left", padx=(6, 0))
        tk.Button(toolbar, text="Add UAV Anchor", command=self.on_add_uav_anchor).pack(side="left", padx=(6, 0))
        tk.Button(toolbar, text="Add Point", command=self.on_add_anchor_from_inputs).pack(side="left", padx=(6, 0))
        tk.Button(toolbar, text="Save Alignment", command=self.on_solve_calibration).pack(side="left", padx=(6, 0))
        tk.Button(toolbar, text="Clear Calib", command=self.on_clear_calibration).pack(side="left", padx=(6, 0))
        tk.Label(toolbar, textvariable=self.map_status_var).pack(side="left", padx=10)
        calib = tk.Frame(self.map_window)
        calib.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(calib, text="World X").pack(side="left")
        tk.Entry(calib, textvariable=self.anchor_world_x_var, width=10).pack(side="left", padx=(4, 8))
        tk.Label(calib, text="World Y").pack(side="left")
        tk.Entry(calib, textvariable=self.anchor_world_y_var, width=10).pack(side="left", padx=(4, 8))
        tk.Label(calib, textvariable=self.calib_var, anchor="w").pack(side="left", padx=8)
        reg = self.safe(self.client.get_json, "/house_registry", label="Load map bounds") or {}
        self._restore_saved_points_from_registry(reg.get("registry", {}))
        wb = reg.get("registry", {}).get("world_bounds", {})
        bounds = (float(wb.get("min_x", 1000.0)), float(wb.get("min_y", -500.0)), float(wb.get("max_x", 5000.0)), float(wb.get("max_y", 3000.0)))
        self.map_widget = OverheadMapWidget(self.map_window, world_bounds=bounds, canvas_w=760, canvas_h=560)
        self.map_widget.canvas.pack(padx=8, pady=8)
        self.map_widget.set_map_click_callback(self.on_map_click)
        self.refresh_map_async(with_background=False)

    def on_auto_select_house(self) -> None: threading.Thread(target=lambda: (self.safe(self.client.post_json, "/select_nearest_unsearched_house", {}, label="Auto-select nearest"), self.refresh_map_async()), daemon=True).start()
    def on_navigate_step_to_house(self) -> None: self.call_async("Nav step to house", lambda: self._apply_response(self.safe(self.client.post_json, "/navigate_step_to_house", {}, label="Nav step to house")))

    def on_mark_explored(self) -> None:
        def worker():
            reg = self.safe(self.client.get_json, "/house_registry", label="Load target house") or {}
            target_id = reg.get("registry", {}).get("current_target_id", "")
            if not target_id: self.root.after(0, lambda: self.status_var.set("No target house set.")); return
            self.safe(self.client.post_json, "/mark_house_explored", {"house_id": target_id, "person_found": False}, label=f"Mark {target_id} explored"); self.refresh_map_async()
        threading.Thread(target=worker, daemon=True).start()

    def on_mark_current_explored(self) -> None: threading.Thread(target=lambda: (self.safe(self.client.post_json, "/mark_current_house_explored", {"person_found": False}, label="Mark current explored"), self.refresh_map_async()), daemon=True).start()

    def on_mark_person_found(self) -> None:
        def worker():
            reg = self.safe(self.client.get_json, "/house_registry", label="Load target house") or {}
            state = self.safe(self.client.get_json, "/state", label="Load state") or {}
            target_id = reg.get("registry", {}).get("current_target_id", "")
            if not target_id: self.root.after(0, lambda: self.status_var.set("No target house set.")); return
            pose = state.get("pose", {})
            loc = {"x": pose.get("x", 0.0), "y": pose.get("y", 0.0), "z": pose.get("z", 0.0)}
            self.safe(self.client.post_json, "/mark_house_explored", {"house_id": target_id, "person_found": True, "person_location": loc, "notes": "Confirmed by operator"}, label=f"Mark {target_id} person found")
            self.refresh_map_async()
        threading.Thread(target=worker, daemon=True).start()

    def on_add_uav_anchor(self) -> None:
        def worker():
            state = self.safe(self.client.get_json, "/state", label="Load pose for anchor") or {}
            pose = state.get("pose", {}) if isinstance(state, dict) else {}
            world_x = float(pose.get("x", 0.0))
            world_y = float(pose.get("y", 0.0))
            self.pending_anchor_world = {"world_x": world_x, "world_y": world_y}
            self.root.after(0, lambda: self.anchor_world_x_var.set(f"{world_x:.1f}"))
            self.root.after(0, lambda: self.anchor_world_y_var.set(f"{world_y:.1f}"))
            self.root.after(0, lambda: self.calib_var.set(f"Calibration: UAV world loaded ({world_x:.1f}, {world_y:.1f}); click image point then Add Point"))
        threading.Thread(target=worker, daemon=True).start()

    def on_map_click(self, image_x: float, image_y: float) -> None:
        if len(self.calibration_anchors) >= 5:
            self.root.after(0, lambda: self.calib_var.set("Calibration: already have 5 points; save or clear first."))
            return
        self.pending_image_anchor = {"image_x": float(image_x), "image_y": float(image_y)}
        if self.pending_anchor_world is not None:
            self.root.after(0, lambda: self.anchor_world_x_var.set(f"{self.pending_anchor_world['world_x']:.1f}"))
            self.root.after(0, lambda: self.anchor_world_y_var.set(f"{self.pending_anchor_world['world_y']:.1f}"))
            self.pending_anchor_world = None
        self.root.after(0, lambda: self.calib_var.set(
            f"Calibration: picked image P{len(self.calibration_anchors) + 1}=({image_x:.1f}, {image_y:.1f}); "
            "enter world X/Y then click Add Point"
        ))
        self.refresh_map_async()

    def on_load_map_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select overhead map image",
            filetypes=[
                ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
                ("All Files", "*.*"),
            ],
        )
        if not path:
            return
        image = cv2.imread(path)
        if image is None:
            self.status_var.set(f"Failed to load image: {path}")
            return
        self.loaded_map_image = image
        self.loaded_map_image_path = os.path.abspath(path)
        self.pending_image_anchor = None
        self.pending_anchor_world = None
        if self.map_widget is not None:
            self.map_widget.set_background_image(image)
            self.map_widget.set_calibration(None, (int(image.shape[1]), int(image.shape[0])), [
                {"label": f"P{i+1}", **anchor} for i, anchor in enumerate(self.calibration_anchors)
            ])
        self.map_status_var.set(f"Map: loaded {os.path.basename(path)}")
        self.calib_var.set(
            f"Calibration: image loaded {os.path.basename(path)}; click image, enter world X/Y, Add Point ({len(self.calibration_anchors)}/5)"
        )

    def on_add_anchor_from_inputs(self) -> None:
        if self.pending_image_anchor is None:
            self.calib_var.set("Calibration: click a point on the image first.")
            return
        if len(self.calibration_anchors) >= 5:
            self.calib_var.set("Calibration: already have 5 points; save or clear first.")
            return
        try:
            world_x = float(self.anchor_world_x_var.get().strip())
            world_y = float(self.anchor_world_y_var.get().strip())
        except ValueError:
            self.calib_var.set("Calibration: invalid world X/Y.")
            return
        anchor = {
            "world_x": world_x,
            "world_y": world_y,
            "image_x": float(self.pending_image_anchor["image_x"]),
            "image_y": float(self.pending_image_anchor["image_y"]),
        }
        self.calibration_anchors.append(anchor)
        idx = len(self.calibration_anchors)
        self.pending_image_anchor = None
        self.pending_anchor_world = None
        self.calib_var.set(
            f"Calibration: added P{idx} world=({world_x:.1f}, {world_y:.1f}) image=({anchor['image_x']:.1f}, {anchor['image_y']:.1f})"
        )
        self.refresh_map_async()

    def on_solve_calibration(self) -> None:
        if len(self.calibration_anchors) < 3:
            self.calib_var.set("Calibration: need at least 3 anchors.")
            return
        def worker():
            image_width = 0
            image_height = 0
            image_path = self.loaded_map_image_path
            if self.loaded_map_image is not None:
                image_width = int(self.loaded_map_image.shape[1])
                image_height = int(self.loaded_map_image.shape[0])
            if image_width <= 0 or image_height <= 0:
                state = self.safe(self.client.get_json, "/state", label="Load map info for calibration") or {}
                overhead = state.get("overhead_map", {}) if isinstance(state.get("overhead_map"), dict) else {}
                image_width = int(overhead.get("image_width", 0))
                image_height = int(overhead.get("image_height", 0))
                if not image_path:
                    image_path = str(overhead.get("saved_image_path", "") or overhead.get("image_path", "") or "")
            if image_width <= 0 or image_height <= 0:
                self.root.after(0, lambda: self.calib_var.set("Calibration: load a map image or capture fixed map first."))
                return
            resp = self.safe(self.client.post_json, "/set_overhead_calibration", {
                "anchors": self.calibration_anchors,
                "image_width": image_width,
                "image_height": image_height,
                "image_path": image_path,
            }, label="Solve calibration")
            if isinstance(resp, dict) and resp.get("status") == "ok":
                calibration = resp.get("calibration", {}) if isinstance(resp.get("calibration", {}), dict) else {}
                saved = calibration.get("anchors", []) if isinstance(calibration.get("anchors", []), list) else []
                restored: List[Dict[str, float]] = []
                for anchor in saved[:5]:
                    try:
                        restored.append(
                            {
                                "world_x": float(anchor["world_x"]),
                                "world_y": float(anchor["world_y"]),
                                "image_x": float(anchor["image_x"]),
                                "image_y": float(anchor["image_y"]),
                            }
                        )
                    except Exception:
                        continue
                self.calibration_anchors = restored or list(self.calibration_anchors)
                self.pending_anchor_world = None
                self.pending_image_anchor = None
                self.root.after(0, lambda: self.calib_var.set(resp.get("message", "Calibration solved.")))
                self.refresh_map_async()
        threading.Thread(target=worker, daemon=True).start()

    def on_clear_calibration(self) -> None:
        def worker():
            self.calibration_anchors = []
            self.pending_anchor_world = None
            self.pending_image_anchor = None
            resp = self.safe(self.client.post_json, "/clear_overhead_calibration", {}, label="Clear calibration")
            if isinstance(resp, dict):
                self.root.after(0, lambda: self.calib_var.set(resp.get("message", "Calibration cleared.")))
                self.refresh_map_async()
        threading.Thread(target=worker, daemon=True).start()

    def on_close(self) -> None:
        for window in (self.preview_window, self.depth_window, self.map_window, self.open_map_window):
            try:
                if window is not None: window.destroy()
            except Exception:
                pass
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic UAV control panel")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5020)
    parser.add_argument("--timeout_s", type=float, default=8.0)
    parser.add_argument("--state_interval_ms", type=int, default=1500)
    parser.add_argument("--preview_interval_ms", type=int, default=1500)
    parser.add_argument("--depth_interval_ms", type=int, default=1800)
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    Panel(args).run()


if __name__ == "__main__":
    main()
