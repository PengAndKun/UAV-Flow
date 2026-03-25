"""
Basic control panel for `uav_control_server_basic.py`.
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import tkinter as tk
from typing import Any, Dict, Optional
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


class BasicControlClient:
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
            raise RuntimeError(f"Failed to decode image from path={path}")
        return image


class BasicUAVControlPanel:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.client = BasicControlClient(f"http://{args.host}:{args.port}", args.timeout_s)
        self.root = tk.Tk()
        self.root.title("UAV Basic Controller")
        self.root.geometry("900x720")
        self.root.minsize(760, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.status_var = tk.StringVar(value="Ready")
        self.pose_var = tk.StringVar(value="Pose: waiting...")
        self.depth_var = tk.StringVar(value="Depth: waiting...")
        self.control_var = tk.StringVar(value="Movement: waiting...")
        self.capture_var = tk.StringVar(value="Capture: waiting...")

        self.task_label_var = tk.StringVar(value="")
        self.capture_label_var = tk.StringVar(value="")
        self.sequence_var = tk.StringVar(value="")
        self.sequence_delay_var = tk.StringVar(value="220")
        self.auto_state_var = tk.BooleanVar(value=True)
        self.auto_preview_var = tk.BooleanVar(value=False)
        self.auto_depth_var = tk.BooleanVar(value=False)

        self.state_refresh_inflight = False
        self.manual_request_inflight = False
        self.move_request_inflight = False
        self.background_pause_until = 0.0
        self.preview_refresh_inflight = False
        self.depth_refresh_inflight = False

        self.preview_window: Optional[tk.Toplevel] = None
        self.preview_label: Optional[tk.Label] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.depth_window: Optional[tk.Toplevel] = None
        self.depth_label: Optional[tk.Label] = None
        self.depth_photo: Optional[ImageTk.PhotoImage] = None
        self.pose_text: Optional[tk.Text] = None

        self.sequence_thread: Optional[threading.Thread] = None
        self.sequence_stop_event = threading.Event()
        self.movement_enabled_state = False

        # --- Mission map state ---
        self.map_window: Optional[tk.Toplevel] = None
        self.map_widget: Optional[OverheadMapWidget] = None
        self.mission_status_var = tk.StringVar(value="Mission: idle")
        self.target_house_var = tk.StringVar(value="Target: none")

        self.build_ui()
        self.root.after(200, self.schedule_state_refresh)
        self.root.after(350, self.schedule_preview_refresh)
        self.root.after(450, self.schedule_depth_refresh)
        self.root.after(1200, self.schedule_map_refresh)

    def pause_background_refresh(self, seconds: float) -> None:
        self.background_pause_until = max(self.background_pause_until, time.time() + max(0.0, seconds))

    def background_refresh_allowed(self) -> bool:
        return (not self.manual_request_inflight) and (not self.move_request_inflight) and time.time() >= self.background_pause_until

    def set_status(self, message: str) -> None:
        self.status_var.set(message)

    def safe_request(self, func, *args, label: str = "Request", **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            logger.warning("%s failed: %s", label, exc)
            self.set_status(f"{label} failed: {exc}")
            return None

    def run_manual_async_request(self, description: str, fn) -> None:
        if self.manual_request_inflight:
            self.set_status(f"{description} skipped while another request is running.")
            return

        def worker() -> None:
            self.manual_request_inflight = True
            self.pause_background_refresh(1.0)
            self.root.after(0, lambda: self.set_status(f"{description}..."))
            try:
                fn()
            finally:
                self.manual_request_inflight = False
                self.pause_background_refresh(0.4)

        threading.Thread(target=worker, daemon=True).start()

    def apply_state(self, state: Dict[str, Any]) -> None:
        pose = state.get("pose", {}) if isinstance(state.get("pose"), dict) else {}
        depth = state.get("depth", {}) if isinstance(state.get("depth"), dict) else {}
        self.pose_var.set(
            "Pose x={x:.1f} y={y:.1f} z={z:.1f} yaw={yaw:.1f} action={action} task={task}".format(
                x=float(pose.get("x", 0.0)),
                y=float(pose.get("y", 0.0)),
                z=float(pose.get("z", 0.0)),
                yaw=float(pose.get("task_yaw", 0.0)),
                action=str(state.get("last_action", "idle")),
                task=str(state.get("task_label", "")),
            )
        )
        self.depth_var.set(
            "Depth frame={frame} min={min_d:.1f} max={max_d:.1f} front_min={front:.1f}".format(
                frame=str(depth.get("frame_id", "")),
                min_d=float(depth.get("min_depth", 0.0)),
                max_d=float(depth.get("max_depth", 0.0)),
                front=float(depth.get("front_min_depth", 0.0)),
            )
        )
        self.movement_enabled_state = bool(state.get("movement_enabled", False))
        self.control_var.set(f"Movement enabled={1 if self.movement_enabled_state else 0} origin={state.get('last_action_origin', 'n/a')}")
        self.capture_var.set(f"Capture last={(state.get('last_capture') or {}).get('capture_id', 'none')}")
        self.movement_toggle_button.configure(text="Disable Basic Movement" if self.movement_enabled_state else "Enable Basic Movement")

    def refresh_state_once(self) -> None:
        if self.state_refresh_inflight:
            return

        def worker() -> None:
            self.state_refresh_inflight = True
            try:
                state = self.safe_request(self.client.get_json, "/state", label="Refresh State")
                if isinstance(state, dict):
                    self.root.after(0, lambda: self.apply_state(state))
            finally:
                self.state_refresh_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def schedule_state_refresh(self) -> None:
        if self.auto_state_var.get() and self.background_refresh_allowed():
            self.refresh_state_once()
        self.root.after(self.args.state_interval_ms, self.schedule_state_refresh)

    def schedule_preview_refresh(self) -> None:
        if (
            self.auto_preview_var.get()
            and self.preview_window is not None
            and self.preview_window.winfo_exists()
            and self.background_refresh_allowed()
            and not self.preview_refresh_inflight
        ):
            self.refresh_preview_window()
        self.root.after(self.args.preview_interval_ms, self.schedule_preview_refresh)

    def schedule_depth_refresh(self) -> None:
        if (
            self.auto_depth_var.get()
            and self.depth_window is not None
            and self.depth_window.winfo_exists()
            and self.background_refresh_allowed()
            and not self.depth_refresh_inflight
        ):
            self.refresh_depth_window()
        self.root.after(self.args.depth_interval_ms, self.schedule_depth_refresh)

    # ------------------------------------------------------------------
    # Map and mission helpers
    # ------------------------------------------------------------------
    def schedule_map_refresh(self) -> None:
        if self.map_widget is not None and self.map_window is not None and self.map_window.winfo_exists():
            self._refresh_map_async()
        self.root.after(1500, self.schedule_map_refresh)

    def _refresh_map_async(self) -> None:
        def worker() -> None:
            try:
                state = self.client.get_json("/state")
                registry = self.client.get_json("/house_registry")
            except Exception:
                return
            self.root.after(0, lambda: self._apply_map_data(state, registry))
        threading.Thread(target=worker, daemon=True).start()

    def _apply_map_data(self, state: Dict[str, Any], registry: Dict[str, Any]) -> None:
        pose = state.get("pose", {})
        uav_x = float(pose.get("x", 0))
        uav_y = float(pose.get("y", 0))
        uav_yaw = float(pose.get("task_yaw", 0))

        reg = registry.get("registry", {})
        target_id = reg.get("target_house_id", "")
        summary = state.get("house_registry", {})
        counts = summary.get("counts", {})
        self.mission_status_var.set(
            f"Mission: unsearched={counts.get('UNSEARCHED',0)} "
            f"in_progress={counts.get('IN_PROGRESS',0)} "
            f"explored={counts.get('EXPLORED',0)} "
            f"found={counts.get('PERSON_FOUND',0)}"
        )
        self.target_house_var.set(f"Target: {target_id or 'none'}")

        if self.map_widget is None:
            return
        self.map_widget.update_uav(uav_x, uav_y, uav_yaw)
        houses_raw = reg.get("houses", [])
        houses_for_map = []
        for h in houses_raw:
            houses_for_map.append({
                "id": h.get("id", ""),
                "name": h.get("name", ""),
                "center_x": float(h.get("center_x", 0)),
                "center_y": float(h.get("center_y", 0)),
                "radius_cm": float(h.get("radius_cm", 600)),
                "status": h.get("status", "UNSEARCHED"),
                "is_target": h.get("id", "") == target_id,
            })
        self.map_widget.update_houses(houses_for_map)

    def toggle_map_window(self) -> None:
        if self.map_window is not None and self.map_window.winfo_exists():
            self.map_window.destroy()
            self.map_window = None
            self.map_widget = None
            return
        self.map_window = tk.Toplevel(self.root)
        self.map_window.title("Mission Map — Overhead View")
        self.map_window.resizable(False, False)
        # Use world bounds from registry if available; fall back to defaults
        try:
            reg = self.client.get_json("/house_registry")
            wb = reg.get("registry", {}).get("world_bounds", {})
            bounds = (
                float(wb.get("min_x", 1000)),
                float(wb.get("min_y", -500)),
                float(wb.get("max_x", 5000)),
                float(wb.get("max_y", 3000)),
            )
        except Exception:
            bounds = (1000, -500, 5000, 3000)
        self.map_widget = OverheadMapWidget(self.map_window, world_bounds=bounds,
                                            canvas_w=520, canvas_h=400)
        self.map_widget.set_click_callback(self._on_map_house_click)
        self._refresh_map_async()

    def _on_map_house_click(self, house_id: str) -> None:
        def worker() -> None:
            resp = self.safe_request(
                self.client.post_json, "/select_target_house",
                {"house_id": house_id}, label=f"Select house {house_id}",
            )
            if isinstance(resp, dict):
                msg = resp.get("message", "")
                self.root.after(0, lambda: self.set_status(f"[Map] {msg}"))
                self._refresh_map_async()
        threading.Thread(target=worker, daemon=True).start()
        self.set_status(f"Selecting target house: {house_id}...")

    def on_auto_select_house(self) -> None:
        def worker() -> None:
            try:
                state = self.client.get_json("/state")
                reg = self.client.get_json("/house_registry")
            except Exception as exc:
                self.root.after(0, lambda: self.set_status(f"Auto-select failed: {exc}"))
                return
            pose = state.get("pose", {})
            uav_x, uav_y = float(pose.get("x", 0)), float(pose.get("y", 0))
            houses = reg.get("registry", {}).get("houses", [])
            best_id, best_dist = None, float("inf")
            for h in houses:
                if h.get("status", "") not in ("UNSEARCHED",):
                    continue
                dx = float(h.get("center_x", 0)) - uav_x
                dy = float(h.get("center_y", 0)) - uav_y
                d = (dx**2 + dy**2) ** 0.5
                if d < best_dist:
                    best_dist, best_id = d, h["id"]
            if best_id is None:
                self.root.after(0, lambda: self.set_status("No unsearched houses remain."))
                return
            resp = self.safe_request(
                self.client.post_json, "/select_target_house",
                {"house_id": best_id}, label=f"Auto-select {best_id}",
            )
            msg = (resp or {}).get("message", best_id)
            self.root.after(0, lambda: self.set_status(f"Auto-selected: {msg}"))
            self._refresh_map_async()
        threading.Thread(target=worker, daemon=True).start()
        self.set_status("Auto-selecting nearest unsearched house...")

    def on_navigate_step_to_house(self) -> None:
        def worker() -> None:
            resp = self.safe_request(
                self.client.post_json, "/navigate_step_to_house", {},
                label="Nav step to house",
            )
            if isinstance(resp, dict):
                self.root.after(0, lambda: self.apply_state(resp))
        self.run_manual_async_request("Nav step to house", worker)

    def on_mark_explored(self) -> None:
        def worker() -> None:
            try:
                reg = self.client.get_json("/house_registry")
            except Exception:
                return
            target_id = reg.get("registry", {}).get("target_house_id", "")
            if not target_id:
                self.root.after(0, lambda: self.set_status("No target house set."))
                return
            resp = self.safe_request(
                self.client.post_json, "/mark_house_explored",
                {"house_id": target_id, "person_found": False},
                label=f"Mark {target_id} explored",
            )
            msg = (resp or {}).get("message", "done")
            self.root.after(0, lambda: self.set_status(msg))
            self._refresh_map_async()
        threading.Thread(target=worker, daemon=True).start()

    def on_mark_person_found(self) -> None:
        def worker() -> None:
            try:
                reg = self.client.get_json("/house_registry")
                state = self.client.get_json("/state")
            except Exception:
                return
            target_id = reg.get("registry", {}).get("target_house_id", "")
            if not target_id:
                self.root.after(0, lambda: self.set_status("No target house set."))
                return
            pose = state.get("pose", {})
            loc = {"x": pose.get("x", 0), "y": pose.get("y", 0), "z": pose.get("z", 0)}
            resp = self.safe_request(
                self.client.post_json, "/mark_house_explored",
                {"house_id": target_id, "person_found": True, "person_location": loc,
                 "notes": "Confirmed by operator"},
                label=f"Mark {target_id} person found",
            )
            msg = (resp or {}).get("message", "done")
            self.root.after(0, lambda: self.set_status(msg))
            self._refresh_map_async()
        threading.Thread(target=worker, daemon=True).start()

    def on_close(self) -> None:
        for window in (self.preview_window, self.depth_window):
            try:
                if window is not None:
                    window.destroy()
            except Exception:
                pass
        self.root.destroy()

    def build_ui(self) -> None:
        main = tk.Frame(self.root, padx=10, pady=10)
        main.pack(fill="both", expand=True)

        status_frame = tk.LabelFrame(main, text="Runtime Status", padx=8, pady=8)
        status_frame.pack(fill="x", pady=(0, 10))
        for var in (self.status_var, self.pose_var, self.depth_var, self.control_var, self.capture_var):
            tk.Label(status_frame, textvariable=var, anchor="w", justify="left").pack(fill="x", pady=2)

        top = tk.Frame(main)
        top.pack(fill="x", pady=(0, 10))

        task_frame = tk.LabelFrame(top, text="Task And Capture", padx=8, pady=8)
        task_frame.pack(side="left", fill="x", expand=True, padx=(0, 8))
        tk.Label(task_frame, text="Task Label").grid(row=0, column=0, sticky="w")
        tk.Entry(task_frame, textvariable=self.task_label_var, width=44).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        tk.Button(task_frame, text="Set Task", width=14, command=self.on_set_task).grid(row=0, column=2)
        tk.Label(task_frame, text="Capture Label").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tk.Entry(task_frame, textvariable=self.capture_label_var, width=44).grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
        tk.Button(task_frame, text="Capture", width=14, command=self.on_capture).grid(row=1, column=2, pady=(8, 0))
        tk.Label(task_frame, text="Init Pose JSON").grid(row=2, column=0, sticky="nw", pady=(8, 0))
        self.pose_text = tk.Text(task_frame, width=44, height=5, wrap="word")
        self.pose_text.grid(row=2, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
        self.pose_text.insert("1.0", '{\n  "x": 2359.9,\n  "y": 85.3,\n  "z": 225.0,\n  "yaw": -1.7\n}')
        tk.Button(task_frame, text="Set Pose", width=14, command=self.on_set_pose).grid(row=2, column=2, pady=(8, 0), sticky="n")
        task_frame.grid_columnconfigure(1, weight=1)

        move_frame = tk.LabelFrame(top, text="Basic Movement", padx=8, pady=8)
        move_frame.pack(side="right", fill="both")
        self.movement_toggle_button = tk.Button(move_frame, text="Enable Basic Movement", width=22, command=self.on_toggle_movement)
        self.movement_toggle_button.grid(row=0, column=0, columnspan=3, pady=(0, 8))
        layout = [
            ("Yaw Left (Q)", "q", 0, 0),
            ("Forward (W)", "w", 0, 1),
            ("Yaw Right (E)", "e", 0, 2),
            ("Left (A)", "a", 1, 0),
            ("Hold (X)", "x", 1, 1),
            ("Right (D)", "d", 1, 2),
            ("Up (R)", "r", 2, 0),
            ("Backward (S)", "s", 2, 1),
            ("Down (F)", "f", 2, 2),
        ]
        for label, symbol, row, col in layout:
            tk.Button(move_frame, text=label, width=16, command=lambda s=symbol: self.send_move_symbol(s)).grid(row=row + 1, column=col, padx=4, pady=4, sticky="nsew")

        bottom = tk.Frame(main)
        bottom.pack(fill="x", pady=(0, 10))

        seq_frame = tk.LabelFrame(bottom, text="Sequence Control", padx=8, pady=8)
        seq_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))
        tk.Label(seq_frame, text="Symbols").grid(row=0, column=0, sticky="w")
        tk.Entry(seq_frame, textvariable=self.sequence_var, width=40).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        tk.Button(seq_frame, text="Execute Sequence", width=16, command=self.on_execute_sequence).grid(row=0, column=2)
        tk.Label(seq_frame, text="Delay ms").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tk.Entry(seq_frame, textvariable=self.sequence_delay_var, width=10).grid(row=1, column=1, sticky="w", padx=(8, 8), pady=(8, 0))
        tk.Button(seq_frame, text="Stop Sequence", width=16, command=self.on_stop_sequence).grid(row=1, column=2, pady=(8, 0))
        tk.Label(seq_frame, text="Use w/s/a/d/r/f/q/e/x. Example: wwwqdd", anchor="w").grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        seq_frame.grid_columnconfigure(1, weight=1)

        preview_frame = tk.LabelFrame(bottom, text="Preview Windows", padx=8, pady=8)
        preview_frame.pack(side="right", fill="both")
        tk.Button(preview_frame, text="Toggle RGB", width=16, command=self.toggle_preview_window).grid(row=0, column=0, padx=4, pady=4)
        tk.Button(preview_frame, text="Refresh RGB", width=16, command=self.refresh_preview_window).grid(row=0, column=1, padx=4, pady=4)
        tk.Checkbutton(preview_frame, text="Auto RGB", variable=self.auto_preview_var).grid(row=0, column=2, padx=4, pady=4, sticky="w")
        tk.Button(preview_frame, text="Toggle Depth", width=16, command=self.toggle_depth_window).grid(row=1, column=0, padx=4, pady=4)
        tk.Button(preview_frame, text="Refresh Depth", width=16, command=self.refresh_depth_window).grid(row=1, column=1, padx=4, pady=4)
        tk.Checkbutton(preview_frame, text="Auto Depth", variable=self.auto_depth_var).grid(row=1, column=2, padx=4, pady=4, sticky="w")

        # --- Mission Map panel ---
        mission_frame = tk.LabelFrame(main, text="Multi-House Mission", padx=8, pady=8)
        mission_frame.pack(fill="x", pady=(0, 8))
        tk.Label(mission_frame, textvariable=self.mission_status_var, anchor="w").pack(fill="x")
        tk.Label(mission_frame, textvariable=self.target_house_var, anchor="w").pack(fill="x")
        mission_btn_row = tk.Frame(mission_frame)
        mission_btn_row.pack(fill="x", pady=(6, 0))
        tk.Button(mission_btn_row, text="Open Map", width=14,
                  command=self.toggle_map_window).pack(side="left", padx=(0, 6))
        tk.Button(mission_btn_row, text="Auto-Select Nearest", width=18,
                  command=self.on_auto_select_house).pack(side="left", padx=(0, 6))
        tk.Button(mission_btn_row, text="Nav Step →House", width=18,
                  command=self.on_navigate_step_to_house).pack(side="left", padx=(0, 6))
        tk.Button(mission_btn_row, text="Mark Explored ✓", width=16,
                  command=self.on_mark_explored).pack(side="left", padx=(0, 6))
        tk.Button(mission_btn_row, text="Mark Person Found !", width=18,
                  command=self.on_mark_person_found).pack(side="left")

        footer = tk.Frame(main)
        footer.pack(fill="x")
        tk.Button(footer, text="Refresh State", width=16, command=lambda: self.run_manual_async_request("Refreshing state", self.refresh_state_once)).pack(side="left")
        tk.Checkbutton(footer, text="Auto State", variable=self.auto_state_var).pack(side="left", padx=(10, 0))
        tk.Label(footer, text=f"State interval: {self.args.state_interval_ms} ms").pack(side="left", padx=(12, 0))

    def on_set_task(self) -> None:
        def worker() -> None:
            response = self.safe_request(self.client.post_json, "/task", {"task_label": self.task_label_var.get().strip()}, label="Set Task")
            if isinstance(response, dict):
                self.root.after(0, lambda: self.apply_state(response))
        self.run_manual_async_request("Setting task", worker)

    def on_capture(self) -> None:
        def worker() -> None:
            response = self.safe_request(self.client.post_json, "/capture", {"label": self.capture_label_var.get().strip()}, label="Capture")
            if isinstance(response, dict):
                state = response.get("state", response)
                if isinstance(state, dict):
                    self.root.after(0, lambda: self.apply_state(state))
        self.run_manual_async_request("Capturing", worker)

    def on_set_pose(self) -> None:
        if self.pose_text is None:
            self.set_status("Pose input is unavailable.")
            return
        raw_text = self.pose_text.get("1.0", "end").strip()
        try:
            payload = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            self.set_status(f"Invalid pose JSON: {exc}")
            return
        if not isinstance(payload, dict):
            self.set_status("Pose JSON must be an object.")
            return

        def worker() -> None:
            response = self.safe_request(self.client.post_json, "/set_pose", payload, label="Set Pose")
            if isinstance(response, dict):
                self.root.after(0, lambda: self.apply_state(response))

        self.run_manual_async_request("Setting pose", worker)

    def on_toggle_movement(self) -> None:
        desired_enabled = not self.movement_enabled_state

        def worker() -> None:
            response = self.safe_request(self.client.post_json, "/basic_movement_enable", {"enabled": desired_enabled}, label="Toggle Movement")
            if isinstance(response, dict):
                self.root.after(0, lambda: self.apply_state(response))
        self.run_manual_async_request("Toggling movement", worker)

    def _execute_move_symbol_blocking(self, symbol: str, *, from_sequence: bool = False) -> bool:
        payload = MOVE_COMMANDS.get(symbol.lower())
        if payload is None:
            self.root.after(0, lambda: self.set_status(f"Unsupported move symbol: {symbol}"))
            return False
        self.move_request_inflight = True
        self.pause_background_refresh(1.0)
        try:
            response = self.safe_request(self.client.post_json, "/move_relative", payload, label=f"Move {symbol}")
            if isinstance(response, dict):
                self.root.after(0, lambda: self.apply_state(response))
                if from_sequence:
                    self.root.after(0, lambda s=symbol: self.set_status(f"Sequence sent: {s}"))
                return True
            return False
        finally:
            self.move_request_inflight = False
            self.pause_background_refresh(0.4)

    def send_move_symbol(self, symbol: str) -> None:
        if self.move_request_inflight:
            self.set_status(f"Move {symbol} ignored while another move is in flight.")
            return

        def worker() -> None:
            self._execute_move_symbol_blocking(symbol)

        threading.Thread(target=worker, daemon=True).start()

    def on_execute_sequence(self) -> None:
        symbols = [symbol for symbol in self.sequence_var.get().strip().lower() if symbol in MOVE_COMMANDS]
        if not symbols:
            self.set_status("No valid sequence symbols.")
            return
        try:
            delay_s = max(0.05, float(self.sequence_delay_var.get().strip()) / 1000.0)
        except ValueError:
            self.set_status("Invalid sequence delay.")
            return
        if self.sequence_thread is not None and self.sequence_thread.is_alive():
            self.set_status("Sequence already running.")
            return

        def worker() -> None:
            self.sequence_stop_event.clear()
            total = len(symbols)
            for index, symbol in enumerate(symbols, start=1):
                if self.sequence_stop_event.is_set():
                    self.root.after(0, lambda: self.set_status(f"Sequence stopped at step {index - 1}/{total}."))
                    return
                while self.move_request_inflight and not self.sequence_stop_event.is_set():
                    time.sleep(0.02)
                if self.sequence_stop_event.is_set():
                    self.root.after(0, lambda: self.set_status(f"Sequence stopped at step {index - 1}/{total}."))
                    return
                self.root.after(0, lambda i=index, t=total, s=symbol: self.set_status(f"Sequence step {i}/{t}: {s}"))
                ok = self._execute_move_symbol_blocking(symbol, from_sequence=True)
                if not ok:
                    self.root.after(0, lambda i=index, t=total: self.set_status(f"Sequence failed at step {i}/{t}."))
                    return
                time.sleep(delay_s)
            self.root.after(0, lambda: self.set_status(f"Sequence completed: {total}/{total} steps"))

        self.sequence_thread = threading.Thread(target=worker, daemon=True)
        self.sequence_thread.start()

    def on_stop_sequence(self) -> None:
        self.sequence_stop_event.set()
        self.set_status("Stopping sequence...")

    def toggle_preview_window(self) -> None:
        if self.preview_window is not None and self.preview_window.winfo_exists():
            self.preview_window.destroy()
            self.preview_window = None
            self.preview_label = None
            self.preview_photo = None
            return
        self.preview_window = tk.Toplevel(self.root)
        self.preview_window.title("UAV RGB Preview")
        self.preview_label = tk.Label(self.preview_window)
        self.preview_label.pack(fill="both", expand=True)
        self.refresh_preview_window()

    def refresh_preview_window(self) -> None:
        if self.preview_window is None or self.preview_label is None:
            return
        if self.preview_refresh_inflight:
            return

        def worker() -> None:
            self.preview_refresh_inflight = True
            try:
                self.safe_request(self.client.post_json, "/refresh", {}, label="Sync Observation")
                image = self.safe_request(self.client.get_image, "/frame", label="Refresh RGB")
                depth_image = self.safe_request(self.client.get_image, "/depth_frame", label="Refresh Depth")
                if isinstance(image, np.ndarray):
                    photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
                    self.root.after(0, lambda: self.apply_preview_photo(photo, rgb=True))
                if isinstance(depth_image, np.ndarray):
                    depth_photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(depth_image, cv2.COLOR_BGR2RGB)))
                    self.root.after(0, lambda: self.apply_preview_photo(depth_photo, rgb=False))
            finally:
                self.preview_refresh_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def toggle_depth_window(self) -> None:
        if self.depth_window is not None and self.depth_window.winfo_exists():
            self.depth_window.destroy()
            self.depth_window = None
            self.depth_label = None
            self.depth_photo = None
            return
        self.depth_window = tk.Toplevel(self.root)
        self.depth_window.title("UAV Depth Preview")
        self.depth_label = tk.Label(self.depth_window)
        self.depth_label.pack(fill="both", expand=True)
        self.refresh_depth_window()

    def refresh_depth_window(self) -> None:
        if self.depth_window is None or self.depth_label is None:
            return
        if self.depth_refresh_inflight:
            return

        def worker() -> None:
            self.depth_refresh_inflight = True
            try:
                self.safe_request(self.client.post_json, "/refresh", {}, label="Sync Observation")
                image = self.safe_request(self.client.get_image, "/depth_frame", label="Refresh Depth")
                rgb_image = self.safe_request(self.client.get_image, "/frame", label="Refresh RGB")
                if isinstance(image, np.ndarray):
                    photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
                    self.root.after(0, lambda: self.apply_preview_photo(photo, rgb=False))
                if isinstance(rgb_image, np.ndarray):
                    rgb_photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)))
                    self.root.after(0, lambda: self.apply_preview_photo(rgb_photo, rgb=True))
            finally:
                self.depth_refresh_inflight = False

        threading.Thread(target=worker, daemon=True).start()

    def apply_preview_photo(self, photo: ImageTk.PhotoImage, *, rgb: bool) -> None:
        if rgb:
            self.preview_photo = photo
            if self.preview_label is not None:
                self.preview_label.configure(image=photo)
        else:
            self.depth_photo = photo
            if self.depth_label is not None:
                self.depth_label.configure(image=photo)

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
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )
    panel = BasicUAVControlPanel(args)
    panel.run()


if __name__ == "__main__":
    main()
