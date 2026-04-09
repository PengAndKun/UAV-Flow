from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import Any, Dict, List, Optional
from urllib import request

import cv2
import numpy as np
from PIL import Image, ImageTk

from map_overhead_widget import OverheadMapWidget

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from phase2_multimodal_fusion_analysis import (
    find_latest_phase2_weights,
    run_phase2_fusion_analysis,
)

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

    def get_json(self, path: str, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        req = request.Request(f"{self.base_url}{path}", method="GET")
        with request.urlopen(req, timeout=self.timeout_s if timeout_s is None else float(timeout_s)) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def post_json(self, path: str, payload: Optional[Dict[str, Any]] = None, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        body = json.dumps(payload or {}).encode("utf-8")
        req = request.Request(f"{self.base_url}{path}", data=body, headers={"Content-Type": "application/json"}, method="POST")
        with request.urlopen(req, timeout=self.timeout_s if timeout_s is None else float(timeout_s)) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def get_image(self, path: str, timeout_s: Optional[float] = None) -> np.ndarray:
        req = request.Request(f"{self.base_url}{path}", method="GET")
        with request.urlopen(req, timeout=self.timeout_s if timeout_s is None else float(timeout_s)) as resp:
            body = resp.read()
        image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Failed to decode image from {path}")
        return image

    def get_image_unchanged(self, path: str, timeout_s: Optional[float] = None) -> np.ndarray:
        req = request.Request(f"{self.base_url}{path}", method="GET")
        with request.urlopen(req, timeout=self.timeout_s if timeout_s is None else float(timeout_s)) as resp:
            body = resp.read()
        image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
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
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.main_canvas = tk.Canvas(self.root, highlightthickness=0)
        self.main_scrollbar = tk.Scrollbar(self.root, orient="vertical", command=self.main_canvas.yview)
        self.main_canvas.configure(yscrollcommand=self.main_scrollbar.set)
        self.main_canvas.grid(row=0, column=0, sticky="nsew")
        self.main_scrollbar.grid(row=0, column=1, sticky="ns")
        self.content_frame = tk.Frame(self.main_canvas)
        self.content_window = self.main_canvas.create_window((0, 0), window=self.content_frame, anchor="nw")
        self.content_frame.bind("<Configure>", self._on_content_frame_configure)
        self.main_canvas.bind("<Configure>", self._on_main_canvas_configure)
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self.root.bind_all("<Button-4>", self._on_mousewheel_linux)
        self.root.bind_all("<Button-5>", self._on_mousewheel_linux)

        self.status_var = tk.StringVar(value="Ready")
        self.pose_var = tk.StringVar(value="Pose: waiting...")
        self.depth_var = tk.StringVar(value="Depth: waiting...")
        self.control_var = tk.StringVar(value="Movement: waiting...")
        self.capture_var = tk.StringVar(value="Capture: waiting...")
        self.fusion_summary_var = tk.StringVar(value="Fusion: idle")
        self.fusion_dir_var = tk.StringVar(value="Fusion dir: none")
        self.fusion_model_var = tk.StringVar(value="Fusion model: waiting...")
        self.mission_var = tk.StringVar(value="Mission: idle")
        self.current_house_var = tk.StringVar(value="Current house: none")
        self.target_house_var = tk.StringVar(value="Target: none")
        self.target_dist_var = tk.StringVar(value="Target distance: n/a")
        self.map_status_var = tk.StringVar(value="Map: closed")
        self.map_pose_var = tk.StringVar(value="Map pose: n/a")
        self.calib_var = tk.StringVar(value="Calibration: none")
        self.anchor_world_x_var = tk.StringVar(value="")
        self.anchor_world_y_var = tk.StringVar(value="")
        self.house_set_var = tk.BooleanVar(value=False)
        self.house_id_var = tk.StringVar(value="")
        self.house_name_var = tk.StringVar(value="")
        self.house_box_var = tk.StringVar(value="House Set: idle")

        self.task_label_var = tk.StringVar(value="")
        self.capture_label_var = tk.StringVar(value="")
        self.phase1_settle_var = tk.StringVar(value="0.20")
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
        self.fusion_preview_label: Optional[tk.Label] = None
        self.fusion_preview_photo: Optional[ImageTk.PhotoImage] = None
        self.fusion_window: Optional[tk.Toplevel] = None
        self.fusion_result_label: Optional[tk.Label] = None
        self.fusion_result_photo: Optional[ImageTk.PhotoImage] = None
        self.map_window: Optional[tk.Toplevel] = None
        self.map_widget: Optional[OverheadMapWidget] = None
        self.open_map_window: Optional[tk.Toplevel] = None
        self.open_map_widget: Optional[OverheadMapWidget] = None
        self.pose_text: Optional[tk.Text] = None
        self.movement_toggle_button: Optional[tk.Button] = None
        self.calibration_anchors: List[Dict[str, float]] = []
        self.pending_anchor_world: Optional[Dict[str, float]] = None
        self.pending_image_anchor: Optional[Dict[str, float]] = None
        self.pending_house_rect: Optional[Dict[str, float]] = None
        self.loaded_map_image_path: str = ""
        self.loaded_map_image: Optional[np.ndarray] = None
        self.last_fusion_overlay_path: str = ""
        self.last_fusion_result_dir: str = ""

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
        self.phase2_fusion_output_root = os.path.join(PROJECT_ROOT, "phase2_multimodal_fusion_analysis", "results")

        self.build_ui()
        self._refresh_default_fusion_model()
        for delay, fn in ((200, self.schedule_state_refresh), (350, self.schedule_preview_refresh), (500, self.schedule_depth_refresh), (1200, self.schedule_map_refresh)):
            self.root.after(delay, fn)

    def build_ui(self) -> None:
        root = self.content_frame
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
        tk.Button(task, text="Phase1 Scan x12", command=self.on_capture_phase1_scan).grid(row=1, column=3, padx=6, pady=6)
        tk.Label(task, text="Settle s").grid(row=1, column=4, sticky="e", padx=(10, 4), pady=6)
        tk.Entry(task, textvariable=self.phase1_settle_var, width=8).grid(row=1, column=5, sticky="w", padx=(0, 6), pady=6)
        tk.Label(task, text="Init Pose JSON").grid(row=2, column=0, sticky="nw", padx=6, pady=6)
        self.pose_text = tk.Text(task, width=42, height=6)
        self.pose_text.grid(row=2, column=1, sticky="ew", padx=6, pady=6)
        self.pose_text.insert("1.0", json.dumps({"x": 2359.9, "y": 85.3, "z": 225.0, "yaw": -1.7}, indent=2))
        tk.Button(task, text="Set Pose", command=self.on_set_pose).grid(row=2, column=2, padx=6, pady=6, sticky="n")

        fusion = tk.LabelFrame(left, text="Phase2 Fusion")
        fusion.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        fusion.grid_columnconfigure(0, weight=1)
        fusion.grid_columnconfigure(1, weight=1)
        tk.Button(fusion, text="Run Fusion Analyze", command=self.on_run_phase2_fusion).grid(row=0, column=0, padx=6, pady=6, sticky="ew")
        tk.Button(fusion, text="Open Fusion Result", command=self.open_fusion_result_window).grid(row=0, column=1, padx=6, pady=6, sticky="ew")
        tk.Label(fusion, textvariable=self.fusion_model_var, anchor="w", justify="left").grid(row=1, column=0, columnspan=2, sticky="ew", padx=6, pady=(2, 2))
        tk.Label(fusion, textvariable=self.fusion_dir_var, anchor="w", justify="left").grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 2))
        tk.Label(
            fusion,
            textvariable=self.fusion_summary_var,
            anchor="w",
            justify="left",
            wraplength=560,
            font=("Consolas", 10),
        ).grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=(2, 4))
        self.fusion_preview_label = tk.Label(fusion, text="No fusion image yet", anchor="center")
        self.fusion_preview_label.grid(row=4, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))

        move = tk.LabelFrame(left, text="Basic Movement"); move.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.movement_toggle_button = tk.Button(move, text="Enable Basic Movement", command=self.on_toggle_movement, width=24)
        self.movement_toggle_button.grid(row=0, column=0, columnspan=3, pady=(6, 10))
        for label, symbol, row, col in (
            ("Yaw Left (Q)", "q", 1, 0), ("Forward (W)", "w", 1, 1), ("Yaw Right (E)", "e", 1, 2),
            ("Left (A)", "a", 2, 0), ("Hold (X)", "x", 2, 1), ("Right (D)", "d", 2, 2),
            ("Up (R)", "r", 3, 0), ("Backward (S)", "s", 3, 1), ("Down (F)", "f", 3, 2),
        ):
            tk.Button(move, text=label, command=lambda s=symbol: self.send_move_symbol(s), width=18).grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
        for col in range(3): move.grid_columnconfigure(col, weight=1)

        seq = tk.LabelFrame(left, text="Sequence Control"); seq.grid(row=3, column=0, sticky="ew", pady=(0, 8)); seq.grid_columnconfigure(1, weight=1)
        tk.Label(seq, text="Symbols").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(seq, textvariable=self.sequence_var).grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        tk.Button(seq, text="Execute Sequence", command=self.on_execute_sequence).grid(row=0, column=2, padx=6, pady=6)
        tk.Label(seq, text="Delay ms").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(seq, textvariable=self.sequence_delay_var, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=6)
        tk.Button(seq, text="Stop Sequence", command=self.on_stop_sequence).grid(row=1, column=2, padx=6, pady=6)
        tk.Label(seq, text="Use w/s/a/d/r/f/q/e/x. Example: wwwqdd", anchor="w").grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 6))

        preview = tk.LabelFrame(left, text="Preview Windows"); preview.grid(row=4, column=0, sticky="ew", pady=(0, 8))
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

    def _on_content_frame_configure(self, _event: tk.Event) -> None:
        self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))

    def _on_main_canvas_configure(self, event: tk.Event) -> None:
        self.main_canvas.itemconfigure(self.content_window, width=event.width)

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        units = -int(delta / 120) if delta % 120 == 0 else (-1 if delta > 0 else 1)
        self.main_canvas.yview_scroll(units, "units")

    def _on_mousewheel_linux(self, event: tk.Event) -> None:
        num = getattr(event, "num", None)
        if num == 4:
            self.main_canvas.yview_scroll(-1, "units")
        elif num == 5:
            self.main_canvas.yview_scroll(1, "units")

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

    def _coerce_anchor_list(self, anchors: Any) -> List[Dict[str, float]]:
        if not isinstance(anchors, list):
            return []
        restored: List[Dict[str, float]] = []
        for index, anchor in enumerate(anchors[:5], start=1):
            if not isinstance(anchor, dict):
                continue
            try:
                restored.append(
                    {
                        "index": float(anchor.get("index", index)),
                        "label": str(anchor.get("label", f"P{index}")),
                        "world_x": float(anchor["world_x"]),
                        "world_y": float(anchor["world_y"]),
                        "image_x": float(anchor["image_x"]),
                        "image_y": float(anchor["image_y"]),
                    }
                )
            except Exception:
                continue
        return restored

    def _solve_affine_from_anchors(self, anchors: List[Dict[str, float]]) -> Optional[List[List[float]]]:
        if len(anchors) < 3:
            return None
        try:
            world = np.asarray(
                [[float(anchor["world_x"]), float(anchor["world_y"]), 1.0] for anchor in anchors],
                dtype=np.float64,
            )
            image_x = np.asarray([float(anchor["image_x"]) for anchor in anchors], dtype=np.float64)
            image_y = np.asarray([float(anchor["image_y"]) for anchor in anchors], dtype=np.float64)
            row_x, *_ = np.linalg.lstsq(world, image_x, rcond=None)
            row_y, *_ = np.linalg.lstsq(world, image_y, rcond=None)
            return [
                [float(row_x[0]), float(row_x[1]), float(row_x[2])],
                [float(row_y[0]), float(row_y[1]), float(row_y[2])],
            ]
        except Exception:
            return None

    def _normalize_calibration_payload(self, payload: Any) -> Dict[str, Any]:
        calibration = payload if isinstance(payload, dict) else {}
        anchors = self._coerce_anchor_list(calibration.get("anchors", []))
        affine = calibration.get("affine_world_to_image")
        if not (isinstance(affine, list) and len(affine) == 2):
            affine = self._solve_affine_from_anchors(anchors)
        if not (isinstance(affine, list) and len(affine) == 2):
            return {}
        normalized: Dict[str, Any] = {
            "anchors": anchors,
            "affine_world_to_image": affine,
        }
        if calibration.get("image_width") is not None:
            try:
                normalized["image_width"] = int(calibration.get("image_width"))
            except Exception:
                pass
        if calibration.get("image_height") is not None:
            try:
                normalized["image_height"] = int(calibration.get("image_height"))
            except Exception:
                pass
        if calibration.get("rmse_px") is not None:
            try:
                normalized["rmse_px"] = float(calibration.get("rmse_px"))
            except Exception:
                pass
        return normalized

    def _get_saved_calibration(self, registry: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        overhead_cfg = self._get_local_overhead_config()
        local_calibration = self._normalize_calibration_payload(overhead_cfg.get("calibration", {}))
        if local_calibration:
            return local_calibration
        reg = registry if isinstance(registry, dict) else {}
        registry_overhead = reg.get("overhead_map", {}) if isinstance(reg.get("overhead_map"), dict) else {}
        return self._normalize_calibration_payload(registry_overhead.get("calibration", {}))

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

    def _refresh_default_fusion_model(self) -> None:
        try:
            weights = find_latest_phase2_weights()
            self.fusion_model_var.set(f"Fusion model: {weights}")
        except Exception as exc:
            self.fusion_model_var.set(f"Fusion model: unavailable ({exc})")

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

    def _image_to_world_point(self, image_x: float, image_y: float, calibration: Dict[str, Any]) -> Optional[tuple[float, float]]:
        affine = calibration.get("affine_world_to_image") if isinstance(calibration, dict) else None
        if not isinstance(affine, list) or len(affine) != 2:
            return None
        try:
            matrix = np.asarray(affine, dtype=np.float64)
            linear = matrix[:, :2]
            offset = matrix[:, 2]
            if abs(np.linalg.det(linear)) < 1e-8:
                return None
            inv_linear = np.linalg.inv(linear)
            image_vec = np.asarray([float(image_x), float(image_y)], dtype=np.float64)
            world_vec = inv_linear.dot(image_vec - offset)
            return float(world_vec[0]), float(world_vec[1])
        except Exception:
            return None

    def _estimate_radius_cm_from_bbox(self, bbox: Dict[str, float], calibration: Dict[str, Any]) -> float:
        cx = 0.5 * (float(bbox["x1"]) + float(bbox["x2"]))
        cy = 0.5 * (float(bbox["y1"]) + float(bbox["y2"]))
        half_w = abs(float(bbox["x2"]) - float(bbox["x1"])) * 0.5
        half_h = abs(float(bbox["y2"]) - float(bbox["y1"])) * 0.5
        center_world = self._image_to_world_point(cx, cy, calibration)
        right_world = self._image_to_world_point(cx + half_w, cy, calibration)
        down_world = self._image_to_world_point(cx, cy + half_h, calibration)
        if center_world is None or right_world is None or down_world is None:
            return 700.0
        dx_r = right_world[0] - center_world[0]
        dy_r = right_world[1] - center_world[1]
        dx_d = down_world[0] - center_world[0]
        dy_d = down_world[1] - center_world[1]
        radius = max(
            float(np.hypot(dx_r, dy_r)),
            float(np.hypot(dx_d, dy_d)),
        )
        return max(300.0, radius)

    def _load_local_house_boxes(self) -> List[Dict[str, Any]]:
        raw = self._read_local_houses_config()
        houses = raw.get("houses", []) if isinstance(raw.get("houses"), list) else []
        result: List[Dict[str, Any]] = []
        for house in houses:
            bbox = house.get("map_bbox_image")
            if not isinstance(bbox, dict):
                continue
            try:
                result.append(
                    {
                        "id": str(house.get("id", "")),
                        "name": str(house.get("name", house.get("id", ""))),
                        "status": str(house.get("status", "UNSEARCHED")),
                        "map_bbox_image": {
                            "x1": float(bbox["x1"]),
                            "y1": float(bbox["y1"]),
                            "x2": float(bbox["x2"]),
                            "y2": float(bbox["y2"]),
                        },
                    }
                )
            except Exception:
                continue
        return result

    def _build_map_display_boxes(
        self,
        reg_houses: List[Dict[str, Any]],
        current_id: str,
        target_id: str,
        calibration: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        local_boxes = self._load_local_house_boxes()
        by_id: Dict[str, Dict[str, Any]] = {
            str(box.get("id", "") or ""): dict(box)
            for box in local_boxes
            if str(box.get("id", "") or "")
        }
        display_boxes: List[Dict[str, Any]] = []
        for house in reg_houses:
            hid = str(house.get("id", "") or "")
            if not hid:
                continue
            box = dict(by_id.get(hid, {}))
            if not box:
                try:
                    cx = float(house.get("center_x", 0.0))
                    cy = float(house.get("center_y", 0.0))
                    radius_cm = float(house.get("radius_cm", 600.0))
                except Exception:
                    continue
                corners_world = [
                    (cx - radius_cm, cy - radius_cm),
                    (cx + radius_cm, cy - radius_cm),
                    (cx + radius_cm, cy + radius_cm),
                    (cx - radius_cm, cy + radius_cm),
                ]
                corners_image = [
                    self._world_to_image_point(wx, wy, calibration)
                    for wx, wy in corners_world
                ]
                if any(pt is None for pt in corners_image):
                    continue
                xs = [float(pt[0]) for pt in corners_image if pt is not None]
                ys = [float(pt[1]) for pt in corners_image if pt is not None]
                if not xs or not ys:
                    continue
                box = {
                    "id": hid,
                    "name": str(house.get("name", hid)),
                    "status": str(house.get("status", "UNSEARCHED")),
                    "map_bbox_image": {
                        "x1": min(xs),
                        "y1": min(ys),
                        "x2": max(xs),
                        "y2": max(ys),
                    },
                }
            box["id"] = hid
            box["name"] = str(house.get("name", box.get("name", hid)))
            box["status"] = str(house.get("status", box.get("status", "UNSEARCHED")))
            box["is_target"] = hid == target_id
            box["is_current"] = hid == current_id
            display_boxes.append(box)
        return display_boxes

    def _load_local_registry_payload(self) -> Dict[str, Any]:
        raw = self._read_local_houses_config()
        if not isinstance(raw, dict):
            return {}
        houses = raw.get("houses", [])
        if not isinstance(houses, list):
            houses = []
        current_target_id = str(raw.get("current_target_id", "") or "")
        return {
            "world_bounds": raw.get("world_bounds", {}),
            "overhead_map": raw.get("overhead_map", {}),
            "current_target_id": current_target_id,
            "houses": houses,
            "status_summary": self._compute_house_status_summary(houses),
        }

    def _compute_house_status_summary(self, houses: List[Dict[str, Any]]) -> Dict[str, int]:
        summary = {"UNSEARCHED": 0, "IN_PROGRESS": 0, "EXPLORED": 0, "PERSON_FOUND": 0}
        for house in houses:
            status = str(house.get("status", "UNSEARCHED"))
            summary[status] = int(summary.get(status, 0)) + 1
        return summary

    def _find_containing_house_id(self, x: float, y: float, houses: List[Dict[str, Any]]) -> str:
        for house in houses:
            try:
                cx = float(house.get("center_x", 0.0))
                cy = float(house.get("center_y", 0.0))
                radius = float(house.get("radius_cm", 0.0))
            except Exception:
                continue
            if radius > 0.0 and float(np.hypot(x - cx, y - cy)) <= radius:
                return str(house.get("id", "") or "")
        return ""

    def _save_local_houses_config(self, payload: Dict[str, Any]) -> None:
        with open(self.houses_config_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)

    def _restore_saved_points_from_registry(self, reg: Dict[str, Any]) -> None:
        if self.calibration_anchors:
            return
        calibration = self._get_saved_calibration(reg)
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
                if isinstance(state, dict) and isinstance(registry, dict):
                    self.root.after(0, lambda: self.apply_map_data(state, registry))
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
        if not self.root.winfo_exists():
            return
        reg = registry.get("registry", {})
        if not isinstance(reg, dict):
            reg = {}
        local_reg = self._load_local_registry_payload()
        if not isinstance(reg.get("houses"), list) or not reg.get("houses"):
            reg["houses"] = list(local_reg.get("houses", []))
        if not str(reg.get("current_target_id", "") or "") and str(local_reg.get("current_target_id", "") or ""):
            reg["current_target_id"] = local_reg.get("current_target_id", "")
        if not isinstance(reg.get("status_summary"), dict) or not reg.get("status_summary"):
            reg["status_summary"] = dict(local_reg.get("status_summary", {}))

        summary = reg.get("status_summary", {})
        mission = state.get("house_mission", {}) if isinstance(state.get("house_mission"), dict) else {}
        overhead = state.get("overhead_map", {}) if isinstance(state.get("overhead_map"), dict) else {}
        calibration = self._get_saved_calibration(reg)
        self._restore_saved_points_from_registry(reg)
        if self.pending_image_anchor is not None:
            self.calib_var.set(
                f"Calibration: clicked image=({self.pending_image_anchor['image_x']:.1f}, {self.pending_image_anchor['image_y']:.1f}) "
                f"anchors={len(self.calibration_anchors)}/5"
            )
        elif calibration:
            self.calib_var.set(
                f"Calibration: saved anchors={len(calibration.get('anchors', []))} "
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
        elif calibration.get("image_width") and calibration.get("image_height"):
            image_size = (int(calibration.get("image_width", 0)), int(calibration.get("image_height", 0)))
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

        current_id = str(mission.get("current_house_id", "") or "")
        target_id = str(reg.get("current_target_id", "") or "")
        if not current_id:
            current_id = self._find_containing_house_id(pose_x, pose_y, reg.get("houses", []))

        house_name_by_id: Dict[str, str] = {}
        houses = []
        target_xy = None
        for h in reg.get("houses", []):
            hid = str(h.get("id", "") or "")
            name = str(h.get("name", hid))
            house_name_by_id[hid] = name
            if hid == current_id:
                name = f"{name} (UAV)"
            cx = float(h.get("center_x", 0.0))
            cy = float(h.get("center_y", 0.0))
            is_target = hid == target_id
            is_current = hid == current_id
            if is_target:
                target_xy = (cx, cy)
            houses.append(
                {
                    "id": hid,
                    "name": name,
                    "center_x": cx,
                    "center_y": cy,
                    "radius_cm": float(h.get("radius_cm", 600.0)),
                    "status": h.get("status", "UNSEARCHED"),
                    "is_target": is_target,
                    "is_current": is_current,
                }
            )

        display_house_boxes = self._build_map_display_boxes(reg.get("houses", []), current_id, target_id, calibration)

        current_name = str(mission.get("current_house_name", "") or house_name_by_id.get(current_id, "") or "none")
        target_name = str(mission.get("target_house_name", "") or house_name_by_id.get(target_id, "") or "none")
        self.mission_var.set(f"Mission: unsearched={summary.get('UNSEARCHED',0)} in_progress={summary.get('IN_PROGRESS',0)} explored={summary.get('EXPLORED',0)} found={summary.get('PERSON_FOUND',0)}")
        self.map_status_var.set(
            f"Map: current={current_name} "
            f"target={target_name} "
            f"bg={overhead.get('refresh_time','-')}"
        )

        if self.map_widget is not None and self.map_window is not None and self.map_window.winfo_exists():
            if self.loaded_map_image is not None:
                self.map_widget.set_background_image(self.loaded_map_image)
            self.map_widget.set_calibration(affine, image_size, anchors)
            self.map_widget.set_house_boxes(display_house_boxes)
            self.map_widget.update_uav(pose_x, pose_y, pose_yaw)
            self.map_widget.update_houses(houses if self.show_houses_var.get() else [])
            self.map_widget.set_route_target(target_xy if (self.show_houses_var.get() and self.show_route_var.get()) else None)

        if self.open_map_widget is not None and self.open_map_window is not None and self.open_map_window.winfo_exists():
            if self.loaded_map_image is not None:
                self.open_map_widget.set_background_image(self.loaded_map_image)
            # Open view mirrors the calibrated map display without the
            # editable calibration anchors or interactive house-setting tools.
            self.open_map_widget.set_calibration(affine, image_size, [])
            self.open_map_widget.set_house_boxes(display_house_boxes)
            self.open_map_widget.update_uav(pose_x, pose_y, pose_yaw)
            self.open_map_widget.update_houses([])
            self.open_map_widget.set_route_target(None)

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
    def on_capture_phase1_scan(self) -> None:
        long_timeout_s = max(float(self.args.timeout_s), 120.0)
        try:
            settle_time_s = max(0.0, float(self.phase1_settle_var.get().strip()))
        except ValueError:
            self.status_var.set("Invalid Phase1 settle time.")
            return

        def _run_scan() -> None:
            response = self.safe(
                self.client.post_json,
                "/capture_phase1_spin_scan",
                {
                    "label": self.capture_label_var.get().strip(),
                    "num_steps": 12,
                    "settle_time_s": settle_time_s,
                },
                label="Phase1 Scan x12",
                timeout_s=long_timeout_s,
            ) or {}
            state = response.get("state")
            if isinstance(state, dict):
                self._apply_response(state)
            scan = response.get("scan", {}) if isinstance(response.get("scan"), dict) else {}
            scan_dir = str(scan.get("scan_dir", "") or response.get("manifest_path", "") or "").strip()
            if scan_dir:
                self.root.after(0, lambda: self.status_var.set(f"Phase1 scan saved: {scan_dir}"))

        self.call_async(
            "Phase1 spin scan",
            _run_scan,
        )
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

    def _load_display_photo(self, image_path: str, *, max_width: int, max_height: int) -> Optional[ImageTk.PhotoImage]:
        if not image_path or not os.path.exists(image_path):
            return None
        image = cv2.imread(image_path)
        if image is None:
            return None
        h, w = image.shape[:2]
        scale = min(float(max_width) / max(1, w), float(max_height) / max(1, h), 1.0)
        if scale < 1.0:
            image = cv2.resize(image, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(Image.fromarray(rgb))

    def apply_fusion_result(self, result: Dict[str, Any]) -> None:
        self.last_fusion_result_dir = str(result.get("labeling_dir", result.get("run_dir", "")) or "")
        self.last_fusion_overlay_path = str(result.get("fusion_overlay_path", "") or "")
        self.fusion_summary_var.set(str(result.get("panel_summary", "Fusion: no summary")))
        self.fusion_dir_var.set(f"Fusion dir: {self.last_fusion_result_dir or 'none'}")
        weights_path = str(result.get("weights_path", "") or "")
        if weights_path:
            self.fusion_model_var.set(f"Fusion model: {weights_path}")
        photo = self._load_display_photo(self.last_fusion_overlay_path, max_width=560, max_height=260)
        if photo is not None and self.fusion_preview_label is not None:
            self.fusion_preview_photo = photo
            self.fusion_preview_label.configure(image=photo, text="")
        elif self.fusion_preview_label is not None:
            self.fusion_preview_label.configure(image="", text="Fusion image unavailable")

    def on_run_phase2_fusion(self) -> None:
        def worker() -> None:
            try:
                self.safe(self.client.post_json, "/refresh", {}, label="Sync Fusion Observation")
                rgb = self.safe(self.client.get_image, "/frame", label="Fusion RGB")
                depth_raw = self.safe(self.client.get_image_unchanged, "/depth_raw", label="Fusion Depth Raw")
                camera_info_resp = self.safe(self.client.get_json, "/camera_info", label="Fusion Camera Info") or {}
                state = self.safe(self.client.get_json, "/state", label="Fusion State") or {}
                if not isinstance(rgb, np.ndarray) or not isinstance(depth_raw, np.ndarray):
                    return
                camera_info = camera_info_resp.get("camera_info") if isinstance(camera_info_resp, dict) else {}
                label = self.capture_label_var.get().strip() or self.task_label_var.get().strip()
                result = run_phase2_fusion_analysis(
                    rgb_bgr=rgb,
                    depth_raw=depth_raw,
                    output_root=Path(self.phase2_fusion_output_root),
                    label=label,
                    camera_info=camera_info if isinstance(camera_info, dict) else {},
                    state=state if isinstance(state, dict) else {},
                )
                self.root.after(0, lambda r=result: self.apply_fusion_result(r))
                self.root.after(0, lambda r=result: self.status_var.set(f"Phase2 fusion saved: {r.get('run_dir', '')}"))
            except Exception as exc:
                logger.warning("Phase2 fusion analysis failed: %s", exc)
                self.root.after(0, lambda e=exc: self.status_var.set(f"Phase2 fusion failed: {e}"))

        self.call_async("Phase2 fusion analysis", worker)

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

    def open_fusion_result_window(self) -> None:
        if not self.last_fusion_overlay_path or not os.path.exists(self.last_fusion_overlay_path):
            self.status_var.set("No fusion result image available yet.")
            return
        if self.fusion_window and self.fusion_window.winfo_exists():
            self.fusion_window.destroy()
            self.fusion_window = None
            self.fusion_result_label = None
            self.fusion_result_photo = None
            return
        self.fusion_window = tk.Toplevel(self.root)
        self.fusion_window.title("Phase2 Fusion Result")
        self.fusion_result_label = tk.Label(self.fusion_window)
        self.fusion_result_label.pack(fill="both", expand=True)
        photo = self._load_display_photo(self.last_fusion_overlay_path, max_width=1280, max_height=760)
        if photo is None:
            self.fusion_result_label.configure(text="Failed to load fusion result image.")
            return
        self.fusion_result_photo = photo
        self.fusion_result_label.configure(image=photo)

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
        self.open_map_window.protocol("WM_DELETE_WINDOW", self._close_open_map_window)
        toolbar = tk.Frame(self.open_map_window)
        toolbar.pack(fill="x", padx=8, pady=(8, 0))
        tk.Label(toolbar, textvariable=self.map_pose_var, anchor="w").pack(side="left", padx=(0, 8))
        tk.Label(toolbar, textvariable=self.map_status_var, anchor="w").pack(side="left")
        self.open_map_widget = OverheadMapWidget(self.open_map_window, world_bounds=bounds, canvas_w=760, canvas_h=560)
        self.open_map_widget.canvas.pack(padx=8, pady=8)
        self.refresh_map_async()

    def toggle_map_window(self) -> None:
        if self.map_window and self.map_window.winfo_exists():
            self._close_map_window(); return
        self._ensure_default_map_loaded(force=True)
        self.map_window = tk.Toplevel(self.root); self.map_window.title("Setting Map - Alignment"); self.map_window.resizable(False, False)
        self.map_window.protocol("WM_DELETE_WINDOW", self._close_map_window)
        toolbar = tk.Frame(self.map_window); toolbar.pack(fill="x", padx=8, pady=(8, 0))
        tk.Button(toolbar, text="Capture Fixed Map", command=self.refresh_map_background_async).pack(side="left")
        tk.Button(toolbar, text="Load Local Image", command=self.on_load_map_image).pack(side="left", padx=(6, 0))
        tk.Button(toolbar, text="Add UAV Anchor", command=self.on_add_uav_anchor).pack(side="left", padx=(6, 0))
        tk.Button(toolbar, text="Add Point", command=self.on_add_anchor_from_inputs).pack(side="left", padx=(6, 0))
        tk.Button(toolbar, text="Save Alignment", command=self.on_solve_calibration).pack(side="left", padx=(6, 0))
        tk.Button(toolbar, text="Clear Calib", command=self.on_clear_calibration).pack(side="left", padx=(6, 0))
        tk.Checkbutton(toolbar, text="House Set", variable=self.house_set_var, command=self.on_toggle_house_set_mode).pack(side="left", padx=(10, 0))
        tk.Button(toolbar, text="Save House", command=self.on_save_house_annotation).pack(side="left", padx=(6, 0))
        tk.Label(toolbar, textvariable=self.map_status_var).pack(side="left", padx=10)
        calib = tk.Frame(self.map_window)
        calib.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(calib, text="World X").pack(side="left")
        tk.Entry(calib, textvariable=self.anchor_world_x_var, width=10).pack(side="left", padx=(4, 8))
        tk.Label(calib, text="World Y").pack(side="left")
        tk.Entry(calib, textvariable=self.anchor_world_y_var, width=10).pack(side="left", padx=(4, 8))
        tk.Label(calib, textvariable=self.calib_var, anchor="w").pack(side="left", padx=8)
        house_row = tk.Frame(self.map_window)
        house_row.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(house_row, text="House ID").pack(side="left")
        tk.Entry(house_row, textvariable=self.house_id_var, width=12).pack(side="left", padx=(4, 8))
        tk.Label(house_row, text="Name").pack(side="left")
        tk.Entry(house_row, textvariable=self.house_name_var, width=16).pack(side="left", padx=(4, 8))
        tk.Label(house_row, textvariable=self.house_box_var, anchor="w").pack(side="left", padx=8)
        reg = self.safe(self.client.get_json, "/house_registry", label="Load map bounds") or {}
        self._restore_saved_points_from_registry(reg.get("registry", {}))
        wb = reg.get("registry", {}).get("world_bounds", {})
        bounds = (float(wb.get("min_x", 1000.0)), float(wb.get("min_y", -500.0)), float(wb.get("max_x", 5000.0)), float(wb.get("max_y", 3000.0)))
        self.map_widget = OverheadMapWidget(self.map_window, world_bounds=bounds, canvas_w=760, canvas_h=560)
        self.map_widget.canvas.pack(padx=8, pady=8)
        self.map_widget.set_map_click_callback(self.on_map_click)
        self.map_widget.set_rect_select_callback(self.on_house_rect_selected)
        self.map_widget.set_rect_select_enabled(bool(self.house_set_var.get()))
        self.refresh_map_async(with_background=False)

    def _close_open_map_window(self) -> None:
        try:
            if self.open_map_window is not None and self.open_map_window.winfo_exists():
                self.open_map_window.destroy()
        except Exception:
            pass
        self.open_map_window = None
        self.open_map_widget = None
        self.map_status_var.set("Map: closed")

    def _close_map_window(self) -> None:
        try:
            if self.map_window is not None and self.map_window.winfo_exists():
                self.map_window.destroy()
        except Exception:
            pass
        self.map_window = None
        self.map_widget = None
        self.map_status_var.set("Map: closed")

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

    def on_toggle_house_set_mode(self) -> None:
        enabled = bool(self.house_set_var.get())
        if self.map_widget is not None:
            self.map_widget.set_rect_select_enabled(enabled)
        if enabled:
            self.house_box_var.set("House Set: drag a rectangle on the map.")
        else:
            self.house_box_var.set("House Set: idle")

    def on_house_rect_selected(self, rect: Dict[str, float]) -> None:
        self.pending_house_rect = {
            "x1": float(rect["x1"]),
            "y1": float(rect["y1"]),
            "x2": float(rect["x2"]),
            "y2": float(rect["y2"]),
        }
        if not self.house_id_var.get().strip():
            existing = self._load_local_house_boxes()
            self.house_id_var.set(f"house_{len(existing) + 1:02d}")
        if not self.house_name_var.get().strip():
            self.house_name_var.set(self.house_id_var.get().strip() or "house")
        self.house_box_var.set(
            "House Set: rect "
            f"({self.pending_house_rect['x1']:.1f}, {self.pending_house_rect['y1']:.1f}) -> "
            f"({self.pending_house_rect['x2']:.1f}, {self.pending_house_rect['y2']:.1f})"
        )
        self.refresh_map_async()

    def on_save_house_annotation(self) -> None:
        if self.pending_house_rect is None:
            self.house_box_var.set("House Set: draw a rectangle first.")
            return
        house_id = self.house_id_var.get().strip()
        if not house_id:
            self.house_box_var.set("House Set: house id required.")
            return
        house_name = self.house_name_var.get().strip() or house_id

        def worker() -> None:
            raw = self._read_local_houses_config()
            overhead_cfg = raw.get("overhead_map", {}) if isinstance(raw.get("overhead_map"), dict) else {}
            calibration = overhead_cfg.get("calibration", {}) if isinstance(overhead_cfg.get("calibration"), dict) else {}
            center_image_x = 0.5 * (float(self.pending_house_rect["x1"]) + float(self.pending_house_rect["x2"]))
            center_image_y = 0.5 * (float(self.pending_house_rect["y1"]) + float(self.pending_house_rect["y2"]))
            center_world = self._image_to_world_point(center_image_x, center_image_y, calibration)
            if center_world is None:
                self.root.after(0, lambda: self.house_box_var.set("House Set: solve calibration first."))
                return

            radius_cm = self._estimate_radius_cm_from_bbox(self.pending_house_rect, calibration)
            houses = raw.get("houses", [])
            if not isinstance(houses, list):
                houses = []
            updated = False
            for house in houses:
                if str(house.get("id", "")) != house_id:
                    continue
                house["name"] = house_name
                house["center_x"] = float(center_world[0])
                house["center_y"] = float(center_world[1])
                house["center_z"] = float(house.get("center_z", 200.0))
                house["approach_z"] = float(house.get("approach_z", 600.0))
                house["radius_cm"] = float(radius_cm)
                house["entry_yaw_hint"] = float(house.get("entry_yaw_hint", 0.0))
                house["map_bbox_image"] = dict(self.pending_house_rect)
                house.setdefault("status", "UNSEARCHED")
                house.setdefault("notes", "")
                updated = True
                break
            if not updated:
                houses.append(
                    {
                        "id": house_id,
                        "name": house_name,
                        "center_x": float(center_world[0]),
                        "center_y": float(center_world[1]),
                        "center_z": 200.0,
                        "approach_z": 600.0,
                        "radius_cm": float(radius_cm),
                        "entry_yaw_hint": 0.0,
                        "status": "UNSEARCHED",
                        "search_start_time": None,
                        "search_end_time": None,
                        "person_location": None,
                        "notes": "",
                        "map_bbox_image": dict(self.pending_house_rect),
                    }
                )
            raw["houses"] = houses
            self._save_local_houses_config(raw)
            self.safe(self.client.post_json, "/reload_house_registry", {}, label="Reload house registry")
            self.pending_house_rect = None
            self.root.after(0, lambda: self.house_box_var.set(f"House Set: saved '{house_id}'"))
            self.root.after(0, lambda: self.house_id_var.set(house_id))
            self.root.after(0, lambda: self.house_name_var.set(house_name))
            self.refresh_map_async()

        threading.Thread(target=worker, daemon=True).start()

    def on_close(self) -> None:
        for window in (self.preview_window, self.depth_window, self.fusion_window, self.map_window, self.open_map_window):
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
