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
from tkinter import filedialog, ttk
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

TARGET_CONDITIONED_STATE_OPTIONS: List[str] = [
    "target_house_not_in_view",
    "target_house_entry_visible",
    "target_house_entry_approachable",
    "target_house_entry_blocked",
    "non_target_house_entry_visible",
    "target_house_geometric_opening_needs_confirmation",
]

TARGET_CONDITIONED_SUBGOAL_OPTIONS: List[str] = [
    "reorient_to_target_house",
    "keep_search_target_house",
    "approach_target_entry",
    "align_target_entry",
    "detour_left_to_target_entry",
    "detour_right_to_target_entry",
    "cross_target_entry",
    "ignore_non_target_entry",
    "backoff_and_reobserve",
]

TARGET_CONDITIONED_ACTION_OPTIONS: List[str] = [
    "forward",
    "yaw_left",
    "yaw_right",
    "left",
    "right",
    "backward",
    "hold",
]

TARGET_CONDITIONED_STATE_NOTES: Dict[str, str] = {
    "target_house_not_in_view": "目标房屋当前不在主要视野内，通常应继续转向目标房屋，而不是处理当前画面中的入口。",
    "target_house_entry_visible": "目标房屋已经在视野内，但入口还不够明确或不够稳定，适合继续围绕目标房屋观察和搜索。",
    "target_house_entry_approachable": "目标房屋入口已经明确，且前方条件允许靠近，通常应继续朝入口接近。",
    "target_house_entry_blocked": "目标房屋入口是对的，但前方有障碍、遮挡或路径受限，需要先绕行或调整位置。",
    "non_target_house_entry_visible": "当前看到的是非目标房屋的入口，即使它看起来可进入，也不应该优先进这个门。",
    "target_house_geometric_opening_needs_confirmation": "几何上像目标房屋有开口，但语义或归属还不够确定，需要继续确认是否真的是目标入口。",
}

TARGET_CONDITIONED_SUBGOAL_NOTES: Dict[str, str] = {
    "reorient_to_target_house": "当前主要任务是重新朝向目标房屋，让目标房屋重新进入有效视野。",
    "keep_search_target_house": "目标房屋已经基本进入观察范围，但还需要继续扫描立面或门口区域，找到真正入口。",
    "approach_target_entry": "目标入口已经确认，下一步应稳定向入口靠近，而不是继续横向搜索。",
    "align_target_entry": "入口已经找到，但机体姿态或中心线还没对齐，需要先调整朝向和位置。",
    "detour_left_to_target_entry": "入口方向是正确的，但左侧绕行更合理，通常是因为左侧更安全或障碍更少。",
    "detour_right_to_target_entry": "入口方向是正确的，但右侧绕行更合理，通常是因为右侧更安全或障碍更少。",
    "cross_target_entry": "已经足够接近且对齐目标入口，可以尝试穿过入口进入。",
    "ignore_non_target_entry": "虽然画面里有入口，但它不属于目标房屋，应忽略并继续找目标房屋入口。",
    "backoff_and_reobserve": "当前判断不稳定或距离太近导致视野差，先后退一点，再重新观察会更稳。",
}

TARGET_CONDITIONED_ACTION_NOTES: Dict[str, str] = {
    "forward": "当前最合理的是向前推进，通常表示入口已较明确且前方空间允许靠近。",
    "yaw_left": "当前更适合向左转头，常见于目标房屋或目标入口偏在左侧，需要重新把它转到中央。",
    "yaw_right": "当前更适合向右转头，常见于目标房屋或目标入口偏在右侧，需要重新把它转到中央。",
    "left": "当前更适合向左平移或左绕行，通常是为了避开障碍或贴近左侧入口通道。",
    "right": "当前更适合向右平移或右绕行，通常是为了避开障碍或贴近右侧入口通道。",
    "backward": "当前更适合后退，通常是因为离得太近、视野过窄或需要先拉开距离再判断。",
    "hold": "当前更适合悬停保持，说明这一步主要是在稳定观察，而不是立刻移动。",
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
        self.door_var = tk.StringVar(value="Door: waiting...")
        self.memory_var = tk.StringVar(value="Memory: waiting...")
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
        self.selected_target_house_var = tk.StringVar(value="")
        self.anchor_world_x_var = tk.StringVar(value="")
        self.anchor_world_y_var = tk.StringVar(value="")
        self.house_set_var = tk.BooleanVar(value=False)
        self.house_id_var = tk.StringVar(value="")
        self.house_name_var = tk.StringVar(value="")
        self.house_box_var = tk.StringVar(value="House Set: idle")

        self.task_label_var = tk.StringVar(value="")
        self.capture_label_var = tk.StringVar(value="")
        self.memory_episode_label_var = tk.StringVar(value="")
        self.memory_capture_label_var = tk.StringVar(value="")
        self.memory_step_display_var = tk.StringVar(value="0")
        self.memory_reset_on_start_var = tk.BooleanVar(value=True)
        self.memory_auto_refresh_var = tk.BooleanVar(value=True)
        self.memory_auto_mode_var = tk.StringVar(value="off")
        self.memory_auto_seconds_var = tk.StringVar(value="5")
        self.memory_auto_steps_var = tk.StringVar(value="3")
        self.memory_auto_status_var = tk.StringVar(value="Auto Capture: off")
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
        self.memory_window: Optional[tk.Toplevel] = None
        self.memory_text: Optional[tk.Text] = None
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
        self.target_house_combo: Optional[ttk.Combobox] = None
        self.target_house_choice_map: Dict[str, str] = {}
        self.target_house_display_by_id: Dict[str, str] = {}
        self.target_house_selection_dirty = False
        self.review_window: Optional[tk.Toplevel] = None
        self.review_image_label: Optional[tk.Label] = None
        self.review_image_photo: Optional[ImageTk.PhotoImage] = None
        self.review_house_var = tk.StringVar(value="")
        self.review_status_var = tk.StringVar(value="Reviewer: idle")
        self.review_info_var = tk.StringVar(value="No sample loaded.")
        self.review_queue: List[Dict[str, Any]] = []
        self.review_current_item: Optional[Dict[str, Any]] = None
        self.indicator_review_window: Optional[tk.Toplevel] = None
        self.indicator_review_image_label: Optional[tk.Label] = None
        self.indicator_review_image_photo: Optional[ImageTk.PhotoImage] = None
        self.indicator_review_status_var = tk.StringVar(value="Indicator reviewer: idle")
        self.indicator_review_info_var = tk.StringVar(value="No sample loaded.")
        self.indicator_review_queue: List[Dict[str, Any]] = []
        self.indicator_review_current_item: Optional[Dict[str, Any]] = None
        self.indicator_review_house_var = tk.StringVar(value="")
        self.indicator_review_state_var = tk.StringVar(value="")
        self.indicator_review_subgoal_var = tk.StringVar(value="")
        self.indicator_review_action_var = tk.StringVar(value="")
        self.indicator_review_state_note_var = tk.StringVar(value="")
        self.indicator_review_subgoal_note_var = tk.StringVar(value="")
        self.indicator_review_action_note_var = tk.StringVar(value="")
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
        self.memory_capture_inflight = False
        self.state_refresh_inflight = False
        self.preview_refresh_inflight = False
        self.depth_refresh_inflight = False
        self.map_refresh_inflight = False
        self.map_background_refresh_inflight = False
        self.memory_refresh_inflight = False
        self.background_pause_until = 0.0
        self.sequence_thread: Optional[threading.Thread] = None
        self.sequence_stop_event = threading.Event()
        self.movement_enabled_state = False
        self.latest_state: Dict[str, Any] = {}
        self.latest_memory_collection_state: Dict[str, Any] = {}
        self.memory_auto_enabled = False
        self.memory_auto_last_capture_time = 0.0
        self.memory_auto_last_capture_step = 0
        self.memory_auto_episode_id = ""

        self.eval_dir = os.path.dirname(os.path.abspath(__file__))
        self.houses_config_path = os.path.join(self.eval_dir, "houses_config.json")
        self.default_map_path = os.path.join(self.eval_dir, "map", "qq.png")
        self.phase2_fusion_output_root = os.path.join(PROJECT_ROOT, "phase2_multimodal_fusion_analysis", "results")

        self.indicator_review_state_var.trace_add("write", lambda *_: self._refresh_indicator_review_explanations())
        self.indicator_review_subgoal_var.trace_add("write", lambda *_: self._refresh_indicator_review_explanations())
        self.indicator_review_action_var.trace_add("write", lambda *_: self._refresh_indicator_review_explanations())

        self.build_ui()
        self._refresh_target_house_choices()
        self._refresh_default_fusion_model()
        for delay, fn in ((200, self.schedule_state_refresh), (350, self.schedule_preview_refresh), (500, self.schedule_depth_refresh), (900, self.schedule_memory_auto_capture), (1200, self.schedule_map_refresh)):
            self.root.after(delay, fn)

    def build_ui(self) -> None:
        root = self.content_frame
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)

        status = tk.LabelFrame(root, text="Runtime Status")
        status.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=8, pady=8)
        for idx, var in enumerate((self.status_var, self.pose_var, self.depth_var, self.control_var, self.door_var, self.memory_var, self.capture_var, self.mission_var, self.current_house_var, self.target_house_var, self.target_dist_var, self.map_status_var, self.map_pose_var, self.calib_var)):
            tk.Label(status, textvariable=var, anchor="w", justify="left", font=("Consolas", 11)).grid(row=idx, column=0, sticky="ew", padx=6, pady=2)
        status.grid_columnconfigure(0, weight=1)

        left = tk.Frame(root); left.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=(0, 8)); left.grid_columnconfigure(0, weight=1)
        right = tk.Frame(root); right.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=(0, 8)); right.grid_columnconfigure(0, weight=1)

        task = tk.LabelFrame(left, text="Task And Capture"); task.grid(row=0, column=0, sticky="ew", pady=(0, 8)); task.grid_columnconfigure(1, weight=1)
        tk.Label(task, text="Task Label").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(task, textvariable=self.task_label_var).grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        tk.Button(task, text="Set Task", command=self.on_set_task).grid(row=0, column=2, padx=6, pady=6)
        tk.Label(task, text="House ID").grid(row=0, column=3, sticky="e", padx=(10, 4), pady=6)
        self.target_house_combo = ttk.Combobox(
            task,
            textvariable=self.selected_target_house_var,
            state="readonly",
            width=22,
        )
        self.target_house_combo.grid(row=0, column=4, columnspan=2, sticky="w", padx=(0, 6), pady=6)
        self.target_house_combo.bind("<<ComboboxSelected>>", self.on_target_house_combo_selected)
        tk.Label(task, text="Run Label").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(task, textvariable=self.capture_label_var).grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        tk.Button(task, text="Phase1 Scan x12", command=self.on_capture_phase1_scan).grid(row=1, column=2, padx=6, pady=6)
        tk.Label(task, text="Settle s").grid(row=1, column=4, sticky="e", padx=(10, 4), pady=6)
        tk.Entry(task, textvariable=self.phase1_settle_var, width=8).grid(row=1, column=5, sticky="w", padx=(0, 6), pady=6)
        tk.Label(task, text="Init Pose JSON").grid(row=2, column=0, sticky="nw", padx=6, pady=6)
        self.pose_text = tk.Text(task, width=42, height=6)
        self.pose_text.grid(row=2, column=1, sticky="ew", padx=6, pady=6)
        self.pose_text.insert("1.0", json.dumps({"x": 2359.9, "y": 85.3, "z": 225.0, "yaw": -1.7}, indent=2))
        tk.Button(task, text="Set Pose", command=self.on_set_pose).grid(row=2, column=2, padx=6, pady=6, sticky="n")

        memory = tk.LabelFrame(left, text="Memory Collection")
        memory.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        memory.grid_columnconfigure(1, weight=1)
        memory.grid_columnconfigure(3, weight=1)
        tk.Label(memory, text="Episode Label").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(memory, textvariable=self.memory_episode_label_var).grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        tk.Checkbutton(memory, text="Reset Store On Start", variable=self.memory_reset_on_start_var).grid(row=0, column=2, columnspan=2, sticky="w", padx=6, pady=6)
        tk.Button(memory, text="Start Episode", command=self.on_memory_collection_start).grid(row=1, column=0, padx=6, pady=6, sticky="ew")
        tk.Button(memory, text="Stop Episode", command=self.on_memory_collection_stop).grid(row=1, column=1, padx=6, pady=6, sticky="ew")
        tk.Button(memory, text="Reset Memory", command=self.on_memory_collection_reset).grid(row=1, column=2, padx=6, pady=6, sticky="ew")
        tk.Button(memory, text="Snapshot Now", command=self.on_memory_snapshot).grid(row=1, column=3, padx=6, pady=6, sticky="ew")
        tk.Label(memory, text="Capture Label").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(memory, textvariable=self.memory_capture_label_var).grid(row=2, column=1, sticky="ew", padx=6, pady=6)
        tk.Label(memory, text="Current Step").grid(row=2, column=2, sticky="e", padx=(10, 4), pady=6)
        tk.Label(memory, textvariable=self.memory_step_display_var, anchor="w").grid(row=2, column=3, sticky="w", padx=(0, 6), pady=6)
        tk.Button(memory, text="Capture Step+Analyze", command=self.on_memory_capture_analyze).grid(row=3, column=0, columnspan=2, padx=6, pady=(0, 6), sticky="ew")
        tk.Button(memory, text="Open Memory Window", command=self.toggle_memory_window).grid(row=3, column=2, columnspan=2, padx=6, pady=(0, 6), sticky="ew")
        tk.Label(memory, text="Auto Mode").grid(row=4, column=0, sticky="w", padx=6, pady=6)
        ttk.Combobox(
            memory,
            textvariable=self.memory_auto_mode_var,
            values=["off", "time", "step"],
            state="readonly",
            width=12,
        ).grid(row=4, column=1, sticky="w", padx=6, pady=6)
        tk.Label(memory, text="Time(s)").grid(row=4, column=2, sticky="e", padx=(10, 4), pady=6)
        tk.Entry(memory, textvariable=self.memory_auto_seconds_var, width=10).grid(row=4, column=3, sticky="w", padx=(0, 6), pady=6)
        tk.Label(memory, text="Step Interval").grid(row=5, column=0, sticky="w", padx=6, pady=(0, 6))
        tk.Entry(memory, textvariable=self.memory_auto_steps_var, width=12).grid(row=5, column=1, sticky="w", padx=6, pady=(0, 6))
        tk.Button(memory, text="Start Auto Capture", command=self.on_memory_auto_capture_start).grid(row=5, column=2, padx=6, pady=(0, 6), sticky="ew")
        tk.Button(memory, text="Stop Auto Capture", command=self.on_memory_auto_capture_stop).grid(row=5, column=3, padx=6, pady=(0, 6), sticky="ew")
        tk.Label(
            memory,
            textvariable=self.memory_auto_status_var,
            anchor="w",
            justify="left",
        ).grid(row=6, column=0, columnspan=4, sticky="ew", padx=6, pady=(0, 6))

        fusion = tk.LabelFrame(left, text="Phase2 Fusion")
        fusion.grid(row=2, column=0, sticky="ew", pady=(0, 8))
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
        tk.Button(fusion, text="House Reviewer", command=self.toggle_result_reviewer_window).grid(row=4, column=0, padx=6, pady=(0, 6), sticky="ew")
        tk.Button(fusion, text="Indicator Reviewer", command=self.toggle_indicator_reviewer_window).grid(row=4, column=1, padx=6, pady=(0, 6), sticky="ew")
        self.fusion_preview_label = tk.Label(fusion, text="No fusion image yet", anchor="center")
        self.fusion_preview_label.grid(row=5, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))

        move = tk.LabelFrame(left, text="Basic Movement"); move.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        self.movement_toggle_button = tk.Button(move, text="Enable Basic Movement", command=self.on_toggle_movement, width=24)
        self.movement_toggle_button.grid(row=0, column=0, columnspan=3, pady=(6, 10))
        for label, symbol, row, col in (
            ("Yaw Left (Q)", "q", 1, 0), ("Forward (W)", "w", 1, 1), ("Yaw Right (E)", "e", 1, 2),
            ("Left (A)", "a", 2, 0), ("Hold (X)", "x", 2, 1), ("Right (D)", "d", 2, 2),
            ("Up (R)", "r", 3, 0), ("Backward (S)", "s", 3, 1), ("Down (F)", "f", 3, 2),
        ):
            tk.Button(move, text=label, command=lambda s=symbol: self.send_move_symbol(s), width=18).grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
        tk.Button(move, text="Open Door", command=self.on_door_open, width=18).grid(row=4, column=0, padx=6, pady=6, sticky="nsew")
        tk.Button(move, text="Toggle Door", command=self.on_door_toggle, width=18).grid(row=4, column=1, padx=6, pady=6, sticky="nsew")
        tk.Button(move, text="Close Door", command=self.on_door_close, width=18).grid(row=4, column=2, padx=6, pady=6, sticky="nsew")
        tk.Button(move, text="Interact Door (E)", command=self.on_door_interact_e, width=18).grid(row=5, column=0, columnspan=3, padx=6, pady=(0, 6), sticky="nsew")
        for col in range(3): move.grid_columnconfigure(col, weight=1)

        seq = tk.LabelFrame(left, text="Sequence Control"); seq.grid(row=4, column=0, sticky="ew", pady=(0, 8)); seq.grid_columnconfigure(1, weight=1)
        tk.Label(seq, text="Symbols").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(seq, textvariable=self.sequence_var).grid(row=0, column=1, sticky="ew", padx=6, pady=6)
        tk.Button(seq, text="Execute Sequence", command=self.on_execute_sequence).grid(row=0, column=2, padx=6, pady=6)
        tk.Label(seq, text="Delay ms").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(seq, textvariable=self.sequence_delay_var, width=10).grid(row=1, column=1, sticky="w", padx=6, pady=6)
        tk.Button(seq, text="Stop Sequence", command=self.on_stop_sequence).grid(row=1, column=2, padx=6, pady=6)
        tk.Label(seq, text="Use w/s/a/d/r/f/q/e/x. Example: wwwqdd", anchor="w").grid(row=2, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 6))

        preview = tk.LabelFrame(left, text="Preview Windows"); preview.grid(row=5, column=0, sticky="ew", pady=(0, 8))
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

    def _resolve_scroll_target(self, event: tk.Event):
        widget = getattr(event, "widget", None)
        if widget is None:
            return None
        try:
            toplevel = widget.winfo_toplevel()
        except Exception:
            return None
        if self.memory_window is not None and self.memory_window.winfo_exists() and toplevel == self.memory_window and self.memory_text is not None:
            return ("memory_text", self.memory_text)
        if toplevel == self.root:
            return ("main_canvas", self.main_canvas)
        return None

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        units = -int(delta / 120) if delta % 120 == 0 else (-1 if delta > 0 else 1)
        target = self._resolve_scroll_target(event)
        if target is None:
            return
        _, widget = target
        widget.yview_scroll(units, "units")

    def _on_mousewheel_linux(self, event: tk.Event) -> None:
        num = getattr(event, "num", None)
        target = self._resolve_scroll_target(event)
        if target is None:
            return
        _, widget = target
        if num == 4:
            widget.yview_scroll(-1, "units")
        elif num == 5:
            widget.yview_scroll(1, "units")

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

    def _refresh_target_house_choices(self, *, preferred_house_id: str = "") -> None:
        raw = self._read_local_houses_config()
        houses = raw.get("houses", []) if isinstance(raw.get("houses"), list) else []
        current_target_id = str(raw.get("current_target_id", "") or "")
        choice_map: Dict[str, str] = {}
        display_by_id: Dict[str, str] = {}
        for house in houses:
            house_id = str(house.get("id", "") or "").strip()
            if not house_id:
                continue
            house_name = str(house.get("name", house_id) or house_id).strip()
            display = f"{house_id} | {house_name}"
            choice_map[display] = house_id
            display_by_id[house_id] = display
        self.target_house_choice_map = choice_map
        self.target_house_display_by_id = display_by_id
        values = list(choice_map.keys())
        if self.target_house_combo is not None:
            self.target_house_combo["values"] = values
        target_id = preferred_house_id.strip() or current_target_id.strip()
        current_display = self.selected_target_house_var.get().strip()
        if self.target_house_selection_dirty and current_display in choice_map and not preferred_house_id.strip():
            return
        if target_id and target_id in display_by_id:
            self.selected_target_house_var.set(display_by_id[target_id])
            self.target_house_selection_dirty = False
        elif values and current_display not in choice_map:
            self.selected_target_house_var.set(values[0])
            self.target_house_selection_dirty = False
        elif not values:
            self.selected_target_house_var.set("")
            self.target_house_selection_dirty = False

    def _get_selected_target_house_id(self) -> str:
        display = self.selected_target_house_var.get().strip()
        if not display:
            return ""
        house_id = self.target_house_choice_map.get(display, "")
        if house_id:
            return house_id
        if "|" in display:
            return display.split("|", 1)[0].strip()
        return display

    def _set_selected_target_house(self, house_id: str, *, mark_clean: bool = True) -> None:
        hid = str(house_id or "").strip()
        if not hid:
            return
        display = self.target_house_display_by_id.get(hid, "")
        if display:
            self.selected_target_house_var.set(display)
            if mark_clean:
                self.target_house_selection_dirty = False
        else:
            self._refresh_target_house_choices(preferred_house_id=hid)
            if mark_clean:
                self.target_house_selection_dirty = False

    def on_target_house_combo_selected(self, _event=None) -> None:
        selected_house_id = self._get_selected_target_house_id()
        if selected_house_id:
            self.target_house_selection_dirty = True
            self.status_var.set(f"Selected target house pending apply: {selected_house_id}")

    def _update_local_current_target_id(self, house_id: str) -> None:
        hid = str(house_id or "").strip()
        if not hid:
            return
        raw = self._read_local_houses_config()
        if not raw:
            return
        raw["current_target_id"] = hid
        self._save_local_houses_config(raw)
        self._refresh_target_house_choices(preferred_house_id=hid)

    def _review_json_path(self, labeling_dir: str) -> str:
        return os.path.join(labeling_dir, "target_house_review.json")

    def _indicator_review_json_path(self, labeling_dir: str) -> str:
        return os.path.join(labeling_dir, "target_conditioned_review.json")

    def _list_pending_review_items(self) -> List[Dict[str, Any]]:
        results_root = Path(self.phase2_fusion_output_root)
        if not results_root.exists():
            return []
        items: List[Dict[str, Any]] = []
        for run_dir in sorted(results_root.iterdir()):
            if not run_dir.is_dir():
                continue
            labeling_dir = run_dir / "labeling"
            overlay_path = labeling_dir / "fusion_overlay.png"
            manifest_path = labeling_dir / "labeling_manifest.json"
            fusion_result_path = labeling_dir / "fusion_result.json"
            review_path = Path(self._review_json_path(str(labeling_dir)))
            if not labeling_dir.exists() or not overlay_path.exists() or review_path.exists():
                continue
            manifest: Dict[str, Any] = {}
            fusion_result: Dict[str, Any] = {}
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            except Exception:
                manifest = {}
            try:
                fusion_result = json.loads(fusion_result_path.read_text(encoding="utf-8")) if fusion_result_path.exists() else {}
            except Exception:
                fusion_result = {}
            fusion_summary = fusion_result.get("fusion_summary", {}) if isinstance(fusion_result.get("fusion_summary"), dict) else {}
            target_context = fusion_result.get("target_context", {}) if isinstance(fusion_result.get("target_context"), dict) else {}
            target_house_id = str(
                fusion_summary.get("target_house_id", "") or
                target_context.get("target_house_id", "") or
                ""
            ).strip()
            target_house_name = str(target_context.get("target_house_name", "") or "").strip()
            items.append(
                {
                    "sample_id": str(manifest.get("sample_id", run_dir.name) or run_dir.name),
                    "run_dir": str(run_dir),
                    "labeling_dir": str(labeling_dir),
                    "overlay_path": str(overlay_path),
                    "task_label": str(manifest.get("task_label", "") or ""),
                    "target_house_id": target_house_id,
                    "target_house_name": target_house_name,
                    "review_json_path": str(review_path),
                }
            )
        return items

    def _build_house_display(self, house_id: str, house_name: str = "") -> str:
        hid = str(house_id or "").strip()
        if not hid:
            return "none"
        display = self.target_house_display_by_id.get(hid, "")
        if display:
            return display
        name = str(house_name or "").strip()
        if name and name.startswith(hid) and "|" in name:
            return name
        return f"{hid} | {name}" if name else hid

    def _refresh_review_queue(self) -> None:
        self._refresh_target_house_choices()
        self.review_queue = self._list_pending_review_items()
        if not self.review_queue:
            self.review_current_item = None
            self.review_status_var.set("Reviewer: no pending samples")
            self.review_info_var.set("All fusion_overlay samples already reviewed.")
            if self.review_image_label is not None:
                self.review_image_label.configure(image="", text="No pending review sample")
            return
        self._show_review_item(0)

    def _advance_review_item(self) -> None:
        if not self.review_queue:
            self._refresh_review_queue()
            return
        if len(self.review_queue) <= 1:
            self.review_status_var.set("Reviewer: skipped current sample, no later pending sample")
            return
        current_sample = str((self.review_current_item or {}).get("sample_id", "") or "")
        if current_sample:
            current_index = next(
                (idx for idx, queued in enumerate(self.review_queue) if str(queued.get("sample_id", "") or "") == current_sample),
                0,
            )
            next_index = (current_index + 1) % len(self.review_queue)
        else:
            next_index = 1
        self._show_review_item(next_index)

    def _show_review_item(self, index: int) -> None:
        if index < 0 or index >= len(self.review_queue):
            return
        item = self.review_queue[index]
        self.review_current_item = item
        display_target = self._build_house_display(item.get("target_house_id", ""), item.get("target_house_name", ""))
        self.review_house_var.set(display_target if display_target != "none" else "")
        self.review_status_var.set(
            f"Reviewer: {index + 1}/{len(self.review_queue)}  sample={item.get('sample_id', '')}"
        )
        self.review_info_var.set(
            f"Task: {item.get('task_label', '') or 'n/a'}\n"
            f"Pred target house: {display_target}\n"
            f"Run: {os.path.basename(str(item.get('run_dir', '')))}"
        )
        image_path = str(item.get("overlay_path", "") or "")
        photo = self._load_display_photo(image_path, max_width=880, max_height=620)
        if self.review_image_label is not None:
            if photo is not None:
                self.review_image_photo = photo
                self.review_image_label.configure(image=photo, text="")
            else:
                self.review_image_photo = None
                self.review_image_label.configure(image="", text=f"Failed to load image:\n{image_path}")

    def _save_review_result(self, *, is_correct: bool, corrected_house_id: str = "") -> None:
        item = self.review_current_item
        if not item:
            return
        original_house_id = str(item.get("target_house_id", "") or "").strip()
        selected_house_id = str(corrected_house_id or "").strip()
        final_house_id = original_house_id if is_correct else selected_house_id
        if not final_house_id:
            self.review_status_var.set("Reviewer: corrected house id required.")
            return
        payload = {
            "sample_id": str(item.get("sample_id", "") or ""),
            "run_dir": str(item.get("run_dir", "") or ""),
            "labeling_dir": str(item.get("labeling_dir", "") or ""),
            "task_label": str(item.get("task_label", "") or ""),
            "original_target_house_id": original_house_id,
            "original_target_house_name": str(item.get("target_house_name", "") or ""),
            "review_result": "correct" if is_correct else "corrected",
            "reviewed_house_id": final_house_id,
            "reviewed_house_name": self._build_house_display(final_house_id),
            "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        review_json_path = str(item.get("review_json_path", "") or "")
        with open(review_json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        current_sample = str(item.get("sample_id", "") or "")
        self.review_queue = [queued for queued in self.review_queue if str(queued.get("sample_id", "") or "") != current_sample]
        if self.review_queue:
            self._show_review_item(0)
            self.review_status_var.set(f"Reviewer: saved {current_sample}, moving to next")
        else:
            self.review_current_item = None
            self.review_status_var.set(f"Reviewer: saved {current_sample}, all pending complete")
            self.review_info_var.set("All fusion_overlay samples already reviewed.")
            if self.review_image_label is not None:
                self.review_image_label.configure(image="", text="No pending review sample")

    def on_review_mark_correct(self) -> None:
        self._save_review_result(is_correct=True)

    def on_review_mark_corrected(self) -> None:
        selected_house_id = self.target_house_choice_map.get(self.review_house_var.get().strip(), "")
        if not selected_house_id:
            display = self.review_house_var.get().strip()
            if "|" in display:
                selected_house_id = display.split("|", 1)[0].strip()
            else:
                selected_house_id = display
        self._save_review_result(is_correct=False, corrected_house_id=selected_house_id)

    def _extract_target_conditioned_fields(
        self,
        *,
        fusion_result: Optional[Dict[str, Any]] = None,
        teacher_root: Optional[Dict[str, Any]] = None,
        target_house_review: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        fusion_result = fusion_result if isinstance(fusion_result, dict) else {}
        teacher_root = teacher_root if isinstance(teacher_root, dict) else {}
        teacher_output = teacher_root.get("teacher_output", {}) if isinstance(teacher_root.get("teacher_output"), dict) else {}
        target_context = fusion_result.get("target_context", {}) if isinstance(fusion_result.get("target_context"), dict) else {}
        house_review = target_house_review if isinstance(target_house_review, dict) else {}

        target_house_id = str(
            house_review.get("reviewed_house_id", "") or
            teacher_output.get("target_house_id", "") or
            target_context.get("target_house_id", "") or
            ""
        ).strip()
        target_house_name = str(
            house_review.get("reviewed_house_name", "") or
            target_context.get("target_house_name", "") or
            ""
        ).strip()
        entry_state = str(
            teacher_output.get("entry_state", "") or
            fusion_result.get("final_entry_state", "") or
            ""
        ).strip()
        subgoal = str(
            teacher_output.get("subgoal", "") or
            fusion_result.get("recommended_subgoal", "") or
            ""
        ).strip()
        action_hint = str(
            teacher_output.get("action_hint", "") or
            fusion_result.get("recommended_action_hint", "") or
            ""
        ).strip()
        target_state = str(
            teacher_output.get("target_conditioned_state", "") or
            fusion_result.get("target_conditioned_state", "") or
            ""
        ).strip()
        target_subgoal = str(
            teacher_output.get("target_conditioned_subgoal", "") or
            fusion_result.get("target_conditioned_subgoal", "") or
            ""
        ).strip()
        target_action = str(
            teacher_output.get("target_conditioned_action_hint", "") or
            fusion_result.get("target_conditioned_action_hint", "") or
            ""
        ).strip()
        current_house_id = str(
            teacher_output.get("current_house_id", "") or
            target_context.get("current_house_id", "") or
            ""
        ).strip()
        return {
            "target_house_id": target_house_id,
            "target_house_name": target_house_name,
            "entry_state": entry_state,
            "subgoal": subgoal,
            "action_hint": action_hint,
            "target_state": target_state,
            "target_subgoal": target_subgoal,
            "target_action": target_action,
            "current_house_id": current_house_id,
        }

    def _list_pending_indicator_review_items(self) -> List[Dict[str, Any]]:
        results_root = Path(self.phase2_fusion_output_root)
        if not results_root.exists():
            return []
        items: List[Dict[str, Any]] = []
        for run_dir in sorted(results_root.iterdir()):
            if not run_dir.is_dir():
                continue
            labeling_dir = run_dir / "labeling"
            overlay_path = labeling_dir / "fusion_overlay.png"
            indicator_review_path = Path(self._indicator_review_json_path(str(labeling_dir)))
            fusion_result_path = labeling_dir / "fusion_result.json"
            teacher_output_path = labeling_dir / "teacher_output.json"
            manifest_path = labeling_dir / "labeling_manifest.json"
            house_review_path = Path(self._review_json_path(str(labeling_dir)))
            if not labeling_dir.exists() or not overlay_path.exists() or indicator_review_path.exists():
                continue
            manifest: Dict[str, Any] = {}
            fusion_result: Dict[str, Any] = {}
            teacher_root: Dict[str, Any] = {}
            house_review: Dict[str, Any] = {}
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
            except Exception:
                manifest = {}
            try:
                fusion_result = json.loads(fusion_result_path.read_text(encoding="utf-8")) if fusion_result_path.exists() else {}
            except Exception:
                fusion_result = {}
            try:
                teacher_root = json.loads(teacher_output_path.read_text(encoding="utf-8")) if teacher_output_path.exists() else {}
            except Exception:
                teacher_root = {}
            try:
                house_review = json.loads(house_review_path.read_text(encoding="utf-8")) if house_review_path.exists() else {}
            except Exception:
                house_review = {}
            teacher_output = teacher_root.get("teacher_output", {}) if isinstance(teacher_root.get("teacher_output"), dict) else {}
            task_label = str(manifest.get("task_label", "") or teacher_output.get("task_label", "") or "").strip()
            fields = self._extract_target_conditioned_fields(
                fusion_result=fusion_result,
                teacher_root=teacher_root,
                target_house_review=house_review,
            )
            items.append(
                {
                    "sample_id": str(manifest.get("sample_id", run_dir.name) or run_dir.name),
                    "run_dir": str(run_dir),
                    "labeling_dir": str(labeling_dir),
                    "overlay_path": str(overlay_path),
                    "task_label": task_label,
                    "indicator_review_json_path": str(indicator_review_path),
                    **fields,
                }
            )
        return items

    def _refresh_indicator_review_queue(self) -> None:
        self._refresh_target_house_choices()
        self.indicator_review_queue = self._list_pending_indicator_review_items()
        if not self.indicator_review_queue:
            self.indicator_review_current_item = None
            self.indicator_review_status_var.set("Indicator reviewer: no pending samples")
            self.indicator_review_info_var.set("All target-conditioned samples already reviewed.")
            if self.indicator_review_image_label is not None:
                self.indicator_review_image_label.configure(image="", text="No pending indicator review sample")
            return
        self._show_indicator_review_item(0)

    def _advance_indicator_review_item(self) -> None:
        if not self.indicator_review_queue:
            self._refresh_indicator_review_queue()
            return
        if len(self.indicator_review_queue) <= 1:
            self.indicator_review_status_var.set("Indicator reviewer: skipped current sample, no later pending sample")
            return
        current_sample = str((self.indicator_review_current_item or {}).get("sample_id", "") or "")
        if current_sample:
            current_index = next(
                (idx for idx, queued in enumerate(self.indicator_review_queue) if str(queued.get("sample_id", "") or "") == current_sample),
                0,
            )
            next_index = (current_index + 1) % len(self.indicator_review_queue)
        else:
            next_index = 1
        self._show_indicator_review_item(next_index)

    def _refresh_indicator_review_explanations(self) -> None:
        state_value = self.indicator_review_state_var.get().strip()
        subgoal_value = self.indicator_review_subgoal_var.get().strip()
        action_value = self.indicator_review_action_var.get().strip()
        self.indicator_review_state_note_var.set(
            TARGET_CONDITIONED_STATE_NOTES.get(state_value, "请选择最符合当前画面含义的目标状态。")
        )
        self.indicator_review_subgoal_note_var.set(
            TARGET_CONDITIONED_SUBGOAL_NOTES.get(subgoal_value, "请选择当前最合理的目标子任务。")
        )
        self.indicator_review_action_note_var.set(
            TARGET_CONDITIONED_ACTION_NOTES.get(action_value, "请选择当前最合理的动作建议。")
        )

    def _show_indicator_review_item(self, index: int) -> None:
        if index < 0 or index >= len(self.indicator_review_queue):
            return
        item = self.indicator_review_queue[index]
        self.indicator_review_current_item = item
        display_target = self._build_house_display(item.get("target_house_id", ""), item.get("target_house_name", ""))
        self.indicator_review_house_var.set("" if display_target == "none" else display_target)
        self.indicator_review_state_var.set(str(item.get("target_state", "") or ""))
        self.indicator_review_subgoal_var.set(str(item.get("target_subgoal", "") or ""))
        self.indicator_review_action_var.set(str(item.get("target_action", "") or ""))
        self._refresh_indicator_review_explanations()
        self.indicator_review_status_var.set(
            f"Indicator reviewer: {index + 1}/{len(self.indicator_review_queue)}  sample={item.get('sample_id', '')}"
        )
        self.indicator_review_info_var.set(
            f"Task: {item.get('task_label', '') or 'n/a'}\n"
            f"Current house: {item.get('current_house_id', '') or 'n/a'}\n"
            f"Reviewed target house: {display_target}\n"
            f"Pred entry/subgoal/action: {item.get('entry_state', '') or 'n/a'} / {item.get('subgoal', '') or 'n/a'} / {item.get('action_hint', '') or 'n/a'}\n"
            f"Pred target state/subgoal/action: {item.get('target_state', '') or 'n/a'} / {item.get('target_subgoal', '') or 'n/a'} / {item.get('target_action', '') or 'n/a'}\n"
            f"Run: {os.path.basename(str(item.get('run_dir', '')))}"
        )
        image_path = str(item.get("overlay_path", "") or "")
        photo = self._load_display_photo(image_path, max_width=880, max_height=620)
        if self.indicator_review_image_label is not None:
            if photo is not None:
                self.indicator_review_image_photo = photo
                self.indicator_review_image_label.configure(image=photo, text="")
            else:
                self.indicator_review_image_photo = None
                self.indicator_review_image_label.configure(image="", text=f"Failed to load image:\n{image_path}")

    def _save_indicator_review_result(
        self,
        *,
        is_correct: bool,
        corrected_house_id: str = "",
        corrected_state: str = "",
        corrected_subgoal: str = "",
        corrected_action: str = "",
    ) -> None:
        item = self.indicator_review_current_item
        if not item:
            return
        original_target_house_id = str(item.get("target_house_id", "") or "").strip()
        predicted_state = str(item.get("target_state", "") or "").strip()
        predicted_subgoal = str(item.get("target_subgoal", "") or "").strip()
        predicted_action = str(item.get("target_action", "") or "").strip()

        final_house_id = original_target_house_id if is_correct else str(corrected_house_id or "").strip()
        final_state = predicted_state if is_correct else str(corrected_state or "").strip()
        final_subgoal = predicted_subgoal if is_correct else str(corrected_subgoal or "").strip()
        final_action = predicted_action if is_correct else str(corrected_action or "").strip()

        if not final_house_id:
            self.indicator_review_status_var.set("Indicator reviewer: corrected house id required.")
            return
        if not final_state or not final_subgoal or not final_action:
            self.indicator_review_status_var.set("Indicator reviewer: target state/subgoal/action are all required.")
            return

        payload = {
            "sample_id": str(item.get("sample_id", "") or ""),
            "run_dir": str(item.get("run_dir", "") or ""),
            "labeling_dir": str(item.get("labeling_dir", "") or ""),
            "task_label": str(item.get("task_label", "") or ""),
            "original_target_house_id": original_target_house_id,
            "original_target_house_name": str(item.get("target_house_name", "") or ""),
            "predicted_entry_state": str(item.get("entry_state", "") or ""),
            "predicted_subgoal": str(item.get("subgoal", "") or ""),
            "predicted_action_hint": str(item.get("action_hint", "") or ""),
            "predicted_target_conditioned_state": predicted_state,
            "predicted_target_conditioned_subgoal": predicted_subgoal,
            "predicted_target_conditioned_action_hint": predicted_action,
            "review_result": "correct" if is_correct else "corrected",
            "reviewed_house_id": final_house_id,
            "reviewed_house_name": self._build_house_display(final_house_id),
            "reviewed_target_conditioned_state": final_state,
            "reviewed_target_conditioned_subgoal": final_subgoal,
            "reviewed_target_conditioned_action_hint": final_action,
            "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        review_json_path = str(item.get("indicator_review_json_path", "") or "")
        with open(review_json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        current_sample = str(item.get("sample_id", "") or "")
        self.indicator_review_queue = [queued for queued in self.indicator_review_queue if str(queued.get("sample_id", "") or "") != current_sample]
        if self.indicator_review_queue:
            self._show_indicator_review_item(0)
            self.indicator_review_status_var.set(f"Indicator reviewer: saved {current_sample}, moving to next")
        else:
            self.indicator_review_current_item = None
            self.indicator_review_status_var.set(f"Indicator reviewer: saved {current_sample}, all pending complete")
            self.indicator_review_info_var.set("All target-conditioned samples already reviewed.")
            if self.indicator_review_image_label is not None:
                self.indicator_review_image_label.configure(image="", text="No pending indicator review sample")

    def on_indicator_review_mark_correct(self) -> None:
        self._save_indicator_review_result(is_correct=True)

    def on_indicator_review_mark_corrected(self) -> None:
        display = self.indicator_review_house_var.get().strip()
        selected_house_id = self.target_house_choice_map.get(display, "")
        if not selected_house_id:
            if "|" in display:
                selected_house_id = display.split("|", 1)[0].strip()
            else:
                selected_house_id = display
        self._save_indicator_review_result(
            is_correct=False,
            corrected_house_id=selected_house_id,
            corrected_state=self.indicator_review_state_var.get().strip(),
            corrected_subgoal=self.indicator_review_subgoal_var.get().strip(),
            corrected_action=self.indicator_review_action_var.get().strip(),
        )

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

    def _reset_memory_auto_tracking(self, episode_id: str, step_index: int) -> None:
        self.memory_auto_episode_id = str(episode_id or "")
        self.memory_auto_last_capture_time = time.time()
        self.memory_auto_last_capture_step = int(step_index)
        self._refresh_memory_auto_status()

    def _refresh_memory_auto_status(self) -> None:
        memory_collection = self.latest_memory_collection_state if isinstance(self.latest_memory_collection_state, dict) else {}
        mode = self.memory_auto_mode_var.get().strip().lower() or "off"
        current_step = int(memory_collection.get("step_index", 0) or 0)
        episode_id = str(memory_collection.get("episode_id", "") or "")
        if not self.memory_auto_enabled:
            self.memory_auto_status_var.set(
                f"Auto Capture: off | mode={mode} | episode={episode_id or 'none'} | step={current_step}"
            )
            return
        if mode == "time":
            rule_text = f"every {self.memory_auto_seconds_var.get().strip() or '0'}s"
            last_text = datetime.fromtimestamp(float(self.memory_auto_last_capture_time)).strftime("%H:%M:%S") if self.memory_auto_last_capture_time else "n/a"
        elif mode == "step":
            rule_text = f"every {self.memory_auto_steps_var.get().strip() or '0'} step(s)"
            last_text = f"step {int(self.memory_auto_last_capture_step)}"
        else:
            rule_text = "mode=off"
            last_text = "n/a"
        state_text = "capturing" if self.memory_capture_inflight else "running"
        self.memory_auto_status_var.set(
            f"Auto Capture: {state_text} | mode={mode} | rule={rule_text} | episode={episode_id or 'none'} | step={current_step} | last={last_text}"
        )

    def apply_state(self, state: Dict[str, Any]) -> None:
        self.latest_state = state if isinstance(state, dict) else {}
        pose, depth = state.get("pose", {}), state.get("depth", {})
        mission = state.get("house_mission", {}) if isinstance(state.get("house_mission"), dict) else {}
        door_control = state.get("door_control", {}) if isinstance(state.get("door_control"), dict) else {}
        self.pose_var.set(f"Pose x={float(pose.get('x',0)):.1f} y={float(pose.get('y',0)):.1f} z={float(pose.get('z',0)):.1f} yaw={float(pose.get('task_yaw',0)):.1f} action={state.get('last_action','idle')} task={state.get('task_label','')}")
        self.depth_var.set(f"Depth frame={depth.get('frame_id','')} min={float(depth.get('min_depth',0)):.1f} max={float(depth.get('max_depth',0)):.1f} front_min={float(depth.get('front_min_depth',0)):.1f}")
        self.movement_enabled_state = bool(state.get("movement_enabled", False))
        self.control_var.set(f"Movement enabled={1 if self.movement_enabled_state else 0} origin={state.get('last_action_origin','n/a')}")
        nearest_door_name = str(door_control.get("nearest_door_name", "") or "")
        nearest_door_dist = door_control.get("nearest_door_distance_cm")
        nearest_door_loc = door_control.get("nearest_door_location") if isinstance(door_control.get("nearest_door_location"), list) else None
        control_player_name = str(door_control.get("control_player_name", "") or "")
        last_target_name = str(door_control.get("last_target_name", "") or "")
        last_ok = door_control.get("last_ok")
        last_response = str(door_control.get("last_response", "") or "")
        last_method = str(door_control.get("last_method", "") or "")
        last_window_title = str(door_control.get("last_window_title", "") or "")
        nearest_door_text = nearest_door_name or "none"
        if nearest_door_dist is not None:
            nearest_door_text = f"{nearest_door_text} ({float(nearest_door_dist):.1f}cm)"
        nearest_door_loc_text = "-"
        if isinstance(nearest_door_loc, list) and len(nearest_door_loc) >= 3:
            nearest_door_loc_text = f"({float(nearest_door_loc[0]):.1f}, {float(nearest_door_loc[1]):.1f}, {float(nearest_door_loc[2]):.1f})"
        response_text = last_response if len(last_response) <= 60 else (last_response[:57] + "...")
        self.door_var.set(
            "Door "
            f"supported={int(bool(door_control.get('supported', False)))} "
            f"mode={str(door_control.get('mode', 'unknown') or 'unknown')} "
            f"player={control_player_name or 'none'} "
            f"nearest={nearest_door_text} "
            f"xyz={nearest_door_loc_text} "
            f"last_target={last_target_name or 'none'} "
            f"method={last_method or 'none'} "
            f"window={last_window_title or 'none'} "
            f"ok={str(last_ok)} "
            f"resp={response_text or '-'} "
            f"last={str(door_control.get('last_requested_label', 'unknown') or 'unknown')} "
            f"time={str(door_control.get('last_control_time', '') or '-')}"
        )
        memory_collection = state.get("memory_collection", {}) if isinstance(state.get("memory_collection"), dict) else {}
        self.latest_memory_collection_state = dict(memory_collection)
        episode_id = str(memory_collection.get("episode_id", "") or "")
        episode_label = str(memory_collection.get("episode_label", "") or "")
        collection_dir = str(memory_collection.get("collection_dir", "") or "")
        collection_dir_name = os.path.basename(collection_dir) if collection_dir else "none"
        current_step_index = int(memory_collection.get("step_index", 0) or 0)
        self.memory_step_display_var.set(str(current_step_index))
        if episode_id and episode_id != self.memory_auto_episode_id:
            self._reset_memory_auto_tracking(episode_id, current_step_index)
        self._refresh_memory_auto_status()
        self.memory_var.set(
            "Memory "
            f"active={int(bool(memory_collection.get('active', False)))} "
            f"episode={episode_id or 'none'} "
            f"label={episode_label or '-'} "
            f"step={current_step_index} "
            f"snapshots={int(memory_collection.get('snapshot_count', 0) or 0)} "
            f"target={str(memory_collection.get('current_target_house_id', '') or 'none')} "
            f"current={str(memory_collection.get('current_house_id', '') or 'none')} "
            f"dir={collection_dir_name}"
        )
        last_capture = state.get("last_capture", {}) if isinstance(state.get("last_capture"), dict) else {}
        self.capture_var.set(
            "Capture "
            f"last={str(last_capture.get('capture_id', 'none') or 'none')} "
            f"source={str(last_capture.get('capture_source', '-') or '-')} "
            f"step={str(last_capture.get('step_index', '-') or '-')}"
        )
        self.mission_var.set(f"Mission: current={mission.get('current_house_name') or 'none'} target={mission.get('target_house_name') or 'none'}")
        self.current_house_var.set(f"Current house: {mission.get('current_house_name') or 'none'} [{mission.get('current_house_status') or '-'}]")
        self.target_house_var.set(f"Target: {mission.get('target_house_name') or 'none'} [{mission.get('target_house_status') or '-'}]  next={mission.get('nearest_unsearched_house_name') or 'none'}")
        mission_target_house_id = str(mission.get("target_house_id", "") or "").strip()
        current_selected_house_id = self._get_selected_target_house_id()
        if mission_target_house_id and current_selected_house_id == mission_target_house_id:
            self._set_selected_target_house(mission_target_house_id, mark_clean=True)
        elif not self.target_house_selection_dirty:
            self._set_selected_target_house(mission_target_house_id, mark_clean=True)
        dist = mission.get("distance_to_target_cm")
        self.target_dist_var.set("Target distance: n/a" if dist is None else f"Target distance: {float(dist):.1f} cm")
        if self.movement_toggle_button is not None:
            self.movement_toggle_button.configure(text="Disable Basic Movement" if self.movement_enabled_state else "Enable Basic Movement")
        if self.memory_window is not None and self.memory_window.winfo_exists() and self.memory_auto_refresh_var.get():
            self.refresh_memory_window()

    def _format_memory_payload(self, payload: Dict[str, Any]) -> str:
        memory_collection = payload.get("memory_collection", {}) if isinstance(payload.get("memory_collection"), dict) else {}
        house_mission = payload.get("house_mission", {}) if isinstance(payload.get("house_mission"), dict) else {}
        memory_store = payload.get("memory_store", {}) if isinstance(payload.get("memory_store"), dict) else {}
        target_house_memory = payload.get("target_house_memory", {}) if isinstance(payload.get("target_house_memory"), dict) else {}
        current_house_memory = payload.get("current_house_memory", {}) if isinstance(payload.get("current_house_memory"), dict) else {}

        lines: List[str] = []
        lines.append("=== Memory Collection ===")
        lines.append(f"active: {bool(memory_collection.get('active', False))}")
        lines.append(f"episode_id: {memory_collection.get('episode_id', '')}")
        lines.append(f"episode_label: {memory_collection.get('episode_label', '')}")
        lines.append(f"collection_dir: {memory_collection.get('collection_dir', '')}")
        lines.append(f"step_index: {int(memory_collection.get('step_index', 0) or 0)}")
        lines.append(f"snapshot_count: {int(memory_collection.get('snapshot_count', 0) or 0)}")
        lines.append(f"current_target_house_id: {memory_collection.get('current_target_house_id', '')}")
        lines.append(f"current_house_id: {memory_collection.get('current_house_id', '')}")
        lines.append(f"capture_root: {memory_collection.get('capture_root', '')}")
        lines.append(f"last_snapshot_before_path: {memory_collection.get('last_snapshot_before_path', '')}")
        lines.append(f"last_snapshot_after_path: {memory_collection.get('last_snapshot_after_path', '')}")
        lines.append(f"last_capture_run_dir: {memory_collection.get('last_capture_run_dir', '')}")
        lines.append(f"last_capture_label: {memory_collection.get('last_capture_label', '')}")
        lines.append(f"last_capture_source: {memory_collection.get('last_capture_source', '')}")
        lines.append(f"last_capture_time: {memory_collection.get('last_capture_time', '')}")
        lines.append("")
        lines.append("=== Local Auto Capture ===")
        lines.append(f"enabled: {bool(self.memory_auto_enabled)}")
        lines.append(f"mode: {self.memory_auto_mode_var.get().strip() or 'off'}")
        lines.append(f"time_seconds: {self.memory_auto_seconds_var.get().strip() or '0'}")
        lines.append(f"step_interval: {self.memory_auto_steps_var.get().strip() or '0'}")
        lines.append("")
        lines.append("=== Mission Context ===")
        lines.append(f"target_house_id: {house_mission.get('target_house_id', '')}")
        lines.append(f"target_house_name: {house_mission.get('target_house_name', '')}")
        lines.append(f"current_house_id: {house_mission.get('current_house_id', '')}")
        lines.append(f"current_house_name: {house_mission.get('current_house_name', '')}")
        lines.append("")
        lines.append("=== Target House Memory ===")
        lines.append(json.dumps(target_house_memory, indent=2, ensure_ascii=False))
        lines.append("")
        lines.append("=== Current House Memory ===")
        lines.append(json.dumps(current_house_memory, indent=2, ensure_ascii=False))
        lines.append("")
        lines.append("=== Memory Store Meta ===")
        lines.append(json.dumps({
            "memory_store_path": payload.get("memory_store_path", ""),
            "version": memory_store.get("version", ""),
            "updated_at": memory_store.get("updated_at", ""),
            "current_target_house_id": memory_store.get("current_target_house_id", ""),
            "memory_house_count": len(memory_store.get("memories", {})) if isinstance(memory_store.get("memories"), dict) else 0,
        }, indent=2, ensure_ascii=False))
        return "\n".join(lines)

    def refresh_memory_window(self) -> None:
        if self.memory_refresh_inflight:
            return
        if self.memory_window is None or not self.memory_window.winfo_exists() or self.memory_text is None:
            return
        def worker():
            self.memory_refresh_inflight = True
            try:
                payload = self.safe(self.client.get_json, "/memory_state", label="Refresh Memory State")
                if not isinstance(payload, dict):
                    return
                text = self._format_memory_payload(payload)
                def apply_text() -> None:
                    if self.memory_text is None:
                        return
                    self.memory_text.configure(state="normal")
                    self.memory_text.delete("1.0", "end")
                    self.memory_text.insert("1.0", text)
                    self.memory_text.configure(state="disabled")
                self.root.after(0, apply_text)
            finally:
                self.memory_refresh_inflight = False
        threading.Thread(target=worker, daemon=True).start()

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

    def _start_memory_auto_capture_local(self) -> None:
        memory_collection = self.latest_memory_collection_state if isinstance(self.latest_memory_collection_state, dict) else {}
        current_step = int(memory_collection.get("step_index", 0) or 0)
        episode_id = str(memory_collection.get("episode_id", "") or "")
        self.memory_auto_enabled = True
        self._reset_memory_auto_tracking(episode_id, current_step)
        mode = self.memory_auto_mode_var.get().strip() or "off"
        self._refresh_memory_auto_status()
        self.status_var.set(f"Memory auto capture started: mode={mode}")

    def _stop_memory_auto_capture_local(self) -> None:
        self.memory_auto_enabled = False
        self._refresh_memory_auto_status()
        self.status_var.set("Memory auto capture stopped.")

    def _maybe_trigger_memory_auto_capture(self) -> None:
        if not self.memory_auto_enabled or self.memory_capture_inflight:
            return
        memory_collection = self.latest_memory_collection_state if isinstance(self.latest_memory_collection_state, dict) else {}
        if not bool(memory_collection.get("active", False)):
            return
        episode_id = str(memory_collection.get("episode_id", "") or "")
        current_step = int(memory_collection.get("step_index", 0) or 0)
        if episode_id and episode_id != self.memory_auto_episode_id:
            self._reset_memory_auto_tracking(episode_id, current_step)

        mode = self.memory_auto_mode_var.get().strip().lower()
        if mode == "time":
            try:
                interval_s = max(1.0, float(self.memory_auto_seconds_var.get().strip()))
            except ValueError:
                return
            if (time.time() - float(self.memory_auto_last_capture_time)) < interval_s:
                return
            threading.Thread(
                target=lambda: self._run_memory_capture_analyze(
                    capture_source="auto_time",
                    note=f"auto_time interval_s={interval_s:.2f}",
                    update_status=False,
                ),
                daemon=True,
            ).start()
        elif mode == "step":
            try:
                step_interval = max(1, int(float(self.memory_auto_steps_var.get().strip())))
            except ValueError:
                return
            if current_step - int(self.memory_auto_last_capture_step) < step_interval:
                return
            threading.Thread(
                target=lambda: self._run_memory_capture_analyze(
                    capture_source="auto_step",
                    note=f"auto_step interval={step_interval}",
                    update_status=False,
                ),
                daemon=True,
            ).start()

    def schedule_memory_auto_capture(self) -> None:
        self._maybe_trigger_memory_auto_capture()
        self.root.after(500, self.schedule_memory_auto_capture)

    def _run_memory_capture_analyze(self, *, capture_source: str, note: str, update_status: bool) -> bool:
        if self.memory_capture_inflight:
            return False
        self.memory_capture_inflight = True
        self.root.after(0, self._refresh_memory_auto_status)
        try:
            response = self.safe(
                self.client.post_json,
                "/memory_capture_analyze",
                {
                    "label": self.memory_capture_label_var.get().strip(),
                    "capture_source": str(capture_source or "manual"),
                    "note": str(note or ""),
                },
                label=f"Memory Capture {capture_source}",
                timeout_s=max(float(self.args.timeout_s), 180.0),
            ) or {}
            if not isinstance(response, dict):
                return False
            ok = str(response.get("status", "")) == "ok"
            state = response.get("state")
            result = response.get("result")
            if isinstance(state, dict):
                self.root.after(0, lambda s=state: self.apply_state(s))
                memory_collection = state.get("memory_collection", {}) if isinstance(state.get("memory_collection"), dict) else {}
                self.memory_auto_last_capture_step = int(memory_collection.get("step_index", self.memory_auto_last_capture_step) or self.memory_auto_last_capture_step)
                self.memory_auto_episode_id = str(memory_collection.get("episode_id", self.memory_auto_episode_id) or self.memory_auto_episode_id)
            if ok:
                self.memory_auto_last_capture_time = time.time()
                self.root.after(0, self._refresh_memory_auto_status)
            if isinstance(result, dict):
                self.root.after(0, lambda r=result: self.apply_fusion_result(r))
            run_dir = str(response.get("run_dir", "") or "")
            if update_status:
                status_message = str(response.get("message", "") or "")
                if ok:
                    self.root.after(0, lambda rd=run_dir: self.status_var.set(f"Memory capture analyze saved: {rd or 'done'}"))
                else:
                    self.root.after(0, lambda msg=status_message: self.status_var.set(msg or "Memory capture analyze failed."))
            return ok
        finally:
            self.memory_capture_inflight = False
            self.root.after(0, self._refresh_memory_auto_status)

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

    def on_set_task(self) -> None:
        task_label = self.task_label_var.get().strip()
        selected_house_id = self._get_selected_target_house_id()

        def worker() -> None:
            if selected_house_id:
                select_resp = self.safe(
                    self.client.post_json,
                    "/select_target_house",
                    {"house_id": selected_house_id},
                    label=f"Select Target House {selected_house_id}",
                )
                if isinstance(select_resp, dict) and select_resp.get("status") == "ok":
                    self._update_local_current_target_id(selected_house_id)
                    self.root.after(0, lambda hid=selected_house_id: self._set_selected_target_house(hid, mark_clean=True))
            task_resp = self.safe(
                self.client.post_json,
                "/task",
                {"task_label": task_label},
                label="Set Task",
            )
            self._apply_response(task_resp)

        self.call_async("Setting task", worker)
    def on_memory_collection_start(self) -> None:
        self.call_async(
            "Starting memory collection",
            lambda: self._apply_response_and_reset_memory_auto(
                self.safe(
                    self.client.post_json,
                    "/memory_collection_start",
                    {
                        "episode_label": self.memory_episode_label_var.get().strip(),
                        "reset_store": bool(self.memory_reset_on_start_var.get()),
                    },
                    label="Start Memory Collection",
                )
            ),
        )
    def on_memory_collection_stop(self) -> None:
        self._stop_memory_auto_capture_local()
        self.call_async(
            "Stopping memory collection",
            lambda: self._apply_response(
                self.safe(self.client.post_json, "/memory_collection_stop", {}, label="Stop Memory Collection")
            ),
        )
    def on_memory_collection_reset(self) -> None:
        self._stop_memory_auto_capture_local()
        self.call_async(
            "Resetting memory store",
            lambda: self._apply_response(
                self.safe(self.client.post_json, "/memory_collection_reset", {}, label="Reset Memory Collection")
            ),
        )
    def on_memory_snapshot(self) -> None:
        self.call_async(
            "Saving memory snapshot",
            lambda: self._apply_response(
                self.safe(
                    self.client.post_json,
                    "/memory_snapshot",
                    {"snapshot_type": "manual", "note": "Manual snapshot from panel"},
                    label="Save Memory Snapshot",
                )
            ),
        )
    def _send_memory_snapshot_quiet(self, snapshot_type: str, note: str) -> None:
        self.safe(
            self.client.post_json,
            "/memory_snapshot",
            {"snapshot_type": snapshot_type, "note": note},
            label=f"Memory Snapshot {snapshot_type}",
        )
    def _apply_response_and_reset_memory_auto(self, response: Any) -> None:
        self._apply_response(response)
        memory_collection = self.latest_memory_collection_state if isinstance(self.latest_memory_collection_state, dict) else {}
        self._reset_memory_auto_tracking(
            str(memory_collection.get("episode_id", "") or ""),
            int(memory_collection.get("step_index", 0) or 0),
        )

    def on_memory_capture_analyze(self) -> None:
        self.call_async(
            "Memory capture analyze",
            lambda: self._run_memory_capture_analyze(
                capture_source="manual",
                note="manual_step_capture_from_panel",
                update_status=True,
            ),
        )

    def on_memory_auto_capture_start(self) -> None:
        memory_collection = self.latest_memory_collection_state if isinstance(self.latest_memory_collection_state, dict) else {}
        if not bool(memory_collection.get("active", False)):
            self.status_var.set("Start Episode first, then start auto capture.")
            return
        mode = self.memory_auto_mode_var.get().strip().lower()
        if mode not in {"time", "step"}:
            self.status_var.set("Auto capture mode must be 'time' or 'step'.")
            return
        if mode == "time":
            try:
                if float(self.memory_auto_seconds_var.get().strip()) <= 0:
                    raise ValueError
            except ValueError:
                self.status_var.set("Invalid auto capture time interval.")
                return
        if mode == "step":
            try:
                if int(float(self.memory_auto_steps_var.get().strip())) <= 0:
                    raise ValueError
            except ValueError:
                self.status_var.set("Invalid auto capture step interval.")
                return
        self._start_memory_auto_capture_local()

    def on_memory_auto_capture_stop(self) -> None:
        self._stop_memory_auto_capture_local()

    def on_capture(self) -> None:
        self.on_memory_capture_analyze()

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
    def on_door_open(self) -> None: self.call_async("Opening door", lambda: self._apply_response(self.safe(self.client.post_json, "/door_open", {}, label="Open Door")))
    def on_door_close(self) -> None: self.call_async("Closing door", lambda: self._apply_response(self.safe(self.client.post_json, "/door_close", {}, label="Close Door")))
    def on_door_toggle(self) -> None: self.call_async("Toggling door", lambda: self._apply_response(self.safe(self.client.post_json, "/door_toggle", {}, label="Toggle Door")))
    def on_door_interact_e(self) -> None: self.call_async("Sending E interaction", lambda: self._apply_response(self.safe(self.client.post_json, "/door_interact_e", {}, label="Door Interact E")))

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
            sequence_id = f"seq_{int(time.time())}"
            self._send_memory_snapshot_quiet("sequence_start", f"sequence_id={sequence_id} symbols={''.join(symbols)} delay_s={delay_s:.3f}")
            for i, s in enumerate(symbols, start=1):
                if self.sequence_stop_event.is_set():
                    self._send_memory_snapshot_quiet("sequence_stop", f"sequence_id={sequence_id} stopped_at={i-1}/{total}")
                    self.root.after(0, lambda i=i, t=total: self.status_var.set(f"Sequence stopped at step {i-1}/{t}."))
                    return
                while self.move_request_inflight and not self.sequence_stop_event.is_set(): time.sleep(0.02)
                self.root.after(0, lambda i=i, t=total, s=s: self.status_var.set(f"Sequence step {i}/{t}: {s}"))
                if not self._execute_move(s, from_sequence=True):
                    self._send_memory_snapshot_quiet("sequence_failed", f"sequence_id={sequence_id} failed_at={i}/{total}")
                    self.root.after(0, lambda i=i, t=total: self.status_var.set(f"Sequence failed at step {i}/{t}."))
                    return
                time.sleep(delay_s)
            self._send_memory_snapshot_quiet("sequence_end", f"sequence_id={sequence_id} completed={total}/{total}")
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

    def toggle_memory_window(self) -> None:
        if self.memory_window and self.memory_window.winfo_exists():
            self._close_memory_window()
            return
        self.memory_window = tk.Toplevel(self.root)
        self.memory_window.title("Memory Collection Inspector")
        self.memory_window.geometry("980x760")
        self.memory_window.protocol("WM_DELETE_WINDOW", self._close_memory_window)
        toolbar = tk.Frame(self.memory_window)
        toolbar.pack(fill="x", padx=8, pady=(8, 4))
        tk.Button(toolbar, text="Refresh", command=self.refresh_memory_window).pack(side="left", padx=(0, 6))
        tk.Button(toolbar, text="Snapshot Now", command=self.on_memory_snapshot).pack(side="left", padx=(0, 6))
        tk.Checkbutton(toolbar, text="Auto Refresh", variable=self.memory_auto_refresh_var).pack(side="left", padx=(6, 0))
        self.memory_text = tk.Text(self.memory_window, wrap="none", font=("Consolas", 10))
        self.memory_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.memory_text.configure(state="disabled")
        self.refresh_memory_window()

    def _close_memory_window(self) -> None:
        try:
            if self.memory_window is not None and self.memory_window.winfo_exists():
                self.memory_window.destroy()
        except Exception:
            pass
        self.memory_window = None
        self.memory_text = None

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

    def toggle_result_reviewer_window(self) -> None:
        if self.review_window and self.review_window.winfo_exists():
            self._close_result_reviewer_window()
            return
        self._refresh_target_house_choices()
        self.review_window = tk.Toplevel(self.root)
        self.review_window.title("Fusion Result Reviewer")
        self.review_window.geometry("1180x900")
        self.review_window.protocol("WM_DELETE_WINDOW", self._close_result_reviewer_window)

        top_bar = tk.Frame(self.review_window)
        top_bar.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(top_bar, textvariable=self.review_status_var, anchor="w", font=("Consolas", 11, "bold")).pack(side="left")
        tk.Button(top_bar, text="Refresh Queue", command=self._refresh_review_queue).pack(side="right", padx=(6, 0))

        info_bar = tk.Frame(self.review_window)
        info_bar.pack(fill="x", padx=8, pady=(0, 4))
        tk.Label(info_bar, textvariable=self.review_info_var, anchor="w", justify="left", wraplength=1140).pack(fill="x")

        choice_bar = tk.Frame(self.review_window)
        choice_bar.pack(fill="x", padx=8, pady=(0, 6))
        tk.Label(choice_bar, text="Correct House ID").pack(side="left")
        review_combo = ttk.Combobox(
            choice_bar,
            textvariable=self.review_house_var,
            state="readonly",
            width=28,
            values=list(self.target_house_choice_map.keys()),
        )
        review_combo.pack(side="left", padx=(6, 12))
        tk.Button(choice_bar, text="Correct", command=self.on_review_mark_correct, width=14).pack(side="left", padx=(0, 6))
        tk.Button(choice_bar, text="Save Corrected", command=self.on_review_mark_corrected, width=16).pack(side="left", padx=(0, 6))
        tk.Button(choice_bar, text="Skip", command=self._advance_review_item).pack(side="left")

        self.review_image_label = tk.Label(self.review_window, text="Loading review sample...", anchor="center")
        self.review_image_label.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._refresh_review_queue()

    def toggle_indicator_reviewer_window(self) -> None:
        if self.indicator_review_window and self.indicator_review_window.winfo_exists():
            self._close_indicator_reviewer_window()
            return
        self._refresh_target_house_choices()
        self.indicator_review_window = tk.Toplevel(self.root)
        self.indicator_review_window.title("Fusion Indicator Reviewer")
        self.indicator_review_window.geometry("1240x980")
        self.indicator_review_window.protocol("WM_DELETE_WINDOW", self._close_indicator_reviewer_window)

        top_bar = tk.Frame(self.indicator_review_window)
        top_bar.pack(fill="x", padx=8, pady=(8, 4))
        tk.Label(top_bar, textvariable=self.indicator_review_status_var, anchor="w", font=("Consolas", 11, "bold")).pack(side="left")
        tk.Button(top_bar, text="Refresh Queue", command=self._refresh_indicator_review_queue).pack(side="right", padx=(6, 0))

        info_bar = tk.Frame(self.indicator_review_window)
        info_bar.pack(fill="x", padx=8, pady=(0, 4))
        tk.Label(info_bar, textvariable=self.indicator_review_info_var, anchor="w", justify="left", wraplength=1200).pack(fill="x")

        choice_bar = tk.Frame(self.indicator_review_window)
        choice_bar.pack(fill="x", padx=8, pady=(0, 6))

        tk.Label(choice_bar, text="House ID").grid(row=0, column=0, sticky="w", padx=(0, 6), pady=4)
        indicator_house_combo = ttk.Combobox(
            choice_bar,
            textvariable=self.indicator_review_house_var,
            state="readonly",
            width=28,
            values=[""] + list(self.target_house_choice_map.keys()),
        )
        indicator_house_combo.grid(row=0, column=1, sticky="w", padx=(0, 12), pady=4)

        tk.Label(choice_bar, text="Target State").grid(row=1, column=0, sticky="w", padx=(0, 6), pady=4)
        indicator_state_combo = ttk.Combobox(
            choice_bar,
            textvariable=self.indicator_review_state_var,
            state="readonly",
            width=42,
            values=[""] + TARGET_CONDITIONED_STATE_OPTIONS,
        )
        indicator_state_combo.grid(row=1, column=1, sticky="w", padx=(0, 12), pady=4)
        tk.Label(
            choice_bar,
            textvariable=self.indicator_review_state_note_var,
            anchor="w",
            justify="left",
            wraplength=420,
            foreground="#2f4f4f",
        ).grid(row=1, column=2, sticky="w", padx=(0, 12), pady=4)

        tk.Label(choice_bar, text="Target Subgoal").grid(row=2, column=0, sticky="w", padx=(0, 6), pady=4)
        indicator_subgoal_combo = ttk.Combobox(
            choice_bar,
            textvariable=self.indicator_review_subgoal_var,
            state="readonly",
            width=42,
            values=[""] + TARGET_CONDITIONED_SUBGOAL_OPTIONS,
        )
        indicator_subgoal_combo.grid(row=2, column=1, sticky="w", padx=(0, 12), pady=4)
        tk.Label(
            choice_bar,
            textvariable=self.indicator_review_subgoal_note_var,
            anchor="w",
            justify="left",
            wraplength=420,
            foreground="#2f4f4f",
        ).grid(row=2, column=2, sticky="w", padx=(0, 12), pady=4)

        tk.Label(choice_bar, text="Target Action").grid(row=3, column=0, sticky="w", padx=(0, 6), pady=4)
        indicator_action_combo = ttk.Combobox(
            choice_bar,
            textvariable=self.indicator_review_action_var,
            state="readonly",
            width=28,
            values=[""] + TARGET_CONDITIONED_ACTION_OPTIONS,
        )
        indicator_action_combo.grid(row=3, column=1, sticky="w", padx=(0, 12), pady=4)
        tk.Label(
            choice_bar,
            textvariable=self.indicator_review_action_note_var,
            anchor="w",
            justify="left",
            wraplength=420,
            foreground="#2f4f4f",
        ).grid(row=3, column=2, sticky="w", padx=(0, 12), pady=4)

        tk.Button(choice_bar, text="Correct", command=self.on_indicator_review_mark_correct, width=14).grid(row=0, column=3, padx=(12, 6), pady=4, sticky="w")
        tk.Button(choice_bar, text="Save Corrected", command=self.on_indicator_review_mark_corrected, width=16).grid(row=1, column=3, padx=(12, 6), pady=4, sticky="w")
        tk.Button(choice_bar, text="Skip", command=self._advance_indicator_review_item).grid(row=2, column=3, padx=(12, 6), pady=4, sticky="w")

        self.indicator_review_image_label = tk.Label(self.indicator_review_window, text="Loading indicator review sample...", anchor="center")
        self.indicator_review_image_label.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._refresh_indicator_review_queue()

    def _close_result_reviewer_window(self) -> None:
        try:
            if self.review_window is not None and self.review_window.winfo_exists():
                self.review_window.destroy()
        except Exception:
            pass
        self.review_window = None
        self.review_image_label = None
        self.review_image_photo = None
        self.review_current_item = None

    def _close_indicator_reviewer_window(self) -> None:
        try:
            if self.indicator_review_window is not None and self.indicator_review_window.winfo_exists():
                self.indicator_review_window.destroy()
        except Exception:
            pass
        self.indicator_review_window = None
        self.indicator_review_image_label = None
        self.indicator_review_image_photo = None
        self.indicator_review_current_item = None

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
            self.root.after(0, self._refresh_target_house_choices)
            self.refresh_map_async()

        threading.Thread(target=worker, daemon=True).start()

    def on_close(self) -> None:
        for window in (self.preview_window, self.depth_window, self.memory_window, self.fusion_window, self.review_window, self.indicator_review_window, self.map_window, self.open_map_window):
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
