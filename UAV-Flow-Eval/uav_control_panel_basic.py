from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
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
YAW_IMMEDIATE_CAPTURE_SYMBOLS = {"q", "e"}
LLM_CONTROL_CONTINUOUS_SYMBOLS = {"w", "a", "s", "d", "r", "f"}
LLM_CONTROL_SINGLE_SYMBOLS = {"q", "e", "x"}
LLM_CONTROL_ALLOWED_SYMBOLS = LLM_CONTROL_CONTINUOUS_SYMBOLS | LLM_CONTROL_SINGLE_SYMBOLS
LLM_CONTROL_ACTION_ALIASES: Dict[str, str] = {
    "forward": "w",
    "backward": "s",
    "left": "a",
    "right": "d",
    "up": "r",
    "down": "f",
    "yaw_left": "q",
    "turn_left": "q",
    "yaw_right": "e",
    "turn_right": "e",
    "hold": "x",
    "stop": "x",
}
LLM_CONTROL_OUTPUT_SCHEMA: Dict[str, Any] = {
    "action_symbol": "w",
    "repeat": 2,
    "stop": False,
    "need_capture_after": True,
    "confidence": 0.85,
    "reason": "Short memory-aware control rationale.",
}
LLM_ENTRY_REACHED_DISTANCE_CM = 300.0
LLM_APPROACHABLE_DISTANCE_MARGIN_CM = 320.0
LLM_ENTRY_REACHED_POSE_PROXY_CM = 350.0
LLM_ENTRY_REACHED_LARGE_BBOX_PROXY_CM = 520.0
LLM_LARGE_ENTRY_BBOX_AREA_RATIO = 0.42
LLM_LARGE_ENTRY_BBOX_HEIGHT_RATIO = 0.82
LLM_LARGE_ENTRY_BBOX_WIDTH_RATIO = 0.68
LLM_YAW_STEP_DEG = 30.0
LLM_TARGET_REACQUIRE_ALIGN_TOLERANCE_DEG = 18.0
LLM_TARGET_REACQUIRE_MAX_YAW_STEPS = 12
LLM_TARGET_REACQUIRE_DEFAULT_YAW_SYMBOL = "q"
LLM_OBSTACLE_FRONT_MIN_BLOCK_CM = 120.0
LLM_OBSTACLE_FRONT_MIN_BACKOFF_CM = 90.0
LLM_OBSTACLE_DETOUR_SIDE_STEPS = 3
LLM_OBSTACLE_DETOUR_MAX_STEPS = 10
LLM_OBSTACLE_DEFAULT_DETOUR_SYMBOL = "a"
LLM_TARGET_BOUNDARY_MARGIN_CM = 800.0
LLM_TARGET_BOUNDARY_MIN_SEARCH_DISTANCE_CM = 1800.0
LLM_TARGET_BOUNDARY_ALIGN_TOLERANCE_DEG = 18.0
LLM_TARGET_TRANSIT_FRONT_MIN_FORWARD_CM = 160.0
LLM_AXIS_TRANSIT_ALIGN_TOLERANCE_DEG = 12.0
LLM_AXIS_TRANSIT_BOUNDARY_MARGIN_CM = 180.0
LLM_AXIS_TRANSIT_WAYPOINT_TOLERANCE_CM = 220.0
LLM_AXIS_TRANSIT_HOUSE_CLEARANCE_CM = 180.0
LLM_DEPTH_MAX_VALID_CM = 1200.0
LLM_DEPTH_CORRIDOR_X_RANGE = (0.35, 0.65)
LLM_DEPTH_BODY_Y_RANGE = (0.35, 0.60)
LLM_DEPTH_LOW_Y_RANGE = (0.60, 0.90)
LLM_DEPTH_UPPER_Y_RANGE = (0.10, 0.35)
LLM_LOW_OBSTACLE_NEAR_CM = 160.0
LLM_BODY_CORRIDOR_CLEAR_P10_CM = 220.0
LLM_BODY_CORRIDOR_MIN_CLEAR_CM = 180.0
LLM_BODY_CLOSE_RATIO_MAX = 0.03
LLM_LOW_CLOSE_RATIO_MIN = 0.08
LLM_CONTROL_LABELING_INPUT_FILES = (
    "fusion_result.json",
    "sample_metadata.json",
    "entry_search_memory_snapshot_after.json",
    "temporal_context.json",
    "action_history_since_last_capture.json",
    "pose_history_summary.json",
    "yolo_result.json",
    "depth_result.json",
    "fusion_summary.txt",
    "labeling_summary.txt",
    "yolo_summary.txt",
    "depth_summary.txt",
    "depth_cm.png",
    "depth_overlay.png",
    "fusion_overlay.png",
)
LLM_TASK_PLAN_OUTPUT_SCHEMA: Dict[str, Any] = {
    "plan_id": "llm_task_plan_20260428_123000",
    "task_text": "先探索 house 1，再探索 house 3。",
    "ordered_targets": [
        {
            "order": 1,
            "house_id": "001",
            "house_alias": "house 1",
            "goal": "search_entry",
            "finish_condition": "target_entry_reached_or_no_entry_after_full_coverage",
            "status": "pending",
        }
    ],
    "execution_policy": {
        "entry_reached_distance_cm": LLM_ENTRY_REACHED_DISTANCE_CM,
        "entry_reached_pose_proxy_cm": LLM_ENTRY_REACHED_POSE_PROXY_CM,
        "entry_reached_large_bbox_proxy_cm": LLM_ENTRY_REACHED_LARGE_BBOX_PROXY_CM,
        "max_decisions_per_house": 40,
        "allow_no_entry_completion": True,
        "stop_on_needs_review": True,
    },
    "reason": "Short plan rationale.",
}
DOORLIKE_CLASS_NAMES = {"door", "open door", "open_door", "close door", "close_door", "closed door", "closed_door"}

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
        self.memory_auto_next_var = tk.StringVar(value="Next Auto: off")
        self.llm_control_base_url_var = tk.StringVar(value=os.environ.get("ANTHROPIC_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE", ""))
        self.llm_control_api_key_var = tk.StringVar(value=os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("OPENAI_API_KEY", ""))
        self.llm_control_model_var = tk.StringVar(value=os.environ.get("ANTHROPIC_MODEL", "gpt-5.5"))
        self.llm_control_max_steps_var = tk.StringVar(value="20")
        self.llm_control_repeat_cap_var = tk.StringVar(value="4")
        self.llm_control_delay_ms_var = tk.StringVar(value="350")
        self.llm_control_timeout_s_var = tk.StringVar(value="60")
        self.llm_control_status_var = tk.StringVar(value="LLM Control: idle")
        self.llm_control_step_var = tk.StringVar(value="0/0")
        self.llm_control_last_action_var = tk.StringVar(value="Last action: none")
        self.llm_control_last_capture_var = tk.StringVar(value="Last capture: none")
        self.llm_control_reason_var = tk.StringVar(value="Reason: -")
        self.llm_task_text_var = tk.StringVar(value="先探索 house 1，再探索 house 3。")
        self.llm_task_status_var = tk.StringVar(value="LLM Task: no plan")
        self.llm_task_plan_summary_var = tk.StringVar(value="Plan: none")
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
        self.llm_control_window: Optional[tk.Toplevel] = None
        self.llm_control_log_text: Optional[tk.Text] = None
        self.llm_task_window: Optional[tk.Toplevel] = None
        self.llm_task_text: Optional[tk.Text] = None
        self.llm_task_preview_text: Optional[tk.Text] = None
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
        self.last_memory_capture_response: Dict[str, Any] = {}
        self.memory_auto_enabled = False
        self.memory_auto_last_capture_time = 0.0
        self.memory_auto_last_capture_step = 0
        self.memory_auto_episode_id = ""
        self.llm_control_thread: Optional[threading.Thread] = None
        self.llm_control_stop_event = threading.Event()
        self.llm_control_running = False
        self.llm_control_decision_history: List[Dict[str, Any]] = []
        self.llm_task_plan: Dict[str, Any] = {}
        self.llm_task_plan_applied = False
        self.llm_task_current_index = 0
        self.llm_task_execution_trace: Dict[str, Any] = {}
        self.llm_target_reacquire_lock: Dict[str, Any] = {}
        self.llm_obstacle_detour_lock: Dict[str, Any] = {}

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
        tk.Label(memory, text="Next Auto In").grid(row=6, column=0, sticky="w", padx=6, pady=(0, 6))
        tk.Label(
            memory,
            textvariable=self.memory_auto_next_var,
            anchor="w",
            justify="left",
        ).grid(row=6, column=1, columnspan=3, sticky="ew", padx=6, pady=(0, 6))
        tk.Label(
            memory,
            textvariable=self.memory_auto_status_var,
            anchor="w",
            justify="left",
        ).grid(row=7, column=0, columnspan=4, sticky="ew", padx=6, pady=(0, 6))
        tk.Button(memory, text="Open LLM Control", command=self.toggle_llm_control_window).grid(row=8, column=0, columnspan=2, padx=6, pady=(0, 6), sticky="ew")
        tk.Button(memory, text="Analyze LLM Task", command=self.toggle_llm_task_window).grid(row=8, column=2, columnspan=2, padx=6, pady=(0, 6), sticky="ew")
        tk.Label(memory, textvariable=self.llm_task_plan_summary_var, anchor="w", justify="left").grid(row=9, column=0, columnspan=4, sticky="ew", padx=6, pady=(0, 6))

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
        if self.llm_control_window is not None and self.llm_control_window.winfo_exists() and toplevel == self.llm_control_window and self.llm_control_log_text is not None:
            return ("llm_control_log", self.llm_control_log_text)
        if self.llm_task_window is not None and self.llm_task_window.winfo_exists() and toplevel == self.llm_task_window and self.llm_task_preview_text is not None:
            return ("llm_task_preview", self.llm_task_preview_text)
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
            self.memory_auto_next_var.set("Next Auto: off")
            self.memory_auto_status_var.set(
                f"Auto Capture: off | mode={mode} | episode={episode_id or 'none'} | step={current_step}"
            )
            return
        next_text = "Next Auto: n/a"
        if mode == "time":
            interval_text = self.memory_auto_seconds_var.get().strip() or "0"
            rule_text = f"every {interval_text}s"
            last_text = datetime.fromtimestamp(float(self.memory_auto_last_capture_time)).strftime("%H:%M:%S") if self.memory_auto_last_capture_time else "n/a"
            try:
                interval_s = max(1.0, float(interval_text))
                elapsed_s = max(0.0, time.time() - float(self.memory_auto_last_capture_time or time.time()))
                remaining_s = max(0.0, interval_s - elapsed_s)
                next_text = f"Next Auto: {remaining_s:.1f}s"
            except ValueError:
                next_text = "Next Auto: invalid time interval"
        elif mode == "step":
            interval_text = self.memory_auto_steps_var.get().strip() or "0"
            rule_text = f"every {interval_text} step(s)"
            last_text = f"step {int(self.memory_auto_last_capture_step)}"
            try:
                step_interval = max(1, int(float(interval_text)))
                steps_since = max(0, current_step - int(self.memory_auto_last_capture_step))
                remaining_steps = max(0, step_interval - steps_since)
                next_text = f"Next Auto: {remaining_steps} step(s) remaining ({steps_since}/{step_interval})"
            except ValueError:
                next_text = "Next Auto: invalid step interval"
        else:
            rule_text = "mode=off"
            last_text = "n/a"
            next_text = "Next Auto: off"
        state_text = "capturing" if self.memory_capture_inflight else "running"
        if self.memory_capture_inflight:
            next_text = "Next Auto: capturing now"
        self.memory_auto_next_var.set(next_text)
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
        lines.append(f"next_auto: {self.memory_auto_next_var.get()}")
        lines.append(f"server_action_count: {int(memory_collection.get('action_count', 0) or 0)}")
        lines.append(f"server_actions_since_last_capture: {int(memory_collection.get('actions_since_last_capture', 0) or 0)}")
        lines.append(f"server_action_log_path: {memory_collection.get('action_log_path', '')}")
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
        self._refresh_memory_auto_status()
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
            self.last_memory_capture_response = dict(response)
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

    def _should_immediate_memory_capture_after_move(self, symbol: str) -> bool:
        if not self.memory_auto_enabled or self.memory_capture_inflight:
            return False
        if symbol.lower() not in YAW_IMMEDIATE_CAPTURE_SYMBOLS:
            return False
        mode = self.memory_auto_mode_var.get().strip().lower()
        if mode not in {"time", "step"}:
            return False
        memory_collection = self.latest_memory_collection_state if isinstance(self.latest_memory_collection_state, dict) else {}
        return bool(memory_collection.get("active", False))

    def _trigger_yaw_memory_capture_after_move(self, symbol: str, action_name: str) -> None:
        self.status_var.set(f"Yaw {symbol.lower()} detected, triggering immediate memory capture.")
        threading.Thread(
            target=lambda: self._run_memory_capture_analyze(
                capture_source="auto_yaw",
                note=f"auto_yaw immediate trigger symbol={symbol.lower()} action={action_name}; reset auto counter",
                update_status=False,
            ),
            daemon=True,
        ).start()

    def _execute_move(self, symbol: str, *, from_sequence: bool = False) -> bool:
        payload = MOVE_COMMANDS.get(symbol.lower())
        if payload is None: return False
        self.move_request_inflight = True; self.pause(1.0)
        try:
            resp = self.safe(self.client.post_json, "/move_relative", payload, label=f"Move {symbol}")
            if isinstance(resp, dict):
                self.root.after(0, lambda: self.apply_state(resp)); self.refresh_map_async()
                if str(resp.get("status", "")).lower() in {"error", "disabled"}:
                    message = str(resp.get("message", "") or f"Move {symbol} failed.")
                    self.root.after(0, lambda msg=message: self.status_var.set(msg))
                    return False
                if from_sequence: self.root.after(0, lambda s=symbol: self.status_var.set(f"Sequence sent: {s}"))
                if self._should_immediate_memory_capture_after_move(symbol):
                    action_name = str(payload.get("action_name", symbol.lower()) or symbol.lower())
                    self.root.after(0, lambda s=symbol, a=action_name: self._trigger_yaw_memory_capture_after_move(s, a))
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

    def toggle_llm_task_window(self) -> None:
        if self.llm_task_window and self.llm_task_window.winfo_exists():
            self._close_llm_task_window()
            return
        self.llm_task_window = tk.Toplevel(self.root)
        self.llm_task_window.title("LLM Task Planner")
        self.llm_task_window.geometry("920x760")
        self.llm_task_window.protocol("WM_DELETE_WINDOW", self._close_llm_task_window)

        config = tk.LabelFrame(self.llm_task_window, text="Task")
        config.pack(fill="x", padx=8, pady=8)
        config.grid_columnconfigure(1, weight=1)
        tk.Label(config, text="Instruction").grid(row=0, column=0, sticky="nw", padx=6, pady=6)
        self.llm_task_text = tk.Text(config, height=4, wrap="word")
        self.llm_task_text.grid(row=0, column=1, columnspan=3, sticky="ew", padx=6, pady=6)
        self.llm_task_text.insert("1.0", self.llm_task_text_var.get().strip())
        tk.Label(config, text="API Base URL").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(config, textvariable=self.llm_control_base_url_var).grid(row=1, column=1, columnspan=3, sticky="ew", padx=6, pady=6)
        tk.Label(config, text="API Key").grid(row=2, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(config, textvariable=self.llm_control_api_key_var, show="*").grid(row=2, column=1, sticky="ew", padx=6, pady=6)
        tk.Label(config, text="Model").grid(row=2, column=2, sticky="e", padx=(10, 4), pady=6)
        tk.Entry(config, textvariable=self.llm_control_model_var).grid(row=2, column=3, sticky="ew", padx=6, pady=6)
        tk.Button(config, text="Analyze Task", command=self.on_llm_task_analyze).grid(row=3, column=0, padx=6, pady=6, sticky="ew")
        tk.Button(config, text="Apply Plan", command=self.on_llm_task_apply).grid(row=3, column=1, padx=6, pady=6, sticky="ew")
        tk.Button(config, text="Clear Plan", command=self.on_llm_task_clear).grid(row=3, column=2, padx=6, pady=6, sticky="ew")
        tk.Label(config, textvariable=self.llm_task_status_var, anchor="w", justify="left").grid(row=4, column=0, columnspan=4, sticky="ew", padx=6, pady=(0, 6))

        preview = tk.LabelFrame(self.llm_task_window, text="Plan Preview")
        preview.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.llm_task_preview_text = tk.Text(preview, wrap="none", font=("Consolas", 10))
        self.llm_task_preview_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.llm_task_preview_text.configure(state="disabled")
        self._refresh_llm_task_preview()

    def _close_llm_task_window(self) -> None:
        try:
            if self.llm_task_window is not None and self.llm_task_window.winfo_exists():
                self.llm_task_window.destroy()
        except Exception:
            pass
        self.llm_task_window = None
        self.llm_task_text = None
        self.llm_task_preview_text = None

    def _get_llm_task_text(self) -> str:
        if self.llm_task_text is not None:
            try:
                text = self.llm_task_text.get("1.0", "end").strip()
                if text:
                    self.llm_task_text_var.set(text)
                    return text
            except Exception:
                pass
        return self.llm_task_text_var.get().strip()

    def _refresh_llm_task_preview(self) -> None:
        plan = self.llm_task_plan if isinstance(self.llm_task_plan, dict) else {}
        trace = self.llm_task_execution_trace if isinstance(self.llm_task_execution_trace, dict) else {}
        payload = {
            "applied": bool(self.llm_task_plan_applied),
            "current_target_index": int(self.llm_task_current_index),
            "plan": plan,
            "execution_trace": trace,
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False)
        if self.llm_task_preview_text is not None and self.llm_task_preview_text.winfo_exists():
            self.llm_task_preview_text.configure(state="normal")
            self.llm_task_preview_text.delete("1.0", "end")
            self.llm_task_preview_text.insert("1.0", text)
            self.llm_task_preview_text.configure(state="disabled")
        targets = plan.get("ordered_targets", []) if isinstance(plan.get("ordered_targets"), list) else []
        target_ids = [str(item.get("house_id", "") or "") for item in targets if isinstance(item, dict)]
        summary = "Plan: none" if not target_ids else f"Plan: {' -> '.join(target_ids)} | applied={int(bool(self.llm_task_plan_applied))}"
        self.llm_task_plan_summary_var.set(summary)

    def on_llm_task_analyze(self) -> None:
        if self.llm_control_thread and self.llm_control_thread.is_alive():
            self.llm_task_status_var.set("LLM Task: stop LLM control before analyzing a new task.")
            return
        task_text = self._get_llm_task_text()
        if not task_text:
            self.llm_task_status_var.set("LLM Task: empty instruction.")
            return
        if not self.llm_control_base_url_var.get().strip() or not self.llm_control_api_key_var.get().strip():
            self.llm_task_status_var.set("LLM Task: missing API base/key.")
            return
        threading.Thread(target=lambda: self._run_llm_task_analyze(task_text), daemon=True).start()

    def _run_llm_task_analyze(self, task_text: str) -> None:
        from anthropic import Anthropic
        from phase2_multimodal_fusion_analysis.memory_aware_llm_teacher_label_validator import extract_json_object

        self._set_llm_control_var(self.llm_task_status_var, "LLM Task: analyzing...")
        session_dir = self._llm_control_session_dir_for_current_episode()
        registry = self._house_registry_for_llm_plan()
        system_prompt = self._build_llm_task_planner_system_prompt()
        user_prompt = self._build_llm_task_planner_user_prompt(task_text, registry)
        prompt_payload = {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "task_text": task_text,
            "house_registry": registry,
            "output_schema": LLM_TASK_PLAN_OUTPUT_SCHEMA,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (session_dir / "task_plan_prompt.json").write_text(
            json.dumps(prompt_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        try:
            client = Anthropic(
                api_key=self.llm_control_api_key_var.get().strip(),
                base_url=self.llm_control_base_url_var.get().strip(),
                timeout=self._llm_control_timeout_s(),
            )
            start_time = time.time()
            response = client.messages.create(
                model=self.llm_control_model_var.get().strip(),
                max_tokens=900,
                system=system_prompt,
                messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
            )
            raw_text = self._extract_anthropic_text(response)
            parsed = extract_json_object(raw_text)
            response_payload = {
                "api_style": "anthropic_sdk_text",
                "model_name": self.llm_control_model_var.get().strip(),
                "latency_ms": round(float((time.time() - start_time) * 1000.0), 3),
                "raw_text": raw_text,
                "parsed": parsed,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            }
            (session_dir / "task_plan_response.json").write_text(
                json.dumps(response_payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            plan = self._normalize_llm_task_plan(parsed, task_text, registry)
            self.llm_task_plan = plan
            self.llm_task_plan_applied = False
            self.llm_task_current_index = 0
            self.llm_task_execution_trace = self._build_initial_llm_execution_trace(plan)
            self._save_llm_task_plan_files()
            self.root.after(0, self._refresh_llm_task_preview)
            self._set_llm_control_var(self.llm_task_status_var, "LLM Task: plan analyzed, review then Apply Plan.")
        except Exception as exc:
            logger.exception("LLM task analysis failed")
            self._set_llm_control_var(self.llm_task_status_var, f"LLM Task: analysis failed: {exc}")

    def _house_registry_for_llm_plan(self) -> Dict[str, Any]:
        raw = self._read_local_houses_config()
        houses = raw.get("houses", []) if isinstance(raw.get("houses"), list) else []
        available: List[Dict[str, Any]] = []
        for house in houses:
            if not isinstance(house, dict):
                continue
            house_id = str(house.get("id", "") or "").strip()
            if not house_id:
                continue
            name = str(house.get("name", house_id) or house_id).strip()
            numeric = str(int(house_id)) if house_id.isdigit() else house_id
            available.append(
                {
                    "house_id": house_id,
                    "house_name": name,
                    "aliases": sorted({house_id, numeric, name, name.lower(), f"house {numeric}", f"house_{numeric}", f"house {house_id}", f"House_{numeric}"}),
                    "status": str(house.get("status", "") or ""),
                    "center_x": house.get("center_x"),
                    "center_y": house.get("center_y"),
                    "radius_cm": house.get("radius_cm"),
                }
            )
        return {
            "current_target_id": str(raw.get("current_target_id", "") or ""),
            "available_houses": available,
        }

    def _build_llm_task_planner_system_prompt(self) -> str:
        return (
            "You are a UAV task planner, not the low-level controller. "
            "Parse the user's natural language instruction into an ordered list of house search targets. "
            "Only use house ids from the provided house registry. Do not invent houses. "
            "Return strict JSON only. No markdown. No commentary."
        )

    def _build_llm_task_planner_user_prompt(self, task_text: str, registry: Dict[str, Any]) -> str:
        return "\n".join(
            [
                "User task:",
                str(task_text or ""),
                "",
                "House registry:",
                json.dumps(registry, indent=2, ensure_ascii=False),
                "",
                "Return a multi-house search plan. Each ordered target goal should be search_entry.",
                "Use finish_condition=target_entry_reached_or_no_entry_after_full_coverage.",
                "If a requested house cannot be matched, return status=needs_user_review and unmatched_targets.",
                "",
                "Expected JSON shape:",
                json.dumps(LLM_TASK_PLAN_OUTPUT_SCHEMA, indent=2, ensure_ascii=False),
            ]
        )

    def _build_house_alias_map(self, registry: Dict[str, Any]) -> Dict[str, str]:
        alias_map: Dict[str, str] = {}
        houses = registry.get("available_houses", []) if isinstance(registry.get("available_houses"), list) else []
        for house in houses:
            if not isinstance(house, dict):
                continue
            house_id = str(house.get("house_id", "") or "").strip()
            if not house_id:
                continue
            aliases = house.get("aliases", []) if isinstance(house.get("aliases"), list) else []
            for alias in aliases + [house_id, str(house.get("house_name", "") or "")]:
                text = str(alias or "").strip().lower().replace("_", " ")
                if text:
                    alias_map[text] = house_id
            if house_id.isdigit():
                alias_map[str(int(house_id))] = house_id
        return alias_map

    def _resolve_plan_house_id(self, value: Any, alias_map: Dict[str, str]) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        direct = raw.lower().replace("_", " ")
        if direct in alias_map:
            return alias_map[direct]
        digits = "".join(ch for ch in raw if ch.isdigit())
        if digits:
            padded = digits.zfill(3)
            if padded.lower() in alias_map:
                return alias_map[padded.lower()]
            if digits in alias_map:
                return alias_map[digits]
        return ""

    def _normalize_llm_task_plan(self, parsed: Dict[str, Any], task_text: str, registry: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(parsed, dict):
            parsed = {}
        alias_map = self._build_house_alias_map(registry)
        raw_targets = parsed.get("ordered_targets", []) if isinstance(parsed.get("ordered_targets"), list) else []
        normalized_targets: List[Dict[str, Any]] = []
        parsed_unmatched = parsed.get("unmatched_targets", []) if isinstance(parsed.get("unmatched_targets"), list) else []
        unmatched: List[str] = [str(item) for item in parsed_unmatched if str(item or "").strip()]
        seen: set[str] = set()
        for item in raw_targets:
            if not isinstance(item, dict):
                continue
            raw_house = item.get("house_id") or item.get("house_alias") or item.get("target") or item.get("name")
            house_id = self._resolve_plan_house_id(raw_house, alias_map)
            if not house_id:
                unmatched.append(str(raw_house or ""))
                continue
            if house_id in seen:
                continue
            seen.add(house_id)
            normalized_targets.append(
                {
                    "order": len(normalized_targets) + 1,
                    "house_id": house_id,
                    "house_alias": str(item.get("house_alias", raw_house) or raw_house or house_id),
                    "goal": "search_entry",
                    "finish_condition": "target_entry_reached_or_no_entry_after_full_coverage",
                    "status": "pending",
                    "finish_type": "",
                    "finished_at_step": None,
                }
            )
        if not normalized_targets and not unmatched:
            fallback_id = self._get_selected_target_house_id() or str(registry.get("current_target_id", "") or "")
            fallback_id = self._resolve_plan_house_id(fallback_id, alias_map)
            if fallback_id:
                normalized_targets.append(
                    {
                        "order": 1,
                        "house_id": fallback_id,
                        "house_alias": fallback_id,
                        "goal": "search_entry",
                        "finish_condition": "target_entry_reached_or_no_entry_after_full_coverage",
                        "status": "pending",
                        "finish_type": "",
                        "finished_at_step": None,
                    }
                )
        status = "ok" if normalized_targets and not unmatched else ("needs_user_review" if unmatched else "error")
        return {
            "version": "llm_multi_house_task_plan_v1",
            "status": status,
            "plan_id": str(parsed.get("plan_id", "") or f"llm_task_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            "task_text": str(task_text or ""),
            "ordered_targets": normalized_targets,
            "unmatched_targets": unmatched,
            "execution_policy": {
                "entry_reached_distance_cm": float(LLM_ENTRY_REACHED_DISTANCE_CM),
                "entry_reached_pose_proxy_cm": float(LLM_ENTRY_REACHED_POSE_PROXY_CM),
                "entry_reached_large_bbox_proxy_cm": float(LLM_ENTRY_REACHED_LARGE_BBOX_PROXY_CM),
                "max_decisions_per_house": 40,
                "allow_no_entry_completion": True,
                "stop_on_needs_review": True,
            },
            "reason": str(parsed.get("reason", "") or "Normalized from LLM task planner output."),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _build_initial_llm_execution_trace(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        memory_collection = self.latest_memory_collection_state if isinstance(self.latest_memory_collection_state, dict) else {}
        return {
            "version": "llm_multi_house_execution_trace_v1",
            "plan_id": str(plan.get("plan_id", "") or ""),
            "episode_id": str(memory_collection.get("episode_id", "") or ""),
            "collection_dir": str(memory_collection.get("collection_dir", "") or ""),
            "current_target_index": 0,
            "events": [],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def _save_llm_task_plan_files(self, labeling_dir: str = "") -> None:
        if str(labeling_dir or "").strip():
            root = self._resolve_llm_control_capture_root(labeling_dir)
            episode_id = self._resolve_llm_control_episode_id(labeling_dir)
            session_dir = root / episode_id
            session_dir.mkdir(parents=True, exist_ok=True)
        else:
            session_dir = self._llm_control_session_dir_for_current_episode()
        self.llm_task_plan["updated_at"] = datetime.now().isoformat(timespec="seconds")
        self.llm_task_execution_trace["updated_at"] = datetime.now().isoformat(timespec="seconds")
        (session_dir / "task_plan.json").write_text(
            json.dumps(self.llm_task_plan, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (session_dir / "execution_trace.json").write_text(
            json.dumps(self.llm_task_execution_trace, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def on_llm_task_apply(self) -> None:
        if not isinstance(self.llm_task_plan, dict) or not self.llm_task_plan.get("ordered_targets"):
            self.llm_task_status_var.set("LLM Task: no analyzed plan to apply.")
            return
        if str(self.llm_task_plan.get("status", "") or "") == "needs_user_review":
            self.llm_task_status_var.set("LLM Task: plan has unmatched targets; review before coding more.")
            return
        self.llm_task_plan_applied = True
        self.llm_task_current_index = 0
        self.llm_target_reacquire_lock = {}
        self.llm_obstacle_detour_lock = {}
        targets = self.llm_task_plan.get("ordered_targets", []) if isinstance(self.llm_task_plan.get("ordered_targets"), list) else []
        for idx, item in enumerate(targets):
            if isinstance(item, dict):
                item["status"] = "active" if idx == 0 else "pending"
        self.llm_task_execution_trace = self._build_initial_llm_execution_trace(self.llm_task_plan)
        first_house = str(targets[0].get("house_id", "") or "") if targets and isinstance(targets[0], dict) else ""
        if first_house:
            self._append_llm_task_event("target_started", first_house, {"reason": "plan_applied"})
            threading.Thread(target=lambda hid=first_house: self._select_target_house_for_llm_plan(hid), daemon=True).start()
        self._save_llm_task_plan_files()
        self._refresh_llm_task_preview()
        self.llm_task_status_var.set(f"LLM Task: plan applied, active target={first_house or 'none'}.")

    def on_llm_task_clear(self) -> None:
        self.llm_task_plan = {}
        self.llm_task_plan_applied = False
        self.llm_task_current_index = 0
        self.llm_task_execution_trace = {}
        self.llm_target_reacquire_lock = {}
        self.llm_obstacle_detour_lock = {}
        self._refresh_llm_task_preview()
        self.llm_task_status_var.set("LLM Task: cleared.")

    def _append_llm_task_event(self, event_type: str, house_id: str, extra: Optional[Dict[str, Any]] = None) -> None:
        if not isinstance(self.llm_task_execution_trace, dict) or not self.llm_task_execution_trace:
            self.llm_task_execution_trace = self._build_initial_llm_execution_trace(self.llm_task_plan if isinstance(self.llm_task_plan, dict) else {})
        memory_collection = self.latest_memory_collection_state if isinstance(self.latest_memory_collection_state, dict) else {}
        event = {
            "event_type": str(event_type),
            "house_id": str(house_id or ""),
            "step_index": int(memory_collection.get("step_index", 0) or 0),
            "time": datetime.now().isoformat(timespec="seconds"),
        }
        if isinstance(extra, dict):
            event.update(extra)
        events = self.llm_task_execution_trace.setdefault("events", [])
        if isinstance(events, list):
            events.append(event)
        self.llm_task_execution_trace["current_target_index"] = int(self.llm_task_current_index)
        self.llm_task_execution_trace["updated_at"] = datetime.now().isoformat(timespec="seconds")

    def toggle_llm_control_window(self) -> None:
        if self.llm_control_window and self.llm_control_window.winfo_exists():
            self._close_llm_control_window()
            return
        self.llm_control_window = tk.Toplevel(self.root)
        self.llm_control_window.title("LLM Control Pilot")
        self.llm_control_window.geometry("940x720")
        self.llm_control_window.protocol("WM_DELETE_WINDOW", self._close_llm_control_window)

        config = tk.LabelFrame(self.llm_control_window, text="LLM API")
        config.pack(fill="x", padx=8, pady=8)
        config.grid_columnconfigure(1, weight=1)
        config.grid_columnconfigure(3, weight=1)
        tk.Label(config, text="API Base URL").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(config, textvariable=self.llm_control_base_url_var).grid(row=0, column=1, columnspan=3, sticky="ew", padx=6, pady=6)
        tk.Label(config, text="API Key").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(config, textvariable=self.llm_control_api_key_var, show="*").grid(row=1, column=1, sticky="ew", padx=6, pady=6)
        tk.Label(config, text="Model").grid(row=1, column=2, sticky="e", padx=(10, 4), pady=6)
        tk.Entry(config, textvariable=self.llm_control_model_var).grid(row=1, column=3, sticky="ew", padx=6, pady=6)

        control = tk.LabelFrame(self.llm_control_window, text="Control Loop")
        control.pack(fill="x", padx=8, pady=(0, 8))
        for col in range(6):
            control.grid_columnconfigure(col, weight=1 if col in {1, 3, 5} else 0)
        tk.Label(control, text="Max Decisions").grid(row=0, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(control, textvariable=self.llm_control_max_steps_var, width=8).grid(row=0, column=1, sticky="w", padx=6, pady=6)
        tk.Label(control, text="Move Repeat Cap").grid(row=0, column=2, sticky="e", padx=(10, 4), pady=6)
        tk.Entry(control, textvariable=self.llm_control_repeat_cap_var, width=8).grid(row=0, column=3, sticky="w", padx=6, pady=6)
        tk.Label(control, text="Delay ms").grid(row=0, column=4, sticky="e", padx=(10, 4), pady=6)
        tk.Entry(control, textvariable=self.llm_control_delay_ms_var, width=8).grid(row=0, column=5, sticky="w", padx=6, pady=6)
        tk.Label(control, text="Timeout s").grid(row=1, column=0, sticky="w", padx=6, pady=6)
        tk.Entry(control, textvariable=self.llm_control_timeout_s_var, width=8).grid(row=1, column=1, sticky="w", padx=6, pady=6)
        tk.Button(control, text="Start LLM Control", command=self.on_llm_control_start).grid(row=1, column=2, padx=6, pady=6, sticky="ew")
        tk.Button(control, text="Single Decision", command=self.on_llm_control_step_once).grid(row=1, column=3, padx=6, pady=6, sticky="ew")
        tk.Button(control, text="Stop", command=self.on_llm_control_stop).grid(row=1, column=4, columnspan=2, padx=6, pady=6, sticky="ew")
        tk.Button(control, text="Analyze LLM Task", command=self.toggle_llm_task_window).grid(row=2, column=0, columnspan=3, padx=6, pady=(0, 6), sticky="ew")
        tk.Label(control, textvariable=self.llm_task_plan_summary_var, anchor="w", justify="left").grid(row=2, column=3, columnspan=3, sticky="ew", padx=6, pady=(0, 6))

        status = tk.LabelFrame(self.llm_control_window, text="Live State")
        status.pack(fill="x", padx=8, pady=(0, 8))
        for idx, var in enumerate(
            (
                self.llm_control_status_var,
                self.llm_control_step_var,
                self.llm_control_last_action_var,
                self.llm_control_last_capture_var,
                self.llm_control_reason_var,
            )
        ):
            tk.Label(status, textvariable=var, anchor="w", justify="left").grid(row=idx, column=0, sticky="ew", padx=6, pady=2)
        status.grid_columnconfigure(0, weight=1)

        log_frame = tk.LabelFrame(self.llm_control_window, text="Decision Log")
        log_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.llm_control_log_text = tk.Text(log_frame, wrap="word", font=("Consolas", 10))
        self.llm_control_log_text.pack(fill="both", expand=True, padx=6, pady=6)
        self.llm_control_log_text.configure(state="disabled")
        self._append_llm_control_log("LLM control window opened. Start a memory episode before running.")

    def _close_llm_control_window(self) -> None:
        self.on_llm_control_stop()
        try:
            if self.llm_control_window is not None and self.llm_control_window.winfo_exists():
                self.llm_control_window.destroy()
        except Exception:
            pass
        self.llm_control_window = None
        self.llm_control_log_text = None

    def _append_llm_control_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")

        def append() -> None:
            if self.llm_control_log_text is None or not self.llm_control_log_text.winfo_exists():
                return
            self.llm_control_log_text.configure(state="normal")
            self.llm_control_log_text.insert("end", f"[{timestamp}] {message}\n")
            self.llm_control_log_text.see("end")
            self.llm_control_log_text.configure(state="disabled")

        self.root.after(0, append)

    def _set_llm_control_var(self, var: tk.StringVar, value: str) -> None:
        self.root.after(0, lambda: var.set(value))

    def on_llm_control_start(self) -> None:
        self._start_llm_control_loop(single_step=False)

    def on_llm_control_step_once(self) -> None:
        self._start_llm_control_loop(single_step=True)

    def _start_llm_control_loop(self, *, single_step: bool) -> None:
        if self.llm_control_thread and self.llm_control_thread.is_alive():
            self.llm_control_status_var.set("LLM Control: already running")
            return
        memory_collection = self.latest_memory_collection_state if isinstance(self.latest_memory_collection_state, dict) else {}
        if not bool(memory_collection.get("active", False)):
            self.llm_control_status_var.set("LLM Control: start memory episode first")
            self.status_var.set("Start Episode first, then start LLM control.")
            return
        base_url = self.llm_control_base_url_var.get().strip()
        api_key = self.llm_control_api_key_var.get().strip()
        model = self.llm_control_model_var.get().strip()
        if not base_url or not api_key or not model:
            self.llm_control_status_var.set("LLM Control: missing API base/key/model")
            return
        if self.memory_auto_enabled:
            self._stop_memory_auto_capture_local()
            self._append_llm_control_log("Panel auto capture stopped to avoid duplicate captures during LLM control.")
        try:
            max_steps = 1 if single_step else max(1, int(float(self.llm_control_max_steps_var.get().strip())))
        except ValueError:
            self.llm_control_status_var.set("LLM Control: invalid max decisions")
            return
        self.llm_control_stop_event.clear()
        self.llm_control_thread = threading.Thread(
            target=lambda: self._llm_control_loop(max_steps=max_steps),
            daemon=True,
        )
        self.llm_control_thread.start()

    def on_llm_control_stop(self) -> None:
        self.llm_control_stop_event.set()
        if self.llm_control_running:
            self.llm_control_status_var.set("LLM Control: stopping...")

    def _llm_control_loop(self, *, max_steps: int) -> None:
        self.llm_control_running = True
        self.llm_control_decision_history = []
        self.llm_target_reacquire_lock = {}
        self.llm_obstacle_detour_lock = {}
        reuse_labeling_dir = ""
        self._set_llm_control_var(self.llm_control_status_var, "LLM Control: running")
        self._append_llm_control_log(f"Started LLM control loop with max_decisions={max_steps}.")
        try:
            if not self._prepare_llm_plan_execution():
                return
            self._ensure_movement_enabled_for_llm()
            for decision_index in range(1, int(max_steps) + 1):
                if self.llm_control_stop_event.is_set():
                    break
                self._set_llm_control_var(self.llm_control_step_var, f"{decision_index}/{max_steps}")
                if reuse_labeling_dir:
                    labeling_dir = reuse_labeling_dir
                    reuse_labeling_dir = ""
                    self._append_llm_control_log(f"Reusing immediate yaw capture: {Path(labeling_dir).parent.name}")
                else:
                    labeling_dir = self._run_llm_control_capture(
                        capture_source="llm_control",
                        note=f"llm_control decision_step={decision_index}",
                    )
                if not labeling_dir:
                    self._set_llm_control_var(self.llm_control_status_var, "LLM Control: capture failed")
                    break

                self._set_llm_control_var(self.llm_control_last_capture_var, f"Last capture: {Path(labeling_dir).parent.name}")
                completion = self._evaluate_llm_house_completion(labeling_dir)
                if bool(completion.get("completed", False)):
                    switched = self._advance_llm_task_plan_after_completion(completion)
                    if switched:
                        continue
                    break
                decision = self._request_llm_control_decision(labeling_dir=labeling_dir, decision_index=decision_index)
                normalized = self._normalize_llm_control_decision(decision)
                normalized = self._apply_llm_control_rule_overrides(normalized, labeling_dir)
                self._write_llm_control_decision(labeling_dir, decision, normalized, decision_index)
                self.llm_control_decision_history.append(normalized)
                self.llm_control_decision_history = self.llm_control_decision_history[-12:]

                symbol = str(normalized.get("action_symbol", "x") or "x")
                repeat = int(normalized.get("repeat", 1) or 1)
                reason = str(normalized.get("reason", "") or "-")
                confidence = float(normalized.get("confidence", 0.0) or 0.0)
                self._set_llm_control_var(
                    self.llm_control_last_action_var,
                    f"Last action: {symbol} x{repeat} conf={confidence:.2f}",
                )
                self._set_llm_control_var(self.llm_control_reason_var, f"Reason: {reason}")
                self._append_llm_control_log(f"Decision {decision_index}: action={symbol} repeat={repeat} stop={int(bool(normalized.get('stop')))} reason={reason}")

                if bool(normalized.get("stop", False)):
                    self._set_llm_control_var(self.llm_control_status_var, "LLM Control: stopped by LLM")
                    break

                move_ok = True
                for repeat_index in range(repeat):
                    if self.llm_control_stop_event.is_set():
                        move_ok = False
                        break
                    move_ok = self._execute_move(symbol, from_sequence=False)
                    if not move_ok:
                        break
                    time.sleep(self._llm_control_delay_s())
                if not move_ok:
                    self._set_llm_control_var(self.llm_control_status_var, "LLM Control: movement failed/stopped")
                    break

                if symbol in YAW_IMMEDIATE_CAPTURE_SYMBOLS and not self.llm_control_stop_event.is_set():
                    yaw_labeling_dir = self._run_llm_control_capture(
                        capture_source="llm_yaw",
                        note=f"llm_control immediate capture after yaw symbol={symbol} decision_step={decision_index}",
                    )
                    if yaw_labeling_dir:
                        reuse_labeling_dir = yaw_labeling_dir
                time.sleep(self._llm_control_delay_s())
            else:
                self._set_llm_control_var(self.llm_control_status_var, "LLM Control: completed max decisions")
        except Exception as exc:
            logger.exception("LLM control loop failed")
            self._set_llm_control_var(self.llm_control_status_var, f"LLM Control: error {exc}")
            self._append_llm_control_log(f"ERROR: {exc}")
        finally:
            self.llm_control_running = False
            if self.llm_control_stop_event.is_set():
                self._set_llm_control_var(self.llm_control_status_var, "LLM Control: stopped")
            self._append_llm_control_log("LLM control loop ended.")

    def _ensure_movement_enabled_for_llm(self) -> None:
        if self.movement_enabled_state:
            return
        resp = self.safe(
            self.client.post_json,
            "/basic_movement_enable",
            {"enabled": True},
            label="Enable movement for LLM control",
        )
        if isinstance(resp, dict):
            self.root.after(0, lambda r=resp: self.apply_state(r))
            self._append_llm_control_log("Basic movement was disabled; enabled it for LLM control.")

    def _select_target_house_for_llm_plan(self, house_id: str) -> bool:
        hid = str(house_id or "").strip()
        if not hid:
            return False
        response = self.safe(
            self.client.post_json,
            "/select_target_house",
            {"house_id": hid},
            label=f"LLM plan select target {hid}",
        )
        if not isinstance(response, dict) or response.get("status") != "ok":
            message = str(response.get("message", "") if isinstance(response, dict) else "" or f"Failed to select {hid}")
            self._append_llm_control_log(f"Target switch failed: {hid} {message}")
            return False
        self._update_local_current_target_id(hid)
        self.root.after(0, lambda h=hid: self._set_selected_target_house(h, mark_clean=True))
        self.root.after(0, lambda r=response: self.apply_state(r))
        self._append_llm_control_log(f"Target switched to house {hid}.")
        return True

    def _current_llm_plan_target(self) -> Dict[str, Any]:
        if not self.llm_task_plan_applied or not isinstance(self.llm_task_plan, dict):
            return {}
        targets = self.llm_task_plan.get("ordered_targets", [])
        if not isinstance(targets, list) or not targets:
            return {}
        index = max(0, min(int(self.llm_task_current_index), len(targets) - 1))
        item = targets[index]
        return item if isinstance(item, dict) else {}

    def _prepare_llm_plan_execution(self) -> bool:
        if not self.llm_task_plan_applied:
            return True
        targets = self.llm_task_plan.get("ordered_targets", []) if isinstance(self.llm_task_plan.get("ordered_targets"), list) else []
        if not targets:
            self._set_llm_control_var(self.llm_control_status_var, "LLM Control: applied plan has no targets")
            return False
        active_index = None
        for idx, item in enumerate(targets):
            if isinstance(item, dict) and str(item.get("status", "") or "") in {"active", "pending"}:
                active_index = idx
                break
        if active_index is None:
            self._set_llm_control_var(self.llm_control_status_var, "LLM Control: plan already completed")
            return False
        self.llm_task_current_index = int(active_index)
        target = targets[self.llm_task_current_index]
        if isinstance(target, dict):
            target["status"] = "active"
            house_id = str(target.get("house_id", "") or "")
        else:
            house_id = ""
        if not house_id:
            self._set_llm_control_var(self.llm_control_status_var, "LLM Control: active plan target missing house_id")
            return False
        if not self._select_target_house_for_llm_plan(house_id):
            self._set_llm_control_var(self.llm_control_status_var, f"LLM Control: failed to select target {house_id}")
            return False
        self.llm_target_reacquire_lock = {}
        self.llm_obstacle_detour_lock = {}
        self._append_llm_task_event("target_started", house_id, {"reason": "llm_control_start"})
        self._save_llm_task_plan_files()
        self.root.after(0, self._refresh_llm_task_preview)
        return True

    def _advance_llm_task_plan_after_completion(self, completion: Dict[str, Any]) -> bool:
        house_id = str(completion.get("house_id", "") or "")
        finish_type = str(completion.get("finish_type", "") or "")
        source_labeling_dir = str(completion.get("source_labeling_dir", "") or "")
        self._append_llm_control_log(f"House search finished: house={house_id} finish_type={finish_type} reason={completion.get('reason', '')}")
        if not self.llm_task_plan_applied:
            self._set_llm_control_var(self.llm_control_status_var, f"LLM Control: house {house_id} finished")
            self.llm_control_stop_event.set()
            return False

        targets = self.llm_task_plan.get("ordered_targets", []) if isinstance(self.llm_task_plan.get("ordered_targets"), list) else []
        if 0 <= int(self.llm_task_current_index) < len(targets) and isinstance(targets[self.llm_task_current_index], dict):
            targets[self.llm_task_current_index]["status"] = "done"
            targets[self.llm_task_current_index]["finish_type"] = finish_type
            targets[self.llm_task_current_index]["finished_at_step"] = completion.get("step_index")
        self._append_llm_task_event("target_finished", house_id, completion)

        next_index = None
        for idx, item in enumerate(targets):
            if isinstance(item, dict) and str(item.get("status", "") or "") == "pending":
                next_index = idx
                break
        if next_index is None:
            self.llm_task_plan["status"] = "completed"
            self._append_llm_task_event("plan_completed", house_id, {"reason": "all_targets_done"})
            self._save_llm_task_plan_files(source_labeling_dir)
            self.root.after(0, self._refresh_llm_task_preview)
            self._set_llm_control_var(self.llm_control_status_var, "LLM Control: multi-house plan completed")
            self.llm_control_stop_event.set()
            return False

        self.llm_task_current_index = int(next_index)
        next_target = targets[next_index]
        next_target["status"] = "active"
        next_house_id = str(next_target.get("house_id", "") or "")
        if not next_house_id:
            self._set_llm_control_var(self.llm_control_status_var, "LLM Control: next target missing house_id")
            self.llm_control_stop_event.set()
            return False
        if not self._select_target_house_for_llm_plan(next_house_id):
            self.llm_control_stop_event.set()
            return False
        self.llm_target_reacquire_lock = {}
        self.llm_obstacle_detour_lock = {}
        self._append_llm_task_event("target_started", next_house_id, {"reason": f"previous_finished:{finish_type}"})
        self._save_llm_task_plan_files(source_labeling_dir)
        self.root.after(0, self._refresh_llm_task_preview)
        self._set_llm_control_var(self.llm_control_status_var, f"LLM Control: switched to house {next_house_id}")
        return True

    def _read_json_file(self, path: Path) -> Dict[str, Any]:
        try:
            value = json.loads(path.read_text(encoding="utf-8-sig"))
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _fusion_payload_from_labeling_dir(self, labeling_dir: str) -> Dict[str, Any]:
        payload = self._read_json_file(Path(labeling_dir) / "fusion_result.json")
        fusion = payload.get("fusion", {}) if isinstance(payload.get("fusion"), dict) else {}
        return fusion if fusion else payload

    def _metadata_from_labeling_dir(self, labeling_dir: str) -> Dict[str, Any]:
        return self._read_json_file(Path(labeling_dir) / "sample_metadata.json")

    def _as_float_or_none(self, value: Any) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None

    def _candidate_entry_distance_cm(self, fusion: Dict[str, Any]) -> Optional[float]:
        direct_keys = ("entry_distance_cm", "distance_cm", "dist_cm")
        for key in direct_keys:
            value = self._as_float_or_none(fusion.get(key))
            if value is not None and value > 0:
                return value
        for section_key in ("semantic_depth_assessment", "matched_depth_candidate", "chosen_depth_candidate", "best_depth_candidate"):
            section = fusion.get(section_key)
            if not isinstance(section, dict):
                continue
            for key in direct_keys + ("depth_cm", "center_depth_cm"):
                value = self._as_float_or_none(section.get(key))
                if value is not None and value > 0:
                    return value
        return None

    def _target_house_distance_cm(self, fusion: Dict[str, Any]) -> Optional[float]:
        target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
        for key in ("target_house_distance_cm", "distance_to_target_cm", "target_distance_cm"):
            value = self._as_float_or_none(target_context.get(key))
            if value is not None and value > 0:
                return value
        return None

    def _semantic_class_name(self, fusion: Dict[str, Any]) -> str:
        semantic = fusion.get("semantic_detection", {}) if isinstance(fusion.get("semantic_detection"), dict) else {}
        return str(
            semantic.get("class_name_normalized")
            or semantic.get("class_name")
            or fusion.get("class_name")
            or fusion.get("semantic_class")
            or ""
        ).strip().lower().replace("_", " ")

    def _is_doorlike_class(self, class_name: str) -> bool:
        normalized = str(class_name or "").strip().lower().replace("_", " ")
        return normalized in {name.replace("_", " ") for name in DOORLIKE_CLASS_NAMES}

    def _semantic_bbox_metrics(self, fusion: Dict[str, Any]) -> Dict[str, float]:
        semantic = fusion.get("semantic_detection", {}) if isinstance(fusion.get("semantic_detection"), dict) else {}
        xyxy = semantic.get("xyxy")
        if not isinstance(xyxy, list) or len(xyxy) < 4:
            return {}
        try:
            x1, y1, x2, y2 = [float(value) for value in xyxy[:4]]
        except Exception:
            return {}
        target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
        image_width = max(1.0, float(target_context.get("image_width", 640) or 640))
        image_height = max(1.0, float(target_context.get("image_height", 480) or 480))
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        area_ratio = (width * height) / max(1.0, image_width * image_height)
        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "width_ratio": width / image_width,
            "height_ratio": height / image_height,
            "area_ratio": area_ratio,
            "touches_left": 1.0 if x1 <= 4.0 else 0.0,
            "touches_right": 1.0 if x2 >= image_width - 4.0 else 0.0,
            "touches_top": 1.0 if y1 <= 8.0 else 0.0,
            "touches_bottom": 1.0 if y2 >= image_height - 4.0 else 0.0,
        }

    def _is_large_near_entry_bbox(self, fusion: Dict[str, Any]) -> bool:
        metrics = self._semantic_bbox_metrics(fusion)
        if not metrics:
            return False
        area_ratio = float(metrics.get("area_ratio", 0.0) or 0.0)
        height_ratio = float(metrics.get("height_ratio", 0.0) or 0.0)
        width_ratio = float(metrics.get("width_ratio", 0.0) or 0.0)
        touches_edge = bool(metrics.get("touches_left", 0.0) or metrics.get("touches_right", 0.0) or metrics.get("touches_bottom", 0.0))
        return (
            area_ratio >= LLM_LARGE_ENTRY_BBOX_AREA_RATIO
            or height_ratio >= LLM_LARGE_ENTRY_BBOX_HEIGHT_RATIO
            or (touches_edge and width_ratio >= LLM_LARGE_ENTRY_BBOX_WIDTH_RATIO)
        )

    def _front_path_clear_for_approach(self, fusion: Dict[str, Any]) -> bool:
        if bool(fusion.get("front_blocked", False)):
            return False
        global_obstacle = fusion.get("global_front_obstacle", False)
        if isinstance(global_obstacle, dict):
            if bool(global_obstacle.get("present", False)):
                severity = str(global_obstacle.get("severity", "") or "").lower()
                return severity not in {"high", "severe", "blocked", "critical"}
        elif bool(global_obstacle):
            return False
        assessment = fusion.get("semantic_depth_assessment", {}) if isinstance(fusion.get("semantic_depth_assessment"), dict) else {}
        if bool(assessment.get("front_obstacle_present", False)):
            severity = str(assessment.get("front_obstacle_severity", "") or "").lower()
            return severity not in {"high", "severe", "blocked", "critical"}
        return True

    def _depth_cm_path_from_labeling_dir(self, labeling_dir: str) -> Optional[Path]:
        if not labeling_dir:
            return None
        labeling_path = Path(labeling_dir)
        direct = labeling_path / "depth_cm.png"
        if direct.is_file():
            return direct
        metadata = self._metadata_from_labeling_dir(labeling_dir)
        images = metadata.get("images", {}) if isinstance(metadata.get("images"), dict) else {}
        for key in ("depth_cm", "depth"):
            candidate = images.get(key)
            if candidate:
                path = Path(str(candidate))
                if path.is_file():
                    return path
        return None

    def _depth_band_summary(
        self,
        depth_cm: np.ndarray,
        *,
        y_range: tuple,
        x_range: tuple = LLM_DEPTH_CORRIDOR_X_RANGE,
    ) -> Dict[str, Any]:
        h, w = depth_cm.shape[:2]
        y0 = max(0, min(h, int(round(float(y_range[0]) * h))))
        y1 = max(y0 + 1, min(h, int(round(float(y_range[1]) * h))))
        x0 = max(0, min(w, int(round(float(x_range[0]) * w))))
        x1 = max(x0 + 1, min(w, int(round(float(x_range[1]) * w))))
        roi = depth_cm[y0:y1, x0:x1].astype(np.float32, copy=False)
        valid = roi[np.isfinite(roi) & (roi > 0.0) & (roi <= float(LLM_DEPTH_MAX_VALID_CM))]
        summary: Dict[str, Any] = {
            "y_range": [float(y_range[0]), float(y_range[1])],
            "x_range": [float(x_range[0]), float(x_range[1])],
            "sample_count": int(valid.size),
            "min_cm": None,
            "p05_cm": None,
            "p10_cm": None,
            "p20_cm": None,
            "median_cm": None,
            "close_ratio_100": 0.0,
            "close_ratio_140": 0.0,
        }
        if valid.size == 0:
            return summary
        percentiles = np.percentile(valid, [5, 10, 20, 50])
        summary.update(
            {
                "min_cm": round(float(np.min(valid)), 2),
                "p05_cm": round(float(percentiles[0]), 2),
                "p10_cm": round(float(percentiles[1]), 2),
                "p20_cm": round(float(percentiles[2]), 2),
                "median_cm": round(float(percentiles[3]), 2),
                "close_ratio_100": round(float(np.mean(valid <= 100.0)), 4),
                "close_ratio_140": round(float(np.mean(valid <= 140.0)), 4),
            }
        )
        return summary

    def _height_aware_front_corridor_status(self, labeling_dir: str) -> Dict[str, Any]:
        depth_path = self._depth_cm_path_from_labeling_dir(labeling_dir)
        status: Dict[str, Any] = {
            "available": False,
            "depth_cm_path": str(depth_path) if depth_path else "",
            "corridor_x_range": [float(LLM_DEPTH_CORRIDOR_X_RANGE[0]), float(LLM_DEPTH_CORRIDOR_X_RANGE[1])],
            "low_obstacle_passable": False,
            "body_corridor_clear": False,
            "body_corridor_blocking": False,
            "classification": "unavailable",
            "reason": "",
        }
        if depth_path is None:
            status["reason"] = "depth_cm.png not found"
            return status
        try:
            depth_cm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
            if depth_cm is None:
                status["reason"] = "failed to read depth_cm.png"
                return status
            if depth_cm.ndim == 3:
                depth_cm = depth_cm[:, :, 0]
            upper = self._depth_band_summary(depth_cm, y_range=LLM_DEPTH_UPPER_Y_RANGE)
            body = self._depth_band_summary(depth_cm, y_range=LLM_DEPTH_BODY_Y_RANGE)
            low = self._depth_band_summary(depth_cm, y_range=LLM_DEPTH_LOW_Y_RANGE)
        except Exception as exc:
            status["reason"] = f"height-aware depth analysis failed: {exc}"
            return status

        body_min = self._as_float_or_none(body.get("min_cm"))
        body_p10 = self._as_float_or_none(body.get("p10_cm"))
        body_close_ratio = float(body.get("close_ratio_140", 0.0) or 0.0)
        low_min = self._as_float_or_none(low.get("min_cm"))
        low_p10 = self._as_float_or_none(low.get("p10_cm"))
        low_close_ratio = float(low.get("close_ratio_140", 0.0) or 0.0)
        upper_close_ratio = float(upper.get("close_ratio_140", 0.0) or 0.0)

        low_near = bool(
            (low_min is not None and low_min <= LLM_LOW_OBSTACLE_NEAR_CM)
            or (low_p10 is not None and low_p10 <= LLM_LOW_OBSTACLE_NEAR_CM)
            or low_close_ratio >= LLM_LOW_CLOSE_RATIO_MIN
        )
        body_clear = bool(
            body_min is not None
            and body_p10 is not None
            and body_min >= LLM_BODY_CORRIDOR_MIN_CLEAR_CM
            and body_p10 >= LLM_BODY_CORRIDOR_CLEAR_P10_CM
            and body_close_ratio <= LLM_BODY_CLOSE_RATIO_MAX
            and upper_close_ratio <= LLM_BODY_CLOSE_RATIO_MAX
        )
        body_blocking = bool(
            (body_min is not None and body_min <= LLM_OBSTACLE_FRONT_MIN_BLOCK_CM)
            or (body_p10 is not None and body_p10 <= LLM_TARGET_TRANSIT_FRONT_MIN_FORWARD_CM)
            or body_close_ratio > LLM_BODY_CLOSE_RATIO_MAX
        )
        low_passable = bool(low_near and body_clear and not body_blocking)
        if low_passable:
            classification = "low_obstacle_passable"
            reason = "near depth is concentrated in the lower image band while the UAV body corridor is clear"
        elif body_blocking:
            classification = "body_corridor_blocked"
            reason = "near depth intersects the central body-height corridor"
        elif body_clear:
            classification = "body_corridor_clear"
            reason = "central body-height corridor has sufficient depth clearance"
        else:
            classification = "uncertain"
            reason = "depth bands do not provide a confident pass/block decision"

        status.update(
            {
                "available": True,
                "upper_band": upper,
                "body_band": body,
                "low_band": low,
                "low_obstacle_passable": low_passable,
                "body_corridor_clear": body_clear,
                "body_corridor_blocking": body_blocking,
                "classification": classification,
                "reason": reason,
                "effective_front_min_depth_cm": body_min,
                "effective_front_p10_depth_cm": body_p10,
            }
        )
        return status

    def _front_obstacle_status(self, fusion: Dict[str, Any], labeling_dir: str = "") -> Dict[str, Any]:
        status: Dict[str, Any] = {
            "present": bool(fusion.get("front_blocked", False)),
            "severity": "",
            "front_min_depth_cm": None,
            "front_mean_depth_cm": None,
            "blocking": False,
            "too_close": False,
            "low_obstacle_passable": False,
        }
        global_obstacle = fusion.get("global_front_obstacle", {}) if isinstance(fusion.get("global_front_obstacle"), dict) else {}
        if global_obstacle:
            status["present"] = bool(status["present"] or global_obstacle.get("present", False))
            status["severity"] = str(global_obstacle.get("severity", "") or "").lower()
            status["front_min_depth_cm"] = self._as_float_or_none(global_obstacle.get("front_min_depth_cm"))
            status["front_mean_depth_cm"] = self._as_float_or_none(global_obstacle.get("front_mean_depth_cm"))
        assessment = fusion.get("semantic_depth_assessment", {}) if isinstance(fusion.get("semantic_depth_assessment"), dict) else {}
        if assessment:
            status["present"] = bool(status["present"] or assessment.get("front_obstacle_present", False))
            if not status["severity"]:
                status["severity"] = str(assessment.get("front_obstacle_severity", "") or "").lower()
            if status["front_min_depth_cm"] is None:
                status["front_min_depth_cm"] = self._as_float_or_none(assessment.get("front_min_depth_cm"))
        severity = str(status.get("severity", "") or "").lower()
        front_min = self._as_float_or_none(status.get("front_min_depth_cm"))
        severe = severity in {"high", "severe", "blocked", "critical"}
        medium_close = severity == "medium" and front_min is not None and front_min <= LLM_OBSTACLE_FRONT_MIN_BLOCK_CM
        too_close = front_min is not None and front_min <= LLM_OBSTACLE_FRONT_MIN_BACKOFF_CM
        status["too_close"] = bool(too_close)
        status["blocking"] = bool(status["present"] and (severe or medium_close or too_close))
        if labeling_dir:
            height_aware = self._height_aware_front_corridor_status(labeling_dir)
            status["height_aware_front_corridor"] = height_aware
            if bool(height_aware.get("available", False)):
                if bool(height_aware.get("body_corridor_blocking", False)):
                    status["blocking"] = bool(status["present"])
                    status["too_close"] = bool(status["too_close"] or severe)
                    status["blocking_reason"] = str(height_aware.get("reason", "") or "")
                elif bool(height_aware.get("low_obstacle_passable", False)):
                    status["low_obstacle_passable"] = True
                    status["blocking"] = False
                    status["too_close"] = False
                    status["severity"] = "low_passable"
                    status["effective_front_min_depth_cm"] = height_aware.get("effective_front_min_depth_cm")
                    status["effective_front_p10_depth_cm"] = height_aware.get("effective_front_p10_depth_cm")
                    status["blocking_reason"] = str(height_aware.get("reason", "") or "")
        return status

    def _detour_side_symbol_from_fusion(self, fusion: Dict[str, Any]) -> str:
        for key in ("recommended_subgoal", "target_conditioned_subgoal"):
            value = str(fusion.get(key, "") or "").lower()
            if "detour_left" in value:
                return "a"
            if "detour_right" in value:
                return "d"
        for key in ("recommended_action_hint", "target_conditioned_action_hint"):
            value = str(fusion.get(key, "") or "").lower().replace("-", "_").replace(" ", "_")
            if value == "left":
                return "a"
            if value == "right":
                return "d"
        lock = self.llm_obstacle_detour_lock if isinstance(self.llm_obstacle_detour_lock, dict) else {}
        locked_side = str(lock.get("side_symbol", "") or "")
        if locked_side in {"a", "d"}:
            return locked_side
        target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
        bearing = self._as_float_or_none(target_context.get("target_house_bearing_deg"))
        if bearing is not None:
            if bearing > 8.0:
                return "d"
            if bearing < -8.0:
                return "a"
        return LLM_OBSTACLE_DEFAULT_DETOUR_SYMBOL

    def _apply_front_obstacle_detour_override(self, normalized: Dict[str, Any], labeling_dir: str) -> Dict[str, Any]:
        if bool(normalized.get("stop", False)):
            return normalized
        fusion = self._fusion_payload_from_labeling_dir(labeling_dir)
        house_id = self._current_target_house_id_from_fusion(fusion)
        if house_id:
            boundary = self._target_boundary_context(fusion, house_id)
            axis_plan = self._axis_aligned_boundary_transit_plan(fusion, house_id, boundary)
            if str(axis_plan.get("action_symbol", "") or "") in YAW_IMMEDIATE_CAPTURE_SYMBOLS:
                return normalized
        obstacle = self._front_obstacle_status(fusion, labeling_dir)
        current_symbol = str(normalized.get("action_symbol", "") or "")
        if not bool(obstacle.get("blocking", False)):
            if self.llm_obstacle_detour_lock:
                self.llm_obstacle_detour_lock = {}
            return normalized
        metadata = self._metadata_from_labeling_dir(labeling_dir)
        step_index = int(metadata.get("step_index", 0) or 0)
        lock = self.llm_obstacle_detour_lock if isinstance(self.llm_obstacle_detour_lock, dict) else {}
        if str(lock.get("target_house_id", "") or "") != house_id:
            lock = {
                "target_house_id": house_id,
                "side_symbol": self._detour_side_symbol_from_fusion(fusion),
                "total_steps": 0,
                "backoff_steps": 0,
                "side_steps": 0,
                "started_at_step": step_index,
            }
        total_steps = int(lock.get("total_steps", 0) or 0)
        if total_steps >= LLM_OBSTACLE_DETOUR_MAX_STEPS:
            self.llm_obstacle_detour_lock = {}
            return normalized

        side_symbol = str(lock.get("side_symbol", "") or LLM_OBSTACLE_DEFAULT_DETOUR_SYMBOL)
        if side_symbol not in {"a", "d"}:
            side_symbol = LLM_OBSTACLE_DEFAULT_DETOUR_SYMBOL
        too_close = bool(obstacle.get("too_close", False))
        backoff_steps = int(lock.get("backoff_steps", 0) or 0)
        side_steps = int(lock.get("side_steps", 0) or 0)
        if too_close and backoff_steps < 2:
            action_symbol = "s"
            repeat = 1
            phase = "backoff"
            lock["backoff_steps"] = backoff_steps + 1
        else:
            if side_steps >= LLM_OBSTACLE_DETOUR_SIDE_STEPS and bool(obstacle.get("too_close", False)):
                side_symbol = "d" if side_symbol == "a" else "a"
                lock["side_symbol"] = side_symbol
                lock["side_steps"] = 0
                side_steps = 0
            action_symbol = side_symbol
            repeat = min(self._llm_control_repeat_cap(), 2)
            phase = "side_detour"
            lock["side_steps"] = side_steps + 1
        lock["total_steps"] = total_steps + 1
        lock["last_step_index"] = step_index
        lock["last_front_obstacle"] = obstacle
        self.llm_obstacle_detour_lock = lock

        original = dict(normalized)
        normalized = dict(normalized)
        normalized["action_symbol"] = action_symbol
        normalized["repeat"] = max(1, int(repeat))
        normalized["stop"] = False
        normalized["need_capture_after"] = False
        normalized["obstacle_detour_lock"] = dict(lock)
        normalized["rule_override"] = {
            "type": "front_obstacle_structured_detour",
            "original_action_symbol": original.get("action_symbol"),
            "original_repeat": original.get("repeat"),
            "target_house_id": house_id,
            "phase": phase,
            "action_symbol": action_symbol,
            "repeat": int(repeat),
            "side_symbol": side_symbol,
            "front_obstacle": obstacle,
            "recommended_subgoal": fusion.get("recommended_subgoal"),
            "recommended_action_hint": fusion.get("recommended_action_hint"),
            "reason": "front obstacle is too close; execute physical backoff/strafe detour instead of yaw-only observation or forward motion",
        }
        reason = str(normalized.get("reason", "") or "")
        detour_text = (
            f"front-obstacle detour override: {phase} with {action_symbol} x{int(repeat)} "
            f"(front_min={obstacle.get('front_min_depth_cm')}cm, severity={obstacle.get('severity')})"
        )
        normalized["reason"] = (reason + " | " if reason else "") + detour_text
        if original.get("action_symbol") != action_symbol:
            self._append_llm_control_log(
                f"Obstacle detour override: action={original.get('action_symbol')} -> {action_symbol} repeat={int(repeat)}; "
                f"front_min={obstacle.get('front_min_depth_cm')} severity={obstacle.get('severity')}"
            )
        return normalized

    def _apply_target_boundary_transit_override(self, normalized: Dict[str, Any], labeling_dir: str) -> Dict[str, Any]:
        if bool(normalized.get("stop", False)):
            return normalized
        fusion = self._fusion_payload_from_labeling_dir(labeling_dir)
        house_id = self._current_target_house_id_from_fusion(fusion)
        if not house_id:
            return normalized
        boundary = self._target_boundary_context(fusion, house_id)
        if not bool(boundary.get("outside_search_boundary", False)):
            return normalized
        axis_plan = self._axis_aligned_boundary_transit_plan(fusion, house_id, boundary)
        if not axis_plan and not bool(boundary.get("target_house_in_fov", False)):
            return normalized
        obstacle = self._front_obstacle_status(fusion, labeling_dir)
        front_min = self._as_float_or_none(obstacle.get("front_min_depth_cm"))
        axis_action = str(axis_plan.get("action_symbol", "") or "") if axis_plan else ""
        if (
            axis_action not in YAW_IMMEDIATE_CAPTURE_SYMBOLS
            and bool(obstacle.get("blocking", False))
        ) or (
            bool(obstacle.get("present", False))
            and front_min is not None
            and front_min <= LLM_TARGET_TRANSIT_FRONT_MIN_FORWARD_CM
            and not bool(obstacle.get("low_obstacle_passable", False))
            and axis_action not in YAW_IMMEDIATE_CAPTURE_SYMBOLS
        ):
            return normalized

        if axis_plan:
            action_symbol = str(axis_plan.get("action_symbol", "") or "w")
            repeat = int(axis_plan.get("repeat", 1) or 1)
            need_capture = bool(axis_plan.get("need_capture_after", action_symbol in YAW_IMMEDIATE_CAPTURE_SYMBOLS))
            phase = str(axis_plan.get("phase", "") or "axis_aligned_target_boundary")
        else:
            bearing = self._as_float_or_none(boundary.get("target_house_bearing_deg"))
            if bearing is not None and abs(bearing) > LLM_TARGET_BOUNDARY_ALIGN_TOLERANCE_DEG:
                action_symbol = "e" if bearing > 0.0 else "q"
                repeat = 1
                need_capture = True
                phase = "align_to_target_boundary"
            else:
                distance_cm = self._as_float_or_none(boundary.get("target_house_distance_cm")) or 0.0
                action_symbol = "w"
                repeat = 4 if distance_cm > 3500.0 else 3 if distance_cm > 2500.0 else 2
                repeat = min(self._llm_control_repeat_cap(), repeat)
                need_capture = False
                phase = "advance_to_target_boundary"

        original = dict(normalized)
        normalized = dict(normalized)
        normalized["action_symbol"] = action_symbol
        normalized["repeat"] = max(1, int(repeat))
        normalized["stop"] = False
        normalized["need_capture_after"] = need_capture
        normalized["target_boundary_context"] = boundary
        normalized["rule_override"] = {
            "type": "target_boundary_transit",
            "original_action_symbol": original.get("action_symbol"),
            "original_repeat": original.get("repeat"),
            "target_house_id": house_id,
            "phase": phase,
            "action_symbol": action_symbol,
            "repeat": int(repeat),
            "target_boundary_context": boundary,
            "axis_aligned_transit_plan": axis_plan,
            "front_obstacle": obstacle,
            "reason": "target house is still far outside its search boundary; use axis-aligned boundary transit to avoid diagonal crossing near intermediate houses",
        }
        reason = str(normalized.get("reason", "") or "")
        transit_reason = (
            f"target-boundary transit: {phase} with {action_symbol} x{int(repeat)}; "
            f"target_distance={boundary.get('target_house_distance_cm')}cm, "
            f"search_allowed_within={boundary.get('search_distance_cm')}cm"
        )
        if axis_plan:
            transit_reason += (
                f", axis={axis_plan.get('selected_axis')} "
                f"remaining={axis_plan.get('axis_remaining_to_boundary_cm')}cm"
            )
        normalized["reason"] = (reason + " | " if reason else "") + transit_reason
        if original.get("action_symbol") != action_symbol:
            self._append_llm_control_log(
                f"Target boundary transit override: action={original.get('action_symbol')} -> {action_symbol} repeat={int(repeat)}; "
                f"target={house_id} distance={boundary.get('target_house_distance_cm')} search_distance={boundary.get('search_distance_cm')} phase={phase}"
            )
        return normalized

    def _target_entry_visible_for_current_house(self, fusion: Dict[str, Any]) -> bool:
        target_state = str(fusion.get("target_conditioned_state", "") or "")
        if target_state in {"target_house_not_in_view", "non_target_house_entry_visible"}:
            return False
        if target_state in {"target_house_entry_visible", "target_house_entry_approachable", "target_house_entry_blocked", "target_house_geometric_opening_needs_confirmation"}:
            return True
        return bool(fusion.get("entry_visible", False))

    def _normalize_angle_deg(self, angle_deg: float) -> float:
        return ((float(angle_deg) + 180.0) % 360.0) - 180.0

    def _house_center_for_id(self, house_id: str) -> Dict[str, float]:
        hid = str(house_id or "").strip()
        if not hid:
            return {}
        raw = self._read_local_houses_config()
        houses = raw.get("houses", []) if isinstance(raw.get("houses"), list) else []
        for house in houses:
            if not isinstance(house, dict) or str(house.get("id", "") or "").strip() != hid:
                continue
            x = self._as_float_or_none(house.get("center_x"))
            y = self._as_float_or_none(house.get("center_y"))
            if x is None or y is None:
                return {}
            return {"x": x, "y": y}
        return {}

    def _house_radius_cm_for_id(self, house_id: str) -> Optional[float]:
        hid = str(house_id or "").strip()
        if not hid:
            return None
        raw = self._read_local_houses_config()
        houses = raw.get("houses", []) if isinstance(raw.get("houses"), list) else []
        for house in houses:
            if isinstance(house, dict) and str(house.get("id", "") or "").strip() == hid:
                return self._as_float_or_none(house.get("radius_cm"))
        return None

    def _house_records_for_route_planning(self) -> List[Dict[str, Any]]:
        raw = self._read_local_houses_config()
        houses = raw.get("houses", []) if isinstance(raw.get("houses"), list) else []
        records: List[Dict[str, Any]] = []
        for house in houses:
            if not isinstance(house, dict):
                continue
            house_id = str(house.get("id", "") or "").strip()
            x = self._as_float_or_none(house.get("center_x"))
            y = self._as_float_or_none(house.get("center_y"))
            radius = self._as_float_or_none(house.get("radius_cm"))
            if not house_id or x is None or y is None or radius is None:
                continue
            records.append({"id": house_id, "x": float(x), "y": float(y), "radius_cm": float(radius)})
        return records

    def _point_to_segment_distance_cm(
        self,
        px: float,
        py: float,
        ax: float,
        ay: float,
        bx: float,
        by: float,
    ) -> float:
        abx = float(bx) - float(ax)
        aby = float(by) - float(ay)
        apx = float(px) - float(ax)
        apy = float(py) - float(ay)
        denom = abx * abx + aby * aby
        if denom <= 1e-6:
            return math.hypot(apx, apy)
        t = max(0.0, min(1.0, (apx * abx + apy * aby) / denom))
        cx = float(ax) + t * abx
        cy = float(ay) + t * aby
        return math.hypot(float(px) - cx, float(py) - cy)

    def _axis_route_clearance(
        self,
        *,
        pose: Dict[str, float],
        target: Dict[str, float],
        first_axis: str,
        target_house_id: str,
        current_house_id: str = "",
    ) -> Dict[str, Any]:
        cx = float(pose["x"])
        cy = float(pose["y"])
        tx = float(target["x"])
        ty = float(target["y"])
        if first_axis == "x":
            waypoint = {"x": tx, "y": cy}
        else:
            waypoint = {"x": cx, "y": ty}
        segments = [
            {"from": {"x": cx, "y": cy}, "to": waypoint},
            {"from": waypoint, "to": {"x": tx, "y": ty}},
        ]
        ignored = {str(target_house_id or "").strip(), str(current_house_id or "").strip()}
        blockers: List[Dict[str, Any]] = []
        min_clearance: Optional[float] = None
        for house in self._house_records_for_route_planning():
            house_id = str(house.get("id", "") or "").strip()
            if not house_id or house_id in ignored:
                continue
            base_radius = float(house.get("radius_cm", 0.0) or 0.0)
            radius = base_radius + float(LLM_AXIS_TRANSIT_HOUSE_CLEARANCE_CM)
            if math.hypot(float(house["x"]) - cx, float(house["y"]) - cy) <= radius:
                continue
            for idx, segment in enumerate(segments):
                dist = self._point_to_segment_distance_cm(
                    float(house["x"]),
                    float(house["y"]),
                    float(segment["from"]["x"]),
                    float(segment["from"]["y"]),
                    float(segment["to"]["x"]),
                    float(segment["to"]["y"]),
                )
                clearance = dist - radius
                min_clearance = clearance if min_clearance is None else min(min_clearance, clearance)
                if clearance < 0.0:
                    blockers.append(
                        {
                            "house_id": house_id,
                            "segment_index": idx,
                            "distance_to_route_cm": round(float(dist), 2),
                            "clearance_cm": round(float(clearance), 2),
                        }
                    )
        return {
            "first_axis": first_axis,
            "waypoint": waypoint,
            "segments": segments,
            "blocker_count": len(blockers),
            "min_clearance_cm": round(float(min_clearance), 2) if min_clearance is not None else None,
            "blockers": blockers[:8],
        }

    def _choose_axis_route_order(
        self,
        *,
        pose: Dict[str, float],
        target: Dict[str, float],
        target_house_id: str,
        current_house_id: str = "",
    ) -> Dict[str, Any]:
        x_route = self._axis_route_clearance(
            pose=pose,
            target=target,
            first_axis="x",
            target_house_id=target_house_id,
            current_house_id=current_house_id,
        )
        y_route = self._axis_route_clearance(
            pose=pose,
            target=target,
            first_axis="y",
            target_house_id=target_house_id,
            current_house_id=current_house_id,
        )
        dx = abs(float(target["x"]) - float(pose["x"]))
        dy = abs(float(target["y"]) - float(pose["y"]))

        def route_key(route: Dict[str, Any]) -> tuple:
            min_clearance = self._as_float_or_none(route.get("min_clearance_cm"))
            return (
                -int(route.get("blocker_count", 0) or 0),
                min_clearance if min_clearance is not None else 1e9,
                dx if route.get("first_axis") == "x" else dy,
            )

        selected = x_route if route_key(x_route) >= route_key(y_route) else y_route
        fallback_axis = "x" if dx >= dy else "y"
        if int(x_route.get("blocker_count", 0) or 0) == int(y_route.get("blocker_count", 0) or 0):
            x_clearance = self._as_float_or_none(x_route.get("min_clearance_cm"))
            y_clearance = self._as_float_or_none(y_route.get("min_clearance_cm"))
            if x_clearance is not None and y_clearance is not None and abs(x_clearance - y_clearance) < 100.0:
                selected = x_route if fallback_axis == "x" else y_route
        return {
            "selected_first_axis": str(selected.get("first_axis", fallback_axis)),
            "fallback_axis": fallback_axis,
            "x_first": x_route,
            "y_first": y_route,
            "reason": "choose the axis-aligned route with fewer non-target house intersections, then larger clearance",
        }

    def _target_boundary_context(self, fusion: Dict[str, Any], house_id: str) -> Dict[str, Any]:
        distance_cm = self._target_house_distance_cm(fusion)
        radius_cm = self._house_radius_cm_for_id(house_id)
        search_distance_cm = max(
            LLM_TARGET_BOUNDARY_MIN_SEARCH_DISTANCE_CM,
            (radius_cm or 0.0) + LLM_TARGET_BOUNDARY_MARGIN_CM,
        )
        target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
        bearing = self._as_float_or_none(target_context.get("target_house_bearing_deg"))
        outside = distance_cm is not None and distance_cm > search_distance_cm
        return {
            "target_house_id": str(house_id or ""),
            "target_house_distance_cm": distance_cm,
            "target_radius_cm": radius_cm,
            "boundary_margin_cm": float(LLM_TARGET_BOUNDARY_MARGIN_CM),
            "search_distance_cm": float(search_distance_cm),
            "outside_search_boundary": bool(outside),
            "target_house_bearing_deg": bearing,
            "target_house_in_fov": bool(target_context.get("target_house_in_fov", False)),
            "mode": "transit_to_target_boundary" if outside else "entry_search_allowed",
        }

    def _axis_aligned_boundary_transit_plan(
        self,
        fusion: Dict[str, Any],
        house_id: str,
        boundary: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not bool(boundary.get("outside_search_boundary", False)):
            return {}
        pose = self._uav_pose_from_fusion_or_state(fusion)
        target = self._target_center_from_fusion_or_config(fusion, house_id)
        if not pose or not target:
            return {}
        search_distance = self._as_float_or_none(boundary.get("search_distance_cm"))
        if search_distance is None or search_distance <= 0.0:
            return {}

        target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
        current_house_id = str(target_context.get("current_house_id", "") or "").strip()
        route_choice = self._choose_axis_route_order(
            pose=pose,
            target=target,
            target_house_id=house_id,
            current_house_id=current_house_id,
        )

        dx = float(target["x"]) - float(pose["x"])
        dy = float(target["y"]) - float(pose["y"])
        abs_dx = abs(dx)
        abs_dy = abs(dy)
        first_axis = str(route_choice.get("selected_first_axis", "") or "x")

        def axis_remaining(axis: str) -> float:
            if axis == "x":
                other = abs_dy
                primary = abs_dx
            else:
                other = abs_dx
                primary = abs_dy
            if other < search_distance:
                allowed_primary = math.sqrt(max(0.0, search_distance * search_distance - other * other))
                return max(0.0, primary - allowed_primary + float(LLM_AXIS_TRANSIT_BOUNDARY_MARGIN_CM))
            return primary

        first_remaining = axis_remaining(first_axis)
        if first_remaining > float(LLM_AXIS_TRANSIT_WAYPOINT_TOLERANCE_CM):
            axis = first_axis
            remaining_cm = first_remaining
            phase = f"axis_{axis}_to_target_boundary"
        else:
            second_axis = "y" if first_axis == "x" else "x"
            second_remaining = axis_remaining(second_axis)
            axis = second_axis if second_remaining > float(LLM_AXIS_TRANSIT_WAYPOINT_TOLERANCE_CM) else first_axis
            remaining_cm = max(first_remaining, second_remaining)
            phase = f"axis_{axis}_finish_target_boundary"

        if axis == "x":
            desired_yaw = 0.0 if dx >= 0.0 else 180.0
            signed_axis_delta_cm = dx
        else:
            desired_yaw = 90.0 if dy >= 0.0 else -90.0
            signed_axis_delta_cm = dy
        current_yaw = float(pose["yaw"])
        yaw_delta = self._normalize_angle_deg(desired_yaw - current_yaw)
        if abs(yaw_delta) > float(LLM_AXIS_TRANSIT_ALIGN_TOLERANCE_DEG):
            action_symbol = "e" if yaw_delta > 0.0 else "q"
            repeat = 1
            need_capture = True
            phase = f"align_{phase}"
        else:
            action_symbol = "w"
            if remaining_cm > 2500.0:
                repeat = 4
            elif remaining_cm > 1200.0:
                repeat = 3
            else:
                repeat = 2
            repeat = min(self._llm_control_repeat_cap(), repeat)
            need_capture = False

        return {
            "mode": "axis_aligned_target_boundary_transit",
            "target_house_id": str(house_id or ""),
            "current_house_id": current_house_id,
            "selected_axis": axis,
            "selected_first_axis": first_axis,
            "signed_axis_delta_cm": round(float(signed_axis_delta_cm), 2),
            "axis_remaining_to_boundary_cm": round(float(remaining_cm), 2),
            "desired_yaw_deg": round(float(desired_yaw), 2),
            "current_yaw_deg": round(float(current_yaw), 2),
            "yaw_delta_deg": round(float(yaw_delta), 2),
            "action_symbol": action_symbol,
            "repeat": int(max(1, repeat)),
            "need_capture_after": bool(need_capture),
            "phase": phase,
            "pose": {"x": float(pose["x"]), "y": float(pose["y"]), "yaw": float(pose["yaw"])},
            "target": {"x": float(target["x"]), "y": float(target["y"])},
            "route_choice": route_choice,
            "reason": "use an axis-aligned route to approach the target search boundary before entrance/house judging, reducing diagonal crossing near intermediate houses",
        }

    def _target_center_from_fusion_or_config(self, fusion: Dict[str, Any], house_id: str) -> Dict[str, float]:
        target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
        center = target_context.get("target_house_center_world", {}) if isinstance(target_context.get("target_house_center_world"), dict) else {}
        x = self._as_float_or_none(center.get("x"))
        y = self._as_float_or_none(center.get("y"))
        if x is not None and y is not None:
            return {"x": x, "y": y}
        return self._house_center_for_id(house_id)

    def _uav_pose_from_fusion_or_state(self, fusion: Dict[str, Any]) -> Dict[str, float]:
        target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
        pose = target_context.get("uav_pose_world", {}) if isinstance(target_context.get("uav_pose_world"), dict) else {}
        x = self._as_float_or_none(pose.get("x"))
        y = self._as_float_or_none(pose.get("y"))
        yaw = self._as_float_or_none(pose.get("yaw"))
        if x is not None and y is not None and yaw is not None:
            return {"x": x, "y": y, "yaw": yaw}
        latest_pose = self.latest_state.get("pose", {}) if isinstance(self.latest_state.get("pose"), dict) else {}
        x = self._as_float_or_none(latest_pose.get("x"))
        y = self._as_float_or_none(latest_pose.get("y"))
        yaw = self._as_float_or_none(latest_pose.get("task_yaw", latest_pose.get("yaw")))
        if x is not None and y is not None and yaw is not None:
            return {"x": x, "y": y, "yaw": yaw}
        return {}

    def _geometric_target_reacquire_plan(self, fusion: Dict[str, Any], house_id: str) -> Dict[str, Any]:
        pose = self._uav_pose_from_fusion_or_state(fusion)
        target = self._target_center_from_fusion_or_config(fusion, house_id)
        if not pose or not target:
            return {}
        dx = float(target["x"]) - float(pose["x"])
        dy = float(target["y"]) - float(pose["y"])
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            return {}
        desired_yaw = math.degrees(math.atan2(dy, dx))
        current_yaw = float(pose["yaw"])
        delta = self._normalize_angle_deg(desired_yaw - current_yaw)
        abs_delta = abs(delta)
        if abs_delta <= LLM_TARGET_REACQUIRE_ALIGN_TOLERANCE_DEG:
            planned_steps = 0
            yaw_symbol = ""
        else:
            planned_steps = max(1, int(round(abs_delta / max(1.0, LLM_YAW_STEP_DEG))))
            planned_steps = min(planned_steps, int(LLM_TARGET_REACQUIRE_MAX_YAW_STEPS))
            yaw_symbol = "e" if delta > 0.0 else "q"
        return {
            "source": "world_geometry",
            "target_house_id": house_id,
            "uav_x": float(pose["x"]),
            "uav_y": float(pose["y"]),
            "uav_yaw_deg": current_yaw,
            "target_x": float(target["x"]),
            "target_y": float(target["y"]),
            "desired_yaw_deg": desired_yaw,
            "signed_delta_deg": delta,
            "abs_delta_deg": abs_delta,
            "yaw_symbol": yaw_symbol,
            "planned_yaw_steps": planned_steps,
            "yaw_step_deg": float(LLM_YAW_STEP_DEG),
            "align_tolerance_deg": float(LLM_TARGET_REACQUIRE_ALIGN_TOLERANCE_DEG),
        }

    def _current_target_house_id_from_fusion(self, fusion: Dict[str, Any]) -> str:
        active = self._current_llm_plan_target()
        house_id = str(active.get("house_id", "") or "").strip()
        if house_id:
            return house_id
        target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
        return str(target_context.get("target_house_id", "") or "").strip()

    def _action_hint_to_yaw_symbol(self, hint: Any) -> str:
        text = str(hint or "").strip().lower().replace("-", "_").replace(" ", "_")
        symbol = LLM_CONTROL_ACTION_ALIASES.get(text, text)
        return symbol if symbol in YAW_IMMEDIATE_CAPTURE_SYMBOLS else ""

    def _target_reacquire_needed(self, fusion: Dict[str, Any], *, active_lock: bool = False) -> bool:
        if self._target_entry_visible_for_current_house(fusion):
            return False
        if active_lock:
            return True
        target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
        if bool(target_context.get("target_house_in_fov", False)):
            return False
        target_state = str(fusion.get("target_conditioned_state", "") or "")
        expected_side = str(target_context.get("target_house_expected_side", "") or "")
        return target_state == "target_house_not_in_view" or expected_side == "out_of_view"

    def _preferred_reacquire_yaw_symbol(self, fusion: Dict[str, Any], normalized: Dict[str, Any]) -> str:
        for key in ("target_conditioned_action_hint", "recommended_action_hint"):
            symbol = self._action_hint_to_yaw_symbol(fusion.get(key))
            if symbol:
                return symbol
        symbol = str(normalized.get("action_symbol", "") or "")
        if symbol in YAW_IMMEDIATE_CAPTURE_SYMBOLS:
            return symbol
        target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
        bearing = self._as_float_or_none(target_context.get("target_house_bearing_deg"))
        if bearing is not None:
            return "q" if bearing >= 0.0 else "e"
        return LLM_TARGET_REACQUIRE_DEFAULT_YAW_SYMBOL

    def _apply_target_reacquire_yaw_lock(self, normalized: Dict[str, Any], labeling_dir: str) -> Dict[str, Any]:
        if bool(normalized.get("stop", False)):
            return normalized
        fusion = self._fusion_payload_from_labeling_dir(labeling_dir)
        house_id = self._current_target_house_id_from_fusion(fusion)
        existing_lock = self.llm_target_reacquire_lock if isinstance(self.llm_target_reacquire_lock, dict) else {}
        lock_for_same_house = str(existing_lock.get("target_house_id", "") or "") == house_id
        active_lock = lock_for_same_house and int(existing_lock.get("yaw_steps", 0) or 0) < int(existing_lock.get("planned_yaw_steps", existing_lock.get("max_yaw_steps", 0)) or 0)
        if house_id:
            boundary = self._target_boundary_context(fusion, house_id)
            if bool(boundary.get("outside_search_boundary", False)) and self._axis_aligned_boundary_transit_plan(fusion, house_id, boundary):
                if lock_for_same_house:
                    self.llm_target_reacquire_lock = {}
                return normalized
        if not house_id or not self._target_reacquire_needed(fusion, active_lock=active_lock):
            if house_id and str(self.llm_target_reacquire_lock.get("target_house_id", "") or "") == house_id:
                self.llm_target_reacquire_lock = {}
            return normalized

        metadata = self._metadata_from_labeling_dir(labeling_dir)
        step_index = int(metadata.get("step_index", 0) or 0)
        lock = existing_lock
        if str(lock.get("target_house_id", "") or "") != house_id or str(lock.get("yaw_symbol", "") or "") not in YAW_IMMEDIATE_CAPTURE_SYMBOLS:
            target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
            geometry_plan = self._geometric_target_reacquire_plan(fusion, house_id)
            yaw_symbol = str(geometry_plan.get("yaw_symbol", "") or "") if geometry_plan else ""
            planned_steps = int(geometry_plan.get("planned_yaw_steps", 0) or 0) if geometry_plan else 0
            plan_source = str(geometry_plan.get("source", "") or "")
            if not yaw_symbol or planned_steps <= 0:
                yaw_symbol = self._preferred_reacquire_yaw_symbol(fusion, normalized)
                planned_steps = int(LLM_TARGET_REACQUIRE_MAX_YAW_STEPS)
                plan_source = "fallback_full_scan"
            lock = {
                "target_house_id": house_id,
                "yaw_symbol": yaw_symbol,
                "yaw_steps": 0,
                "planned_yaw_steps": max(1, planned_steps),
                "max_yaw_steps": max(1, planned_steps),
                "plan_source": plan_source,
                "geometry_plan": geometry_plan,
                "started_at_step": step_index,
                "initial_target_bearing_deg": target_context.get("target_house_bearing_deg"),
                "initial_action_hint": fusion.get("target_conditioned_action_hint") or fusion.get("recommended_action_hint"),
            }

        yaw_symbol = str(lock.get("yaw_symbol", "") or LLM_TARGET_REACQUIRE_DEFAULT_YAW_SYMBOL)
        yaw_steps = int(lock.get("yaw_steps", 0) or 0)
        planned_steps = max(1, int(lock.get("planned_yaw_steps", lock.get("max_yaw_steps", LLM_TARGET_REACQUIRE_MAX_YAW_STEPS)) or LLM_TARGET_REACQUIRE_MAX_YAW_STEPS))
        if yaw_steps >= planned_steps:
            self.llm_target_reacquire_lock = {}
            return normalized

        original = dict(normalized)
        lock["yaw_steps"] = yaw_steps + 1
        lock["last_step_index"] = step_index
        self.llm_target_reacquire_lock = lock
        normalized = dict(normalized)
        normalized["action_symbol"] = yaw_symbol
        normalized["repeat"] = 1
        normalized["stop"] = False
        normalized["need_capture_after"] = True
        normalized["target_reacquire_lock"] = dict(lock)
        normalized["rule_override"] = {
            "type": "geometric_target_reacquire_yaw_plan",
            "original_action_symbol": original.get("action_symbol"),
            "original_repeat": original.get("repeat"),
            "target_house_id": house_id,
            "yaw_symbol": yaw_symbol,
            "yaw_steps": int(lock.get("yaw_steps", 0) or 0),
            "planned_yaw_steps": planned_steps,
            "max_yaw_steps": planned_steps,
            "plan_source": str(lock.get("plan_source", "") or ""),
            "geometry_plan": lock.get("geometry_plan", {}),
            "target_state": str(fusion.get("target_conditioned_state", "") or ""),
            "target_bearing_deg": (fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}).get("target_house_bearing_deg"),
            "reason": "target house is not in view; execute the precomputed geometric yaw plan before judging again",
        }
        reason = str(normalized.get("reason", "") or "")
        lock_reason = (
            f"geometric reacquire plan keeps yaw {'left' if yaw_symbol == 'q' else 'right'} "
            f"for house {house_id} step {lock['yaw_steps']}/{planned_steps}"
        )
        normalized["reason"] = (reason + " | " if reason else "") + lock_reason
        if original.get("action_symbol") != yaw_symbol:
            self._append_llm_control_log(
                f"Geometric reacquire yaw plan: action={original.get('action_symbol')} -> {yaw_symbol}; target={house_id} step={lock['yaw_steps']}/{planned_steps}"
            )
        return normalized

    def _memory_completion_for_house(self, labeling_dir: str, house_id: str) -> Dict[str, Any]:
        snapshot = self._read_json_file(Path(labeling_dir) / "entry_search_memory_snapshot_after.json")
        memory = snapshot.get("memory", {}) if isinstance(snapshot.get("memory"), dict) else {}
        memories = memory.get("memories", {}) if isinstance(memory.get("memories"), dict) else {}
        house_memory = memories.get(str(house_id), {}) if isinstance(memories, dict) else {}
        semantic = house_memory.get("semantic_memory", {}) if isinstance(house_memory.get("semantic_memory"), dict) else {}
        completion = semantic.get("search_completion_evidence", {}) if isinstance(semantic.get("search_completion_evidence"), dict) else {}
        return completion

    def _best_reliable_memory_entry_for_house(self, labeling_dir: str, house_id: str) -> Dict[str, Any]:
        snapshot = self._read_json_file(Path(labeling_dir) / "entry_search_memory_snapshot_after.json")
        memory = snapshot.get("memory", {}) if isinstance(snapshot.get("memory"), dict) else {}
        memories = memory.get("memories", {}) if isinstance(memory.get("memories"), dict) else {}
        house_memory = memories.get(str(house_id), {}) if isinstance(memories, dict) else {}
        semantic = house_memory.get("semantic_memory", {}) if isinstance(house_memory.get("semantic_memory"), dict) else {}
        entries = semantic.get("candidate_entries", []) if isinstance(semantic.get("candidate_entries"), list) else []
        best: Dict[str, Any] = {}
        best_score = -1.0
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("associated_house_id", "") or "") != str(house_id):
                continue
            if not self._is_doorlike_class(str(entry.get("entry_type", "") or "")):
                continue
            assoc = self._as_float_or_none(entry.get("association_confidence")) or 0.0
            match = self._as_float_or_none(entry.get("target_match_score")) or 0.0
            obs = self._as_float_or_none(entry.get("observation_count")) or 0.0
            score = assoc + match + min(obs, 5.0) * 0.05
            if score > best_score:
                best = entry
                best_score = score
        if not best:
            return {}
        assoc = self._as_float_or_none(best.get("association_confidence")) or 0.0
        match = self._as_float_or_none(best.get("target_match_score")) or 0.0
        obs = self._as_float_or_none(best.get("observation_count")) or 0.0
        if assoc >= 0.55 and obs >= 2 and match >= 0.45:
            return best
        return {}

    def _evaluate_llm_house_completion(self, labeling_dir: str) -> Dict[str, Any]:
        metadata = self._metadata_from_labeling_dir(labeling_dir)
        fusion = self._fusion_payload_from_labeling_dir(labeling_dir)
        active_target = self._current_llm_plan_target()
        house_id = str(active_target.get("house_id") or metadata.get("target_house_id") or "").strip()
        step_index = int(metadata.get("step_index", 0) or 0)
        if not house_id:
            return {"completed": False, "reason": "no_active_house_id"}

        memory_completion = self._memory_completion_for_house(labeling_dir, house_id)
        if bool(memory_completion.get("no_entry_after_full_coverage", False)):
            return {
                "completed": True,
                "finish_type": "no_entry_after_full_coverage",
                "house_id": house_id,
                "step_index": step_index,
                "source_labeling_dir": str(labeling_dir),
                "reason": "memory reports no entry after full coverage",
                "memory_completion": memory_completion,
            }

        distance_cm = self._candidate_entry_distance_cm(fusion)
        class_name = self._semantic_class_name(fusion)
        target_entry_visible = self._target_entry_visible_for_current_house(fusion)
        target_house_distance_cm = self._target_house_distance_cm(fusion)
        bbox_metrics = self._semantic_bbox_metrics(fusion)
        large_entry_bbox = self._is_large_near_entry_bbox(fusion)
        reliable_memory_entry = self._best_reliable_memory_entry_for_house(labeling_dir, house_id)
        front_path_clear = self._front_path_clear_for_approach(fusion)
        threshold = float((self.llm_task_plan.get("execution_policy", {}) if isinstance(self.llm_task_plan, dict) else {}).get("entry_reached_distance_cm", LLM_ENTRY_REACHED_DISTANCE_CM))
        if (
            target_entry_visible
            and self._is_doorlike_class(class_name)
            and distance_cm is not None
            and distance_cm <= threshold
        ):
            return {
                "completed": True,
                "finish_type": "target_entry_reached",
                "house_id": house_id,
                "step_index": step_index,
                "source_labeling_dir": str(labeling_dir),
                "entry_distance_cm": float(distance_cm),
                "distance_threshold_cm": threshold,
                "semantic_class": class_name,
                "reason": f"target-house door-like entry reached within {threshold:.0f}cm",
            }
        if (
            target_entry_visible
            and self._is_doorlike_class(class_name)
            and large_entry_bbox
            and front_path_clear
            and (
                (target_house_distance_cm is not None and target_house_distance_cm <= LLM_ENTRY_REACHED_LARGE_BBOX_PROXY_CM)
                or (distance_cm is not None and distance_cm <= LLM_ENTRY_REACHED_LARGE_BBOX_PROXY_CM)
            )
        ):
            return {
                "completed": True,
                "finish_type": "target_entry_reached_large_bbox_proxy",
                "house_id": house_id,
                "step_index": step_index,
                "source_labeling_dir": str(labeling_dir),
                "entry_distance_cm": distance_cm,
                "target_house_distance_cm": target_house_distance_cm,
                "distance_threshold_cm": threshold,
                "large_bbox_proxy_cm": float(LLM_ENTRY_REACHED_LARGE_BBOX_PROXY_CM),
                "semantic_class": class_name,
                "bbox_metrics": bbox_metrics,
                "reason": "target-house entry fills most of the view and the UAV is close enough to treat the doorway as reached",
            }
        if (
            reliable_memory_entry
            and target_house_distance_cm is not None
            and target_house_distance_cm <= LLM_ENTRY_REACHED_POSE_PROXY_CM
        ):
            return {
                "completed": True,
                "finish_type": "target_entry_reached_pose_memory_proxy",
                "house_id": house_id,
                "step_index": step_index,
                "source_labeling_dir": str(labeling_dir),
                "entry_distance_cm": distance_cm,
                "target_house_distance_cm": target_house_distance_cm,
                "pose_proxy_cm": float(LLM_ENTRY_REACHED_POSE_PROXY_CM),
                "semantic_class": class_name,
                "reliable_memory_entry_id": str(reliable_memory_entry.get("entry_id", "") or ""),
                "reliable_memory_entry_type": str(reliable_memory_entry.get("entry_type", "") or ""),
                "reason": "semantic door evidence became unreliable at close range, but remembered target-house entry and pose distance indicate the entry is reached",
            }
        return {
            "completed": False,
            "house_id": house_id,
            "step_index": step_index,
            "entry_distance_cm": distance_cm,
            "target_house_distance_cm": target_house_distance_cm,
            "semantic_class": class_name,
            "target_entry_visible": target_entry_visible,
            "large_entry_bbox": large_entry_bbox,
            "bbox_metrics": bbox_metrics,
            "front_path_clear": front_path_clear,
            "reliable_memory_entry_id": str(reliable_memory_entry.get("entry_id", "") or "") if reliable_memory_entry else "",
        }

    def _should_override_to_forward_approach(self, normalized: Dict[str, Any], labeling_dir: str) -> Dict[str, Any]:
        fusion = self._fusion_payload_from_labeling_dir(labeling_dir)
        distance_cm = self._candidate_entry_distance_cm(fusion)
        class_name = self._semantic_class_name(fusion)
        target_entry_visible = self._target_entry_visible_for_current_house(fusion)
        if not target_entry_visible or not self._is_doorlike_class(class_name):
            return {"override": False}
        if distance_cm is None or distance_cm <= LLM_APPROACHABLE_DISTANCE_MARGIN_CM:
            return {"override": False}
        if not self._front_path_clear_for_approach(fusion):
            return {"override": False}
        current_symbol = str(normalized.get("action_symbol", "") or "")
        if current_symbol == "w":
            return {"override": False}
        return {
            "override": True,
            "distance_cm": float(distance_cm),
            "semantic_class": class_name,
            "target_state": str(fusion.get("target_conditioned_state", "") or ""),
            "reason": "target-house door is visible and farther than 3m; cross_ready=false does not block approach",
        }

    def _apply_llm_control_rule_overrides(self, normalized: Dict[str, Any], labeling_dir: str) -> Dict[str, Any]:
        if bool(normalized.get("stop", False)):
            return normalized
        normalized = self._apply_target_reacquire_yaw_lock(normalized, labeling_dir)
        if bool(normalized.get("stop", False)):
            return normalized
        rule_type = str((normalized.get("rule_override", {}) if isinstance(normalized.get("rule_override"), dict) else {}).get("type", "") or "")
        if rule_type == "geometric_target_reacquire_yaw_plan" and str(normalized.get("action_symbol", "") or "") in YAW_IMMEDIATE_CAPTURE_SYMBOLS:
            fusion = self._fusion_payload_from_labeling_dir(labeling_dir)
            obstacle = self._front_obstacle_status(fusion, labeling_dir)
            if bool(obstacle.get("too_close", False)):
                return self._apply_front_obstacle_detour_override(normalized, labeling_dir)
            return normalized
        normalized = self._apply_front_obstacle_detour_override(normalized, labeling_dir)
        if bool(normalized.get("stop", False)) or str(normalized.get("action_symbol", "") or "") in YAW_IMMEDIATE_CAPTURE_SYMBOLS:
            return normalized
        normalized = self._apply_target_boundary_transit_override(normalized, labeling_dir)
        if bool(normalized.get("stop", False)) or str(normalized.get("action_symbol", "") or "") in YAW_IMMEDIATE_CAPTURE_SYMBOLS:
            return normalized
        override = self._should_override_to_forward_approach(normalized, labeling_dir)
        if not bool(override.get("override", False)):
            return normalized
        distance_cm = float(override.get("distance_cm", 0.0) or 0.0)
        repeat = 1
        if distance_cm > 800.0:
            repeat = min(self._llm_control_repeat_cap(), 3)
        elif distance_cm > 450.0:
            repeat = min(self._llm_control_repeat_cap(), 2)
        original = dict(normalized)
        normalized = dict(normalized)
        normalized["action_symbol"] = "w"
        normalized["repeat"] = max(1, int(repeat))
        normalized["rule_override"] = {
            "type": "approach_visible_target_entry",
            "original_action_symbol": original.get("action_symbol"),
            "original_repeat": original.get("repeat"),
            **override,
        }
        reason = str(normalized.get("reason", "") or "")
        normalized["reason"] = (reason + " | " if reason else "") + str(override.get("reason", "rule override to forward approach"))
        self._append_llm_control_log(
            f"Rule override: action={original.get('action_symbol')} -> w repeat={repeat}; distance={distance_cm:.1f}cm"
        )
        return normalized

    def _llm_control_delay_s(self) -> float:
        try:
            return max(0.05, float(self.llm_control_delay_ms_var.get().strip()) / 1000.0)
        except ValueError:
            return 0.35

    def _llm_control_repeat_cap(self) -> int:
        try:
            return max(1, min(12, int(float(self.llm_control_repeat_cap_var.get().strip()))))
        except ValueError:
            return 4

    def _llm_control_timeout_s(self) -> float:
        try:
            return max(10.0, float(self.llm_control_timeout_s_var.get().strip()))
        except ValueError:
            return 60.0

    def _run_llm_control_capture(self, *, capture_source: str, note: str) -> str:
        wait_deadline = time.time() + 240.0
        while self.memory_capture_inflight and not self.llm_control_stop_event.is_set() and time.time() < wait_deadline:
            time.sleep(0.1)
        if self.llm_control_stop_event.is_set():
            return ""
        self._set_llm_control_var(self.llm_control_status_var, f"LLM Control: capturing {capture_source}")
        ok = self._run_memory_capture_analyze(capture_source=capture_source, note=note, update_status=False)
        response = self.last_memory_capture_response if isinstance(self.last_memory_capture_response, dict) else {}
        labeling_dir = str(response.get("labeling_dir", "") or "")
        if ok and labeling_dir:
            self._append_llm_control_log(f"Capture ready: {Path(labeling_dir).parent.name}")
            return labeling_dir
        message = str(response.get("message", "") or "unknown capture error")
        self._append_llm_control_log(f"Capture failed: {message}")
        return ""

    def _build_llm_control_prompt_payload(self, labeling_dir: str, decision_index: int) -> Dict[str, Any]:
        from phase2_multimodal_fusion_analysis.memory_aware_llm_teacher_prompt_builder import build_prompt_payload

        teacher_payload = build_prompt_payload(Path(labeling_dir), memory_snapshot="after")
        structured_input = teacher_payload.get("structured_input", {}) if isinstance(teacher_payload.get("structured_input"), dict) else {}
        fusion = self._fusion_payload_from_labeling_dir(labeling_dir)
        house_id = self._current_target_house_id_from_fusion(fusion)
        target_boundary_context = self._target_boundary_context(fusion, house_id) if house_id else {}
        axis_aligned_transit_plan = self._axis_aligned_boundary_transit_plan(fusion, house_id, target_boundary_context) if house_id else {}
        height_aware_front_obstacle = self._front_obstacle_status(fusion, labeling_dir)
        return {
            "version": "memory_aware_llm_control_prompt_v1",
            "prompt_type": "memory_aware_direct_control",
            "decision_index": int(decision_index),
            "labeling_dir": str(labeling_dir),
            "allowed_action_symbols": sorted(LLM_CONTROL_ALLOWED_SYMBOLS),
            "continuous_action_symbols": sorted(LLM_CONTROL_CONTINUOUS_SYMBOLS),
            "single_action_symbols": sorted(LLM_CONTROL_SINGLE_SYMBOLS),
            "repeat_cap": self._llm_control_repeat_cap(),
            "output_schema": LLM_CONTROL_OUTPUT_SCHEMA,
            "active_task_plan": {
                "applied": bool(self.llm_task_plan_applied),
                "current_target_index": int(self.llm_task_current_index),
                "current_target": self._current_llm_plan_target(),
                "ordered_targets": self.llm_task_plan.get("ordered_targets", []) if isinstance(self.llm_task_plan, dict) else [],
                "entry_reached_distance_cm": float(LLM_ENTRY_REACHED_DISTANCE_CM),
                "entry_reached_pose_proxy_cm": float(LLM_ENTRY_REACHED_POSE_PROXY_CM),
                "entry_reached_large_bbox_proxy_cm": float(LLM_ENTRY_REACHED_LARGE_BBOX_PROXY_CM),
            },
            "target_reacquire_lock": dict(self.llm_target_reacquire_lock) if isinstance(self.llm_target_reacquire_lock, dict) else {},
            "obstacle_detour_lock": dict(self.llm_obstacle_detour_lock) if isinstance(self.llm_obstacle_detour_lock, dict) else {},
            "target_boundary_context": target_boundary_context,
            "axis_aligned_transit_plan": axis_aligned_transit_plan,
            "height_aware_front_obstacle": height_aware_front_obstacle,
            "completion_rules": {
                "target_entry_reached": "Finish the current house when a reliable target-house door/open door/close door is within 300cm.",
                "large_bbox_proxy": "Also treat the current house as reached when a target-house door/open door/close door fills most of the image and target-house distance is near the doorway, because close-range depth may overestimate.",
                "pose_memory_proxy": "If semantic detection disappears at close range, a reliable remembered target-house entry plus target-house distance below about 350cm is enough to finish the current house.",
                "geometric_reacquire": "When a new target house is outside the view, the executor computes the world-angle difference from UAV pose to the target house center and locks q/e for the planned yaw step count before judging again.",
                "front_obstacle_detour": "If front_obstacle is high/severe or front_min_depth is very close, do not use yaw-only observation or forward motion as detour; physically back off and strafe left/right until the front path clears.",
                "low_obstacle_passable": "If height_aware_front_obstacle.low_obstacle_passable=true, the near depth is only in the lower image band and the UAV body-height corridor is clear; do not treat it as a blocking front obstacle.",
                "target_boundary_transit": "If target_boundary_context.mode=transit_to_target_boundary, ignore intermediate non-target house entrances and follow axis_aligned_transit_plan first; start entrance/house judging only after reaching the target search boundary.",
                "approach_vs_cross": "cross_ready=false means do not enter/cross the doorway yet; it does not mean you cannot approach the doorway.",
                "forward_preference": "If a target-house entry is visible, associated with the active target, farther than 300cm, and the front path is not severely blocked, prefer forward.",
            },
            "structured_observation": structured_input,
            "recent_llm_decisions": self.llm_control_decision_history[-6:],
        }

    def _build_llm_control_system_prompt(self) -> str:
        return (
            "You are a memory-aware UAV control pilot. You directly choose low-level control symbols "
            "for collecting target-house entry search data.\n"
            "Use the provided YOLO/RGB evidence, depth traversability evidence, target-house context, "
            "temporal actions, and structured memory.\n"
            "Allowed controls: w=forward, s=backward, a=left, d=right, r=up, f=down, q=yaw_left, "
            "e=yaw_right, x=hold.\n"
            "w/a/s/d/r/f may use repeat>1 for a short continuous move. q/e must always use repeat=1 "
            "because yaw changes the view and the controller will capture immediately after yaw.\n"
            "Do not chase non-target entries. If the target house is not in view, reorient with q/e.\n"
            "When switching to a new target house that is outside the field of view, keep yawing in the same "
            "geometrically planned direction for the planned number of yaw steps. The executor estimates this from the "
            "UAV world pose and the target house center. Do not alternate q/e based on short-term noisy bearing wrap-around.\n"
            "Important: cross_ready=false only means do not cross/enter the doorway yet. It does not mean you cannot "
            "approach a visible target-house door. If a target-house door/open door/close door is visible and the UAV "
            "is farther than 3m from it, prefer small forward repeats when the front path is not severely blocked.\n"
            "Treat rule-based labels such as target_house_entry_blocked, blocked_temporary, or persistent_blocked_shift "
            "as warnings against crossing the doorway, not as automatic reasons to stop approaching. If global_front_obstacle.present=false "
            "and the target door is still around 3-12m away, forward approach is usually the preferred data-collection action.\n"
            "If global_front_obstacle.present=true with high/severe severity or front_min_depth is around 120cm or less, "
            "do not keep moving forward and do not treat yaw-only observation as a real detour. First back off if the "
            "obstacle is very close, then strafe left/right according to detour guidance until the front path clears.\n"
            "Depth is height-sensitive: if height_aware_front_obstacle.low_obstacle_passable=true, the near depth is "
            "concentrated in the lower image band while the UAV body-height corridor is clear. In that case do not "
            "treat bushes/curbs/low foreground objects as a hard front blockage; continue target-boundary transit or "
            "approach cautiously instead of backoff/side oscillation.\n"
            "Before the UAV reaches the target house boundary/search radius, treat visible door/window evidence from "
            "nearby buildings as intermediate distractors. Do not start target-entry search or chase those candidates; "
            "follow axis_aligned_transit_plan first. Prefer x-first or y-first horizontal/vertical world-axis movement "
            "that has fewer non-target house intersections; avoid diagonal shortcuts through intermediate houses. "
            "Only after reaching the target search boundary should you begin detailed building/entry judgment.\n"
            "The current house search is finished when the UAV is within about 300cm of a reliable target-house entry; "
            "the executor will switch to the next house. Close-range depth can be noisy: if the target-house entry bbox "
            "fills most of the image or a reliable remembered target-house entry exists while the target-house distance "
            "is near 3-5m, treat it as reached rather than continuing through the doorway. Avoid left/right oscillation; "
            "if lateral moves do not improve evidence, move forward, back off, yaw to a new sector, or stop for review.\n"
            "If the search appears complete or unsafe, set stop=true.\n"
            "Return only one JSON object. No markdown. No commentary."
        )

    def _build_llm_control_user_prompt(self, payload: Dict[str, Any]) -> str:
        return "\n".join(
            [
                "Task:",
                "Directly control the UAV to search for and approach the entrance of the target house while collecting memory-aware training data.",
                "",
                "Controller behavior:",
                "The chosen action will be executed before the next visual analysis. Continuous moves repeat without intermediate screenshots. q/e yaw actions are single-step and trigger an immediate capture.",
                "",
                "Control input:",
                json.dumps(payload, indent=2, ensure_ascii=False),
                "",
                "Return strict JSON with this schema:",
                json.dumps(LLM_CONTROL_OUTPUT_SCHEMA, indent=2, ensure_ascii=False),
            ]
        )

    def _safe_llm_control_path_fragment(self, value: str) -> str:
        text = str(value or "").strip()
        safe = "".join(ch if (ch.isalnum() or ch in {"-", "_", "."}) else "_" for ch in text)
        return safe[:120] or "unknown"

    def _llm_control_session_dir_for_current_episode(self) -> Path:
        memory_collection = self.latest_memory_collection_state if isinstance(self.latest_memory_collection_state, dict) else {}
        collection_dir_text = str(memory_collection.get("collection_dir", "") or "").strip()
        episode_id = str(memory_collection.get("episode_id", "") or "").strip()
        root = Path(PROJECT_ROOT) / "captures_remote" / "llm_control_sessions"
        if collection_dir_text:
            collection_dir = Path(collection_dir_text)
            if collection_dir.parent.name == "memory_collection_sessions":
                root = collection_dir.parent.parent / "llm_control_sessions"
                episode_id = episode_id or collection_dir.name
        if not episode_id:
            episode_id = f"no_episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        session_dir = root / self._safe_llm_control_path_fragment(episode_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _resolve_llm_control_capture_root(self, labeling_dir: str) -> Path:
        labeling_path = Path(labeling_dir).resolve()
        for parent in labeling_path.parents:
            if parent.name == "memory_collection_sessions":
                return parent.parent / "llm_control_sessions"
        memory_collection = self.latest_memory_collection_state if isinstance(self.latest_memory_collection_state, dict) else {}
        collection_dir = Path(str(memory_collection.get("collection_dir", "") or "")).resolve()
        if collection_dir:
            for parent in collection_dir.parents:
                if parent.name == "captures_remote":
                    return parent / "llm_control_sessions"
            if collection_dir.parent.name == "memory_collection_sessions":
                return collection_dir.parent.parent / "llm_control_sessions"
        return Path(PROJECT_ROOT) / "captures_remote" / "llm_control_sessions"

    def _resolve_llm_control_episode_id(self, labeling_dir: str) -> str:
        labeling_path = Path(labeling_dir).resolve()
        for parent in labeling_path.parents:
            if parent.parent.name == "memory_collection_sessions":
                return self._safe_llm_control_path_fragment(parent.name)
        memory_collection = self.latest_memory_collection_state if isinstance(self.latest_memory_collection_state, dict) else {}
        episode_id = str(memory_collection.get("episode_id", "") or "")
        return self._safe_llm_control_path_fragment(episode_id or "no_episode")

    def _llm_control_artifact_dir(self, labeling_dir: str, decision_index: int) -> Path:
        root = self._resolve_llm_control_capture_root(labeling_dir)
        episode_id = self._resolve_llm_control_episode_id(labeling_dir)
        capture_name = self._safe_llm_control_path_fragment(Path(labeling_dir).parent.name)
        artifact_dir = root / episode_id / f"decision_{int(decision_index):04d}_{capture_name}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        return artifact_dir

    def _sync_llm_control_labeling_inputs(self, labeling_dir: str, artifact_dir: Path) -> Dict[str, Any]:
        source_dir = Path(labeling_dir)
        target_dir = artifact_dir / "labeling_inputs"
        target_dir.mkdir(parents=True, exist_ok=True)
        copied: List[str] = []
        missing: List[str] = []
        for filename in LLM_CONTROL_LABELING_INPUT_FILES:
            source_path = source_dir / filename
            if source_path.is_file():
                shutil.copy2(source_path, target_dir / filename)
                copied.append(filename)
            else:
                missing.append(filename)
        manifest = {
            "version": "llm_control_labeling_inputs_v1",
            "source_labeling_dir": str(source_dir),
            "labeling_inputs_dir": str(target_dir),
            "copied_files": copied,
            "missing_files": missing,
            "note": "Only key labeling artifacts are mirrored here; raw capture folders/images are not duplicated.",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        (target_dir / "labeling_inputs_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return manifest

    def _request_llm_control_decision(self, *, labeling_dir: str, decision_index: int) -> Dict[str, Any]:
        from anthropic import Anthropic
        from phase2_multimodal_fusion_analysis.memory_aware_llm_teacher_label_validator import extract_json_object

        prompt_payload = self._build_llm_control_prompt_payload(labeling_dir, decision_index)
        system_prompt = self._build_llm_control_system_prompt()
        user_prompt = self._build_llm_control_user_prompt(prompt_payload)
        artifact_dir = self._llm_control_artifact_dir(labeling_dir, decision_index)
        if isinstance(self.llm_task_plan, dict) and self.llm_task_plan:
            self._save_llm_task_plan_files(labeling_dir)
        labeling_inputs_manifest = self._sync_llm_control_labeling_inputs(labeling_dir, artifact_dir)
        prompt_path = artifact_dir / "llm_control_prompt.json"
        prompt_path.write_text(
            json.dumps(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "control_prompt_payload": prompt_payload,
                    "source_labeling_dir": str(labeling_dir),
                    "labeling_inputs_manifest": labeling_inputs_manifest,
                    "artifact_dir": str(artifact_dir),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        self._set_llm_control_var(self.llm_control_status_var, "LLM Control: waiting for LLM")
        start_time = time.time()
        client = Anthropic(
            api_key=self.llm_control_api_key_var.get().strip(),
            base_url=self.llm_control_base_url_var.get().strip(),
            timeout=self._llm_control_timeout_s(),
        )
        response = client.messages.create(
            model=self.llm_control_model_var.get().strip(),
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": [{"type": "text", "text": user_prompt}]}],
        )
        latency_ms = (time.time() - start_time) * 1000.0
        raw_text = self._extract_anthropic_text(response)
        parsed = extract_json_object(raw_text)
        response_payload = {
            "api_style": "anthropic_sdk_text",
            "model_name": self.llm_control_model_var.get().strip(),
            "latency_ms": round(float(latency_ms), 3),
            "raw_text": raw_text,
            "parsed": parsed,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "source_labeling_dir": str(labeling_dir),
            "artifact_dir": str(artifact_dir),
        }
        (artifact_dir / "llm_control_response.json").write_text(
            json.dumps(response_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if not parsed:
            raise RuntimeError("LLM returned no parseable JSON control decision.")
        return parsed

    def _extract_anthropic_text(self, response: Any) -> str:
        content = getattr(response, "content", None)
        if not isinstance(content, list):
            return ""
        parts: List[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
        return "\n".join(parts).strip()

    def _as_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        text = str(value or "").strip().lower()
        if text in {"1", "true", "yes", "y", "stop"}:
            return True
        if text in {"0", "false", "no", "n", ""}:
            return False
        return False

    def _normalize_llm_control_decision(self, decision: Dict[str, Any]) -> Dict[str, Any]:
        raw_symbol = str(
            decision.get("action_symbol")
            or decision.get("symbol")
            or decision.get("action")
            or decision.get("target_conditioned_action_hint")
            or "x"
        ).strip().lower().replace(" ", "_")
        symbol = LLM_CONTROL_ACTION_ALIASES.get(raw_symbol, raw_symbol)
        if symbol not in LLM_CONTROL_ALLOWED_SYMBOLS:
            symbol = "x"
        try:
            repeat = int(float(decision.get("repeat", 1)))
        except Exception:
            repeat = 1
        if symbol in LLM_CONTROL_CONTINUOUS_SYMBOLS:
            repeat = max(1, min(self._llm_control_repeat_cap(), repeat))
        else:
            repeat = 1
        try:
            confidence = float(decision.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        return {
            "action_symbol": symbol,
            "repeat": repeat,
            "stop": self._as_bool(decision.get("stop", False)),
            "need_capture_after": self._as_bool(decision.get("need_capture_after", True)),
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": str(decision.get("reason", "") or "").strip(),
            "raw_decision": decision,
        }

    def _write_llm_control_decision(
        self,
        labeling_dir: str,
        raw_decision: Dict[str, Any],
        normalized: Dict[str, Any],
        decision_index: int,
    ) -> None:
        artifact_dir = self._llm_control_artifact_dir(labeling_dir, decision_index)
        payload = {
            "version": "memory_aware_llm_control_decision_v1",
            "decision_index": int(decision_index),
            "labeling_dir": str(labeling_dir),
            "artifact_dir": str(artifact_dir),
            "labeling_inputs_dir": str(artifact_dir / "labeling_inputs"),
            "raw_decision": raw_decision,
            "normalized_decision": normalized,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        (artifact_dir / "llm_control_decision.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

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
        self.on_llm_control_stop()
        for window in (self.preview_window, self.depth_window, self.memory_window, self.llm_control_window, self.fusion_window, self.review_window, self.indicator_review_window, self.map_window, self.open_map_window):
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
