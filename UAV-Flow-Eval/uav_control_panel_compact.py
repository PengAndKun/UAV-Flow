"""
Compact UAV control panel for `uav_control_server.py`.

Goals:
- keep only the most-used manual controls
- support compact symbol sequences like `wwwq`
- avoid heavy realtime refresh pressure by default
- expose planner/LLM configuration and API history debugging
- provide simple RGB/depth preview window toggles
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
import tkinter.ttk as ttk
from typing import Any, Dict, List, Optional, Tuple
from urllib import request

import cv2
import numpy as np
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)

PLANNER_PRESETS: Dict[str, Dict[str, Any]] = {
    "Heuristic Baseline": {
        "planner_name": "external_heuristic_planner",
        "planner_mode": "heuristic",
        "planner_route_mode": "heuristic_only",
        "fallback_to_heuristic": True,
        "planner_request_timeout_s": 5.0,
    },
    "Gemini Lite": {
        "planner_name": "external_llm_planner",
        "planner_mode": "llm",
        "planner_route_mode": "llm_only",
        "llm_api_style": "google_genai_sdk",
        "llm_model": "gemini-3.1-flash-lite-preview",
        "llm_input_mode": "text_image",
        "llm_base_url": "google-genai-sdk",
        "llm_api_key_env": "GEMINI_API_KEY",
        "fallback_to_heuristic": False,
        "planner_request_timeout_s": 25.0,
    },
    "Gemini Flash": {
        "planner_name": "external_llm_planner",
        "planner_mode": "llm",
        "planner_route_mode": "llm_only",
        "llm_api_style": "google_genai_sdk",
        "llm_model": "gemini-3-flash-preview",
        "llm_input_mode": "text_image",
        "llm_base_url": "google-genai-sdk",
        "llm_api_key_env": "GEMINI_API_KEY",
        "fallback_to_heuristic": False,
        "planner_request_timeout_s": 25.0,
    },
    "Search Hybrid": {
        "planner_name": "external_llm_planner",
        "planner_mode": "hybrid",
        "planner_route_mode": "search_hybrid",
        "llm_api_style": "google_genai_sdk",
        "llm_model": "gemini-3.1-flash-lite-preview",
        "llm_input_mode": "text_image",
        "llm_base_url": "google-genai-sdk",
        "llm_api_key_env": "GEMINI_API_KEY",
        "fallback_to_heuristic": True,
        "planner_request_timeout_s": 25.0,
    },
    "Anthropic Qwen Next": {
        "planner_name": "external_llm_planner",
        "planner_mode": "llm",
        "planner_route_mode": "llm_only",
        "llm_api_style": "anthropic_sdk",
        "llm_model": "qwen3-coder-next",
        "llm_input_mode": "text_image",
        "llm_base_url": "http://1.95.142.151:3000",
        "llm_api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "fallback_to_heuristic": False,
        "planner_request_timeout_s": 20.0,
    },
    "Anthropic Sonnet": {
        "planner_name": "external_llm_planner",
        "planner_mode": "llm",
        "planner_route_mode": "llm_only",
        "llm_api_style": "anthropic_sdk",
        "llm_model": "claude-sonnet-4-6",
        "llm_input_mode": "text_image",
        "llm_base_url": "http://1.95.142.151:3000",
        "llm_api_key_env": "ANTHROPIC_AUTH_TOKEN",
        "fallback_to_heuristic": False,
        "planner_request_timeout_s": 20.0,
    },
}

MOVE_COMMANDS: Dict[str, Dict[str, Any]] = {
    "w": {"forward_cm": 20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "forward"},
    "s": {"forward_cm": -20.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "backward"},
    "a": {"forward_cm": 0.0, "right_cm": -20.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "left"},
    "d": {"forward_cm": 0.0, "right_cm": 20.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "right"},
    "r": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 20.0, "yaw_delta_deg": 0.0, "action_name": "up"},
    "f": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": -20.0, "yaw_delta_deg": 0.0, "action_name": "down"},
    "q": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": -5.0, "action_name": "yaw_left"},
    "e": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 5.0, "action_name": "yaw_right"},
    "x": {"forward_cm": 0.0, "right_cm": 0.0, "up_cm": 0.0, "yaw_delta_deg": 0.0, "action_name": "hold"},
}


class RemoteControlClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def get_json(self, path: str) -> Dict[str, Any]:
        req = request.Request(f"{self.base_url}{path}", method="GET")
        with request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def post_json(self, path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = json.dumps(payload or {}).encode("utf-8")
        req = request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
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


class CompactUAVControlPanel:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.client = RemoteControlClient(f"http://{args.host}:{args.port}", args.timeout_s)
        self.root = tk.Tk()
        self.root.title("UAV Remote Compact Controller")
        self.root.geometry("1120x820")
        self.root.minsize(860, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.default_font = tkfont.nametofont("TkDefaultFont").copy()
        self.default_font.configure(size=10)
        self.status_font = tkfont.nametofont("TkFixedFont").copy()
        self.status_font.configure(size=10)
        self.title_font = tkfont.nametofont("TkDefaultFont").copy()
        self.title_font.configure(size=11, weight="bold")
        self.small_font = tkfont.nametofont("TkDefaultFont").copy()
        self.small_font.configure(size=9)

        self.root.option_add("*Font", self.default_font)
        self.style = ttk.Style(self.root)
        self.style.configure(".", font=self.default_font)
        self.style.configure("TLabelframe.Label", font=self.title_font)

        self.last_state: Optional[Dict[str, Any]] = None
        self.last_api_history: Optional[Dict[str, Any]] = None
        self.state_refresh_inflight = False
        self.preview_refresh_inflight = False
        self.depth_refresh_inflight = False
        self.manual_request_inflight = False
        self.background_pause_until = 0.0

        self.preview_window: Optional[tk.Toplevel] = None
        self.preview_label: Optional[tk.Label] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.depth_window: Optional[tk.Toplevel] = None
        self.depth_label: Optional[tk.Label] = None
        self.depth_photo: Optional[ImageTk.PhotoImage] = None
        self.api_history_window: Optional[tk.Toplevel] = None
        self.api_history_text_widget: Optional[tk.Text] = None

        self.sequence_thread: Optional[threading.Thread] = None
        self.sequence_stop_event = threading.Event()
        self.move_queue: List[str] = []
        self.move_queue_worker: Optional[threading.Thread] = None
        self.move_request_inflight = False

        self.status_labels: List[tk.Label] = []
        self.canvas: Optional[tk.Canvas] = None
        self.canvas_window: Optional[int] = None
        self.scrollable_body: Optional[tk.Frame] = None
        self.last_font_bucket: Optional[int] = None

        self.auto_state_var = tk.BooleanVar(value=True)
        self.auto_preview_var = tk.BooleanVar(value=False)
        self.auto_depth_var = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="Ready")
        self.pose_var = tk.StringVar(value="Pose: waiting...")
        self.plan_var = tk.StringVar(value="Plan: waiting...")
        self.mission_var = tk.StringVar(value="Mission: waiting...")
        self.llm_action_var = tk.StringVar(value="LLMAct: waiting...")
        self.api_hist_var = tk.StringVar(value="APIHist: waiting...")
        self.planner_cfg_var = tk.StringVar(value="Planner config: waiting...")

        self.task_label_var = tk.StringVar(value=args.default_task_label)
        self.capture_label_var = tk.StringVar(value="")
        self.sequence_var = tk.StringVar(value="")
        self.sequence_delay_ms_var = tk.StringVar(value="220")

        self.plan_segment_steps_var = tk.StringVar(value="8")
        self.plan_replan_every_var = tk.StringVar(value="0")
        self.hold_retry_var = tk.StringVar(value="2")
        self.continuous_llm_var = tk.BooleanVar(value=True)

        self.preset_var = tk.StringVar(value="Gemini Lite")
        self.planner_name_var = tk.StringVar(value="external_llm_planner")
        self.planner_mode_var = tk.StringVar(value="llm")
        self.route_mode_var = tk.StringVar(value="llm_only")
        self.api_style_var = tk.StringVar(value="google_genai_sdk")
        self.model_var = tk.StringVar(value="gemini-3.1-flash-lite-preview")
        self.input_mode_var = tk.StringVar(value="text_image")
        self.base_url_var = tk.StringVar(value="google-genai-sdk")
        self.api_key_var = tk.StringVar(value="")
        self.api_env_var = tk.StringVar(value="GEMINI_API_KEY")
        self.req_timeout_var = tk.StringVar(value="25.0")
        self.allow_fallback_var = tk.BooleanVar(value=False)

        self.build_ui()
        self.refresh_client_timeout()
        self.root.bind("<Configure>", self.handle_root_resize, add="+")

    def build_ui(self) -> None:
        outer = tk.Frame(self.root)
        outer.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        body = tk.Frame(self.canvas, padx=10, pady=10)
        self.scrollable_body = body
        self.canvas_window = self.canvas.create_window((0, 0), window=body, anchor="nw")
        body.bind("<Configure>", self.on_body_configure)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.bind_mousewheel(self.root)

        main = body

        status_frame = tk.LabelFrame(main, text="Runtime Status", padx=8, pady=8)
        status_frame.pack(fill="x", pady=(0, 10))
        self.build_status_label(status_frame, self.status_var)
        self.build_status_label(status_frame, self.pose_var)
        self.build_status_label(status_frame, self.plan_var)
        self.build_status_label(status_frame, self.mission_var)
        self.build_status_label(status_frame, self.llm_action_var)
        self.build_status_label(status_frame, self.api_hist_var)
        self.build_status_label(status_frame, self.planner_cfg_var)

        controls = tk.Frame(main)
        controls.pack(fill="both", expand=True)
        controls.grid_columnconfigure(0, weight=1)
        controls.grid_columnconfigure(1, weight=1)

        left = tk.Frame(controls)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        right = tk.Frame(controls)
        right.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self.build_task_frame(left)
        self.build_move_frame(left)
        self.build_sequence_frame(left)
        self.build_preview_frame(left)
        self.build_refresh_frame(left)

        self.build_llm_frame(right)

    def build_status_label(self, parent: tk.Widget, textvariable: tk.StringVar) -> None:
        label = tk.Label(
            parent,
            textvariable=textvariable,
            anchor="w",
            justify="left",
            font=self.status_font,
            relief="groove",
            borderwidth=1,
            padx=8,
            pady=6,
        )
        label.pack(fill="x", pady=(0, 5))
        self.status_labels.append(label)
        label.bind("<Configure>", lambda event, widget=label: widget.configure(wraplength=max(420, event.width - 16)))

    def build_task_frame(self, parent: tk.Widget) -> None:
        frame = tk.LabelFrame(parent, text="Task And Capture", padx=8, pady=8)
        frame.pack(fill="x", pady=(0, 10))

        tk.Label(frame, text="Task Label").grid(row=0, column=0, sticky="w")
        tk.Entry(frame, textvariable=self.task_label_var).grid(row=0, column=1, columnspan=2, sticky="ew", padx=(6, 0))
        tk.Button(frame, text="Set Task", command=self.set_task_label, width=12).grid(row=0, column=3, padx=(8, 0))

        tk.Label(frame, text="Capture Label").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tk.Entry(frame, textvariable=self.capture_label_var).grid(row=1, column=1, columnspan=2, sticky="ew", padx=(6, 0), pady=(8, 0))
        tk.Button(frame, text="Capture", command=self.capture, width=12).grid(row=1, column=3, padx=(8, 0), pady=(8, 0))

        frame.grid_columnconfigure(1, weight=1)
        frame.grid_columnconfigure(2, weight=1)

    def build_move_frame(self, parent: tk.Widget) -> None:
        frame = tk.LabelFrame(parent, text="Basic Movement", padx=8, pady=8)
        frame.pack(fill="x", pady=(0, 10))
        pad = tk.Frame(frame)
        pad.pack(fill="x", expand=True)

        buttons: List[Tuple[str, str, int, int]] = [
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
        for text, symbol, row, column in buttons:
            tk.Button(pad, text=text, width=12, command=lambda symbol=symbol: self.send_move_symbol(symbol)).grid(
                row=row,
                column=column,
                padx=3,
                pady=3,
                sticky="nsew",
            )
        for row in range(3):
            pad.grid_rowconfigure(row, weight=1)
        for column in range(3):
            pad.grid_columnconfigure(column, weight=1)

    def build_sequence_frame(self, parent: tk.Widget) -> None:
        frame = tk.LabelFrame(parent, text="Sequence Control", padx=8, pady=8)
        frame.pack(fill="x", pady=(0, 10))

        tk.Label(frame, text="Symbols").grid(row=0, column=0, sticky="w")
        tk.Entry(frame, textvariable=self.sequence_var).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        tk.Label(frame, text="Delay ms").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tk.Entry(frame, textvariable=self.sequence_delay_ms_var, width=12).grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

        button_row = tk.Frame(frame)
        button_row.grid(row=0, column=2, rowspan=2, padx=(10, 0))
        tk.Button(button_row, text="Execute Sequence", command=self.execute_sequence, width=16).pack(fill="x")
        tk.Button(button_row, text="Stop Sequence", command=self.stop_sequence, width=16).pack(fill="x", pady=(6, 0))

        helper = tk.Label(
            frame,
            text="Use w/s/a/d/r/f/q/e/x. Example: wwwqdd",
            anchor="w",
            justify="left",
            font=self.small_font,
        )
        helper.grid(row=2, column=0, columnspan=3, sticky="w", pady=(8, 0))
        frame.grid_columnconfigure(1, weight=1)

    def build_preview_frame(self, parent: tk.Widget) -> None:
        frame = tk.LabelFrame(parent, text="Preview Windows", padx=8, pady=8)
        frame.pack(fill="x", pady=(0, 10))

        rgb_row = tk.Frame(frame)
        rgb_row.pack(fill="x")
        tk.Button(rgb_row, text="Toggle RGB", command=self.toggle_preview_window, width=12).pack(side="left")
        tk.Button(rgb_row, text="Refresh RGB", command=self.request_preview_refresh_async, width=12).pack(side="left", padx=6)
        tk.Checkbutton(rgb_row, text="Auto RGB", variable=self.auto_preview_var).pack(side="left", padx=(10, 0))

        depth_row = tk.Frame(frame)
        depth_row.pack(fill="x", pady=(8, 0))
        tk.Button(depth_row, text="Toggle Depth", command=self.toggle_depth_window, width=12).pack(side="left")
        tk.Button(depth_row, text="Refresh Depth", command=self.request_depth_refresh_async, width=12).pack(side="left", padx=6)
        tk.Checkbutton(depth_row, text="Auto Depth", variable=self.auto_depth_var).pack(side="left", padx=(10, 0))

    def build_refresh_frame(self, parent: tk.Widget) -> None:
        frame = tk.LabelFrame(parent, text="Refresh", padx=8, pady=8)
        frame.pack(fill="x", pady=(0, 10))
        tk.Button(frame, text="Refresh State", command=self.request_state_refresh_async, width=14).pack(side="left")
        tk.Checkbutton(frame, text="Auto State", variable=self.auto_state_var).pack(side="left", padx=(10, 0))
        tk.Label(frame, text=f"State interval: {self.args.state_interval_ms} ms", font=self.small_font).pack(side="left", padx=(12, 0))

    def build_llm_frame(self, parent: tk.Widget) -> None:
        frame = tk.LabelFrame(parent, text="Paper Experiment Controls", padx=8, pady=8)
        frame.pack(fill="both", expand=True)

        row = 0

        def add_row(label_text: str, widget: tk.Widget) -> None:
            nonlocal row
            tk.Label(frame, text=label_text, anchor="w").grid(row=row, column=0, sticky="w", pady=3)
            widget.grid(row=row, column=1, sticky="ew", pady=3)
            row += 1

        preset_combo = ttk.Combobox(frame, textvariable=self.preset_var, values=list(PLANNER_PRESETS.keys()), state="readonly")
        add_row("Preset", preset_combo)
        tk.Button(frame, text="Use Preset", command=self.apply_preset, width=14).grid(row=0, column=2, padx=(8, 0))

        add_row("Mode", ttk.Combobox(frame, textvariable=self.planner_mode_var, values=["heuristic", "llm", "hybrid"], state="readonly"))
        add_row("Route", ttk.Combobox(frame, textvariable=self.route_mode_var, values=["heuristic_only", "llm_only", "search_hybrid", "auto"], state="readonly"))
        add_row(
            "API Style",
            ttk.Combobox(
                frame,
                textvariable=self.api_style_var,
                values=["google_genai_sdk", "google_gemini", "anthropic_sdk", "anthropic_messages", "openai_chat", "openai_responses"],
                state="readonly",
            ),
        )
        add_row("Model", tk.Entry(frame, textvariable=self.model_var))
        add_row("Input", ttk.Combobox(frame, textvariable=self.input_mode_var, values=["text", "text_image"], state="readonly"))
        add_row("Base URL", tk.Entry(frame, textvariable=self.base_url_var))
        api_key_entry = tk.Entry(frame, textvariable=self.api_key_var, show="*")
        add_row("API Key", api_key_entry)
        add_row("API Env", tk.Entry(frame, textvariable=self.api_env_var))
        add_row("Req Timeout", tk.Entry(frame, textvariable=self.req_timeout_var))
        tk.Checkbutton(frame, text="Allow heuristic fallback", variable=self.allow_fallback_var).grid(row=row, column=1, sticky="w", pady=4)
        row += 1

        seg_frame = tk.Frame(frame)
        seg_frame.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(6, 6))
        tk.Label(seg_frame, text="LLM Steps").pack(side="left")
        tk.Entry(seg_frame, textvariable=self.plan_segment_steps_var, width=6).pack(side="left", padx=(6, 12))
        tk.Label(seg_frame, text="Plan Every").pack(side="left")
        tk.Entry(seg_frame, textvariable=self.plan_replan_every_var, width=6).pack(side="left", padx=(6, 12))
        tk.Label(seg_frame, text="Hold Retry").pack(side="left")
        tk.Entry(seg_frame, textvariable=self.hold_retry_var, width=6).pack(side="left", padx=(6, 12))
        tk.Checkbutton(seg_frame, text="Continuous LLM", variable=self.continuous_llm_var).pack(side="left")
        row += 1

        buttons = tk.Frame(frame)
        buttons.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        tk.Button(buttons, text="Apply Manual", command=self.apply_planner_config, width=14).grid(row=0, column=0, padx=4, pady=4)
        tk.Button(buttons, text="Refresh Planner", command=self.refresh_planner_config, width=14).grid(row=0, column=1, padx=4, pady=4)
        tk.Button(buttons, text="Request Plan", command=self.request_plan, width=14).grid(row=0, column=2, padx=4, pady=4)
        tk.Button(buttons, text="Request Scene", command=self.request_scene_waypoints, width=14).grid(row=1, column=0, padx=4, pady=4)
        tk.Button(buttons, text="Request LLM Action", command=self.request_llm_action, width=14).grid(row=1, column=1, padx=4, pady=4)
        tk.Button(buttons, text="Execute LLM Segment", command=self.execute_llm_action_segment, width=16).grid(row=1, column=2, padx=4, pady=4)
        tk.Button(buttons, text="View API History", command=self.refresh_api_history_view, width=14).grid(row=2, column=0, padx=4, pady=4)
        tk.Button(buttons, text="Refresh State", command=self.request_state_refresh_async, width=14).grid(row=2, column=1, padx=4, pady=4)

        frame.grid_columnconfigure(1, weight=1)

    def refresh_client_timeout(self) -> None:
        try:
            planner_timeout = float(self.req_timeout_var.get().strip() or "8.0")
        except ValueError:
            planner_timeout = 8.0
        self.client.timeout_s = max(float(self.args.timeout_s), planner_timeout + 4.0)

    def pause_background_refresh(self, seconds: float = 1.2) -> None:
        self.background_pause_until = max(self.background_pause_until, time.time() + max(0.0, seconds))

    def background_refresh_allowed(self) -> bool:
        return (
            not self.manual_request_inflight
            and not self.move_request_inflight
            and time.time() >= self.background_pause_until
        )

    def run_async_request(self, work, *, busy_message: str = "", on_success=None, on_error=None) -> None:
        if busy_message:
            self.status_var.set(busy_message)

        def worker() -> None:
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001
                if on_error is not None:
                    self.root.after(0, lambda exc=exc: on_error(exc))
                else:
                    self.root.after(0, lambda exc=exc: self.handle_async_error("Request", exc))
                return
            if on_success is not None:
                self.root.after(0, lambda result=result: on_success(result))

        threading.Thread(target=worker, daemon=True).start()

    def run_manual_async_request(
        self,
        work,
        *,
        busy_message: str = "",
        pause_s: float = 1.5,
        on_success=None,
        on_error=None,
    ) -> None:
        if self.manual_request_inflight:
            self.status_var.set("Another request is still running...")
            return
        self.manual_request_inflight = True
        self.pause_background_refresh(pause_s)

        def handle_success(result):
            self.manual_request_inflight = False
            if on_success is not None:
                on_success(result)

        def handle_error(exc: Exception) -> None:
            self.manual_request_inflight = False
            if on_error is not None:
                on_error(exc)
            else:
                self.handle_async_error("Request", exc)

        self.run_async_request(
            work,
            busy_message=busy_message,
            on_success=handle_success,
            on_error=handle_error,
        )

    def handle_async_error(self, prefix: str, exc: Exception) -> None:
        self.status_var.set(f"{prefix} failed: {exc}")
        logger.warning("%s failed: %s", prefix, exc)

    def request_state_refresh_async(self) -> None:
        if self.state_refresh_inflight or not self.background_refresh_allowed():
            return
        self.state_refresh_inflight = True

        def on_done(result: Optional[Dict[str, Any]]) -> None:
            self.state_refresh_inflight = False
            if isinstance(result, dict):
                self.last_state = result
                self.update_status_from_state(result)

        def on_error(exc: Exception) -> None:
            self.state_refresh_inflight = False
            self.handle_async_error("Refresh State", exc)

        self.run_async_request(lambda: self.client.get_json("/state"), on_success=on_done, on_error=on_error)

    def schedule_state_refresh(self) -> None:
        if self.auto_state_var.get() and self.background_refresh_allowed():
            self.request_state_refresh_async()
        self.root.after(max(400, int(self.args.state_interval_ms)), self.schedule_state_refresh)

    def schedule_preview_refresh(self) -> None:
        if (
            self.auto_preview_var.get()
            and self.preview_window is not None
            and self.preview_window.winfo_exists()
            and self.background_refresh_allowed()
        ):
            self.request_preview_refresh_async()
        self.root.after(max(400, int(self.args.preview_interval_ms)), self.schedule_preview_refresh)

    def schedule_depth_refresh(self) -> None:
        if (
            self.auto_depth_var.get()
            and self.depth_window is not None
            and self.depth_window.winfo_exists()
            and self.background_refresh_allowed()
        ):
            self.request_depth_refresh_async()
        self.root.after(max(400, int(self.args.depth_interval_ms)), self.schedule_depth_refresh)

    def send_move_symbol(self, symbol: str) -> None:
        self.move_queue.append(symbol)
        if self.move_queue_worker is not None and self.move_queue_worker.is_alive():
            self.status_var.set(f"Queued {MOVE_COMMANDS[symbol]['action_name']} ({len(self.move_queue)} pending)")
            return

        def worker() -> None:
            while self.move_queue:
                next_symbol = self.move_queue.pop(0)
                payload = MOVE_COMMANDS[next_symbol]
                self.move_request_inflight = True
                self.pause_background_refresh(0.9)
                try:
                    result = self.client.post_json("/move_relative", payload)
                except Exception as exc:  # noqa: BLE001
                    self.root.after(0, lambda exc=exc, next_symbol=next_symbol: self.handle_async_error(f"Move {next_symbol}", exc))
                else:
                    self.root.after(0, lambda result=result: self.handle_move_response(result))
                finally:
                    self.move_request_inflight = False
                time.sleep(0.04)

        self.status_var.set(f"Sending {MOVE_COMMANDS[symbol]['action_name']}...")
        self.move_queue_worker = threading.Thread(target=worker, daemon=True)
        self.move_queue_worker.start()

    def handle_move_response(self, result: Optional[Dict[str, Any]]) -> None:
        if isinstance(result, dict):
            self.last_state = result
            self.update_status_from_state(result)

    def parse_sequence(self, raw: str) -> List[str]:
        sequence: List[str] = []
        for char in raw.lower():
            if char in MOVE_COMMANDS:
                sequence.append(char)
            elif char in {" ", ",", ";", "\n", "\t"}:
                continue
        return sequence

    def execute_sequence(self) -> None:
        sequence = self.parse_sequence(self.sequence_var.get())
        if not sequence:
            self.status_var.set("Sequence is empty or has no valid symbols")
            return
        if self.sequence_thread is not None and self.sequence_thread.is_alive():
            self.status_var.set("Sequence already running")
            return
        try:
            delay_ms = max(0, int(self.sequence_delay_ms_var.get().strip() or "220"))
        except ValueError:
            delay_ms = 220
        self.sequence_stop_event.clear()
        self.status_var.set(f"Executing sequence: {''.join(sequence)}")

        def worker() -> None:
            executed = 0
            try:
                for index, symbol in enumerate(sequence, start=1):
                    if self.sequence_stop_event.is_set():
                        break
                    payload = MOVE_COMMANDS[symbol]
                    self.move_request_inflight = True
                    self.pause_background_refresh(max(0.9, delay_ms / 1000.0))
                    result = self.client.post_json("/move_relative", payload)
                    executed = index
                    self.root.after(0, lambda result=result: self.handle_move_response(result))
                    self.move_request_inflight = False
                    remaining_s = delay_ms / 1000.0
                    while remaining_s > 0.0 and not self.sequence_stop_event.is_set():
                        sleep_s = min(0.05, remaining_s)
                        time.sleep(sleep_s)
                        remaining_s -= sleep_s
            except Exception as exc:  # noqa: BLE001
                self.move_request_inflight = False
                self.root.after(0, lambda exc=exc: self.handle_async_error("Sequence", exc))
                return
            self.move_request_inflight = False
            stop_text = "stopped" if self.sequence_stop_event.is_set() else "completed"
            self.root.after(0, lambda executed=executed, stop_text=stop_text: self.status_var.set(f"Sequence {stop_text}: {executed}/{len(sequence)} steps"))

        self.sequence_thread = threading.Thread(target=worker, daemon=True)
        self.sequence_thread.start()

    def stop_sequence(self) -> None:
        self.sequence_stop_event.set()
        self.status_var.set("Stopping sequence...")

    def set_task_label(self) -> None:
        payload = {"task_label": self.task_label_var.get().strip()}
        self.run_manual_async_request(
            lambda: self.client.post_json("/task", payload),
            busy_message="Setting task...",
            on_success=self.handle_set_task_result,
            on_error=lambda exc: self.handle_async_error("Set Task", exc),
        )

    def capture(self) -> None:
        payload = {"label": self.capture_label_var.get().strip(), "task_label": self.task_label_var.get().strip()}
        self.run_manual_async_request(
            lambda: self.client.post_json("/capture", payload),
            busy_message="Capturing...",
            pause_s=2.0,
            on_success=self.handle_capture_result,
            on_error=lambda exc: self.handle_async_error("Capture", exc),
        )

    def handle_set_task_result(self, result: Optional[Dict[str, Any]]) -> None:
        if isinstance(result, dict):
            self.status_var.set(f"Task set: {result.get('task_label', '')}")
            self.request_state_refresh_async()

    def handle_capture_result(self, result: Optional[Dict[str, Any]]) -> None:
        if isinstance(result, dict):
            self.status_var.set(f"Captured: {result.get('meta_path', 'ok')}")
            self.request_state_refresh_async()

    def build_planner_config_payload(self) -> Dict[str, Any]:
        payload = {
            "planner_name": self.planner_name_var.get().strip() or "external_llm_planner",
            "planner_mode": self.planner_mode_var.get().strip() or "heuristic",
            "planner_route_mode": self.route_mode_var.get().strip() or "heuristic_only",
            "llm_api_style": self.api_style_var.get().strip() or "google_genai_sdk",
            "llm_model": self.model_var.get().strip(),
            "llm_input_mode": self.input_mode_var.get().strip() or "text",
            "llm_base_url": self.base_url_var.get().strip(),
            "llm_api_key_env": self.api_env_var.get().strip(),
            "fallback_to_heuristic": bool(self.allow_fallback_var.get()),
            "planner_request_timeout_s": float(self.req_timeout_var.get().strip() or "8.0"),
        }
        inline_key = self.api_key_var.get().strip()
        if inline_key:
            payload["llm_api_key"] = inline_key
        return payload

    def sync_planner_controls_from_config(self, config: Dict[str, Any]) -> None:
        self.planner_name_var.set(str(config.get("planner_name", self.planner_name_var.get()) or self.planner_name_var.get()))
        self.planner_mode_var.set(str(config.get("planner_mode", self.planner_mode_var.get()) or self.planner_mode_var.get()))
        self.route_mode_var.set(str(config.get("planner_route_mode_raw", config.get("planner_route_mode", self.route_mode_var.get())) or self.route_mode_var.get()))
        self.api_style_var.set(str(config.get("llm_api_style", self.api_style_var.get()) or self.api_style_var.get()))
        self.model_var.set(str(config.get("llm_model", self.model_var.get()) or self.model_var.get()))
        self.input_mode_var.set(str(config.get("llm_input_mode", self.input_mode_var.get()) or self.input_mode_var.get()))
        self.base_url_var.set(str(config.get("llm_base_url", self.base_url_var.get()) or self.base_url_var.get()))
        self.api_env_var.set(str(config.get("llm_api_key_env", self.api_env_var.get()) or self.api_env_var.get()))
        self.allow_fallback_var.set(bool(config.get("fallback_to_heuristic", self.allow_fallback_var.get())))
        self.req_timeout_var.set(str(float(config.get("planner_request_timeout_s", self.req_timeout_var.get() or 8.0))))
        self.refresh_client_timeout()
        self.planner_cfg_var.set(
            "PlannerCfg "
            f"mode={self.planner_mode_var.get()} "
            f"route={config.get('planner_route_mode', self.route_mode_var.get())} "
            f"api={self.api_style_var.get() or '-'} "
            f"model={self.model_var.get() or '-'} "
            f"key_cfg={int(bool(config.get('llm_api_key_configured', False)))} "
            f"fallback={int(bool(self.allow_fallback_var.get()))} "
            f"req_timeout={self.req_timeout_var.get()}s "
            f"enabled={int(bool(config.get('llm_enabled', False)))}"
        )

    def refresh_planner_config(self) -> None:
        self.run_manual_async_request(
            lambda: self.client.get_json("/planner_config"),
            busy_message="Refreshing planner config...",
            pause_s=1.2,
            on_success=lambda result: self.sync_planner_controls_from_config(result) if isinstance(result, dict) else None,
            on_error=lambda exc: self.handle_async_error("Refresh Planner", exc),
        )

    def apply_planner_config(self) -> None:
        payload = self.build_planner_config_payload()
        self.refresh_client_timeout()
        self.run_manual_async_request(
            lambda: self.client.post_json("/planner_config", payload),
            busy_message="Applying planner config...",
            pause_s=1.8,
            on_success=self.handle_apply_planner_config,
            on_error=lambda exc: self.handle_async_error("Apply Planner", exc),
        )

    def handle_apply_planner_config(self, result: Optional[Dict[str, Any]]) -> None:
        if isinstance(result, dict):
            self.sync_planner_controls_from_config(result)
            self.status_var.set("Planner config applied")

    def apply_preset(self) -> None:
        preset = PLANNER_PRESETS.get(self.preset_var.get().strip())
        if not preset:
            self.status_var.set("Unknown preset")
            return
        self.planner_name_var.set(str(preset.get("planner_name", self.planner_name_var.get())))
        self.planner_mode_var.set(str(preset.get("planner_mode", self.planner_mode_var.get())))
        self.route_mode_var.set(str(preset.get("planner_route_mode", self.route_mode_var.get())))
        self.api_style_var.set(str(preset.get("llm_api_style", self.api_style_var.get())))
        self.model_var.set(str(preset.get("llm_model", self.model_var.get())))
        self.input_mode_var.set(str(preset.get("llm_input_mode", self.input_mode_var.get())))
        self.base_url_var.set(str(preset.get("llm_base_url", self.base_url_var.get())))
        self.api_env_var.set(str(preset.get("llm_api_key_env", self.api_env_var.get())))
        self.allow_fallback_var.set(bool(preset.get("fallback_to_heuristic", self.allow_fallback_var.get())))
        self.req_timeout_var.set(str(float(preset.get("planner_request_timeout_s", self.req_timeout_var.get() or 8.0))))
        self.apply_planner_config()

    def request_plan(self) -> None:
        payload = {"task_label": self.task_label_var.get().strip()}
        self.run_manual_async_request(
            lambda: self.client.post_json("/request_plan", payload),
            busy_message="Requesting planner...",
            pause_s=3.0,
            on_success=self.handle_request_state_result,
            on_error=lambda exc: self.handle_async_error("Request Plan", exc),
        )

    def request_scene_waypoints(self) -> None:
        self.run_manual_async_request(
            lambda: self.client.post_json("/request_scene_waypoints", {"trigger": "manual_request", "refresh_observations": True}),
            busy_message="Requesting scene waypoints...",
            pause_s=3.0,
            on_success=self.handle_request_state_result,
            on_error=lambda exc: self.handle_async_error("Request Scene", exc),
        )

    def request_llm_action(self) -> None:
        self.run_manual_async_request(
            lambda: self.client.post_json("/request_llm_action", {"trigger": "manual_request", "refresh_observations": True}),
            busy_message="Requesting LLM action...",
            pause_s=3.0,
            on_success=self.handle_request_state_result,
            on_error=lambda exc: self.handle_async_error("Request LLM Action", exc),
        )

    def execute_llm_action_segment(self) -> None:
        try:
            step_budget = max(1, min(50, int(self.plan_segment_steps_var.get().strip() or "8")))
        except ValueError:
            step_budget = 8
        try:
            plan_every = max(0, int(self.plan_replan_every_var.get().strip() or "0"))
        except ValueError:
            plan_every = 0
        try:
            hold_retry = max(0, min(10, int(self.hold_retry_var.get().strip() or "2")))
        except ValueError:
            hold_retry = 2
        payload = {
            "step_budget": step_budget,
            "refresh_plan": True,
            "plan_refresh_interval_steps": plan_every,
            "continuous_mode": bool(self.continuous_llm_var.get()),
            "hold_retry_budget": hold_retry,
            "trigger": "manual_llm_action_segment",
        }
        self.run_manual_async_request(
            lambda: self.client.post_json("/execute_llm_action_segment", payload),
            busy_message=f"Running LLM action segment ({step_budget} steps)...",
            pause_s=6.0,
            on_success=self.handle_request_state_result,
            on_error=lambda exc: self.handle_async_error("Execute LLM Segment", exc),
        )

    def handle_request_state_result(self, result: Optional[Dict[str, Any]]) -> None:
        if isinstance(result, dict):
            state = result.get("state") if isinstance(result.get("state"), dict) else None
            if state is not None:
                self.last_state = state
                self.update_status_from_state(state)
            else:
                self.request_state_refresh_async()

    def toggle_preview_window(self) -> None:
        if self.preview_window is None or not self.preview_window.winfo_exists():
            self.open_preview_window()
            return
        self.close_preview_window()

    def toggle_depth_window(self) -> None:
        if self.depth_window is None or not self.depth_window.winfo_exists():
            self.open_depth_window()
            return
        self.close_depth_window()

    def open_preview_window(self) -> None:
        if self.preview_window is None or not self.preview_window.winfo_exists():
            self.preview_window = tk.Toplevel(self.root)
            self.preview_window.title("RGB Preview")
            self.preview_label = tk.Label(self.preview_window)
            self.preview_label.pack(fill="both", expand=True)
            self.preview_window.protocol("WM_DELETE_WINDOW", self.close_preview_window)
        self.request_preview_refresh_async()

    def close_preview_window(self) -> None:
        if self.preview_window is not None and self.preview_window.winfo_exists():
            self.preview_window.destroy()
        self.preview_window = None
        self.preview_label = None
        self.preview_photo = None

    def open_depth_window(self) -> None:
        if self.depth_window is None or not self.depth_window.winfo_exists():
            self.depth_window = tk.Toplevel(self.root)
            self.depth_window.title("Depth Preview")
            self.depth_label = tk.Label(self.depth_window)
            self.depth_label.pack(fill="both", expand=True)
            self.depth_window.protocol("WM_DELETE_WINDOW", self.close_depth_window)
        self.request_depth_refresh_async()

    def close_depth_window(self) -> None:
        if self.depth_window is not None and self.depth_window.winfo_exists():
            self.depth_window.destroy()
        self.depth_window = None
        self.depth_label = None
        self.depth_photo = None

    def request_preview_refresh_async(self) -> None:
        if self.preview_refresh_inflight or self.preview_window is None or not self.preview_window.winfo_exists():
            return
        self.preview_refresh_inflight = True

        def on_done(frame: Optional[np.ndarray]) -> None:
            self.preview_refresh_inflight = False
            if frame is not None:
                self.update_preview_frame(frame)

        def on_error(exc: Exception) -> None:
            self.preview_refresh_inflight = False
            self.handle_async_error("Refresh RGB", exc)

        self.run_async_request(lambda: self.client.get_image("/frame"), on_success=on_done, on_error=on_error)

    def request_depth_refresh_async(self) -> None:
        if self.depth_refresh_inflight or self.depth_window is None or not self.depth_window.winfo_exists():
            return
        self.depth_refresh_inflight = True

        def on_done(frame: Optional[np.ndarray]) -> None:
            self.depth_refresh_inflight = False
            if frame is not None:
                self.update_depth_frame(frame)

        def on_error(exc: Exception) -> None:
            self.depth_refresh_inflight = False
            self.handle_async_error("Refresh Depth", exc)

        self.run_async_request(lambda: self.client.get_image("/depth_frame"), on_success=on_done, on_error=on_error)

    def update_preview_frame(self, frame: np.ndarray) -> None:
        if self.preview_label is None:
            return
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(image))
        self.preview_photo = photo
        self.preview_label.configure(image=photo)

    def update_depth_frame(self, frame: np.ndarray) -> None:
        if self.depth_label is None:
            return
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(image))
        self.depth_photo = photo
        self.depth_label.configure(image=photo)

    def refresh_api_history_view(self) -> None:
        self.run_manual_async_request(
            lambda: self.client.get_json("/api_history"),
            busy_message="Loading API history...",
            pause_s=1.5,
            on_success=self.handle_api_history_response,
            on_error=lambda exc: self.handle_async_error("Load API History", exc),
        )

    def handle_api_history_response(self, result: Optional[Dict[str, Any]]) -> None:
        if not isinstance(result, dict):
            return
        self.last_api_history = result
        self.open_text_viewer("API Request History", self.build_api_history_view_text())

    def build_api_history_view_text(self) -> str:
        if not isinstance(self.last_api_history, dict):
            return "No API history available."
        summary = self.last_api_history.get("api_history_summary", {})
        records = self.last_api_history.get("api_request_history", [])
        lines = [
            "API History Summary",
            json.dumps(summary, ensure_ascii=True, indent=2),
            "",
            "Recent Records",
            "",
        ]
        for record in records:
            lines.append(
                f"[{record.get('history_id', '-')}] kind={record.get('kind', '-')}"
                f" status={record.get('status', '-')}"
                f" trigger={record.get('trigger', '-')}"
                f" model={record.get('model_name', '-')}"
                f" latency={record.get('latency_ms', 0.0)}"
                f" fallback={int(bool(record.get('fallback_used', False)))}"
            )
            lines.append(f"  error={record.get('error', '') or '-'}")
            lines.append("  request_payload:")
            lines.append(json.dumps(record.get("request_payload", {}), ensure_ascii=True, indent=2))
            lines.append("  parsed_payload:")
            lines.append(json.dumps(record.get("parsed_payload", {}), ensure_ascii=True, indent=2))
            raw_text = str(record.get("raw_text", "") or "")
            lines.append("  raw_text:")
            lines.append(raw_text or "(empty)")
            sys_excerpt = str(record.get("system_prompt_excerpt", "") or "")
            usr_excerpt = str(record.get("user_prompt_excerpt", "") or "")
            if sys_excerpt or usr_excerpt:
                lines.append("  prompt_excerpt:")
                lines.append(sys_excerpt or "(no system excerpt)")
                lines.append("")
                lines.append(usr_excerpt or "(no user excerpt)")
            lines.append("")
        return "\n".join(lines)

    def open_text_viewer(self, title: str, content: str) -> None:
        if (
            self.api_history_window is None
            or not self.api_history_window.winfo_exists()
            or self.api_history_text_widget is None
            or not self.api_history_text_widget.winfo_exists()
        ):
            window = tk.Toplevel(self.root)
            window.title(title)
            window.geometry("900x640")
            container = tk.Frame(window)
            container.pack(fill="both", expand=True)
            scrollbar = tk.Scrollbar(container)
            scrollbar.pack(side="right", fill="y")
            text = tk.Text(container, wrap="word", font=self.status_font, yscrollcommand=scrollbar.set)
            text.pack(side="left", fill="both", expand=True)
            scrollbar.configure(command=text.yview)
            self.api_history_window = window
            self.api_history_text_widget = text
        assert self.api_history_text_widget is not None
        self.api_history_window.deiconify()
        self.api_history_window.lift()
        self.api_history_text_widget.configure(state="normal")
        self.api_history_text_widget.delete("1.0", "end")
        self.api_history_text_widget.insert("1.0", content)
        self.api_history_text_widget.configure(state="disabled")

    def update_status_from_state(self, state: Dict[str, Any]) -> None:
        pose = state.get("pose", {}) if isinstance(state.get("pose"), dict) else {}
        runtime_debug = state.get("runtime_debug", {}) if isinstance(state.get("runtime_debug"), dict) else {}
        task_label = str(state.get("task_label", "") or "")
        self.pose_var.set(
            "Pose "
            f"x={float(pose.get('x', 0.0)):.1f} "
            f"y={float(pose.get('y', 0.0)):.1f} "
            f"z={float(pose.get('z', 0.0)):.1f} "
            f"yaw={float(pose.get('yaw', 0.0)):.1f} "
            f"cmd={float(pose.get('command_yaw', 0.0)):.1f} "
            f"task={float(pose.get('task_yaw', 0.0)):.1f} "
            f"uav={float(pose.get('uav_yaw', 0.0)):.1f} "
            f"action={state.get('last_action', 'idle')} "
            f"risk={float(runtime_debug.get('risk_score', 0.0)):.2f} "
            f"task_label={task_label or '-'}"
        )

        plan = state.get("plan", {}) if isinstance(state.get("plan"), dict) else {}
        planner_runtime = state.get("planner_runtime", {}) if isinstance(state.get("planner_runtime"), dict) else {}
        usage = planner_runtime.get("last_usage", {}) if isinstance(planner_runtime.get("last_usage"), dict) else {}
        tokens = int(
            usage.get(
                "total_tokens",
                usage.get(
                    "totalTokenCount",
                    usage.get(
                        "total_token_count",
                        usage.get("input_tokens", usage.get("promptTokenCount", usage.get("prompt_token_count", 0))),
                    ),
                ),
            )
            or 0
        )
        self.plan_var.set(
            "Plan "
            f"planner={plan.get('planner_name', 'n/a')} "
            f"subgoal={plan.get('semantic_subgoal', 'idle')} "
            f"status={planner_runtime.get('planner_status', 'idle')} "
            f"detail={planner_runtime.get('planner_source_detail', 'none')} "
            f"route={planner_runtime.get('planner_route_mode', 'n/a')} "
            f"model={planner_runtime.get('last_model_name', '-') or '-'} "
            f"fallback={int(bool(planner_runtime.get('fallback_used', False)))} "
            f"tokens={tokens} "
            f"lat={float(planner_runtime.get('last_latency_ms', 0.0)):.1f}ms"
        )

        mission = state.get("mission", {}) if isinstance(state.get("mission"), dict) else {}
        search_runtime = state.get("search_runtime", {}) if isinstance(state.get("search_runtime"), dict) else {}
        doorway_runtime = state.get("doorway_runtime", {}) if isinstance(state.get("doorway_runtime"), dict) else {}
        phase6_runtime = state.get("phase6_mission_runtime", {}) if isinstance(state.get("phase6_mission_runtime"), dict) else {}
        scene_waypoint_runtime = state.get("scene_waypoint_runtime", {}) if isinstance(state.get("scene_waypoint_runtime"), dict) else {}
        self.mission_var.set(
            "Mission "
            f"type={mission.get('mission_type', 'n/a')} "
            f"subgoal={search_runtime.get('current_search_subgoal', 'idle')} "
            f"scope={mission.get('search_scope', 'local')} "
            f"door={int(doorway_runtime.get('candidate_count', 0))}/{int(doorway_runtime.get('traversable_candidate_count', 0))} "
            f"phase6={phase6_runtime.get('active_stage_id', 'none')} "
            f"scene={scene_waypoint_runtime.get('scene_state', 'unknown')}"
        )

        llm_action_runtime = state.get("llm_action_runtime", {}) if isinstance(state.get("llm_action_runtime"), dict) else {}
        llm_usage = llm_action_runtime.get("usage", {}) if isinstance(llm_action_runtime.get("usage"), dict) else {}
        llm_tokens = int(
            llm_usage.get(
                "total_tokens",
                llm_usage.get(
                    "totalTokenCount",
                    llm_usage.get(
                        "total_token_count",
                        llm_usage.get("input_tokens", llm_usage.get("promptTokenCount", llm_usage.get("prompt_token_count", 0))),
                    ),
                ),
            )
            or 0
        )
        self.llm_action_var.set(
            "LLMAct "
            f"status={llm_action_runtime.get('status', 'idle')} "
            f"source={llm_action_runtime.get('source', 'none')} "
            f"suggested={llm_action_runtime.get('suggested_action', 'hold')} "
            f"exec={int(bool(llm_action_runtime.get('should_execute', False)))} "
            f"conf={float(llm_action_runtime.get('confidence', 0.0)):.2f} "
            f"tokens={llm_tokens} "
            f"model={llm_action_runtime.get('model_name', '-') or '-'}"
        )

        hist = state.get("api_history_summary", {}) if isinstance(state.get("api_history_summary"), dict) else {}
        self.api_hist_var.set(
            "APIHist "
            f"count={int(hist.get('count', 0))} "
            f"last={hist.get('last_kind', 'none')}:{hist.get('last_status', 'idle')} "
            f"model={hist.get('last_model_name', '-') or '-'} "
            f"preview={str(hist.get('last_preview', '') or '-')[:72]}"
        )

    def on_body_configure(self, _event: tk.Event) -> None:
        if self.canvas is not None:
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def on_canvas_configure(self, event: tk.Event) -> None:
        if self.canvas is not None and self.canvas_window is not None:
            self.canvas.itemconfigure(self.canvas_window, width=event.width)
        self.update_status_wraplengths(event.width)

    def update_status_wraplengths(self, width: int) -> None:
        wrap = max(420, width - 56)
        for label in self.status_labels:
            if label.winfo_exists():
                label.configure(wraplength=wrap)

    def bind_mousewheel(self, widget: tk.Widget) -> None:
        widget.bind_all("<MouseWheel>", self.on_mousewheel, add="+")
        widget.bind_all("<Button-4>", self.on_mousewheel, add="+")
        widget.bind_all("<Button-5>", self.on_mousewheel, add="+")

    def on_mousewheel(self, event: tk.Event) -> None:
        if self.canvas is None:
            return
        if getattr(event, "num", None) == 4:
            delta = -1
        elif getattr(event, "num", None) == 5:
            delta = 1
        else:
            delta = -1 * int(getattr(event, "delta", 0) / 120) if getattr(event, "delta", 0) else 0
        if delta:
            self.canvas.yview_scroll(delta, "units")

    def handle_root_resize(self, _event: tk.Event) -> None:
        width = max(860, int(self.root.winfo_width()))
        bucket = max(8, min(13, int(round(width / 112.0))))
        if self.last_font_bucket == bucket:
            return
        self.last_font_bucket = bucket
        self.default_font.configure(size=bucket)
        self.small_font.configure(size=max(7, bucket - 1))
        self.status_font.configure(size=max(8, bucket))
        self.title_font.configure(size=max(9, bucket + 1))
        self.update_status_wraplengths(width)

    def on_close(self) -> None:
        self.stop_sequence()
        self.close_preview_window()
        self.close_depth_window()
        self.root.destroy()

    def run(self) -> None:
        self.refresh_planner_config()
        self.request_state_refresh_async()
        self.schedule_state_refresh()
        self.schedule_preview_refresh()
        self.schedule_depth_refresh()
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compact remote control panel for uav_control_server.py")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5020)
    parser.add_argument("--timeout_s", type=float, default=10.0)
    parser.add_argument("--state_interval_ms", type=int, default=1800)
    parser.add_argument("--preview_interval_ms", type=int, default=1800)
    parser.add_argument("--depth_interval_ms", type=int, default=2200)
    parser.add_argument("--default_task_label", default="")
    parser.add_argument("--log_level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )
    panel = CompactUAVControlPanel(args)
    panel.run()


if __name__ == "__main__":
    main()
