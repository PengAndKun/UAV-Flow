"""
Remote control panel for `uav_control_server.py`.

This file is the "controller side" process:
- connects to the running UAV control server
- sends move commands
- fetches the latest RGB preview and depth preview
- triggers screenshots on the server
"""

import argparse
import json
import logging
import tkinter as tk
import tkinter.font as tkfont
from typing import Any, Dict, Optional
from urllib import error, request

import cv2
import numpy as np
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)

PREVIEW_WINDOW_NAME = "UAV Remote Preview"
DEPTH_WINDOW_NAME = "UAV Depth Preview"


class RemoteControlClient:
    """Small HTTP client for the UAV control server."""

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

    def get_frame(self) -> np.ndarray:
        return self.get_image("/frame")

    def get_depth_frame(self) -> np.ndarray:
        return self.get_image("/depth_frame")

    def get_image(self, path: str) -> np.ndarray:
        req = request.Request(f"{self.base_url}{path}", method="GET")
        with request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read()
        data = np.frombuffer(body, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to decode image returned by server path={path}")
        return frame


class UAVControlPanel:
    """Tk control panel that talks to the remote UAV control server."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.client = RemoteControlClient(f"http://{args.host}:{args.port}", args.timeout_s)
        self.root = tk.Tk()
        self.root.title("UAV Remote Control Panel")
        self.root.geometry("920x780")
        self.root.minsize(780, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.status_font = tkfont.nametofont("TkFixedFont").copy()
        self.status_font.configure(size=10)
        self.status_var = tk.StringVar(value="Connecting...")
        self.depth_var = tk.StringVar(value="Depth: waiting...")
        self.plan_var = tk.StringVar(value="Plan: idle")
        self.mission_var = tk.StringVar(value="Mission: idle")
        self.evidence_var = tk.StringVar(value="Evidence: idle")
        self.archive_var = tk.StringVar(value="Archive: idle")
        self.reflex_var = tk.StringVar(value="Reflex: idle")
        self.executor_var = tk.StringVar(value="Executor: idle")
        self.takeover_var = tk.StringVar(value="Takeover: idle")
        self.last_state: Optional[Dict[str, Any]] = None
        self.preview_window: Optional[tk.Toplevel] = None
        self.preview_label: Optional[tk.Label] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.depth_window: Optional[tk.Toplevel] = None
        self.depth_label: Optional[tk.Label] = None
        self.depth_photo: Optional[ImageTk.PhotoImage] = None
        self.scroll_canvas: Optional[tk.Canvas] = None
        self.scroll_content: Optional[tk.Frame] = None
        self.scroll_window_id: Optional[int] = None
        self.task_label_var = tk.StringVar(value=args.default_task_label)
        self.capture_label_var = tk.StringVar(value="")
        self.takeover_note_var = tk.StringVar(value="")
        self.evidence_note_var = tk.StringVar(value="")

    def safe_request(self, func, *call_args, **call_kwargs):
        try:
            return func(*call_args, **call_kwargs)
        except error.URLError as exc:
            self.status_var.set(f"Server connection failed: {exc}")
            logger.warning("Server request failed: %s", exc)
        except Exception as exc:
            self.status_var.set(f"Request failed: {exc}")
            logger.warning("Request failed: %s", exc)
        return None

    @staticmethod
    def _frame_to_photo(frame: np.ndarray, width: int, height: int) -> ImageTk.PhotoImage:
        if width > 0 and height > 0:
            frame = cv2.resize(frame, (width, height))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        return ImageTk.PhotoImage(image=pil_image)

    def send_move(
        self,
        forward_cm: float = 0.0,
        right_cm: float = 0.0,
        up_cm: float = 0.0,
        yaw_delta_deg: float = 0.0,
        action_name: str = "custom",
    ) -> None:
        payload = {
            "forward_cm": forward_cm,
            "right_cm": right_cm,
            "up_cm": up_cm,
            "yaw_delta_deg": yaw_delta_deg,
            "action_name": action_name,
        }
        state = self.safe_request(self.client.post_json, "/move_relative", payload)
        if state:
            self.last_state = state
            self.update_status_from_state(state)
            self.update_preview_once()
            self.update_depth_once()

    def capture(self) -> None:
        payload = {
            "label": self.capture_label_var.get().strip(),
            "task_label": self.task_label_var.get().strip(),
        }
        result = self.safe_request(self.client.post_json, "/capture", payload)
        if result:
            self.status_var.set(f"Captured: {result['meta_path']}")
            self.update_preview_once()
            self.update_depth_once()

    def set_task_label(self) -> None:
        result = self.safe_request(
            self.client.post_json,
            "/task",
            {"task_label": self.task_label_var.get().strip()},
        )
        if result:
            self.status_var.set(f"Task label set: {result.get('task_label', '')}")

    def request_plan(self) -> None:
        payload = {"task_label": self.task_label_var.get().strip()}
        result = self.safe_request(self.client.post_json, "/request_plan", payload)
        if result:
            if isinstance(result.get("plan"), dict):
                self.plan_var.set(self.format_plan_summary(result["plan"]))
            self.update_state_once()

    def request_reflex(self) -> None:
        result = self.safe_request(self.client.post_json, "/request_reflex", {"trigger": "manual_request"})
        if result:
            self.update_state_once()

    def execute_reflex(self) -> None:
        result = self.safe_request(
            self.client.post_json,
            "/execute_reflex",
            {
                "trigger": "manual_execute",
                "refresh_policy": True,
                "allow_auto_plan": True,
                "sync_after_execution": True,
            },
        )
        if result:
            state = result.get("state") if isinstance(result.get("state"), dict) else None
            if state:
                self.last_state = state
                self.update_status_from_state(state)
            else:
                self.update_state_once()
            self.update_preview_once()
            self.update_depth_once()

    def start_takeover(self) -> None:
        note = self.takeover_note_var.get().strip()
        result = self.safe_request(
            self.client.post_json,
            "/takeover",
            {"action": "start", "reason": note or "manual_takeover", "note": note},
        )
        if result:
            self.update_state_once()

    def end_takeover(self) -> None:
        note = self.takeover_note_var.get().strip()
        result = self.safe_request(
            self.client.post_json,
            "/takeover",
            {"action": "end", "reason": "resolved", "note": note},
        )
        if result:
            self.update_state_once()

    def submit_person_evidence(self, action: str) -> None:
        payload = {
            "action": action,
            "note": self.evidence_note_var.get().strip(),
            "capture_label": self.capture_label_var.get().strip(),
        }
        result = self.safe_request(self.client.post_json, "/person_evidence", payload)
        if result:
            state = result.get("state") if isinstance(result.get("state"), dict) else None
            if state:
                self.last_state = state
                self.update_status_from_state(state)
            else:
                self.update_state_once()

    def mark_suspect(self) -> None:
        self.submit_person_evidence("suspect")

    def confirm_person(self) -> None:
        self.submit_person_evidence("confirm_present")

    def reject_person(self) -> None:
        self.submit_person_evidence("confirm_absent")

    def reset_evidence(self) -> None:
        self.submit_person_evidence("reset")

    def shutdown_server(self) -> None:
        result = self.safe_request(self.client.post_json, "/shutdown", {})
        if result:
            self.status_var.set("Server shutdown requested")

    def format_plan_summary(self, plan: Dict[str, Any]) -> str:
        waypoints = plan.get("candidate_waypoints") or []
        waypoint_text = "none"
        if waypoints:
            wp = waypoints[0]
            waypoint_text = f"({wp.get('x', 0.0):.0f}, {wp.get('y', 0.0):.0f}, {wp.get('z', 0.0):.0f})"
        return (
            f"Plan planner={plan.get('planner_name', 'n/a')} "
            f"subgoal={plan.get('semantic_subgoal', 'idle')} "
            f"sector={plan.get('sector_id', '-')} "
            f"conf={plan.get('planner_confidence', 0.0):.2f} "
            f"wp={waypoint_text}"
        )

    @staticmethod
    def shorten_cell_id(cell_id: str, limit: int = 52) -> str:
        text = str(cell_id or "")
        if len(text) <= limit:
            return text
        return f"{text[:24]}...{text[-18:]}"

    def adjust_status_font(self, delta: int) -> None:
        current_size = int(self.status_font.cget("size"))
        next_size = max(8, min(16, current_size + delta))
        self.status_font.configure(size=next_size)

    def _scroll_canvas_by(self, delta_units: int) -> None:
        if self.scroll_canvas is not None:
            self.scroll_canvas.yview_scroll(delta_units, "units")

    def on_mousewheel(self, event: tk.Event) -> None:
        delta = getattr(event, "delta", 0)
        if delta:
            steps = int(-delta / 120)
            if steps == 0:
                steps = -1 if delta > 0 else 1
            self._scroll_canvas_by(steps)

    def on_mousewheel_linux_up(self, _event: tk.Event) -> None:
        self._scroll_canvas_by(-1)

    def on_mousewheel_linux_down(self, _event: tk.Event) -> None:
        self._scroll_canvas_by(1)

    def refresh_scroll_region(self, _event: Optional[tk.Event] = None) -> None:
        if self.scroll_canvas is not None:
            self.scroll_canvas.configure(scrollregion=self.scroll_canvas.bbox("all"))

    def resize_scroll_content(self, event: tk.Event) -> None:
        if self.scroll_canvas is not None and self.scroll_window_id is not None:
            self.scroll_canvas.itemconfigure(self.scroll_window_id, width=event.width)

    def build_status_label(self, parent: tk.Widget, textvariable: tk.StringVar) -> tk.Label:
        label = tk.Label(
            parent,
            textvariable=textvariable,
            anchor="w",
            justify="left",
            font=self.status_font,
            padx=8,
            pady=6,
            relief="groove",
            borderwidth=1,
        )
        label.pack(fill="x", pady=(0, 6))
        label.bind(
            "<Configure>",
            lambda event, widget=label: widget.configure(wraplength=max(320, event.width - 16)),
        )
        return label

    def update_status_from_state(self, state: Dict[str, Any]) -> None:
        pose = state.get("pose", {})
        depth = state.get("depth", {})
        camera_info = state.get("camera_info", depth.get("camera_info", {}))
        runtime_debug = state.get("runtime_debug", {})
        plan = state.get("plan", {})
        mission = state.get("mission", {})
        search_runtime = state.get("search_runtime", {})
        person_evidence = state.get("person_evidence_runtime", {})
        search_result = state.get("search_result", {})
        planner_runtime = state.get("planner_runtime", {})
        archive = state.get("archive", {})
        reflex_runtime = state.get("reflex_runtime", {})
        reflex_execution = state.get("reflex_execution", {})
        takeover_runtime = state.get("takeover_runtime", {})
        takeover_events = state.get("takeover_recent_events", [])
        archive_current = archive.get("current_cell") if isinstance(archive.get("current_cell"), dict) else {}
        archive_candidates = archive.get("top_cells") or []
        archive_hint = "none"
        if archive_candidates:
            archive_hint = ",".join(
                self.shorten_cell_id(str(item.get("cell_id", "")), limit=18)
                for item in archive_candidates[:2]
            )
        self.status_var.set(
            "Pose "
            f"x={pose.get('x', 0.0):.1f} "
            f"y={pose.get('y', 0.0):.1f} "
            f"z={pose.get('z', 0.0):.1f} "
            f"yaw={pose.get('yaw', 0.0):.1f} "
            f"cmd={pose.get('command_yaw', 0.0):.1f} "
            f"task={pose.get('task_yaw', 0.0):.1f} "
            f"uav={pose.get('uav_yaw', 0.0):.1f} "
            f"action={state.get('last_action', 'idle')} "
            f"risk={runtime_debug.get('risk_score', 0.0):.2f} "
            f"task_label={state.get('task_label', '')} | "
            f"server={self.client.base_url}"
        )
        self.depth_var.set(
            f"Depth frame={depth.get('frame_id', 'n/a')} "
            f"size={depth.get('image_width', 0)}x{depth.get('image_height', 0)} "
            f"range={float(depth.get('min_depth', 0.0)):.1f}->{float(depth.get('max_depth', 0.0)):.1f} cm "
            f"fov={float(depth.get('fov_deg', 0.0)):.1f} "
            f"fx={float((camera_info.get('k') or [0.0])[0]):.1f} "
            f"pipe={depth.get('pipeline', depth.get('source_mode', 'n/a'))}"
        )
        if plan:
            self.plan_var.set(
                f"{self.format_plan_summary(plan)} | "
                f"status={planner_runtime.get('planner_status', 'idle')} "
                f"source={planner_runtime.get('planner_source', 'none')} "
                f"trigger={planner_runtime.get('last_trigger', 'n/a')} "
                f"latency={float(planner_runtime.get('last_latency_ms', 0.0)):.1f}ms "
                f"auto={planner_runtime.get('auto_mode', 'manual')} "
                f"next={planner_runtime.get('next_auto_trigger_step', 0)}"
            )
        else:
            self.plan_var.set("Plan: idle")
        mission_priority = mission.get("priority_regions") if isinstance(mission.get("priority_regions"), list) else []
        runtime_candidates = search_runtime.get("candidate_regions") if isinstance(search_runtime.get("candidate_regions"), list) else []
        priority_region = search_runtime.get("priority_region") if isinstance(search_runtime.get("priority_region"), dict) else {}
        region_label = str(priority_region.get("region_label", "")) or "none"
        region_status = str(priority_region.get("status", "")) or "n/a"
        self.mission_var.set(
            "Mission "
            f"type={mission.get('mission_type', 'n/a')} "
            f"status={mission.get('status', 'idle')} "
            f"subgoal={search_runtime.get('current_search_subgoal', 'idle')} "
            f"scope={mission.get('search_scope', 'local')} "
            f"priority={region_label}:{region_status} "
            f"candidates={len(runtime_candidates) or len(mission_priority)} "
            f"detect={search_runtime.get('detection_state', 'unknown')} "
            f"confirm={int(bool(mission.get('confirm_target', False)))} "
            f"visited={int(search_runtime.get('visited_region_count', 0))}"
        )
        estimated_position = search_result.get("estimated_person_position") if isinstance(search_result.get("estimated_person_position"), dict) else {}
        estimated_label = "none"
        person_exists = search_result.get("person_exists")
        if person_exists is None:
            person_exists_label = "unknown"
        else:
            person_exists_label = str(int(bool(person_exists)))
        if estimated_position:
            estimated_label = (
                f"({float(estimated_position.get('x', 0.0)):.0f},"
                f"{float(estimated_position.get('y', 0.0)):.0f},"
                f"{float(estimated_position.get('z', 0.0)):.0f})"
            )
        self.evidence_var.set(
            "Evidence "
            f"status={person_evidence.get('evidence_status', 'idle')} "
            f"result={search_result.get('result_status', 'unknown')} "
            f"exists={person_exists_label} "
            f"conf={float(person_evidence.get('confidence', search_result.get('confidence', 0.0))):.2f} "
            f"suspect={int(person_evidence.get('suspect_count', 0))} "
            f"present={int(person_evidence.get('confirm_present_count', 0))} "
            f"absent={int(person_evidence.get('confirm_absent_count', 0))} "
            f"events={int(person_evidence.get('evidence_event_count', 0))} "
            f"loc={estimated_label}"
        )
        self.archive_var.set(
            "Archive "
            f"cell={self.shorten_cell_id(str(runtime_debug.get('archive_cell_id', archive.get('current_cell_id', '')))) or 'none'} "
            f"visits={int(archive_current.get('visit_count', 0))} "
            f"cells={int(archive.get('cell_count', 0))} "
            f"transitions={int(archive.get('transition_count', 0))} "
            f"recent={len(archive.get('recent_cell_ids', [])) if isinstance(archive.get('recent_cell_ids'), list) else 0} "
            f"hint={archive_hint}"
        )
        self.reflex_var.set(
            "Reflex "
            f"mode={reflex_runtime.get('mode', 'n/a')} "
            f"policy={reflex_runtime.get('policy_name', 'n/a')} "
            f"source={reflex_runtime.get('source', 'n/a')} "
            f"status={reflex_runtime.get('status', 'idle')} "
            f"suggested={reflex_runtime.get('suggested_action', 'idle')} "
            f"exec={int(bool(reflex_runtime.get('should_execute', False)))} "
            f"lat={float(reflex_runtime.get('last_latency_ms', 0.0)):.1f}ms "
            f"conf={float(reflex_runtime.get('policy_confidence', 0.0)):.2f} "
            f"wp_dist={float(reflex_runtime.get('waypoint_distance_cm', 0.0)):.1f} "
            f"yaw_err={float(reflex_runtime.get('yaw_error_deg', 0.0)):.1f} "
            f"progress={float(reflex_runtime.get('progress_to_waypoint_cm', 0.0)):.1f} "
            f"retrieval={self.shorten_cell_id(str(reflex_runtime.get('retrieval_cell_id', '')), limit=18) or 'none'}"
        )
        self.executor_var.set(
            "Executor "
            f"mode={reflex_execution.get('mode', 'manual')} "
            f"status={reflex_execution.get('last_status', 'idle')} "
            f"reason={reflex_execution.get('last_reason', 'none')} "
            f"req={reflex_execution.get('last_requested_action', 'idle')} "
            f"exec={reflex_execution.get('last_executed_action', '') or 'none'} "
            f"count={int(reflex_execution.get('execution_count', 0))}"
        )
        self.takeover_var.set(
            "Takeover "
            f"active={int(bool(takeover_runtime.get('active', False)))} "
            f"reason={takeover_runtime.get('current_reason', 'none') or 'none'} "
            f"last_reason={takeover_runtime.get('last_intervention_reason', takeover_runtime.get('last_event_reason', 'none')) or 'none'} "
            f"note={takeover_runtime.get('current_note', '') or '-'} "
            f"interventions={int(takeover_runtime.get('intervention_count', 0))} "
            f"events={int(takeover_runtime.get('event_count', 0))} "
            f"last={takeover_runtime.get('last_event_type', 'none')} "
            f"recent={len(takeover_events) if isinstance(takeover_events, list) else 0}"
        )

    def update_state_once(self) -> None:
        state = self.safe_request(self.client.get_json, "/state")
        if state:
            self.last_state = state
            self.update_status_from_state(state)

    def update_preview_once(self) -> None:
        if self.args.hide_preview_window:
            return
        frame = self.safe_request(self.client.get_frame)
        if frame is None:
            return
        self.preview_photo = self._frame_to_photo(frame, self.args.preview_width, self.args.preview_height)
        if self.preview_label is not None:
            self.preview_label.configure(image=self.preview_photo)
            self.preview_label.image = self.preview_photo

    def update_depth_once(self) -> None:
        if self.args.hide_depth_window:
            return
        frame = self.safe_request(self.client.get_depth_frame)
        if frame is None:
            return
        self.depth_photo = self._frame_to_photo(frame, self.args.depth_width, self.args.depth_height)
        if self.depth_label is not None:
            self.depth_label.configure(image=self.depth_photo)
            self.depth_label.image = self.depth_photo

    def schedule_state_refresh(self) -> None:
        self.update_state_once()
        self.root.after(self.args.state_interval_ms, self.schedule_state_refresh)

    def schedule_preview_refresh(self) -> None:
        if self.args.hide_preview_window:
            return
        self.update_preview_once()
        self.root.after(self.args.preview_interval_ms, self.schedule_preview_refresh)

    def schedule_depth_refresh(self) -> None:
        if self.args.hide_depth_window:
            return
        self.update_depth_once()
        self.root.after(self.args.depth_interval_ms, self.schedule_depth_refresh)

    def bind_keys(self) -> None:
        bindings = {
            "w": lambda: self.send_move(forward_cm=self.args.move_step_cm, action_name="forward(W)"),
            "s": lambda: self.send_move(forward_cm=-self.args.move_step_cm, action_name="backward(S)"),
            "a": lambda: self.send_move(right_cm=-self.args.move_step_cm, action_name="left(A)"),
            "d": lambda: self.send_move(right_cm=self.args.move_step_cm, action_name="right(D)"),
            "r": lambda: self.send_move(up_cm=self.args.vertical_step_cm, action_name="up(R)"),
            "f": lambda: self.send_move(up_cm=-self.args.vertical_step_cm, action_name="down(F)"),
            "q": lambda: self.send_move(yaw_delta_deg=-self.args.yaw_step_deg, action_name="yaw_left(Q)"),
            "e": lambda: self.send_move(yaw_delta_deg=self.args.yaw_step_deg, action_name="yaw_right(E)"),
            "c": self.capture,
            "p": self.request_plan,
            "t": self.request_reflex,
            "y": self.execute_reflex,
            "v": lambda: (self.update_preview_once(), self.update_depth_once()),
            "u": self.start_takeover,
            "i": self.end_takeover,
            "g": self.mark_suspect,
            "h": self.confirm_person,
            "j": self.reject_person,
            "o": self.reset_evidence,
        }
        for key, callback in bindings.items():
            self.root.bind(f"<KeyPress-{key}>", lambda _event, cb=callback: cb())
            self.root.bind(f"<KeyPress-{key.upper()}>", lambda _event, cb=callback: cb())

    def build_ui(self) -> None:
        if not self.args.hide_preview_window:
            self.preview_window = tk.Toplevel(self.root)
            self.preview_window.title(PREVIEW_WINDOW_NAME)
            self.preview_window.geometry(f"{self.args.preview_width}x{self.args.preview_height}")
            self.preview_window.protocol("WM_DELETE_WINDOW", self.on_close)
            self.preview_label = tk.Label(self.preview_window)
            self.preview_label.pack(fill="both", expand=True)

        if not self.args.hide_depth_window:
            self.depth_window = tk.Toplevel(self.root)
            self.depth_window.title(DEPTH_WINDOW_NAME)
            self.depth_window.geometry(f"{self.args.depth_width}x{self.args.depth_height}")
            self.depth_window.protocol("WM_DELETE_WINDOW", self.on_close)
            self.depth_label = tk.Label(self.depth_window)
            self.depth_label.pack(fill="both", expand=True)
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(fill="both", expand=True)
        self.scroll_canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        scrollbar = tk.Scrollbar(canvas_frame, orient="vertical", command=self.scroll_canvas.yview)
        self.scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.scroll_canvas.pack(side="left", fill="both", expand=True)
        self.scroll_content = tk.Frame(self.scroll_canvas)
        self.scroll_window_id = self.scroll_canvas.create_window((0, 0), window=self.scroll_content, anchor="nw")
        self.scroll_content.bind("<Configure>", self.refresh_scroll_region)
        self.scroll_canvas.bind("<Configure>", self.resize_scroll_content)
        self.root.bind_all("<MouseWheel>", self.on_mousewheel)
        self.root.bind_all("<Button-4>", self.on_mousewheel_linux_up)
        self.root.bind_all("<Button-5>", self.on_mousewheel_linux_down)
        content = self.scroll_content

        header = tk.Label(
            content,
            text="Keyboard: W/S/A/D move, R/F up-down, Q/E yaw, C capture, V refresh preview, P request plan, T request reflex, Y execute reflex, U start takeover, I end takeover, G suspect, H confirm person, J reject person, O reset evidence",
            anchor="w",
            justify="left",
            wraplength=860,
        )
        header.pack(fill="x", padx=12, pady=(10, 6))

        status_frame = tk.LabelFrame(content, text="Runtime Status", padx=10, pady=8)
        status_frame.pack(fill="x", padx=12, pady=(0, 10))
        status_toolbar = tk.Frame(status_frame)
        status_toolbar.pack(fill="x", pady=(0, 8))
        tk.Label(
            status_toolbar,
            text="Use scroll to inspect long status blocks. Zoom changes the status text size only.",
            anchor="w",
            justify="left",
        ).pack(side="left", fill="x", expand=True)
        tk.Button(status_toolbar, text="A-", width=4, command=lambda: self.adjust_status_font(-1)).pack(side="right")
        tk.Button(status_toolbar, text="A+", width=4, command=lambda: self.adjust_status_font(1)).pack(side="right", padx=(0, 6))

        self.build_status_label(status_frame, self.status_var)
        self.build_status_label(status_frame, self.depth_var)
        self.build_status_label(status_frame, self.plan_var)
        self.build_status_label(status_frame, self.mission_var)
        self.build_status_label(status_frame, self.evidence_var)
        self.build_status_label(status_frame, self.archive_var)
        self.build_status_label(status_frame, self.reflex_var)
        self.build_status_label(status_frame, self.executor_var)
        self.build_status_label(status_frame, self.takeover_var)

        mission_frame = tk.LabelFrame(content, text="Mission And Notes", padx=10, pady=8)
        mission_frame.pack(fill="x", padx=12, pady=(0, 8))
        tk.Label(mission_frame, text="Task Label").grid(row=0, column=0, sticky="w")
        tk.Entry(mission_frame, textvariable=self.task_label_var).grid(row=0, column=1, sticky="ew", padx=(8, 10))
        tk.Label(mission_frame, text="Capture Label").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tk.Entry(mission_frame, textvariable=self.capture_label_var).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 10),
            pady=(8, 0),
        )
        tk.Label(mission_frame, text="Takeover Note").grid(row=2, column=0, sticky="w", pady=(8, 0))
        tk.Entry(mission_frame, textvariable=self.takeover_note_var).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(8, 10),
            pady=(8, 0),
        )
        tk.Label(mission_frame, text="Evidence Note").grid(row=3, column=0, sticky="w", pady=(8, 0))
        tk.Entry(mission_frame, textvariable=self.evidence_note_var).grid(
            row=3,
            column=1,
            sticky="ew",
            padx=(8, 10),
            pady=(8, 0),
        )
        mission_actions = tk.Frame(mission_frame)
        mission_actions.grid(row=0, column=2, rowspan=4, sticky="ns")
        for idx, (label, callback) in enumerate(
            [
                ("Set Task", self.set_task_label),
                ("Capture", self.capture),
                ("Request Plan", self.request_plan),
                ("Request Reflex", self.request_reflex),
                ("Execute Reflex", self.execute_reflex),
            ]
        ):
            tk.Button(mission_actions, text=label, command=callback, width=16).grid(
                row=idx,
                column=0,
                sticky="ew",
                pady=(0 if idx == 0 else 8, 0),
            )
        mission_frame.grid_columnconfigure(1, weight=1)

        task_help = tk.Label(
            content,
            text=(
                "Task Label controls planner/capture mission metadata. Capture Label is a one-shot suffix for the next saved sample. "
                "Takeover Note and Evidence Note are stored with manual interventions and suspect/confirm/reject person events."
            ),
            anchor="w",
            justify="left",
            wraplength=860,
        )
        task_help.pack(fill="x", padx=12, pady=(0, 8))

        control_groups = tk.Frame(content)
        control_groups.pack(fill="x", padx=12, pady=6)

        movement_frame = tk.LabelFrame(control_groups, text="Movement", padx=8, pady=8)
        movement_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        movement_buttons = [
            ("Forward (W)", lambda: self.send_move(forward_cm=self.args.move_step_cm, action_name="forward(W)")),
            ("Backward (S)", lambda: self.send_move(forward_cm=-self.args.move_step_cm, action_name="backward(S)")),
            ("Left (A)", lambda: self.send_move(right_cm=-self.args.move_step_cm, action_name="left(A)")),
            ("Right (D)", lambda: self.send_move(right_cm=self.args.move_step_cm, action_name="right(D)")),
            ("Up (R)", lambda: self.send_move(up_cm=self.args.vertical_step_cm, action_name="up(R)")),
            ("Down (F)", lambda: self.send_move(up_cm=-self.args.vertical_step_cm, action_name="down(F)")),
            ("Yaw Left (Q)", lambda: self.send_move(yaw_delta_deg=-self.args.yaw_step_deg, action_name="yaw_left(Q)")),
            ("Yaw Right (E)", lambda: self.send_move(yaw_delta_deg=self.args.yaw_step_deg, action_name="yaw_right(E)")),
        ]
        for idx, (label, callback) in enumerate(movement_buttons):
            tk.Button(movement_frame, text=label, command=callback, width=18).grid(
                row=idx // 2,
                column=idx % 2,
                padx=6,
                pady=6,
                sticky="ew",
            )
        movement_frame.grid_columnconfigure(0, weight=1)
        movement_frame.grid_columnconfigure(1, weight=1)

        planner_frame = tk.LabelFrame(control_groups, text="Planner And Reflex", padx=8, pady=8)
        planner_frame.grid(row=0, column=1, sticky="nsew", pady=(0, 6))
        for idx, (label, callback) in enumerate(
            [
                ("Request Plan (P)", self.request_plan),
                ("Request Reflex (T)", self.request_reflex),
                ("Execute Reflex (Y)", self.execute_reflex),
                ("Refresh (V)", lambda: (self.update_preview_once(), self.update_depth_once())),
            ]
        ):
            tk.Button(planner_frame, text=label, command=callback, width=18).grid(
                row=idx,
                column=0,
                padx=6,
                pady=6,
                sticky="ew",
            )
        planner_frame.grid_columnconfigure(0, weight=1)

        takeover_frame = tk.LabelFrame(control_groups, text="Takeover", padx=8, pady=8)
        takeover_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        for idx, (label, callback) in enumerate(
            [
                ("Start Takeover (U)", self.start_takeover),
                ("End Takeover (I)", self.end_takeover),
            ]
        ):
            tk.Button(takeover_frame, text=label, command=callback, width=18).grid(
                row=idx,
                column=0,
                padx=6,
                pady=6,
                sticky="ew",
            )
        takeover_frame.grid_columnconfigure(0, weight=1)

        evidence_frame = tk.LabelFrame(control_groups, text="Evidence", padx=8, pady=8)
        evidence_frame.grid(row=1, column=1, sticky="nsew", pady=(0, 6))
        for idx, (label, callback) in enumerate(
            [
                ("Mark Suspect (G)", self.mark_suspect),
                ("Confirm Person (H)", self.confirm_person),
                ("Reject Person (J)", self.reject_person),
                ("Reset Evidence (O)", self.reset_evidence),
            ]
        ):
            tk.Button(evidence_frame, text=label, command=callback, width=18).grid(
                row=idx,
                column=0,
                padx=6,
                pady=6,
                sticky="ew",
            )
        evidence_frame.grid_columnconfigure(0, weight=1)

        system_frame = tk.LabelFrame(control_groups, text="System", padx=8, pady=8)
        system_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        tk.Button(system_frame, text="Capture (C)", command=self.capture, width=18).grid(
            row=0,
            column=0,
            padx=6,
            pady=6,
            sticky="ew",
        )
        tk.Button(system_frame, text="Shutdown Server", command=self.shutdown_server, width=18).grid(
            row=0,
            column=1,
            padx=6,
            pady=6,
            sticky="ew",
        )
        system_frame.grid_columnconfigure(0, weight=1)
        system_frame.grid_columnconfigure(1, weight=1)
        control_groups.grid_columnconfigure(0, weight=1)
        control_groups.grid_columnconfigure(1, weight=1)

        shown_windows = []
        if not self.args.hide_preview_window:
            shown_windows.append(PREVIEW_WINDOW_NAME)
        if not self.args.hide_depth_window:
            shown_windows.append(DEPTH_WINDOW_NAME)
        windows_text = " / ".join(shown_windows) if shown_windows else "(none)"
        hint = tk.Label(
            content,
            text=(
                f"Preview windows: {windows_text}\n"
                f"Control server: {self.client.base_url}\n"
                "The main panel supports mouse-wheel scrolling when the content becomes taller than the window."
            ),
            anchor="w",
            justify="left",
            wraplength=860,
        )
        hint.pack(fill="x", padx=12, pady=(6, 10))

        self.bind_keys()

    def on_close(self) -> None:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        if self.preview_window is not None:
            try:
                self.preview_window.destroy()
            except Exception:
                pass
        if self.depth_window is not None:
            try:
                self.depth_window.destroy()
            except Exception:
                pass
        self.root.destroy()

    def run(self) -> None:
        self.build_ui()
        self.update_state_once()
        self.update_preview_once()
        self.update_depth_once()
        self.schedule_state_refresh()
        self.schedule_preview_refresh()
        self.schedule_depth_refresh()
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote control panel for uav_control_server.py")
    parser.add_argument("--host", default="127.0.0.1", help="Control server host")
    parser.add_argument("--port", type=int, default=5020, help="Control server port")
    parser.add_argument("--timeout_s", type=float, default=5.0, help="HTTP request timeout")
    parser.add_argument("--move_step_cm", type=float, default=20.0, help="Forward/left-right translation step")
    parser.add_argument("--vertical_step_cm", type=float, default=20.0, help="Up/down translation step")
    parser.add_argument("--yaw_step_deg", type=float, default=5.0, help="Yaw step")
    parser.add_argument("--preview_width", type=int, default=960, help="Preview display width")
    parser.add_argument("--preview_height", type=int, default=540, help="Preview display height")
    parser.add_argument("--depth_width", type=int, default=480, help="Depth preview width")
    parser.add_argument("--depth_height", type=int, default=360, help="Depth preview height")
    parser.add_argument("--preview_interval_ms", type=int, default=180, help="Preview refresh interval")
    parser.add_argument("--depth_interval_ms", type=int, default=250, help="Depth preview refresh interval")
    parser.add_argument("--hide_preview_window", action="store_true", help="Do not open or refresh the RGB preview window")
    parser.add_argument("--hide_depth_window", action="store_true", help="Do not open or refresh the depth preview window")
    parser.add_argument("--state_interval_ms", type=int, default=500, help="State refresh interval")
    parser.add_argument("--default_task_label", default="", help="Initial task label shown in the panel")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Logging level")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="[%(levelname)s] %(asctime)s - %(name)s - %(message)s",
    )
    panel = UAVControlPanel(args)
    panel.run()


if __name__ == "__main__":
    main()
