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
        self.root.geometry("760x520")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.status_var = tk.StringVar(value="Connecting...")
        self.depth_var = tk.StringVar(value="Depth: waiting...")
        self.plan_var = tk.StringVar(value="Plan: idle")
        self.archive_var = tk.StringVar(value="Archive: idle")
        self.reflex_var = tk.StringVar(value="Reflex: idle")
        self.executor_var = tk.StringVar(value="Executor: idle")
        self.last_state: Optional[Dict[str, Any]] = None
        self.preview_window: Optional[tk.Toplevel] = None
        self.preview_label: Optional[tk.Label] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.depth_window: Optional[tk.Toplevel] = None
        self.depth_label: Optional[tk.Label] = None
        self.depth_photo: Optional[ImageTk.PhotoImage] = None
        self.task_label_var = tk.StringVar(value=args.default_task_label)
        self.capture_label_var = tk.StringVar(value="")

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

    def update_status_from_state(self, state: Dict[str, Any]) -> None:
        pose = state.get("pose", {})
        depth = state.get("depth", {})
        camera_info = state.get("camera_info", depth.get("camera_info", {}))
        runtime_debug = state.get("runtime_debug", {})
        plan = state.get("plan", {})
        planner_runtime = state.get("planner_runtime", {})
        archive = state.get("archive", {})
        reflex_runtime = state.get("reflex_runtime", {})
        reflex_execution = state.get("reflex_execution", {})
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

        header = tk.Label(
            self.root,
            text="Keyboard: W/S/A/D move, R/F up-down, Q/E yaw, C capture, V refresh preview, P request plan, T request reflex, Y execute reflex",
            anchor="w",
            justify="left",
        )
        header.pack(fill="x", padx=12, pady=(10, 6))

        status = tk.Label(self.root, textvariable=self.status_var, anchor="w", justify="left")
        status.pack(fill="x", padx=12, pady=(0, 10))

        depth_status = tk.Label(self.root, textvariable=self.depth_var, anchor="w", justify="left")
        depth_status.pack(fill="x", padx=12, pady=(0, 6))

        plan_status = tk.Label(self.root, textvariable=self.plan_var, anchor="w", justify="left")
        plan_status.pack(fill="x", padx=12, pady=(0, 6))

        archive_status = tk.Label(self.root, textvariable=self.archive_var, anchor="w", justify="left")
        archive_status.pack(fill="x", padx=12, pady=(0, 10))

        reflex_status = tk.Label(self.root, textvariable=self.reflex_var, anchor="w", justify="left")
        reflex_status.pack(fill="x", padx=12, pady=(0, 10))

        executor_status = tk.Label(self.root, textvariable=self.executor_var, anchor="w", justify="left")
        executor_status.pack(fill="x", padx=12, pady=(0, 10))

        task_frame = tk.Frame(self.root)
        task_frame.pack(fill="x", padx=12, pady=(0, 8))

        tk.Label(task_frame, text="Task Label").grid(row=0, column=0, sticky="w")
        tk.Entry(task_frame, textvariable=self.task_label_var).grid(row=0, column=1, sticky="ew", padx=(8, 8))
        tk.Button(task_frame, text="Set Task", command=self.set_task_label, width=14).grid(row=0, column=2, sticky="ew")

        tk.Label(task_frame, text="Capture Label").grid(row=1, column=0, sticky="w", pady=(8, 0))
        tk.Entry(task_frame, textvariable=self.capture_label_var).grid(row=1, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
        tk.Button(task_frame, text="Request Plan", command=self.request_plan, width=14).grid(row=1, column=2, sticky="ew", pady=(8, 0))
        tk.Button(task_frame, text="Request Reflex", command=self.request_reflex, width=14).grid(row=2, column=2, sticky="ew", pady=(8, 0))
        tk.Button(task_frame, text="Execute Reflex", command=self.execute_reflex, width=14).grid(row=3, column=2, sticky="ew", pady=(8, 0))
        task_frame.grid_columnconfigure(1, weight=1)

        task_help = tk.Label(
            self.root,
            text=(
                "Task Label: current semantic task for planner and capture metadata. "
                "Capture Label: one-shot note/suffix for the next saved sample. "
                "Capture saves RGB + depth together. "
                "Request Reflex refreshes the local policy suggestion state. "
                "Execute Reflex performs one gated autonomous reflex step."
            ),
            anchor="w",
            justify="left",
        )
        task_help.pack(fill="x", padx=12, pady=(0, 8))

        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=12, pady=6)

        buttons = [
            ("Forward (W)", lambda: self.send_move(forward_cm=self.args.move_step_cm, action_name="forward(W)")),
            ("Backward (S)", lambda: self.send_move(forward_cm=-self.args.move_step_cm, action_name="backward(S)")),
            ("Left (A)", lambda: self.send_move(right_cm=-self.args.move_step_cm, action_name="left(A)")),
            ("Right (D)", lambda: self.send_move(right_cm=self.args.move_step_cm, action_name="right(D)")),
            ("Up (R)", lambda: self.send_move(up_cm=self.args.vertical_step_cm, action_name="up(R)")),
            ("Down (F)", lambda: self.send_move(up_cm=-self.args.vertical_step_cm, action_name="down(F)")),
            ("Yaw Left (Q)", lambda: self.send_move(yaw_delta_deg=-self.args.yaw_step_deg, action_name="yaw_left(Q)")),
            ("Yaw Right (E)", lambda: self.send_move(yaw_delta_deg=self.args.yaw_step_deg, action_name="yaw_right(E)")),
            ("Capture (C)", self.capture),
            ("Request Plan (P)", self.request_plan),
            ("Request Reflex (T)", self.request_reflex),
            ("Execute Reflex (Y)", self.execute_reflex),
            ("Refresh (V)", lambda: (self.update_preview_once(), self.update_depth_once())),
            ("Shutdown Server", self.shutdown_server),
        ]

        for idx, (label, callback) in enumerate(buttons):
            btn = tk.Button(controls, text=label, command=callback, width=22)
            row = idx // 2
            col = idx % 2
            btn.grid(row=row, column=col, padx=6, pady=6, sticky="ew")

        controls.grid_columnconfigure(0, weight=1)
        controls.grid_columnconfigure(1, weight=1)

        shown_windows = []
        if not self.args.hide_preview_window:
            shown_windows.append(PREVIEW_WINDOW_NAME)
        if not self.args.hide_depth_window:
            shown_windows.append(DEPTH_WINDOW_NAME)
        windows_text = " / ".join(shown_windows) if shown_windows else "(none)"
        hint = tk.Label(
            self.root,
            text=(
                f"Preview windows: {windows_text}\n"
                f"Control server: {self.client.base_url}"
            ),
            anchor="w",
            justify="left",
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
