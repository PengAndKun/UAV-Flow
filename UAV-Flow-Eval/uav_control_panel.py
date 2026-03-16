"""
Remote control panel for `uav_control_server.py`.

This file is the "controller side" process:
- connects to the running UAV control server
- sends move commands
- fetches the latest camera frame
- triggers screenshots on the server
"""

import argparse
import json
import logging
import tkinter as tk
from typing import Any, Dict, Optional, Tuple
from urllib import error, request

import cv2
import numpy as np
from PIL import Image, ImageTk

logger = logging.getLogger(__name__)

PREVIEW_WINDOW_NAME = "UAV Remote Preview"


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
        req = request.Request(f"{self.base_url}/frame", method="GET")
        with request.urlopen(req, timeout=self.timeout_s) as resp:
            body = resp.read()
        data = np.frombuffer(body, dtype=np.uint8)
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Failed to decode frame returned by server")
        return frame


class UAVControlPanel:
    """Tk control panel that talks to the remote UAV control server."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.client = RemoteControlClient(f"http://{args.host}:{args.port}", args.timeout_s)
        self.root = tk.Tk()
        self.root.title("UAV Remote Control Panel")
        self.root.geometry("560x300")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.status_var = tk.StringVar(value="Connecting...")
        self.last_state: Optional[Dict[str, Any]] = None
        self.preview_window: Optional[tk.Toplevel] = None
        self.preview_label: Optional[tk.Label] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None

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

    def capture(self) -> None:
        result = self.safe_request(self.client.post_json, "/capture", {})
        if result:
            self.status_var.set(f"Captured: {result['image_path']}")
            self.update_preview_once()

    def shutdown_server(self) -> None:
        result = self.safe_request(self.client.post_json, "/shutdown", {})
        if result:
            self.status_var.set("Server shutdown requested")

    def update_status_from_state(self, state: Dict[str, Any]) -> None:
        pose = state.get("pose", {})
        self.status_var.set(
            "Pose "
            f"x={pose.get('x', 0.0):.1f} "
            f"y={pose.get('y', 0.0):.1f} "
            f"z={pose.get('z', 0.0):.1f} "
            f"yaw={pose.get('yaw', 0.0):.1f} "
            f"cmd={pose.get('command_yaw', 0.0):.1f} "
            f"task={pose.get('task_yaw', 0.0):.1f} "
            f"uav={pose.get('uav_yaw', 0.0):.1f} "
            f"action={state.get('last_action', 'idle')} | "
            f"server={self.client.base_url}"
        )

    def update_state_once(self) -> None:
        state = self.safe_request(self.client.get_json, "/state")
        if state:
            self.last_state = state
            self.update_status_from_state(state)

    def update_preview_once(self) -> None:
        frame = self.safe_request(self.client.get_frame)
        if frame is None:
            return
        if self.args.preview_width > 0 and self.args.preview_height > 0:
            frame = cv2.resize(frame, (self.args.preview_width, self.args.preview_height))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        self.preview_photo = ImageTk.PhotoImage(image=pil_image)
        if self.preview_label is not None:
            self.preview_label.configure(image=self.preview_photo)
            self.preview_label.image = self.preview_photo

    def schedule_state_refresh(self) -> None:
        self.update_state_once()
        self.root.after(self.args.state_interval_ms, self.schedule_state_refresh)

    def schedule_preview_refresh(self) -> None:
        self.update_preview_once()
        self.root.after(self.args.preview_interval_ms, self.schedule_preview_refresh)

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
            "v": self.update_preview_once,
        }
        for key, callback in bindings.items():
            self.root.bind(f"<KeyPress-{key}>", lambda _event, cb=callback: cb())
            self.root.bind(f"<KeyPress-{key.upper()}>", lambda _event, cb=callback: cb())

    def build_ui(self) -> None:
        self.preview_window = tk.Toplevel(self.root)
        self.preview_window.title(PREVIEW_WINDOW_NAME)
        self.preview_window.geometry(f"{self.args.preview_width}x{self.args.preview_height}")
        self.preview_window.protocol("WM_DELETE_WINDOW", self.on_close)
        self.preview_label = tk.Label(self.preview_window)
        self.preview_label.pack(fill="both", expand=True)

        header = tk.Label(
            self.root,
            text="Keyboard: W/S/A/D move, R/F up-down, Q/E yaw, C capture, V refresh preview",
            anchor="w",
            justify="left",
        )
        header.pack(fill="x", padx=12, pady=(10, 6))

        status = tk.Label(self.root, textvariable=self.status_var, anchor="w", justify="left")
        status.pack(fill="x", padx=12, pady=(0, 10))

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
            ("Refresh (V)", self.update_preview_once),
            ("Shutdown Server", self.shutdown_server),
        ]

        for idx, (label, callback) in enumerate(buttons):
            btn = tk.Button(controls, text=label, command=callback, width=22)
            row = idx // 2
            col = idx % 2
            btn.grid(row=row, column=col, padx=6, pady=6, sticky="ew")

        controls.grid_columnconfigure(0, weight=1)
        controls.grid_columnconfigure(1, weight=1)

        hint = tk.Label(
            self.root,
            text=f"Preview window: {PREVIEW_WINDOW_NAME}\nTarget server: {self.client.base_url}",
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
        self.root.destroy()

    def run(self) -> None:
        self.build_ui()
        self.update_state_once()
        self.update_preview_once()
        self.schedule_state_refresh()
        self.schedule_preview_refresh()
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
    parser.add_argument("--preview_interval_ms", type=int, default=180, help="Preview refresh interval")
    parser.add_argument("--state_interval_ms", type=int, default=500, help="State refresh interval")
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
