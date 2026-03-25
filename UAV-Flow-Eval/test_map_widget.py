"""
Layer 3 — Visual test for map_overhead_widget.py
Opens a real Tkinter window with animated fake UAV movement.
No server needed. Close the window to exit.

Run:
    cd UAV-Flow-Eval
    python test_map_widget.py
"""
import math
import sys
import threading
import time
import tkinter as tk

from map_overhead_widget import OverheadMapWidget

HOUSES = [
    {"id": "house_A", "name": "House A", "center_x": 2400.0, "center_y": 100.0,
     "radius_cm": 700.0, "status": "UNSEARCHED", "is_target": False},
    {"id": "house_B", "name": "House B", "center_x": 3800.0, "center_y": 800.0,
     "radius_cm": 750.0, "status": "IN_PROGRESS", "is_target": True},
    {"id": "house_C", "name": "House C", "center_x": 2100.0, "center_y": 2200.0,
     "radius_cm": 680.0, "status": "EXPLORED", "is_target": False},
    {"id": "house_D", "name": "House D", "center_x": 4200.0, "center_y": 2400.0,
     "radius_cm": 720.0, "status": "PERSON_FOUND", "is_target": False},
]

WORLD_BOUNDS = (1000.0, -500.0, 5000.0, 3000.0)
STATUS_CYCLE = ["UNSEARCHED", "IN_PROGRESS", "EXPLORED", "PERSON_FOUND"]


def run_visual_test():
    root = tk.Tk()
    root.title("Map Widget Visual Test — close window to exit")

    info = tk.Label(root, text="Click a house marker to select it as target.",
                    font=("Arial", 11), pady=6)
    info.pack(fill="x")

    click_var = tk.StringVar(value="Last click: none")
    tk.Label(root, textvariable=click_var, fg="#444").pack(fill="x")

    widget = OverheadMapWidget(root, world_bounds=WORLD_BOUNDS, canvas_w=600, canvas_h=460)

    status_idx = [0]
    target_id = ["house_B"]
    houses_state = {h["id"]: dict(h) for h in HOUSES}

    def on_house_click(house_id: str):
        old_target = target_id[0]
        target_id[0] = house_id
        if old_target in houses_state:
            houses_state[old_target]["is_target"] = False
        houses_state[house_id]["is_target"] = True
        click_var.set(f"Last click: {house_id} (now target)")
        _redraw_houses()

    def _redraw_houses():
        widget.update_houses(list(houses_state.values()))

    widget.set_click_callback(on_house_click)

    # Control buttons
    btn_frame = tk.Frame(root)
    btn_frame.pack(fill="x", pady=4)

    def cycle_status():
        """Cycle the target house through all status values."""
        tid = target_id[0]
        if tid in houses_state:
            cur = houses_state[tid]["status"]
            idx = STATUS_CYCLE.index(cur) if cur in STATUS_CYCLE else 0
            houses_state[tid]["status"] = STATUS_CYCLE[(idx + 1) % len(STATUS_CYCLE)]
            _redraw_houses()

    tk.Button(btn_frame, text="Cycle Target Status", command=cycle_status,
              width=20).pack(side="left", padx=4)
    tk.Button(btn_frame, text="Quit", command=root.destroy,
              width=10).pack(side="right", padx=4)

    # Initial draw
    widget.update_uav(2400.0, 100.0, -90.0)
    _redraw_houses()

    # Animate UAV in background thread
    stop_event = threading.Event()

    def animate():
        t = 0.0
        while not stop_event.is_set():
            cx = 2400.0 + 600.0 * math.sin(t * 0.4)
            cy = 1000.0 + 800.0 * math.cos(t * 0.3)
            yaw = math.degrees(t * 0.5) % 360
            root.after(0, lambda x=cx, y=cy, a=yaw: widget.update_uav(x, y, a))
            t += 0.08
            time.sleep(0.08)

    anim_thread = threading.Thread(target=animate, daemon=True)
    anim_thread.start()

    def on_close():
        stop_event.set()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
    stop_event.set()
    print("Visual test closed.")


if __name__ == "__main__":
    run_visual_test()
