"""
map_overhead_widget.py
======================
A standalone Tkinter overhead (top-down) map widget for visualising a UAV
person-search mission in real time.

The widget can be embedded in any tk container (Frame, Toplevel, Tk root) by
passing it as the *parent* argument, or used standalone via the demo at the
bottom of this file.

Coordinate system
-----------------
World coordinates are in Unreal Engine cm-scale (x = forward, y = right).
The widget transforms them to canvas pixel space via world_to_canvas().

Color scheme
------------
Canvas background : #1e1e2e  (dark navy)
Grid lines        : #2a2a3e  (slightly lighter navy)
Text labels       : #e0e0f0  (off-white)
UAV marker        : red triangle + small circle
House status      :
    UNSEARCHED    – gray fill,   gray outline
    IN_PROGRESS   – yellow fill, orange outline (thick)
    EXPLORED      – green fill,  dark-green outline
    PERSON_FOUND  – red fill,    dark-red outline
Target indicator  – double concentric red rings around the house circle

Usage
-----
    import tkinter as tk
    from map_overhead_widget import OverheadMapWidget

    root = tk.Tk()
    bounds = (1000, -500, 5000, 3000)   # (min_x, min_y, max_x, max_y) in cm
    widget = OverheadMapWidget(root, world_bounds=bounds)
    widget.canvas.pack(fill="both", expand=True)

    widget.update_uav(x=2400.0, y=100.0, yaw_deg=-90.0)
    widget.update_houses([
        {"id": "house_A", "name": "House A", "center_x": 2400.0, "center_y": 100.0,
         "radius_cm": 700.0, "status": "UNSEARCHED", "is_target": False},
    ])
    root.mainloop()
"""

from __future__ import annotations

import math
import tkinter as tk
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

# ---------------------------------------------------------------------------
# Color constants
# ---------------------------------------------------------------------------

BG_COLOR      = "#1e1e2e"
GRID_COLOR    = "#2a2a3e"
TEXT_COLOR    = "#e0e0f0"
UAV_COLOR     = "#ff4444"
UAV_DOT_COLOR = "#ffaaaa"
CURRENT_RING_COLOR = "#55ccff"
ROUTE_COLOR = "#66ddff"

# House status → (fill, outline, outline_width)
_STATUS_STYLE: Dict[str, Tuple[str, str, int]] = {
    "UNSEARCHED":   ("#888888", "#888888", 1),
    "IN_PROGRESS":  ("#ffdd44", "#ff8800", 3),
    "EXPLORED":     ("#44cc66", "#228844", 2),
    "PERSON_FOUND": ("#ee3333", "#990000", 2),
}
_DEFAULT_STYLE = ("#888888", "#888888", 1)

# Minimum pixel radius for drawn house circles
_MIN_CIRCLE_PX = 8


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class OverheadMapWidget:
    """
    Top-down overhead map widget.

    Parameters
    ----------
    parent      : tk widget that owns this canvas.
    world_bounds: (min_x, min_y, max_x, max_y) in world cm units.
    canvas_w    : canvas width in pixels.
    canvas_h    : canvas height in pixels.
    """

    def __init__(
        self,
        parent: tk.Widget,
        world_bounds: Tuple[float, float, float, float],
        canvas_w: int = 480,
        canvas_h: int = 380,
    ) -> None:
        self._world_min_x, self._world_min_y, \
        self._world_max_x, self._world_max_y = world_bounds
        self._canvas_w = canvas_w
        self._canvas_h = canvas_h

        # Canvas
        self.canvas = tk.Canvas(
            parent,
            width=canvas_w,
            height=canvas_h,
            bg=BG_COLOR,
            highlightthickness=0,
        )
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        # Internal state
        self._uav_x: float   = 0.0
        self._uav_y: float   = 0.0
        self._uav_yaw: float = 0.0
        self._houses: List[dict] = []
        self._background_bgr: Optional[np.ndarray] = None
        self._background_photo: Optional[ImageTk.PhotoImage] = None
        self._route_target: Optional[Tuple[float, float]] = None
        self._image_size: Optional[Tuple[int, int]] = None
        self._affine_world_to_image: Optional[np.ndarray] = None
        self._calibration_anchors: List[dict] = []

        # House bounding boxes in canvas pixels, keyed by house id,
        # used for click-hit detection: {id: (cx, cy, r_px)}
        self._house_canvas_circles: Dict[str, Tuple[float, float, float]] = {}

        # Optional callback: fn(house_id: str)
        self._click_callback: Optional[Callable[[str], None]] = None
        self._map_click_callback: Optional[Callable[[float, float], None]] = None

        # Draw initial background
        self._draw_grid()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def update_uav(self, x: float, y: float, yaw_deg: float) -> None:
        """Update the UAV marker position and heading, then redraw."""
        self._uav_x   = x
        self._uav_y   = y
        self._uav_yaw = yaw_deg
        self._redraw()

    def update_houses(self, houses: List[dict]) -> None:
        """
        Replace the house list and redraw.

        Each dict must contain:
            id, name, center_x, center_y, radius_cm, status, is_target
        """
        self._houses = list(houses)
        self._redraw()

    def set_click_callback(self, fn: Callable[[str], None]) -> None:
        """
        Register a callback that fires when the user clicks a house circle.
        The callback receives the house id as its sole argument.
        """
        self._click_callback = fn

    def clear(self) -> None:
        """Clear all drawn items (UAV, houses) but keep the grid background."""
        self.canvas.delete("dynamic")
        self._house_canvas_circles.clear()

    def set_background_image(self, image_bgr: Optional[np.ndarray]) -> None:
        """
        Set an optional background image for the overhead map.

        The image is interpreted as a top-down map already aligned to the
        configured world bounds and is scaled to the canvas size.
        """
        self._background_bgr = None if image_bgr is None else image_bgr.copy()
        self._redraw()

    def set_route_target(self, world_xy: Optional[Tuple[float, float]]) -> None:
        """
        Set an optional route target in world coordinates.

        When present, a dashed line is drawn from the UAV to the target house.
        """
        self._route_target = None if world_xy is None else (float(world_xy[0]), float(world_xy[1]))
        self._redraw()

    def set_calibration(
        self,
        affine_world_to_image: Optional[List[List[float]]],
        image_size: Optional[Tuple[int, int]],
        anchors: Optional[List[dict]] = None,
    ) -> None:
        """Set an optional affine world->image calibration for the background map."""
        self._affine_world_to_image = None if affine_world_to_image is None else np.asarray(affine_world_to_image, dtype=np.float32)
        self._image_size = None if image_size is None else (int(image_size[0]), int(image_size[1]))
        self._calibration_anchors = list(anchors or [])
        self._redraw()

    def set_map_click_callback(self, fn: Callable[[float, float], None]) -> None:
        """Register a callback for raw background clicks in image-pixel space."""
        self._map_click_callback = fn

    def world_to_canvas(self, wx: float, wy: float) -> Tuple[float, float]:
        """
        Transform world (wx, wy) in cm to canvas pixel coordinates.

        World x (forward/east)  → canvas x (left → right)
        World y (right/south)   → canvas y (top → bottom)
        A small margin of 5 % is kept on each side.
        """
        if self._affine_world_to_image is not None and self._image_size is not None:
            image_x = (
                float(self._affine_world_to_image[0, 0]) * float(wx)
                + float(self._affine_world_to_image[0, 1]) * float(wy)
                + float(self._affine_world_to_image[0, 2])
            )
            image_y = (
                float(self._affine_world_to_image[1, 0]) * float(wx)
                + float(self._affine_world_to_image[1, 1]) * float(wy)
                + float(self._affine_world_to_image[1, 2])
            )
            return self.image_to_canvas(image_x, image_y)

        margin_x = self._canvas_w * 0.05
        margin_y = self._canvas_h * 0.05
        avail_w  = self._canvas_w - 2 * margin_x
        avail_h  = self._canvas_h - 2 * margin_y

        world_w = self._world_max_x - self._world_min_x
        world_h = self._world_max_y - self._world_min_y

        # Avoid division by zero
        sx = avail_w / world_w if world_w > 0 else 1.0
        sy = avail_h / world_h if world_h > 0 else 1.0

        cx = margin_x + (wx - self._world_min_x) * sx
        cy = margin_y + (wy - self._world_min_y) * sy

        return cx, cy

    def image_to_canvas(self, image_x: float, image_y: float) -> Tuple[float, float]:
        """Convert background-image pixel coordinates to canvas coordinates."""
        if self._image_size is None:
            return float(image_x), float(image_y)
        image_w = max(1, int(self._image_size[0]))
        image_h = max(1, int(self._image_size[1]))
        return (
            float(image_x) * self._canvas_w / float(image_w),
            float(image_y) * self._canvas_h / float(image_h),
        )

    def canvas_to_image(self, canvas_x: float, canvas_y: float) -> Tuple[float, float]:
        """Convert canvas coordinates to background-image pixel coordinates."""
        if self._image_size is None:
            return float(canvas_x), float(canvas_y)
        image_w = max(1, int(self._image_size[0]))
        image_h = max(1, int(self._image_size[1]))
        return (
            float(canvas_x) * float(image_w) / self._canvas_w,
            float(canvas_y) * float(image_h) / self._canvas_h,
        )

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_grid(self) -> None:
        """Draw a subtle 10x10 grid on the canvas background."""
        self.canvas.delete("grid")
        cols = 10
        rows = 10
        for i in range(1, cols):
            x = self._canvas_w * i / cols
            self.canvas.create_line(
                x, 0, x, self._canvas_h,
                fill=GRID_COLOR, width=1, tags="grid"
            )
        for j in range(1, rows):
            y = self._canvas_h * j / rows
            self.canvas.create_line(
                0, y, self._canvas_w, y,
                fill=GRID_COLOR, width=1, tags="grid"
            )

    def _redraw(self) -> None:
        """Clear dynamic items and redraw everything."""
        self.canvas.delete("background")
        self.canvas.delete("dynamic")
        self._house_canvas_circles.clear()

        if self._background_bgr is not None:
            self._draw_background_image()
        else:
            self._background_photo = None

        # Draw houses first (so UAV appears on top)
        for house in self._houses:
            self._draw_house(house)

        if self._route_target is not None:
            self._draw_route_line(*self._route_target)

        for anchor in self._calibration_anchors:
            self._draw_calibration_anchor(anchor)

        # Draw UAV marker
        cx, cy = self.world_to_canvas(self._uav_x, self._uav_y)
        self._draw_uav_marker(cx, cy, self._uav_yaw)

    def _draw_background_image(self) -> None:
        """Draw the optional top-down background image behind the grid/markers."""
        if self._background_bgr is None or self._background_bgr.size == 0:
            return
        preview = cv2.resize(
            self._background_bgr,
            (self._canvas_w, self._canvas_h),
            interpolation=cv2.INTER_AREA,
        )
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        self._background_photo = ImageTk.PhotoImage(Image.fromarray(preview_rgb))
        self.canvas.create_image(0, 0, image=self._background_photo, anchor="nw", tags="background")
        self.canvas.tag_lower("background", "grid")

    def _draw_house(self, house: dict) -> None:
        """Draw a single house circle, label, and optional target rings."""
        hid     = house.get("id", "?")
        name    = house.get("name", hid)
        wx      = float(house.get("center_x", 0.0))
        wy      = float(house.get("center_y", 0.0))
        r_cm    = float(house.get("radius_cm", 700.0))
        status  = str(house.get("status", "UNSEARCHED"))
        is_tgt  = bool(house.get("is_target", False))
        is_current = bool(house.get("is_current", False))

        cx, cy = self.world_to_canvas(wx, wy)

        # Scale radius: use the x scale factor (world_w → canvas_w)
        world_w = self._world_max_x - self._world_min_x
        margin_x = self._canvas_w * 0.05
        avail_w  = self._canvas_w - 2 * margin_x
        scale_x  = avail_w / world_w if world_w > 0 else 1.0
        r_px = max(_MIN_CIRCLE_PX, r_cm * scale_x)

        fill_color, outline_color, outline_width = _STATUS_STYLE.get(
            status, _DEFAULT_STYLE
        )

        # If this is the target, draw double concentric red rings first
        # (behind the main circle so they appear as a halo)
        if is_tgt:
            ring_gap = 4
            for ring_r in (r_px + ring_gap * 2, r_px + ring_gap):
                self.canvas.create_oval(
                    cx - ring_r, cy - ring_r,
                    cx + ring_r, cy + ring_r,
                    outline="#ff2222",
                    width=1,
                    tags="dynamic",
                )

        if is_current:
            current_r = r_px + 6
            self.canvas.create_oval(
                cx - current_r, cy - current_r,
                cx + current_r, cy + current_r,
                outline=CURRENT_RING_COLOR,
                width=2,
                dash=(5, 3),
                tags="dynamic",
            )

        # Main house circle
        self.canvas.create_oval(
            cx - r_px, cy - r_px,
            cx + r_px, cy + r_px,
            fill=outline_color if status == "IN_PROGRESS" else fill_color,
            outline=outline_color,
            width=outline_width,
            stipple="" if status != "IN_PROGRESS" else "",
            tags="dynamic",
        )

        # For IN_PROGRESS use a lighter inner fill to distinguish from outline
        if status == "IN_PROGRESS":
            inner_r = r_px - outline_width
            if inner_r > 2:
                self.canvas.create_oval(
                    cx - inner_r, cy - inner_r,
                    cx + inner_r, cy + inner_r,
                    fill=fill_color,
                    outline="",
                    tags="dynamic",
                )

        # House name label below the circle
        self.canvas.create_text(
            cx,
            cy + r_px + 8,
            text=name,
            fill=TEXT_COLOR,
            font=("Consolas", 8),
            anchor="n",
            tags="dynamic",
        )

        # Store canvas-space bounding circle for click detection
        self._house_canvas_circles[hid] = (cx, cy, r_px)

    def _draw_route_line(self, wx: float, wy: float) -> None:
        """Draw a dashed route line from the UAV to the current target."""
        ux, uy = self.world_to_canvas(self._uav_x, self._uav_y)
        tx, ty = self.world_to_canvas(wx, wy)
        self.canvas.create_line(
            ux, uy, tx, ty,
            fill=ROUTE_COLOR,
            width=2,
            dash=(8, 6),
            tags="dynamic",
        )
        dot_r = 4
        self.canvas.create_oval(
            tx - dot_r, ty - dot_r,
            tx + dot_r, ty + dot_r,
            fill=ROUTE_COLOR,
            outline="",
            tags="dynamic",
        )

    def _draw_calibration_anchor(self, anchor: dict) -> None:
        """Draw a numbered calibration anchor marker on the map."""
        ix = float(anchor.get("image_x", 0.0))
        iy = float(anchor.get("image_y", 0.0))
        label = str(anchor.get("label", anchor.get("index", "")))
        cx, cy = self.image_to_canvas(ix, iy)
        r = 5
        self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            fill="#00d5ff", outline="#ffffff", width=1, tags="dynamic",
        )
        if label:
            self.canvas.create_text(
                cx + 8, cy - 2,
                text=label,
                fill="#00f0ff",
                font=("Consolas", 8, "bold"),
                anchor="w",
                tags="dynamic",
            )

    def _draw_uav_marker(self, cx: float, cy: float, yaw_deg: float) -> None:
        """
        Draw a small filled triangle pointing in *yaw_deg* direction.

        Convention: yaw_deg = 0 → pointing right (+x canvas direction).
        Positive yaw is clockwise in canvas space (matches Unreal).
        """
        size   = 10   # half-length of the triangle
        width  = 6    # half-width at the base

        angle_rad = math.radians(yaw_deg)

        # Tip of the triangle (in the direction of travel)
        tip_x = cx + size * math.cos(angle_rad)
        tip_y = cy + size * math.sin(angle_rad)

        # Two base corners (perpendicular to travel direction)
        perp = angle_rad + math.pi / 2
        bl_x = cx - (size * 0.4) * math.cos(angle_rad) + width * math.cos(perp)
        bl_y = cy - (size * 0.4) * math.sin(angle_rad) + width * math.sin(perp)
        br_x = cx - (size * 0.4) * math.cos(angle_rad) - width * math.cos(perp)
        br_y = cy - (size * 0.4) * math.sin(angle_rad) - width * math.sin(perp)

        self.canvas.create_polygon(
            tip_x, tip_y,
            bl_x,  bl_y,
            br_x,  br_y,
            fill=UAV_COLOR,
            outline="#ffffff",
            width=1,
            tags="dynamic",
        )

        # Small dot at the center of the UAV body
        dot_r = 3
        self.canvas.create_oval(
            cx - dot_r, cy - dot_r,
            cx + dot_r, cy + dot_r,
            fill=UAV_DOT_COLOR,
            outline="",
            tags="dynamic",
        )

    # ------------------------------------------------------------------
    # Click handling
    # ------------------------------------------------------------------

    def _on_canvas_click(self, event: tk.Event) -> None:
        """Find which house (if any) was clicked and fire the callback."""
        ex, ey = event.x, event.y
        if self._click_callback is not None:
            for house_id, (cx, cy, r_px) in self._house_canvas_circles.items():
                dist = math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
                if dist <= r_px:
                    self._click_callback(house_id)
                    return
        if self._map_click_callback is not None:
            image_x, image_y = self.canvas_to_image(ex, ey)
            self._map_click_callback(image_x, image_y)


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import random

    root = tk.Tk()
    root.title("Overhead Map – Demo")
    root.configure(bg=BG_COLOR)

    bounds = (1000.0, -500.0, 5000.0, 3000.0)
    widget = OverheadMapWidget(root, world_bounds=bounds, canvas_w=640, canvas_h=480)
    widget.canvas.pack(padx=8, pady=8)

    HOUSES = [
        {"id": "house_A", "name": "House A", "center_x": 2400.0, "center_y": 100.0,
         "radius_cm": 700.0, "status": "IN_PROGRESS", "is_target": True},
        {"id": "house_B", "name": "House B", "center_x": 3800.0, "center_y": 800.0,
         "radius_cm": 750.0, "status": "UNSEARCHED",  "is_target": False},
        {"id": "house_C", "name": "House C", "center_x": 2100.0, "center_y": 2200.0,
         "radius_cm": 680.0, "status": "EXPLORED",    "is_target": False},
    ]

    uav_x, uav_y, uav_yaw = 2400.0, 100.0, -90.0

    def click_handler(hid: str) -> None:
        print(f"Clicked house: {hid}")

    widget.set_click_callback(click_handler)

    def animate() -> None:
        global uav_x, uav_y, uav_yaw
        uav_x   += random.uniform(-20, 20)
        uav_y   += random.uniform(-20, 20)
        uav_yaw  = (uav_yaw + 5) % 360
        widget.update_uav(uav_x, uav_y, uav_yaw)
        widget.update_houses(HOUSES)
        root.after(150, animate)

    animate()
    root.mainloop()
