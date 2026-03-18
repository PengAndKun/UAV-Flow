"""
Minimal lesson4-style depth planar helpers.

This module recreates the subset of the old `lesson4` utilities that the
current `uav_control_server.py` still depends on:

- `coerce_depth_planar_image`
- `generate_camera_info`

It intentionally stays lightweight and ROS-free.
"""

from __future__ import annotations

from math import tan
from typing import Any, Dict

import numpy as np


def _to_numpy_depth(raw_depth: Any) -> np.ndarray:
    """Convert arbitrary depth-like input into a 2D float32 numpy array."""
    if isinstance(raw_depth, np.ndarray):
        depth = raw_depth
    else:
        depth = np.asarray(raw_depth)

    if depth.ndim == 3:
        # Some providers return HxWx1 or HxWxC. Keep the first channel.
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise ValueError(f"Expected a 2D depth image, got shape={depth.shape!r}")
    return depth.astype(np.float32, copy=False)


def coerce_depth_planar_image(raw_depth: Any) -> np.ndarray:
    """
    Normalize a depth-planar image into float32 centimeters.

    Heuristics:
    - if values look meter-scale (`max <= 20`), convert to centimeters
    - otherwise assume the input is already in centimeters
    - keep invalid values as-is so downstream code can clip/filter them
    """
    depth = _to_numpy_depth(raw_depth)
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        return depth

    max_value = float(np.max(finite))
    min_value = float(np.min(finite))

    # Unreal / simulators often provide meters in a small float range.
    if 0.0 <= max_value <= 20.0:
        depth = depth * 100.0
    # If everything is in [0, 1], it's almost certainly meters as normalized-ish
    # planner depth; this branch is redundant but keeps the intent obvious.
    elif 0.0 <= max_value <= 1.0:
        depth = depth * 100.0

    # Negative distances are non-physical; mark them invalid for downstream use.
    depth = np.where(depth < 0.0, np.nan, depth)

    # Depth images that are entirely zeros are effectively invalid.
    if min_value == 0.0 and max_value == 0.0:
        depth = np.where(depth == 0.0, np.nan, depth)

    return depth.astype(np.float32, copy=False)


def generate_camera_info(width: int, height: int, fov_deg: float, frame_id: str) -> Dict[str, Any]:
    """
    Build a ROS-style camera_info dictionary from image size and horizontal FOV.

    This mirrors the structure previously used in the project and is sufficient
    for depth preview, replay metadata, and later point projection logic.
    """
    width = int(width)
    height = int(height)
    fov_deg = float(fov_deg)
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid camera size: width={width}, height={height}")
    if fov_deg <= 0.0:
        raise ValueError(f"Invalid camera FOV: {fov_deg}")

    fov_rad = np.deg2rad(fov_deg)
    fx = width / (2.0 * tan(fov_rad / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    return {
        "frame_id": frame_id,
        "width": width,
        "height": height,
        "distortion_model": "plumb_bob",
        "d": [0.0, 0.0, 0.0, 0.0, 0.0],
        "k": [
            float(fx), 0.0, float(cx),
            0.0, float(fy), float(cy),
            0.0, 0.0, 1.0,
        ],
        "r": [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ],
        "p": [
            float(fx), 0.0, float(cx), 0.0,
            0.0, float(fy), float(cy), 0.0,
            0.0, 0.0, 1.0, 0.0,
        ],
        "fov_deg": float(fov_deg),
    }
