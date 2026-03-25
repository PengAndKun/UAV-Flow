"""
multi_house_mission.py
======================
Multi-house mission controller for a UAV person-search mission.

Runs as a background thread and communicates with the UAV via HTTP calls to
uav_control_server_basic.py.  The mission proceeds through a series of phases:

    IDLE  →  FLYING_TO_HOUSE  →  APPROACHING_ENTRY  →  ENTERING
          →  INDOOR_SEARCH   →  RETURNING_TO_START  →  COMPLETE

After each house is finished (searched or timed out) the controller
automatically selects the nearest remaining unsearched house, or terminates if
no houses remain.

HTTP API assumed (uav_control_server_basic.py defaults to 127.0.0.1:5020):
    GET  /state          → {"pose": {"x", "y", "z", "task_yaw"}, ...}
    GET  /frame          → JPEG bytes
    POST /move_relative  → {"forward_cm", "right_cm", "up_cm",
                            "yaw_delta_deg", "action_name"}

Usage
-----
    from house_registry import HouseRegistry
    from multi_house_mission import MultiHouseMission

    registry = HouseRegistry("houses_config.json")
    config   = {}   # use all defaults
    mission  = MultiHouseMission("http://127.0.0.1:5020", registry, config)
    mission.start()
    mission.auto_select_nearest()
    # ... later ...
    mission.stop()
"""

from __future__ import annotations

import logging
import math
import threading
import time
from enum import Enum
from typing import Any, Dict, Optional

import requests

from house_registry import House, HouseRegistry, HouseStatus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Phase enum
# ---------------------------------------------------------------------------

class MissionPhase(str, Enum):
    IDLE               = "IDLE"
    FLYING_TO_HOUSE    = "FLYING_TO_HOUSE"
    APPROACHING_ENTRY  = "APPROACHING_ENTRY"
    ENTERING           = "ENTERING"
    INDOOR_SEARCH      = "INDOOR_SEARCH"
    RETURNING_TO_START = "RETURNING_TO_START"
    COMPLETE           = "COMPLETE"


# ---------------------------------------------------------------------------
# Default configuration values
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    "fly_altitude_cm":          600,    # target z while flying between houses
    "step_size_cm":             25,     # cm per single forward/back/left/right move
    "yaw_step_deg":             20,     # degrees per single yaw command
    "max_steps_per_house":      400,    # total steps before giving up on indoor search
    "entry_dist_threshold_cm":  180,    # horizontal dist to consider "at entry"
    "indoor_search_steps":      200,    # steps dedicated to the indoor sweep
    "step_delay_s":             0.35,   # seconds between successive steps
}

# Number of yaw steps for a full 360° rotation (at yaw_step_deg = 20°)
_FULL_ROTATION_STEPS = 18   # 18 × 20° = 360°

# How close (horizontal) we need to be to the house center to transition phases
_HOUSE_CENTER_THRESHOLD_CM = 150.0


# ---------------------------------------------------------------------------
# Move definitions: action name → (forward_cm, right_cm, up_cm, yaw_delta_deg)
# ---------------------------------------------------------------------------

def _build_move_table(step: float, yaw: float) -> Dict[str, tuple]:
    return {
        "forward":   (step,  0,    0,  0),
        "backward":  (-step, 0,    0,  0),
        "left":      (0,    -step, 0,  0),
        "right":     (0,     step, 0,  0),
        "up":        (0,     0,    step,  0),
        "down":      (0,     0,   -step, 0),
        "yaw_left":  (0,     0,    0,  -yaw),
        "yaw_right": (0,     0,    0,   yaw),
        "hold":      (0,     0,    0,   0),
    }


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class MultiHouseMission:
    """
    Background-thread mission controller.

    Parameters
    ----------
    server_url : Base URL of uav_control_server_basic.py, e.g.
                 "http://127.0.0.1:5020".
    registry   : Populated HouseRegistry instance.
    config     : Optional overrides for _DEFAULT_CONFIG keys.
    """

    def __init__(
        self,
        server_url: str,
        registry: HouseRegistry,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._server_url = server_url.rstrip("/")
        self._registry   = registry

        # Merge user config over defaults
        cfg = dict(_DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        self._cfg = cfg

        # Build move table from config
        self._move_table = _build_move_table(
            self._cfg["step_size_cm"],
            self._cfg["yaw_step_deg"],
        )

        # Mission state – access always under self._lock
        self._phase: MissionPhase  = MissionPhase.IDLE
        self._step_count: int      = 0
        self._last_action: str     = "none"
        self._last_perception: str = ""
        self._approach_yaw_steps: int = 0   # cumulative yaw steps in approach phase

        # Threading primitives
        self._lock        = threading.Lock()
        self._stop_event  = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the mission background thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("MultiHouseMission: already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._mission_loop,
            name="MultiHouseMission",
            daemon=True,
        )
        self._thread.start()
        logger.info("MultiHouseMission: thread started")

    def stop(self) -> None:
        """Signal the mission thread to stop and wait for it to exit."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("MultiHouseMission: thread stopped")

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------

    def select_target(self, house_id: str) -> bool:
        """
        Manually select *house_id* as the next target.

        Returns True if the house was found in the registry.
        """
        ok = self._registry.set_target(house_id)
        if ok:
            with self._lock:
                self._phase      = MissionPhase.FLYING_TO_HOUSE
                self._step_count = 0
                self._approach_yaw_steps = 0
                self._last_action = f"target selected: {house_id}"
            logger.info("MultiHouseMission: target set to '%s'", house_id)
        return ok

    def auto_select_nearest(self) -> Optional[str]:
        """
        Query the UAV pose and select the nearest unsearched house.

        Returns the selected house id, or None if no unsearched houses remain.
        """
        try:
            pose = self._get_uav_pose()
        except Exception as exc:
            logger.error("auto_select_nearest: could not get pose – %s", exc)
            return None

        uav_x = float(pose.get("x", 0.0))
        uav_y = float(pose.get("y", 0.0))

        nearest = self._registry.get_nearest_unsearched(uav_x, uav_y)
        if nearest is None:
            logger.info("auto_select_nearest: no unsearched houses remain")
            with self._lock:
                self._phase = MissionPhase.COMPLETE
            return None

        self.select_target(nearest.id)
        return nearest.id

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        """Return a snapshot of the current mission state."""
        with self._lock:
            target_house = self._registry.get_target_house()
            return {
                "phase":           self._phase.value,
                "target_house_id": target_house.id if target_house else None,
                "step_count":      self._step_count,
                "last_action":     self._last_action,
                "last_perception": self._last_perception,
                "registry":        self._registry.get_status_summary(),
            }

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _mission_loop(self) -> None:
        """Mission thread entry point.  Runs until _stop_event is set."""
        logger.info("MultiHouseMission: mission loop started")

        while not self._stop_event.is_set():
            with self._lock:
                phase = self._phase

            try:
                if phase == MissionPhase.IDLE:
                    # Nothing to do – wait for a target to be assigned
                    time.sleep(0.25)
                    continue

                elif phase == MissionPhase.FLYING_TO_HOUSE:
                    self._fly_to_house_step()

                elif phase == MissionPhase.APPROACHING_ENTRY:
                    self._approach_entry_step()

                elif phase == MissionPhase.ENTERING:
                    self._entering_step()

                elif phase == MissionPhase.INDOOR_SEARCH:
                    self._indoor_search_step()

                elif phase in (MissionPhase.RETURNING_TO_START,
                               MissionPhase.COMPLETE):
                    # Terminal / not-yet-implemented phases – just idle
                    time.sleep(0.5)
                    continue

            except Exception as exc:
                logger.exception("Mission loop error in phase %s: %s", phase, exc)
                # Brief back-off on error so we don't hammer the server
                time.sleep(1.0)
                continue

            time.sleep(self._cfg["step_delay_s"])

        logger.info("MultiHouseMission: mission loop exited")

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _get_uav_pose(self) -> Dict[str, float]:
        """
        GET /state and return the pose sub-dict.

        Returns a dict with keys: x, y, z, task_yaw (all floats).
        Raises requests.RequestException on network errors.
        """
        resp = requests.get(f"{self._server_url}/state", timeout=3.0)
        resp.raise_for_status()
        data = resp.json()
        pose = data.get("pose", {})
        return {
            "x":        float(pose.get("x",        0.0)),
            "y":        float(pose.get("y",        0.0)),
            "z":        float(pose.get("z",        0.0)),
            "task_yaw": float(pose.get("task_yaw", 0.0)),
        }

    def _get_rgb_frame(self) -> bytes:
        """GET /frame and return raw JPEG bytes."""
        resp = requests.get(f"{self._server_url}/frame", timeout=5.0)
        resp.raise_for_status()
        return resp.content

    def _get_depth_summary(self) -> Dict[str, Any]:
        """GET /state and extract the depth summary sub-dict."""
        resp = requests.get(f"{self._server_url}/state", timeout=3.0)
        resp.raise_for_status()
        return resp.json().get("depth", {})

    def _send_move(self, action_name: str) -> Dict[str, Any]:
        """
        Look up *action_name* in the move table and POST /move_relative.

        Supported actions: forward, backward, left, right, up, down,
                           yaw_left, yaw_right, hold.

        Returns the server response JSON.
        Raises ValueError if action_name is not recognised.
        """
        if action_name not in self._move_table:
            raise ValueError(
                f"Unknown action '{action_name}'. "
                f"Valid actions: {list(self._move_table)}"
            )

        fwd, right, up, yaw = self._move_table[action_name]
        payload = {
            "forward_cm":    fwd,
            "right_cm":      right,
            "up_cm":         up,
            "yaw_delta_deg": yaw,
            "action_name":   action_name,
        }
        resp = requests.post(
            f"{self._server_url}/move_relative",
            json=payload,
            timeout=5.0,
        )
        resp.raise_for_status()

        with self._lock:
            self._last_action = action_name
            self._step_count += 1

        return resp.json()

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _fly_to_house_step(self) -> None:
        """
        Phase: FLYING_TO_HOUSE
        -----------------------
        1. If below fly_altitude_cm → climb.
        2. If more than _HOUSE_CENTER_THRESHOLD_CM away from house center → yaw
           toward house then move forward.
        3. If close enough → descend and transition to APPROACHING_ENTRY.
        """
        target = self._registry.get_target_house()
        if target is None:
            logger.warning("_fly_to_house_step: no target house selected")
            with self._lock:
                self._phase = MissionPhase.IDLE
            return

        pose = self._get_uav_pose()
        uav_x, uav_y, uav_z = pose["x"], pose["y"], pose["z"]
        uav_yaw = pose["task_yaw"]

        fly_alt = float(self._cfg["fly_altitude_cm"])

        # Step 1: climb if needed
        if uav_z < fly_alt - 30.0:
            self._send_move("up")
            logger.debug("_fly_to_house_step: climbing (%s cm < %s cm)", uav_z, fly_alt)
            return

        # Step 2: navigate toward the house center
        dx = target.center_x - uav_x
        dy = target.center_y - uav_y
        horiz_dist = math.sqrt(dx * dx + dy * dy)

        logger.debug(
            "_fly_to_house_step: dist=%.1f cm to '%s'", horiz_dist, target.id
        )

        if horiz_dist > _HOUSE_CENTER_THRESHOLD_CM:
            # Bearing to target (degrees, 0 = +x axis, clockwise)
            target_bearing = math.degrees(math.atan2(dy, dx))
            yaw_error = _normalize_angle(target_bearing - uav_yaw)

            if abs(yaw_error) > self._cfg["yaw_step_deg"] + 5:
                # Turn toward the house before advancing
                action = "yaw_right" if yaw_error > 0 else "yaw_left"
                self._send_move(action)
                logger.debug(
                    "_fly_to_house_step: yaw error %.1f° → %s", yaw_error, action
                )
            else:
                self._send_move("forward")
        else:
            # Close enough – begin approach
            logger.info(
                "_fly_to_house_step: reached house '%s', transitioning to APPROACHING_ENTRY",
                target.id,
            )
            with self._lock:
                self._phase = MissionPhase.APPROACHING_ENTRY
                self._approach_yaw_steps = 0

    def _approach_entry_step(self) -> None:
        """
        Phase: APPROACHING_ENTRY
        -------------------------
        Descend to ~200 cm (center_z) while slowly rotating to find the door.
        After a full 360° rotation without finding the door, assume we are
        close enough and transition to INDOOR_SEARCH.

        Door detection hook (_detect_door_in_frame) always returns False in this
        placeholder implementation; replace with a YOLO call when available.
        """
        target = self._registry.get_target_house()
        if target is None:
            with self._lock:
                self._phase = MissionPhase.IDLE
            return

        pose    = self._get_uav_pose()
        uav_z   = pose["z"]
        goal_z  = target.center_z   # ground-level z

        # Descend if still above target height
        if uav_z > goal_z + 50.0:
            self._send_move("down")
            logger.debug("_approach_entry_step: descending (%s → %s)", uav_z, goal_z)
            return

        # Check for door (placeholder – always False)
        door_found = self._detect_door_in_frame()
        if door_found:
            logger.info(
                "_approach_entry_step: door detected at '%s', entering", target.id
            )
            with self._lock:
                self._phase = MissionPhase.ENTERING
            return

        # Rotate slowly scanning for door
        self._send_move("yaw_right")

        with self._lock:
            self._approach_yaw_steps += 1
            yaw_steps = self._approach_yaw_steps

        logger.debug(
            "_approach_entry_step: yaw_right step %d / %d",
            yaw_steps, _FULL_ROTATION_STEPS,
        )

        if yaw_steps >= _FULL_ROTATION_STEPS:
            # Full rotation without finding a door – proceed anyway
            logger.info(
                "_approach_entry_step: full rotation done, transitioning to INDOOR_SEARCH"
            )
            with self._lock:
                self._phase = MissionPhase.INDOOR_SEARCH
                self._step_count = 0

    def _entering_step(self) -> None:
        """
        Phase: ENTERING
        ----------------
        Move forward a few steps to cross the threshold, then transition
        to INDOOR_SEARCH.
        """
        target = self._registry.get_target_house()
        if target is None:
            with self._lock:
                self._phase = MissionPhase.IDLE
            return

        # Simple approach: advance toward the center of the house
        pose = self._get_uav_pose()
        uav_x, uav_y = pose["x"], pose["y"]
        dx = target.center_x - uav_x
        dy = target.center_y - uav_y
        dist = math.sqrt(dx * dx + dy * dy)

        entry_threshold = float(self._cfg["entry_dist_threshold_cm"])

        if dist > entry_threshold:
            self._send_move("forward")
            logger.debug("_entering_step: approaching entry, dist=%.1f cm", dist)
        else:
            logger.info(
                "_entering_step: inside threshold for '%s', starting INDOOR_SEARCH",
                target.id,
            )
            with self._lock:
                self._phase      = MissionPhase.INDOOR_SEARCH
                self._step_count = 0

    def _indoor_search_step(self) -> None:
        """
        Phase: INDOOR_SEARCH
        ---------------------
        Systematic sweep pattern: alternate forward movement steps with
        periodic yaw sweeps to cover the interior.

        After max_steps_per_house the house is marked as explored (no person
        found) and the controller moves on to the next house.

        Person detection is not implemented here – extend this method with
        a YOLO or VLM call on _get_rgb_frame() and call
        _complete_house_search(person_found=True) when a person is detected.
        """
        target = self._registry.get_target_house()
        if target is None:
            with self._lock:
                self._phase = MissionPhase.IDLE
            return

        with self._lock:
            steps = self._step_count
            max_steps = int(self._cfg["max_steps_per_house"])

        if steps >= max_steps:
            logger.info(
                "_indoor_search_step: max steps reached for '%s', completing search",
                target.id,
            )
            self._complete_house_search(person_found=False)
            return

        # Simple sweep pattern:
        #   Steps 0-9:    move forward
        #   Steps 10-13:  yaw right 4×
        #   Steps 14-23:  move forward
        #   Steps 24-27:  yaw left 4×
        #   ... repeat
        sweep_cycle = 28
        pos_in_cycle = steps % sweep_cycle

        if 0 <= pos_in_cycle < 10:
            action = "forward"
        elif 10 <= pos_in_cycle < 14:
            action = "yaw_right"
        elif 14 <= pos_in_cycle < 24:
            action = "forward"
        else:
            action = "yaw_left"

        self._send_move(action)
        logger.debug(
            "_indoor_search_step: step %d/%d – %s", steps, max_steps, action
        )

        with self._lock:
            self._last_perception = f"indoor_sweep step {steps}"

    # ------------------------------------------------------------------
    # House completion
    # ------------------------------------------------------------------

    def _complete_house_search(
        self,
        person_found: bool,
        person_location: Optional[Dict] = None,
    ) -> None:
        """
        Mark the current target house as explored/found and move on.

        If more unsearched houses remain, auto-selects the nearest one.
        Otherwise transitions to COMPLETE.
        """
        target = self._registry.get_target_house()
        if target is None:
            with self._lock:
                self._phase = MissionPhase.IDLE
            return

        self._registry.mark_explored(
            target.id,
            person_found=person_found,
            person_location=person_location,
            notes=f"Completed after {self._step_count} indoor steps.",
        )
        logger.info(
            "_complete_house_search: '%s' done (person_found=%s)",
            target.id, person_found,
        )

        with self._lock:
            self._phase      = MissionPhase.IDLE
            self._step_count = 0

        # Try to move on to the next house
        next_id = self.auto_select_nearest()
        if next_id is None:
            with self._lock:
                self._phase = MissionPhase.COMPLETE
            logger.info("_complete_house_search: all houses searched – mission COMPLETE")

    # ------------------------------------------------------------------
    # Door detection hook (placeholder)
    # ------------------------------------------------------------------

    def _detect_door_in_frame(self) -> bool:
        """
        Placeholder door-detection hook.

        Always returns False.  Replace the body with a YOLO model call:

            frame_bytes = self._get_rgb_frame()
            results     = yolo_model.infer_bytes(frame_bytes)
            return any(cls == "door" for cls, conf in results if conf > 0.6)
        """
        return False

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def phase(self) -> MissionPhase:
        with self._lock:
            return self._phase

    @property
    def step_count(self) -> int:
        with self._lock:
            return self._step_count


# ---------------------------------------------------------------------------
# Angle utility
# ---------------------------------------------------------------------------

def _normalize_angle(angle_deg: float) -> float:
    """Wrap *angle_deg* into the range (-180, 180]."""
    return (angle_deg + 180.0) % 360.0 - 180.0


# ---------------------------------------------------------------------------
# Quick smoke-test (does NOT require a running server)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import os
    import tempfile

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Build a minimal registry from an inline config
    cfg = {
        "houses": [
            {"id": "h1", "name": "House 1", "center_x": 2400.0, "center_y": 100.0,
             "center_z": 200.0, "approach_z": 600.0, "radius_cm": 700.0, "entry_yaw_hint": -90.0},
            {"id": "h2", "name": "House 2", "center_x": 3800.0, "center_y": 800.0,
             "center_z": 200.0, "approach_z": 600.0, "radius_cm": 750.0, "entry_yaw_hint": 180.0},
        ]
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        tmp = f.name

    try:
        registry = HouseRegistry(tmp)
        mission  = MultiHouseMission(
            server_url="http://127.0.0.1:5020",
            registry=registry,
            config={"step_delay_s": 0.1},
        )

        print("Initial status:", mission.get_status())

        # Manually set target without a running server
        ok = registry.set_target("h1")
        print("Set target h1:", ok)
        print("Summary:", registry.get_status_summary())

        # Show move table
        print("\nMove table:")
        for name, params in mission._move_table.items():
            print(f"  {name:12s} → fwd={params[0]:+6.1f}  right={params[1]:+6.1f}"
                  f"  up={params[2]:+6.1f}  yaw={params[3]:+6.1f}")

    finally:
        os.unlink(tmp)

    print("\nSmoke-test passed.")
