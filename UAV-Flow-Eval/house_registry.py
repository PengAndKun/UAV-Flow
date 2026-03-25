"""
house_registry.py
=================
Tracks the multi-house search state for a UAV person-search mission.

Each house has a status (UNSEARCHED → IN_PROGRESS → EXPLORED / PERSON_FOUND)
and spatial metadata (center position, bounding radius, fly-over altitude, entry
yaw hint).  The registry can be loaded from and saved to a JSON config file so
state persists across process restarts.

Typical usage
-------------
    registry = HouseRegistry("houses_config.json")
    registry.set_target("house_A")
    ...
    registry.mark_explored("house_A", person_found=True,
                           person_location={"x": 2420.0, "y": 85.0, "z": 130.0})
    registry.save_to_file("houses_config_state.json")
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status enum
# ---------------------------------------------------------------------------

class HouseStatus(str, Enum):
    """Search status for a single house."""
    UNSEARCHED   = "UNSEARCHED"
    IN_PROGRESS  = "IN_PROGRESS"
    EXPLORED     = "EXPLORED"
    PERSON_FOUND = "PERSON_FOUND"


# ---------------------------------------------------------------------------
# House dataclass
# ---------------------------------------------------------------------------

@dataclass
class House:
    """All metadata and runtime state for one house in the mission area."""

    # --- Static / config fields ---
    id: str
    name: str
    center_x: float
    center_y: float
    center_z: float                      # ground-level z of the house
    approach_z: float = 500.0            # fly-over altitude in cm (above world origin)
    radius_cm: float = 700.0             # bounding-circle radius used for proximity tests
    entry_yaw_hint: float = 0.0          # suggested UAV yaw (deg) to face the entrance

    # --- Runtime state fields ---
    status: HouseStatus = HouseStatus.UNSEARCHED
    search_start_time: Optional[float]   = field(default=None)
    search_end_time: Optional[float]     = field(default=None)
    person_location: Optional[Dict]      = field(default=None)  # {"x", "y", "z"}
    notes: str                           = ""

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def distance_to(self, x: float, y: float) -> float:
        """Horizontal Euclidean distance from an (x, y) world position."""
        return math.sqrt((self.center_x - x) ** 2 + (self.center_y - y) ** 2)

    def to_dict(self) -> dict:
        """Return a fully serialisable representation of this house."""
        return {
            "id":                self.id,
            "name":              self.name,
            "center_x":         self.center_x,
            "center_y":         self.center_y,
            "center_z":         self.center_z,
            "approach_z":       self.approach_z,
            "radius_cm":        self.radius_cm,
            "entry_yaw_hint":   self.entry_yaw_hint,
            "status":           self.status.value,
            "search_start_time": self.search_start_time,
            "search_end_time":   self.search_end_time,
            "person_location":   self.person_location,
            "notes":             self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "House":
        """Reconstruct a House from a dict (e.g. loaded from JSON)."""
        return cls(
            id              = str(d["id"]),
            name            = str(d.get("name", d["id"])),
            center_x        = float(d["center_x"]),
            center_y        = float(d["center_y"]),
            center_z        = float(d.get("center_z", 200.0)),
            approach_z      = float(d.get("approach_z", 500.0)),
            radius_cm       = float(d.get("radius_cm", 700.0)),
            entry_yaw_hint  = float(d.get("entry_yaw_hint", 0.0)),
            status          = HouseStatus(d.get("status", HouseStatus.UNSEARCHED.value)),
            search_start_time = d.get("search_start_time"),
            search_end_time   = d.get("search_end_time"),
            person_location   = d.get("person_location"),
            notes             = str(d.get("notes", "")),
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class HouseRegistry:
    """
    Manages the full set of houses for a multi-house UAV search mission.

    Houses are loaded from a JSON config file whose ``"houses"`` array
    contains dicts compatible with House.from_dict().  An optional
    ``"world_bounds"`` key is preserved and re-emitted on save so that
    map widgets can read it from the same file.
    """

    def __init__(self, config_path: str) -> None:
        self._houses: Dict[str, House] = {}
        self._current_target_id: str = ""
        self._world_bounds: dict = {}   # preserved verbatim for serialisation
        self._overhead_map: dict = {}   # optional top-down map config

        try:
            self.load_from_file(config_path)
            logger.info("HouseRegistry: loaded %d houses from %s",
                        len(self._houses), config_path)
        except FileNotFoundError:
            logger.warning("HouseRegistry: config file not found (%s). "
                           "Starting with empty registry.", config_path)
        except Exception as exc:
            logger.error("HouseRegistry: failed to load %s – %s. "
                         "Starting with empty registry.", config_path, exc)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load_from_file(self, path: str) -> None:
        """
        Parse *path* and replace the current registry contents.

        Expected JSON schema::

            {
              "world_bounds": {...},   // optional
              "houses": [ {house dict}, ... ]
            }

        Raises FileNotFoundError if the file does not exist.
        """
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        self._world_bounds = raw.get("world_bounds", {})
        self._overhead_map = raw.get("overhead_map", {})
        houses_raw = raw.get("houses", [])
        # Restore persisted target (saved by save_to_file)
        self._current_target_id = str(raw.get("current_target_id", ""))

        self._houses = {}
        for h_dict in houses_raw:
            house = House.from_dict(h_dict)
            self._houses[house.id] = house

    def save_to_file(self, path: str) -> None:
        """Serialise the current registry state to *path* (JSON)."""
        payload = {
            "world_bounds": self._world_bounds,
            "overhead_map": self._overhead_map,
            "current_target_id": self._current_target_id,
            "houses": [h.to_dict() for h in self._houses.values()],
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.debug("HouseRegistry: saved state to %s", path)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all_houses(self) -> List[House]:
        """Return all houses in insertion order."""
        return list(self._houses.values())

    def get_house(self, house_id: str) -> Optional[House]:
        """Return the House with *house_id*, or None if not found."""
        return self._houses.get(house_id)

    def get_target_house(self) -> Optional[House]:
        """Return the currently selected target house, or None."""
        if self._current_target_id:
            return self._houses.get(self._current_target_id)
        return None

    def get_nearest_unsearched(self, uav_x: float, uav_y: float) -> Optional[House]:
        """
        Return the UNSEARCHED house whose horizontal center is closest to
        (*uav_x*, *uav_y*), or None if no unsearched houses remain.
        """
        candidates = [h for h in self._houses.values()
                      if h.status == HouseStatus.UNSEARCHED]
        if not candidates:
            return None
        return min(candidates, key=lambda h: h.distance_to(uav_x, uav_y))

    def get_nearest_house(self, uav_x: float, uav_y: float) -> Optional[House]:
        """Return the geometrically nearest house regardless of status."""
        if not self._houses:
            return None
        return min(self._houses.values(), key=lambda h: h.distance_to(uav_x, uav_y))

    def get_containing_house(self, uav_x: float, uav_y: float, *, margin_cm: float = 0.0) -> Optional[House]:
        """
        Return the house whose radius currently contains the UAV position.

        If multiple houses overlap, prefer the one with the smallest center distance.
        """
        containing = [
            h for h in self._houses.values()
            if h.distance_to(uav_x, uav_y) <= float(h.radius_cm) + float(margin_cm)
        ]
        if not containing:
            return None
        return min(containing, key=lambda h: h.distance_to(uav_x, uav_y))

    def get_status_summary(self) -> dict:
        """
        Return a summary dict with per-status counts and the current
        target house id.

        Example::

            {
              "UNSEARCHED": 2,
              "IN_PROGRESS": 1,
              "EXPLORED": 0,
              "PERSON_FOUND": 0,
              "target_house_id": "house_B"
            }
        """
        counts: Dict[str, int] = {s.value: 0 for s in HouseStatus}
        for house in self._houses.values():
            counts[house.status.value] += 1
        counts["target_house_id"] = self._current_target_id
        return counts

    def to_dict(self) -> dict:
        """
        Full serialisable state suitable for embedding in API responses.
        """
        return {
            "world_bounds":      self._world_bounds,
            "overhead_map":      self._overhead_map,
            "current_target_id": self._current_target_id,
            "houses":            [h.to_dict() for h in self._houses.values()],
            "status_summary":    self.get_status_summary(),
        }

    def update_overhead_map(self, patch: dict) -> None:
        """Shallow-merge config into the persisted overhead-map settings."""
        if not isinstance(patch, dict):
            return
        self._overhead_map.update(patch)

    def set_overhead_calibration(
        self,
        *,
        anchors: List[dict],
        affine_world_to_image: List[List[float]],
        image_width: int,
        image_height: int,
        rmse_px: float,
    ) -> None:
        """Persist a solved world->image affine calibration for the overhead map."""
        self._overhead_map["calibration"] = {
            "anchors": list(anchors),
            "affine_world_to_image": affine_world_to_image,
            "image_width": int(image_width),
            "image_height": int(image_height),
            "rmse_px": float(rmse_px),
            "updated_at": time.time(),
        }

    def clear_overhead_calibration(self) -> None:
        """Remove the stored overhead-map calibration if present."""
        if "calibration" in self._overhead_map:
            self._overhead_map.pop("calibration", None)

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def set_target(self, house_id: str) -> bool:
        """
        Set *house_id* as the current mission target.

        If the house was UNSEARCHED, its status is promoted to IN_PROGRESS
        and search_start_time is recorded.  Returns True on success, False
        if the house_id is unknown.
        """
        house = self._houses.get(house_id)
        if house is None:
            logger.warning("HouseRegistry.set_target: unknown id '%s'", house_id)
            return False

        self._current_target_id = house_id

        if house.status == HouseStatus.UNSEARCHED:
            house.status = HouseStatus.IN_PROGRESS
            house.search_start_time = time.time()
            logger.info("HouseRegistry: started search of '%s'", house_id)

        return True

    def mark_explored(
        self,
        house_id: str,
        *,
        person_found: bool = False,
        person_location: Optional[dict] = None,
        notes: str = "",
    ) -> bool:
        """
        Mark a house as fully searched.

        Parameters
        ----------
        house_id:        The house to update.
        person_found:    If True, status becomes PERSON_FOUND; else EXPLORED.
        person_location: Optional {"x", "y", "z"} dict of the detected person.
        notes:           Free-form text to append / replace notes.

        Returns True on success, False if house_id is unknown.
        """
        house = self._houses.get(house_id)
        if house is None:
            logger.warning("HouseRegistry.mark_explored: unknown id '%s'", house_id)
            return False

        house.status = (
            HouseStatus.PERSON_FOUND if person_found else HouseStatus.EXPLORED
        )
        house.search_end_time = time.time()

        if person_found and person_location is not None:
            house.person_location = person_location

        if notes:
            house.notes = notes

        logger.info(
            "HouseRegistry: '%s' marked as %s", house_id, house.status.value
        )
        return True

    def reset_house(self, house_id: str) -> bool:
        """
        Reset a house back to UNSEARCHED, clearing all runtime state.

        Returns True on success, False if house_id is unknown.
        """
        house = self._houses.get(house_id)
        if house is None:
            logger.warning("HouseRegistry.reset_house: unknown id '%s'", house_id)
            return False

        house.status            = HouseStatus.UNSEARCHED
        house.search_start_time = None
        house.search_end_time   = None
        house.person_location   = None
        house.notes             = ""

        if self._current_target_id == house_id:
            self._current_target_id = ""

        logger.info("HouseRegistry: '%s' reset to UNSEARCHED", house_id)
        return True

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def _current_target_id(self) -> str:  # type: ignore[override]
        return self.__current_target_id

    @_current_target_id.setter
    def _current_target_id(self, value: str) -> None:
        self.__current_target_id = value

    # Provide direct attribute-style access that the spec calls for.
    # The property above already does this; we expose the raw string
    # via the public alias used in multi_house_mission.py.
    def _get_current_target_id(self) -> str:
        return self.__current_target_id


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os, tempfile

    logging.basicConfig(level=logging.DEBUG)

    cfg = {
        "world_bounds": {"min_x": 0, "min_y": 0, "max_x": 5000, "max_y": 3000},
        "houses": [
            {"id": "h1", "name": "House 1", "center_x": 1000.0, "center_y": 500.0,
             "center_z": 200.0, "approach_z": 600.0, "radius_cm": 700.0, "entry_yaw_hint": 0.0},
            {"id": "h2", "name": "House 2", "center_x": 3000.0, "center_y": 1500.0,
             "center_z": 200.0, "approach_z": 600.0, "radius_cm": 750.0, "entry_yaw_hint": 90.0},
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cfg, f)
        tmp_path = f.name

    try:
        reg = HouseRegistry(tmp_path)
        print("All houses:", [h.id for h in reg.get_all_houses()])

        reg.set_target("h1")
        print("Target:", reg.get_target_house().name)
        print("Summary:", reg.get_status_summary())

        nearest = reg.get_nearest_unsearched(900.0, 450.0)
        print("Nearest unsearched:", nearest.id if nearest else None)

        reg.mark_explored("h1", person_found=True,
                          person_location={"x": 1020.0, "y": 510.0, "z": 130.0})

        save_path = tmp_path.replace(".json", "_saved.json")
        reg.save_to_file(save_path)
        print("Saved to", save_path)

        reg2 = HouseRegistry(save_path)
        print("Re-loaded h1 status:", reg2.get_house("h1").status)
    finally:
        os.unlink(tmp_path)
