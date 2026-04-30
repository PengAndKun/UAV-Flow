from __future__ import annotations

import copy
import json
import math
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_ENTRY_SEARCH_MEMORY_PATH = os.path.join(
    PROJECT_ROOT,
    "phase2_multimodal_fusion_analysis",
    "entry_search_memory.json",
)
DEFAULT_HOUSES_CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "UAV-Flow-Eval",
    "houses_config.json",
)

DEFAULT_VERSION = "v1"
DEFAULT_RECENT_ACTIONS_LIMIT = 5
DEFAULT_RECENT_DECISIONS_LIMIT = 5
DEFAULT_TOP_CANDIDATES_LIMIT = 3
DEFAULT_EPISODIC_LIMIT = 64
DEFAULT_PERCEPTION_BUFFER_LIMIT = 5
DEFAULT_BBOX_HISTORY_LIMIT = 6
DEFAULT_SOURCE_FRAMES_LIMIT = 8
DEFAULT_PERIMETER_BIN_IDS = [
    "east",
    "north_east",
    "north",
    "north_west",
    "west",
    "south_west",
    "south",
    "south_east",
]
DEFAULT_FACE_IDS = ["east", "north", "west", "south"]
DEFAULT_FACE_SEGMENT_COUNT = 5
DEFAULT_FACE_OBSERVED_RANGE_HALF = 0.18
NO_ENTRY_MIN_VISITED_COVERAGE_RATIO = 0.75
NO_ENTRY_MIN_OBSERVED_COVERAGE_RATIO = 0.50
NO_ENTRY_MIN_FACE_OBSERVED_COVERAGE_RATIO = 0.45
NO_ENTRY_MIN_OBSERVED_FACE_COUNT = 3
NO_ENTRY_MIN_TOTAL_OBSERVATIONS = 8
RELIABLE_ENTRY_ASSOCIATION_THRESHOLD = 0.65
NON_ENTERABLE_MAX_OPENING_WIDTH_CM = 90.0
DEFAULT_SECTOR_IDS = [
    "front_left",
    "front_center",
    "front_right",
    "left_side",
    "right_side",
]


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _deep_merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge_dict(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def _as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def _merge_recent_unique_strings(existing: Any, incoming: Any, limit: int) -> List[str]:
    merged: List[str] = []
    for item in _as_list(existing) + _as_list(incoming):
        text = str(item or "").strip()
        if not text:
            continue
        if text in merged:
            merged.remove(text)
        merged.append(text)
    return merged[-max(1, int(limit)) :]


def _merge_recent_dict_history(existing: Any, incoming: Any, limit: int) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    keys: List[str] = []
    for item in _as_list(existing) + _as_list(incoming):
        if not isinstance(item, dict) or not item:
            continue
        payload = copy.deepcopy(item)
        key = str(payload.get("frame_id") or payload.get("source_frame") or "").strip()
        if not key:
            try:
                key = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            except Exception:
                key = str(payload)
        if key in keys:
            index = keys.index(key)
            del keys[index]
            del merged[index]
        keys.append(key)
        merged.append(payload)
    return merged[-max(1, int(limit)) :]


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def _safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _clamp_float(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _normalize_angle_360(angle_deg: float) -> float:
    return float(angle_deg) % 360.0


def _perimeter_bin_from_angle(angle_deg: float) -> str:
    normalized = _normalize_angle_360(float(angle_deg))
    index = int(((normalized + 22.5) % 360.0) // 45.0)
    return DEFAULT_PERIMETER_BIN_IDS[index]


def _default_perimeter_bin() -> Dict[str, Any]:
    return {
        "visited": False,
        "observed": False,
        "visit_count": 0,
        "visual_observation_count": 0,
        "last_visit_time": None,
        "last_viewpoint_angle_deg": None,
        "last_target_bearing_deg": None,
        "last_distance_cm": None,
    }


def _default_perimeter_coverage() -> Dict[str, Any]:
    return {
        "bin_count": len(DEFAULT_PERIMETER_BIN_IDS),
        "visited_bin_count": 0,
        "observed_bin_count": 0,
        "visited_coverage_ratio": 0.0,
        "observed_coverage_ratio": 0.0,
        "total_observations": 0,
        "visual_observation_count": 0,
        "closed_loop_score": 0.0,
        "last_viewpoint_angle_deg": None,
        "last_viewpoint_bin": "",
        "last_target_bearing_deg": None,
        "last_distance_cm": None,
        "bins": {bin_id: _default_perimeter_bin() for bin_id in DEFAULT_PERIMETER_BIN_IDS},
    }


def _default_face_segment() -> Dict[str, Any]:
    return {
        "visited": False,
        "observed": False,
        "visit_count": 0,
        "visual_observation_count": 0,
        "last_visit_time": None,
        "last_viewpoint_angle_deg": None,
        "last_target_bearing_deg": None,
        "last_distance_cm": None,
    }


def _default_face_record() -> Dict[str, Any]:
    return {
        "visited": False,
        "observed": False,
        "visit_count": 0,
        "visual_observation_count": 0,
        "visited_segment_count": 0,
        "observed_segment_count": 0,
        "coverage_ratio": 0.0,
        "observed_coverage_ratio": 0.0,
        "last_visit_time": None,
        "last_edge_t": None,
        "last_segment_index": None,
        "segments": [_default_face_segment() for _ in range(DEFAULT_FACE_SEGMENT_COUNT)],
    }


def _default_face_coverage() -> Dict[str, Any]:
    segment_count = max(1, int(DEFAULT_FACE_SEGMENT_COUNT))
    total_segment_count = len(DEFAULT_FACE_IDS) * segment_count
    return {
        "face_count": len(DEFAULT_FACE_IDS),
        "segment_count_per_face": segment_count,
        "total_segment_count": total_segment_count,
        "visited_face_count": 0,
        "observed_face_count": 0,
        "visited_segment_count": 0,
        "observed_segment_count": 0,
        "visited_coverage_ratio": 0.0,
        "observed_coverage_ratio": 0.0,
        "total_observations": 0,
        "visual_observation_count": 0,
        "last_face_id": "",
        "last_segment_index": None,
        "last_edge_t": None,
        "last_viewpoint_angle_deg": None,
        "last_distance_cm": None,
        "faces": {face_id: _default_face_record() for face_id in DEFAULT_FACE_IDS},
    }


def _default_search_completion_evidence() -> Dict[str, Any]:
    return {
        "no_entry_after_full_coverage": False,
        "full_coverage_ready": False,
        "perimeter_coverage_ready": False,
        "face_coverage_ready": False,
        "has_reliable_entry": False,
        "visited_coverage_ratio": 0.0,
        "observed_coverage_ratio": 0.0,
        "face_visited_coverage_ratio": 0.0,
        "face_observed_coverage_ratio": 0.0,
        "observed_face_count": 0,
        "total_observations": 0,
        "candidate_entry_count": 0,
        "rejected_entry_count": 0,
        "non_enterable_entry_count": 0,
        "unresolved_entry_count": 0,
        "approachable_entry_count": 0,
        "blocked_entry_count": 0,
        "reasons": [],
        "thresholds": {
            "min_visited_coverage_ratio": NO_ENTRY_MIN_VISITED_COVERAGE_RATIO,
            "min_observed_coverage_ratio": NO_ENTRY_MIN_OBSERVED_COVERAGE_RATIO,
            "min_face_observed_coverage_ratio": NO_ENTRY_MIN_FACE_OBSERVED_COVERAGE_RATIO,
            "min_observed_face_count": NO_ENTRY_MIN_OBSERVED_FACE_COUNT,
            "min_total_observations": NO_ENTRY_MIN_TOTAL_OBSERVATIONS,
            "reliable_entry_association_threshold": RELIABLE_ENTRY_ASSOCIATION_THRESHOLD,
            "non_enterable_max_opening_width_cm": NON_ENTERABLE_MAX_OPENING_WIDTH_CM,
        },
        "updated_at": _now_text(),
    }


def _default_working_memory() -> Dict[str, Any]:
    return {
        "target_house_id": "",
        "current_house_id": "",
        "last_best_entry_id": "",
        "perception_buffer": {
            "max_frames": DEFAULT_PERCEPTION_BUFFER_LIMIT,
            "frames": [],
        },
        "recent_actions": [],
        "recent_target_decisions": [],
        "top_candidates": [],
    }


def _default_semantic_memory() -> Dict[str, Any]:
    return {
        "entry_search_status": "not_started",
        "last_best_entry_id": "",
        "search_summary": {
            "observed_sector_count": 0,
            "candidate_entry_count": 0,
            "approachable_entry_count": 0,
            "blocked_entry_count": 0,
            "rejected_entry_count": 0,
        },
        "perimeter_coverage": _default_perimeter_coverage(),
        "face_coverage": _default_face_coverage(),
        "search_completion_evidence": _default_search_completion_evidence(),
        "searched_sectors": {
            sector_id: {
                "observed": False,
                "observation_count": 0,
                "last_visit_time": None,
                "best_entry_state": "",
                "best_target_conditioned_subgoal": "",
                "best_target_match_score": 0.0,
            }
            for sector_id in DEFAULT_SECTOR_IDS
        },
        "candidate_entries": [],
    }


def _default_house_registry_entry(
    house_id: str,
    *,
    house_name: str = "",
    house_status: str = "UNSEARCHED",
) -> Dict[str, Any]:
    return {
        "house_id": str(house_id),
        "house_name": str(house_name or house_id),
        "house_status": str(house_status or "UNSEARCHED"),
        "mission_status": "NOT_TARGET",
        "search_status": "UNSEARCHED",
        "entry_search_status": "not_started",
        "candidate_entry_count": 0,
        "best_entry_id": "",
        "best_target_match_score": 0.0,
        "searched": False,
        "updated_at": _now_text(),
    }


def _default_planner_context() -> Dict[str, Any]:
    return {
        "target_house_id": "",
        "current_house_id": "",
        "current_best_entry_id": "",
        "unsearched_houses": [],
        "decision_hint": "",
        "updated_at": _now_text(),
    }


def _default_house_memory(house_id: str, house_name: str = "", house_status: str = "UNSEARCHED") -> Dict[str, Any]:
    timestamp = _now_text()
    return {
        "house_id": str(house_id),
        "house_name": str(house_name or house_id),
        "house_status": str(house_status or "UNSEARCHED"),
        "mission_status": "NOT_TARGET",
        "search_status": "UNSEARCHED",
        "target_match_active": False,
        "created_at": timestamp,
        "updated_at": timestamp,
        "working_memory": _default_working_memory(),
        "episodic_memory": [],
        "semantic_memory": _default_semantic_memory(),
    }


class EntrySearchMemoryStore:
    def __init__(
        self,
        path: str = DEFAULT_ENTRY_SEARCH_MEMORY_PATH,
        *,
        recent_actions_limit: int = DEFAULT_RECENT_ACTIONS_LIMIT,
        recent_decisions_limit: int = DEFAULT_RECENT_DECISIONS_LIMIT,
        top_candidates_limit: int = DEFAULT_TOP_CANDIDATES_LIMIT,
        episodic_limit: int = DEFAULT_EPISODIC_LIMIT,
    ) -> None:
        self.path = os.path.abspath(path)
        self.recent_actions_limit = max(1, int(recent_actions_limit))
        self.recent_decisions_limit = max(1, int(recent_decisions_limit))
        self.top_candidates_limit = max(1, int(top_candidates_limit))
        self.episodic_limit = max(1, int(episodic_limit))
        self._houses_config_cache: Optional[Dict[str, Any]] = None
        self.data: Dict[str, Any] = {
            "version": DEFAULT_VERSION,
            "updated_at": _now_text(),
            "current_target_house_id": "",
            "house_registry": {},
            "planner_context": _default_planner_context(),
            "memories": {},
        }

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            self.data = {
                "version": DEFAULT_VERSION,
                "updated_at": _now_text(),
                "current_target_house_id": "",
                "house_registry": {},
                "planner_context": _default_planner_context(),
                "memories": {},
            }
            return self.data
        with open(self.path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.data = raw if isinstance(raw, dict) else {}
        self.data.setdefault("version", DEFAULT_VERSION)
        self.data.setdefault("updated_at", _now_text())
        self.data.setdefault("current_target_house_id", "")
        house_registry = self.data.get("house_registry")
        self.data["house_registry"] = house_registry if isinstance(house_registry, dict) else {}
        planner_context = self.data.get("planner_context")
        self.data["planner_context"] = planner_context if isinstance(planner_context, dict) else _default_planner_context()
        memories = self.data.get("memories")
        self.data["memories"] = memories if isinstance(memories, dict) else {}
        self._normalize_root()
        return self.data

    def save(self) -> str:
        self._normalize_root()
        self.data["updated_at"] = _now_text()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False)
        return self.path

    def to_dict(self) -> Dict[str, Any]:
        self._normalize_root()
        return copy.deepcopy(self.data)

    def set_current_target_house(self, house_id: str) -> None:
        self.data["current_target_house_id"] = str(house_id or "")
        target_house_id = str(house_id or "")
        for memory_house_id, memory in self.memories.items():
            if isinstance(memory, dict):
                memory["target_match_active"] = memory_house_id == target_house_id
                memory["updated_at"] = _now_text()
        self._normalize_root()

    @property
    def memories(self) -> Dict[str, Dict[str, Any]]:
        memories = self.data.setdefault("memories", {})
        return memories if isinstance(memories, dict) else {}

    def ensure_house(
        self,
        house_id: str,
        *,
        house_name: str = "",
        house_status: str = "UNSEARCHED",
    ) -> Dict[str, Any]:
        hid = str(house_id or "").strip()
        if not hid:
            raise ValueError("house_id is required")
        memory = self.memories.get(hid)
        if not isinstance(memory, dict):
            memory = _default_house_memory(hid, house_name=house_name, house_status=house_status)
            self.memories[hid] = memory
        else:
            memory.setdefault("house_id", hid)
            memory.setdefault("house_name", str(house_name or hid))
            memory.setdefault("house_status", str(house_status or "UNSEARCHED"))
            memory.setdefault("target_match_active", False)
            memory.setdefault("created_at", _now_text())
            memory.setdefault("updated_at", _now_text())
            if not isinstance(memory.get("working_memory"), dict):
                memory["working_memory"] = _default_working_memory()
            if not isinstance(memory.get("episodic_memory"), list):
                memory["episodic_memory"] = []
            if not isinstance(memory.get("semantic_memory"), dict):
                memory["semantic_memory"] = _default_semantic_memory()
        memory["house_name"] = str(house_name or memory.get("house_name") or hid)
        memory["house_status"] = str(house_status or memory.get("house_status") or "UNSEARCHED")
        self._normalize_house_memory(memory)
        return memory

    def ensure_from_houses_config(self, houses_config_path: str = DEFAULT_HOUSES_CONFIG_PATH) -> Dict[str, Dict[str, Any]]:
        path = os.path.abspath(houses_config_path)
        if not os.path.exists(path):
            return self.memories
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        houses = raw.get("houses", [])
        if isinstance(houses, list):
            for house in houses:
                if not isinstance(house, dict):
                    continue
                self.ensure_house(
                    str(house.get("id", "") or ""),
                    house_name=str(house.get("name", "") or ""),
                    house_status=str(house.get("status", "UNSEARCHED") or "UNSEARCHED"),
                )
        current_target_id = str(raw.get("current_target_id", "") or "")
        if current_target_id:
            self.set_current_target_house(current_target_id)
        self._normalize_root()
        return self.memories

    def get_house_memory(self, house_id: str, *, ensure: bool = False) -> Optional[Dict[str, Any]]:
        hid = str(house_id or "").strip()
        if not hid:
            return None
        memory = self.memories.get(hid)
        if isinstance(memory, dict):
            self._normalize_house_memory(memory)
            return memory
        if ensure:
            return self.ensure_house(hid)
        return None

    def _load_houses_config(self) -> Dict[str, Any]:
        if isinstance(self._houses_config_cache, dict):
            return self._houses_config_cache
        if not os.path.exists(DEFAULT_HOUSES_CONFIG_PATH):
            self._houses_config_cache = {}
            return self._houses_config_cache
        try:
            with open(DEFAULT_HOUSES_CONFIG_PATH, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
            self._houses_config_cache = payload if isinstance(payload, dict) else {}
        except Exception:
            self._houses_config_cache = {}
        return self._houses_config_cache

    def _house_config_for_id(self, house_id: str) -> Dict[str, Any]:
        config = self._load_houses_config()
        houses = config.get("houses", []) if isinstance(config.get("houses"), list) else []
        target_id = str(house_id or "").strip()
        for house in houses:
            if isinstance(house, dict) and str(house.get("id", "") or "").strip() == target_id:
                return house
        return {}

    def _image_to_world_from_houses_config(self, image_x: float, image_y: float) -> Optional[Dict[str, float]]:
        config = self._load_houses_config()
        overhead = config.get("overhead_map", {}) if isinstance(config.get("overhead_map"), dict) else {}
        calibration = overhead.get("calibration", {}) if isinstance(overhead.get("calibration"), dict) else {}
        affine = calibration.get("affine_world_to_image", [])
        if (
            not isinstance(affine, list)
            or len(affine) < 2
            or not isinstance(affine[0], list)
            or not isinstance(affine[1], list)
            or len(affine[0]) < 3
            or len(affine[1]) < 3
        ):
            return None
        a = _safe_float(affine[0][0], None)
        b = _safe_float(affine[0][1], None)
        c = _safe_float(affine[0][2], None)
        d = _safe_float(affine[1][0], None)
        e = _safe_float(affine[1][1], None)
        f = _safe_float(affine[1][2], None)
        if None in (a, b, c, d, e, f):
            return None
        det = float(a) * float(e) - float(b) * float(d)
        if abs(det) < 1e-8:
            return None
        ix = float(image_x) - float(c)
        iy = float(image_y) - float(f)
        world_x = (float(e) * ix - float(b) * iy) / det
        world_y = (-float(d) * ix + float(a) * iy) / det
        return {"x": float(world_x), "y": float(world_y)}

    def _house_world_bbox_for_face_memory(self, house_id: str, target_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        house = self._house_config_for_id(house_id)
        bbox = house.get("map_bbox_image", {}) if isinstance(house.get("map_bbox_image"), dict) else {}
        points: List[Dict[str, float]] = []
        if bbox:
            x1 = _safe_float(bbox.get("x1"), None)
            y1 = _safe_float(bbox.get("y1"), None)
            x2 = _safe_float(bbox.get("x2"), None)
            y2 = _safe_float(bbox.get("y2"), None)
            if None not in (x1, y1, x2, y2):
                for px, py in ((x1, y1), (x1, y2), (x2, y1), (x2, y2)):
                    world = self._image_to_world_from_houses_config(float(px), float(py))
                    if world:
                        points.append(world)
        if points:
            xs = [float(point["x"]) for point in points]
            ys = [float(point["y"]) for point in points]
            return {
                "min_x": min(xs),
                "max_x": max(xs),
                "min_y": min(ys),
                "max_y": max(ys),
                "center_x": (min(xs) + max(xs)) / 2.0,
                "center_y": (min(ys) + max(ys)) / 2.0,
                "source": "map_bbox_image_affine",
            }

        target_context = target_context if isinstance(target_context, dict) else {}
        center = target_context.get("target_house_center_world", {})
        center = center if isinstance(center, dict) else {}
        center_x = _safe_float(house.get("center_x"), None)
        center_y = _safe_float(house.get("center_y"), None)
        if center_x is None:
            center_x = _safe_float(center.get("x"), None)
        if center_y is None:
            center_y = _safe_float(center.get("y"), None)
        radius = _safe_float(house.get("radius_cm"), None)
        if radius is None:
            radius = _safe_float(target_context.get("target_house_radius_cm"), None)
        if None in (center_x, center_y, radius):
            return None
        half = max(150.0, float(radius) * 0.45)
        return {
            "min_x": float(center_x) - half,
            "max_x": float(center_x) + half,
            "min_y": float(center_y) - half,
            "max_y": float(center_y) + half,
            "center_x": float(center_x),
            "center_y": float(center_y),
            "source": "center_radius_fallback",
        }

    def _face_segment_from_pose(
        self,
        house_id: str,
        target_context: Dict[str, Any],
        *,
        segment_count: int = DEFAULT_FACE_SEGMENT_COUNT,
    ) -> Optional[Dict[str, Any]]:
        target_context = target_context if isinstance(target_context, dict) else {}
        pose = target_context.get("uav_pose_world", {})
        pose = pose if isinstance(pose, dict) else {}
        uav_x = _safe_float(pose.get("x"), None)
        uav_y = _safe_float(pose.get("y"), None)
        bbox = self._house_world_bbox_for_face_memory(house_id, target_context)
        if bbox is None or None in (uav_x, uav_y):
            return None
        min_x = float(bbox["min_x"])
        max_x = float(bbox["max_x"])
        min_y = float(bbox["min_y"])
        max_y = float(bbox["max_y"])
        center_x = float(bbox["center_x"])
        center_y = float(bbox["center_y"])
        half_x = max(1.0, (max_x - min_x) / 2.0)
        half_y = max(1.0, (max_y - min_y) / 2.0)
        dx = float(uav_x) - center_x
        dy = float(uav_y) - center_y
        if abs(dx) / half_x >= abs(dy) / half_y:
            face_id = "east" if dx >= 0.0 else "west"
            edge_t = (float(uav_y) - min_y) / max(1.0, max_y - min_y)
        else:
            face_id = "north" if dy >= 0.0 else "south"
            edge_t = (float(uav_x) - min_x) / max(1.0, max_x - min_x)
        edge_t = _clamp_float(edge_t, 0.0, 1.0)
        count = max(1, int(segment_count))
        segment_index = min(count - 1, max(0, int(edge_t * float(count))))
        return {
            "face_id": face_id,
            "edge_t": round(float(edge_t), 4),
            "segment_index": int(segment_index),
            "bbox_world": bbox,
            "uav_pose_world": {"x": float(uav_x), "y": float(uav_y)},
        }

    @property
    def house_registry(self) -> Dict[str, Dict[str, Any]]:
        registry = self.data.setdefault("house_registry", {})
        return registry if isinstance(registry, dict) else {}

    def set_planner_context(self, patch: Dict[str, Any], *, deep_merge: bool = True) -> Dict[str, Any]:
        planner_context = self.data.setdefault("planner_context", _default_planner_context())
        if not isinstance(planner_context, dict):
            planner_context = _default_planner_context()
            self.data["planner_context"] = planner_context
        if deep_merge:
            _deep_merge_dict(planner_context, patch)
        else:
            for key, value in patch.items():
                planner_context[key] = copy.deepcopy(value)
        planner_context["updated_at"] = _now_text()
        self._normalize_root()
        return planner_context

    def update_working_memory(self, house_id: str, patch: Dict[str, Any], *, deep_merge: bool = True) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        working_memory = memory["working_memory"]
        if deep_merge:
            _deep_merge_dict(working_memory, patch)
        else:
            for key, value in patch.items():
                working_memory[key] = copy.deepcopy(value)
        self._normalize_working_memory(memory)
        memory["updated_at"] = _now_text()
        self._normalize_root()
        return memory

    def update_semantic_memory(self, house_id: str, patch: Dict[str, Any], *, deep_merge: bool = True) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        semantic_memory = memory["semantic_memory"]
        if deep_merge:
            _deep_merge_dict(semantic_memory, patch)
        else:
            for key, value in patch.items():
                semantic_memory[key] = copy.deepcopy(value)
        self._normalize_semantic_memory(memory)
        memory["updated_at"] = _now_text()
        self._normalize_root()
        return memory

    def append_perception_frame(self, house_id: str, frame_payload: Dict[str, Any]) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        working_memory = memory["working_memory"]
        perception_buffer = working_memory.setdefault(
            "perception_buffer",
            {"max_frames": DEFAULT_PERCEPTION_BUFFER_LIMIT, "frames": []},
        )
        if not isinstance(perception_buffer, dict):
            perception_buffer = {"max_frames": DEFAULT_PERCEPTION_BUFFER_LIMIT, "frames": []}
            working_memory["perception_buffer"] = perception_buffer
        frames = perception_buffer.get("frames", [])
        if not isinstance(frames, list):
            frames = []
            perception_buffer["frames"] = frames
        payload = copy.deepcopy(frame_payload)
        payload.setdefault("timestamp", _now_text())
        payload.setdefault("house_id", str(house_id))
        frames.append(payload)
        max_frames = max(1, int(perception_buffer.get("max_frames", DEFAULT_PERCEPTION_BUFFER_LIMIT) or DEFAULT_PERCEPTION_BUFFER_LIMIT))
        if len(frames) > max_frames:
            del frames[:-max_frames]
        perception_buffer["max_frames"] = max_frames
        memory["updated_at"] = _now_text()
        self._normalize_root()
        return memory

    def append_recent_action(self, house_id: str, action_name: str) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        recent_actions = memory["working_memory"].setdefault("recent_actions", [])
        if not isinstance(recent_actions, list):
            recent_actions = []
            memory["working_memory"]["recent_actions"] = recent_actions
        recent_actions.append(str(action_name or ""))
        if len(recent_actions) > self.recent_actions_limit:
            del recent_actions[:-self.recent_actions_limit]
        memory["updated_at"] = _now_text()
        self._normalize_root()
        return memory

    def append_recent_target_decision(self, house_id: str, decision: Dict[str, Any]) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        recent_decisions = memory["working_memory"].setdefault("recent_target_decisions", [])
        if not isinstance(recent_decisions, list):
            recent_decisions = []
            memory["working_memory"]["recent_target_decisions"] = recent_decisions
        payload = copy.deepcopy(decision)
        payload.setdefault("timestamp", _now_text())
        recent_decisions.append(payload)
        if len(recent_decisions) > self.recent_decisions_limit:
            del recent_decisions[:-self.recent_decisions_limit]
        memory["updated_at"] = _now_text()
        self._normalize_root()
        return memory

    def set_top_candidates(self, house_id: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        trimmed = [copy.deepcopy(candidate) for candidate in list(candidates or [])[: self.top_candidates_limit]]
        memory["working_memory"]["top_candidates"] = trimmed
        if trimmed:
            best_candidate_id = str(trimmed[0].get("candidate_id", "") or "")
            memory["working_memory"]["last_best_entry_id"] = best_candidate_id
            memory["semantic_memory"]["last_best_entry_id"] = best_candidate_id
        self._normalize_working_memory(memory)
        self._normalize_semantic_memory(memory)
        memory["updated_at"] = _now_text()
        self._normalize_root()
        return memory

    def set_entry_search_status(self, house_id: str, status: str) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        memory["semantic_memory"]["entry_search_status"] = str(status or "not_started")
        self._normalize_semantic_memory(memory)
        memory["updated_at"] = _now_text()
        self._normalize_root()
        return memory

    def update_sector(
        self,
        house_id: str,
        sector_id: str,
        *,
        observed: bool = True,
        best_entry_state: Optional[str] = None,
        best_target_conditioned_subgoal: Optional[str] = None,
        best_target_match_score: Optional[float] = None,
        visit_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        searched_sectors = memory["semantic_memory"].setdefault("searched_sectors", {})
        if not isinstance(searched_sectors, dict):
            searched_sectors = {}
            memory["semantic_memory"]["searched_sectors"] = searched_sectors
        sector_key = str(sector_id or "").strip() or "unknown_sector"
        sector = searched_sectors.get(sector_key)
        if not isinstance(sector, dict):
            sector = {
                "observed": False,
                "observation_count": 0,
                "last_visit_time": None,
                "best_entry_state": "",
                "best_target_conditioned_subgoal": "",
                "best_target_match_score": 0.0,
            }
            searched_sectors[sector_key] = sector
        sector["observed"] = bool(observed)
        sector["observation_count"] = int(sector.get("observation_count", 0) or 0) + 1
        sector["last_visit_time"] = visit_time if visit_time is not None else datetime.now().timestamp()
        if best_entry_state is not None:
            sector["best_entry_state"] = str(best_entry_state or "")
        if best_target_conditioned_subgoal is not None:
            sector["best_target_conditioned_subgoal"] = str(best_target_conditioned_subgoal or "")
        if best_target_match_score is not None:
            sector["best_target_match_score"] = float(best_target_match_score)
        self._normalize_semantic_memory(memory)
        memory["updated_at"] = _now_text()
        self._normalize_root()
        return memory

    def _update_face_coverage(
        self,
        semantic_memory: Dict[str, Any],
        house_id: str,
        target_context: Dict[str, Any],
        *,
        viewpoint_angle: float,
        target_bearing: Optional[float],
        distance_cm: Optional[float],
        visually_observed: bool,
        visit_time: Optional[float],
    ) -> None:
        coverage = semantic_memory.setdefault("face_coverage", _default_face_coverage())
        if not isinstance(coverage, dict):
            coverage = _default_face_coverage()
            semantic_memory["face_coverage"] = coverage
        segment_count = max(1, int(coverage.get("segment_count_per_face", DEFAULT_FACE_SEGMENT_COUNT) or DEFAULT_FACE_SEGMENT_COUNT))
        face_hit = self._face_segment_from_pose(house_id, target_context, segment_count=segment_count)
        if not face_hit:
            return
        faces = coverage.setdefault("faces", {})
        if not isinstance(faces, dict):
            faces = {}
            coverage["faces"] = faces
        face_id = str(face_hit.get("face_id") or "")
        if face_id not in DEFAULT_FACE_IDS:
            return
        face = faces.get(face_id)
        if not isinstance(face, dict):
            face = _default_face_record()
            faces[face_id] = face
        segments = face.get("segments", [])
        if not isinstance(segments, list):
            segments = []
        while len(segments) < segment_count:
            segments.append(_default_face_segment())
        if len(segments) > segment_count:
            del segments[segment_count:]
        face["segments"] = segments

        edge_t = float(face_hit.get("edge_t", 0.0) or 0.0)
        start_t = _clamp_float(edge_t - DEFAULT_FACE_OBSERVED_RANGE_HALF, 0.0, 1.0)
        end_t = _clamp_float(edge_t + DEFAULT_FACE_OBSERVED_RANGE_HALF, 0.0, 1.0)
        touched_indices: List[int] = []
        timestamp = visit_time if visit_time is not None else datetime.now().timestamp()
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                segment = _default_face_segment()
                segments[index] = segment
            seg_start = float(index) / float(segment_count)
            seg_end = float(index + 1) / float(segment_count)
            if seg_end < start_t or seg_start > end_t:
                continue
            touched_indices.append(index)
            segment["visited"] = True
            segment["visit_count"] = int(segment.get("visit_count", 0) or 0) + 1
            if visually_observed:
                segment["observed"] = True
                segment["visual_observation_count"] = int(segment.get("visual_observation_count", 0) or 0) + 1
            segment["last_visit_time"] = timestamp
            segment["last_viewpoint_angle_deg"] = round(float(viewpoint_angle), 3)
            segment["last_target_bearing_deg"] = None if target_bearing is None else round(float(target_bearing), 3)
            segment["last_distance_cm"] = None if distance_cm is None else round(float(distance_cm), 3)

        face["visited"] = True
        face["visit_count"] = int(face.get("visit_count", 0) or 0) + 1
        if visually_observed:
            face["observed"] = True
            face["visual_observation_count"] = int(face.get("visual_observation_count", 0) or 0) + 1
        face["last_visit_time"] = timestamp
        face["last_edge_t"] = round(edge_t, 4)
        face["last_segment_index"] = int(face_hit.get("segment_index", 0) or 0)

        coverage["last_face_id"] = face_id
        coverage["last_segment_index"] = int(face_hit.get("segment_index", 0) or 0)
        coverage["last_edge_t"] = round(edge_t, 4)
        coverage["last_viewpoint_angle_deg"] = round(float(viewpoint_angle), 3)
        coverage["last_distance_cm"] = None if distance_cm is None else round(float(distance_cm), 3)
        coverage["last_bbox_world"] = face_hit.get("bbox_world", {})
        coverage["last_touched_segments"] = touched_indices

    def update_perimeter_coverage(
        self,
        house_id: str,
        target_context: Dict[str, Any],
        *,
        visit_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        semantic_memory = memory["semantic_memory"]
        coverage = semantic_memory.setdefault("perimeter_coverage", _default_perimeter_coverage())
        if not isinstance(coverage, dict):
            coverage = _default_perimeter_coverage()
            semantic_memory["perimeter_coverage"] = coverage
        bins = coverage.setdefault("bins", {})
        if not isinstance(bins, dict):
            bins = {}
            coverage["bins"] = bins
        for bin_id in DEFAULT_PERIMETER_BIN_IDS:
            bin_payload = bins.get(bin_id)
            if not isinstance(bin_payload, dict):
                bins[bin_id] = _default_perimeter_bin()

        target_context = target_context if isinstance(target_context, dict) else {}
        center = target_context.get("target_house_center_world", {})
        pose = target_context.get("uav_pose_world", {})
        center = center if isinstance(center, dict) else {}
        pose = pose if isinstance(pose, dict) else {}
        house_x = _safe_float(center.get("x"), None)
        house_y = _safe_float(center.get("y"), None)
        uav_x = _safe_float(pose.get("x"), None)
        uav_y = _safe_float(pose.get("y"), None)
        if None in (house_x, house_y, uav_x, uav_y):
            self._normalize_semantic_memory(memory)
            memory["updated_at"] = _now_text()
            self._normalize_root()
            return memory

        viewpoint_angle = _normalize_angle_360(math.degrees(math.atan2(float(uav_y) - float(house_y), float(uav_x) - float(house_x))))
        bin_id = _perimeter_bin_from_angle(viewpoint_angle)
        target_bearing = _safe_float(target_context.get("target_house_bearing_deg"), None)
        distance_cm = _safe_float(target_context.get("target_house_distance_cm"), None)
        if distance_cm is None:
            distance_cm = float(math.hypot(float(uav_x) - float(house_x), float(uav_y) - float(house_y)))
        visually_observed = bool(target_context.get("target_house_in_fov", False))
        if target_bearing is not None and abs(float(target_bearing)) <= 75.0:
            visually_observed = True

        bin_payload = bins.setdefault(bin_id, _default_perimeter_bin())
        bin_payload["visited"] = True
        bin_payload["visit_count"] = int(bin_payload.get("visit_count", 0) or 0) + 1
        if visually_observed:
            bin_payload["observed"] = True
            bin_payload["visual_observation_count"] = int(bin_payload.get("visual_observation_count", 0) or 0) + 1
        bin_payload["last_visit_time"] = visit_time if visit_time is not None else datetime.now().timestamp()
        bin_payload["last_viewpoint_angle_deg"] = round(float(viewpoint_angle), 3)
        bin_payload["last_target_bearing_deg"] = None if target_bearing is None else round(float(target_bearing), 3)
        bin_payload["last_distance_cm"] = None if distance_cm is None else round(float(distance_cm), 3)

        coverage["last_viewpoint_angle_deg"] = round(float(viewpoint_angle), 3)
        coverage["last_viewpoint_bin"] = bin_id
        coverage["last_target_bearing_deg"] = None if target_bearing is None else round(float(target_bearing), 3)
        coverage["last_distance_cm"] = None if distance_cm is None else round(float(distance_cm), 3)
        coverage["total_observations"] = int(coverage.get("total_observations", 0) or 0) + 1
        if visually_observed:
            coverage["visual_observation_count"] = int(coverage.get("visual_observation_count", 0) or 0) + 1

        self._update_face_coverage(
            semantic_memory,
            house_id,
            target_context,
            viewpoint_angle=float(viewpoint_angle),
            target_bearing=target_bearing,
            distance_cm=distance_cm,
            visually_observed=visually_observed,
            visit_time=visit_time,
        )
        self._normalize_semantic_memory(memory)
        memory["updated_at"] = _now_text()
        self._normalize_root()
        return memory

    def upsert_candidate_entry(self, house_id: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        entries = memory["semantic_memory"].setdefault("candidate_entries", [])
        if not isinstance(entries, list):
            entries = []
            memory["semantic_memory"]["candidate_entries"] = entries
        candidate_id = str(candidate.get("entry_id") or candidate.get("candidate_id") or "").strip()
        if not candidate_id:
            raise ValueError("entry_id/candidate_id is required for candidate entry upsert")
        payload = copy.deepcopy(candidate)
        payload["entry_id"] = candidate_id
        payload["candidate_id"] = candidate_id
        payload.setdefault("entry_type", str(payload.get("semantic_class") or payload.get("class_name") or ""))
        payload.setdefault("semantic_class", str(payload.get("entry_type") or payload.get("class_name") or ""))
        payload.setdefault("source_frames", [])
        payload.setdefault("bbox_history", [])
        payload.setdefault("first_seen_time", _now_text())
        payload["last_seen_time"] = str(payload.get("last_seen_time") or _now_text())
        payload.setdefault("observation_count", 0)
        payload.setdefault("sector", "")
        payload.setdefault("associated_house_id", str(house_id))
        payload.setdefault("target_match_score", 0.0)
        payload.setdefault("association_confidence", float(payload.get("candidate_total_score", payload.get("target_match_score", 0.0)) or 0.0))
        payload.setdefault(
            "association_evidence",
            {
                "distance_score": 0.0,
                "view_consistency_score": 0.0,
                "appearance_score": 0.0,
                "language_score": 0.0,
            },
        )
        payload.setdefault("entry_state", str(payload.get("status") or "unverified"))
        payload.setdefault("status", str(payload.get("entry_state") or "unverified"))
        payload.setdefault("is_best_candidate", False)
        payload.setdefault("is_searched", False)
        payload.setdefault("is_entered", False)
        payload.setdefault("world_position", {"x": None, "y": None, "z": None, "source": ""})
        payload.setdefault("attempt_count", 0)
        incoming_source_frames = payload.pop("source_frames", [])
        incoming_bbox_history = payload.pop("bbox_history", [])
        requested_best = _truthy(payload.get("is_best_candidate", False))
        existing_index = next(
            (
                index
                for index, item in enumerate(entries)
                if isinstance(item, dict)
                and str(item.get("entry_id") or item.get("candidate_id") or "") == candidate_id
            ),
            None,
        )
        if existing_index is None:
            payload["observation_count"] = max(1, int(payload.get("observation_count", 0) or 0))
            payload["source_frames"] = _merge_recent_unique_strings(
                [],
                incoming_source_frames,
                DEFAULT_SOURCE_FRAMES_LIMIT,
            )
            payload["bbox_history"] = _merge_recent_dict_history(
                [],
                incoming_bbox_history,
                DEFAULT_BBOX_HISTORY_LIMIT,
            )
            entries.append(payload)
        else:
            merged = entries[existing_index]
            if not isinstance(merged, dict):
                merged = {}
            existing_observation_count = int(merged.get("observation_count", 0) or 0)
            existing_first_seen_time = str(merged.get("first_seen_time") or "")
            payload_observation_count = int(payload.get("observation_count", 0) or 0)
            observation_increment = int(payload.pop("observation_increment", 1) or 1)
            _deep_merge_dict(merged, payload)
            merged["entry_id"] = candidate_id
            merged["candidate_id"] = candidate_id
            merged["first_seen_time"] = str(existing_first_seen_time or payload.get("first_seen_time") or _now_text())
            merged["last_seen_time"] = str(payload.get("last_seen_time") or _now_text())
            merged["observation_count"] = max(existing_observation_count + max(1, observation_increment), payload_observation_count, 1)
            merged["entry_state"] = str(merged.get("entry_state") or merged.get("status") or "unverified")
            merged["status"] = str(merged.get("status") or merged.get("entry_state") or "unverified")
            merged["associated_house_id"] = str(merged.get("associated_house_id") or house_id)
            merged["source_frames"] = _merge_recent_unique_strings(
                merged.get("source_frames", []),
                incoming_source_frames,
                DEFAULT_SOURCE_FRAMES_LIMIT,
            )
            merged["bbox_history"] = _merge_recent_dict_history(
                merged.get("bbox_history", []),
                incoming_bbox_history,
                DEFAULT_BBOX_HISTORY_LIMIT,
            )
            if bool(merged.get("is_entered", False)):
                merged["is_searched"] = True
            entries[existing_index] = merged
        if requested_best:
            memory["working_memory"]["last_best_entry_id"] = candidate_id
            memory["semantic_memory"]["last_best_entry_id"] = candidate_id
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                entry_id = str(entry.get("entry_id") or entry.get("candidate_id") or "").strip()
                entry["is_best_candidate"] = entry_id == candidate_id
        self._normalize_semantic_memory(memory)
        memory["updated_at"] = _now_text()
        self._normalize_root()
        return memory

    def append_episodic_snapshot(self, house_id: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        episodic_memory = memory.setdefault("episodic_memory", [])
        if not isinstance(episodic_memory, list):
            episodic_memory = []
            memory["episodic_memory"] = episodic_memory
        payload = copy.deepcopy(snapshot)
        payload.setdefault("house_id", str(house_id))
        payload.setdefault("timestamp", datetime.now().timestamp())
        snapshot_id = str(payload.get("snapshot_id", "") or "").strip()
        if snapshot_id:
            existing_index = next(
                (index for index, item in enumerate(episodic_memory) if isinstance(item, dict) and str(item.get("snapshot_id", "") or "") == snapshot_id),
                None,
            )
            if existing_index is None:
                episodic_memory.append(payload)
            else:
                merged = episodic_memory[existing_index]
                if not isinstance(merged, dict):
                    merged = {}
                _deep_merge_dict(merged, payload)
                episodic_memory[existing_index] = merged
        else:
            episodic_memory.append(payload)
        if len(episodic_memory) > self.episodic_limit:
            del episodic_memory[:-self.episodic_limit]
        memory["updated_at"] = _now_text()
        self._normalize_root()
        return memory

    def append_episodic_event(self, house_id: str, event_payload: Dict[str, Any]) -> Dict[str, Any]:
        payload = copy.deepcopy(event_payload)
        payload.setdefault("event_id", f"evt_{int(datetime.now().timestamp() * 1000)}")
        payload.setdefault("time", _now_text())
        payload.setdefault("house_id", str(house_id))
        payload.setdefault("timestamp", datetime.now().timestamp())
        return self.append_episodic_snapshot(house_id, payload)

    def _derive_search_status(self, memory: Dict[str, Any]) -> str:
        semantic_memory = memory.get("semantic_memory", {}) if isinstance(memory.get("semantic_memory"), dict) else {}
        working_memory = memory.get("working_memory", {}) if isinstance(memory.get("working_memory"), dict) else {}
        search_summary = semantic_memory.get("search_summary", {}) if isinstance(semantic_memory.get("search_summary"), dict) else {}
        candidate_entries = semantic_memory.get("candidate_entries", []) if isinstance(semantic_memory.get("candidate_entries"), list) else []
        entry_search_status = str(semantic_memory.get("entry_search_status", "") or "").strip()
        latest_decision = working_memory.get("recent_target_decisions", [])
        latest_subgoal = ""
        if isinstance(latest_decision, list) and latest_decision:
            latest = latest_decision[-1]
            if isinstance(latest, dict):
                latest_subgoal = str(latest.get("target_conditioned_subgoal") or latest.get("subgoal") or "").strip()

        if any(bool(entry.get("is_entered", False)) for entry in candidate_entries if isinstance(entry, dict)):
            return "ENTERED"
        if entry_search_status in {"entered_house"}:
            return "ENTERED"
        if entry_search_status == "no_entry_found_after_full_coverage":
            return "NO_ENTRY_FOUND"
        if latest_subgoal in {"approach_target_entry", "align_target_entry", "cross_target_entry"}:
            return "APPROACHING_ENTRY"
        if any(float(entry.get("association_confidence", 0.0) or 0.0) >= 0.7 for entry in candidate_entries if isinstance(entry, dict)):
            return "ENTRY_ASSOCIATED"
        if any(isinstance(entry, dict) for entry in candidate_entries):
            return "ENTRY_CANDIDATE_FOUND"
        if int(search_summary.get("observed_sector_count", 0) or 0) > 0:
            return "OBSERVING"
        if str(memory.get("house_status", "") or "").strip() in {"EXPLORED", "PERSON_FOUND"}:
            return "SEARCHED"
        return "UNSEARCHED"

    def _derive_mission_status(self, memory: Dict[str, Any]) -> str:
        house_status = str(memory.get("house_status", "") or "").strip()
        working_memory = memory.get("working_memory", {}) if isinstance(memory.get("working_memory"), dict) else {}
        observed_sector_count = (
            memory.get("semantic_memory", {}).get("search_summary", {}).get("observed_sector_count", 0)
            if isinstance(memory.get("semantic_memory"), dict)
            else 0
        )
        if house_status in {"EXPLORED", "PERSON_FOUND"}:
            return "COMPLETED"
        entry_search_status = (
            memory.get("semantic_memory", {}).get("entry_search_status", "")
            if isinstance(memory.get("semantic_memory"), dict)
            else ""
        )
        if str(entry_search_status or "").strip() == "no_entry_found_after_full_coverage":
            return "COMPLETED"
        if bool(memory.get("target_match_active", False)):
            return "TARGET_ACTIVE"
        if str(working_memory.get("current_house_id", "") or "").strip() == str(memory.get("house_id", "") or "").strip():
            return "IN_PROGRESS"
        if int(observed_sector_count or 0) > 0:
            return "IN_PROGRESS"
        return "NOT_TARGET"

    def _normalize_root(self) -> None:
        self.data.setdefault("house_registry", {})
        if not isinstance(self.data.get("house_registry"), dict):
            self.data["house_registry"] = {}
        if not isinstance(self.data.get("planner_context"), dict):
            self.data["planner_context"] = _default_planner_context()
        for house_id, memory in self.memories.items():
            if isinstance(memory, dict):
                self._normalize_house_memory(memory)
                self._refresh_house_registry_entry(house_id, memory)
        self._refresh_planner_context()

    def _refresh_house_registry_entry(self, house_id: str, memory: Dict[str, Any]) -> None:
        entry = self.house_registry.get(house_id)
        if not isinstance(entry, dict):
            entry = _default_house_registry_entry(
                house_id,
                house_name=str(memory.get("house_name", "") or house_id),
                house_status=str(memory.get("house_status", "UNSEARCHED") or "UNSEARCHED"),
            )
            self.house_registry[house_id] = entry
        search_status = self._derive_search_status(memory)
        mission_status = self._derive_mission_status(memory)
        semantic_memory = memory.get("semantic_memory", {}) if isinstance(memory.get("semantic_memory"), dict) else {}
        candidate_entries = semantic_memory.get("candidate_entries", []) if isinstance(semantic_memory.get("candidate_entries"), list) else []
        coverage = semantic_memory.get("perimeter_coverage", {}) if isinstance(semantic_memory.get("perimeter_coverage"), dict) else {}
        face_coverage = semantic_memory.get("face_coverage", {}) if isinstance(semantic_memory.get("face_coverage"), dict) else {}
        completion_evidence = (
            semantic_memory.get("search_completion_evidence", {})
            if isinstance(semantic_memory.get("search_completion_evidence"), dict)
            else {}
        )
        best_entry_id = str(semantic_memory.get("last_best_entry_id") or memory.get("working_memory", {}).get("last_best_entry_id") or "").strip()
        best_target_match_score = 0.0
        for candidate in candidate_entries:
            if not isinstance(candidate, dict):
                continue
            if best_entry_id and str(candidate.get("entry_id") or candidate.get("candidate_id") or "") == best_entry_id:
                best_target_match_score = float(candidate.get("target_match_score", 0.0) or 0.0)
                break
        if not best_target_match_score:
            for candidate in candidate_entries:
                if isinstance(candidate, dict):
                    best_target_match_score = max(best_target_match_score, float(candidate.get("target_match_score", 0.0) or 0.0))
        entry.update(
            {
                "house_id": str(house_id),
                "house_name": str(memory.get("house_name", "") or house_id),
                "house_status": str(memory.get("house_status", "UNSEARCHED") or "UNSEARCHED"),
                "mission_status": mission_status,
                "search_status": search_status,
                "entry_search_status": str(semantic_memory.get("entry_search_status", "") or "not_started"),
                "candidate_entry_count": len(candidate_entries),
                "best_entry_id": best_entry_id,
                "best_target_match_score": round(float(best_target_match_score), 4),
                "searched": bool(search_status in {"SEARCHED", "ENTERED", "NO_ENTRY_FOUND"}) or str(memory.get("house_status", "") or "").strip() in {"EXPLORED", "PERSON_FOUND"},
                "visited_coverage_ratio": round(float(coverage.get("visited_coverage_ratio", 0.0) or 0.0), 4),
                "observed_coverage_ratio": round(float(coverage.get("observed_coverage_ratio", 0.0) or 0.0), 4),
                "face_observed_coverage_ratio": round(float(face_coverage.get("observed_coverage_ratio", 0.0) or 0.0), 4),
                "observed_face_count": int(face_coverage.get("observed_face_count", 0) or 0),
                "no_entry_after_full_coverage": bool(completion_evidence.get("no_entry_after_full_coverage", False)),
                "updated_at": _now_text(),
            }
        )
        memory["mission_status"] = mission_status
        memory["search_status"] = search_status

    def _refresh_planner_context(self) -> None:
        planner_context = self.data.setdefault("planner_context", _default_planner_context())
        target_house_id = str(self.data.get("current_target_house_id", "") or "").strip()
        target_memory = self.memories.get(target_house_id, {}) if target_house_id else {}
        target_working = target_memory.get("working_memory", {}) if isinstance(target_memory, dict) and isinstance(target_memory.get("working_memory"), dict) else {}
        target_recent_decisions = target_working.get("recent_target_decisions", []) if isinstance(target_working.get("recent_target_decisions"), list) else []
        decision_hint = ""
        if target_recent_decisions:
            latest = target_recent_decisions[-1]
            if isinstance(latest, dict):
                decision_hint = str(latest.get("target_conditioned_subgoal") or latest.get("subgoal") or latest.get("decision_hint") or "").strip()
        planner_context.update(
            {
                "target_house_id": target_house_id,
                "current_house_id": str(target_working.get("current_house_id", "") or ""),
                "current_best_entry_id": str(
                    (
                        target_memory.get("semantic_memory", {}).get("last_best_entry_id")
                        if isinstance(target_memory, dict) and isinstance(target_memory.get("semantic_memory"), dict)
                        else ""
                    )
                    or target_working.get("last_best_entry_id")
                    or ""
                ),
                "unsearched_houses": [
                    hid
                    for hid, item in sorted(self.house_registry.items())
                    if isinstance(item, dict) and not bool(item.get("searched", False))
                ],
                "decision_hint": decision_hint,
                "updated_at": _now_text(),
            }
        )

    def _normalize_house_memory(self, memory: Dict[str, Any]) -> None:
        if not isinstance(memory.get("working_memory"), dict):
            memory["working_memory"] = _default_working_memory()
        if not isinstance(memory.get("episodic_memory"), list):
            memory["episodic_memory"] = []
        if not isinstance(memory.get("semantic_memory"), dict):
            memory["semantic_memory"] = _default_semantic_memory()
        self._normalize_working_memory(memory)
        self._normalize_semantic_memory(memory)

    def _normalize_working_memory(self, memory: Dict[str, Any]) -> None:
        working_memory = memory.get("working_memory", {})
        if not isinstance(working_memory, dict):
            working_memory = _default_working_memory()
            memory["working_memory"] = working_memory
        for key, value in _default_working_memory().items():
            working_memory.setdefault(key, copy.deepcopy(value))
        perception_buffer = working_memory.get("perception_buffer", {})
        if not isinstance(perception_buffer, dict):
            perception_buffer = {"max_frames": DEFAULT_PERCEPTION_BUFFER_LIMIT, "frames": []}
            working_memory["perception_buffer"] = perception_buffer
        perception_buffer["max_frames"] = max(
            1,
            int(perception_buffer.get("max_frames", DEFAULT_PERCEPTION_BUFFER_LIMIT) or DEFAULT_PERCEPTION_BUFFER_LIMIT),
        )
        frames = perception_buffer.get("frames", [])
        perception_buffer["frames"] = list(frames)[-perception_buffer["max_frames"] :] if isinstance(frames, list) else []
        recent_actions = working_memory.get("recent_actions", [])
        working_memory["recent_actions"] = list(recent_actions)[-self.recent_actions_limit :] if isinstance(recent_actions, list) else []
        recent_decisions = working_memory.get("recent_target_decisions", [])
        working_memory["recent_target_decisions"] = (
            list(recent_decisions)[-self.recent_decisions_limit :] if isinstance(recent_decisions, list) else []
        )
        top_candidates = working_memory.get("top_candidates", [])
        working_memory["top_candidates"] = list(top_candidates)[: self.top_candidates_limit] if isinstance(top_candidates, list) else []

    def _refresh_perimeter_coverage(self, semantic_memory: Dict[str, Any]) -> Dict[str, Any]:
        coverage = semantic_memory.get("perimeter_coverage", {})
        if not isinstance(coverage, dict):
            coverage = _default_perimeter_coverage()
            semantic_memory["perimeter_coverage"] = coverage
        defaults = _default_perimeter_coverage()
        for key, value in defaults.items():
            coverage.setdefault(key, copy.deepcopy(value))
        bins = coverage.get("bins", {})
        if not isinstance(bins, dict):
            bins = {}
            coverage["bins"] = bins
        visited_bin_count = 0
        observed_bin_count = 0
        total_observations = 0
        visual_observation_count = 0
        for bin_id in DEFAULT_PERIMETER_BIN_IDS:
            bin_payload = bins.get(bin_id)
            if not isinstance(bin_payload, dict):
                bin_payload = _default_perimeter_bin()
                bins[bin_id] = bin_payload
            for key, value in _default_perimeter_bin().items():
                bin_payload.setdefault(key, copy.deepcopy(value))
            bin_payload["visit_count"] = max(0, int(bin_payload.get("visit_count", 0) or 0))
            bin_payload["visual_observation_count"] = max(0, int(bin_payload.get("visual_observation_count", 0) or 0))
            bin_payload["visited"] = bool(bin_payload.get("visited", False) or bin_payload["visit_count"] > 0)
            bin_payload["observed"] = bool(bin_payload.get("observed", False) or bin_payload["visual_observation_count"] > 0)
            if bin_payload["visited"]:
                visited_bin_count += 1
            if bin_payload["observed"]:
                observed_bin_count += 1
            total_observations += bin_payload["visit_count"]
            visual_observation_count += bin_payload["visual_observation_count"]
        bin_count = max(1, len(DEFAULT_PERIMETER_BIN_IDS))
        coverage["bin_count"] = bin_count
        coverage["visited_bin_count"] = visited_bin_count
        coverage["observed_bin_count"] = observed_bin_count
        coverage["visited_coverage_ratio"] = round(float(visited_bin_count) / float(bin_count), 4)
        coverage["observed_coverage_ratio"] = round(float(observed_bin_count) / float(bin_count), 4)
        coverage["total_observations"] = max(int(coverage.get("total_observations", 0) or 0), total_observations)
        coverage["visual_observation_count"] = max(int(coverage.get("visual_observation_count", 0) or 0), visual_observation_count)
        coverage["closed_loop_score"] = round(min(float(coverage["visited_coverage_ratio"]), 1.0), 4)
        coverage["visited_bins"] = [bin_id for bin_id in DEFAULT_PERIMETER_BIN_IDS if bins[bin_id].get("visited")]
        coverage["observed_bins"] = [bin_id for bin_id in DEFAULT_PERIMETER_BIN_IDS if bins[bin_id].get("observed")]
        return coverage

    def _refresh_face_coverage(self, semantic_memory: Dict[str, Any]) -> Dict[str, Any]:
        coverage = semantic_memory.get("face_coverage", {})
        if not isinstance(coverage, dict):
            coverage = _default_face_coverage()
            semantic_memory["face_coverage"] = coverage
        defaults = _default_face_coverage()
        for key, value in defaults.items():
            coverage.setdefault(key, copy.deepcopy(value))
        segment_count = max(1, int(coverage.get("segment_count_per_face", DEFAULT_FACE_SEGMENT_COUNT) or DEFAULT_FACE_SEGMENT_COUNT))
        coverage["segment_count_per_face"] = segment_count
        faces = coverage.get("faces", {})
        if not isinstance(faces, dict):
            faces = {}
            coverage["faces"] = faces
        visited_face_count = 0
        observed_face_count = 0
        visited_segment_count = 0
        observed_segment_count = 0
        total_observations = 0
        visual_observation_count = 0
        for face_id in DEFAULT_FACE_IDS:
            face = faces.get(face_id)
            if not isinstance(face, dict):
                face = _default_face_record()
                faces[face_id] = face
            for key, value in _default_face_record().items():
                if key == "segments":
                    continue
                face.setdefault(key, copy.deepcopy(value))
            segments = face.get("segments", [])
            if not isinstance(segments, list):
                segments = []
            while len(segments) < segment_count:
                segments.append(_default_face_segment())
            if len(segments) > segment_count:
                del segments[segment_count:]
            face["segments"] = segments
            face_visited_segments = 0
            face_observed_segments = 0
            face_visits = 0
            face_visuals = 0
            for index, segment in enumerate(segments):
                if not isinstance(segment, dict):
                    segment = _default_face_segment()
                    segments[index] = segment
                for key, value in _default_face_segment().items():
                    segment.setdefault(key, copy.deepcopy(value))
                segment["visit_count"] = max(0, int(segment.get("visit_count", 0) or 0))
                segment["visual_observation_count"] = max(0, int(segment.get("visual_observation_count", 0) or 0))
                segment["visited"] = bool(segment.get("visited", False) or segment["visit_count"] > 0)
                segment["observed"] = bool(segment.get("observed", False) or segment["visual_observation_count"] > 0)
                if segment["visited"]:
                    face_visited_segments += 1
                if segment["observed"]:
                    face_observed_segments += 1
                face_visits += int(segment["visit_count"])
                face_visuals += int(segment["visual_observation_count"])
            face["visit_count"] = max(int(face.get("visit_count", 0) or 0), face_visits)
            face["visual_observation_count"] = max(int(face.get("visual_observation_count", 0) or 0), face_visuals)
            face["visited_segment_count"] = face_visited_segments
            face["observed_segment_count"] = face_observed_segments
            face["visited"] = bool(face.get("visited", False) or face_visited_segments > 0)
            face["observed"] = bool(face.get("observed", False) or face_observed_segments > 0)
            face["coverage_ratio"] = round(float(face_visited_segments) / float(segment_count), 4)
            face["observed_coverage_ratio"] = round(float(face_observed_segments) / float(segment_count), 4)
            if face["visited"]:
                visited_face_count += 1
            if face["observed"]:
                observed_face_count += 1
            visited_segment_count += face_visited_segments
            observed_segment_count += face_observed_segments
            total_observations += int(face["visit_count"])
            visual_observation_count += int(face["visual_observation_count"])
        total_segment_count = max(1, len(DEFAULT_FACE_IDS) * segment_count)
        coverage["face_count"] = len(DEFAULT_FACE_IDS)
        coverage["total_segment_count"] = total_segment_count
        coverage["visited_face_count"] = visited_face_count
        coverage["observed_face_count"] = observed_face_count
        coverage["visited_segment_count"] = visited_segment_count
        coverage["observed_segment_count"] = observed_segment_count
        coverage["visited_coverage_ratio"] = round(float(visited_segment_count) / float(total_segment_count), 4)
        coverage["observed_coverage_ratio"] = round(float(observed_segment_count) / float(total_segment_count), 4)
        coverage["total_observations"] = max(int(coverage.get("total_observations", 0) or 0), total_observations)
        coverage["visual_observation_count"] = max(int(coverage.get("visual_observation_count", 0) or 0), visual_observation_count)
        coverage["visited_faces"] = [face_id for face_id in DEFAULT_FACE_IDS if faces[face_id].get("visited")]
        coverage["observed_faces"] = [face_id for face_id in DEFAULT_FACE_IDS if faces[face_id].get("observed")]
        coverage["unobserved_faces"] = [face_id for face_id in DEFAULT_FACE_IDS if not faces[face_id].get("observed")]
        return coverage

    def _entry_is_reliable_target_entry(self, entry: Dict[str, Any]) -> bool:
        status = str(entry.get("status") or entry.get("entry_state") or "").strip().lower()
        entry_type = str(entry.get("entry_type") or entry.get("semantic_class") or "").strip().lower().replace("_", " ")
        if "window" in entry_type or status in {"window_rejected", "non_target"}:
            return False
        if self._entry_is_non_enterable_candidate(entry):
            return False
        if bool(entry.get("is_entered", False)):
            return True
        if status in {"approachable"}:
            return True
        try:
            association_confidence = float(entry.get("association_confidence", 0.0) or 0.0)
        except Exception:
            association_confidence = 0.0
        try:
            target_match_score = float(entry.get("target_match_score", 0.0) or 0.0)
        except Exception:
            target_match_score = 0.0
        return association_confidence >= RELIABLE_ENTRY_ASSOCIATION_THRESHOLD and target_match_score >= 0.45

    def _entry_is_rejected_candidate(self, entry: Dict[str, Any]) -> bool:
        status = str(entry.get("status") or entry.get("entry_state") or "").strip().lower()
        entry_type = str(entry.get("entry_type") or entry.get("semantic_class") or "").strip().lower().replace("_", " ")
        return bool("window" in entry_type or status in {"window_rejected", "non_target", "rejected"})

    def _entry_is_non_enterable_candidate(self, entry: Dict[str, Any]) -> bool:
        status = str(entry.get("status") or entry.get("entry_state") or "").strip().lower()
        entry_type = str(entry.get("entry_type") or entry.get("semantic_class") or "").strip().lower().replace("_", " ")
        if entry_type in {"close door", "closed door"}:
            return True
        if status in {"closed", "close_door", "closed_door", "blocked_confirmed"}:
            return True
        opening_width = _safe_float(entry.get("opening_width_cm"), None)
        if status in {"blocked_temporary", "target_house_entry_blocked"} and opening_width is not None:
            return opening_width < NON_ENTERABLE_MAX_OPENING_WIDTH_CM
        bbox_history = entry.get("bbox_history", []) if isinstance(entry.get("bbox_history"), list) else []
        if status in {"blocked_temporary", "target_house_entry_blocked"} and bbox_history:
            closed_votes = 0
            open_votes = 0
            for item in bbox_history:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("class_name", "") or "").strip().lower().replace("_", " ")
                if name in {"close door", "closed door"}:
                    closed_votes += 1
                elif name == "open door":
                    open_votes += 1
            if closed_votes > 0 and open_votes == 0:
                return True
        return False

    def _refresh_search_completion_evidence(self, memory: Dict[str, Any]) -> None:
        semantic_memory = memory.get("semantic_memory", {}) if isinstance(memory.get("semantic_memory"), dict) else {}
        search_summary = semantic_memory.get("search_summary", {}) if isinstance(semantic_memory.get("search_summary"), dict) else {}
        coverage = semantic_memory.get("perimeter_coverage", {}) if isinstance(semantic_memory.get("perimeter_coverage"), dict) else {}
        face_coverage = semantic_memory.get("face_coverage", {}) if isinstance(semantic_memory.get("face_coverage"), dict) else {}
        candidate_entries = semantic_memory.get("candidate_entries", []) if isinstance(semantic_memory.get("candidate_entries"), list) else []
        candidate_count = len(candidate_entries)
        rejected_count = int(search_summary.get("rejected_entry_count", 0) or 0)
        approachable_count = int(search_summary.get("approachable_entry_count", 0) or 0)
        blocked_count = int(search_summary.get("blocked_entry_count", 0) or 0)
        visited_ratio = float(coverage.get("visited_coverage_ratio", 0.0) or 0.0)
        observed_ratio = float(coverage.get("observed_coverage_ratio", 0.0) or 0.0)
        total_observations = int(coverage.get("total_observations", 0) or 0)
        face_visited_ratio = float(face_coverage.get("visited_coverage_ratio", 0.0) or 0.0)
        face_observed_ratio = float(face_coverage.get("observed_coverage_ratio", 0.0) or 0.0)
        observed_face_count = int(face_coverage.get("observed_face_count", 0) or 0)
        face_observations = int(face_coverage.get("total_observations", 0) or 0)
        has_reliable_entry = any(
            self._entry_is_reliable_target_entry(entry)
            for entry in candidate_entries
            if isinstance(entry, dict)
        )
        non_enterable_count = sum(
            1
            for entry in candidate_entries
            if isinstance(entry, dict) and self._entry_is_non_enterable_candidate(entry)
        )
        rejected_or_non_enterable_count = sum(
            1
            for entry in candidate_entries
            if isinstance(entry, dict)
            and (self._entry_is_rejected_candidate(entry) or self._entry_is_non_enterable_candidate(entry))
        )
        unresolved_count = max(0, candidate_count - rejected_or_non_enterable_count)
        perimeter_coverage_ready = (
            visited_ratio >= NO_ENTRY_MIN_VISITED_COVERAGE_RATIO
            and observed_ratio >= NO_ENTRY_MIN_OBSERVED_COVERAGE_RATIO
            and total_observations >= NO_ENTRY_MIN_TOTAL_OBSERVATIONS
        )
        face_coverage_ready = (
            face_observed_ratio >= NO_ENTRY_MIN_FACE_OBSERVED_COVERAGE_RATIO
            and observed_face_count >= NO_ENTRY_MIN_OBSERVED_FACE_COUNT
        )
        full_coverage_ready = bool(perimeter_coverage_ready and (face_coverage_ready or face_observations <= 0))
        all_candidates_rejected_or_none = candidate_count == 0 or rejected_or_non_enterable_count >= candidate_count
        no_entry_after_full_coverage = bool(
            full_coverage_ready
            and not has_reliable_entry
            and approachable_count == 0
            and all_candidates_rejected_or_none
        )
        reasons: List[str] = []
        if perimeter_coverage_ready:
            reasons.append("perimeter_coverage_ready")
        else:
            reasons.append("perimeter_coverage_incomplete")
        if face_coverage_ready:
            reasons.append("face_coverage_ready")
        elif face_observations > 0:
            reasons.append("face_coverage_incomplete")
        if has_reliable_entry:
            reasons.append("reliable_entry_present")
        else:
            reasons.append("no_reliable_entry")
        if all_candidates_rejected_or_none:
            reasons.append("all_candidates_rejected_non_enterable_or_no_candidate")
        else:
            reasons.append("unrejected_candidates_remain")
        if approachable_count > 0:
            reasons.append("approachable_entry_present")
        if non_enterable_count > 0:
            reasons.append("non_enterable_entry_present")
        evidence = _default_search_completion_evidence()
        evidence.update(
            {
                "no_entry_after_full_coverage": no_entry_after_full_coverage,
                "full_coverage_ready": full_coverage_ready,
                "perimeter_coverage_ready": perimeter_coverage_ready,
                "face_coverage_ready": face_coverage_ready,
                "has_reliable_entry": has_reliable_entry,
                "visited_coverage_ratio": round(visited_ratio, 4),
                "observed_coverage_ratio": round(observed_ratio, 4),
                "face_visited_coverage_ratio": round(face_visited_ratio, 4),
                "face_observed_coverage_ratio": round(face_observed_ratio, 4),
                "observed_face_count": observed_face_count,
                "total_observations": total_observations,
                "candidate_entry_count": candidate_count,
                "rejected_entry_count": rejected_count,
                "non_enterable_entry_count": non_enterable_count,
                "rejected_or_non_enterable_entry_count": rejected_or_non_enterable_count,
                "unresolved_entry_count": unresolved_count,
                "approachable_entry_count": approachable_count,
                "blocked_entry_count": blocked_count,
                "reasons": reasons,
                "updated_at": _now_text(),
            }
        )
        semantic_memory["search_completion_evidence"] = evidence
        current_status = str(semantic_memory.get("entry_search_status") or "").strip()
        if no_entry_after_full_coverage and current_status not in {"entry_found", "entered_house"}:
            semantic_memory["entry_search_status"] = "no_entry_found_after_full_coverage"
        elif current_status == "no_entry_found_after_full_coverage" and not no_entry_after_full_coverage:
            if has_reliable_entry or approachable_count > 0 or blocked_count > 0:
                semantic_memory["entry_search_status"] = "entry_found"
            else:
                semantic_memory["entry_search_status"] = "searching_entry"

    def _normalize_semantic_memory(self, memory: Dict[str, Any]) -> None:
        semantic_memory = memory.get("semantic_memory", {})
        if not isinstance(semantic_memory, dict):
            semantic_memory = _default_semantic_memory()
            memory["semantic_memory"] = semantic_memory
        defaults = _default_semantic_memory()
        for key, value in defaults.items():
            semantic_memory.setdefault(key, copy.deepcopy(value))
        searched_sectors = semantic_memory.get("searched_sectors", {})
        if not isinstance(searched_sectors, dict):
            searched_sectors = {}
            semantic_memory["searched_sectors"] = searched_sectors
        for sector_id, sector_default in defaults["searched_sectors"].items():
            sector = searched_sectors.get(sector_id)
            if not isinstance(sector, dict):
                searched_sectors[sector_id] = copy.deepcopy(sector_default)
                continue
            for key, value in sector_default.items():
                sector.setdefault(key, copy.deepcopy(value))
        candidate_entries = semantic_memory.get("candidate_entries", [])
        semantic_memory["candidate_entries"] = list(candidate_entries) if isinstance(candidate_entries, list) else []
        observed_sector_count = 0
        approachable_entry_count = 0
        blocked_entry_count = 0
        rejected_entry_count = 0
        best_entry_id = str(
            semantic_memory.get("last_best_entry_id")
            or memory.get("working_memory", {}).get("last_best_entry_id")
            or ""
        ).strip()
        if not best_entry_id:
            best_entries = [
                str(entry.get("entry_id") or entry.get("candidate_id") or "").strip()
                for entry in semantic_memory["candidate_entries"]
                if isinstance(entry, dict) and _truthy(entry.get("is_best_candidate", False))
            ]
            if best_entries:
                best_entry_id = best_entries[-1]
                semantic_memory["last_best_entry_id"] = best_entry_id
                memory["working_memory"]["last_best_entry_id"] = best_entry_id
        for sector in searched_sectors.values():
            if isinstance(sector, dict) and bool(sector.get("observed", False)):
                observed_sector_count += 1
        for entry in semantic_memory["candidate_entries"]:
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("entry_id") or entry.get("candidate_id") or "").strip()
            entry["entry_id"] = entry_id
            entry["candidate_id"] = entry_id
            entry["entry_type"] = str(entry.get("entry_type") or entry.get("semantic_class") or entry.get("class_name") or "")
            entry["semantic_class"] = str(entry.get("semantic_class") or entry.get("entry_type") or entry.get("class_name") or "")
            entry["associated_house_id"] = str(entry.get("associated_house_id") or memory.get("house_id") or "")
            entry["entry_state"] = str(entry.get("entry_state") or entry.get("status") or "unverified")
            entry["status"] = str(entry.get("status") or entry.get("entry_state") or "unverified")
            entry["first_seen_time"] = str(entry.get("first_seen_time") or _now_text())
            entry["last_seen_time"] = str(entry.get("last_seen_time") or entry["first_seen_time"])
            entry["observation_count"] = max(1, int(entry.get("observation_count", 0) or 0))
            entry["source_frames"] = list(entry.get("source_frames", []))[-DEFAULT_SOURCE_FRAMES_LIMIT:] if isinstance(entry.get("source_frames"), list) else []
            entry["bbox_history"] = list(entry.get("bbox_history", []))[-DEFAULT_BBOX_HISTORY_LIMIT:] if isinstance(entry.get("bbox_history"), list) else []
            entry["is_best_candidate"] = bool(best_entry_id and entry_id == best_entry_id)
            if not isinstance(entry.get("association_evidence"), dict):
                entry["association_evidence"] = {
                    "distance_score": 0.0,
                    "view_consistency_score": 0.0,
                    "appearance_score": 0.0,
                    "language_score": 0.0,
                }
            if not isinstance(entry.get("world_position"), dict):
                entry["world_position"] = {"x": None, "y": None, "z": None, "source": ""}
            status = str(entry.get("status", "") or "")
            if status == "approachable":
                approachable_entry_count += 1
            elif status in {"blocked_temporary", "blocked_confirmed"}:
                blocked_entry_count += 1
            elif status in {"non_target", "window_rejected"}:
                rejected_entry_count += 1
        semantic_memory["search_summary"] = {
            "observed_sector_count": observed_sector_count,
            "candidate_entry_count": len(semantic_memory["candidate_entries"]),
            "approachable_entry_count": approachable_entry_count,
            "blocked_entry_count": blocked_entry_count,
            "rejected_entry_count": rejected_entry_count,
        }
        self._refresh_perimeter_coverage(semantic_memory)
        self._refresh_face_coverage(semantic_memory)
        self._refresh_search_completion_evidence(memory)


__all__ = [
    "DEFAULT_ENTRY_SEARCH_MEMORY_PATH",
    "DEFAULT_HOUSES_CONFIG_PATH",
    "DEFAULT_PERIMETER_BIN_IDS",
    "DEFAULT_FACE_IDS",
    "DEFAULT_SECTOR_IDS",
    "EntrySearchMemoryStore",
]
