from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import cv2
    import numpy as np
except Exception:  # pragma: no cover - reported at runtime by load_depth_image
    cv2 = None
    np = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SESSIONS_ROOT = PROJECT_ROOT / "captures_remote" / "memory_collection_sessions"
DEFAULT_HOUSES_CONFIG = PROJECT_ROOT / "UAV-Flow-Eval" / "houses_config.json"

DEFAULT_SEGMENT_COUNT_PER_FACE = 4
DEFAULT_MIN_TARGET_BEARING_FOR_VISIBLE = 80.0
DEFAULT_MIN_YOLO_CONFIDENCE = 0.25
DEFAULT_DEPTH_VALID_RATIO = 0.45
DEFAULT_OPEN_DEPTH_GAP_CM = 120.0
DEFAULT_CLOSED_DEPTH_GAP_CM = 80.0
DEFAULT_CENTER_FAR_RATIO = 0.20
DEFAULT_VERTICAL_CLEARANCE_RATIO = 0.22

DOOR_CLASSES = {"door", "open door", "open_door"}
CLOSED_DOOR_CLASSES = {"close door", "close_door", "closed door", "closed_door"}
WINDOW_CLASSES = {"window"}

FACE_IDS = ("east", "west", "north", "south")
COMPLETED_STATES = {
    "searched_no_entry",
    "searched_closed_entry",
    "searched_window_only",
    "entry_found_open",
}


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def normalize_class_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def discover_labeling_dirs(session_dir: Path, max_captures: int = 0) -> List[Path]:
    items: List[Path] = []
    seen: set[str] = set()

    def add_candidate(labeling_dir: Path) -> None:
        if not labeling_dir.is_dir() or not (labeling_dir / "fusion_result.json").is_file():
            return
        key = str(labeling_dir.resolve()).lower()
        if key in seen:
            return
        seen.add(key)
        items.append(labeling_dir)

    # Memory collection sessions store captures as memory_fusion_captures/*/labeling.
    capture_root = session_dir / "memory_fusion_captures"
    if capture_root.exists():
        for capture_dir in sorted(capture_root.iterdir(), key=lambda p: p.name):
            add_candidate(capture_dir / "labeling")

    # LLM control sessions store each decision snapshot as decision_*/labeling_inputs.
    if session_dir.exists():
        for decision_dir in sorted(session_dir.iterdir(), key=lambda p: p.name):
            if not decision_dir.is_dir():
                continue
            add_candidate(decision_dir / "labeling_inputs")
            add_candidate(decision_dir / "labeling")

    if max_captures > 0:
        return items[:max_captures]
    return items


def image_to_world_from_affine(affine: Any, image_x: float, image_y: float) -> Optional[Dict[str, float]]:
    if (
        not isinstance(affine, list)
        or len(affine) < 2
        or not isinstance(affine[0], list)
        or not isinstance(affine[1], list)
        or len(affine[0]) < 3
        or len(affine[1]) < 3
    ):
        return None
    a = safe_float(affine[0][0], None)
    b = safe_float(affine[0][1], None)
    c = safe_float(affine[0][2], None)
    d = safe_float(affine[1][0], None)
    e = safe_float(affine[1][1], None)
    f = safe_float(affine[1][2], None)
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


def load_house_registry(houses_config_path: Path) -> Dict[str, Dict[str, Any]]:
    data = read_json(houses_config_path)
    houses = data.get("houses", []) if isinstance(data.get("houses"), list) else []
    affine = (
        data.get("overhead_map", {})
        if isinstance(data.get("overhead_map"), dict)
        else {}
    ).get("calibration", {})
    affine = affine.get("affine_world_to_image", []) if isinstance(affine, dict) else []
    registry: Dict[str, Dict[str, Any]] = {}
    for house in houses:
        if not isinstance(house, dict):
            continue
        house_id = str(house.get("id", "") or "").strip()
        if not house_id:
            continue
        entry = dict(house)
        bbox = house.get("map_bbox_image", {}) if isinstance(house.get("map_bbox_image"), dict) else {}
        points: List[Dict[str, float]] = []
        if bbox:
            x1 = safe_float(bbox.get("x1"), None)
            y1 = safe_float(bbox.get("y1"), None)
            x2 = safe_float(bbox.get("x2"), None)
            y2 = safe_float(bbox.get("y2"), None)
            if None not in (x1, y1, x2, y2):
                for px, py in ((x1, y1), (x1, y2), (x2, y1), (x2, y2)):
                    world = image_to_world_from_affine(affine, float(px), float(py))
                    if world:
                        points.append(world)
        if points:
            xs = [float(point["x"]) for point in points]
            ys = [float(point["y"]) for point in points]
            entry["bbox_world"] = {
                "min_x": min(xs),
                "max_x": max(xs),
                "min_y": min(ys),
                "max_y": max(ys),
                "center_x": (min(xs) + max(xs)) / 2.0,
                "center_y": (min(ys) + max(ys)) / 2.0,
                "source": "map_bbox_image_affine",
            }
        else:
            center_x = safe_float(house.get("center_x"), None)
            center_y = safe_float(house.get("center_y"), None)
            radius = safe_float(house.get("radius_cm"), None)
            if None not in (center_x, center_y, radius):
                half = max(150.0, float(radius) * 0.45)
                entry["bbox_world"] = {
                    "min_x": float(center_x) - half,
                    "max_x": float(center_x) + half,
                    "min_y": float(center_y) - half,
                    "max_y": float(center_y) + half,
                    "center_x": float(center_x),
                    "center_y": float(center_y),
                    "source": "center_radius_fallback",
                }
        registry[house_id] = entry
    return registry


def target_context_from_fusion(fusion_payload: Dict[str, Any]) -> Dict[str, Any]:
    fusion = fusion_payload.get("fusion", {}) if isinstance(fusion_payload.get("fusion"), dict) else fusion_payload
    context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
    return context


def target_house_id_from_labeling(labeling_dir: Path, fusion_payload: Dict[str, Any]) -> str:
    metadata = read_json(labeling_dir / "sample_metadata.json")
    context = target_context_from_fusion(fusion_payload)
    snapshot = read_json(labeling_dir / "entry_search_memory_snapshot_after.json")
    mission = snapshot.get("house_mission", {}) if isinstance(snapshot.get("house_mission"), dict) else {}
    for value in (
        metadata.get("target_house_id"),
        context.get("target_house_id"),
        mission.get("target_house_id"),
    ):
        text = str(value or "").strip()
        if text:
            return text
    return ""


def pose_from_labeling(labeling_dir: Path, fusion_payload: Dict[str, Any]) -> Dict[str, float]:
    context = target_context_from_fusion(fusion_payload)
    pose = context.get("uav_pose_world", {}) if isinstance(context.get("uav_pose_world"), dict) else {}
    if not pose:
        snapshot = read_json(labeling_dir / "entry_search_memory_snapshot_after.json")
        pose = snapshot.get("pose", {}) if isinstance(snapshot.get("pose"), dict) else {}
    return {
        "x": float(safe_float(pose.get("x"), 0.0) or 0.0),
        "y": float(safe_float(pose.get("y"), 0.0) or 0.0),
        "z": float(safe_float(pose.get("z"), 0.0) or 0.0),
        "yaw": float(safe_float(pose.get("yaw", pose.get("task_yaw", 0.0)), 0.0) or 0.0),
    }


def face_segment_from_pose(
    house: Dict[str, Any],
    pose: Dict[str, float],
    segment_count: int,
) -> Optional[Dict[str, Any]]:
    bbox = house.get("bbox_world", {}) if isinstance(house.get("bbox_world"), dict) else {}
    if not bbox:
        return None
    min_x = safe_float(bbox.get("min_x"), None)
    max_x = safe_float(bbox.get("max_x"), None)
    min_y = safe_float(bbox.get("min_y"), None)
    max_y = safe_float(bbox.get("max_y"), None)
    center_x = safe_float(bbox.get("center_x"), None)
    center_y = safe_float(bbox.get("center_y"), None)
    if None in (min_x, max_x, min_y, max_y, center_x, center_y):
        return None
    uav_x = float(pose.get("x", 0.0))
    uav_y = float(pose.get("y", 0.0))
    half_x = max(1.0, (float(max_x) - float(min_x)) / 2.0)
    half_y = max(1.0, (float(max_y) - float(min_y)) / 2.0)
    dx = uav_x - float(center_x)
    dy = uav_y - float(center_y)
    if abs(dx) / half_x >= abs(dy) / half_y:
        face_id = "east" if dx >= 0.0 else "west"
        edge_t = (uav_y - float(min_y)) / max(1.0, float(max_y) - float(min_y))
    else:
        face_id = "north" if dy >= 0.0 else "south"
        edge_t = (uav_x - float(min_x)) / max(1.0, float(max_x) - float(min_x))
    edge_t = clamp(edge_t, 0.0, 1.0)
    count = max(1, int(segment_count))
    segment_index = min(count - 1, max(0, int(edge_t * float(count))))
    return {
        "face_id": face_id,
        "segment_index": int(segment_index),
        "edge_t": round(float(edge_t), 4),
        "bbox_world": bbox,
    }


def target_house_visible(fusion_payload: Dict[str, Any]) -> Tuple[bool, str]:
    context = target_context_from_fusion(fusion_payload)
    if bool(context.get("target_house_in_fov", False)):
        return True, "target_house_in_fov"
    bearing = safe_float(context.get("target_house_bearing_deg"), None)
    if bearing is not None and abs(float(bearing)) <= DEFAULT_MIN_TARGET_BEARING_FOR_VISIBLE:
        return True, "target_bearing_in_view"
    return False, "target_not_in_view"


def load_depth_image(labeling_dir: Path, depth_result: Dict[str, Any]) -> Optional[Any]:
    if cv2 is None or np is None:
        return None
    candidates = [
        Path(str(depth_result.get("depth_path", "") or "")),
        labeling_dir / "depth_cm.png",
        labeling_dir.parent / "inputs" / "depth_cm.png",
    ]
    for path in candidates:
        if not str(path) or not path.is_file():
            continue
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            return image
    return None


def valid_depth_values(depth: Any, max_valid_depth_cm: float) -> Any:
    if np is None:
        return depth
    values = depth.astype("float32")
    mask = (values > 1.0) & (values <= float(max_valid_depth_cm))
    return values[mask]


def depth_roi_stats(
    depth_image: Any,
    xyxy: List[float],
    *,
    max_valid_depth_cm: float,
) -> Dict[str, Any]:
    if np is None or depth_image is None:
        return {
            "available": False,
            "roi_valid_ratio": 0.0,
            "reason": "depth_image_unavailable",
        }
    height, width = depth_image.shape[:2]
    x1 = int(clamp(math.floor(float(xyxy[0])), 0, width - 1))
    y1 = int(clamp(math.floor(float(xyxy[1])), 0, height - 1))
    x2 = int(clamp(math.ceil(float(xyxy[2])), x1 + 1, width))
    y2 = int(clamp(math.ceil(float(xyxy[3])), y1 + 1, height))
    roi = depth_image[y1:y2, x1:x2].astype("float32")
    valid_mask = (roi > 1.0) & (roi <= float(max_valid_depth_cm))
    valid_values = roi[valid_mask]
    roi_area = max(1, int(roi.size))
    roi_valid_ratio = float(valid_values.size) / float(roi_area)

    margin_x = max(4, int((x2 - x1) * 0.6))
    margin_y = max(4, int((y2 - y1) * 0.6))
    sx1 = max(0, x1 - margin_x)
    sy1 = max(0, y1 - margin_y)
    sx2 = min(width, x2 + margin_x)
    sy2 = min(height, y2 + margin_y)
    surrounding = depth_image[sy1:sy2, sx1:sx2].astype("float32")
    surrounding_mask = np.ones(surrounding.shape, dtype=bool)
    inner_x1 = x1 - sx1
    inner_y1 = y1 - sy1
    inner_x2 = inner_x1 + (x2 - x1)
    inner_y2 = inner_y1 + (y2 - y1)
    surrounding_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False
    surrounding_valid_mask = (
        surrounding_mask
        & (surrounding > 1.0)
        & (surrounding <= float(max_valid_depth_cm))
    )
    surrounding_values = surrounding[surrounding_valid_mask]

    if valid_values.size <= 0:
        roi_median = 0.0
        roi_p10 = 0.0
        roi_p90 = 0.0
    else:
        roi_median = float(np.median(valid_values))
        roi_p10 = float(np.percentile(valid_values, 10))
        roi_p90 = float(np.percentile(valid_values, 90))
    surrounding_median = float(np.median(surrounding_values)) if surrounding_values.size > 0 else roi_median
    far_threshold = surrounding_median + float(DEFAULT_OPEN_DEPTH_GAP_CM)
    center_x1 = int((x2 - x1) * 0.25)
    center_x2 = max(center_x1 + 1, int((x2 - x1) * 0.75))
    center_y1 = int((y2 - y1) * 0.20)
    center_y2 = max(center_y1 + 1, int((y2 - y1) * 0.90))
    center = roi[center_y1:center_y2, center_x1:center_x2]
    center_valid = (center > 1.0) & (center <= float(max_valid_depth_cm))
    center_far = center_valid & (center >= far_threshold)
    center_band_far_ratio = float(center_far.sum()) / float(max(1, center_valid.sum()))

    row_far_ratios: List[float] = []
    for row_index in range(roi.shape[0]):
        row = roi[row_index, :]
        row_valid = (row > 1.0) & (row <= float(max_valid_depth_cm))
        if int(row_valid.sum()) <= 0:
            continue
        row_far = row_valid & (row >= far_threshold)
        row_far_ratios.append(float(row_far.sum()) / float(row_valid.sum()))
    vertical_clearance_ratio = float(sum(1 for value in row_far_ratios if value >= 0.20)) / float(max(1, len(row_far_ratios)))

    return {
        "available": True,
        "depth_bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
        "roi_valid_ratio": round(roi_valid_ratio, 4),
        "roi_depth_median_cm": round(roi_median, 3),
        "roi_depth_p10_cm": round(roi_p10, 3),
        "roi_depth_p90_cm": round(roi_p90, 3),
        "surrounding_depth_median_cm": round(surrounding_median, 3),
        "depth_gap_cm": round(roi_median - surrounding_median, 3),
        "center_band_far_ratio": round(center_band_far_ratio, 4),
        "vertical_clearance_ratio": round(vertical_clearance_ratio, 4),
    }


def classify_candidate(det: Dict[str, Any], depth_stats: Dict[str, Any]) -> Dict[str, Any]:
    cls = normalize_class_name(det.get("class_name_normalized") or det.get("class_name"))
    confidence = float(safe_float(det.get("confidence"), 0.0) or 0.0)
    depth_available = bool(depth_stats.get("available", False))
    valid_ratio = float(depth_stats.get("roi_valid_ratio", 0.0) or 0.0)
    depth_gap = float(depth_stats.get("depth_gap_cm", 0.0) or 0.0)
    center_far = float(depth_stats.get("center_band_far_ratio", 0.0) or 0.0)
    vertical_clearance = float(depth_stats.get("vertical_clearance_ratio", 0.0) or 0.0)

    if confidence < DEFAULT_MIN_YOLO_CONFIDENCE:
        return {
            "deterministic_state": "ignored_low_confidence",
            "segment_update": "observed_uncertain",
            "is_search_complete": False,
            "reason": f"YOLO confidence {confidence:.2f} below threshold.",
        }

    if cls in WINDOW_CLASSES:
        return {
            "deterministic_state": "window_only",
            "segment_update": "searched_window_only",
            "is_search_complete": True,
            "reason": "YOLO window evidence; window does not trigger entry approach.",
        }

    is_door_like = cls in DOOR_CLASSES or cls in CLOSED_DOOR_CLASSES
    if not is_door_like:
        return {
            "deterministic_state": "non_entry_object",
            "segment_update": "observed_uncertain",
            "is_search_complete": False,
            "reason": f"Unsupported class for entry search: {cls}",
        }

    if not depth_available or valid_ratio < DEFAULT_DEPTH_VALID_RATIO:
        return {
            "deterministic_state": "door_depth_uncertain",
            "segment_update": "observed_uncertain",
            "is_search_complete": False,
            "reason": "Door-like YOLO evidence but depth ROI is missing or invalid.",
        }

    open_like = (
        cls in DOOR_CLASSES
        and depth_gap >= DEFAULT_OPEN_DEPTH_GAP_CM
        and center_far >= DEFAULT_CENTER_FAR_RATIO
        and vertical_clearance >= DEFAULT_VERTICAL_CLEARANCE_RATIO
    )
    if open_like:
        return {
            "deterministic_state": "open_entry",
            "segment_update": "entry_found_open",
            "is_search_complete": True,
            "reason": "Door-like YOLO evidence with strong far-depth opening in the ROI.",
        }

    closed_like = (
        cls in CLOSED_DOOR_CLASSES
        or (
            abs(depth_gap) <= DEFAULT_CLOSED_DEPTH_GAP_CM
            and center_far < DEFAULT_CENTER_FAR_RATIO
            and vertical_clearance < DEFAULT_VERTICAL_CLEARANCE_RATIO
        )
    )
    if closed_like:
        if cls in CLOSED_DOOR_CLASSES and depth_gap >= DEFAULT_OPEN_DEPTH_GAP_CM and center_far >= DEFAULT_CENTER_FAR_RATIO:
            return {
                "deterministic_state": "conflicting_closed_class_open_depth",
                "segment_update": "needs_review",
                "is_search_complete": False,
                "reason": "YOLO says closed door but depth suggests an opening.",
            }
        return {
            "deterministic_state": "closed_entry",
            "segment_update": "searched_closed_entry",
            "is_search_complete": True,
            "reason": "Door-like YOLO evidence with no body-height far-depth opening.",
        }

    return {
        "deterministic_state": "door_depth_uncertain",
        "segment_update": "observed_uncertain",
        "is_search_complete": False,
        "reason": "Door-like evidence is not clearly open or closed under current thresholds.",
    }


def default_segment(segment_id: str) -> Dict[str, Any]:
    return {
        "segment_id": segment_id,
        "state": "unseen",
        "is_search_complete": False,
        "completion_confidence": 0.0,
        "observation_count": 0,
        "visibility_ratio_best": 0.0,
        "last_observed_step": None,
        "last_observed_time": "",
        "last_capture_name": "",
        "last_labeling_dir": "",
        "last_observed_pose": {},
        "observed_pose_extent": {},
        "edge_t_min": None,
        "edge_t_max": None,
        "evidence_type": "",
        "best_evidence": {},
        "post_completion_conflicts": [],
        "candidate_entries": [],
    }


def initialize_segment_memory(house_id: str, segment_count: int) -> Dict[str, Any]:
    faces: Dict[str, Any] = {}
    for face_id in FACE_IDS:
        faces[face_id] = {
            "segments": [
                default_segment(f"H{house_id}_{face_id}_{index}")
                for index in range(max(1, int(segment_count)))
            ]
        }
    return {
        "version": "segment_search_memory_replay_v1",
        "house_id": str(house_id),
        "segment_count_per_face": max(1, int(segment_count)),
        "faces": faces,
        "summary": {},
    }


def state_priority(state: str) -> int:
    order = {
        "unseen": 0,
        "observed_uncertain": 1,
        "blocked_or_occluded": 2,
        "searched_no_entry": 3,
        "searched_window_only": 4,
        "searched_closed_entry": 5,
        "entry_found_open": 6,
        "needs_review": 7,
    }
    return order.get(str(state), 0)


def resolve_segment_state(previous_state: str, requested_state: str, previous_complete: bool) -> Tuple[str, bool]:
    previous_state = str(previous_state or "unseen")
    requested_state = str(requested_state or "observed_uncertain")
    if previous_complete and requested_state not in COMPLETED_STATES:
        return previous_state, True
    if previous_state == "entry_found_open" and requested_state != "entry_found_open":
        return previous_state, True
    if requested_state == "entry_found_open":
        return requested_state, False
    if state_priority(requested_state) >= state_priority(previous_state):
        return requested_state, False
    return previous_state, False


def update_segment(
    memory: Dict[str, Any],
    face_id: str,
    segment_index: int,
    *,
    new_state: str,
    evidence: Dict[str, Any],
    step_index: int,
    labeling_dir: Path,
    visibility_ratio: float,
    pose: Optional[Dict[str, float]] = None,
    edge_t: Optional[float] = None,
) -> Dict[str, Any]:
    faces = memory.setdefault("faces", {})
    face = faces.setdefault(face_id, {"segments": []})
    segments = face.setdefault("segments", [])
    while len(segments) <= int(segment_index):
        segments.append(default_segment(f"H{memory.get('house_id', '')}_{face_id}_{len(segments)}"))
    segment = segments[int(segment_index)]
    previous_state = str(segment.get("state", "unseen") or "unseen")
    previous_complete = bool(segment.get("is_search_complete", False))
    requested_state = str(new_state)
    selected_state, completion_lock_applied = resolve_segment_state(previous_state, requested_state, previous_complete)
    segment["state"] = selected_state
    segment["is_search_complete"] = selected_state in COMPLETED_STATES
    segment["completion_confidence"] = 1.0 if segment["is_search_complete"] else 0.0
    segment["observation_count"] = int(segment.get("observation_count", 0) or 0) + 1
    segment["visibility_ratio_best"] = max(
        float(segment.get("visibility_ratio_best", 0.0) or 0.0),
        float(visibility_ratio),
    )
    segment["last_observed_step"] = int(step_index)
    segment["last_observed_time"] = datetime.now().isoformat(timespec="seconds")
    segment["last_capture_name"] = labeling_dir.parent.name
    segment["last_labeling_dir"] = str(labeling_dir)
    pose_payload: Dict[str, float] = {}
    if isinstance(pose, dict):
        pose_x = safe_float(pose.get("x"), None)
        pose_y = safe_float(pose.get("y"), None)
        pose_z = safe_float(pose.get("z"), None)
        pose_yaw = safe_float(pose.get("yaw"), None)
        if pose_x is not None and pose_y is not None:
            pose_payload = {"x": float(pose_x), "y": float(pose_y)}
            if pose_z is not None:
                pose_payload["z"] = float(pose_z)
            if pose_yaw is not None:
                pose_payload["yaw"] = float(pose_yaw)
            extent = segment.get("observed_pose_extent", {}) if isinstance(segment.get("observed_pose_extent"), dict) else {}
            if not extent:
                extent = {
                    "min_x": float(pose_x),
                    "max_x": float(pose_x),
                    "min_y": float(pose_y),
                    "max_y": float(pose_y),
                    "sample_count": 0,
                }
            extent["min_x"] = min(float(extent.get("min_x", pose_x)), float(pose_x))
            extent["max_x"] = max(float(extent.get("max_x", pose_x)), float(pose_x))
            extent["min_y"] = min(float(extent.get("min_y", pose_y)), float(pose_y))
            extent["max_y"] = max(float(extent.get("max_y", pose_y)), float(pose_y))
            extent["sample_count"] = int(extent.get("sample_count", 0) or 0) + 1
            segment["observed_pose_extent"] = extent
            segment["last_observed_pose"] = pose_payload
    edge_t_value = safe_float(edge_t, None)
    if edge_t_value is not None:
        previous_min = safe_float(segment.get("edge_t_min"), None)
        previous_max = safe_float(segment.get("edge_t_max"), None)
        segment["edge_t_min"] = float(edge_t_value) if previous_min is None else min(float(previous_min), float(edge_t_value))
        segment["edge_t_max"] = float(edge_t_value) if previous_max is None else max(float(previous_max), float(edge_t_value))
    if completion_lock_applied:
        conflicts = segment.setdefault("post_completion_conflicts", [])
        if isinstance(conflicts, list):
            conflicts.append(
                {
                    "requested_state": requested_state,
                    "kept_state": selected_state,
                    "step_index": int(step_index),
                    "capture_id": labeling_dir.parent.name,
                    "labeling_dir": str(labeling_dir),
                    "evidence_type": str(evidence.get("evidence_type", requested_state) or requested_state),
                    "evidence": evidence,
                }
            )
            segment["post_completion_conflicts"] = conflicts[-8:]
    else:
        segment["evidence_type"] = str(evidence.get("evidence_type", requested_state) or requested_state)
        segment["best_evidence"] = evidence
    candidate = evidence.get("candidate")
    if isinstance(candidate, dict):
        candidates = segment.setdefault("candidate_entries", [])
        if isinstance(candidates, list):
            candidates.append(candidate)
            segment["candidate_entries"] = candidates[-8:]
    return {
        "segment_id": segment.get("segment_id", ""),
        "face_id": face_id,
        "segment_index": int(segment_index),
        "previous_state": previous_state,
        "new_state": selected_state,
        "requested_state": requested_state,
        "was_complete": previous_complete,
        "is_search_complete": bool(segment["is_search_complete"]),
        "repeat_completed_observation": bool(previous_complete),
        "completion_lock_applied": bool(completion_lock_applied),
        "evidence_type": segment.get("evidence_type", ""),
        "pose": pose_payload,
        "observed_pose_extent": segment.get("observed_pose_extent", {}),
        "edge_t": float(edge_t_value) if edge_t_value is not None else None,
        "edge_t_range": {
            "min": segment.get("edge_t_min"),
            "max": segment.get("edge_t_max"),
        },
    }


def summarize_segment_memory(memory: Dict[str, Any]) -> Dict[str, Any]:
    counts: Counter = Counter()
    complete_segments: List[str] = []
    remaining_segments: List[str] = []
    entry_found_segments: List[str] = []
    blocked_or_occluded_segments: List[str] = []
    for face_id, face in (memory.get("faces", {}) if isinstance(memory.get("faces"), dict) else {}).items():
        segments = face.get("segments", []) if isinstance(face, dict) else []
        for index, segment in enumerate(segments):
            if not isinstance(segment, dict):
                continue
            state = str(segment.get("state", "unseen") or "unseen")
            segment_key = f"{face_id}_{index}"
            counts[state] += 1
            if bool(segment.get("is_search_complete", False)):
                complete_segments.append(segment_key)
            else:
                remaining_segments.append(segment_key)
            if state == "entry_found_open":
                entry_found_segments.append(segment_key)
            if state == "blocked_or_occluded":
                blocked_or_occluded_segments.append(segment_key)
    total = sum(counts.values())
    return {
        "total_segments": int(total),
        "state_counts": {str(key): int(value) for key, value in sorted(counts.items())},
        "complete_segment_count": len(complete_segments),
        "remaining_segment_count": len(remaining_segments),
        "completion_ratio": round(float(len(complete_segments)) / float(max(1, total)), 4),
        "complete_segments": complete_segments,
        "remaining_segments": remaining_segments,
        "entry_found_segments": entry_found_segments,
        "blocked_or_occluded_segments": blocked_or_occluded_segments,
        "next_unsearched_segment": remaining_segments[0] if remaining_segments else None,
    }


def yolo_detections(labeling_dir: Path) -> List[Dict[str, Any]]:
    yolo = read_json(labeling_dir / "yolo_result.json")
    detections = yolo.get("detections", []) if isinstance(yolo.get("detections"), list) else []
    result: List[Dict[str, Any]] = []
    for index, det in enumerate(detections):
        if not isinstance(det, dict):
            continue
        xyxy = det.get("xyxy", [])
        if not isinstance(xyxy, list) or len(xyxy) < 4:
            continue
        item = dict(det)
        item.setdefault("candidate_id", index)
        result.append(item)
    return result


def process_capture(
    labeling_dir: Path,
    *,
    house_registry: Dict[str, Dict[str, Any]],
    segment_memories: Dict[str, Dict[str, Any]],
    segment_count: int,
    target_house_override: str = "",
) -> Dict[str, Any]:
    fusion_payload = read_json(labeling_dir / "fusion_result.json")
    depth_result = read_json(labeling_dir / "depth_result.json")
    target_house_id = str(target_house_override or target_house_id_from_labeling(labeling_dir, fusion_payload) or "").strip()
    metadata = read_json(labeling_dir / "sample_metadata.json")
    step_index = safe_int(metadata.get("memory_step_index", metadata.get("step_index", 0)), 0)
    pose = pose_from_labeling(labeling_dir, fusion_payload)
    visible, visible_reason = target_house_visible(fusion_payload)
    house = house_registry.get(target_house_id, {}) if target_house_id else {}
    face_hit = face_segment_from_pose(house, pose, segment_count) if house else None
    max_valid_depth = float(safe_float(depth_result.get("max_valid_depth_cm"), 1200.0) or 1200.0)
    depth_image = load_depth_image(labeling_dir, depth_result)
    detections = yolo_detections(labeling_dir)
    classes = [normalize_class_name(det.get("class_name_normalized") or det.get("class_name")) for det in detections]
    updates: List[Dict[str, Any]] = []
    candidate_evidence: List[Dict[str, Any]] = []
    skip_reason = ""

    if not target_house_id:
        skip_reason = "missing_target_house_id"
    elif not house:
        skip_reason = "target_house_not_in_houses_config"
    elif not face_hit:
        skip_reason = "face_segment_assignment_failed"
    elif not visible:
        skip_reason = visible_reason

    if target_house_id and target_house_id not in segment_memories:
        segment_memories[target_house_id] = initialize_segment_memory(target_house_id, segment_count)
    memory = segment_memories.get(target_house_id, {})

    if not skip_reason and memory:
        face_id = str(face_hit.get("face_id", ""))
        segment_index = int(face_hit.get("segment_index", 0) or 0)
        relevant_candidates = []
        for det in detections:
            cls = normalize_class_name(det.get("class_name_normalized") or det.get("class_name"))
            if cls not in (DOOR_CLASSES | CLOSED_DOOR_CLASSES | WINDOW_CLASSES):
                continue
            xyxy = [float(value) for value in det.get("xyxy", [])[:4]]
            stats = depth_roi_stats(depth_image, xyxy, max_valid_depth_cm=max_valid_depth)
            classification = classify_candidate(det, stats)
            candidate_payload = {
                "candidate_id": str(det.get("candidate_id", "")),
                "class_name": cls,
                "confidence": round(float(safe_float(det.get("confidence"), 0.0) or 0.0), 4),
                "bbox_xyxy": xyxy,
                "depth_roi": stats,
                "deterministic_state": classification.get("deterministic_state"),
                "segment_update": classification.get("segment_update"),
                "is_search_complete": classification.get("is_search_complete"),
                "reason": classification.get("reason"),
            }
            candidate_evidence.append(candidate_payload)
            relevant_candidates.append((classification, candidate_payload))

        if relevant_candidates:
            review_candidates = [
                item for item in relevant_candidates if str(item[0].get("segment_update", "")) == "needs_review"
            ]
            selectable = review_candidates if review_candidates else relevant_candidates
            best_classification, best_candidate = sorted(
                selectable,
                key=lambda item: (
                    state_priority(str(item[0].get("segment_update", ""))),
                    float(item[1].get("confidence", 0.0) or 0.0),
                ),
                reverse=True,
            )[0]
            updates.append(
                update_segment(
                    memory,
                    face_id,
                    segment_index,
                    new_state=str(best_classification.get("segment_update", "observed_uncertain")),
                    evidence={
                        "evidence_type": str(best_classification.get("segment_update", "observed_uncertain")),
                        "capture_id": labeling_dir.parent.name,
                        "labeling_dir": str(labeling_dir),
                        "visible_reason": visible_reason,
                        "candidate": best_candidate,
                        "all_candidate_count": len(relevant_candidates),
                    },
                    step_index=step_index,
                    labeling_dir=labeling_dir,
                    visibility_ratio=0.82,
                    pose=pose,
                    edge_t=safe_float(face_hit.get("edge_t"), None),
                )
            )
        else:
            front_obstacle = depth_result.get("front_obstacle", {}) if isinstance(depth_result.get("front_obstacle"), dict) else {}
            severity = str(front_obstacle.get("severity", "") or "").lower()
            if severity in {"blocked", "severe", "critical"}:
                state = "blocked_or_occluded"
                evidence_type = "front_obstacle_occlusion"
            else:
                state = "searched_no_entry"
                evidence_type = "visible_wall_no_entry"
            updates.append(
                update_segment(
                    memory,
                    face_id,
                    segment_index,
                    new_state=state,
                    evidence={
                        "evidence_type": evidence_type,
                        "capture_id": labeling_dir.parent.name,
                        "labeling_dir": str(labeling_dir),
                        "visible_reason": visible_reason,
                        "yolo_classes": classes,
                        "front_obstacle": front_obstacle,
                    },
                    step_index=step_index,
                    labeling_dir=labeling_dir,
                    visibility_ratio=0.74,
                    pose=pose,
                    edge_t=safe_float(face_hit.get("edge_t"), None),
                )
            )

    if memory:
        memory["summary"] = summarize_segment_memory(memory)

    return {
        "capture_name": labeling_dir.parent.name,
        "labeling_dir": str(labeling_dir),
        "step_index": int(step_index),
        "target_house_id": target_house_id,
        "pose": pose,
        "target_visible": bool(visible),
        "visible_reason": visible_reason,
        "face_hit": face_hit,
        "yolo_class_counts": dict(Counter(classes)),
        "candidate_evidence": candidate_evidence,
        "updates": updates,
        "skip_reason": skip_reason,
    }


def replay_session(
    session_dir: Path,
    *,
    houses_config_path: Path = DEFAULT_HOUSES_CONFIG,
    output_path: Optional[Path] = None,
    target_house_id: str = "",
    segment_count: int = DEFAULT_SEGMENT_COUNT_PER_FACE,
    max_captures: int = 0,
) -> Dict[str, Any]:
    session_dir = session_dir.resolve()
    labeling_dirs = discover_labeling_dirs(session_dir, max_captures=max_captures)
    house_registry = load_house_registry(houses_config_path)
    segment_memories: Dict[str, Dict[str, Any]] = {}
    timeline: List[Dict[str, Any]] = []
    for labeling_dir in labeling_dirs:
        event = process_capture(
            labeling_dir,
            house_registry=house_registry,
            segment_memories=segment_memories,
            segment_count=segment_count,
            target_house_override=target_house_id,
        )
        timeline.append(event)
    for memory in segment_memories.values():
        memory["summary"] = summarize_segment_memory(memory)
    repeated_completed = [
        update
        for event in timeline
        for update in event.get("updates", [])
        if isinstance(update, dict) and bool(update.get("repeat_completed_observation", False))
    ]
    report = {
        "version": "segment_search_memory_replay_report_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session_dir": str(session_dir),
        "houses_config_path": str(houses_config_path),
        "segment_count_per_face": int(segment_count),
        "target_house_override": str(target_house_id or ""),
        "capture_count": len(labeling_dirs),
        "processed_event_count": len(timeline),
        "segment_memories": segment_memories,
        "timeline": timeline,
        "summary": {
            "house_count": len(segment_memories),
            "repeated_completed_segment_observations": len(repeated_completed),
            "skipped_capture_count": sum(1 for event in timeline if event.get("skip_reason")),
            "skip_reasons": dict(Counter(str(event.get("skip_reason", "")) for event in timeline if event.get("skip_reason"))),
        },
    }
    if output_path is None:
        output_path = session_dir / "segment_search_memory_report.json"
    write_json(output_path, report)
    report["output_path"] = str(output_path)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a memory collection session and build deterministic segment search memory.")
    parser.add_argument("--session_dir", required=True, help="Path to captures_remote/memory_collection_sessions/memory_episode_*")
    parser.add_argument("--houses_config", default=str(DEFAULT_HOUSES_CONFIG), help="Path to UAV-Flow-Eval/houses_config.json")
    parser.add_argument("--output_path", default="", help="Optional report JSON path. Defaults to <session_dir>/segment_search_memory_report.json")
    parser.add_argument("--target_house_id", default="", help="Optional target house id override, e.g. 002")
    parser.add_argument("--segment_count_per_face", type=int, default=DEFAULT_SEGMENT_COUNT_PER_FACE)
    parser.add_argument("--max_captures", type=int, default=0, help="Only process first N captures for quick tests.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    session_dir = Path(args.session_dir)
    output_path = Path(args.output_path) if str(args.output_path or "").strip() else None
    report = replay_session(
        session_dir,
        houses_config_path=Path(args.houses_config),
        output_path=output_path,
        target_house_id=str(args.target_house_id or "").strip(),
        segment_count=max(1, int(args.segment_count_per_face)),
        max_captures=max(0, int(args.max_captures)),
    )
    summary = report.get("summary", {})
    print(
        "[segment-replay] done "
        f"captures={report.get('capture_count', 0)} "
        f"houses={summary.get('house_count', 0)} "
        f"repeated_completed={summary.get('repeated_completed_segment_observations', 0)} "
        f"skipped={summary.get('skipped_capture_count', 0)}"
    )
    print(f"[segment-replay] report -> {report.get('output_path')}")


if __name__ == "__main__":
    main()
