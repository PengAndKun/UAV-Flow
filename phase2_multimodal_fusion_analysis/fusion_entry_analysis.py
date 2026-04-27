from __future__ import annotations

import copy
import json
import math
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PHASE2_YOLO_DIR = ROOT / "phase2_door_or_window_yolo26_training"
PHASE2_DEPTH_DIR = ROOT / "phase2_modality2_depth_analysis"
for candidate in (PHASE2_YOLO_DIR, PHASE2_DEPTH_DIR):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from run_phase2_door_or_window_predict import find_latest_best_weights
from run_phase2_depth_entry_analysis import (
    analyze_depth_entry,
    build_depth_preview,
    draw_overlay,
    estimate_opening_width_cm,
    extract_ring_mask,
    load_depth_image,
    write_text_summary,
)
try:
    from .entry_search_memory import DEFAULT_SECTOR_IDS, EntrySearchMemoryStore
except ImportError:
    from entry_search_memory import DEFAULT_SECTOR_IDS, EntrySearchMemoryStore


DOORLIKE_CLASSES = {
    "door",
    "open door",
    "open_door",
    "close door",
    "close_door",
    "closed door",
    "closed_door",
}
OPEN_DOOR_CLASSES = {"open door", "open_door"}
WINDOW_CLASSES = {"window"}
CROSSING_READY_MAX_DISTANCE_CM = 320.0
DEFAULT_HOUSES_CONFIG_PATH = ROOT / "UAV-Flow-Eval" / "houses_config.json"
TARGET_MATCH_MIN_SCORE = 0.55
TARGET_MATCH_MIN_TARGET_SCORE = 0.45
MEMORY_REJECTED_CANDIDATE_STATUSES = {"non_target", "window_rejected", "blocked_confirmed"}
MEMORY_REVISIT_CANDIDATE_STATUSES = {"approachable", "blocked_temporary", "unverified"}
MEMORY_LOW_YIELD_ENTRY_STATES = {"no_entry", "weak_entry", "non_target_entry_visible"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def copy_if_exists(src: Optional[str], dst: Path) -> Optional[str]:
    if not src:
        return None
    src_path = Path(str(src))
    if not src_path.exists():
        return None
    shutil.copy2(src_path, dst)
    return str(dst)


def find_latest_phase2_weights(project_root: Optional[Path] = None) -> Path:
    root = (project_root or PHASE2_YOLO_DIR).resolve()
    return find_latest_best_weights(root)


def normalize_class_name(name: Any) -> str:
    return str(name or "").strip().lower().replace("_", " ")


def normalize_angle_deg(angle_deg: float) -> float:
    value = float(angle_deg)
    while value > 180.0:
        value -= 360.0
    while value <= -180.0:
        value += 360.0
    return value


def side_from_x_norm(x_norm: Optional[float]) -> str:
    if x_norm is None:
        return "unknown"
    value = float(x_norm)
    if value < 0.44:
        return "left"
    if value > 0.56:
        return "right"
    return "center"


def side_from_bearing_deg(relative_bearing_deg: Optional[float], hfov_deg: float) -> str:
    if relative_bearing_deg is None:
        return "unknown"
    bearing = float(relative_bearing_deg)
    if abs(bearing) > float(hfov_deg) * 0.5:
        return "out_of_view"
    if bearing > 12.0:
        return "left"
    if bearing < -12.0:
        return "right"
    return "center"


def detection_center_x_norm(detection: Dict[str, Any], image_width: int) -> Optional[float]:
    det_box = detection.get("xyxy", [])
    if not isinstance(det_box, list) or len(det_box) != 4 or image_width <= 0:
        return None
    return clamp(((float(det_box[0]) + float(det_box[2])) * 0.5) / float(image_width), 0.0, 1.0)


def bbox_from_xyxy(xyxy: Any) -> Dict[str, Any]:
    if not isinstance(xyxy, list) or len(xyxy) != 4:
        return {}
    x1, y1, x2, y2 = [float(value) for value in xyxy]
    return {
        "x": x1,
        "y": y1,
        "width": max(0.0, x2 - x1),
        "height": max(0.0, y2 - y1),
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def build_observation_frame_id(state: Optional[Dict[str, Any]], observation_id: str) -> str:
    if str(observation_id or "").strip():
        return str(observation_id).strip()
    state = state if isinstance(state, dict) else {}
    for key in ("frame_id", "capture_id", "image_id"):
        value = str(state.get(key) or "").strip()
        if value:
            return value
    memory_collection = state.get("memory_collection", {}) if isinstance(state.get("memory_collection"), dict) else {}
    for value in (
        state.get("memory_step_index"),
        state.get("step_index"),
        memory_collection.get("step_index"),
    ):
        try:
            return f"frame_{int(value):06d}"
        except Exception:
            text = str(value or "").strip()
            if text:
                return text
    return f"frame_{int(datetime.now().timestamp() * 1000)}"


def build_bbox_history_payload(
    candidate: Dict[str, Any],
    *,
    frame_id: str,
    observation_time: str,
) -> Dict[str, Any]:
    xyxy = candidate.get("xyxy", []) if isinstance(candidate.get("xyxy"), list) else []
    bbox = (
        candidate.get("bbox", {})
        if isinstance(candidate.get("bbox"), dict) and candidate.get("bbox")
        else bbox_from_xyxy(xyxy)
    )
    return {
        "frame_id": str(frame_id or ""),
        "time": str(observation_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "class_name": str(candidate.get("class_name") or ""),
        "confidence": float(candidate.get("confidence", 0.0) or 0.0),
        "bbox": copy.deepcopy(bbox),
        "xyxy": [float(value) for value in xyxy] if len(xyxy) == 4 else [],
        "center_x_norm": candidate.get("center_x_norm"),
        "entry_distance_cm": candidate.get("entry_distance_cm"),
        "opening_width_cm": candidate.get("opening_width_cm"),
        "candidate_total_score": float(candidate.get("candidate_total_score", 0.0) or 0.0),
        "target_match_score": float(candidate.get("candidate_target_match_score", 0.0) or 0.0),
    }


def build_association_evidence(candidate: Dict[str, Any]) -> Dict[str, float]:
    components = (
        candidate.get("candidate_target_score_components", {})
        if isinstance(candidate.get("candidate_target_score_components"), dict)
        else {}
    )
    bearing_score = float(components.get("bearing_score", 0.0) or 0.0)
    image_side_score = float(components.get("image_side_score", 0.0) or 0.0)
    return {
        "distance_score": float(components.get("range_score", 0.0) or 0.0),
        "view_consistency_score": float(0.5 * bearing_score + 0.5 * image_side_score),
        "appearance_score": float(candidate.get("candidate_semantic_score", 0.0) or 0.0),
        "language_score": 0.0,
        "geometry_score": float(candidate.get("candidate_geometry_score", 0.0) or 0.0),
        "memory_similarity_score": float(candidate.get("memory_best_similarity", 0.0) or 0.0),
    }


def house_lookup_from_config(houses_config: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    houses = houses_config.get("houses", []) if isinstance(houses_config, dict) else []
    lookup: Dict[str, Dict[str, Any]] = {}
    for house in houses:
        if not isinstance(house, dict):
            continue
        house_id = str(house.get("id") or "").strip()
        if house_id:
            lookup[house_id] = house
    return lookup


def load_houses_config(path: Optional[Path] = None) -> Dict[str, Any]:
    config_path = (path or DEFAULT_HOUSES_CONFIG_PATH).resolve()
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_target_context(
    *,
    state: Optional[Dict[str, Any]],
    houses_config: Dict[str, Any],
    image_shape: Tuple[int, int],
    hfov_deg: float,
) -> Dict[str, Any]:
    state = state if isinstance(state, dict) else {}
    pose = state.get("pose", {}) if isinstance(state.get("pose"), dict) else {}
    house_mission = state.get("house_mission", {}) if isinstance(state.get("house_mission"), dict) else {}
    house_registry = state.get("house_registry", {}) if isinstance(state.get("house_registry"), dict) else {}
    houses_by_id = house_lookup_from_config(houses_config)

    target_house_id = str(
        house_mission.get("target_house_id")
        or state.get("target_house_id")
        or house_registry.get("target_house_id")
        or houses_config.get("current_target_id")
        or ""
    ).strip()
    current_house_id = str(
        house_mission.get("current_house_id")
        or state.get("current_house_id")
        or state.get("current_house")
        or ""
    ).strip()
    target_house = houses_by_id.get(target_house_id, {})

    pose_x = pose.get("x", house_mission.get("uav_x"))
    pose_y = pose.get("y", house_mission.get("uav_y"))
    pose_yaw = pose.get("uav_yaw", pose.get("task_yaw", house_mission.get("uav_yaw")))
    try:
        pose_x = float(pose_x) if pose_x is not None else None
        pose_y = float(pose_y) if pose_y is not None else None
        pose_yaw = float(pose_yaw) if pose_yaw is not None else None
    except Exception:
        pose_x = pose_y = pose_yaw = None

    target_x = target_house.get("center_x")
    target_y = target_house.get("center_y")
    try:
        target_x = float(target_x) if target_x is not None else None
        target_y = float(target_y) if target_y is not None else None
    except Exception:
        target_x = target_y = None

    target_distance_cm = house_mission.get("distance_to_target_cm", state.get("target_distance_cm"))
    try:
        target_distance_cm = float(target_distance_cm) if target_distance_cm is not None else None
    except Exception:
        target_distance_cm = None
    if target_distance_cm is None and None not in (pose_x, pose_y, target_x, target_y):
        target_distance_cm = float(math.hypot(target_x - pose_x, target_y - pose_y))

    target_bearing_deg: Optional[float] = None
    if None not in (pose_x, pose_y, pose_yaw, target_x, target_y):
        absolute_bearing = math.degrees(math.atan2(float(target_y - pose_y), float(target_x - pose_x)))
        target_bearing_deg = normalize_angle_deg(absolute_bearing - float(pose_yaw))

    expected_side = side_from_bearing_deg(target_bearing_deg, hfov_deg)
    target_in_fov = expected_side != "out_of_view" if target_bearing_deg is not None else False
    expected_image_x: Optional[float] = None
    if target_bearing_deg is not None and target_in_fov:
        expected_image_x = clamp(0.5 - (float(target_bearing_deg) / max(1.0, float(hfov_deg))), 0.0, 1.0)

    image_bbox = target_house.get("map_bbox_image", {}) if isinstance(target_house.get("map_bbox_image"), dict) else {}
    return {
        "target_house_id": target_house_id or None,
        "target_house_name": target_house.get("name"),
        "current_house_id": current_house_id or None,
        "current_house_name": house_mission.get("current_house_name"),
        "target_house_distance_cm": target_distance_cm,
        "target_house_bearing_deg": target_bearing_deg,
        "target_house_in_fov": bool(target_in_fov),
        "target_house_expected_side": expected_side,
        "target_house_expected_image_x": expected_image_x,
        "target_house_center_world": {
            "x": target_x,
            "y": target_y,
        },
        "target_house_map_bbox_image": image_bbox or None,
        "uav_pose_world": {
            "x": pose_x,
            "y": pose_y,
            "yaw": pose_yaw,
        },
        "image_width": int(image_shape[1]) if len(image_shape) >= 2 else None,
        "image_height": int(image_shape[0]) if len(image_shape) >= 1 else None,
    }


def infer_target_sector_id(target_context: Dict[str, Any]) -> str:
    bearing = target_context.get("target_house_bearing_deg")
    in_fov = bool(target_context.get("target_house_in_fov", False))
    try:
        bearing_value = float(bearing) if bearing is not None else None
    except Exception:
        bearing_value = None
    if bearing_value is None:
        return "front_center"
    if not in_fov:
        if bearing_value > 0.0:
            return "left_side"
        if bearing_value < 0.0:
            return "right_side"
        return "front_center"
    if bearing_value > 12.0:
        return "front_left"
    if bearing_value < -12.0:
        return "front_right"
    return "front_center"


def infer_entry_search_status(
    *,
    target_house_id: Optional[str],
    target_conditioned_state: str,
    target_conditioned_subgoal: str,
) -> str:
    if not str(target_house_id or "").strip():
        return "not_started"
    state_value = str(target_conditioned_state or "").strip()
    subgoal_value = str(target_conditioned_subgoal or "").strip()
    if subgoal_value == "cross_target_entry":
        return "entry_found"
    if state_value == "target_house_entry_approachable":
        return "entry_found"
    if state_value in {
        "target_house_entry_visible",
        "target_house_entry_blocked",
        "non_target_house_entry_visible",
        "target_house_not_in_view",
        "target_house_geometric_opening_needs_confirmation",
    }:
        return "searching_entry"
    return "searching_entry"


def infer_observation_result(target_conditioned_state: str) -> str:
    state_value = str(target_conditioned_state or "").strip()
    if state_value == "target_house_entry_blocked":
        return "blocked_entry"
    if state_value == "target_house_entry_approachable":
        return "approachable_entry"
    if state_value == "non_target_house_entry_visible":
        return "non_target_entry_visible"
    if state_value == "target_house_entry_visible":
        return "weak_entry"
    return "no_entry"


def infer_candidate_entry_status(
    *,
    candidate: Dict[str, Any],
    target_conditioned_state: str,
) -> str:
    normalized_name = normalize_class_name(candidate.get("class_name"))
    if bool(candidate.get("candidate_is_target_house_entry", False)):
        if target_conditioned_state == "target_house_entry_blocked":
            return "blocked_temporary"
        if target_conditioned_state == "target_house_entry_approachable":
            return "approachable"
        return "unverified"
    if normalized_name in WINDOW_CLASSES:
        return "window_rejected"
    if target_conditioned_state == "non_target_house_entry_visible":
        return "non_target"
    return "unverified"


def update_entry_search_memory_from_fusion(
    *,
    houses_config: Dict[str, Any],
    fusion_result: Dict[str, Any],
    state: Optional[Dict[str, Any]] = None,
    yolo_result: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    store = EntrySearchMemoryStore()
    store.load()
    store.ensure_from_houses_config()

    target_context = fusion_result.get("target_context", {}) if isinstance(fusion_result.get("target_context"), dict) else {}
    target_house_id = str(target_context.get("target_house_id") or "").strip()
    if target_house_id:
        store.set_current_target_house(target_house_id)
    else:
        store.save()
        return {
            "status": "ok",
            "memory_path": store.path,
            "house_id": None,
            "sector_id": None,
            "semantic_memory": None,
        }

    houses_by_id = house_lookup_from_config(houses_config)
    house_meta = houses_by_id.get(target_house_id, {})
    store.ensure_house(
        target_house_id,
        house_name=str(house_meta.get("name", "") or target_house_id),
        house_status=str(house_meta.get("status", "UNSEARCHED") or "UNSEARCHED"),
    )
    house_mission = state.get("house_mission", {}) if isinstance(state, dict) and isinstance(state.get("house_mission"), dict) else {}
    pose = state.get("pose", {}) if isinstance(state, dict) and isinstance(state.get("pose"), dict) else {}
    current_house_id = str(
        house_mission.get("current_house_id")
        or target_context.get("current_house_id")
        or ""
    ).strip()
    observation_id = str(state.get("observation_id") or "") if isinstance(state, dict) else ""
    observation_time = str(state.get("observation_time") or "") if isinstance(state, dict) else ""
    frame_id = build_observation_frame_id(state, observation_id)
    last_action = str(state.get("last_action") or "") if isinstance(state, dict) else ""
    store.update_working_memory(
        target_house_id,
        {
            "target_house_id": target_house_id,
            "current_house_id": current_house_id,
        },
    )

    target_conditioned_state = str(fusion_result.get("target_conditioned_state", "") or "")
    target_conditioned_subgoal = str(fusion_result.get("target_conditioned_subgoal", "") or "")
    target_conditioned_action = str(fusion_result.get("target_conditioned_action_hint", "") or "")
    sector_id = infer_target_sector_id(target_context)
    entry_search_status = infer_entry_search_status(
        target_house_id=target_house_id,
        target_conditioned_state=target_conditioned_state,
        target_conditioned_subgoal=target_conditioned_subgoal,
    )
    store.set_entry_search_status(target_house_id, entry_search_status)
    store.append_recent_target_decision(
        target_house_id,
        {
            "target_conditioned_state": target_conditioned_state,
            "target_conditioned_subgoal": target_conditioned_subgoal,
            "target_conditioned_action_hint": target_conditioned_action,
            "target_reason": str(fusion_result.get("target_conditioned_reason") or ""),
            "timestamp": observation_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    if last_action:
        store.append_recent_action(target_house_id, last_action)

    detections = yolo_result.get("detections", []) if isinstance(yolo_result, dict) and isinstance(yolo_result.get("detections"), list) else []
    if detections:
        perception_candidates: List[Dict[str, Any]] = []
        for rank, detection in enumerate(detections[:3]):
            if not isinstance(detection, dict):
                continue
            det_xyxy = detection.get("xyxy", []) if isinstance(detection.get("xyxy"), list) else []
            bbox_payload: Dict[str, Any] = {}
            if len(det_xyxy) == 4:
                bbox_payload = {
                    "x1": float(det_xyxy[0]),
                    "y1": float(det_xyxy[1]),
                    "x2": float(det_xyxy[2]),
                    "y2": float(det_xyxy[3]),
                    "width": float(det_xyxy[2]) - float(det_xyxy[0]),
                    "height": float(det_xyxy[3]) - float(det_xyxy[1]),
                }
            perception_candidates.append(
                {
                    "candidate_id": str(detection.get("candidate_id") or rank),
                    "class_name": str(detection.get("class_name") or ""),
                    "confidence": float(detection.get("confidence", 0.0) or 0.0),
                    "bbox": bbox_payload,
                    "xyxy": [float(v) for v in det_xyxy] if len(det_xyxy) == 4 else [],
                }
            )
        store.append_perception_frame(
            target_house_id,
            {
                "frame_id": observation_id or f"obs_{int(datetime.now().timestamp())}",
                "source_frame_id": frame_id,
                "time": observation_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "pose": {
                    "x": float(pose.get("x", 0.0) or 0.0),
                    "y": float(pose.get("y", 0.0) or 0.0),
                    "z": float(pose.get("z", 0.0) or 0.0),
                    "yaw": float(pose.get("task_yaw", 0.0) or 0.0),
                },
                "detections": perception_candidates,
                "target_house_id": target_house_id,
                "current_house_id": current_house_id,
            },
        )

    candidate_target_scores = fusion_result.get("candidate_target_scores", [])
    if isinstance(candidate_target_scores, list) and candidate_target_scores:
        top_candidates: List[Dict[str, Any]] = []
        for item in candidate_target_scores[:3]:
            if not isinstance(item, dict):
                continue
            item_xyxy = item.get("xyxy", []) if isinstance(item.get("xyxy"), list) else []
            bbox_payload: Dict[str, Any] = {}
            if len(item_xyxy) == 4:
                bbox_payload = {
                    "x1": float(item_xyxy[0]),
                    "y1": float(item_xyxy[1]),
                    "x2": float(item_xyxy[2]),
                    "y2": float(item_xyxy[3]),
                    "width": float(item_xyxy[2]) - float(item_xyxy[0]),
                    "height": float(item_xyxy[3]) - float(item_xyxy[1]),
                }
            top_candidates.append(
                {
                    "candidate_id": str(item.get("candidate_id", "") or ""),
                    "class_name": str(item.get("class_name", "") or ""),
                    "confidence": float(item.get("confidence", 0.0) or 0.0),
                    "bbox": bbox_payload,
                    "target_match_score": float(item.get("candidate_target_match_score", 0.0) or 0.0),
                    "association_confidence": float(item.get("candidate_total_score", 0.0) or 0.0),
                    "center_x_norm": item.get("center_x_norm"),
                    "entry_distance_cm": float(item.get("entry_distance_cm", 0.0) or 0.0),
                    "sector": sector_id,
                    "is_target_house_entry": bool(item.get("candidate_is_target_house_entry", False)),
                }
            )
        if top_candidates:
            store.set_top_candidates(target_house_id, top_candidates)

    if bool(target_context.get("target_house_in_fov", False)):
        store.update_sector(
            target_house_id,
            sector_id,
            observed=True,
            best_entry_state=infer_observation_result(target_conditioned_state),
            best_target_conditioned_subgoal=target_conditioned_subgoal,
            best_target_match_score=float((fusion_result.get("best_target_candidate") or {}).get("candidate_target_match_score") or 0.0),
        )

    best_target_candidate = fusion_result.get("best_target_candidate", {}) if isinstance(fusion_result.get("best_target_candidate"), dict) else {}
    candidate_id_value = best_target_candidate.get("candidate_id", "")
    candidate_id = ""
    if candidate_id_value is not None:
        candidate_id = str(candidate_id_value).strip()
    planner_patch: Dict[str, Any] = {
        "target_house_id": target_house_id,
        "current_house_id": current_house_id,
        "current_best_entry_id": candidate_id,
        "decision_hint": target_conditioned_subgoal,
    }
    if candidate_id:
        memory = store.get_house_memory(target_house_id, ensure=True) or {}
        existing_attempt_count = 0
        semantic_memory = memory.get("semantic_memory", {}) if isinstance(memory.get("semantic_memory"), dict) else {}
        existing_entries = semantic_memory.get("candidate_entries", []) if isinstance(semantic_memory.get("candidate_entries"), list) else []
        for entry in existing_entries:
            if isinstance(entry, dict) and str(entry.get("entry_id") or entry.get("candidate_id") or "") == candidate_id:
                try:
                    existing_attempt_count = int(entry.get("attempt_count", 0) or 0)
                except Exception:
                    existing_attempt_count = 0
                break
        house_center_x = float(house_meta.get("center_x", 0.0) or 0.0)
        house_center_y = float(house_meta.get("center_y", 0.0) or 0.0)
        association_evidence = build_association_evidence(best_target_candidate)
        store.upsert_candidate_entry(
            target_house_id,
            {
                "entry_id": candidate_id,
                "candidate_id": candidate_id,
                "entry_type": best_target_candidate.get("class_name"),
                "semantic_class": best_target_candidate.get("class_name"),
                "source_frames": [frame_id] if frame_id else [],
                "bbox_history": [
                    build_bbox_history_payload(
                        best_target_candidate,
                        frame_id=frame_id,
                        observation_time=observation_time,
                    )
                ],
                "target_match_score": float(best_target_candidate.get("candidate_target_match_score", 0.0) or 0.0),
                "association_confidence": float(best_target_candidate.get("candidate_total_score", 0.0) or best_target_candidate.get("candidate_target_match_score", 0.0) or 0.0),
                "association_evidence": association_evidence,
                "associated_house_id": target_house_id,
                "world_position": {
                    "x": house_center_x,
                    "y": house_center_y,
                    "z": 0.0,
                    "source": "associated_house_center_proxy",
                },
                "sector": sector_id,
                "entry_state": infer_candidate_entry_status(
                    candidate=best_target_candidate,
                    target_conditioned_state=target_conditioned_state,
                ),
                "distance_cm": float(best_target_candidate.get("entry_distance_cm", 0.0) or 0.0),
                "opening_width_cm": float(best_target_candidate.get("opening_width_cm", 0.0) or 0.0),
                "center_x_norm": best_target_candidate.get("center_x_norm"),
                "candidate_total_score": float(best_target_candidate.get("candidate_total_score", 0.0) or 0.0),
                "status": infer_candidate_entry_status(
                    candidate=best_target_candidate,
                    target_conditioned_state=target_conditioned_state,
                ),
                "is_best_candidate": True,
                "is_searched": target_conditioned_subgoal == "cross_target_entry",
                "is_entered": target_conditioned_subgoal == "cross_target_entry",
                "observation_increment": 1,
                "attempt_count": existing_attempt_count + 1,
                "last_checked_time": datetime.now().timestamp(),
                "last_seen_time": observation_time or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        store.update_semantic_memory(target_house_id, {"last_best_entry_id": candidate_id})
    store.update_perimeter_coverage(
        target_house_id,
        target_context,
        visit_time=datetime.now().timestamp(),
    )
    store.set_planner_context(planner_patch)

    store.append_episodic_event(
        target_house_id,
        {
            "event_type": "fusion_decision",
            "house_id": target_house_id,
            "entry_id": candidate_id or "",
            "sector": sector_id,
            "details": {
                "target_conditioned_state": target_conditioned_state,
                "target_conditioned_subgoal": target_conditioned_subgoal,
                "target_conditioned_action_hint": target_conditioned_action,
                "last_action": last_action,
                "best_candidate_id": candidate_id or "",
                "best_candidate_class": str(best_target_candidate.get("class_name") or ""),
                "best_candidate_status": infer_candidate_entry_status(
                    candidate=best_target_candidate,
                    target_conditioned_state=target_conditioned_state,
                ) if candidate_id else "",
                "best_candidate_target_match_score": float(best_target_candidate.get("candidate_target_match_score", 0.0) or 0.0),
                "best_candidate_association_confidence": float(
                    best_target_candidate.get("candidate_total_score", 0.0)
                    or best_target_candidate.get("candidate_target_match_score", 0.0)
                    or 0.0
                ),
                "association_evidence": build_association_evidence(best_target_candidate) if candidate_id else {
                    "distance_score": 0.0,
                    "view_consistency_score": 0.0,
                    "appearance_score": 0.0,
                    "language_score": 0.0,
                    "geometry_score": 0.0,
                    "memory_similarity_score": 0.0,
                },
                "bbox_history_item": build_bbox_history_payload(
                    best_target_candidate,
                    frame_id=frame_id,
                    observation_time=observation_time,
                ) if candidate_id else {},
            },
        },
    )

    store.save()
    updated_memory = store.get_house_memory(target_house_id, ensure=True) or {}
    return {
        "status": "ok",
        "memory_path": store.path,
        "house_id": target_house_id,
        "sector_id": sector_id if bool(target_context.get("target_house_in_fov", False)) else None,
        "entry_search_status": entry_search_status,
        "house_registry_entry": copy.deepcopy(store.to_dict().get("house_registry", {}).get(target_house_id, {})),
        "planner_context": copy.deepcopy(store.to_dict().get("planner_context", {})),
        "semantic_memory": copy.deepcopy(updated_memory.get("semantic_memory", {})),
    }


def score_candidate_for_target_house(
    *,
    detection: Dict[str, Any],
    semantic_depth_assessment: Optional[Dict[str, Any]],
    target_context: Dict[str, Any],
    image_shape: Tuple[int, int],
) -> Dict[str, Any]:
    image_width = int(image_shape[1]) if len(image_shape) >= 2 else 0
    center_x_norm = detection_center_x_norm(detection, image_width)
    image_side = side_from_x_norm(center_x_norm)
    expected_x = target_context.get("target_house_expected_image_x")
    expected_side = str(target_context.get("target_house_expected_side") or "unknown")
    target_in_fov = bool(target_context.get("target_house_in_fov", False))
    target_distance_cm = target_context.get("target_house_distance_cm")

    bearing_score = 0.0
    if expected_x is not None and center_x_norm is not None and target_in_fov:
        bearing_score = clamp(1.0 - abs(float(center_x_norm) - float(expected_x)) / 0.5, 0.0, 1.0)

    image_side_score = 0.0
    if expected_side == "center":
        image_side_score = clamp(1.0 - abs(float(center_x_norm or 0.5) - 0.5) / 0.25, 0.0, 1.0)
    elif expected_side in {"left", "right"}:
        image_side_score = 1.0 if image_side == expected_side else 0.0
    elif expected_side == "out_of_view":
        image_side_score = 0.0

    range_score = 0.5
    if semantic_depth_assessment and target_distance_cm is not None:
        try:
            entry_distance_cm = float(semantic_depth_assessment.get("entry_distance_cm"))
            target_distance_val = float(target_distance_cm)
            tolerance_cm = max(1200.0, target_distance_val * 0.75)
            range_score = clamp(1.0 - abs(entry_distance_cm - target_distance_val) / tolerance_cm, 0.0, 1.0)
        except Exception:
            range_score = 0.5

    normalized_name = normalize_class_name(detection.get("class_name"))
    class_prior_norm = clamp(class_priority(normalized_name) / 3.0, 0.0, 1.0)
    semantic_score = clamp(
        0.65 * class_prior_norm + 0.35 * float(detection.get("confidence", 0.0)),
        0.0,
        1.0,
    )

    geometry_score = 0.0
    entry_distance_cm = None
    opening_width_cm = None
    traversable = False
    crossing_ready = False
    depth_confidence = 0.0
    if semantic_depth_assessment:
        entry_distance_cm = float(semantic_depth_assessment.get("entry_distance_cm", 0.0))
        opening_width_cm = float(semantic_depth_assessment.get("opening_width_cm", 0.0))
        traversable = bool(semantic_depth_assessment.get("traversable", False))
        crossing_ready = bool(semantic_depth_assessment.get("crossing_ready", False))
        depth_confidence = float(semantic_depth_assessment.get("confidence", 0.0))
        geometry_score = clamp(
            0.55 * depth_confidence
            + 0.30 * (1.0 if traversable else 0.0)
            + 0.15 * (1.0 if crossing_ready else 0.0),
            0.0,
            1.0,
        )

    target_match_score = 0.0
    if target_in_fov:
        target_match_score = clamp(
            0.45 * bearing_score
            + 0.20 * image_side_score
            + 0.35 * range_score,
            0.0,
            1.0,
        )

    candidate_total_score = clamp(
        0.50 * target_match_score
        + 0.25 * semantic_score
        + 0.25 * geometry_score,
        0.0,
        1.0,
    )
    candidate_is_target_house_entry = bool(
        normalized_name in DOORLIKE_CLASSES
        and target_match_score >= TARGET_MATCH_MIN_TARGET_SCORE
        and candidate_total_score >= TARGET_MATCH_MIN_SCORE
    )

    return {
        "candidate_id": int(detection.get("candidate_id", -1)),
        "class_name": detection.get("class_name"),
        "class_name_normalized": normalized_name,
        "confidence": float(detection.get("confidence", 0.0)),
        "xyxy": [float(v) for v in detection.get("xyxy", [])] if isinstance(detection.get("xyxy"), list) else [],
        "bbox": bbox_from_xyxy(detection.get("xyxy", [])),
        "center_x_norm": center_x_norm,
        "candidate_image_side": image_side,
        "candidate_target_match_score": target_match_score,
        "candidate_semantic_score": semantic_score,
        "candidate_geometry_score": geometry_score,
        "candidate_total_score": candidate_total_score,
        "candidate_target_score_components": {
            "bearing_score": bearing_score,
            "image_side_score": image_side_score,
            "range_score": range_score,
            "expected_image_x": expected_x,
            "expected_side": expected_side,
        },
        "candidate_house_id": target_context.get("target_house_id") if candidate_is_target_house_entry else None,
        "candidate_is_target_house_entry": candidate_is_target_house_entry,
        "entry_distance_cm": entry_distance_cm,
        "opening_width_cm": opening_width_cm,
        "traversable": traversable,
        "crossing_ready": crossing_ready,
        "semantic_depth_assessment": semantic_depth_assessment,
    }


def load_target_house_memory(target_house_id: Optional[str]) -> Dict[str, Any]:
    house_id = str(target_house_id or "").strip()
    if not house_id:
        return {}
    try:
        store = EntrySearchMemoryStore()
        store.load()
        memory = store.get_house_memory(house_id, ensure=False)
        return copy.deepcopy(memory) if isinstance(memory, dict) else {}
    except Exception:
        return {}


def score_memory_candidate_similarity(
    candidate: Dict[str, Any],
    memory_entry: Dict[str, Any],
) -> float:
    candidate_name = normalize_class_name(candidate.get("class_name"))
    memory_name = normalize_class_name(
        memory_entry.get("semantic_class") or memory_entry.get("class_name")
    )
    if candidate_name == memory_name and candidate_name:
        class_score = 1.0
    elif candidate_name in DOORLIKE_CLASSES and memory_name in DOORLIKE_CLASSES:
        class_score = 0.75
    elif candidate_name in WINDOW_CLASSES and memory_name in WINDOW_CLASSES:
        class_score = 0.7
    else:
        class_score = 0.0

    center_score = 0.5
    try:
        candidate_center = float(candidate.get("center_x_norm"))
        memory_center = float(memory_entry.get("center_x_norm"))
        center_score = clamp(1.0 - abs(candidate_center - memory_center) / 0.35, 0.0, 1.0)
    except Exception:
        center_score = 0.5

    distance_score = 0.5
    try:
        candidate_distance = float(candidate.get("entry_distance_cm"))
        memory_distance = float(memory_entry.get("distance_cm"))
        tolerance_cm = max(150.0, max(candidate_distance, memory_distance) * 0.5)
        distance_score = clamp(
            1.0 - abs(candidate_distance - memory_distance) / tolerance_cm,
            0.0,
            1.0,
        )
    except Exception:
        distance_score = 0.5

    target_score = 0.5
    try:
        candidate_target_score = float(candidate.get("candidate_target_match_score"))
        memory_target_score = float(memory_entry.get("target_match_score"))
        target_score = clamp(
            1.0 - abs(candidate_target_score - memory_target_score) / 0.45,
            0.0,
            1.0,
        )
    except Exception:
        target_score = 0.5

    return clamp(
        0.35 * class_score
        + 0.25 * center_score
        + 0.25 * distance_score
        + 0.15 * target_score,
        0.0,
        1.0,
    )


def compute_sector_repeat_penalty(
    semantic_memory: Dict[str, Any],
    sector_id: Optional[str],
    candidate: Dict[str, Any],
) -> float:
    if not isinstance(semantic_memory, dict):
        return 0.0
    sector_key = str(sector_id or "").strip()
    if not sector_key:
        return 0.0
    searched_sectors = semantic_memory.get("searched_sectors", {})
    if not isinstance(searched_sectors, dict):
        return 0.0
    sector = searched_sectors.get(sector_key)
    if not isinstance(sector, dict):
        return 0.0

    try:
        observation_count = int(sector.get("observation_count", 0) or 0)
    except Exception:
        observation_count = 0
    if observation_count < 2:
        return 0.0

    best_entry_state = str(sector.get("best_entry_state", "") or "").strip()
    best_subgoal = str(sector.get("best_target_conditioned_subgoal", "") or "").strip()
    try:
        best_target_match_score = float(sector.get("best_target_match_score", 0.0) or 0.0)
    except Exception:
        best_target_match_score = 0.0
    try:
        candidate_target_score = float(candidate.get("candidate_target_match_score", 0.0) or 0.0)
    except Exception:
        candidate_target_score = 0.0

    low_yield = (
        best_entry_state in MEMORY_LOW_YIELD_ENTRY_STATES
        or best_subgoal in {"keep_search_target_house", "ignore_non_target_entry"}
    ) and best_target_match_score < 0.6
    if not low_yield:
        return 0.0
    if candidate_target_score >= 0.72:
        return 0.0

    penalty = min(0.12, 0.03 * float(observation_count - 1))
    if normalize_class_name(candidate.get("class_name")) in WINDOW_CLASSES:
        penalty += 0.02
    return clamp(penalty, 0.0, 0.15)


def apply_memory_aware_candidate_adjustments(
    candidate_target_scores: List[Dict[str, Any]],
    *,
    target_context: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    target_house_id = str(target_context.get("target_house_id") or "").strip()
    house_memory = load_target_house_memory(target_house_id)
    semantic_memory = house_memory.get("semantic_memory", {}) if isinstance(house_memory.get("semantic_memory"), dict) else {}
    sector_id = infer_target_sector_id(target_context)
    candidate_entries = semantic_memory.get("candidate_entries", []) if isinstance(semantic_memory.get("candidate_entries"), list) else []
    last_best_entry_id = str(semantic_memory.get("last_best_entry_id", "") or "").strip()

    adjusted_scores: List[Dict[str, Any]] = []
    sector_penalty_applied = False
    candidate_history_used = False
    last_best_tracking_used = False

    for item in candidate_target_scores:
        candidate = copy.deepcopy(item)
        raw_total_score = float(candidate.get("candidate_total_score", 0.0) or 0.0)
        sector_penalty = compute_sector_repeat_penalty(semantic_memory, sector_id, candidate)
        sector_penalty_applied = sector_penalty_applied or sector_penalty > 0.0

        best_similarity = 0.0
        best_memory_entry: Dict[str, Any] = {}
        for memory_entry in candidate_entries:
            if not isinstance(memory_entry, dict):
                continue
            similarity = score_memory_candidate_similarity(candidate, memory_entry)
            if similarity > best_similarity:
                best_similarity = similarity
                best_memory_entry = memory_entry

        history_penalty = 0.0
        history_boost = 0.0
        tracking_boost = 0.0
        memory_status = ""
        memory_candidate_id = ""
        if best_similarity >= 0.55 and best_memory_entry:
            candidate_history_used = True
            memory_status = str(best_memory_entry.get("status", "") or "").strip()
            memory_candidate_id = str(best_memory_entry.get("candidate_id", "") or "").strip()
            if memory_status == "window_rejected":
                history_penalty = 0.18 * best_similarity
            elif memory_status == "non_target":
                history_penalty = 0.16 * best_similarity
            elif memory_status == "blocked_confirmed":
                history_penalty = 0.14 * best_similarity
            elif memory_status == "blocked_temporary":
                history_boost = 0.05 * best_similarity
            elif memory_status == "approachable":
                history_boost = 0.10 * best_similarity
            elif memory_status == "unverified" and float(candidate.get("candidate_target_match_score", 0.0) or 0.0) >= 0.65:
                history_boost = 0.03 * best_similarity

            if memory_candidate_id and memory_candidate_id == last_best_entry_id and memory_status in MEMORY_REVISIT_CANDIDATE_STATUSES:
                tracking_boost = 0.06 * best_similarity
                last_best_tracking_used = True

        adjusted_total_score = clamp(
            raw_total_score - sector_penalty - history_penalty + history_boost + tracking_boost,
            0.0,
            1.0,
        )
        normalized_name = normalize_class_name(candidate.get("class_name"))
        candidate["candidate_total_score_raw"] = round(raw_total_score, 4)
        candidate["memory_sector_penalty"] = round(sector_penalty, 4)
        candidate["memory_history_penalty"] = round(history_penalty, 4)
        candidate["memory_history_boost"] = round(history_boost, 4)
        candidate["memory_tracking_boost"] = round(tracking_boost, 4)
        candidate["memory_best_similarity"] = round(best_similarity, 4)
        candidate["memory_best_match_status"] = memory_status or None
        candidate["memory_best_match_candidate_id"] = memory_candidate_id or None
        candidate["candidate_total_score"] = round(adjusted_total_score, 4)
        candidate["candidate_is_target_house_entry"] = bool(
            normalized_name in DOORLIKE_CLASSES
            and float(candidate.get("candidate_target_match_score", 0.0) or 0.0) >= TARGET_MATCH_MIN_TARGET_SCORE
            and adjusted_total_score >= TARGET_MATCH_MIN_SCORE
        )
        adjusted_scores.append(candidate)

    memory_guidance = {
        "target_house_id": target_house_id or None,
        "memory_available": bool(house_memory),
        "sector_id": sector_id if target_house_id else None,
        "sector_penalty_applied": sector_penalty_applied,
        "candidate_history_used": candidate_history_used,
        "last_best_entry_id": last_best_entry_id or None,
        "last_best_tracking_used": last_best_tracking_used,
    }
    return adjusted_scores, memory_guidance


def find_best_memory_entry_match(
    candidate: Dict[str, Any],
    semantic_memory: Dict[str, Any],
) -> Tuple[Dict[str, Any], float]:
    candidate_entries = semantic_memory.get("candidate_entries", []) if isinstance(semantic_memory, dict) else []
    if not isinstance(candidate_entries, list):
        return {}, 0.0
    best_entry: Dict[str, Any] = {}
    best_similarity = 0.0
    for memory_entry in candidate_entries:
        if not isinstance(memory_entry, dict):
            continue
        similarity = score_memory_candidate_similarity(candidate, memory_entry)
        if similarity > best_similarity:
            best_similarity = similarity
            best_entry = memory_entry
    return best_entry, float(best_similarity)


def preferred_alternate_sectors(current_sector_id: Optional[str]) -> List[str]:
    sector = str(current_sector_id or "").strip()
    mapping = {
        "front_center": ["front_left", "front_right", "left_side", "right_side"],
        "front_left": ["front_right", "left_side", "front_center", "right_side"],
        "front_right": ["front_left", "right_side", "front_center", "left_side"],
        "left_side": ["front_left", "front_center", "front_right", "right_side"],
        "right_side": ["front_right", "front_center", "front_left", "left_side"],
    }
    preferred = [item for item in mapping.get(sector, []) if item in DEFAULT_SECTOR_IDS]
    for sector_id in DEFAULT_SECTOR_IDS:
        if sector_id != sector and sector_id not in preferred:
            preferred.append(sector_id)
    return preferred


def select_alternate_sector(
    semantic_memory: Dict[str, Any],
    current_sector_id: Optional[str],
) -> Optional[str]:
    searched_sectors = semantic_memory.get("searched_sectors", {}) if isinstance(semantic_memory, dict) else {}
    if not isinstance(searched_sectors, dict):
        searched_sectors = {}
    candidates: List[Tuple[int, int, float, str]] = []
    for sector_id in preferred_alternate_sectors(current_sector_id):
        sector = searched_sectors.get(sector_id, {})
        if not isinstance(sector, dict):
            sector = {}
        observed = bool(sector.get("observed", False))
        try:
            observation_count = int(sector.get("observation_count", 0) or 0)
        except Exception:
            observation_count = 0
        try:
            best_target_match_score = float(sector.get("best_target_match_score", 0.0) or 0.0)
        except Exception:
            best_target_match_score = 0.0
        unexplored_priority = 0 if not observed else 1
        candidates.append((unexplored_priority, observation_count, -best_target_match_score, sector_id))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][3]


def action_for_sector_shift(
    current_sector_id: Optional[str],
    target_sector_id: Optional[str],
    target_bearing_deg: Optional[float],
) -> str:
    current_sector = str(current_sector_id or "").strip()
    target_sector = str(target_sector_id or "").strip()
    if target_sector in {"front_left", "left_side"}:
        return "yaw_left"
    if target_sector in {"front_right", "right_side"}:
        return "yaw_right"
    if target_sector == "front_center":
        if current_sector in {"front_right", "right_side"}:
            return "yaw_left"
        if current_sector in {"front_left", "left_side"}:
            return "yaw_right"
    if target_bearing_deg is not None:
        return "yaw_left" if float(target_bearing_deg) > 0.0 else "yaw_right"
    return "hold"


def apply_memory_aware_target_decision(
    *,
    target_context: Dict[str, Any],
    target_state: str,
    target_subgoal: str,
    target_action: str,
    target_reason: str,
    best_target_candidate: Dict[str, Any],
) -> Tuple[str, str, str, Dict[str, Any]]:
    target_house_id = str(target_context.get("target_house_id") or "").strip()
    current_sector_id = infer_target_sector_id(target_context)
    guidance: Dict[str, Any] = {
        "memory_available": False,
        "decision_override_applied": False,
        "override_reason": "",
        "target_house_id": target_house_id or None,
        "current_sector_id": current_sector_id or None,
        "alternate_sector_id": None,
        "best_memory_status": None,
        "best_memory_similarity": 0.0,
        "blocked_attempt_count": 0,
        "target_reason": target_reason,
    }
    if not target_house_id:
        return target_state, target_subgoal, target_action, guidance

    house_memory = load_target_house_memory(target_house_id)
    semantic_memory = house_memory.get("semantic_memory", {}) if isinstance(house_memory.get("semantic_memory"), dict) else {}
    if not semantic_memory:
        return target_state, target_subgoal, target_action, guidance

    guidance["memory_available"] = True
    searched_sectors = semantic_memory.get("searched_sectors", {}) if isinstance(semantic_memory.get("searched_sectors"), dict) else {}
    current_sector = searched_sectors.get(current_sector_id, {}) if current_sector_id else {}
    if not isinstance(current_sector, dict):
        current_sector = {}

    best_memory_entry, best_similarity = find_best_memory_entry_match(best_target_candidate, semantic_memory)
    memory_status = str(best_memory_entry.get("status", "") or "").strip()
    guidance["best_memory_status"] = memory_status or None
    guidance["best_memory_similarity"] = round(best_similarity, 4)
    try:
        guidance["blocked_attempt_count"] = int(best_memory_entry.get("attempt_count", 0) or 0)
    except Exception:
        guidance["blocked_attempt_count"] = 0

    try:
        sector_observation_count = int(current_sector.get("observation_count", 0) or 0)
    except Exception:
        sector_observation_count = 0
    sector_entry_state = str(current_sector.get("best_entry_state", "") or "").strip()
    sector_subgoal = str(current_sector.get("best_target_conditioned_subgoal", "") or "").strip()
    try:
        sector_target_match_score = float(current_sector.get("best_target_match_score", 0.0) or 0.0)
    except Exception:
        sector_target_match_score = 0.0

    low_yield_sector = (
        target_state == "target_house_entry_visible"
        and sector_observation_count >= 2
        and (
            sector_entry_state in MEMORY_LOW_YIELD_ENTRY_STATES
            or sector_subgoal in {"keep_search_target_house", "ignore_non_target_entry"}
        )
        and sector_target_match_score < 0.6
    )
    if low_yield_sector:
        alternate_sector_id = select_alternate_sector(semantic_memory, current_sector_id)
        if alternate_sector_id:
            guidance["decision_override_applied"] = True
            guidance["override_reason"] = "low_yield_sector_shift"
            guidance["alternate_sector_id"] = alternate_sector_id
            target_subgoal = "keep_search_target_house"
            target_action = action_for_sector_shift(
                current_sector_id=current_sector_id,
                target_sector_id=alternate_sector_id,
                target_bearing_deg=target_context.get("target_house_bearing_deg"),
            )
            guidance["target_reason"] = (
                f"{target_reason}; repeated low-yield observations in sector {current_sector_id or 'unknown'} "
                f"(count={sector_observation_count}) so search shifts toward {alternate_sector_id}"
            )
            return target_state, target_subgoal, target_action, guidance

    persistent_blocked = (
        target_state == "target_house_entry_blocked"
        and best_similarity >= 0.55
        and memory_status in {"blocked_temporary", "blocked_confirmed"}
        and guidance["blocked_attempt_count"] >= 2
    )
    if persistent_blocked:
        alternate_sector_id = select_alternate_sector(semantic_memory, current_sector_id)
        if alternate_sector_id:
            guidance["decision_override_applied"] = True
            guidance["override_reason"] = "persistent_blocked_shift"
            guidance["alternate_sector_id"] = alternate_sector_id
            target_subgoal = "keep_search_target_house"
            target_action = action_for_sector_shift(
                current_sector_id=current_sector_id,
                target_sector_id=alternate_sector_id,
                target_bearing_deg=target_context.get("target_house_bearing_deg"),
            )
            guidance["target_reason"] = (
                f"{target_reason}; similar blocked target-entry candidate was already checked "
                f"{guidance['blocked_attempt_count']} times, so search shifts toward {alternate_sector_id}"
            )
            return target_state, target_subgoal, target_action, guidance

    return target_state, target_subgoal, target_action, guidance


def box_iou_xyxy(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-6:
        return 0.0
    return float(inter / union)


def center_distance_score(box_a: List[float], box_b: List[float], image_shape: Tuple[int, int]) -> float:
    height, width = image_shape[:2]
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    acx = 0.5 * (ax1 + ax2)
    acy = 0.5 * (ay1 + ay2)
    bcx = 0.5 * (bx1 + bx2)
    bcy = 0.5 * (by1 + by2)
    diag = math.hypot(max(1.0, float(width)), max(1.0, float(height)))
    dist = math.hypot(acx - bcx, acy - bcy)
    return max(0.0, 1.0 - float(dist / max(1.0, diag)))


def candidate_box_xyxy(candidate: Dict[str, Any]) -> List[float]:
    bbox = candidate.get("bbox", {}) if isinstance(candidate, dict) else {}
    x = float(bbox.get("x", 0.0))
    y = float(bbox.get("y", 0.0))
    w = float(bbox.get("width", 0.0))
    h = float(bbox.get("height", 0.0))
    return [x, y, x + w, y + h]


def scale_xyxy_between_images(
    box_xyxy: List[float],
    src_shape: Tuple[int, int],
    dst_shape: Tuple[int, int],
) -> List[int]:
    if len(box_xyxy) != 4:
        return [0, 0, 0, 0]
    src_h, src_w = src_shape[:2]
    dst_h, dst_w = dst_shape[:2]
    if src_h <= 0 or src_w <= 0 or dst_h <= 0 or dst_w <= 0:
        return [0, 0, 0, 0]
    sx = float(dst_w) / float(src_w)
    sy = float(dst_h) / float(src_h)
    x1 = int(round(float(box_xyxy[0]) * sx))
    y1 = int(round(float(box_xyxy[1]) * sy))
    x2 = int(round(float(box_xyxy[2]) * sx))
    y2 = int(round(float(box_xyxy[3]) * sy))
    x1 = max(0, min(dst_w - 1, x1))
    y1 = max(0, min(dst_h - 1, y1))
    x2 = max(x1 + 1, min(dst_w, x2))
    y2 = max(y1 + 1, min(dst_h, y2))
    return [x1, y1, x2, y2]


def summarize_semantic_region_depth(
    *,
    depth_cm: np.ndarray,
    detection: Dict[str, Any],
    rgb_shape: Tuple[int, int],
    hfov_deg: float,
    traversable_min_width_cm: float,
    traversable_min_clearance_cm: float,
    traversable_min_depth_gain_cm: float,
    crossing_ready_max_distance_cm: float,
) -> Optional[Dict[str, Any]]:
    if depth_cm is None or not isinstance(depth_cm, np.ndarray):
        return None
    det_box = detection.get("xyxy", [])
    if not isinstance(det_box, list) or len(det_box) != 4:
        return None

    depth_h, depth_w = depth_cm.shape[:2]
    rgb_h, rgb_w = rgb_shape[:2]
    dx1, dy1, dx2, dy2 = scale_xyxy_between_images(det_box, (rgb_h, rgb_w), (depth_h, depth_w))
    if dx2 - dx1 < 4 or dy2 - dy1 < 8:
        return None

    box_w = dx2 - dx1
    box_h = dy2 - dy1
    shrink_x = max(2, int(box_w * 0.12))
    shrink_y = max(2, int(box_h * 0.08))
    ix1 = min(dx2 - 1, dx1 + shrink_x)
    iy1 = min(dy2 - 1, dy1 + shrink_y)
    ix2 = max(ix1 + 1, dx2 - shrink_x)
    iy2 = max(iy1 + 1, dy2 - shrink_y)

    region = depth_cm[iy1:iy2, ix1:ix2]
    region_values = region[np.isfinite(region)]
    if region_values.size < 16:
        return None

    far_percentile = float(np.nanpercentile(region_values, 72))
    opening_mask = np.isfinite(region) & (region >= far_percentile)
    opening_values = region[opening_mask]
    if opening_values.size < 8:
        opening_values = region_values
        opening_mask = np.isfinite(region)
    if opening_values.size < 8:
        return None

    pad = max(12, int(max(box_w, box_h) * 0.18))
    ring_mask = extract_ring_mask(depth_cm.shape, x=dx1, y=dy1, width=box_w, height=box_h, pad=pad)
    surround_values = depth_cm[ring_mask]
    surround_values = surround_values[np.isfinite(surround_values)]
    if surround_values.size < 12:
        surround_values = region_values

    opening_depth_cm = float(np.nanmean(opening_values))
    surround_depth_cm = float(np.nanmean(surround_values))
    depth_gain_cm = max(0.0, opening_depth_cm - surround_depth_cm)

    lower_cut = iy1 + int((iy2 - iy1) * 0.55)
    lower_region = depth_cm[lower_cut:iy2, ix1:ix2]
    lower_values = lower_region[np.isfinite(lower_region)]
    if lower_values.size < 8:
        lower_values = opening_values
    clearance_depth_cm = float(np.nanmean(lower_values))

    column_support = opening_mask.sum(axis=0)
    supported_columns = column_support >= max(1, int(opening_mask.shape[0] * 0.18))
    opening_width_px = int(np.count_nonzero(supported_columns))
    if opening_width_px <= 0:
        opening_width_px = max(1, ix2 - ix1)
    opening_width_ratio = opening_width_px / float(max(1, depth_w))
    opening_width_cm = estimate_opening_width_cm(
        opening_depth_cm=opening_depth_cm,
        width_ratio=opening_width_ratio,
        hfov_deg=hfov_deg,
    )

    center_x_norm = ((dx1 + dx2) * 0.5) / float(max(1, depth_w))
    center_y_norm = ((dy1 + dy2) * 0.5) / float(max(1, depth_h))
    center_bias = 1.0 - min(1.0, abs(center_x_norm - 0.5) / 0.5)
    traversable = bool(
        opening_width_cm >= traversable_min_width_cm
        and clearance_depth_cm >= traversable_min_clearance_cm
        and depth_gain_cm >= traversable_min_depth_gain_cm
    )
    crossing_ready = bool(traversable and opening_depth_cm <= float(crossing_ready_max_distance_cm))

    depth_score = clamp(depth_gain_cm / 320.0, 0.0, 1.0)
    width_score = clamp((opening_width_cm - traversable_min_width_cm) / 120.0, 0.0, 1.0)
    clearance_score = clamp((clearance_depth_cm - traversable_min_clearance_cm) / 250.0, 0.0, 1.0)
    confidence = clamp(
        depth_score * 0.42
        + width_score * 0.23
        + clearance_score * 0.20
        + center_bias * 0.10
        + (0.05 if traversable else 0.0),
        0.0,
        1.0,
    )

    return {
        "candidate_id": "semantic_region_depth",
        "source": "semantic_region",
        "bbox": {
            "x": float(det_box[0]),
            "y": float(det_box[1]),
            "width": float(det_box[2] - det_box[0]),
            "height": float(det_box[3] - det_box[1]),
        },
        "depth_bbox": {
            "x": dx1,
            "y": dy1,
            "width": box_w,
            "height": box_h,
        },
        "rgb_bbox_xyxy": [float(v) for v in det_box],
        "center_x_norm": center_x_norm,
        "center_y_norm": center_y_norm,
        "entry_distance_cm": opening_depth_cm,
        "surrounding_depth_cm": surround_depth_cm,
        "clearance_depth_cm": clearance_depth_cm,
        "depth_gain_cm": depth_gain_cm,
        "opening_width_cm": opening_width_cm,
        "traversable": traversable,
        "crossing_ready": crossing_ready,
        "confidence": confidence,
        "rationale": (
            f"semantic-guided depth check: distance={opening_depth_cm:.0f}cm, "
            f"width={opening_width_cm:.0f}cm, clearance={clearance_depth_cm:.0f}cm, "
            f"gain={depth_gain_cm:.0f}cm"
        ),
    }


def class_priority(normalized_name: str) -> int:
    if normalized_name in OPEN_DOOR_CLASSES:
        return 3
    if normalized_name in DOORLIKE_CLASSES:
        return 2
    if normalized_name in WINDOW_CLASSES:
        return 1
    return 0


def run_yolo_inference(
    *,
    rgb_bgr: np.ndarray,
    weights: Path,
    output_dir: Path,
    conf: float,
    imgsz: int,
    device: str,
) -> Dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Fusion analysis requires ultralytics. Install it with: pip install -U ultralytics") from exc

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    ensure_dir(output_dir)

    model = YOLO(str(weights))
    results = model.predict(
        source=rgb_bgr,
        conf=float(conf),
        imgsz=int(imgsz),
        device=str(device),
        verbose=False,
    )
    if not results:
        raise RuntimeError("YOLO model returned no results.")

    result = results[0]
    annotated = result.plot()
    annotated_path = output_dir / "yolo_annotated.jpg"
    cv2.imwrite(str(annotated_path), annotated)

    detections: List[Dict[str, Any]] = []
    names = result.names if isinstance(result.names, dict) else {}
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf_val = float(box.conf.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            class_name = str(names.get(cls_id, str(cls_id)))
            detections.append(
                {
                    "class_id": cls_id,
                    "class_name": class_name,
                    "class_name_normalized": normalize_class_name(class_name),
                    "confidence": conf_val,
                    "xyxy": xyxy,
                }
            )

    detections.sort(
        key=lambda item: (
            class_priority(str(item.get("class_name_normalized", ""))),
            float(item.get("confidence", 0.0)),
        ),
        reverse=True,
    )

    json_path = output_dir / "yolo_result.json"
    txt_path = output_dir / "yolo_result.txt"
    json_path.write_text(
        json.dumps(
            {
                "weights": str(weights),
                "annotated_path": str(annotated_path),
                "num_detections": len(detections),
                "detections": detections,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    lines = [
        (
            f"{idx:02d} class={det['class_name']} "
            f"conf={float(det['confidence']):.4f} "
            f"xyxy=({det['xyxy'][0]:.1f}, {det['xyxy'][1]:.1f}, {det['xyxy'][2]:.1f}, {det['xyxy'][3]:.1f})"
        )
        for idx, det in enumerate(detections, start=1)
    ]
    if not lines:
        lines.append("No detections.")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "weights_path": str(weights),
        "annotated_path": str(annotated_path),
        "json_path": str(json_path),
        "txt_path": str(txt_path),
        "num_detections": len(detections),
        "detections": detections,
        "annotated_image": annotated,
    }


def run_depth_inference(
    *,
    depth_cm_path: Path,
    output_dir: Path,
    hfov_deg: float = 90.0,
    max_valid_depth_cm: float = 1200.0,
    front_obstacle_threshold_cm: float = 140.0,
    traversable_min_width_cm: float = 90.0,
    traversable_min_clearance_cm: float = 160.0,
    traversable_min_depth_gain_cm: float = 80.0,
) -> Dict[str, Any]:
    ensure_dir(output_dir)
    depth_cm = load_depth_image(depth_cm_path, max_valid_depth_cm=float(max_valid_depth_cm))
    preview = build_depth_preview(depth_cm, max_valid_depth_cm=float(max_valid_depth_cm))
    analysis = analyze_depth_entry(
        depth_cm=depth_cm,
        hfov_deg=float(hfov_deg),
        max_valid_depth_cm=float(max_valid_depth_cm),
        front_obstacle_threshold_cm=float(front_obstacle_threshold_cm),
        traversable_min_width_cm=float(traversable_min_width_cm),
        traversable_min_clearance_cm=float(traversable_min_clearance_cm),
        traversable_min_depth_gain_cm=float(traversable_min_depth_gain_cm),
    )
    analysis.update(
        {
            "depth_path": str(depth_cm_path),
            "hfov_deg": float(hfov_deg),
            "max_valid_depth_cm": float(max_valid_depth_cm),
        }
    )

    overlay = draw_overlay(preview, analysis)
    overlay_path = output_dir / "depth_overlay.png"
    preview_path = output_dir / "depth_preview.png"
    json_path = output_dir / "depth_result.json"
    txt_path = output_dir / "depth_summary.txt"
    cv2.imwrite(str(preview_path), preview)
    cv2.imwrite(str(overlay_path), overlay)
    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    write_text_summary(txt_path, analysis)
    return {
        "preview_path": str(preview_path),
        "overlay_path": str(overlay_path),
        "json_path": str(json_path),
        "txt_path": str(txt_path),
        "analysis": analysis,
        "overlay_image": overlay,
        "preview_image": preview,
        "depth_cm_array": depth_cm,
        "hfov_deg": float(hfov_deg),
        "thresholds": {
            "front_obstacle_threshold_cm": float(front_obstacle_threshold_cm),
            "traversable_min_width_cm": float(traversable_min_width_cm),
            "traversable_min_clearance_cm": float(traversable_min_clearance_cm),
            "traversable_min_depth_gain_cm": float(traversable_min_depth_gain_cm),
        },
    }


def choose_best_semantic_detection(detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    semantic = [det for det in detections if class_priority(str(det.get("class_name_normalized", ""))) > 0]
    return semantic[0] if semantic else None


def recommend_subgoal_and_action(
    *,
    final_state: str,
    center_x_norm: Optional[float],
    crossing_ready: bool,
    front_blocked: bool,
) -> Tuple[str, str]:
    center = 0.5 if center_x_norm is None else float(center_x_norm)
    if front_blocked or final_state == "front_blocked_detour":
        if center < 0.42:
            return "detour_left", "yaw_left"
        if center > 0.58:
            return "detour_right", "yaw_right"
        return "backoff_and_reobserve", "backward"

    if final_state in {"enterable_open_door", "enterable_door"}:
        if not crossing_ready:
            if center < 0.44:
                return "approach_entry", "yaw_left"
            if center > 0.56:
                return "approach_entry", "yaw_right"
            return "approach_entry", "forward"
        if center < 0.47:
            return "align_entry", "yaw_left"
        if center > 0.53:
            return "align_entry", "yaw_right"
        return "cross_entry", "forward"

    if final_state == "visible_but_blocked_entry":
        if center < 0.45:
            return "detour_left", "left"
        if center > 0.55:
            return "detour_right", "right"
        return "backoff_and_reobserve", "backward"

    if final_state == "window_visible_keep_search":
        if center < 0.5:
            return "keep_search", "yaw_right"
        return "keep_search", "yaw_left"

    if final_state == "geometric_opening_needs_confirmation":
        if center < 0.44:
            return "approach_entry", "yaw_left"
        if center > 0.56:
            return "approach_entry", "yaw_right"
        return "approach_entry", "forward"

    return "keep_search", "hold"


def recommend_target_conditioned_action(
    *,
    target_state: str,
    target_bearing_deg: Optional[float],
    candidate_center_x_norm: Optional[float],
    crossing_ready: bool,
) -> Tuple[str, str]:
    bearing = float(target_bearing_deg) if target_bearing_deg is not None else None
    center = 0.5 if candidate_center_x_norm is None else float(candidate_center_x_norm)
    if target_state == "target_house_not_in_view":
        if bearing is None:
            return "keep_search_target_house", "hold"
        return ("reorient_to_target_house", "yaw_left") if bearing > 0.0 else ("reorient_to_target_house", "yaw_right")
    if target_state == "non_target_house_entry_visible":
        if bearing is None or abs(bearing) < 8.0:
            return "ignore_non_target_entry", "hold"
        return ("ignore_non_target_entry", "yaw_left") if bearing > 0.0 else ("ignore_non_target_entry", "yaw_right")
    if target_state == "target_house_entry_blocked":
        if center < 0.48:
            return "detour_left_to_target_entry", "left"
        return "detour_right_to_target_entry", "right"
    if target_state == "target_house_entry_approachable":
        if crossing_ready and 0.45 <= center <= 0.55:
            return "cross_target_entry", "forward"
        if center < 0.44:
            return "approach_target_entry", "yaw_left"
        if center > 0.56:
            return "approach_target_entry", "yaw_right"
        return "approach_target_entry", "forward"
    if target_state == "target_house_entry_visible":
        if candidate_center_x_norm is None:
            if bearing is not None and abs(bearing) > 8.0:
                return ("keep_search_target_house", "yaw_left") if bearing > 0.0 else ("keep_search_target_house", "yaw_right")
            return "keep_search_target_house", "hold"
        if center < 0.44:
            return "keep_search_target_house", "yaw_left"
        if center > 0.56:
            return "keep_search_target_house", "yaw_right"
        return "keep_search_target_house", "hold"
    return "keep_search_target_house", "hold"


def match_detection_to_candidate(
    detection: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    image_shape: Tuple[int, int],
) -> Tuple[Optional[Dict[str, Any]], float, float, float]:
    best_candidate: Optional[Dict[str, Any]] = None
    best_score = -1.0
    best_iou = 0.0
    best_center = 0.0
    det_box = detection.get("xyxy", [])
    if not isinstance(det_box, list) or len(det_box) != 4:
        return None, 0.0, 0.0, 0.0
    for candidate in candidates:
        cand_box = candidate_box_xyxy(candidate)
        iou = box_iou_xyxy(det_box, cand_box)
        center = center_distance_score(det_box, cand_box, image_shape)
        score = 0.65 * iou + 0.35 * center
        if score > best_score:
            best_score = score
            best_candidate = candidate
            best_iou = iou
            best_center = center
    if best_candidate is None:
        return None, 0.0, 0.0, 0.0
    return best_candidate, float(best_score), float(best_iou), float(best_center)


def build_fusion_decision(
    *,
    yolo_result: Dict[str, Any],
    depth_result: Dict[str, Any],
    image_shape: Tuple[int, int],
    state: Optional[Dict[str, Any]] = None,
    houses_config: Optional[Dict[str, Any]] = None,
    crossing_ready_max_distance_cm: float = CROSSING_READY_MAX_DISTANCE_CM,
) -> Dict[str, Any]:
    detections = list(yolo_result.get("detections", []))
    detections = [
        {
            **det,
            "candidate_id": idx,
        }
        for idx, det in enumerate(detections)
    ]
    depth_analysis = dict(depth_result.get("analysis", {}))
    candidates = list(depth_analysis.get("entry_assessment", {}).get("candidates", []))
    best_depth_candidate = depth_analysis.get("entry_assessment", {}).get("best_candidate", {}) or {}
    front_obstacle = depth_analysis.get("front_obstacle", {}) or {}
    houses_cfg = houses_config if isinstance(houses_config, dict) else {}
    target_context = build_target_context(
        state=state,
        houses_config=houses_cfg,
        image_shape=image_shape,
        hfov_deg=float(depth_result.get("hfov_deg", 90.0)),
    )

    semantic_detection = choose_best_semantic_detection(detections)
    matched_candidate: Optional[Dict[str, Any]] = None
    match_score = 0.0
    match_iou = 0.0
    match_center = 0.0
    if semantic_detection is not None and candidates:
        matched_candidate, match_score, match_iou, match_center = match_detection_to_candidate(
            semantic_detection,
            candidates,
            image_shape,
        )

    thresholds = depth_result.get("thresholds", {}) if isinstance(depth_result.get("thresholds"), dict) else {}
    semantic_depth_assessment: Optional[Dict[str, Any]] = None
    semantic_depth_by_candidate_id: Dict[int, Dict[str, Any]] = {}
    semantic_detections = [det for det in detections if class_priority(str(det.get("class_name_normalized", ""))) > 0]
    for det in semantic_detections:
        assessment = summarize_semantic_region_depth(
            depth_cm=depth_result.get("depth_cm_array"),
            detection=det,
            rgb_shape=image_shape,
            hfov_deg=float(depth_result.get("hfov_deg", 90.0)),
            traversable_min_width_cm=float(thresholds.get("traversable_min_width_cm", 90.0)),
            traversable_min_clearance_cm=float(thresholds.get("traversable_min_clearance_cm", 160.0)),
            traversable_min_depth_gain_cm=float(thresholds.get("traversable_min_depth_gain_cm", 80.0)),
            crossing_ready_max_distance_cm=float(crossing_ready_max_distance_cm),
        )
        if assessment:
            semantic_depth_by_candidate_id[int(det.get("candidate_id", -1))] = assessment
    if semantic_detection is not None:
        semantic_depth_assessment = semantic_depth_by_candidate_id.get(int(semantic_detection.get("candidate_id", -1)))

    normalized_name = normalize_class_name((semantic_detection or {}).get("class_name"))
    chosen_candidate: Dict[str, Any] = {}
    chosen_depth_source = "none"
    if semantic_depth_assessment:
        chosen_candidate = semantic_depth_assessment
        chosen_depth_source = "semantic_region"
    elif matched_candidate:
        chosen_candidate = matched_candidate
        chosen_depth_source = "matched_global_candidate"
    elif isinstance(best_depth_candidate, dict) and best_depth_candidate:
        chosen_candidate = best_depth_candidate
        chosen_depth_source = "best_global_candidate"

    traversable = bool(chosen_candidate.get("traversable", False))
    front_min_depth_cm = front_obstacle.get("front_min_depth_cm")
    try:
        front_min_depth_cm = float(front_min_depth_cm) if front_min_depth_cm is not None else None
    except Exception:
        front_min_depth_cm = None
    front_blocked = bool(front_obstacle.get("present", False)) and (
        str(front_obstacle.get("severity", "")).lower() == "high"
        or (front_min_depth_cm is not None and front_min_depth_cm <= 80.0)
    )
    chosen_distance_cm = float(chosen_candidate.get("entry_distance_cm", 0.0)) if chosen_candidate else 0.0
    crossing_ready = bool(chosen_candidate.get("crossing_ready", False))
    if chosen_candidate and "crossing_ready" not in chosen_candidate:
        crossing_ready = bool(traversable and chosen_distance_cm <= float(crossing_ready_max_distance_cm))

    if front_blocked:
        final_state = "front_blocked_detour"
        decision_reason = (
            f"front obstacle is too close "
            f"(min={front_min_depth_cm if front_min_depth_cm is not None else 'n/a'}cm, "
            f"severity={front_obstacle.get('severity', 'unknown')})"
        )
    elif semantic_detection is not None and normalized_name in OPEN_DOOR_CLASSES and traversable:
        final_state = "enterable_open_door"
        if crossing_ready:
            decision_reason = "semantic open door matched a traversable depth opening"
        else:
            decision_reason = "semantic open door is traversable, but it is still too far away and should be approached first"
    elif semantic_detection is not None and normalized_name in DOORLIKE_CLASSES and traversable:
        final_state = "enterable_door"
        if crossing_ready:
            decision_reason = "semantic door matched a traversable depth opening"
        else:
            decision_reason = "semantic door is traversable, but it is still too far away and should be approached first"
    elif semantic_detection is not None and normalized_name in DOORLIKE_CLASSES and not traversable:
        final_state = "visible_but_blocked_entry"
        decision_reason = "door-like region is visible but geometry is not safely traversable"
    elif semantic_detection is not None and normalized_name in WINDOW_CLASSES:
        final_state = "window_visible_keep_search"
        decision_reason = "semantic detection is window, not an entry door"
    elif semantic_detection is None and bool(best_depth_candidate.get("traversable", False)):
        final_state = "geometric_opening_needs_confirmation"
        decision_reason = "depth suggests a traversable opening but semantic confirmation is missing"
    else:
        final_state = "no_entry_confirmed"
        decision_reason = "no reliable entry evidence was confirmed"

    center_x_norm = None
    if semantic_depth_assessment:
        center_x_norm = semantic_depth_assessment.get("center_x_norm")
    elif semantic_detection is not None:
        det_box = semantic_detection.get("xyxy", [])
        if isinstance(det_box, list) and len(det_box) == 4 and image_shape[1] > 0:
            center_x_norm = ((float(det_box[0]) + float(det_box[2])) * 0.5) / float(image_shape[1])
    elif chosen_candidate:
        center_x_norm = chosen_candidate.get("center_x_norm")

    recommended_subgoal, recommended_action_hint = recommend_subgoal_and_action(
        final_state=final_state,
        center_x_norm=center_x_norm,
        crossing_ready=crossing_ready,
        front_blocked=front_blocked,
    )

    candidate_target_scores: List[Dict[str, Any]] = []
    for det in semantic_detections:
        candidate_target_scores.append(
            score_candidate_for_target_house(
                detection=det,
                semantic_depth_assessment=semantic_depth_by_candidate_id.get(int(det.get("candidate_id", -1))),
                target_context=target_context,
                image_shape=image_shape,
            )
        )
    candidate_target_scores, memory_guidance = apply_memory_aware_candidate_adjustments(
        candidate_target_scores,
        target_context=target_context,
    )
    candidate_target_scores.sort(key=lambda item: float(item.get("candidate_total_score", 0.0)), reverse=True)
    best_target_candidate = candidate_target_scores[0] if candidate_target_scores else {}
    best_target_candidate_is_target_house_entry = bool(best_target_candidate.get("candidate_is_target_house_entry", False))
    target_conditioned_state = "target_house_not_in_view"
    target_conditioned_reason = "target house is not visible in the current field of view"
    if not target_context.get("target_house_id"):
        target_conditioned_state = "target_house_not_in_view"
        target_conditioned_reason = "target house context is unavailable"
    elif not bool(target_context.get("target_house_in_fov", False)):
        target_conditioned_state = "target_house_not_in_view"
        target_conditioned_reason = "target house is outside the current field of view"
    elif best_target_candidate_is_target_house_entry:
        if front_blocked or not bool(best_target_candidate.get("traversable", False)):
            target_conditioned_state = "target_house_entry_blocked"
            target_conditioned_reason = "target-house entry candidate exists but the path is blocked or not safely traversable"
        elif bool(best_target_candidate.get("crossing_ready", False)):
            target_conditioned_state = "target_house_entry_approachable"
            target_conditioned_reason = "target-house entry candidate is aligned, traversable, and close enough to approach/cross"
        else:
            target_conditioned_state = "target_house_entry_approachable"
            target_conditioned_reason = "target-house entry candidate is visible and traversable, but it should be approached first"
    elif candidate_target_scores:
        if any(normalize_class_name(item.get("class_name")) in DOORLIKE_CLASSES for item in candidate_target_scores):
            target_conditioned_state = "non_target_house_entry_visible"
            target_conditioned_reason = "visible door-like candidate is more likely to belong to a non-target house"
        else:
            target_conditioned_state = "target_house_entry_visible"
            target_conditioned_reason = "target house is in view, but only weak or non-door entry evidence is available"
    else:
        target_conditioned_state = "target_house_entry_visible"
        target_conditioned_reason = "target house is in view, but no reliable entry candidate has been detected yet"

    target_conditioned_subgoal, target_conditioned_action_hint = recommend_target_conditioned_action(
        target_state=target_conditioned_state,
        target_bearing_deg=target_context.get("target_house_bearing_deg"),
        candidate_center_x_norm=best_target_candidate.get("center_x_norm"),
        crossing_ready=bool(best_target_candidate.get("crossing_ready", False)),
    )
    (
        target_conditioned_state,
        target_conditioned_subgoal,
        target_conditioned_action_hint,
        memory_decision_guidance,
    ) = apply_memory_aware_target_decision(
        target_context=target_context,
        target_state=target_conditioned_state,
        target_subgoal=target_conditioned_subgoal,
        target_action=target_conditioned_action_hint,
        target_reason=target_conditioned_reason,
        best_target_candidate=best_target_candidate,
    )
    target_conditioned_reason = str(memory_decision_guidance.get("target_reason") or target_conditioned_reason)
    decision_text = (
        f"{final_state}; "
        f"class={semantic_detection.get('class_name') if semantic_detection else 'none'}; "
        f"dist={float(chosen_candidate.get('entry_distance_cm', 0.0)):.0f}cm; "
        f"width={float(chosen_candidate.get('opening_width_cm', 0.0)):.0f}cm; "
        f"trav={int(bool(chosen_candidate.get('traversable', False)))}; "
        f"cross_ready={int(bool(crossing_ready))}; "
        f"front_obstacle={int(bool(front_obstacle.get('present', False)))}; "
        f"reason={decision_reason}"
    )
    return {
        "final_entry_state": final_state,
        "decision_text": decision_text,
        "decision_reason": decision_reason,
        "semantic_detection": semantic_detection,
        "semantic_depth_assessment": semantic_depth_assessment,
        "matched_depth_candidate": matched_candidate,
        "best_depth_candidate": best_depth_candidate,
        "chosen_depth_candidate": chosen_candidate,
        "chosen_depth_source": chosen_depth_source,
        "global_front_obstacle": front_obstacle,
        "front_blocked": front_blocked,
        "match_score": round(match_score, 4),
        "match_iou": round(match_iou, 4),
        "match_center_score": round(match_center, 4),
        "entry_visible": semantic_detection is not None,
        "entry_class": semantic_detection.get("class_name") if semantic_detection else None,
        "entry_semantic_confidence": float(semantic_detection.get("confidence", 0.0)) if semantic_detection else None,
        "entry_distance_cm": float(chosen_candidate.get("entry_distance_cm", 0.0)) if chosen_candidate else None,
        "opening_width_cm": float(chosen_candidate.get("opening_width_cm", 0.0)) if chosen_candidate else None,
        "traversable": bool(chosen_candidate.get("traversable", False)) if chosen_candidate else False,
        "crossing_ready": crossing_ready,
        "crossing_ready_max_distance_cm": float(crossing_ready_max_distance_cm),
        "recommended_subgoal": recommended_subgoal,
        "recommended_action_hint": recommended_action_hint,
        "target_context": target_context,
        "memory_guidance": memory_guidance,
        "memory_decision_guidance": memory_decision_guidance,
        "candidate_target_scores": candidate_target_scores,
        "best_target_candidate": best_target_candidate,
        "best_target_candidate_is_target_house_entry": best_target_candidate_is_target_house_entry,
        "target_conditioned_state": target_conditioned_state,
        "target_conditioned_subgoal": target_conditioned_subgoal,
        "target_conditioned_action_hint": target_conditioned_action_hint,
        "target_conditioned_reason": target_conditioned_reason,
    }


def draw_fusion_overlay(
    *,
    rgb_bgr: np.ndarray,
    yolo_result: Dict[str, Any],
    depth_result: Dict[str, Any],
    fusion_result: Dict[str, Any],
) -> np.ndarray:
    rgb_annotated = np.asarray(yolo_result.get("annotated_image"), dtype=np.uint8)
    if rgb_annotated.ndim != 3:
        rgb_annotated = rgb_bgr.copy()
    rgb_canvas = rgb_annotated.copy()

    chosen = fusion_result.get("chosen_depth_candidate") or fusion_result.get("matched_depth_candidate") or fusion_result.get("best_depth_candidate") or {}
    if isinstance(chosen, dict) and chosen:
        rgb_box = chosen.get("rgb_bbox_xyxy")
        if isinstance(rgb_box, list) and len(rgb_box) == 4:
            x = int(round(float(rgb_box[0])))
            y = int(round(float(rgb_box[1])))
            w = int(round(float(rgb_box[2]) - float(rgb_box[0])))
            h = int(round(float(rgb_box[3]) - float(rgb_box[1])))
        else:
            bbox = chosen.get("bbox", {})
            x = int(bbox.get("x", 0))
            y = int(bbox.get("y", 0))
            w = int(bbox.get("width", 0))
            h = int(bbox.get("height", 0))
        color = (0, 255, 0) if bool(chosen.get("traversable", False)) else (0, 0, 255)
        cv2.rectangle(rgb_canvas, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            rgb_canvas,
            f"depth {float(chosen.get('entry_distance_cm', 0.0)):.0f}cm {float(chosen.get('opening_width_cm', 0.0)):.0f}cm",
            (x, max(24, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    depth_overlay = np.asarray(depth_result.get("overlay_image"), dtype=np.uint8)
    if depth_overlay.shape[:2] != rgb_canvas.shape[:2]:
        depth_overlay = cv2.resize(depth_overlay, (rgb_canvas.shape[1], rgb_canvas.shape[0]), interpolation=cv2.INTER_LINEAR)

    combined = cv2.hconcat([rgb_canvas, depth_overlay])
    summary_lines = [
        str(fusion_result.get("decision_text", "fusion unavailable")),
        (
            f"semantic={fusion_result.get('entry_class') or 'none'} "
            f"conf={float(fusion_result.get('entry_semantic_confidence') or 0.0):.2f} "
            f"match={float(fusion_result.get('match_score') or 0.0):.2f}"
        ),
        (
            f"subgoal={fusion_result.get('recommended_subgoal') or 'n/a'} "
            f"action={fusion_result.get('recommended_action_hint') or 'n/a'} "
            f"cross_ready={int(bool(fusion_result.get('crossing_ready', False)))}"
        ),
    ]
    for idx, line in enumerate(summary_lines):
        cv2.putText(
            combined,
            line,
            (18, 28 + idx * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return combined


def build_panel_summary(result: Dict[str, Any]) -> str:
    fusion = result.get("fusion", {}) if isinstance(result.get("fusion"), dict) else {}
    depth_analysis = result.get("depth", {}).get("analysis", {}) if isinstance(result.get("depth", {}), dict) else {}
    best = fusion.get("chosen_depth_candidate") or fusion.get("matched_depth_candidate") or fusion.get("best_depth_candidate") or {}
    obstacle = depth_analysis.get("front_obstacle", {}) if isinstance(depth_analysis, dict) else {}
    memory = fusion.get("entry_search_memory", {}) if isinstance(fusion.get("entry_search_memory"), dict) else {}
    return "\n".join(
        [
            f"State: {fusion.get('final_entry_state', 'unknown')}",
            (
                f"YOLO: {fusion.get('entry_class') or 'none'} "
                f"(conf={float(fusion.get('entry_semantic_confidence') or 0.0):.2f})"
            ),
            (
                f"Depth: dist={float(best.get('entry_distance_cm', 0.0)):.0f}cm "
                f"width={float(best.get('opening_width_cm', 0.0)):.0f}cm "
                f"trav={int(bool(best.get('traversable', False)))} "
                f"cross_ready={int(bool(fusion.get('crossing_ready', False)))}"
            ),
            (
                f"Front obstacle: {int(bool(obstacle.get('present', False)))} "
                f"(min={obstacle.get('front_min_depth_cm', 'n/a')}cm)"
            ),
            f"Match score: {float(fusion.get('match_score') or 0.0):.2f}",
            (
                f"Decision: subgoal={fusion.get('recommended_subgoal') or 'n/a'} "
                f"action={fusion.get('recommended_action_hint') or 'n/a'}"
            ),
            (
                f"Target: state={fusion.get('target_conditioned_state') or 'n/a'} "
                f"subgoal={fusion.get('target_conditioned_subgoal') or 'n/a'} "
                f"action={fusion.get('target_conditioned_action_hint') or 'n/a'}"
            ),
            (
                f"Memory: status={memory.get('entry_search_status') or 'n/a'} "
                f"sector={memory.get('sector_id') or 'n/a'} "
                f"house={memory.get('house_id') or 'n/a'}"
            ),
            (
                f"MemoryGuide: history={int(bool((fusion.get('memory_guidance') or {}).get('candidate_history_used', False)))} "
                f"sector_penalty={int(bool((fusion.get('memory_guidance') or {}).get('sector_penalty_applied', False)))} "
                f"track={int(bool((fusion.get('memory_guidance') or {}).get('last_best_tracking_used', False)))}"
            ),
            (
                f"MemoryDecision: override={int(bool((fusion.get('memory_decision_guidance') or {}).get('decision_override_applied', False)))} "
                f"reason={((fusion.get('memory_decision_guidance') or {}).get('override_reason') or 'none')} "
                f"alt_sector={((fusion.get('memory_decision_guidance') or {}).get('alternate_sector_id') or 'n/a')}"
            ),
            f"Reason: {fusion.get('decision_reason', 'n/a')}",
        ]
    )


def extract_pose_history_summary(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    state = state if isinstance(state, dict) else {}
    pose = state.get("pose", {}) if isinstance(state.get("pose"), dict) else {}
    summary: Dict[str, Any] = {
        "task_label": state.get("task_label"),
        "pose": {
            "x": pose.get("x"),
            "y": pose.get("y"),
            "z": pose.get("z"),
            "yaw": pose.get("task_yaw", pose.get("yaw", pose.get("uav_yaw"))),
            "uav_yaw": pose.get("uav_yaw"),
            "frame_name": pose.get("frame_name"),
        },
        "action": state.get("action_name", state.get("action")),
        "movement_enabled": state.get("movement_enabled"),
        "current_house": state.get("current_house_id", state.get("current_house")),
        "target_house": state.get("target_house_id", state.get("target_house")),
        "target_distance_cm": state.get("target_distance_cm", state.get("target_distance")),
        "mission": state.get("mission"),
        "phase2_fusion": state.get("phase2_fusion"),
        "phase2_decision": state.get("phase2_decision"),
    }
    for key in (
        "recent_actions",
        "action_history",
        "history",
        "navigation_history",
        "event_history",
        "recent_events",
        "phase_history",
        "pose_history",
        "movement_history",
        "decision_history",
        "fusion_history",
    ):
        value = state.get(key)
        if value is not None:
            summary[key] = value
    return summary


def extract_state_excerpt(state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    state = state if isinstance(state, dict) else {}
    excerpt: Dict[str, Any] = {}
    for key in (
        "task_label",
        "action",
        "action_name",
        "movement_enabled",
        "current_house_id",
        "target_house_id",
        "target_distance_cm",
        "mission",
        "phase2_fusion",
        "phase2_decision",
        "camera_info",
    ):
        if key in state:
            excerpt[key] = state.get(key)
    pose = state.get("pose")
    if isinstance(pose, dict):
        excerpt["pose"] = pose
    return excerpt


def build_labeling_manifest(
    *,
    sample_id: str,
    inputs: Dict[str, Any],
    yolo_section: Dict[str, Any],
    depth_section: Dict[str, Any],
    fusion_section: Dict[str, Any],
    pose_history: Dict[str, Any],
    state_excerpt: Dict[str, Any],
    camera_info: Optional[Dict[str, Any]],
    run_dir: Path,
    labeling_dir: Path,
) -> Dict[str, Any]:
    detections = list(yolo_section.get("detections", []))
    yolo_brief: List[Dict[str, Any]] = []
    for det in detections[:5]:
        yolo_brief.append(
            {
                "class_name": det.get("class_name"),
                "confidence": det.get("confidence"),
                "xyxy": det.get("xyxy"),
            }
        )
    depth_analysis = depth_section.get("analysis", {}) if isinstance(depth_section.get("analysis"), dict) else {}
    best_entry = (
        fusion_section.get("chosen_depth_candidate")
        or fusion_section.get("matched_depth_candidate")
        or fusion_section.get("best_depth_candidate")
        or {}
    )
    front_obstacle = fusion_section.get("global_front_obstacle") or depth_analysis.get("front_obstacle") or {}
    manifest = {
        "sample_id": sample_id,
        "run_dir": str(run_dir),
        "labeling_dir": str(labeling_dir),
        "task_label": pose_history.get("task_label"),
        "images": {
            "rgb": str(labeling_dir / "rgb.png"),
            "depth_cm": str(labeling_dir / "depth_cm.png"),
            "depth_preview": str(labeling_dir / "depth_preview.png"),
            "yolo_annotated": str(labeling_dir / "yolo_annotated.jpg"),
            "depth_overlay": str(labeling_dir / "depth_overlay.png"),
            "fusion_overlay": str(labeling_dir / "fusion_overlay.png"),
        },
        "artifacts": {
            "camera_info": str(labeling_dir / "camera_info.json"),
            "state": str(labeling_dir / "state.json"),
            "pose_history_summary": str(labeling_dir / "pose_history_summary.json"),
            "yolo_result": str(labeling_dir / "yolo_result.json"),
            "yolo_summary": str(labeling_dir / "yolo_summary.txt"),
            "depth_result": str(labeling_dir / "depth_result.json"),
            "depth_summary": str(labeling_dir / "depth_summary.txt"),
            "fusion_result": str(labeling_dir / "fusion_result.json"),
            "fusion_summary": str(labeling_dir / "fusion_summary.txt"),
            "labeling_summary": str(labeling_dir / "labeling_summary.txt"),
            "entry_search_memory": str(fusion_section.get("entry_search_memory", {}).get("memory_path") or ""),
        },
        "camera_info": camera_info or {},
        "pose_history": pose_history,
        "state_excerpt": state_excerpt,
        "yolo_summary": {
            "num_detections": yolo_section.get("num_detections"),
            "top_detections": yolo_brief,
        },
        "depth_summary": {
            "front_obstacle": front_obstacle,
            "best_entry": {
                "entry_distance_cm": best_entry.get("entry_distance_cm"),
                "opening_width_cm": best_entry.get("opening_width_cm"),
                "traversable": best_entry.get("traversable"),
                "crossing_ready": best_entry.get("crossing_ready", fusion_section.get("crossing_ready")),
                "confidence": best_entry.get("confidence"),
            },
        },
        "fusion_summary": {
            "final_entry_state": fusion_section.get("final_entry_state"),
            "decision_reason": fusion_section.get("decision_reason"),
            "entry_class": fusion_section.get("entry_class"),
            "entry_semantic_confidence": fusion_section.get("entry_semantic_confidence"),
            "match_score": fusion_section.get("match_score"),
            "match_iou": fusion_section.get("match_iou"),
            "match_center_score": fusion_section.get("match_center_score"),
            "front_blocked": fusion_section.get("front_blocked"),
            "crossing_ready": fusion_section.get("crossing_ready"),
            "recommended_subgoal": fusion_section.get("recommended_subgoal"),
            "recommended_action_hint": fusion_section.get("recommended_action_hint"),
            "target_house_id": (fusion_section.get("target_context") or {}).get("target_house_id"),
            "target_house_in_fov": (fusion_section.get("target_context") or {}).get("target_house_in_fov"),
            "target_conditioned_state": fusion_section.get("target_conditioned_state"),
            "target_conditioned_subgoal": fusion_section.get("target_conditioned_subgoal"),
            "target_conditioned_action_hint": fusion_section.get("target_conditioned_action_hint"),
            "target_conditioned_reason": fusion_section.get("target_conditioned_reason"),
            "entry_search_memory_status": fusion_section.get("entry_search_memory", {}).get("entry_search_status"),
            "entry_search_memory_sector_id": fusion_section.get("entry_search_memory", {}).get("sector_id"),
            "entry_search_memory_house_id": fusion_section.get("entry_search_memory", {}).get("house_id"),
            "memory_guidance_history_used": fusion_section.get("memory_guidance", {}).get("candidate_history_used"),
            "memory_guidance_sector_penalty_applied": fusion_section.get("memory_guidance", {}).get("sector_penalty_applied"),
            "memory_guidance_last_best_tracking_used": fusion_section.get("memory_guidance", {}).get("last_best_tracking_used"),
            "memory_decision_override_applied": fusion_section.get("memory_decision_guidance", {}).get("decision_override_applied"),
            "memory_decision_override_reason": fusion_section.get("memory_decision_guidance", {}).get("override_reason"),
            "memory_decision_alternate_sector_id": fusion_section.get("memory_decision_guidance", {}).get("alternate_sector_id"),
        },
        "annotation_template": {
            "gt_entry_state": "",
            "gt_subgoal": "",
            "gt_action_hint": "",
            "reviewer": "",
            "notes": "",
        },
    }
    return manifest


def build_labeling_summary_text(manifest: Dict[str, Any]) -> str:
    fusion = manifest.get("fusion_summary", {}) if isinstance(manifest.get("fusion_summary"), dict) else {}
    depth = manifest.get("depth_summary", {}) if isinstance(manifest.get("depth_summary"), dict) else {}
    front_obstacle = depth.get("front_obstacle", {}) if isinstance(depth.get("front_obstacle"), dict) else {}
    best_entry = depth.get("best_entry", {}) if isinstance(depth.get("best_entry"), dict) else {}
    pose = manifest.get("pose_history", {}).get("pose", {}) if isinstance(manifest.get("pose_history"), dict) else {}
    top_detections = manifest.get("yolo_summary", {}).get("top_detections", [])
    top_line = "none"
    if top_detections:
        first = top_detections[0] if isinstance(top_detections[0], dict) else {}
        top_line = f"{first.get('class_name')} ({float(first.get('confidence') or 0.0):.2f})"
    lines = [
        f"sample_id: {manifest.get('sample_id', '')}",
        f"task_label: {manifest.get('task_label', '')}",
        (
            "pose: "
            f"x={pose.get('x')} y={pose.get('y')} z={pose.get('z')} "
            f"yaw={pose.get('yaw')}"
        ),
        f"top_yolo: {top_line}",
        (
            "depth: "
            f"front_obstacle={int(bool(front_obstacle.get('present', False)))} "
            f"min={front_obstacle.get('front_min_depth_cm', 'n/a')}cm "
            f"severity={front_obstacle.get('severity', 'n/a')}"
        ),
        (
            "best_entry: "
            f"dist={best_entry.get('entry_distance_cm', 'n/a')}cm "
            f"width={best_entry.get('opening_width_cm', 'n/a')}cm "
            f"trav={int(bool(best_entry.get('traversable', False)))} "
            f"cross_ready={int(bool(best_entry.get('crossing_ready', False)))}"
        ),
        f"fusion_state: {fusion.get('final_entry_state', 'unknown')}",
        (
            f"fusion_decision: subgoal={fusion.get('recommended_subgoal', 'n/a')} "
            f"action={fusion.get('recommended_action_hint', 'n/a')}"
        ),
        (
            f"target_fusion: state={fusion.get('target_conditioned_state', 'n/a')} "
            f"subgoal={fusion.get('target_conditioned_subgoal', 'n/a')} "
            f"action={fusion.get('target_conditioned_action_hint', 'n/a')}"
        ),
        (
            f"entry_memory: house={fusion.get('entry_search_memory_house_id', 'n/a')} "
            f"status={fusion.get('entry_search_memory_status', 'n/a')} "
            f"sector={fusion.get('entry_search_memory_sector_id', 'n/a')}"
        ),
        (
            f"memory_guidance: history={int(bool(fusion.get('memory_guidance_history_used', False)))} "
            f"sector_penalty={int(bool(fusion.get('memory_guidance_sector_penalty_applied', False)))} "
            f"track={int(bool(fusion.get('memory_guidance_last_best_tracking_used', False)))}"
        ),
        (
            f"memory_decision: override={int(bool(fusion.get('memory_decision_override_applied', False)))} "
            f"reason={fusion.get('memory_decision_override_reason', 'none')} "
            f"alt_sector={fusion.get('memory_decision_alternate_sector_id', 'n/a')}"
        ),
        f"target_reason: {fusion.get('target_conditioned_reason', 'n/a')}",
        f"fusion_reason: {fusion.get('decision_reason', 'n/a')}",
    ]
    return "\n".join(lines) + "\n"


def run_phase2_fusion_analysis(
    *,
    rgb_bgr: np.ndarray,
    depth_raw: np.ndarray,
    output_root: Optional[Path] = None,
    existing_run_dir: Optional[Path] = None,
    weights: Optional[Path] = None,
    label: str = "",
    camera_info: Optional[Dict[str, Any]] = None,
    state: Optional[Dict[str, Any]] = None,
    houses_config: Optional[Dict[str, Any]] = None,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "0",
) -> Dict[str, Any]:
    if rgb_bgr is None or not isinstance(rgb_bgr, np.ndarray):
        raise RuntimeError("rgb_bgr is required for fusion analysis.")
    if depth_raw is None or not isinstance(depth_raw, np.ndarray):
        raise RuntimeError("depth_raw is required for fusion analysis.")
    houses_cfg = houses_config if isinstance(houses_config, dict) else load_houses_config()

    run_root = (output_root or (ROOT / "phase2_multimodal_fusion_analysis" / "results")).resolve()
    ensure_dir(run_root)
    if existing_run_dir is not None:
        run_dir = existing_run_dir.resolve()
        ensure_dir(run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_suffix = f"_{str(label).strip()}" if str(label or "").strip() else ""
        run_dir = run_root / f"fusion_{timestamp}{label_suffix}"
    inputs_dir = run_dir / "inputs"
    yolo_dir = run_dir / "yolo"
    depth_dir = run_dir / "depth"
    fusion_dir = run_dir / "fusion"
    labeling_dir = run_dir / "labeling"
    for path in (inputs_dir, yolo_dir, depth_dir, fusion_dir, labeling_dir):
        ensure_dir(path)

    rgb_path = inputs_dir / "rgb.png"
    depth_cm_path = inputs_dir / "depth_cm.png"
    cv2.imwrite(str(rgb_path), rgb_bgr)
    cv2.imwrite(str(depth_cm_path), depth_raw)
    if camera_info is not None:
        (inputs_dir / "camera_info.json").write_text(json.dumps(camera_info, indent=2, ensure_ascii=False), encoding="utf-8")
    if state is not None:
        (inputs_dir / "state.json").write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

    selected_weights = (weights.resolve() if weights else find_latest_phase2_weights())
    yolo_result = run_yolo_inference(
        rgb_bgr=rgb_bgr,
        weights=selected_weights,
        output_dir=yolo_dir,
        conf=float(conf),
        imgsz=int(imgsz),
        device=str(device),
    )
    depth_result = run_depth_inference(depth_cm_path=depth_cm_path, output_dir=depth_dir)
    fusion = build_fusion_decision(
        yolo_result=yolo_result,
        depth_result=depth_result,
        image_shape=rgb_bgr.shape[:2],
        state=state,
        houses_config=houses_cfg,
    )
    try:
        entry_search_memory = update_entry_search_memory_from_fusion(
            houses_config=houses_cfg,
            fusion_result=fusion,
            state=state,
            yolo_result=yolo_result,
        )
    except Exception as exc:
        entry_search_memory = {
            "status": "error",
            "error": str(exc),
        }
    fusion["entry_search_memory"] = entry_search_memory
    fusion_overlay = draw_fusion_overlay(
        rgb_bgr=rgb_bgr,
        yolo_result=yolo_result,
        depth_result=depth_result,
        fusion_result=fusion,
    )
    fusion_overlay_path = fusion_dir / "fusion_overlay.png"
    fusion_json_path = fusion_dir / "fusion_result.json"
    fusion_txt_path = fusion_dir / "fusion_summary.txt"
    cv2.imwrite(str(fusion_overlay_path), fusion_overlay)

    result = {
        "status": "ok",
        "run_dir": str(run_dir),
        "labeling_dir": str(labeling_dir),
        "weights_path": str(selected_weights),
        "houses_config_path": str(DEFAULT_HOUSES_CONFIG_PATH.resolve()),
        "inputs": {
            "rgb_path": str(rgb_path),
            "depth_cm_path": str(depth_cm_path),
            "camera_info_path": str(inputs_dir / "camera_info.json") if camera_info is not None else None,
            "state_path": str(inputs_dir / "state.json") if state is not None else None,
        },
        "yolo": {
            "weights_path": yolo_result.get("weights_path"),
            "annotated_path": yolo_result.get("annotated_path"),
            "json_path": yolo_result.get("json_path"),
            "txt_path": yolo_result.get("txt_path"),
            "num_detections": yolo_result.get("num_detections"),
            "detections": yolo_result.get("detections"),
        },
        "depth": {
            "preview_path": depth_result.get("preview_path"),
            "overlay_path": depth_result.get("overlay_path"),
            "json_path": depth_result.get("json_path"),
            "txt_path": depth_result.get("txt_path"),
            "analysis": depth_result.get("analysis"),
        },
        "fusion": fusion,
        "fusion_overlay_path": str(fusion_overlay_path),
    }
    result["panel_summary"] = build_panel_summary(result)

    pose_history = extract_pose_history_summary(state)
    state_excerpt = extract_state_excerpt(state)
    pose_history_json_path = labeling_dir / "pose_history_summary.json"
    pose_history_txt_path = labeling_dir / "pose_history_summary.txt"
    state_excerpt_json_path = labeling_dir / "state_excerpt.json"
    pose_history_json_path.write_text(json.dumps(pose_history, indent=2, ensure_ascii=False), encoding="utf-8")
    pose_history_txt_path.write_text(
        json.dumps(pose_history, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    state_excerpt_json_path.write_text(json.dumps(state_excerpt, indent=2, ensure_ascii=False), encoding="utf-8")

    copy_if_exists(str(rgb_path), labeling_dir / "rgb.png")
    copy_if_exists(str(depth_cm_path), labeling_dir / "depth_cm.png")
    copy_if_exists(str(depth_result.get("preview_path")), labeling_dir / "depth_preview.png")
    copy_if_exists(str(yolo_result.get("annotated_path")), labeling_dir / "yolo_annotated.jpg")
    copy_if_exists(str(depth_result.get("overlay_path")), labeling_dir / "depth_overlay.png")
    copy_if_exists(str(fusion_overlay_path), labeling_dir / "fusion_overlay.png")
    copy_if_exists(str(yolo_result.get("json_path")), labeling_dir / "yolo_result.json")
    copy_if_exists(str(yolo_result.get("txt_path")), labeling_dir / "yolo_summary.txt")
    copy_if_exists(str(depth_result.get("json_path")), labeling_dir / "depth_result.json")
    copy_if_exists(str(depth_result.get("txt_path")), labeling_dir / "depth_summary.txt")
    if camera_info is not None:
        copy_if_exists(str(inputs_dir / "camera_info.json"), labeling_dir / "camera_info.json")
    if state is not None:
        copy_if_exists(str(inputs_dir / "state.json"), labeling_dir / "state.json")

    manifest = build_labeling_manifest(
        sample_id=run_dir.name,
        inputs={
            **result["inputs"],
            "fusion_overlay_path": str(fusion_overlay_path),
        },
        yolo_section=result["yolo"],
        depth_section=result["depth"],
        fusion_section=result["fusion"],
        pose_history=pose_history,
        state_excerpt=state_excerpt,
        camera_info=camera_info if isinstance(camera_info, dict) else {},
        run_dir=run_dir,
        labeling_dir=labeling_dir,
    )
    labeling_manifest_path = labeling_dir / "labeling_manifest.json"
    annotation_template_path = labeling_dir / "annotation_template.json"
    labeling_summary_path = labeling_dir / "labeling_summary.txt"
    labeling_manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    annotation_template_path.write_text(
        json.dumps(manifest.get("annotation_template", {}), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    labeling_summary_path.write_text(build_labeling_summary_text(manifest), encoding="utf-8")
    result["labeling_manifest_path"] = str(labeling_manifest_path)
    fusion_json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    fusion_txt_path.write_text(result["panel_summary"] + "\n", encoding="utf-8")
    copy_if_exists(str(fusion_json_path), labeling_dir / "fusion_result.json")
    copy_if_exists(str(fusion_txt_path), labeling_dir / "fusion_summary.txt")
    return result


def reprocess_phase2_fusion_run(
    run_dir: Path,
    *,
    weights: Optional[Path] = None,
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "0",
) -> Dict[str, Any]:
    run_dir = run_dir.resolve()
    inputs_dir = run_dir / "inputs"
    rgb_path = inputs_dir / "rgb.png"
    depth_cm_path = inputs_dir / "depth_cm.png"
    if not rgb_path.exists():
        raise FileNotFoundError(f"Missing rgb input: {rgb_path}")
    if not depth_cm_path.exists():
        raise FileNotFoundError(f"Missing depth_cm input: {depth_cm_path}")

    rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise RuntimeError(f"Could not read rgb image: {rgb_path}")
    depth_raw = cv2.imread(str(depth_cm_path), cv2.IMREAD_UNCHANGED)
    if depth_raw is None:
        raise RuntimeError(f"Could not read depth image: {depth_cm_path}")

    camera_info_path = inputs_dir / "camera_info.json"
    state_path = inputs_dir / "state.json"
    camera_info = None
    state = None
    if camera_info_path.exists():
        camera_info = json.loads(camera_info_path.read_text(encoding="utf-8"))
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    houses_config = load_houses_config()

    return run_phase2_fusion_analysis(
        rgb_bgr=rgb_bgr,
        depth_raw=depth_raw,
        existing_run_dir=run_dir,
        weights=weights,
        label="",
        camera_info=camera_info,
        state=state,
        houses_config=houses_config,
        conf=float(conf),
        imgsz=int(imgsz),
        device=str(device),
    )
