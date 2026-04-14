from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RESULTS_ROOT = Path(__file__).resolve().parent / "results"
TOP_K_CANDIDATES = 3
MAX_DEPTH_CM = 1200.0
MAX_TARGET_DISTANCE_CM = 2000.0
MAX_DEPTH_GAIN_CM = 400.0
MAX_OPENING_WIDTH_CM = 300.0
CLASS_ORDER = ["open door", "door", "close door", "window"]
CLASS_PRIORITY = {
    "open door": 0,
    "door": 1,
    "close door": 2,
    "window": 3,
}
EXPECTED_SIDE_TO_ID = {
    "left": 0,
    "center": 1,
    "right": 2,
    "out_of_view": 3,
    "unknown": 4,
}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _normalize_class_name(value: Any) -> str:
    token = str(value or "").strip().lower().replace("_", " ")
    token = re.sub(r"\s+", " ", token)
    if token == "closed door":
        return "close door"
    return token


def _house_id_to_int(value: Any) -> int:
    text = str(value or "").strip()
    if not text:
        return -1
    match = re.search(r"(\d+)$", text)
    if match:
        return int(match.group(1))
    return -1


def _yaw_to_sincos(yaw_deg: float) -> Tuple[float, float]:
    radians = math.radians(float(yaw_deg))
    return math.sin(radians), math.cos(radians)


def _bbox_dict_to_xyxy(bbox: Dict[str, Any]) -> List[float]:
    x = _safe_float(bbox.get("x"), 0.0)
    y = _safe_float(bbox.get("y"), 0.0)
    w = _safe_float(bbox.get("width"), 0.0)
    h = _safe_float(bbox.get("height"), 0.0)
    return [x, y, x + w, y + h]


def _box_iou(box_a: List[float], box_b: List[float]) -> float:
    if len(box_a) != 4 or len(box_b) != 4:
        return 0.0
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


def _class_onehot(name: str) -> List[int]:
    normalized = _normalize_class_name(name)
    return [1 if normalized == cls else 0 for cls in CLASS_ORDER]


def _expected_side_to_id(value: Any) -> int:
    token = str(value or "").strip().lower()
    return EXPECTED_SIDE_TO_ID.get(token, EXPECTED_SIDE_TO_ID["unknown"])


def _zero_candidate(slot_id: int) -> Dict[str, Any]:
    return {
        "candidate_id": -1,
        "raw_candidate_id": -1,
        "valid_mask": 0,
        "class_name": "",
        "class_onehot": [0, 0, 0, 0],
        "confidence": 0.0,
        "bbox_cx": 0.0,
        "bbox_cy": 0.0,
        "bbox_w": 0.0,
        "bbox_h": 0.0,
        "bbox_area_ratio": 0.0,
        "aspect_ratio": 0.0,
        "entry_distance_cm": 0.0,
        "surrounding_depth_cm": 0.0,
        "clearance_depth_cm": 0.0,
        "depth_gain_cm": 0.0,
        "opening_width_cm": 0.0,
        "traversable": 0,
        "crossing_ready": 0,
        "candidate_rank": -1,
        "entry_distance_norm": 0.0,
        "surrounding_depth_norm": 0.0,
        "clearance_depth_norm": 0.0,
        "depth_gain_norm": 0.0,
        "opening_width_norm": 0.0,
        "source": "",
        "source_slot": slot_id,
        "match_iou": 0.0,
        "candidate_target_match_score": 0.0,
        "candidate_semantic_score": 0.0,
        "candidate_geometry_score": 0.0,
        "candidate_total_score": 0.0,
        "candidate_house_id": -1,
        "candidate_is_target_house_entry": 0,
        "candidate_target_side_match": 0.0,
        "candidate_center_in_target_bbox": 0,
        "candidate_near_target_bbox": 0,
        "candidate_image_side": "",
        "target_expected_side": "",
        "target_expected_image_x": 0.5,
    }


def _sort_detection_key(detection: Dict[str, Any]) -> Tuple[int, float]:
    class_name = _normalize_class_name(detection.get("class_name_normalized") or detection.get("class_name"))
    priority = CLASS_PRIORITY.get(class_name, 99)
    confidence = _safe_float(detection.get("confidence"), 0.0)
    return (priority, -confidence)


def _extract_depth_candidates(fusion: Dict[str, Any], depth_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []

    def _append_candidate(candidate: Optional[Dict[str, Any]], source_name: str) -> None:
        if not isinstance(candidate, dict):
            return
        item = dict(candidate)
        item["source"] = source_name
        output.append(item)

    _append_candidate(fusion.get("semantic_depth_assessment"), "semantic_region")
    _append_candidate(fusion.get("chosen_depth_candidate"), "chosen_depth")
    _append_candidate(fusion.get("matched_depth_candidate"), "matched_depth")
    _append_candidate(fusion.get("best_depth_candidate"), "best_depth")

    entry_assessment = depth_result.get("entry_assessment", {}) if isinstance(depth_result.get("entry_assessment"), dict) else {}
    for candidate in entry_assessment.get("candidates", []) if isinstance(entry_assessment.get("candidates"), list) else []:
        _append_candidate(candidate, "depth_candidate")
    return output


def _extract_target_score_candidates(fusion: Dict[str, Any]) -> List[Dict[str, Any]]:
    value = fusion.get("candidate_target_scores")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _match_target_score_for_detection(detection: Dict[str, Any], fusion: Dict[str, Any]) -> Dict[str, Any]:
    detection_box = detection.get("xyxy", [])
    if not isinstance(detection_box, list) or len(detection_box) != 4:
        return {}

    det_class = _normalize_class_name(detection.get("class_name_normalized") or detection.get("class_name"))
    best_item: Dict[str, Any] = {}
    best_iou = 0.0
    for candidate in _extract_target_score_candidates(fusion):
        candidate_box = candidate.get("xyxy", [])
        if not isinstance(candidate_box, list) or len(candidate_box) != 4:
            continue
        candidate_class = _normalize_class_name(candidate.get("class_name_normalized") or candidate.get("class_name"))
        if candidate_class and det_class and candidate_class != det_class:
            continue
        iou = _box_iou([float(v) for v in detection_box], [float(v) for v in candidate_box])
        if iou > best_iou:
            best_iou = iou
            best_item = dict(candidate)
    if best_iou >= 0.10:
        best_item["match_iou"] = best_iou
        return best_item
    return {}


def _match_depth_features_for_detection(
    detection: Dict[str, Any],
    fusion: Dict[str, Any],
    depth_result: Dict[str, Any],
) -> Dict[str, Any]:
    detection_box = detection.get("xyxy", [])
    if not isinstance(detection_box, list) or len(detection_box) != 4:
        return {}

    semantic_detection = fusion.get("semantic_detection", {}) if isinstance(fusion.get("semantic_detection"), dict) else {}
    semantic_box = semantic_detection.get("xyxy", []) if isinstance(semantic_detection.get("xyxy"), list) else []
    semantic_depth = fusion.get("semantic_depth_assessment", {}) if isinstance(fusion.get("semantic_depth_assessment"), dict) else {}
    if semantic_box and semantic_depth:
        if _box_iou([float(v) for v in detection_box], [float(v) for v in semantic_box]) >= 0.5:
            item = dict(semantic_depth)
            item["source"] = "semantic_region"
            item["match_iou"] = _box_iou([float(v) for v in detection_box], [float(v) for v in semantic_box])
            return item

    best_item: Dict[str, Any] = {}
    best_iou = 0.0
    for candidate in _extract_depth_candidates(fusion, depth_result):
        if "rgb_bbox_xyxy" in candidate and isinstance(candidate.get("rgb_bbox_xyxy"), list):
            candidate_box = [float(v) for v in candidate.get("rgb_bbox_xyxy", [])]
        else:
            candidate_box = _bbox_dict_to_xyxy(candidate.get("bbox", {}))
        iou = _box_iou([float(v) for v in detection_box], candidate_box)
        if iou > best_iou:
            best_iou = iou
            best_item = dict(candidate)
            best_item["match_iou"] = iou
    if best_iou >= 0.10:
        return best_item
    return {}


def _build_global_state(
    state_excerpt: Dict[str, Any],
    fusion: Dict[str, Any],
    depth_result: Dict[str, Any],
    pose_history: Dict[str, Any],
) -> Dict[str, Any]:
    pose = state_excerpt.get("pose", {}) if isinstance(state_excerpt.get("pose"), dict) else {}
    front_obstacle = depth_result.get("front_obstacle", {}) if isinstance(depth_result.get("front_obstacle"), dict) else {}
    target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
    yaw_deg = _safe_float(pose.get("uav_yaw", pose.get("yaw", pose.get("task_yaw", 0.0))), 0.0)
    yaw_sin, yaw_cos = _yaw_to_sincos(yaw_deg)
    target_distance_cm = target_context.get("target_house_distance_cm", state_excerpt.get("target_distance_cm"))
    target_distance_cm = None if target_distance_cm is None else _safe_float(target_distance_cm, 0.0)
    target_bearing_deg = target_context.get("target_house_bearing_deg", state_excerpt.get("target_bearing_deg"))
    target_bearing_deg = None if target_bearing_deg is None else _safe_float(target_bearing_deg, 0.0)
    target_bearing_sin, target_bearing_cos = (0.0, 0.0)
    if target_bearing_deg is not None:
        target_bearing_sin, target_bearing_cos = _yaw_to_sincos(target_bearing_deg)

    history_actions: List[str] = []
    action_value = pose_history.get("action")
    if action_value:
        history_actions.append(str(action_value))
    for field in ("recent_actions", "action_history", "navigation_history"):
        value = state_excerpt.get(field)
        if isinstance(value, list):
            history_actions.extend([str(item) for item in value if str(item).strip()])
    deduped_history: List[str] = []
    for action in history_actions:
        if action not in deduped_history:
            deduped_history.append(action)

    return {
        "pose_x": _safe_float(pose.get("x"), 0.0),
        "pose_y": _safe_float(pose.get("y"), 0.0),
        "pose_z": _safe_float(pose.get("z"), 0.0),
        "yaw_deg": yaw_deg,
        "yaw_sin": round(yaw_sin, 6),
        "yaw_cos": round(yaw_cos, 6),
        "front_obstacle_present": int(bool(front_obstacle.get("present", False))),
        "front_min_depth_cm": _safe_float(front_obstacle.get("front_min_depth_cm"), 0.0),
        "front_obstacle_severity": str(front_obstacle.get("severity") or "").strip().lower(),
        "target_house_id": _house_id_to_int(target_context.get("target_house_id") or state_excerpt.get("target_house") or state_excerpt.get("target_house_id")),
        "current_house_id": _house_id_to_int(target_context.get("current_house_id") or state_excerpt.get("current_house") or state_excerpt.get("current_house_id")),
        "target_house_known_mask": int(bool(target_context.get("target_house_id"))),
        "target_house_in_fov": int(bool(target_context.get("target_house_in_fov", False))),
        "target_house_expected_side": _expected_side_to_id(target_context.get("target_house_expected_side")),
        "target_house_expected_side_text": str(target_context.get("target_house_expected_side") or ""),
        "target_house_expected_image_x": round(
            _clamp(_safe_float(target_context.get("target_house_expected_image_x"), 0.5), 0.0, 1.0), 6
        ),
        "target_house_bbox_available_mask": int(bool(target_context.get("target_house_map_bbox_image"))),
        "target_distance_cm": target_distance_cm if target_distance_cm is not None else 0.0,
        "target_bearing_deg": target_bearing_deg if target_bearing_deg is not None else 0.0,
        "target_bearing_sin": round(target_bearing_sin, 6),
        "target_bearing_cos": round(target_bearing_cos, 6),
        "movement_enabled": int(bool(state_excerpt.get("movement_enabled", False))),
        "history_actions": deduped_history[:4],
        "front_min_depth_norm": round(_clamp(_safe_float(front_obstacle.get("front_min_depth_cm"), 0.0) / MAX_DEPTH_CM, 0.0, 1.0), 6),
        "target_distance_norm": round(
            _clamp(((target_distance_cm or 0.0) / MAX_TARGET_DISTANCE_CM), 0.0, 1.0), 6
        ),
    }


def _build_candidate(detection: Dict[str, Any], fusion: Dict[str, Any], depth_result: Dict[str, Any], rank: int) -> Dict[str, Any]:
    xyxy = detection.get("xyxy", [])
    x1, y1, x2, y2 = [float(v) for v in xyxy] if isinstance(xyxy, list) and len(xyxy) == 4 else [0.0, 0.0, 0.0, 0.0]
    bbox_w = max(0.0, x2 - x1)
    bbox_h = max(0.0, y2 - y1)
    class_name = _normalize_class_name(detection.get("class_name_normalized") or detection.get("class_name"))
    depth_features = _match_depth_features_for_detection(detection, fusion, depth_result)
    target_features = _match_target_score_for_detection(detection, fusion)
    target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}

    entry_distance_cm = _safe_float(depth_features.get("entry_distance_cm"), 0.0)
    surrounding_depth_cm = _safe_float(depth_features.get("surrounding_depth_cm"), 0.0)
    clearance_depth_cm = _safe_float(depth_features.get("clearance_depth_cm"), 0.0)
    depth_gain_cm = _safe_float(depth_features.get("depth_gain_cm"), 0.0)
    opening_width_cm = _safe_float(depth_features.get("opening_width_cm"), 0.0)
    expected_x = target_features.get("candidate_target_score_components", {}).get("expected_image_x")
    expected_x = None if expected_x is None else _safe_float(expected_x, 0.5)
    bbox_cx_norm = ((x1 + x2) * 0.5) / 640.0
    if expected_x is not None:
        delta = abs(float(bbox_cx_norm) - float(expected_x))
        center_in_target_bbox = int(delta <= 0.08)
        near_target_bbox = int(delta <= 0.16)
    else:
        center_in_target_bbox = 0
        near_target_bbox = 0
    target_side_match = _safe_float(
        target_features.get("candidate_target_score_components", {}).get("image_side_score"),
        0.0,
    )
    candidate_house_id = _house_id_to_int(target_features.get("candidate_house_id") or target_context.get("target_house_id"))
    if not bool(target_features.get("candidate_is_target_house_entry", False)):
        candidate_house_id = -1

    return {
        "candidate_id": rank,
        "raw_candidate_id": _safe_int(target_features.get("candidate_id"), rank),
        "valid_mask": 1,
        "class_name": class_name,
        "class_onehot": _class_onehot(class_name),
        "confidence": round(_clamp(_safe_float(detection.get("confidence"), 0.0), 0.0, 1.0), 6),
        "bbox_cx": round(bbox_cx_norm, 6),
        "bbox_cy": round(((y1 + y2) * 0.5) / 480.0, 6),
        "bbox_w": round(bbox_w / 640.0, 6),
        "bbox_h": round(bbox_h / 480.0, 6),
        "bbox_area_ratio": round((bbox_w * bbox_h) / float(640 * 480), 6),
        "aspect_ratio": round(bbox_w / max(1.0, bbox_h), 6),
        "entry_distance_cm": round(entry_distance_cm, 4),
        "surrounding_depth_cm": round(surrounding_depth_cm, 4),
        "clearance_depth_cm": round(clearance_depth_cm, 4),
        "depth_gain_cm": round(depth_gain_cm, 4),
        "opening_width_cm": round(opening_width_cm, 4),
        "traversable": int(bool(depth_features.get("traversable", False))),
        "crossing_ready": int(bool(depth_features.get("crossing_ready", False))),
        "candidate_rank": rank,
        "entry_distance_norm": round(_clamp(entry_distance_cm / MAX_DEPTH_CM, 0.0, 1.0), 6),
        "surrounding_depth_norm": round(_clamp(surrounding_depth_cm / MAX_DEPTH_CM, 0.0, 1.0), 6),
        "clearance_depth_norm": round(_clamp(clearance_depth_cm / MAX_DEPTH_CM, 0.0, 1.0), 6),
        "depth_gain_norm": round(_clamp(depth_gain_cm / MAX_DEPTH_GAIN_CM, 0.0, 1.0), 6),
        "opening_width_norm": round(_clamp(opening_width_cm / MAX_OPENING_WIDTH_CM, 0.0, 1.0), 6),
        "source": str(depth_features.get("source") or ""),
        "source_slot": rank,
        "match_iou": round(_safe_float(depth_features.get("match_iou"), 0.0), 6),
        "candidate_target_match_score": round(_clamp(_safe_float(target_features.get("candidate_target_match_score"), 0.0), 0.0, 1.0), 6),
        "candidate_semantic_score": round(_clamp(_safe_float(target_features.get("candidate_semantic_score"), 0.0), 0.0, 1.0), 6),
        "candidate_geometry_score": round(_clamp(_safe_float(target_features.get("candidate_geometry_score"), 0.0), 0.0, 1.0), 6),
        "candidate_total_score": round(_clamp(_safe_float(target_features.get("candidate_total_score"), 0.0), 0.0, 1.0), 6),
        "candidate_house_id": candidate_house_id,
        "candidate_is_target_house_entry": int(bool(target_features.get("candidate_is_target_house_entry", False))),
        "candidate_target_side_match": round(_clamp(target_side_match, 0.0, 1.0), 6),
        "candidate_center_in_target_bbox": center_in_target_bbox,
        "candidate_near_target_bbox": near_target_bbox,
        "candidate_image_side": str(target_features.get("candidate_image_side") or ""),
        "target_expected_side": str(target_features.get("candidate_target_score_components", {}).get("expected_side") or ""),
        "target_expected_image_x": round(_clamp(_safe_float(expected_x, 0.5), 0.0, 1.0), 6),
    }


def _build_teacher_targets(labeling_dir: Path) -> Dict[str, Any]:
    teacher_output_path = labeling_dir / "teacher_output.json"
    if not teacher_output_path.exists():
        return {
            "teacher_available": 0,
            "entry_state": "",
            "subgoal": "",
            "action_hint": "",
            "target_candidate_id": -1,
            "risk_level": "",
            "teacher_reason_text": "",
            "teacher_reason_embedding": [],
            "confidence": 0.0,
            "target_conditioned_teacher_available": 0,
            "target_conditioned_state": "",
            "target_conditioned_subgoal": "",
            "target_conditioned_action_hint": "",
            "target_conditioned_target_candidate_id": -1,
            "target_conditioned_reason_text": "",
            "target_conditioned_reason_embedding": [],
            "target_conditioned_confidence": 0.0,
        }

    teacher_payload = _read_json(teacher_output_path)
    teacher_output = teacher_payload.get("teacher_output", {}) if isinstance(teacher_payload.get("teacher_output"), dict) else {}
    return {
        "teacher_available": 1,
        "entry_state": str(teacher_output.get("entry_state") or ""),
        "subgoal": str(teacher_output.get("subgoal") or ""),
        "action_hint": str(teacher_output.get("action_hint") or ""),
        "target_candidate_id": _safe_int(teacher_output.get("target_candidate_id"), -1),
        "risk_level": str(teacher_output.get("risk_level") or ""),
        "teacher_reason_text": str(teacher_output.get("reason") or ""),
        "teacher_reason_embedding": [],
        "confidence": round(_clamp(_safe_float(teacher_output.get("confidence"), 0.0), 0.0, 1.0), 6),
        "target_conditioned_teacher_available": int(bool(teacher_output.get("target_conditioned_state"))),
        "target_conditioned_state": str(teacher_output.get("target_conditioned_state") or ""),
        "target_conditioned_subgoal": str(teacher_output.get("target_conditioned_subgoal") or ""),
        "target_conditioned_action_hint": str(teacher_output.get("target_conditioned_action_hint") or ""),
        "target_conditioned_target_candidate_id": _safe_int(teacher_output.get("target_conditioned_candidate_id"), -1),
        "target_conditioned_reason_text": str(teacher_output.get("target_conditioned_reason") or ""),
        "target_conditioned_reason_embedding": [],
        "target_conditioned_confidence": round(
            _clamp(_safe_float(teacher_output.get("target_confidence"), 0.0), 0.0, 1.0), 6
        ),
    }


def build_entry_state_for_labeling_dir(labeling_dir: Path) -> Dict[str, Any]:
    labeling_dir = labeling_dir.resolve()
    sample_id = labeling_dir.parent.name
    yolo_result = _read_json(labeling_dir / "yolo_result.json")
    depth_result = _read_json(labeling_dir / "depth_result.json")
    fusion_result = _read_json(labeling_dir / "fusion_result.json")
    state_excerpt = _read_json(labeling_dir / "state_excerpt.json")
    pose_history = _read_json(labeling_dir / "pose_history_summary.json")

    fusion = fusion_result.get("fusion", {}) if isinstance(fusion_result.get("fusion"), dict) else fusion_result
    depth_analysis = depth_result.get("analysis", {}) if isinstance(depth_result.get("analysis"), dict) else depth_result
    detections = yolo_result.get("detections", []) if isinstance(yolo_result.get("detections"), list) else []
    sorted_detections = sorted(detections, key=_sort_detection_key)

    candidates: List[Dict[str, Any]] = []
    for rank, detection in enumerate(sorted_detections[:TOP_K_CANDIDATES]):
        candidates.append(_build_candidate(detection, fusion, depth_analysis, rank))
    while len(candidates) < TOP_K_CANDIDATES:
        candidates.append(_zero_candidate(len(candidates)))

    global_state = _build_global_state(state_excerpt, fusion, depth_analysis, pose_history)
    teacher_targets = _build_teacher_targets(labeling_dir)
    metadata = {
        "sample_id": sample_id,
        "task_label": str(state_excerpt.get("task_label") or ""),
        "source_run_dir": str(labeling_dir.parent),
        "labeling_dir": str(labeling_dir),
        "rgb_path": str(labeling_dir / "rgb.png"),
        "depth_cm_path": str(labeling_dir / "depth_cm.png"),
        "depth_preview_path": str(labeling_dir / "depth_preview.png"),
        "teacher_source": str((labeling_dir / "teacher_output.json").name) if (labeling_dir / "teacher_output.json").exists() else "",
        "fusion_source": "fusion_result.json",
        "top_k": TOP_K_CANDIDATES,
        "target_conditioning_enabled": int(bool(fusion.get("target_context"))),
        "target_house_id": str((fusion.get("target_context") or {}).get("target_house_id") or ""),
    }

    entry_state = {
        "sample_id": sample_id,
        "global_state": global_state,
        "candidates": candidates,
        "teacher_targets": teacher_targets,
        "metadata": metadata,
    }
    _write_json(labeling_dir / "entry_state.json", entry_state)
    return {
        "sample_id": sample_id,
        "entry_state_path": str(labeling_dir / "entry_state.json"),
        "teacher_available": teacher_targets.get("teacher_available", 0),
        "target_conditioned_teacher_available": teacher_targets.get("target_conditioned_teacher_available", 0),
        "top_candidate_class": candidates[0]["class_name"] if candidates and candidates[0]["valid_mask"] else "",
        "top_candidate_traversable": candidates[0]["traversable"] if candidates else 0,
        "target_candidate_id": teacher_targets.get("target_candidate_id", -1),
        "target_conditioned_state": teacher_targets.get("target_conditioned_state", ""),
    }


def _discover_labeling_dirs(results_root: Path) -> List[Path]:
    output: List[Path] = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        labeling_dir = child / "labeling"
        if labeling_dir.is_dir():
            output.append(labeling_dir)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build candidate-level entry_state.json files from Phase 2 multimodal fusion labeling packages."
    )
    parser.add_argument("--results_root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--labeling_dir", type=Path, default=None, help="Build one labeling directory only.")
    parser.add_argument("--only_dir", type=str, default="", help="Only process one fusion_xxx directory under results_root.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.labeling_dir:
        summary = build_entry_state_for_labeling_dir(args.labeling_dir)
        print(
            f"[entry-state] done -> {summary['sample_id']} "
            f"(teacher={summary['teacher_available']}; top={summary['top_candidate_class']}; "
            f"trav={summary['top_candidate_traversable']}; "
            f"target_teacher={summary['target_conditioned_teacher_available']}; "
            f"target_state={summary['target_conditioned_state']})"
        )
        print(f"[entry-state] entry_state -> {summary['entry_state_path']}")
        return

    results_root = args.results_root.resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    labeling_dirs = _discover_labeling_dirs(results_root)
    if args.only_dir:
        labeling_dirs = [path for path in labeling_dirs if path.parent.name == args.only_dir]
        if not labeling_dirs:
            raise FileNotFoundError(f"Could not find run directory named: {args.only_dir}")

    total = len(labeling_dirs)
    print(f"[entry-state] discovered {total} labeling directories")
    summary_items: List[Dict[str, Any]] = []
    ok_count = 0
    error_count = 0
    top_candidate_class_counts: Counter[str] = Counter()
    target_conditioned_state_counts: Counter[str] = Counter()
    for idx, labeling_dir in enumerate(labeling_dirs, start=1):
        sample_name = labeling_dir.parent.name
        print(f"[{idx}/{total}] start -> {sample_name}")
        try:
            summary = build_entry_state_for_labeling_dir(labeling_dir)
            print(
                f"[{idx}/{total}] done -> {sample_name} "
                f"(teacher={summary['teacher_available']}; top={summary['top_candidate_class']}; "
                f"trav={summary['top_candidate_traversable']}; "
                f"target_teacher={summary['target_conditioned_teacher_available']}; "
                f"target_state={summary['target_conditioned_state']})"
            )
            summary_items.append({"run_dir": str(labeling_dir.parent), "status": "ok", **summary})
            ok_count += 1
            top_candidate_class_counts[str(summary.get("top_candidate_class") or "")] += 1
            target_conditioned_state_counts[str(summary.get("target_conditioned_state") or "")] += 1
        except Exception as exc:
            print(f"[{idx}/{total}] error -> {sample_name}: {exc}")
            summary_items.append(
                {
                    "run_dir": str(labeling_dir.parent),
                    "sample_id": sample_name,
                    "status": "error",
                    "error": str(exc),
                }
            )
            error_count += 1

    summary_path = results_root / f"entry_state_builder_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _write_json(
        summary_path,
        {
            "results_root": str(results_root),
            "ok_count": ok_count,
            "error_count": error_count,
            "top_candidate_class_counts": dict(top_candidate_class_counts),
            "target_conditioned_state_counts": dict(target_conditioned_state_counts),
            "items": summary_items,
        },
    )
    print(f"[entry-state] finished: ok={ok_count} error={error_count}")
    print(f"[entry-state] summary -> {summary_path}")


if __name__ == "__main__":
    main()
