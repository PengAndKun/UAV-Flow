from __future__ import annotations

from typing import Any, Dict, List, Mapping

from .label_schema import (
    DEFAULT_TOP_K_CANDIDATES,
    encode_action_hint,
    encode_entry_state,
    encode_subgoal,
    encode_target_action,
    encode_target_candidate_id,
    encode_target_state,
    encode_target_subgoal,
)


SEVERITY_LABELS = ("clear", "low", "medium", "high", "unknown")
SIDE_LABELS = ("left", "center", "right", "out_of_view", "unknown")
SOURCE_LABELS = ("semantic_region", "depth_candidate", "none", "unknown")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _onehot(label: str, labels: tuple[str, ...]) -> List[float]:
    value = str(label or "").strip()
    index = labels.index(value) if value in labels else labels.index("unknown")
    output = [0.0] * len(labels)
    output[index] = 1.0
    return output


def _normalize_binary(value: Any) -> float:
    return 1.0 if _safe_int(value, 0) != 0 else 0.0


def _normalize_ratio(value: Any) -> float:
    return _clamp(_safe_float(value, 0.0), 0.0, 1.0)


def _normalize_aspect_ratio(value: Any) -> float:
    return _clamp(_safe_float(value, 0.0), 0.0, 4.0) / 4.0


def build_global_feature_vector(sample: Mapping[str, Any]) -> Dict[str, Any]:
    global_state = sample.get("global_state", {}) if isinstance(sample.get("global_state"), dict) else {}
    metadata = sample.get("metadata", {}) if isinstance(sample.get("metadata"), dict) else {}

    target_house_id = _safe_int(global_state.get("target_house_id"), -1)
    current_house_id = _safe_int(global_state.get("current_house_id"), -1)
    history_actions = global_state.get("history_actions", [])
    history_len = len(history_actions) if isinstance(history_actions, list) else 0

    severity_onehot = _onehot(str(global_state.get("front_obstacle_severity") or "unknown"), SEVERITY_LABELS)
    target_side_onehot = _onehot(
        str(global_state.get("target_house_expected_side_text") or "unknown"),
        SIDE_LABELS,
    )

    features: List[float] = [
        _clamp(_safe_float(global_state.get("yaw_sin"), 0.0), -1.0, 1.0),
        _clamp(_safe_float(global_state.get("yaw_cos"), 0.0), -1.0, 1.0),
        _normalize_binary(global_state.get("front_obstacle_present")),
        _normalize_ratio(global_state.get("front_min_depth_norm")),
        *severity_onehot,
        _normalize_binary(global_state.get("target_house_known_mask")),
        _normalize_binary(global_state.get("target_house_in_fov")),
        *target_side_onehot,
        _normalize_ratio(global_state.get("target_house_expected_image_x")),
        _normalize_binary(global_state.get("target_house_bbox_available_mask")),
        _normalize_ratio(global_state.get("target_distance_norm")),
        _clamp(_safe_float(global_state.get("target_bearing_sin"), 0.0), -1.0, 1.0),
        _clamp(_safe_float(global_state.get("target_bearing_cos"), 0.0), -1.0, 1.0),
        _normalize_binary(global_state.get("movement_enabled")),
        _normalize_binary(1 if current_house_id >= 0 else 0),
        _normalize_binary(1 if target_house_id >= 0 and target_house_id == current_house_id else 0),
        _clamp(float(history_len), 0.0, 10.0) / 10.0,
        _normalize_binary(metadata.get("target_conditioning_enabled", 1)),
    ]

    feature_names = [
        "yaw_sin",
        "yaw_cos",
        "front_obstacle_present",
        "front_min_depth_norm",
        "front_obstacle_severity_clear",
        "front_obstacle_severity_low",
        "front_obstacle_severity_medium",
        "front_obstacle_severity_high",
        "front_obstacle_severity_unknown",
        "target_house_known_mask",
        "target_house_in_fov",
        "target_side_left",
        "target_side_center",
        "target_side_right",
        "target_side_out_of_view",
        "target_side_unknown",
        "target_house_expected_image_x",
        "target_house_bbox_available_mask",
        "target_distance_norm",
        "target_bearing_sin",
        "target_bearing_cos",
        "movement_enabled",
        "current_house_known_mask",
        "current_equals_target_mask",
        "history_len_norm",
        "target_conditioning_enabled",
    ]

    return {
        "vector": features,
        "feature_names": feature_names,
        "target_house_id": target_house_id,
        "current_house_id": current_house_id,
    }


def build_candidate_feature_vector(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    class_onehot = candidate.get("class_onehot", [])
    if not isinstance(class_onehot, list):
        class_onehot = []
    class_onehot = [float(_safe_float(value, 0.0)) for value in class_onehot[:4]]
    if len(class_onehot) < 4:
        class_onehot.extend([0.0] * (4 - len(class_onehot)))

    source_onehot = _onehot(str(candidate.get("source") or "unknown"), SOURCE_LABELS)
    candidate_side_onehot = _onehot(str(candidate.get("candidate_image_side") or "unknown"), SIDE_LABELS)
    target_side_onehot = _onehot(str(candidate.get("target_expected_side") or "unknown"), SIDE_LABELS)

    features: List[float] = [
        _normalize_binary(candidate.get("valid_mask")),
        *class_onehot,
        _normalize_ratio(candidate.get("confidence")),
        _normalize_ratio(candidate.get("bbox_cx")),
        _normalize_ratio(candidate.get("bbox_cy")),
        _normalize_ratio(candidate.get("bbox_w")),
        _normalize_ratio(candidate.get("bbox_h")),
        _normalize_ratio(candidate.get("bbox_area_ratio")),
        _normalize_aspect_ratio(candidate.get("aspect_ratio")),
        _normalize_ratio(candidate.get("entry_distance_norm")),
        _normalize_ratio(candidate.get("surrounding_depth_norm")),
        _normalize_ratio(candidate.get("clearance_depth_norm")),
        _normalize_ratio(candidate.get("depth_gain_norm")),
        _normalize_ratio(candidate.get("opening_width_norm")),
        _normalize_binary(candidate.get("traversable")),
        _normalize_binary(candidate.get("crossing_ready")),
        _normalize_ratio(candidate.get("candidate_rank", 0) / max(1, DEFAULT_TOP_K_CANDIDATES - 1)),
        _normalize_ratio(candidate.get("match_iou")),
        _normalize_ratio(candidate.get("candidate_target_match_score")),
        _normalize_ratio(candidate.get("candidate_semantic_score")),
        _normalize_ratio(candidate.get("candidate_geometry_score")),
        _normalize_ratio(candidate.get("candidate_total_score")),
        _normalize_binary(candidate.get("candidate_is_target_house_entry")),
        _normalize_ratio(candidate.get("candidate_target_side_match")),
        _normalize_binary(candidate.get("candidate_center_in_target_bbox")),
        _normalize_binary(candidate.get("candidate_near_target_bbox")),
        *candidate_side_onehot,
        *target_side_onehot,
        _normalize_ratio(candidate.get("target_expected_image_x")),
        *source_onehot,
    ]

    feature_names = [
        "valid_mask",
        "class_open_door",
        "class_door",
        "class_close_door",
        "class_window",
        "confidence",
        "bbox_cx",
        "bbox_cy",
        "bbox_w",
        "bbox_h",
        "bbox_area_ratio",
        "aspect_ratio_norm",
        "entry_distance_norm",
        "surrounding_depth_norm",
        "clearance_depth_norm",
        "depth_gain_norm",
        "opening_width_norm",
        "traversable",
        "crossing_ready",
        "candidate_rank_norm",
        "match_iou",
        "candidate_target_match_score",
        "candidate_semantic_score",
        "candidate_geometry_score",
        "candidate_total_score",
        "candidate_is_target_house_entry",
        "candidate_target_side_match",
        "candidate_center_in_target_bbox",
        "candidate_near_target_bbox",
        "candidate_side_left",
        "candidate_side_center",
        "candidate_side_right",
        "candidate_side_out_of_view",
        "candidate_side_unknown",
        "target_side_left",
        "target_side_center",
        "target_side_right",
        "target_side_out_of_view",
        "target_side_unknown",
        "target_expected_image_x",
        "source_semantic_region",
        "source_depth_candidate",
        "source_none",
        "source_unknown",
    ]

    return {
        "vector": features,
        "feature_names": feature_names,
    }


def build_teacher_targets(sample: Mapping[str, Any]) -> Dict[str, Any]:
    teacher_targets = sample.get("teacher_targets", {}) if isinstance(sample.get("teacher_targets"), dict) else {}

    teacher_available = _safe_int(teacher_targets.get("teacher_available"), 0)
    target_teacher_available = _safe_int(teacher_targets.get("target_conditioned_teacher_available"), 0)

    labels = {
        "entry_state": encode_entry_state(teacher_targets.get("entry_state")) if teacher_available else -1,
        "subgoal": encode_subgoal(teacher_targets.get("subgoal")) if teacher_available else -1,
        "action_hint": encode_action_hint(teacher_targets.get("action_hint")) if teacher_available else -1,
        "target_conditioned_state": (
            encode_target_state(teacher_targets.get("target_conditioned_state"))
            if target_teacher_available
            else -1
        ),
        "target_conditioned_subgoal": (
            encode_target_subgoal(teacher_targets.get("target_conditioned_subgoal"))
            if target_teacher_available
            else -1
        ),
        "target_conditioned_action_hint": (
            encode_target_action(teacher_targets.get("target_conditioned_action_hint"))
            if target_teacher_available
            else -1
        ),
        "target_conditioned_target_candidate_id": (
            encode_target_candidate_id(teacher_targets.get("target_conditioned_target_candidate_id"))
            if target_teacher_available
            else encode_target_candidate_id(-1)
        ),
    }

    masks = {
        "teacher_available": float(1 if teacher_available else 0),
        "target_conditioned_teacher_available": float(1 if target_teacher_available else 0),
    }

    confidences = {
        "teacher_confidence": _clamp(_safe_float(teacher_targets.get("confidence"), 0.0), 0.0, 1.0),
        "target_conditioned_confidence": _clamp(
            _safe_float(teacher_targets.get("target_conditioned_confidence"), 0.0), 0.0, 1.0
        ),
    }

    return {
        "labels": labels,
        "masks": masks,
        "confidences": confidences,
    }


def build_training_example(sample: Mapping[str, Any], *, top_k: int = DEFAULT_TOP_K_CANDIDATES) -> Dict[str, Any]:
    global_payload = build_global_feature_vector(sample)

    candidates = sample.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []
    candidate_payloads = [
        build_candidate_feature_vector(candidate if isinstance(candidate, dict) else {})
        for candidate in candidates[:top_k]
    ]
    while len(candidate_payloads) < int(top_k):
        candidate_payloads.append(build_candidate_feature_vector({}))

    candidate_vectors = [payload["vector"] for payload in candidate_payloads]
    candidate_masks = [payload["vector"][0] for payload in candidate_payloads]
    candidate_feature_names = candidate_payloads[0]["feature_names"] if candidate_payloads else []

    teacher_payload = build_teacher_targets(sample)

    return {
        "sample_id": str(sample.get("sample_id") or ""),
        "split": str(sample.get("split") or ""),
        "global_features": global_payload["vector"],
        "global_feature_names": global_payload["feature_names"],
        "candidate_features": candidate_vectors,
        "candidate_feature_names": candidate_feature_names,
        "candidate_valid_mask": candidate_masks,
        "labels": teacher_payload["labels"],
        "label_masks": teacher_payload["masks"],
        "label_confidences": teacher_payload["confidences"],
        "metadata": {
            "target_house_id": global_payload["target_house_id"],
            "current_house_id": global_payload["current_house_id"],
            "num_candidates": len(candidates),
        },
    }

