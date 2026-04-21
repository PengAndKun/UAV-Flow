from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ENTRY_STATES = {
    "enterable_open_door",
    "enterable_door",
    "visible_but_blocked_entry",
    "front_blocked_detour",
    "window_visible_keep_search",
    "geometric_opening_needs_confirmation",
    "no_entry_confirmed",
}

SUBGOALS = {
    "keep_search",
    "approach_entry",
    "align_entry",
    "detour_left",
    "detour_right",
    "cross_entry",
    "backoff_and_reobserve",
}

ACTION_HINTS = {
    "forward",
    "yaw_left",
    "yaw_right",
    "left",
    "right",
    "backward",
    "hold",
}

RISK_LEVELS = {"low", "medium", "high"}

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

BLOCKED_STATES = {"visible_but_blocked_entry", "front_blocked_detour"}
ENTERABLE_STATES = {"enterable_open_door", "enterable_door"}
SEARCH_STATES = {"window_visible_keep_search", "geometric_opening_needs_confirmation", "no_entry_confirmed"}

RESULTS_ROOT = Path(__file__).resolve().parent / "results"

TARGET_CONDITIONED_STATES = {
    "target_house_not_in_view",
    "target_house_visible_keep_search",
    "target_house_entry_visible",
    "target_house_entry_approachable",
    "target_house_entry_blocked",
    "non_target_house_entry_visible",
    "target_house_geometric_opening_needs_confirmation",
}

TARGET_CONDITIONED_SUBGOALS = {
    "reorient_to_target_house",
    "keep_search_target_house",
    "approach_target_entry",
    "align_target_entry",
    "detour_left_to_target_entry",
    "detour_right_to_target_entry",
    "cross_target_entry",
    "ignore_non_target_entry",
    "backoff_and_reobserve",
}


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_target_house_review(labeling_dir: Path) -> Dict[str, Any]:
    review_path = labeling_dir / "target_house_review.json"
    if not review_path.exists():
        return {}
    try:
        value = _read_json(review_path)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _normalize_token(value: Any) -> str:
    token = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    token = re.sub(r"[^a-z0-9_]+", "", token)
    return token


def _canonical_entry_state(value: Any) -> str:
    token = _normalize_token(value)
    mapping = {
        "enterable_open_door": "enterable_open_door",
        "enterable_door": "enterable_door",
        "visible_but_blocked_entry": "visible_but_blocked_entry",
        "front_blocked_detour": "front_blocked_detour",
        "window_visible_keep_search": "window_visible_keep_search",
        "geometric_opening_needs_confirmation": "geometric_opening_needs_confirmation",
        "no_entry_confirmed": "no_entry_confirmed",
    }
    return mapping.get(token, token)


def _canonical_subgoal(value: Any) -> str:
    token = _normalize_token(value)
    mapping = {
        "keep_search": "keep_search",
        "approach_entry": "approach_entry",
        "align_entry": "align_entry",
        "detour_left": "detour_left",
        "detour_right": "detour_right",
        "cross_entry": "cross_entry",
        "backoff_and_reobserve": "backoff_and_reobserve",
        "outside_localization": "keep_search",
        "entry_search": "keep_search",
        "target_house_verification": "keep_search",
    }
    return mapping.get(token, token)


def _canonical_action_hint(value: Any) -> str:
    token = _normalize_token(value)
    mapping = {
        "forward": "forward",
        "yaw_left": "yaw_left",
        "turn_left": "yaw_left",
        "yaw_right": "yaw_right",
        "turn_right": "yaw_right",
        "left": "left",
        "right": "right",
        "backward": "backward",
        "hold": "hold",
        "stop": "hold",
    }
    return mapping.get(token, token)


def _canonical_risk_level(value: Any) -> str:
    token = _normalize_token(value)
    mapping = {
        "low": "low",
        "medium": "medium",
        "mid": "medium",
        "high": "high",
    }
    return mapping.get(token, token)


def _canonical_target_conditioned_state(value: Any) -> str:
    token = _normalize_token(value)
    mapping = {
        "target_house_not_in_view": "target_house_not_in_view",
        "target_house_visible_keep_search": "target_house_visible_keep_search",
        "target_house_entry_visible": "target_house_entry_visible",
        "target_house_entry_approachable": "target_house_entry_approachable",
        "target_house_entry_blocked": "target_house_entry_blocked",
        "non_target_house_entry_visible": "non_target_house_entry_visible",
        "target_house_geometric_opening_needs_confirmation": "target_house_geometric_opening_needs_confirmation",
    }
    return mapping.get(token, token)


def _canonical_target_conditioned_subgoal(value: Any) -> str:
    token = _normalize_token(value)
    mapping = {
        "reorient_to_target_house": "reorient_to_target_house",
        "keep_search_target_house": "keep_search_target_house",
        "approach_target_entry": "approach_target_entry",
        "align_target_entry": "align_target_entry",
        "detour_left_to_target_entry": "detour_left_to_target_entry",
        "detour_right_to_target_entry": "detour_right_to_target_entry",
        "cross_target_entry": "cross_target_entry",
        "ignore_non_target_entry": "ignore_non_target_entry",
        "backoff_and_reobserve": "backoff_and_reobserve",
    }
    return mapping.get(token, token)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _short_reason_from_state(entry_state: str, crossing_ready: bool, front_severity: str, yolo_class: str) -> str:
    if front_severity == "high" or entry_state == "front_blocked_detour":
        return "Front obstacle is too close; avoid pushing forward."
    if entry_state == "window_visible_keep_search":
        return "Visible opening is a window, not a valid entry."
    if entry_state == "visible_but_blocked_entry":
        return "Entry is visible but the current corridor is blocked."
    if entry_state == "enterable_open_door":
        if crossing_ready:
            return "Open door is clear and ready to cross."
        return "Open door is visible and traversable, but it is still far away."
    if entry_state == "enterable_door":
        if crossing_ready:
            return "Door is visible, clear, and ready to cross."
        return "Door is visible and traversable, but it still needs approach."
    if entry_state == "geometric_opening_needs_confirmation":
        return "A geometric opening exists, but semantic confirmation is still weak."
    if yolo_class in WINDOW_CLASSES:
        return "Window evidence is stronger than entry evidence in this view."
    return "No reliable entry is confirmed in the current view."


def _pick_target_candidate_id(entry_state: str, detections: List[Dict[str, Any]]) -> int:
    preferred_classes: Tuple[str, ...]
    if entry_state in ENTERABLE_STATES or entry_state == "visible_but_blocked_entry":
        preferred_classes = tuple(DOORLIKE_CLASSES)
    elif entry_state == "window_visible_keep_search":
        preferred_classes = tuple(WINDOW_CLASSES)
    else:
        return -1

    best_idx = -1
    best_conf = -1.0
    for idx, detection in enumerate(detections):
        cls_name = str(detection.get("class_name_normalized") or detection.get("class_name") or "").strip().lower()
        conf = _safe_float(detection.get("confidence"), 0.0)
        if cls_name in preferred_classes and conf > best_conf:
            best_conf = conf
            best_idx = idx
    return best_idx


def _choose_search_action(entry_state: str, target_candidate_id: int, detections: List[Dict[str, Any]]) -> str:
    if entry_state == "window_visible_keep_search" and target_candidate_id >= 0 and target_candidate_id < len(detections):
        det = detections[target_candidate_id]
        xyxy = det.get("xyxy", [])
        if isinstance(xyxy, list) and len(xyxy) == 4:
            center_x = 0.5 * (_safe_float(xyxy[0]) + _safe_float(xyxy[2]))
            if center_x < 300:
                return "yaw_left"
            if center_x > 340:
                return "yaw_right"
    return "hold"


def _subgoal_to_action(subgoal: str, entry_state: str, target_candidate_id: int, detections: List[Dict[str, Any]]) -> str:
    mapping = {
        "approach_entry": "forward",
        "align_entry": "hold",
        "detour_left": "left",
        "detour_right": "right",
        "cross_entry": "forward",
        "backoff_and_reobserve": "backward",
    }
    if subgoal in mapping:
        return mapping[subgoal]
    return _choose_search_action(entry_state, target_candidate_id, detections)


def _target_conditioned_subgoal_to_action(
    subgoal: str,
    target_state: str,
    fusion_result: Dict[str, Any],
) -> str:
    mapping = {
        "approach_target_entry": "forward",
        "align_target_entry": "hold",
        "detour_left_to_target_entry": "left",
        "detour_right_to_target_entry": "right",
        "cross_target_entry": "forward",
        "backoff_and_reobserve": "backward",
        "ignore_non_target_entry": "hold",
    }
    if subgoal in mapping:
        return mapping[subgoal]

    if subgoal == "reorient_to_target_house":
        action_hint = _canonical_action_hint(fusion_result.get("target_conditioned_action_hint"))
        if action_hint in ACTION_HINTS:
            return action_hint
        expected_side = str(
            ((fusion_result.get("target_context") or {}) if isinstance(fusion_result.get("target_context"), dict) else {}).get(
                "target_house_expected_side"
            )
            or ""
        ).strip().lower()
        if expected_side == "left":
            return "yaw_left"
        if expected_side == "right":
            return "yaw_right"
        return "hold"

    if subgoal == "keep_search_target_house":
        action_hint = _canonical_action_hint(fusion_result.get("target_conditioned_action_hint"))
        if action_hint in ACTION_HINTS:
            return action_hint
        return "hold"

    if target_state == "target_house_entry_visible":
        return _canonical_action_hint(fusion_result.get("target_conditioned_action_hint")) or "hold"
    return "hold"


def _target_conditioned_reason_from_fusion(fusion_result: Dict[str, Any], fallback_reason: str) -> str:
    reason = str(fusion_result.get("target_conditioned_reason") or "").strip()
    if reason:
        return _truncate_reason(reason)
    return _truncate_reason(fallback_reason)


def _infer_entry_state(
    *,
    parsed: Dict[str, Any],
    fusion: Dict[str, Any],
    top_yolo_class: str,
    front_severity: str,
) -> str:
    entry_visible = bool(parsed.get("entry_door_visible", False))
    entry_traversable = bool(parsed.get("entry_door_traversable", False))
    fusion_state = _canonical_entry_state(fusion.get("final_entry_state"))

    if front_severity == "high":
        return "front_blocked_detour"
    if top_yolo_class in WINDOW_CLASSES:
        return "window_visible_keep_search"
    if entry_visible and entry_traversable:
        if top_yolo_class in OPEN_DOOR_CLASSES:
            return "enterable_open_door"
        return "enterable_door"
    if entry_visible and not entry_traversable:
        return "visible_but_blocked_entry"
    if fusion_state in ENTRY_STATES:
        return fusion_state
    return "no_entry_confirmed"


def _infer_subgoal(
    *,
    parsed: Dict[str, Any],
    entry_state: str,
    crossing_ready: bool,
) -> str:
    active_stage = _canonical_subgoal(parsed.get("active_stage"))
    if entry_state == "front_blocked_detour":
        return "backoff_and_reobserve"
    if entry_state == "visible_but_blocked_entry":
        return "backoff_and_reobserve"
    if entry_state == "window_visible_keep_search":
        return "keep_search"
    if entry_state == "no_entry_confirmed":
        return "keep_search"
    if entry_state == "geometric_opening_needs_confirmation":
        return "keep_search"
    if entry_state in ENTERABLE_STATES:
        if active_stage == "cross_entry" and crossing_ready:
            return "cross_entry"
        if active_stage in {"approach_entry", "align_entry"}:
            return active_stage
        return "approach_entry"
    return "keep_search"


def _infer_risk_level(entry_state: str, front_severity: str, crossing_ready: bool) -> str:
    if front_severity == "high" or entry_state in BLOCKED_STATES:
        return "high"
    if entry_state in SEARCH_STATES:
        return "medium"
    if entry_state in ENTERABLE_STATES and not crossing_ready:
        return "medium"
    return "low"


def _extract_top_yolo_class(yolo_result: Dict[str, Any]) -> str:
    detections = yolo_result.get("detections", [])
    if isinstance(detections, list) and detections:
        first = detections[0]
        return str(first.get("class_name_normalized") or first.get("class_name") or "").strip().lower()
    return ""


def _truncate_reason(text: str, max_chars: int = 180) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3].rstrip() + "..."


def normalize_teacher_output(
    *,
    teacher_payload: Dict[str, Any],
    fusion_result: Dict[str, Any],
    yolo_result: Dict[str, Any],
    depth_result: Dict[str, Any],
    state_excerpt: Optional[Dict[str, Any]] = None,
    target_house_review: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    parsed = teacher_payload.get("parsed", {}) if isinstance(teacher_payload.get("parsed"), dict) else {}
    detections = yolo_result.get("detections", []) if isinstance(yolo_result.get("detections"), list) else []
    top_yolo_class = _extract_top_yolo_class(yolo_result)
    front_obstacle = depth_result.get("front_obstacle", {}) if isinstance(depth_result.get("front_obstacle"), dict) else {}
    front_severity = str(front_obstacle.get("severity") or "").strip().lower()
    crossing_ready = bool(fusion_result.get("crossing_ready", False))
    target_context = fusion_result.get("target_context", {}) if isinstance(fusion_result.get("target_context"), dict) else {}
    fusion_target_state = _canonical_target_conditioned_state(fusion_result.get("target_conditioned_state"))
    fusion_target_subgoal = _canonical_target_conditioned_subgoal(fusion_result.get("target_conditioned_subgoal"))
    fusion_target_action = _canonical_action_hint(fusion_result.get("target_conditioned_action_hint"))
    best_target_candidate = (
        fusion_result.get("best_target_candidate", {}) if isinstance(fusion_result.get("best_target_candidate"), dict) else {}
    )

    entry_state = _infer_entry_state(
        parsed=parsed,
        fusion=fusion_result,
        top_yolo_class=top_yolo_class,
        front_severity=front_severity,
    )
    target_candidate_id = _pick_target_candidate_id(entry_state, detections)
    subgoal = _infer_subgoal(parsed=parsed, entry_state=entry_state, crossing_ready=crossing_ready)
    action_hint = _subgoal_to_action(subgoal, entry_state, target_candidate_id, detections)
    risk_level = _infer_risk_level(entry_state, front_severity, crossing_ready)

    scene_description = str(parsed.get("scene_description") or "").strip()
    reason = _short_reason_from_state(entry_state, crossing_ready, front_severity, top_yolo_class)
    if scene_description and entry_state in ENTERABLE_STATES and "far" in scene_description.lower():
        reason = "Door is visible and traversable, but it is still far away."
    reason = _truncate_reason(reason)

    confidence = _clamp(_safe_float(parsed.get("confidence"), 0.5), 0.0, 1.0)
    current_house_id = ""
    target_house_id = ""
    task_label = ""
    if isinstance(state_excerpt, dict):
        current_house_id = str(state_excerpt.get("current_house_id") or state_excerpt.get("current_house") or "").strip()
        target_house_id = str(state_excerpt.get("target_house_id") or state_excerpt.get("target_house") or "").strip()
        task_label = str(state_excerpt.get("task_label") or "").strip()
    if not current_house_id:
        current_house_id = str(target_context.get("current_house_id") or "").strip()
    if not target_house_id:
        target_house_id = str(target_context.get("target_house_id") or "").strip()

    review_payload = target_house_review if isinstance(target_house_review, dict) else {}
    review_result = str(review_payload.get("review_result") or "").strip().lower()
    reviewed_house_id = str(review_payload.get("reviewed_house_id") or "").strip()
    reviewed_house_name = str(review_payload.get("reviewed_house_name") or "").strip()
    original_target_house_id = str(target_house_id or "").strip()
    target_house_review_applied = bool(reviewed_house_id)
    target_house_review_changed = bool(reviewed_house_id) and bool(original_target_house_id) and reviewed_house_id != original_target_house_id
    target_house_review_filled_missing = bool(reviewed_house_id) and not bool(original_target_house_id)
    if reviewed_house_id:
        target_house_id = reviewed_house_id

    target_house_in_fov = bool(target_context.get("target_house_in_fov", False))
    target_house_expected_side = str(target_context.get("target_house_expected_side") or "").strip().lower()
    target_conditioned_state = fusion_target_state if fusion_target_state in TARGET_CONDITIONED_STATES else ""
    if not target_conditioned_state:
        target_conditioned_state = "target_house_not_in_view" if not target_house_in_fov else "target_house_entry_visible"

    target_conditioned_subgoal = (
        fusion_target_subgoal if fusion_target_subgoal in TARGET_CONDITIONED_SUBGOALS else ""
    )
    if not target_conditioned_subgoal:
        fallback_mapping = {
            "target_house_not_in_view": "reorient_to_target_house",
            "target_house_visible_keep_search": "keep_search_target_house",
            "target_house_entry_visible": "keep_search_target_house",
            "target_house_entry_approachable": "approach_target_entry",
            "target_house_entry_blocked": "backoff_and_reobserve",
            "non_target_house_entry_visible": "ignore_non_target_entry",
            "target_house_geometric_opening_needs_confirmation": "keep_search_target_house",
        }
        target_conditioned_subgoal = fallback_mapping.get(target_conditioned_state, "keep_search_target_house")

    target_conditioned_candidate_id = best_target_candidate.get("candidate_id", -1)
    if not isinstance(target_conditioned_candidate_id, int):
        try:
            target_conditioned_candidate_id = int(target_conditioned_candidate_id)
        except (TypeError, ValueError):
            target_conditioned_candidate_id = -1

    target_conditioned_action_hint = fusion_target_action if fusion_target_action in ACTION_HINTS else ""
    if not target_conditioned_action_hint:
        target_conditioned_action_hint = _target_conditioned_subgoal_to_action(
            target_conditioned_subgoal,
            target_conditioned_state,
            fusion_result,
        )

    target_conditioned_reason = _target_conditioned_reason_from_fusion(fusion_result, reason)
    best_target_score = _safe_float(best_target_candidate.get("candidate_total_score"), confidence)
    target_confidence = _clamp(max(confidence, best_target_score), 0.0, 1.0)
    if target_conditioned_state in {"target_house_not_in_view", "target_house_entry_visible"}:
        target_confidence = _clamp(min(target_confidence, 0.85), 0.0, 1.0)
    if target_house_review_changed or target_house_review_filled_missing:
        target_conditioned_state = ""
        target_conditioned_subgoal = ""
        target_conditioned_action_hint = ""
        target_conditioned_candidate_id = -1
        target_conditioned_reason = ""
        target_confidence = 0.0

    normalized = {
        "entry_state": entry_state,
        "subgoal": subgoal,
        "action_hint": action_hint,
        "target_candidate_id": int(target_candidate_id),
        "risk_level": risk_level,
        "reason": reason,
        "confidence": confidence,
        "teacher_source_stage": str(parsed.get("active_stage") or "").strip(),
        "teacher_source_scene_state": str(parsed.get("scene_state") or "").strip(),
        "crossing_ready": bool(crossing_ready),
        "task_label": task_label,
        "current_house_id": current_house_id,
        "target_house_id": target_house_id,
        "original_target_house_id": original_target_house_id,
        "target_house_review_result": review_result,
        "reviewed_house_id": reviewed_house_id,
        "reviewed_house_name": reviewed_house_name,
        "target_house_review_applied": int(target_house_review_applied),
        "target_house_review_changed": int(target_house_review_changed),
        "target_house_review_filled_missing": int(target_house_review_filled_missing),
        "target_house_in_fov": target_house_in_fov,
        "target_house_expected_side": target_house_expected_side,
        "target_conditioned_state": target_conditioned_state,
        "target_conditioned_subgoal": target_conditioned_subgoal,
        "target_conditioned_action_hint": target_conditioned_action_hint,
        "target_conditioned_candidate_id": int(target_conditioned_candidate_id),
        "target_conditioned_reason": target_conditioned_reason,
        "target_confidence": round(float(target_confidence), 4),
    }
    return normalized


def _add_issue(items: List[Dict[str, str]], code: str, message: str) -> None:
    items.append({"code": code, "message": message})


def validate_teacher_output(
    *,
    teacher_output: Dict[str, Any],
    fusion_result: Dict[str, Any],
    yolo_result: Dict[str, Any],
    depth_result: Dict[str, Any],
) -> Dict[str, Any]:
    errors: List[Dict[str, str]] = []
    warnings: List[Dict[str, str]] = []
    score = 1.0

    required_fields = [
        "entry_state",
        "subgoal",
        "action_hint",
        "target_candidate_id",
        "risk_level",
        "reason",
        "confidence",
    ]
    for field in required_fields:
        if field not in teacher_output:
            _add_issue(errors, "missing_required_field", f"Missing required field: {field}")

    entry_state = _canonical_entry_state(teacher_output.get("entry_state"))
    subgoal = _canonical_subgoal(teacher_output.get("subgoal"))
    action_hint = _canonical_action_hint(teacher_output.get("action_hint"))
    risk_level = _canonical_risk_level(teacher_output.get("risk_level"))
    confidence = _safe_float(teacher_output.get("confidence"), -1.0)
    target_candidate_id = teacher_output.get("target_candidate_id")
    reason = str(teacher_output.get("reason") or "").strip()
    target_conditioned_state = _canonical_target_conditioned_state(teacher_output.get("target_conditioned_state"))
    target_conditioned_subgoal = _canonical_target_conditioned_subgoal(teacher_output.get("target_conditioned_subgoal"))
    target_conditioned_action_hint = _canonical_action_hint(teacher_output.get("target_conditioned_action_hint"))
    target_conditioned_candidate_id = teacher_output.get("target_conditioned_candidate_id")
    target_conditioned_reason = str(teacher_output.get("target_conditioned_reason") or "").strip()
    target_confidence = _safe_float(teacher_output.get("target_confidence"), -1.0)

    if entry_state not in ENTRY_STATES:
        _add_issue(errors, "invalid_entry_state", f"Unsupported entry_state: {teacher_output.get('entry_state')}")
    if subgoal not in SUBGOALS:
        _add_issue(errors, "invalid_subgoal", f"Unsupported subgoal: {teacher_output.get('subgoal')}")
    if action_hint not in ACTION_HINTS:
        _add_issue(errors, "invalid_action_hint", f"Unsupported action_hint: {teacher_output.get('action_hint')}")
    if risk_level not in RISK_LEVELS:
        _add_issue(errors, "invalid_risk_level", f"Unsupported risk_level: {teacher_output.get('risk_level')}")
    if not (0.0 <= confidence <= 1.0):
        _add_issue(errors, "invalid_confidence", f"Confidence must be in [0,1], got {confidence}")
    if not isinstance(target_candidate_id, int) or int(target_candidate_id) < -1:
        _add_issue(errors, "invalid_target_candidate_id", f"Invalid target_candidate_id: {target_candidate_id}")
    if target_conditioned_state and target_conditioned_state not in TARGET_CONDITIONED_STATES:
        _add_issue(
            errors,
            "invalid_target_conditioned_state",
            f"Unsupported target_conditioned_state: {teacher_output.get('target_conditioned_state')}",
        )
    if target_conditioned_subgoal and target_conditioned_subgoal not in TARGET_CONDITIONED_SUBGOALS:
        _add_issue(
            errors,
            "invalid_target_conditioned_subgoal",
            f"Unsupported target_conditioned_subgoal: {teacher_output.get('target_conditioned_subgoal')}",
        )
    if target_conditioned_action_hint and target_conditioned_action_hint not in ACTION_HINTS:
        _add_issue(
            errors,
            "invalid_target_conditioned_action_hint",
            f"Unsupported target_conditioned_action_hint: {teacher_output.get('target_conditioned_action_hint')}",
        )
    if target_conditioned_candidate_id is not None and (
        not isinstance(target_conditioned_candidate_id, int) or int(target_conditioned_candidate_id) < -1
    ):
        _add_issue(
            errors,
            "invalid_target_conditioned_candidate_id",
            f"Invalid target_conditioned_candidate_id: {target_conditioned_candidate_id}",
        )
    if target_confidence != -1.0 and not (0.0 <= target_confidence <= 1.0):
        _add_issue(errors, "invalid_target_confidence", f"target_confidence must be in [0,1], got {target_confidence}")

    detections = yolo_result.get("detections", []) if isinstance(yolo_result.get("detections"), list) else []
    if isinstance(target_candidate_id, int) and target_candidate_id >= len(detections) and detections:
        _add_issue(errors, "candidate_index_out_of_range", f"target_candidate_id {target_candidate_id} exceeds detections")
    if isinstance(target_conditioned_candidate_id, int) and target_conditioned_candidate_id >= len(detections) and detections:
        _add_issue(
            errors,
            "target_conditioned_candidate_index_out_of_range",
            f"target_conditioned_candidate_id {target_conditioned_candidate_id} exceeds detections",
        )

    front_obstacle = depth_result.get("front_obstacle", {}) if isinstance(depth_result.get("front_obstacle"), dict) else {}
    front_severity = str(front_obstacle.get("severity") or "").strip().lower()
    top_yolo_class = _extract_top_yolo_class(yolo_result)
    fusion_state = _canonical_entry_state(fusion_result.get("final_entry_state"))
    fusion_target_state = _canonical_target_conditioned_state(fusion_result.get("target_conditioned_state"))
    fusion_target_subgoal = _canonical_target_conditioned_subgoal(fusion_result.get("target_conditioned_subgoal"))
    fusion_target_action_hint = _canonical_action_hint(fusion_result.get("target_conditioned_action_hint"))
    target_context = fusion_result.get("target_context", {}) if isinstance(fusion_result.get("target_context"), dict) else {}
    target_house_in_fov = bool(target_context.get("target_house_in_fov", False))
    best_target_candidate_is_target = bool(fusion_result.get("best_target_candidate_is_target_house_entry", False))

    if entry_state == "window_visible_keep_search" and subgoal == "cross_entry":
        _add_issue(errors, "window_cross_conflict", "Window state cannot directly request cross_entry.")
    if entry_state == "window_visible_keep_search" and action_hint == "forward":
        _add_issue(errors, "window_forward_conflict", "Window state cannot use forward as the main action.")
    if entry_state == "front_blocked_detour" and action_hint == "forward":
        _add_issue(errors, "blocked_forward_conflict", "Blocked state cannot use forward as the main action.")
    if front_severity == "high" and subgoal == "cross_entry":
        _add_issue(errors, "high_obstacle_cross_conflict", "High front obstacle cannot request cross_entry.")
    if front_severity == "high" and action_hint == "forward":
        _add_issue(errors, "high_obstacle_forward_conflict", "High front obstacle cannot request forward as the main action.")
    if top_yolo_class in WINDOW_CLASSES and entry_state in ENTERABLE_STATES:
        _add_issue(errors, "window_marked_enterable", "Top semantic class is window, but teacher marked it as enterable.")
    if entry_state == "no_entry_confirmed" and subgoal == "cross_entry":
        _add_issue(errors, "no_entry_cross_conflict", "No-entry state cannot request cross_entry.")
    if target_conditioned_state == "target_house_not_in_view" and target_house_in_fov:
        _add_issue(
            errors,
            "target_in_view_state_conflict",
            "target_house_in_fov is true, but target_conditioned_state is target_house_not_in_view.",
        )
    if target_conditioned_state == "non_target_house_entry_visible" and best_target_candidate_is_target:
        _add_issue(
            errors,
            "non_target_state_conflict",
            "best_target_candidate is marked as target-house entry, but target_conditioned_state says non-target.",
        )
    if target_conditioned_state == "target_house_entry_approachable" and target_conditioned_action_hint != "forward":
        _add_issue(
            warnings,
            "approachable_without_forward",
            "Approachable target entry usually should bias toward forward.",
        )
        score -= 0.05
    if target_conditioned_state == "target_house_entry_blocked" and target_conditioned_action_hint not in {"left", "right", "backward"}:
        _add_issue(
            errors,
            "blocked_target_wrong_action",
            "Blocked target entry should detour or back off, not keep a neutral/forward action.",
        )
    if target_conditioned_state == "non_target_house_entry_visible" and target_conditioned_action_hint == "forward":
        _add_issue(
            errors,
            "non_target_forward_conflict",
            "Non-target entry should not directly request forward approach.",
        )
    if target_conditioned_subgoal == "reorient_to_target_house" and target_conditioned_action_hint not in {"yaw_left", "yaw_right", "hold"}:
        _add_issue(
            errors,
            "reorient_wrong_action",
            "reorient_to_target_house should use yaw_left, yaw_right, or hold.",
        )
    if target_conditioned_subgoal == "approach_target_entry" and target_conditioned_action_hint not in {"forward", "hold"}:
        _add_issue(
            warnings,
            "approach_target_nonforward",
            "approach_target_entry usually maps to forward or short hold.",
        )
        score -= 0.05

    if confidence < 0.55:
        _add_issue(warnings, "low_confidence", f"Teacher confidence is low: {confidence:.2f}")
        score -= 0.12
    if len(reason) < 12:
        _add_issue(warnings, "reason_too_short", "Reason text is too short to be informative.")
        score -= 0.05
    if len(reason) > 180:
        _add_issue(warnings, "reason_too_long", "Reason text is longer than the recommended limit.")
        score -= 0.05
    if entry_state in ENTERABLE_STATES and subgoal == "keep_search":
        _add_issue(warnings, "overly_conservative_subgoal", "Teacher stayed in keep_search despite enterable evidence.")
        score -= 0.08
    if target_candidate_id == -1 and entry_state in {"enterable_open_door", "enterable_door", "window_visible_keep_search"}:
        _add_issue(warnings, "target_candidate_missing", "Teacher did not pick a target candidate despite clear semantic evidence.")
        score -= 0.08
    if fusion_state and fusion_state != entry_state and not errors:
        _add_issue(
            warnings,
            "fusion_state_mismatch",
            f"Teacher entry_state ({entry_state}) differs from fusion final_entry_state ({fusion_state}).",
        )
        score -= 0.08
    if fusion_target_state and target_conditioned_state and fusion_target_state != target_conditioned_state and not errors:
        _add_issue(
            warnings,
            "fusion_target_state_mismatch",
            f"Teacher target_conditioned_state ({target_conditioned_state}) differs from fusion target_conditioned_state ({fusion_target_state}).",
        )
        score -= 0.08
    if fusion_target_subgoal and target_conditioned_subgoal and fusion_target_subgoal != target_conditioned_subgoal and not errors:
        _add_issue(
            warnings,
            "fusion_target_subgoal_mismatch",
            f"Teacher target_conditioned_subgoal ({target_conditioned_subgoal}) differs from fusion target_conditioned_subgoal ({fusion_target_subgoal}).",
        )
        score -= 0.06

    normalized = dict(teacher_output)
    normalized["entry_state"] = entry_state
    normalized["subgoal"] = subgoal
    normalized["action_hint"] = action_hint
    normalized["risk_level"] = risk_level
    normalized["confidence"] = _clamp(confidence, 0.0, 1.0)
    normalized["reason"] = _truncate_reason(reason or "")
    normalized["target_conditioned_state"] = target_conditioned_state
    normalized["target_conditioned_subgoal"] = target_conditioned_subgoal
    normalized["target_conditioned_action_hint"] = target_conditioned_action_hint
    normalized["target_conditioned_candidate_id"] = (
        int(target_conditioned_candidate_id) if isinstance(target_conditioned_candidate_id, int) else -1
    )
    normalized["target_conditioned_reason"] = _truncate_reason(target_conditioned_reason or "")
    normalized["target_confidence"] = _clamp(target_confidence if target_confidence != -1.0 else confidence, 0.0, 1.0)

    if errors:
        status = "invalid"
        score = min(score, 0.30)
    else:
        score = _clamp(score, 0.0, 1.0)
        if score >= 0.80:
            status = "valid"
        elif score >= 0.55:
            status = "weak_valid"
        else:
            status = "invalid"

    return {
        "status": status,
        "score": round(float(score), 4),
        "errors": errors,
        "warnings": warnings,
        "normalized_teacher_output": normalized,
        "evidence": {
            "fusion_final_entry_state": fusion_state,
            "fusion_target_conditioned_state": fusion_target_state,
            "fusion_target_conditioned_subgoal": fusion_target_subgoal,
            "fusion_target_conditioned_action_hint": fusion_target_action_hint,
            "top_yolo_class": top_yolo_class,
            "front_obstacle_severity": front_severity,
            "front_min_depth_cm": front_obstacle.get("front_min_depth_cm"),
            "target_house_in_fov": target_house_in_fov,
            "best_target_candidate_is_target_house_entry": best_target_candidate_is_target,
        },
    }


def pick_teacher_source_file(labeling_dir: Path, teacher_filename: str = "") -> Path:
    if teacher_filename:
        path = labeling_dir / teacher_filename
        if not path.exists():
            raise FileNotFoundError(f"Teacher file not found: {path}")
        return path

    candidates = sorted(
        [
            path
            for path in labeling_dir.glob("anthropic*_scene_result.json")
            if "_vs_labeling_compare" not in path.name
        ],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No anthropic *_scene_result.json found in {labeling_dir}")
    return candidates[0]


def validate_labeling_dir(labeling_dir: Path, teacher_filename: str = "") -> Dict[str, Any]:
    labeling_dir = labeling_dir.resolve()
    teacher_source_path = pick_teacher_source_file(labeling_dir, teacher_filename=teacher_filename)
    teacher_payload = _read_json(teacher_source_path)
    fusion_result = _read_json(labeling_dir / "fusion_result.json")
    yolo_result = _read_json(labeling_dir / "yolo_result.json")
    depth_result = _read_json(labeling_dir / "depth_result.json")
    state_excerpt_path = labeling_dir / "state_excerpt.json"
    state_excerpt = _read_json(state_excerpt_path) if state_excerpt_path.exists() else {}
    target_house_review = _read_target_house_review(labeling_dir)

    fusion = fusion_result.get("fusion", {}) if isinstance(fusion_result.get("fusion"), dict) else fusion_result
    depth_analysis = depth_result.get("analysis", {}) if isinstance(depth_result.get("analysis"), dict) else depth_result

    teacher_output = normalize_teacher_output(
        teacher_payload=teacher_payload,
        fusion_result=fusion,
        yolo_result=yolo_result,
        depth_result=depth_analysis,
        state_excerpt=state_excerpt,
        target_house_review=target_house_review,
    )
    validation = validate_teacher_output(
        teacher_output=teacher_output,
        fusion_result=fusion,
        yolo_result=yolo_result,
        depth_result=depth_analysis,
    )

    sample_id = labeling_dir.parent.name
    teacher_output_payload = {
        "sample_id": sample_id,
        "teacher_source_path": str(teacher_source_path),
        "teacher_source_name": teacher_source_path.name,
        "teacher_model": teacher_payload.get("model_name"),
        "teacher_family": "anthropic_scene_descriptor",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "target_house_review": target_house_review,
        "teacher_output": validation["normalized_teacher_output"],
    }
    validation_payload = {
        "sample_id": sample_id,
        "teacher_source_path": str(teacher_source_path),
        "status": validation["status"],
        "score": validation["score"],
        "errors": validation["errors"],
        "warnings": validation["warnings"],
        "normalized_teacher_output": validation["normalized_teacher_output"],
        "evidence": validation["evidence"],
        "target_house_review": target_house_review,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }

    _write_json(labeling_dir / "teacher_output.json", teacher_output_payload)
    _write_json(labeling_dir / "teacher_validation.json", validation_payload)
    return {
        "sample_id": sample_id,
        "status": validation_payload["status"],
        "score": validation_payload["score"],
        "teacher_source_path": str(teacher_source_path),
        "teacher_output_path": str(labeling_dir / "teacher_output.json"),
        "teacher_validation_path": str(labeling_dir / "teacher_validation.json"),
        "entry_state": validation["normalized_teacher_output"]["entry_state"],
        "subgoal": validation["normalized_teacher_output"]["subgoal"],
        "action_hint": validation["normalized_teacher_output"]["action_hint"],
        "target_conditioned_state": validation["normalized_teacher_output"].get("target_conditioned_state", ""),
        "target_conditioned_subgoal": validation["normalized_teacher_output"].get("target_conditioned_subgoal", ""),
        "target_conditioned_action_hint": validation["normalized_teacher_output"].get("target_conditioned_action_hint", ""),
        "error_count": len(validation["errors"]),
        "warning_count": len(validation["warnings"]),
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
        description="Normalize and validate Step C teacher outputs for Phase 2 multimodal fusion labeling packages."
    )
    parser.add_argument("--results_root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--labeling_dir", type=Path, default=None, help="Validate a single labeling directory.")
    parser.add_argument("--only_dir", type=str, default="", help="Only process one fusion_xxx directory under results_root.")
    parser.add_argument("--teacher_filename", type=str, default="", help="Optional teacher result filename inside labeling/.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.labeling_dir:
        summary = validate_labeling_dir(args.labeling_dir, teacher_filename=str(args.teacher_filename or ""))
        print(
            f"[teacher-validator] done -> {summary['sample_id']} "
            f"({summary['status']}; score={summary['score']}; "
            f"entry={summary['entry_state']}; subgoal={summary['subgoal']}; "
            f"target={summary['target_conditioned_state']}; target_subgoal={summary['target_conditioned_subgoal']})"
        )
        print(f"[teacher-validator] teacher_output -> {summary['teacher_output_path']}")
        print(f"[teacher-validator] teacher_validation -> {summary['teacher_validation_path']}")
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
    print(f"[teacher-validator] discovered {total} labeling directories")
    summary_items: List[Dict[str, Any]] = []
    ok_count = 0
    error_count = 0
    entry_state_counts: Counter[str] = Counter()
    subgoal_counts: Counter[str] = Counter()
    target_conditioned_state_counts: Counter[str] = Counter()
    target_conditioned_subgoal_counts: Counter[str] = Counter()
    target_conditioned_action_hint_counts: Counter[str] = Counter()
    for idx, labeling_dir in enumerate(labeling_dirs, start=1):
        sample_name = labeling_dir.parent.name
        print(f"[{idx}/{total}] start -> {sample_name}")
        try:
            summary = validate_labeling_dir(labeling_dir, teacher_filename=str(args.teacher_filename or ""))
            print(
                f"[{idx}/{total}] done -> {sample_name} "
                f"({summary['status']}; score={summary['score']}; "
                f"entry={summary['entry_state']}; subgoal={summary['subgoal']}; "
                f"target={summary['target_conditioned_state']}; "
                f"target_subgoal={summary['target_conditioned_subgoal']}; "
                f"warnings={summary['warning_count']})"
            )
            summary_items.append({"run_dir": str(labeling_dir.parent), "status": "ok", **summary})
            ok_count += 1
            entry_state_counts[summary["entry_state"]] += 1
            subgoal_counts[summary["subgoal"]] += 1
            target_conditioned_state_counts[summary["target_conditioned_state"]] += 1
            target_conditioned_subgoal_counts[summary["target_conditioned_subgoal"]] += 1
            target_conditioned_action_hint_counts[summary["target_conditioned_action_hint"]] += 1
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

    summary_path = results_root / f"teacher_validator_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _write_json(
        summary_path,
        {
            "results_root": str(results_root),
            "ok_count": ok_count,
            "error_count": error_count,
            "entry_state_counts": dict(entry_state_counts),
            "subgoal_counts": dict(subgoal_counts),
            "target_conditioned_state_counts": dict(target_conditioned_state_counts),
            "target_conditioned_subgoal_counts": dict(target_conditioned_subgoal_counts),
            "target_conditioned_action_hint_counts": dict(target_conditioned_action_hint_counts),
            "items": summary_items,
        },
    )
    print(f"[teacher-validator] finished: ok={ok_count} error={error_count}")
    print(f"[teacher-validator] summary -> {summary_path}")


if __name__ == "__main__":
    main()
