"""
Minimal trainable local-policy utilities for Phase 3.

This module keeps the first learned reflex policy intentionally simple:
- extract a fixed set of scalar features from dataset samples / runtime requests
- train per-action feature prototypes
- infer the nearest prototype at runtime

It is not the final reflex navigator, but it gives the current stack a stable
"model artifact -> policy server" path before a neural model replaces it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from runtime_interfaces import build_reflex_runtime_state, now_timestamp


FEATURE_NAMES: List[str] = [
    "waypoint_distance_cm",
    "yaw_error_deg",
    "vertical_error_cm",
    "progress_to_waypoint_cm",
    "risk_score",
    "retrieval_score",
    "front_min_depth_cm",
    "front_mean_depth_cm",
    "planner_confidence",
    "archive_visit_count",
]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _extract_depth_features(depth: Dict[str, Any]) -> Tuple[float, float]:
    if not isinstance(depth, dict):
        return 0.0, 0.0
    return (
        _safe_float(depth.get("front_min_depth", 0.0)),
        _safe_float(depth.get("front_mean_depth", 0.0)),
    )


def extract_sample_features(sample: Dict[str, Any]) -> List[float]:
    depth = sample.get("depth", {}) if isinstance(sample.get("depth"), dict) else {}
    front_min_depth_cm, front_mean_depth_cm = _extract_depth_features(depth)
    return [
        _safe_float(sample.get("waypoint_distance_cm", 0.0)),
        _safe_float(sample.get("yaw_error_deg", 0.0)),
        _safe_float(sample.get("vertical_error_cm", 0.0)),
        _safe_float(sample.get("progress_to_waypoint_cm", 0.0)),
        _safe_float(sample.get("risk_score", 0.0)),
        _safe_float(sample.get("retrieval_score", 0.0)),
        front_min_depth_cm,
        front_mean_depth_cm,
        _safe_float(sample.get("planner_confidence", 0.0)),
        _safe_float(sample.get("archive_visit_count", 0.0)),
    ]


def extract_request_features(request_payload: Dict[str, Any]) -> List[float]:
    pose = request_payload.get("pose", {}) if isinstance(request_payload.get("pose"), dict) else {}
    waypoint = request_payload.get("current_waypoint", {}) if isinstance(request_payload.get("current_waypoint"), dict) else {}
    depth = request_payload.get("depth", {}) if isinstance(request_payload.get("depth"), dict) else {}
    archive = request_payload.get("archive", {}) if isinstance(request_payload.get("archive"), dict) else {}
    runtime_debug = request_payload.get("runtime_debug", {}) if isinstance(request_payload.get("runtime_debug"), dict) else {}
    active_retrieval = archive.get("active_retrieval", {}) if isinstance(archive.get("active_retrieval"), dict) else {}
    plan = request_payload.get("plan", {}) if isinstance(request_payload.get("plan"), dict) else {}

    px = _safe_float(pose.get("x", 0.0))
    py = _safe_float(pose.get("y", 0.0))
    pz = _safe_float(pose.get("z", 0.0))
    pyaw = _safe_float(pose.get("yaw", 0.0))

    if waypoint:
        dx = _safe_float(waypoint.get("x", px)) - px
        dy = _safe_float(waypoint.get("y", py)) - py
        dz = _safe_float(waypoint.get("z", pz)) - pz
        waypoint_distance_cm = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        vertical_error_cm = dz
        desired_yaw = pyaw if abs(dx) < 1e-6 and abs(dy) < 1e-6 else float(np.degrees(np.arctan2(dy, dx)))
        yaw_error_deg = ((desired_yaw - pyaw + 180.0) % 360.0) - 180.0
    else:
        waypoint_distance_cm = 0.0
        vertical_error_cm = 0.0
        yaw_error_deg = 0.0

    front_min_depth_cm, front_mean_depth_cm = _extract_depth_features(depth)
    previous_distance = _safe_float(runtime_debug.get("previous_waypoint_distance_cm", waypoint_distance_cm))
    progress_to_waypoint_cm = previous_distance - waypoint_distance_cm

    current_cell = archive.get("current_cell", {}) if isinstance(archive.get("current_cell"), dict) else {}
    return [
        waypoint_distance_cm,
        yaw_error_deg,
        vertical_error_cm,
        progress_to_waypoint_cm,
        _safe_float(runtime_debug.get("risk_score", 0.0)),
        _safe_float(active_retrieval.get("retrieval_score", 0.0)),
        front_min_depth_cm,
        front_mean_depth_cm,
        _safe_float(plan.get("planner_confidence", 0.0)),
        _safe_float(current_cell.get("visit_count", 0.0)),
    ]


def load_dataset_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                records.append(payload)
    return records


def train_prototype_artifact(
    records: Iterable[Dict[str, Any]],
    *,
    policy_name: str,
    min_count_per_action: int = 1,
) -> Dict[str, Any]:
    samples = list(records)
    if not samples:
        raise ValueError("No training samples provided.")

    feature_matrix = np.asarray([extract_sample_features(sample) for sample in samples], dtype=np.float32)
    feature_means = feature_matrix.mean(axis=0)
    feature_stds = feature_matrix.std(axis=0)
    feature_stds = np.where(feature_stds < 1e-6, 1.0, feature_stds)

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        action = str(sample.get("suggested_action", "idle") or "idle")
        grouped.setdefault(action, []).append(sample)

    actions: Dict[str, Any] = {}
    default_action = "hold_position" if "hold_position" in grouped else max(grouped.items(), key=lambda item: len(item[1]))[0]
    for action_name, group in grouped.items():
        if len(group) < int(min_count_per_action):
            continue
        matrix = np.asarray([extract_sample_features(sample) for sample in group], dtype=np.float32)
        normalized = (matrix - feature_means) / feature_stds
        actions[action_name] = {
            "count": len(group),
            "prototype": normalized.mean(axis=0).tolist(),
            "should_execute_rate": float(np.mean([1.0 if sample.get("should_execute", False) else 0.0 for sample in group])),
            "avg_risk_score": float(np.mean([_safe_float(sample.get("risk_score", 0.0)) for sample in group])),
            "avg_waypoint_distance_cm": float(np.mean([_safe_float(sample.get("waypoint_distance_cm", 0.0)) for sample in group])),
        }

    if not actions:
        raise ValueError("No actions met the minimum sample threshold.")

    return {
        "schema_version": "phase3.reflex_policy_artifact.v1",
        "policy_name": policy_name,
        "trained_at": now_timestamp(),
        "feature_names": list(FEATURE_NAMES),
        "feature_means": feature_means.tolist(),
        "feature_stds": feature_stds.tolist(),
        "default_action": default_action,
        "actions": actions,
        "training_summary": {
            "sample_count": len(samples),
            "action_counts": {action: len(group) for action, group in grouped.items()},
        },
    }


def save_artifact(artifact: Dict[str, Any], path: str) -> None:
    Path(path).write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")


def load_artifact(path: str) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid reflex policy artifact: {path}")
    return payload


@dataclass
class PrototypeInferenceResult:
    action_name: str
    distance: float
    should_execute: bool


def predict_action_from_artifact(features: List[float], artifact: Dict[str, Any]) -> PrototypeInferenceResult:
    means = np.asarray(artifact.get("feature_means", [0.0] * len(FEATURE_NAMES)), dtype=np.float32)
    stds = np.asarray(artifact.get("feature_stds", [1.0] * len(FEATURE_NAMES)), dtype=np.float32)
    stds = np.where(stds < 1e-6, 1.0, stds)
    vector = np.asarray(features, dtype=np.float32)
    normalized = (vector - means) / stds

    best_action = str(artifact.get("default_action", "hold_position"))
    best_distance = float("inf")
    should_execute = False
    for action_name, action_payload in (artifact.get("actions", {}) or {}).items():
        prototype = np.asarray(action_payload.get("prototype", []), dtype=np.float32)
        if prototype.shape != normalized.shape:
            continue
        distance = float(np.linalg.norm(normalized - prototype))
        if distance < best_distance:
            best_distance = distance
            best_action = str(action_name)
            should_execute = float(action_payload.get("should_execute_rate", 0.0)) >= 0.5

    return PrototypeInferenceResult(
        action_name=best_action,
        distance=0.0 if best_distance == float("inf") else best_distance,
        should_execute=should_execute,
    )


def build_model_reflex_runtime(
    *,
    request_payload: Dict[str, Any],
    artifact: Dict[str, Any],
) -> Dict[str, Any]:
    features = extract_request_features(request_payload)
    result = predict_action_from_artifact(features, artifact)

    archive = request_payload.get("archive", {}) if isinstance(request_payload.get("archive"), dict) else {}
    active_retrieval = archive.get("active_retrieval", {}) if isinstance(archive.get("active_retrieval"), dict) else {}
    runtime_debug = request_payload.get("runtime_debug", {}) if isinstance(request_payload.get("runtime_debug"), dict) else {}

    return build_reflex_runtime_state(
        mode="prototype_policy",
        policy_name=str(artifact.get("policy_name", "")),
        source="external_model",
        status="policy_inference",
        suggested_action=result.action_name,
        should_execute=result.should_execute,
        last_trigger=str(request_payload.get("context", {}).get("trigger", "")),
        last_latency_ms=0.0,
        waypoint_distance_cm=features[0],
        yaw_error_deg=features[1],
        vertical_error_cm=features[2],
        progress_to_waypoint_cm=features[3],
        retrieval_cell_id=str(active_retrieval.get("cell_id", "")),
        retrieval_score=float(active_retrieval.get("retrieval_score", 0.0) or 0.0),
        retrieval_semantic_subgoal=str(active_retrieval.get("semantic_subgoal", "")),
        risk_score=float(runtime_debug.get("risk_score", 0.0)),
        shield_triggered=bool(runtime_debug.get("shield_triggered", False)),
    ) | {"prototype_distance": float(result.distance)}
