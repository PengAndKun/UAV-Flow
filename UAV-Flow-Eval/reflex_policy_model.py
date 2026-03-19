"""
Minimal trainable local-policy utilities for Phase 3.

This module now supports two lightweight learned baselines:
- prototype: nearest-prototype action selection
- mlp_classifier: one-hidden-layer MLP trained with NumPy

It keeps the runtime path stable while letting Phase 3 move from a pure
heuristic/debug stack to a small but real trainable local policy.
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
    "waypoint_forward_error_cm",
    "waypoint_right_error_cm",
    "yaw_error_deg",
    "vertical_error_cm",
    "progress_to_waypoint_cm",
    "risk_score",
    "retrieval_score",
    "front_min_depth_cm",
    "front_mean_depth_cm",
    "planner_confidence",
    "planner_sector_id",
    "archive_visit_count",
    "has_retrieval",
    "retrieval_matches_subgoal",
    "subgoal_forward",
    "subgoal_backward",
    "subgoal_lateral",
    "subgoal_turn",
    "subgoal_vertical",
]


ACTION_NAME_ALIASES: Dict[str, str] = {
    "forward(w)": "forward",
    "backward(s)": "backward",
    "left(a)": "left",
    "right(d)": "right",
    "up(r)": "up",
    "down(f)": "down",
    "yaw_left(q)": "yaw_left",
    "yaw_right(e)": "yaw_right",
    "hold_position": "hold_position",
    "shield_hold": "shield_hold",
    "scan_hover": "scan_hover",
    "idle": "idle",
    "forward": "forward",
    "backward": "backward",
    "left": "left",
    "right": "right",
    "up": "up",
    "down": "down",
    "yaw_left": "yaw_left",
    "yaw_right": "yaw_right",
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def normalize_action_name(action_name: Any) -> str:
    text = str(action_name or "idle").strip().lower()
    return ACTION_NAME_ALIASES.get(text, text or "idle")


def resolve_target_action(sample: Dict[str, Any]) -> str:
    for field_name in ("target_action", "executed_action", "suggested_action"):
        raw_value = sample.get(field_name, "")
        if str(raw_value or "").strip():
            return normalize_action_name(raw_value)
    return "idle"


def _extract_depth_features(depth: Dict[str, Any]) -> Tuple[float, float]:
    if not isinstance(depth, dict):
        return 0.0, 0.0
    return (
        _safe_float(depth.get("front_min_depth", 0.0)),
        _safe_float(depth.get("front_mean_depth", 0.0)),
    )


def _normalize_text(text: Any) -> str:
    return str(text or "").strip().lower().replace("-", "_").replace(" ", "_")


def _extract_subgoal_features(text: Any) -> List[float]:
    normalized = _normalize_text(text)
    return [
        1.0 if "forward" in normalized else 0.0,
        1.0 if "backward" in normalized else 0.0,
        1.0 if "left" in normalized or "right" in normalized else 0.0,
        1.0 if "turn" in normalized else 0.0,
        1.0 if "ascend" in normalized or "descend" in normalized or "up" in normalized or "down" in normalized else 0.0,
    ]


def _extract_waypoint_local_components(
    *,
    px: float,
    py: float,
    pz: float,
    yaw_deg: float,
    waypoint: Dict[str, Any],
) -> Tuple[float, float, float, float, float]:
    if not isinstance(waypoint, dict) or not waypoint:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    dx = _safe_float(waypoint.get("x", px)) - px
    dy = _safe_float(waypoint.get("y", py)) - py
    dz = _safe_float(waypoint.get("z", pz)) - pz
    yaw_rad = float(np.deg2rad(yaw_deg))
    local_forward = float(np.cos(yaw_rad) * dx + np.sin(yaw_rad) * dy)
    local_right = float(-np.sin(yaw_rad) * dx + np.cos(yaw_rad) * dy)
    waypoint_distance_cm = float(np.sqrt(dx * dx + dy * dy + dz * dz))
    desired_yaw = yaw_deg if abs(dx) < 1e-6 and abs(dy) < 1e-6 else float(np.degrees(np.arctan2(dy, dx)))
    yaw_error_deg = ((desired_yaw - yaw_deg + 180.0) % 360.0) - 180.0
    return local_forward, local_right, dz, waypoint_distance_cm, yaw_error_deg


def _resolve_waypoint_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    current_waypoint = sample.get("current_waypoint", {})
    if isinstance(current_waypoint, dict) and current_waypoint:
        return current_waypoint
    target_waypoint = sample.get("target_waypoint", {})
    if isinstance(target_waypoint, dict) and target_waypoint:
        return target_waypoint
    return {}


def extract_sample_features(sample: Dict[str, Any]) -> List[float]:
    depth = sample.get("depth", {}) if isinstance(sample.get("depth"), dict) else {}
    front_min_depth_cm, front_mean_depth_cm = _extract_depth_features(depth)
    pose = sample.get("pose", {}) if isinstance(sample.get("pose"), dict) else {}
    px = _safe_float(pose.get("x", 0.0))
    py = _safe_float(pose.get("y", 0.0))
    pz = _safe_float(pose.get("z", 0.0))
    pyaw = _safe_float(pose.get("yaw", 0.0))
    waypoint = _resolve_waypoint_from_sample(sample)
    local_forward, local_right, vertical_error_cm, waypoint_distance_cm, yaw_error_deg = _extract_waypoint_local_components(
        px=px,
        py=py,
        pz=pz,
        yaw_deg=pyaw,
        waypoint=waypoint,
    )
    planner_subgoal = str(sample.get("planner_subgoal", "") or "")
    retrieval_subgoal = str(sample.get("retrieval_semantic_subgoal", "") or "")
    subgoal_features = _extract_subgoal_features(planner_subgoal)
    retrieval_cell_id = str(sample.get("retrieval_cell_id", "") or "")
    return [
        waypoint_distance_cm if waypoint else _safe_float(sample.get("waypoint_distance_cm", 0.0)),
        local_forward,
        local_right,
        yaw_error_deg if waypoint else _safe_float(sample.get("yaw_error_deg", 0.0)),
        vertical_error_cm if waypoint else _safe_float(sample.get("vertical_error_cm", 0.0)),
        _safe_float(sample.get("progress_to_waypoint_cm", 0.0)),
        _safe_float(sample.get("risk_score", 0.0)),
        _safe_float(sample.get("retrieval_score", 0.0)),
        front_min_depth_cm,
        front_mean_depth_cm,
        _safe_float(sample.get("planner_confidence", 0.0)),
        _safe_float(sample.get("planner_sector_id", 0.0)),
        _safe_float(sample.get("archive_visit_count", 0.0)),
        0.0 if retrieval_cell_id in {"", "none"} else 1.0,
        1.0 if planner_subgoal and planner_subgoal == retrieval_subgoal else 0.0,
        *subgoal_features,
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

    local_forward, local_right, vertical_error_cm, waypoint_distance_cm, yaw_error_deg = _extract_waypoint_local_components(
        px=px,
        py=py,
        pz=pz,
        yaw_deg=pyaw,
        waypoint=waypoint,
    )

    front_min_depth_cm, front_mean_depth_cm = _extract_depth_features(depth)
    previous_distance = _safe_float(runtime_debug.get("previous_waypoint_distance_cm", waypoint_distance_cm))
    progress_to_waypoint_cm = previous_distance - waypoint_distance_cm

    current_cell = archive.get("current_cell", {}) if isinstance(archive.get("current_cell"), dict) else {}
    planner_subgoal = str(plan.get("semantic_subgoal", "") or "")
    retrieval_subgoal = str(active_retrieval.get("semantic_subgoal", "") or "")
    retrieval_cell_id = str(active_retrieval.get("cell_id", "") or "")
    subgoal_features = _extract_subgoal_features(planner_subgoal)
    return [
        waypoint_distance_cm,
        local_forward,
        local_right,
        yaw_error_deg,
        vertical_error_cm,
        progress_to_waypoint_cm,
        _safe_float(runtime_debug.get("risk_score", 0.0)),
        _safe_float(active_retrieval.get("retrieval_score", 0.0)),
        front_min_depth_cm,
        front_mean_depth_cm,
        _safe_float(plan.get("planner_confidence", 0.0)),
        _safe_float(plan.get("sector_id", 0.0)),
        _safe_float(current_cell.get("visit_count", 0.0)),
        0.0 if retrieval_cell_id in {"", "none"} else 1.0,
        1.0 if planner_subgoal and planner_subgoal == retrieval_subgoal else 0.0,
        *subgoal_features,
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
        action = resolve_target_action(sample)
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
        "model_type": "prototype",
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
    confidence: float = 0.0


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.clip(np.sum(exp_logits, axis=1, keepdims=True), 1e-8, None)


def predict_action_from_artifact(features: List[float], artifact: Dict[str, Any]) -> PrototypeInferenceResult:
    model_type = str(artifact.get("model_type", "prototype"))
    if model_type == "mlp_classifier":
        return predict_action_from_mlp_artifact(features, artifact)

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
        confidence=float(1.0 / (1.0 + max(0.0, best_distance))),
    )


def train_mlp_artifact(
    records: Iterable[Dict[str, Any]],
    *,
    policy_name: str,
    hidden_dim: int = 32,
    epochs: int = 250,
    learning_rate: float = 0.02,
    weight_decay: float = 1e-4,
    seed: int = 0,
    class_weight_power: float = 0.0,
) -> Dict[str, Any]:
    samples = list(records)
    if not samples:
        raise ValueError("No training samples provided.")

    x = np.asarray([extract_sample_features(sample) for sample in samples], dtype=np.float32)
    feature_means = x.mean(axis=0)
    feature_stds = x.std(axis=0)
    feature_stds = np.where(feature_stds < 1e-6, 1.0, feature_stds)
    x_norm = (x - feature_means) / feature_stds

    resolved_targets = [resolve_target_action(sample) for sample in samples]
    action_names = sorted(set(resolved_targets))
    action_to_index = {action_name: idx for idx, action_name in enumerate(action_names)}
    y = np.asarray([action_to_index[action_name] for action_name in resolved_targets], dtype=np.int64)
    num_classes = max(1, len(action_names))
    num_features = x_norm.shape[1]
    hidden_dim = max(4, int(hidden_dim))

    rng = np.random.default_rng(int(seed))
    w1 = (rng.standard_normal((num_features, hidden_dim)).astype(np.float32) * np.sqrt(2.0 / max(1, num_features)))
    b1 = np.zeros((1, hidden_dim), dtype=np.float32)
    w2 = (rng.standard_normal((hidden_dim, num_classes)).astype(np.float32) * np.sqrt(2.0 / max(1, hidden_dim)))
    b2 = np.zeros((1, num_classes), dtype=np.float32)

    one_hot = np.zeros((len(samples), num_classes), dtype=np.float32)
    one_hot[np.arange(len(samples)), y] = 1.0
    class_counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    sample_weights = np.ones((len(samples), 1), dtype=np.float32)
    if float(class_weight_power) > 0.0:
        safe_counts = np.where(class_counts < 1.0, 1.0, class_counts)
        inv = np.power(np.max(safe_counts) / safe_counts, float(class_weight_power)).astype(np.float32)
        inv = inv / np.clip(np.mean(inv), 1e-8, None)
        sample_weights = inv[y].reshape(-1, 1).astype(np.float32)
    sample_weight_sum = float(np.sum(sample_weights))

    final_loss = 0.0
    for _epoch in range(max(1, int(epochs))):
        z1 = x_norm @ w1 + b1
        h1 = np.maximum(z1, 0.0)
        logits = h1 @ w2 + b2
        probs = _stable_softmax(logits)
        ce_loss = -np.sum(sample_weights * one_hot * np.log(np.clip(probs, 1e-8, 1.0))) / max(1e-8, sample_weight_sum)
        l2_loss = 0.5 * float(weight_decay) * (float(np.sum(w1 * w1)) + float(np.sum(w2 * w2)))
        final_loss = float(ce_loss + l2_loss)

        grad_logits = (sample_weights * (probs - one_hot)) / max(1e-8, sample_weight_sum)
        grad_w2 = h1.T @ grad_logits + float(weight_decay) * w2
        grad_b2 = np.sum(grad_logits, axis=0, keepdims=True)
        grad_h1 = grad_logits @ w2.T
        grad_z1 = grad_h1 * (z1 > 0.0).astype(np.float32)
        grad_w1 = x_norm.T @ grad_z1 + float(weight_decay) * w1
        grad_b1 = np.sum(grad_z1, axis=0, keepdims=True)

        lr = float(learning_rate)
        w1 -= lr * grad_w1
        b1 -= lr * grad_b1
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2

    predicted = np.argmax(_stable_softmax(np.maximum(x_norm @ w1 + b1, 0.0) @ w2 + b2), axis=1)
    train_accuracy = float(np.mean(predicted == y)) if len(samples) else 0.0
    default_action = max(action_to_index, key=lambda name: int(np.sum(y == action_to_index[name])))

    return {
        "schema_version": "phase3.reflex_policy_artifact.v2",
        "model_type": "mlp_classifier",
        "policy_name": policy_name,
        "trained_at": now_timestamp(),
        "feature_names": list(FEATURE_NAMES),
        "feature_means": feature_means.tolist(),
        "feature_stds": feature_stds.tolist(),
        "default_action": default_action,
        "action_names": action_names,
        "network": {
            "hidden_dim": hidden_dim,
            "input_dim": num_features,
            "output_dim": num_classes,
            "w1": w1.tolist(),
            "b1": b1.tolist(),
            "w2": w2.tolist(),
            "b2": b2.tolist(),
        },
        "actions": {
            action_name: {
                "count": int(np.sum(y == action_to_index[action_name])),
                "should_execute_rate": float(
                    np.mean(
                        [
                            1.0 if sample.get("should_execute", False) else 0.0
                            for sample in samples
                            if resolve_target_action(sample) == action_name
                        ]
                    )
                ),
            }
            for action_name in action_names
        },
        "training_summary": {
            "sample_count": len(samples),
            "action_counts": {action: int(np.sum(y == action_to_index[action])) for action in action_names},
            "train_accuracy": train_accuracy,
            "final_loss": final_loss,
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "class_weight_power": float(class_weight_power),
        },
    }


def predict_action_from_mlp_artifact(features: List[float], artifact: Dict[str, Any]) -> PrototypeInferenceResult:
    means = np.asarray(artifact.get("feature_means", [0.0] * len(FEATURE_NAMES)), dtype=np.float32)
    stds = np.asarray(artifact.get("feature_stds", [1.0] * len(FEATURE_NAMES)), dtype=np.float32)
    stds = np.where(stds < 1e-6, 1.0, stds)
    vector = np.asarray(features, dtype=np.float32).reshape(1, -1)
    normalized = (vector - means.reshape(1, -1)) / stds.reshape(1, -1)

    network = artifact.get("network", {}) if isinstance(artifact.get("network"), dict) else {}
    w1 = np.asarray(network.get("w1", []), dtype=np.float32)
    b1 = np.asarray(network.get("b1", []), dtype=np.float32)
    w2 = np.asarray(network.get("w2", []), dtype=np.float32)
    b2 = np.asarray(network.get("b2", []), dtype=np.float32)

    action_names = list(artifact.get("action_names", []))
    if w1.size == 0 or w2.size == 0 or not action_names:
        default_action = str(artifact.get("default_action", "hold_position"))
        return PrototypeInferenceResult(action_name=default_action, distance=0.0, should_execute=False, confidence=0.0)

    hidden = np.maximum(normalized @ w1 + b1, 0.0)
    probs = _stable_softmax(hidden @ w2 + b2)
    class_index = int(np.argmax(probs[0]))
    action_name = str(action_names[class_index]) if class_index < len(action_names) else str(artifact.get("default_action", "hold_position"))
    action_payload = artifact.get("actions", {}).get(action_name, {})
    should_execute = float(action_payload.get("should_execute_rate", 0.0)) >= 0.5
    confidence = float(probs[0, class_index]) if probs.shape[1] else 0.0
    return PrototypeInferenceResult(
        action_name=action_name,
        distance=float(1.0 - confidence),
        should_execute=should_execute,
        confidence=confidence,
    )


def build_model_reflex_runtime(
    *,
    request_payload: Dict[str, Any],
    artifact: Dict[str, Any],
) -> Dict[str, Any]:
    features = extract_request_features(request_payload)
    result = predict_action_from_artifact(features, artifact)
    model_type = str(artifact.get("model_type", "prototype"))

    archive = request_payload.get("archive", {}) if isinstance(request_payload.get("archive"), dict) else {}
    active_retrieval = archive.get("active_retrieval", {}) if isinstance(archive.get("active_retrieval"), dict) else {}
    runtime_debug = request_payload.get("runtime_debug", {}) if isinstance(request_payload.get("runtime_debug"), dict) else {}

    return build_reflex_runtime_state(
        mode="mlp_policy" if model_type == "mlp_classifier" else "prototype_policy",
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
    ) | {
        "prototype_distance": float(result.distance),
        "policy_confidence": float(result.confidence),
        "model_type": model_type,
    }
