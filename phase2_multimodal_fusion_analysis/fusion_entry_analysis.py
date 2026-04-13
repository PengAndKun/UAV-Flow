from __future__ import annotations

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
    crossing_ready_max_distance_cm: float = CROSSING_READY_MAX_DISTANCE_CM,
) -> Dict[str, Any]:
    detections = list(yolo_result.get("detections", []))
    depth_analysis = dict(depth_result.get("analysis", {}))
    candidates = list(depth_analysis.get("entry_assessment", {}).get("candidates", []))
    best_depth_candidate = depth_analysis.get("entry_assessment", {}).get("best_candidate", {}) or {}
    front_obstacle = depth_analysis.get("front_obstacle", {}) or {}

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
    if semantic_detection is not None:
        semantic_depth_assessment = summarize_semantic_region_depth(
            depth_cm=depth_result.get("depth_cm_array"),
            detection=semantic_detection,
            rgb_shape=image_shape,
            hfov_deg=float(depth_result.get("hfov_deg", 90.0)),
            traversable_min_width_cm=float(thresholds.get("traversable_min_width_cm", 90.0)),
            traversable_min_clearance_cm=float(thresholds.get("traversable_min_clearance_cm", 160.0)),
            traversable_min_depth_gain_cm=float(thresholds.get("traversable_min_depth_gain_cm", 80.0)),
            crossing_ready_max_distance_cm=float(crossing_ready_max_distance_cm),
        )

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
    conf: float = 0.25,
    imgsz: int = 640,
    device: str = "0",
) -> Dict[str, Any]:
    if rgb_bgr is None or not isinstance(rgb_bgr, np.ndarray):
        raise RuntimeError("rgb_bgr is required for fusion analysis.")
    if depth_raw is None or not isinstance(depth_raw, np.ndarray):
        raise RuntimeError("depth_raw is required for fusion analysis.")

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
    )
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

    return run_phase2_fusion_analysis(
        rgb_bgr=rgb_bgr,
        depth_raw=depth_raw,
        existing_run_dir=run_dir,
        weights=weights,
        label="",
        camera_info=camera_info,
        state=state,
        conf=float(conf),
        imgsz=int(imgsz),
        device=str(device),
    )
