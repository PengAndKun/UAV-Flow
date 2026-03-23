"""RGB+depth doorway detection heuristics for Phase 5 house-search."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from lesson4.depth_planar_pipeline import coerce_depth_planar_image
from runtime_interfaces import build_doorway_candidate, build_doorway_runtime_state


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bbox_to_dict(x: int, y: int, width: int, height: int) -> Dict[str, int]:
    return {
        "x": int(x),
        "y": int(y),
        "width": int(width),
        "height": int(height),
    }


def _extract_ring_mask(
    shape: Tuple[int, int],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    pad: int,
) -> np.ndarray:
    ring_mask = np.zeros(shape, dtype=np.uint8)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(shape[1], x + width + pad)
    y1 = min(shape[0], y + height + pad)
    ring_mask[y0:y1, x0:x1] = 1
    ring_mask[y : y + height, x : x + width] = 0
    return ring_mask.astype(bool)


def detect_doorway_runtime(
    *,
    rgb_frame: Optional[np.ndarray],
    depth_frame: Optional[np.ndarray],
    depth_summary: Optional[Dict[str, Any]] = None,
    detector_name: str = "rgb_depth_doorway_heuristic",
) -> Dict[str, Any]:
    """Detect doorway candidates from RGB + depth and score traversability."""

    if rgb_frame is None or depth_frame is None:
        return build_doorway_runtime_state(
            status="unavailable",
            detector_name=detector_name,
            available=False,
            summary="RGB or depth observation missing.",
        )

    if not isinstance(rgb_frame, np.ndarray) or not isinstance(depth_frame, np.ndarray):
        return build_doorway_runtime_state(
            status="unavailable",
            detector_name=detector_name,
            available=False,
            summary="Doorway detector received invalid frame payloads.",
        )

    depth = coerce_depth_planar_image(depth_frame)
    if depth.ndim != 2:
        return build_doorway_runtime_state(
            status="unavailable",
            detector_name=detector_name,
            available=False,
            summary="Depth frame is not a planar 2D image.",
        )

    rgb = rgb_frame
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)

    height, width = depth.shape[:2]
    frame_id = str((depth_summary or {}).get("frame_id", ""))
    if height < 40 or width < 40:
        return build_doorway_runtime_state(
            frame_id=frame_id,
            status="unavailable",
            detector_name=detector_name,
            available=False,
            summary="Frame resolution too small for doorway detection.",
        )

    configured_min_depth = _safe_float((depth_summary or {}).get("min_depth", 20.0), 20.0)
    configured_max_depth = _safe_float((depth_summary or {}).get("max_depth", 1200.0), 1200.0)
    finite_depth = depth[np.isfinite(depth)]
    if finite_depth.size < 32:
        return build_doorway_runtime_state(
            frame_id=frame_id,
            status="no_depth",
            detector_name=detector_name,
            available=False,
            summary="Depth frame did not contain enough valid values.",
        )

    roi_x0 = int(width * 0.12)
    roi_x1 = int(width * 0.88)
    roi_y0 = int(height * 0.16)
    roi_y1 = int(height * 0.94)
    roi_depth = depth[roi_y0:roi_y1, roi_x0:roi_x1]
    if roi_depth.size == 0:
        return build_doorway_runtime_state(
            frame_id=frame_id,
            status="no_roi",
            detector_name=detector_name,
            available=False,
            summary="Doorway ROI collapsed to an empty region.",
        )

    valid_roi = roi_depth[np.isfinite(roi_depth)]
    if valid_roi.size < 32:
        return build_doorway_runtime_state(
            frame_id=frame_id,
            status="no_depth",
            detector_name=detector_name,
            available=False,
            summary="Doorway ROI did not contain enough valid depth.",
        )

    front_mean_depth = _safe_float((depth_summary or {}).get("front_mean_depth", np.nanpercentile(valid_roi, 35)), np.nanpercentile(valid_roi, 35))
    global_far_threshold = max(
        front_mean_depth + 110.0,
        _safe_float(np.nanpercentile(valid_roi, 72), configured_min_depth + 160.0),
        configured_min_depth + 160.0,
    )
    far_mask = (
        np.isfinite(roi_depth)
        & (roi_depth >= global_far_threshold)
        & (roi_depth <= configured_max_depth)
    ).astype(np.uint8)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 15))
    far_mask = cv2.morphologyEx(far_mask, cv2.MORPH_OPEN, kernel_open)
    far_mask = cv2.morphologyEx(far_mask, cv2.MORPH_CLOSE, kernel_close)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(far_mask, connectivity=8)
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    candidates: List[Dict[str, Any]] = []

    min_width_px = max(18, int(width * 0.07))
    min_height_px = max(48, int(height * 0.24))
    min_area_px = max(1800, int(width * height * 0.015))

    for label_idx in range(1, int(num_labels)):
        stat = stats[label_idx]
        comp_x, comp_y, comp_w, comp_h, comp_area = [int(value) for value in stat]
        if comp_w < min_width_px or comp_h < min_height_px or comp_area < min_area_px:
            continue
        aspect_ratio = comp_w / float(max(comp_h, 1))
        if aspect_ratio < 0.18 or aspect_ratio > 1.8:
            continue

        bbox_x = roi_x0 + comp_x
        bbox_y = roi_y0 + comp_y
        component_mask = labels == label_idx
        global_mask = np.zeros(depth.shape, dtype=bool)
        global_mask[roi_y0:roi_y1, roi_x0:roi_x1] = component_mask

        candidate_depth_values = depth[global_mask]
        candidate_depth_values = candidate_depth_values[np.isfinite(candidate_depth_values)]
        if candidate_depth_values.size < 24:
            continue

        ring_mask = _extract_ring_mask(
            depth.shape,
            x=bbox_x,
            y=bbox_y,
            width=comp_w,
            height=comp_h,
            pad=max(18, int(max(comp_w, comp_h) * 0.18)),
        )
        surround_depth_values = depth[ring_mask]
        surround_depth_values = surround_depth_values[np.isfinite(surround_depth_values)]
        if surround_depth_values.size < 12:
            surround_depth_values = valid_roi

        opening_depth_cm = float(np.nanmean(candidate_depth_values))
        surrounding_depth_cm = float(np.nanmean(surround_depth_values))
        depth_gain_cm = max(0.0, opening_depth_cm - surrounding_depth_cm)

        lower_band_start = bbox_y + int(comp_h * 0.55)
        lower_band_end = min(height, bbox_y + comp_h)
        lower_band_mask = global_mask.copy()
        lower_band_mask[:lower_band_start, :] = False
        lower_band_mask[lower_band_end:, :] = False
        lower_depth_values = depth[lower_band_mask]
        lower_depth_values = lower_depth_values[np.isfinite(lower_depth_values)]
        if lower_depth_values.size < 8:
            lower_depth_values = candidate_depth_values
        clearance_depth_cm = float(np.nanmean(lower_depth_values))

        candidate_gray_values = grayscale[global_mask]
        candidate_gray_values = candidate_gray_values[np.isfinite(candidate_gray_values)]
        surround_gray_values = grayscale[ring_mask]
        surround_gray_values = surround_gray_values[np.isfinite(surround_gray_values)]
        if surround_gray_values.size < 12:
            surround_gray_values = grayscale[roi_y0:roi_y1, roi_x0:roi_x1].reshape(-1)

        inside_luma = float(np.nanmean(candidate_gray_values)) if candidate_gray_values.size else 0.0
        outside_luma = float(np.nanmean(surround_gray_values)) if surround_gray_values.size else inside_luma
        darkness_gain = max(0.0, outside_luma - inside_luma)

        left_strip = grayscale[bbox_y : bbox_y + comp_h, max(0, bbox_x - 4) : min(width, bbox_x + 4)]
        right_strip = grayscale[bbox_y : bbox_y + comp_h, max(0, bbox_x + comp_w - 4) : min(width, bbox_x + comp_w + 4)]
        doorway_edge_score = 0.0
        if left_strip.size and right_strip.size:
            doorway_edge_score = min(
                1.0,
                (float(np.std(left_strip)) + float(np.std(right_strip))) / 80.0,
            )

        width_ratio = comp_w / float(width)
        height_ratio = comp_h / float(height)
        center_x_norm = (bbox_x + (comp_w / 2.0)) / float(width)
        center_y_norm = (bbox_y + (comp_h / 2.0)) / float(height)
        center_bias = 1.0 - min(1.0, abs(center_x_norm - 0.5) / 0.5)

        depth_score = _clamp(depth_gain_cm / 280.0, 0.0, 1.0)
        rgb_score = _clamp((darkness_gain / 70.0) * 0.75 + doorway_edge_score * 0.25, 0.0, 1.0)
        width_score = _clamp((width_ratio - 0.06) / 0.18, 0.0, 1.0)
        height_score = _clamp((height_ratio - 0.22) / 0.45, 0.0, 1.0)

        traversable = bool(
            clearance_depth_cm >= max(150.0, surrounding_depth_cm + 55.0)
            and width_ratio >= 0.075
            and height_ratio >= 0.23
            and depth_gain_cm >= 80.0
        )

        confidence = _clamp(
            (depth_score * 0.42)
            + (rgb_score * 0.23)
            + (width_score * 0.12)
            + (height_score * 0.10)
            + (center_bias * 0.08)
            + (0.05 if traversable else 0.0),
            0.0,
            1.0,
        )

        label = "entry_doorway" if traversable else "door_candidate"
        rationale_parts = [
            f"depth_gain={depth_gain_cm:.0f}cm",
            f"clearance={clearance_depth_cm:.0f}cm",
            f"darkness_gain={darkness_gain:.0f}",
        ]
        if traversable:
            rationale_parts.append("opening appears traversable")

        candidates.append(
            build_doorway_candidate(
                candidate_id=f"doorway_{label_idx:02d}",
                label=label,
                bbox=_bbox_to_dict(bbox_x, bbox_y, comp_w, comp_h),
                center_x_norm=center_x_norm,
                center_y_norm=center_y_norm,
                width_ratio=width_ratio,
                height_ratio=height_ratio,
                opening_depth_cm=opening_depth_cm,
                surrounding_depth_cm=surrounding_depth_cm,
                clearance_depth_cm=clearance_depth_cm,
                depth_gain_cm=depth_gain_cm,
                rgb_door_score=rgb_score,
                depth_opening_score=depth_score,
                confidence=confidence,
                traversable=traversable,
                rationale=", ".join(rationale_parts),
            )
        )

    candidates.sort(
        key=lambda item: (
            int(bool(item.get("traversable", False))),
            float(item.get("confidence", 0.0)),
            float(item.get("depth_gain_cm", 0.0)),
        ),
        reverse=True,
    )
    best_candidate = candidates[0] if candidates else {}
    traversable_candidates = [item for item in candidates if bool(item.get("traversable", False))]

    if best_candidate:
        best_label = str(best_candidate.get("label", "door_candidate") or "door_candidate")
        best_depth_gain = float(best_candidate.get("depth_gain_cm", 0.0))
        best_clearance = float(best_candidate.get("clearance_depth_cm", 0.0))
        summary = (
            f"best={best_label} traversable={int(bool(best_candidate.get('traversable', False)))} "
            f"gain={best_depth_gain:.0f}cm clearance={best_clearance:.0f}cm "
            f"cand={len(candidates)} traversable_cand={len(traversable_candidates)}"
        )
        status = "candidate_found"
        focus_label = best_label
    else:
        summary = "No doorway candidate matched RGB+depth opening heuristics."
        status = "no_candidate"
        focus_label = ""

    return build_doorway_runtime_state(
        frame_id=frame_id,
        status=status,
        detector_name=detector_name,
        available=True,
        candidate_count=len(candidates),
        traversable_candidate_count=len(traversable_candidates),
        best_candidate=best_candidate,
        candidates=candidates[:5],
        focus_label=focus_label,
        summary=summary,
    )
