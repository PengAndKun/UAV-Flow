from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
UAV_FLOW_EVAL = ROOT / "UAV-Flow-Eval"
if str(UAV_FLOW_EVAL) not in sys.path:
    sys.path.insert(0, str(UAV_FLOW_EVAL))

from lesson4.depth_planar_pipeline import coerce_depth_planar_image


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def load_depth_image(depth_path: Path, max_valid_depth_cm: float) -> np.ndarray:
    suffix = depth_path.suffix.lower()
    if suffix == ".npy":
        raw = np.load(depth_path)
    else:
        raw = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not read depth image: {depth_path}")
    depth_cm = coerce_depth_planar_image(raw)
    depth_cm = np.where(np.isfinite(depth_cm), depth_cm, np.nan)
    depth_cm = np.where(depth_cm <= 0.0, np.nan, depth_cm)
    depth_cm = np.where(depth_cm > float(max_valid_depth_cm), np.nan, depth_cm)
    return depth_cm.astype(np.float32, copy=False)


def build_depth_preview(depth_cm: np.ndarray, max_valid_depth_cm: float) -> np.ndarray:
    finite = depth_cm[np.isfinite(depth_cm)]
    if finite.size == 0:
        canvas = np.zeros((depth_cm.shape[0], depth_cm.shape[1], 3), dtype=np.uint8)
        canvas[:] = (20, 20, 20)
        return canvas

    min_depth = float(np.nanpercentile(finite, 3))
    max_depth = float(min(max_valid_depth_cm, np.nanpercentile(finite, 98)))
    if max_depth <= min_depth:
        max_depth = min_depth + 1.0

    normalized = (depth_cm - min_depth) / (max_depth - min_depth)
    normalized = np.clip(normalized, 0.0, 1.0)
    normalized = np.where(np.isfinite(normalized), normalized, 1.0)
    image_8u = (normalized * 255.0).astype(np.uint8)
    preview = cv2.applyColorMap(255 - image_8u, cv2.COLORMAP_TURBO)
    return preview


def extract_ring_mask(
    shape: Tuple[int, int],
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    pad: int,
) -> np.ndarray:
    ring = np.zeros(shape, dtype=np.uint8)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(shape[1], x + width + pad)
    y1 = min(shape[0], y + height + pad)
    ring[y0:y1, x0:x1] = 1
    ring[y : y + height, x : x + width] = 0
    return ring.astype(bool)


def estimate_opening_width_cm(
    *,
    opening_depth_cm: float,
    width_ratio: float,
    hfov_deg: float,
) -> float:
    half_width_cm = opening_depth_cm * math.tan(math.radians(hfov_deg / 2.0))
    full_view_width_cm = 2.0 * half_width_cm
    return max(0.0, full_view_width_cm * width_ratio)


def analyze_depth_entry(
    *,
    depth_cm: np.ndarray,
    hfov_deg: float,
    max_valid_depth_cm: float,
    front_obstacle_threshold_cm: float,
    traversable_min_width_cm: float,
    traversable_min_clearance_cm: float,
    traversable_min_depth_gain_cm: float,
) -> Dict[str, Any]:
    height, width = depth_cm.shape[:2]
    finite = depth_cm[np.isfinite(depth_cm)]
    if finite.size < 32:
        return {
            "status": "invalid_depth",
            "summary": "Depth image did not contain enough valid values.",
            "front_obstacle": {
                "present": False,
                "front_min_depth_cm": None,
                "front_mean_depth_cm": None,
                "severity": "unknown",
            },
            "entry_assessment": {
                "candidate_count": 0,
                "best_candidate": {},
                "candidates": [],
            },
        }

    front_y0 = int(height * 0.42)
    front_y1 = int(height * 0.88)
    front_x0 = int(width * 0.38)
    front_x1 = int(width * 0.62)
    front_roi = depth_cm[front_y0:front_y1, front_x0:front_x1]
    front_values = front_roi[np.isfinite(front_roi)]
    front_min_depth_cm = float(np.nanmin(front_values)) if front_values.size else float("nan")
    front_mean_depth_cm = float(np.nanmean(front_values)) if front_values.size else float("nan")
    front_obstacle_present = bool(
        front_values.size > 0 and (
            front_min_depth_cm <= front_obstacle_threshold_cm
            or front_mean_depth_cm <= (front_obstacle_threshold_cm + 40.0)
        )
    )
    if not np.isfinite(front_min_depth_cm):
        front_obstacle_present = False

    if not front_obstacle_present:
        severity = "clear"
    elif front_min_depth_cm <= front_obstacle_threshold_cm * 0.6:
        severity = "high"
    elif front_min_depth_cm <= front_obstacle_threshold_cm:
        severity = "medium"
    else:
        severity = "low"

    roi_x0 = int(width * 0.10)
    roi_x1 = int(width * 0.90)
    roi_y0 = int(height * 0.16)
    roi_y1 = int(height * 0.94)
    roi_depth = depth_cm[roi_y0:roi_y1, roi_x0:roi_x1]
    valid_roi = roi_depth[np.isfinite(roi_depth)]

    far_threshold = max(
        float(np.nanpercentile(valid_roi, 72)),
        front_mean_depth_cm + 120.0 if np.isfinite(front_mean_depth_cm) else 220.0,
        220.0,
    )
    far_mask = (
        np.isfinite(roi_depth)
        & (roi_depth >= far_threshold)
        & (roi_depth <= float(max_valid_depth_cm))
    ).astype(np.uint8)
    far_mask = cv2.morphologyEx(
        far_mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 9)),
    )
    far_mask = cv2.morphologyEx(
        far_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 15)),
    )

    min_width_px = max(18, int(width * 0.06))
    min_height_px = max(40, int(height * 0.20))
    min_area_px = max(1200, int(width * height * 0.010))

    candidates: List[Dict[str, Any]] = []
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(far_mask, connectivity=8)
    for label_idx in range(1, int(num_labels)):
        comp_x, comp_y, comp_w, comp_h, comp_area = [int(v) for v in stats[label_idx]]
        if comp_w < min_width_px or comp_h < min_height_px or comp_area < min_area_px:
            continue

        width_ratio = comp_w / float(width)
        height_ratio = comp_h / float(height)
        aspect_ratio = comp_w / float(max(comp_h, 1))
        if aspect_ratio < 0.15 or aspect_ratio > 1.8:
            continue

        bbox_x = roi_x0 + comp_x
        bbox_y = roi_y0 + comp_y
        component_mask = labels == label_idx
        global_mask = np.zeros(depth_cm.shape, dtype=bool)
        global_mask[roi_y0:roi_y1, roi_x0:roi_x1] = component_mask

        opening_values = depth_cm[global_mask]
        opening_values = opening_values[np.isfinite(opening_values)]
        if opening_values.size < 24:
            continue

        ring_mask = extract_ring_mask(
            depth_cm.shape,
            x=bbox_x,
            y=bbox_y,
            width=comp_w,
            height=comp_h,
            pad=max(20, int(max(comp_w, comp_h) * 0.18)),
        )
        surround_values = depth_cm[ring_mask]
        surround_values = surround_values[np.isfinite(surround_values)]
        if surround_values.size < 12:
            surround_values = valid_roi

        opening_depth_cm = float(np.nanmean(opening_values))
        surround_depth_cm = float(np.nanmean(surround_values))
        depth_gain_cm = max(0.0, opening_depth_cm - surround_depth_cm)

        lower_band_start = bbox_y + int(comp_h * 0.55)
        lower_mask = global_mask.copy()
        lower_mask[:lower_band_start, :] = False
        lower_values = depth_cm[lower_mask]
        lower_values = lower_values[np.isfinite(lower_values)]
        if lower_values.size < 8:
            lower_values = opening_values
        clearance_depth_cm = float(np.nanmean(lower_values))

        center_x_norm = (bbox_x + comp_w / 2.0) / float(width)
        center_y_norm = (bbox_y + comp_h / 2.0) / float(height)
        center_bias = 1.0 - min(1.0, abs(center_x_norm - 0.5) / 0.5)

        opening_width_cm = estimate_opening_width_cm(
            opening_depth_cm=opening_depth_cm,
            width_ratio=width_ratio,
            hfov_deg=hfov_deg,
        )
        traversable = bool(
            opening_width_cm >= traversable_min_width_cm
            and clearance_depth_cm >= traversable_min_clearance_cm
            and depth_gain_cm >= traversable_min_depth_gain_cm
            and height_ratio >= 0.20
        )

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

        candidates.append(
            {
                "candidate_id": f"depth_entry_{label_idx:02d}",
                "bbox": {
                    "x": bbox_x,
                    "y": bbox_y,
                    "width": comp_w,
                    "height": comp_h,
                },
                "center_x_norm": center_x_norm,
                "center_y_norm": center_y_norm,
                "width_ratio": width_ratio,
                "height_ratio": height_ratio,
                "entry_distance_cm": opening_depth_cm,
                "surrounding_depth_cm": surround_depth_cm,
                "clearance_depth_cm": clearance_depth_cm,
                "depth_gain_cm": depth_gain_cm,
                "opening_width_cm": opening_width_cm,
                "traversable": traversable,
                "confidence": confidence,
                "rationale": (
                    f"distance={opening_depth_cm:.0f}cm, width={opening_width_cm:.0f}cm, "
                    f"clearance={clearance_depth_cm:.0f}cm, gain={depth_gain_cm:.0f}cm"
                ),
            }
        )

    candidates.sort(
        key=lambda item: (
            int(bool(item.get("traversable", False))),
            float(item.get("confidence", 0.0)),
            float(item.get("opening_width_cm", 0.0)),
        ),
        reverse=True,
    )

    best_candidate = candidates[0] if candidates else {}
    summary = (
        f"front_obstacle={int(front_obstacle_present)} "
        f"front_min={front_min_depth_cm:.1f}cm "
        f"candidates={len(candidates)} "
        f"best_traversable={int(bool(best_candidate.get('traversable', False)))}"
        if np.isfinite(front_min_depth_cm)
        else f"front_obstacle={int(front_obstacle_present)} candidates={len(candidates)}"
    )

    return {
        "status": "ok",
        "summary": summary,
        "front_obstacle": {
            "present": front_obstacle_present,
            "front_min_depth_cm": round(front_min_depth_cm, 2) if np.isfinite(front_min_depth_cm) else None,
            "front_mean_depth_cm": round(front_mean_depth_cm, 2) if np.isfinite(front_mean_depth_cm) else None,
            "severity": severity,
        },
        "entry_assessment": {
            "candidate_count": len(candidates),
            "best_candidate": best_candidate,
            "candidates": candidates[:5],
        },
    }


def draw_overlay(
    preview: np.ndarray,
    analysis: Dict[str, Any],
) -> np.ndarray:
    canvas = preview.copy()
    height, width = canvas.shape[:2]

    front_x0 = int(width * 0.38)
    front_x1 = int(width * 0.62)
    front_y0 = int(height * 0.42)
    front_y1 = int(height * 0.88)
    obstacle = analysis.get("front_obstacle", {})
    obstacle_color = (0, 0, 255) if obstacle.get("present") else (0, 200, 0)
    cv2.rectangle(canvas, (front_x0, front_y0), (front_x1, front_y1), obstacle_color, 2)

    for idx, candidate in enumerate(analysis.get("entry_assessment", {}).get("candidates", []), start=1):
        bbox = candidate.get("bbox", {})
        x = int(bbox.get("x", 0))
        y = int(bbox.get("y", 0))
        w = int(bbox.get("width", 0))
        h = int(bbox.get("height", 0))
        traversable = bool(candidate.get("traversable", False))
        color = (0, 255, 0) if traversable else (0, 180, 255)
        cv2.rectangle(canvas, (x, y), (x + w, y + h), color, 2)
        label = (
            f"C{idx} "
            f"{candidate.get('entry_distance_cm', 0.0):.0f}cm "
            f"{candidate.get('opening_width_cm', 0.0):.0f}cm"
        )
        cv2.putText(
            canvas,
            label,
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    title_lines = [
        f"Front obstacle: {int(bool(obstacle.get('present', False)))} "
        f"(min={obstacle.get('front_min_depth_cm', 'n/a')}cm)",
    ]
    best = analysis.get("entry_assessment", {}).get("best_candidate", {})
    if best:
        title_lines.append(
            f"Best entry: dist={best.get('entry_distance_cm', 0.0):.0f}cm "
            f"width={best.get('opening_width_cm', 0.0):.0f}cm "
            f"trav={int(bool(best.get('traversable', False)))}"
        )
    else:
        title_lines.append("Best entry: none")

    for idx, line in enumerate(title_lines):
        cv2.putText(
            canvas,
            line,
            (16, 28 + idx * 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return canvas


def write_text_summary(path: Path, result: Dict[str, Any]) -> None:
    obstacle = result.get("front_obstacle", {})
    best = result.get("entry_assessment", {}).get("best_candidate", {})
    lines = [
        f"status={result.get('status', 'unknown')}",
        f"summary={result.get('summary', '')}",
        "",
        f"front_obstacle.present={obstacle.get('present')}",
        f"front_obstacle.front_min_depth_cm={obstacle.get('front_min_depth_cm')}",
        f"front_obstacle.front_mean_depth_cm={obstacle.get('front_mean_depth_cm')}",
        f"front_obstacle.severity={obstacle.get('severity')}",
        "",
        f"entry_assessment.candidate_count={result.get('entry_assessment', {}).get('candidate_count', 0)}",
        f"entry_assessment.best_candidate.entry_distance_cm={best.get('entry_distance_cm')}",
        f"entry_assessment.best_candidate.opening_width_cm={best.get('opening_width_cm')}",
        f"entry_assessment.best_candidate.clearance_depth_cm={best.get('clearance_depth_cm')}",
        f"entry_assessment.best_candidate.depth_gain_cm={best.get('depth_gain_cm')}",
        f"entry_assessment.best_candidate.traversable={best.get('traversable')}",
        f"entry_assessment.best_candidate.confidence={best.get('confidence')}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    default_root = ROOT / "phase2_modality2_depth_analysis"
    parser = argparse.ArgumentParser(description="Phase 2 modality-2 single depth image entry analysis")
    parser.add_argument("--depth_path", type=Path, required=True, help="Path to a depth_cm PNG or NPY file.")
    parser.add_argument("--output_root", type=Path, default=default_root / "outputs")
    parser.add_argument("--hfov_deg", type=float, default=90.0)
    parser.add_argument("--max_valid_depth_cm", type=float, default=1200.0)
    parser.add_argument("--front_obstacle_threshold_cm", type=float, default=140.0)
    parser.add_argument("--traversable_min_width_cm", type=float, default=90.0)
    parser.add_argument("--traversable_min_clearance_cm", type=float, default=160.0)
    parser.add_argument("--traversable_min_depth_gain_cm", type=float, default=80.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    depth_path = args.depth_path.resolve()
    output_root = args.output_root.resolve()
    ensure_dir(output_root)

    run_dir = output_root / f"depth_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{depth_path.stem}"
    ensure_dir(run_dir)

    depth_cm = load_depth_image(depth_path, max_valid_depth_cm=float(args.max_valid_depth_cm))
    preview = build_depth_preview(depth_cm, max_valid_depth_cm=float(args.max_valid_depth_cm))
    analysis = analyze_depth_entry(
        depth_cm=depth_cm,
        hfov_deg=float(args.hfov_deg),
        max_valid_depth_cm=float(args.max_valid_depth_cm),
        front_obstacle_threshold_cm=float(args.front_obstacle_threshold_cm),
        traversable_min_width_cm=float(args.traversable_min_width_cm),
        traversable_min_clearance_cm=float(args.traversable_min_clearance_cm),
        traversable_min_depth_gain_cm=float(args.traversable_min_depth_gain_cm),
    )
    analysis.update(
        {
            "depth_path": str(depth_path),
            "hfov_deg": float(args.hfov_deg),
            "max_valid_depth_cm": float(args.max_valid_depth_cm),
            "run_dir": str(run_dir),
        }
    )

    overlay = draw_overlay(preview, analysis)
    overlay_path = run_dir / "analysis_overlay.png"
    json_path = run_dir / "analysis_result.json"
    txt_path = run_dir / "analysis_summary.txt"

    cv2.imwrite(str(overlay_path), overlay)
    json_path.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    write_text_summary(txt_path, analysis)

    print("[phase2_depth_analysis] completed")
    print(f"[phase2_depth_analysis] overlay={overlay_path}")
    print(f"[phase2_depth_analysis] json={json_path}")
    print(f"[phase2_depth_analysis] txt={txt_path}")
    print(f"[phase2_depth_analysis] summary={analysis.get('summary', '')}")


if __name__ == "__main__":
    main()
