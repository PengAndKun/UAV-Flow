r"""
Standalone validator for Phase 5 doorway detection.

Typical usage:

python E:\github\UAV-Flow\UAV-Flow-Eval\validate_doorway_detection.py ^
  --bundle_json E:\github\UAV-Flow\captures_remote\capture_xxx_bundle.json ^
  --save_overlay
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from doorway_detection import detect_doorway_runtime


def _load_bundle_paths(bundle_json: Path) -> Tuple[Path, Path]:
    payload = json.loads(bundle_json.read_text(encoding="utf-8"))
    rgb_path = Path(str(payload.get("rgb_image_path", "")))
    depth_path = Path(str(payload.get("depth_image_path", "")))
    if not rgb_path.is_file():
        raise FileNotFoundError(f"RGB image not found in bundle: {rgb_path}")
    if not depth_path.is_file():
        raise FileNotFoundError(f"Depth image not found in bundle: {depth_path}")
    return rgb_path, depth_path


def _draw_overlay(rgb: np.ndarray, runtime: Dict[str, Any]) -> np.ndarray:
    overlay = rgb.copy()
    candidates = runtime.get("candidates", []) if isinstance(runtime.get("candidates"), list) else []
    for idx, candidate in enumerate(candidates):
        bbox = candidate.get("bbox", {}) if isinstance(candidate.get("bbox"), dict) else {}
        x = int(bbox.get("x", 0))
        y = int(bbox.get("y", 0))
        w = int(bbox.get("width", 0))
        h = int(bbox.get("height", 0))
        traversable = bool(candidate.get("traversable", False))
        confidence = float(candidate.get("confidence", 0.0) or 0.0)
        label = str(candidate.get("label", "door_candidate") or "door_candidate")
        color = (60, 220, 60) if traversable else (0, 180, 255)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
        text = f"{idx + 1}:{label} conf={confidence:.2f}"
        cv2.putText(
            overlay,
            text,
            (x, max(16, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    summary = str(runtime.get("summary", "") or "no summary")
    cv2.putText(
        overlay,
        summary[:110],
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay,
        summary[:110],
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        (30, 30, 30),
        1,
        cv2.LINE_AA,
    )
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate doorway detection on a captured RGB+depth pair.")
    parser.add_argument("--bundle_json", type=Path, help="Capture bundle JSON containing rgb_image_path and depth_image_path.")
    parser.add_argument("--rgb_path", type=Path, help="RGB image path if not using --bundle_json.")
    parser.add_argument("--depth_path", type=Path, help="Depth planar image path if not using --bundle_json.")
    parser.add_argument("--save_overlay", action="store_true", help="Save an RGB overlay with doorway candidates.")
    parser.add_argument("--output_json", type=Path, help="Optional output JSON path for detector runtime.")
    parser.add_argument("--output_overlay", type=Path, help="Optional overlay path. Defaults next to bundle/rgb.")
    args = parser.parse_args()

    if args.bundle_json:
        rgb_path, depth_path = _load_bundle_paths(args.bundle_json)
        output_base = args.bundle_json.with_suffix("")
    else:
        if not args.rgb_path or not args.depth_path:
            raise SystemExit("Provide either --bundle_json or both --rgb_path and --depth_path.")
        rgb_path = args.rgb_path
        depth_path = args.depth_path
        output_base = rgb_path.with_suffix("")

    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if rgb is None:
        raise FileNotFoundError(f"Failed to read RGB image: {rgb_path}")
    if depth is None:
        raise FileNotFoundError(f"Failed to read depth image: {depth_path}")

    runtime = detect_doorway_runtime(
        rgb_frame=rgb,
        depth_frame=depth,
        depth_summary={},
    )

    print(json.dumps(runtime, indent=2, ensure_ascii=False))

    output_json = args.output_json or output_base.parent / f"{output_base.name}_doorway_runtime.json"
    output_json.write_text(json.dumps(runtime, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.save_overlay:
        overlay = _draw_overlay(rgb, runtime)
        output_overlay = args.output_overlay or output_base.parent / f"{output_base.name}_doorway_overlay.png"
        cv2.imwrite(str(output_overlay), overlay)
        print(f"overlay_saved={output_overlay}")


if __name__ == "__main__":
    main()
