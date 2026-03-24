"""
Reference-house matcher for Phase 6.

This is a lightweight image similarity module that compares the current RGB
view against an optional target-house reference image.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import cv2
import numpy as np

from runtime_interfaces import build_reference_match_runtime_state


def _compute_descriptor(image: np.ndarray) -> Optional[np.ndarray]:
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return None
    resized = cv2.resize(image, (160, 120), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    edges = cv2.Canny(cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY), 60, 160)
    edge_density = np.array([float(np.mean(edges > 0))], dtype=np.float32)
    return np.concatenate([hist.astype(np.float32), edge_density], axis=0)


def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_norm = float(np.linalg.norm(lhs))
    rhs_norm = float(np.linalg.norm(rhs))
    if lhs_norm <= 1e-8 or rhs_norm <= 1e-8:
        return 0.0
    return float(np.dot(lhs, rhs) / (lhs_norm * rhs_norm))


class ReferenceHouseMatcher:
    def __init__(self, reference_image_path: str = "", threshold: float = 0.78) -> None:
        self.reference_image_path = str(reference_image_path or "").strip()
        self.threshold = float(threshold)
        self._cached_descriptor: Optional[np.ndarray] = None
        self._cached_path: str = ""
        self._last_error: str = ""

    def reset(self) -> None:
        self._cached_descriptor = None
        self._cached_path = ""
        self._last_error = ""

    def _load_reference_descriptor(self) -> Optional[np.ndarray]:
        path = self.reference_image_path
        if not path:
            self._last_error = "reference_image_unconfigured"
            return None
        if self._cached_descriptor is not None and self._cached_path == path:
            return self._cached_descriptor
        if not os.path.exists(path):
            self._last_error = f"reference_image_missing:{path}"
            return None
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            self._last_error = f"reference_image_unreadable:{path}"
            return None
        descriptor = _compute_descriptor(image)
        if descriptor is None:
            self._last_error = "reference_descriptor_empty"
            return None
        self._cached_descriptor = descriptor
        self._cached_path = path
        self._last_error = ""
        return descriptor

    def update_config(self, *, reference_image_path: str, threshold: float) -> None:
        changed = str(reference_image_path or "").strip() != self.reference_image_path
        self.reference_image_path = str(reference_image_path or "").strip()
        self.threshold = float(threshold)
        if changed:
            self.reset()

    def match(
        self,
        *,
        rgb_frame: Optional[np.ndarray],
        mission: Optional[Dict[str, Any]] = None,
        task_label: str = "",
    ) -> Dict[str, Any]:
        mission_payload = mission if isinstance(mission, dict) else {}
        if not self.reference_image_path:
            return build_reference_match_runtime_state(
                mission_id=str(mission_payload.get("mission_id", "")),
                task_label=str(task_label or ""),
                status="not_configured",
                source="reference_matcher",
                method="rgb_hist_cosine_v0",
                reference_image_path="",
                threshold=self.threshold,
                match_state="not_configured",
                match_confidence=0.0,
                summary="Target-house reference image is not configured.",
            )

        reference_descriptor = self._load_reference_descriptor()
        if reference_descriptor is None:
            return build_reference_match_runtime_state(
                mission_id=str(mission_payload.get("mission_id", "")),
                task_label=str(task_label or ""),
                status="error",
                source="reference_matcher",
                method="rgb_hist_cosine_v0",
                reference_image_path=self.reference_image_path,
                threshold=self.threshold,
                match_state="error",
                match_confidence=0.0,
                summary=self._last_error or "Failed to load reference descriptor.",
            )

        current_descriptor = _compute_descriptor(rgb_frame) if rgb_frame is not None else None
        if current_descriptor is None:
            return build_reference_match_runtime_state(
                mission_id=str(mission_payload.get("mission_id", "")),
                task_label=str(task_label or ""),
                status="no_frame",
                source="reference_matcher",
                method="rgb_hist_cosine_v0",
                reference_image_path=self.reference_image_path,
                threshold=self.threshold,
                match_state="unknown",
                match_confidence=0.0,
                summary="Current RGB frame is unavailable for target-house matching.",
            )

        similarity = _cosine_similarity(reference_descriptor, current_descriptor)
        if similarity >= self.threshold:
            match_state = "matched"
        elif similarity >= max(0.0, self.threshold - 0.12):
            match_state = "candidate"
        else:
            match_state = "not_matched"
        return build_reference_match_runtime_state(
            mission_id=str(mission_payload.get("mission_id", "")),
            task_label=str(task_label or ""),
            status="ok",
            source="reference_matcher",
            method="rgb_hist_cosine_v0",
            reference_image_path=self.reference_image_path,
            threshold=self.threshold,
            match_state=match_state,
            match_confidence=similarity,
            summary=f"reference_match={match_state} similarity={similarity:.3f} threshold={self.threshold:.3f}",
        )
