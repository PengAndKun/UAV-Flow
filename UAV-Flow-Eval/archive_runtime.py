"""
Lightweight Phase 3 archive runtime for the staged UAV navigation stack.

This file intentionally keeps the first archive implementation simple:
- goal-conditioned cell ids
- quantized pose bins
- depth/risk-aware summaries
- in-memory visit statistics and transition history

It is designed as a runtime scaffold for Phase 3 experiments, not as the final
offline Go-Explore archive builder.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional


def normalize_angle_deg(angle_deg: float) -> float:
    return (float(angle_deg) + 180.0) % 360.0 - 180.0


def slugify_label(text: str, default: str = "idle") -> str:
    raw = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text or "").strip())
    compact = "_".join(part for part in raw.split("_") if part)
    return compact or default


def quantize_value(value: float, bin_size: float) -> int:
    size = max(float(bin_size), 1e-6)
    return int(round(float(value) / size))


def build_depth_signature(depth_summary: Optional[Dict[str, Any]]) -> Dict[str, float]:
    payload = depth_summary if isinstance(depth_summary, dict) else {}
    return {
        "min_depth_cm": float(payload.get("min_depth", 0.0)),
        "max_depth_cm": float(payload.get("max_depth", 0.0)),
        "front_min_depth_cm": float(payload.get("front_min_depth", payload.get("min_depth", 0.0))),
        "front_mean_depth_cm": float(payload.get("front_mean_depth", payload.get("max_depth", 0.0))),
    }


def build_pose_bin(
    pose: Dict[str, Any],
    *,
    pos_bin_cm: float,
    yaw_bin_deg: float,
) -> Dict[str, int]:
    return {
        "qx": quantize_value(float(pose.get("x", 0.0)), pos_bin_cm),
        "qy": quantize_value(float(pose.get("y", 0.0)), pos_bin_cm),
        "qz": quantize_value(float(pose.get("z", 0.0)), pos_bin_cm),
        "qyaw": quantize_value(normalize_angle_deg(float(pose.get("yaw", 0.0))), yaw_bin_deg),
    }


def build_archive_cell_id(
    *,
    task_label: str,
    semantic_subgoal: str,
    pose_bin: Dict[str, int],
    depth_signature: Dict[str, float],
    depth_bin_cm: float,
) -> str:
    task_key = slugify_label(task_label or "idle")
    subgoal_key = slugify_label(semantic_subgoal or "idle")
    front_depth_bin = quantize_value(depth_signature.get("front_min_depth_cm", 0.0), depth_bin_cm)
    return (
        f"{task_key}__{subgoal_key}"
        f"__x{pose_bin['qx']}_y{pose_bin['qy']}_z{pose_bin['qz']}_yaw{pose_bin['qyaw']}"
        f"__fd{front_depth_bin}"
    )


@dataclass
class ArchiveRuntime:
    pos_bin_cm: float = 200.0
    yaw_bin_deg: float = 30.0
    depth_bin_cm: float = 100.0
    recent_limit: int = 6
    cells: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    transitions: Dict[str, int] = field(default_factory=dict)
    recent_cell_ids: Deque[str] = field(default_factory=lambda: deque(maxlen=6))
    current_cell_id: str = ""
    last_transition_key: str = ""
    last_retrieval: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.recent_cell_ids = deque(maxlen=max(1, int(self.recent_limit)))

    def _summarize_cell(self, cell: Dict[str, Any], retrieval_score: Optional[float] = None) -> Dict[str, Any]:
        return {
            "cell_id": cell["cell_id"],
            "task_label": cell["task_label"],
            "semantic_subgoal": cell["semantic_subgoal"],
            "visit_count": int(cell["visit_count"]),
            "last_seen_at": cell["last_seen_at"],
            "last_frame_id": cell["last_frame_id"],
            "last_action": cell["last_action"],
            "risk_score": float(cell["risk_score"]),
            "pose_bin": dict(cell["pose_bin"]),
            "latest_pose": dict(cell["latest_pose"]),
            "depth_signature": dict(cell["depth_signature"]),
            "current_waypoint": dict(cell["current_waypoint"]) if isinstance(cell.get("current_waypoint"), dict) else None,
            "retrieval_score": None if retrieval_score is None else float(retrieval_score),
        }

    def _rank_candidates(
        self,
        *,
        task_label: str,
        semantic_subgoal: str,
        current_cell_id: str,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        task_key = slugify_label(task_label or "idle")
        subgoal_key = slugify_label(semantic_subgoal or "idle")
        ranked: List[tuple] = []
        for cell in self.cells.values():
            if cell["cell_id"] == current_cell_id:
                continue
            task_match = 1 if slugify_label(cell["task_label"]) == task_key else 0
            subgoal_match = 1 if slugify_label(cell["semantic_subgoal"]) == subgoal_key else 0
            retrieval_score = (
                (2.0 * task_match)
                + (1.5 * subgoal_match)
                + min(int(cell["visit_count"]), 10) * 0.1
                - float(cell["risk_score"]) * 0.5
            )
            ranked.append(
                (
                    retrieval_score,
                    int(cell["visit_count"]),
                    -float(cell["risk_score"]),
                    cell["cell_id"],
                )
            )
        ranked.sort(reverse=True)
        result: List[Dict[str, Any]] = []
        for retrieval_score, _, _, cell_id in ranked[: max(0, int(limit))]:
            result.append(self._summarize_cell(self.cells[cell_id], retrieval_score=float(retrieval_score)))
        return result

    def register_observation(
        self,
        *,
        timestamp: str,
        frame_id: str,
        task_label: str,
        semantic_subgoal: str,
        pose: Dict[str, Any],
        depth_summary: Optional[Dict[str, Any]] = None,
        current_waypoint: Optional[Dict[str, Any]] = None,
        action_label: str = "idle",
        risk_score: float = 0.0,
    ) -> Dict[str, Any]:
        pose_bin = build_pose_bin(
            pose,
            pos_bin_cm=self.pos_bin_cm,
            yaw_bin_deg=self.yaw_bin_deg,
        )
        depth_signature = build_depth_signature(depth_summary)
        cell_id = build_archive_cell_id(
            task_label=task_label,
            semantic_subgoal=semantic_subgoal,
            pose_bin=pose_bin,
            depth_signature=depth_signature,
            depth_bin_cm=self.depth_bin_cm,
        )

        cell = self.cells.get(cell_id)
        if cell is None:
            cell = {
                "cell_id": cell_id,
                "task_label": str(task_label or ""),
                "semantic_subgoal": str(semantic_subgoal or "idle"),
                "created_at": timestamp,
                "first_seen_at": timestamp,
                "last_seen_at": timestamp,
                "last_frame_id": frame_id,
                "visit_count": 0,
                "last_action": "idle",
                "risk_score": 0.0,
                "pose_bin": pose_bin,
                "latest_pose": dict(pose),
                "depth_signature": depth_signature,
                "current_waypoint": current_waypoint,
                "transition_in_count": 0,
                "transition_out_count": 0,
            }
            self.cells[cell_id] = cell

        previous_cell_id = self.current_cell_id
        if previous_cell_id and previous_cell_id != cell_id:
            transition_key = f"{previous_cell_id}->{cell_id}"
            self.transitions[transition_key] = int(self.transitions.get(transition_key, 0)) + 1
            self.last_transition_key = transition_key
            if previous_cell_id in self.cells:
                self.cells[previous_cell_id]["transition_out_count"] = int(self.cells[previous_cell_id].get("transition_out_count", 0)) + 1
            cell["transition_in_count"] = int(cell.get("transition_in_count", 0)) + 1

        cell["last_seen_at"] = timestamp
        cell["last_frame_id"] = frame_id
        cell["visit_count"] = int(cell.get("visit_count", 0)) + 1
        cell["last_action"] = str(action_label or "idle")
        cell["risk_score"] = float(risk_score)
        cell["latest_pose"] = dict(pose)
        cell["pose_bin"] = pose_bin
        cell["depth_signature"] = depth_signature
        cell["current_waypoint"] = dict(current_waypoint) if isinstance(current_waypoint, dict) else None

        self.current_cell_id = cell_id
        self.recent_cell_ids.append(cell_id)

        retrieval_candidates = self._rank_candidates(
            task_label=task_label,
            semantic_subgoal=semantic_subgoal,
            current_cell_id=cell_id,
        )
        self.last_retrieval = retrieval_candidates[0] if retrieval_candidates else {}
        return {
            "cell": self._summarize_cell(cell),
            "current_cell_id": cell_id,
            "retrieval_candidates": retrieval_candidates,
            "active_retrieval": dict(self.last_retrieval) if self.last_retrieval else None,
            "cell_count": len(self.cells),
            "transition_count": sum(self.transitions.values()),
            "recent_cell_ids": list(self.recent_cell_ids),
            "last_transition_key": self.last_transition_key,
        }

    def get_state(self, *, limit: int = 6) -> Dict[str, Any]:
        current_cell = self.cells.get(self.current_cell_id) if self.current_cell_id else None
        top_cells = sorted(
            (self._summarize_cell(cell) for cell in self.cells.values()),
            key=lambda item: (int(item["visit_count"]), item["last_seen_at"]),
            reverse=True,
        )[: max(1, int(limit))]
        return {
            "enabled": True,
            "current_cell_id": self.current_cell_id,
            "cell_count": len(self.cells),
            "transition_count": sum(self.transitions.values()),
            "recent_cell_ids": list(self.recent_cell_ids),
            "last_transition_key": self.last_transition_key,
            "current_cell": self._summarize_cell(current_cell) if current_cell else None,
            "active_retrieval": dict(self.last_retrieval) if self.last_retrieval else None,
            "top_cells": top_cells,
        }

    def get_planner_context(
        self,
        *,
        task_label: str,
        semantic_subgoal: str,
        limit: int = 3,
    ) -> Dict[str, Any]:
        retrieval_candidates = self._rank_candidates(
            task_label=task_label,
            semantic_subgoal=semantic_subgoal,
            current_cell_id=self.current_cell_id,
            limit=limit,
        )
        active_retrieval = retrieval_candidates[0] if retrieval_candidates else None
        self.last_retrieval = dict(active_retrieval) if active_retrieval else {}
        return {
            "current_cell_id": self.current_cell_id,
            "cell_count": len(self.cells),
            "recent_cell_ids": list(self.recent_cell_ids),
            "retrieval_candidates": retrieval_candidates,
            "active_retrieval": active_retrieval,
        }
