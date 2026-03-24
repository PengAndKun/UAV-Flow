"""
Semantic archive runtime for Phase 6.

Stores language-centric scene summaries and retrieves semantically similar
entries using a simple hashed bag-of-words representation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from runtime_interfaces import build_semantic_archive_runtime_state, now_timestamp


TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(str(text or "").strip().lower())


def _counter_similarity(lhs: Counter[str], rhs: Counter[str]) -> float:
    if not lhs or not rhs:
        return 0.0
    dot = 0.0
    for token, lhs_count in lhs.items():
        dot += float(lhs_count * rhs.get(token, 0))
    lhs_norm = math.sqrt(sum(float(v * v) for v in lhs.values()))
    rhs_norm = math.sqrt(sum(float(v * v) for v in rhs.values()))
    if lhs_norm <= 1e-8 or rhs_norm <= 1e-8:
        return 0.0
    return dot / (lhs_norm * rhs_norm)


class SemanticArchiveRuntime:
    def __init__(self, *, max_entries: int = 256, retrieval_limit: int = 5) -> None:
        self.max_entries = max(16, int(max_entries))
        self.retrieval_limit = max(1, int(retrieval_limit))
        self.entries: List[Dict[str, Any]] = []
        self.counter: int = 0

    def reset(self) -> None:
        self.entries = []
        self.counter = 0

    def _build_entry(
        self,
        *,
        mission_id: str,
        task_label: str,
        stage_label: str,
        scene_description: str,
        semantic_text: str,
        pose: Optional[Dict[str, Any]],
        doorway_runtime: Optional[Dict[str, Any]],
        reference_match_runtime: Optional[Dict[str, Any]],
        current_plan: Optional[Dict[str, Any]],
        outcome_status: str,
    ) -> Dict[str, Any]:
        combined_text = " ".join(part for part in [scene_description, semantic_text, stage_label, task_label] if part)
        return {
            "archive_id": f"sem_arch_{self.counter:06d}",
            "mission_id": str(mission_id or ""),
            "task_label": str(task_label or ""),
            "stage_label": str(stage_label or ""),
            "scene_description": str(scene_description or ""),
            "semantic_text": str(semantic_text or ""),
            "pose": dict(pose or {}),
            "doorway_summary": str(_as_dict(doorway_runtime).get("summary", "") or ""),
            "target_house_match_state": str(_as_dict(reference_match_runtime).get("match_state", "unknown") or "unknown"),
            "action_hint": str(_as_dict(current_plan).get("semantic_subgoal", "") or ""),
            "outcome_status": str(outcome_status or "unknown"),
            "visit_count": 1,
            "updated_at": now_timestamp(),
            "_token_counter": Counter(_tokenize(combined_text)),
        }

    def update(
        self,
        *,
        mission_id: str,
        task_label: str,
        stage_label: str,
        scene_description: str,
        semantic_text: str,
        pose: Optional[Dict[str, Any]] = None,
        doorway_runtime: Optional[Dict[str, Any]] = None,
        reference_match_runtime: Optional[Dict[str, Any]] = None,
        current_plan: Optional[Dict[str, Any]] = None,
        outcome_status: str = "active",
    ) -> Dict[str, Any]:
        self.counter += 1
        new_entry = self._build_entry(
            mission_id=mission_id,
            task_label=task_label,
            stage_label=stage_label,
            scene_description=scene_description,
            semantic_text=semantic_text,
            pose=pose,
            doorway_runtime=doorway_runtime,
            reference_match_runtime=reference_match_runtime,
            current_plan=current_plan,
            outcome_status=outcome_status,
        )
        best_index = -1
        best_similarity = 0.0
        for index, entry in enumerate(self.entries):
            if str(entry.get("mission_id", "")) != str(mission_id or ""):
                continue
            if str(entry.get("stage_label", "")) != str(stage_label or ""):
                continue
            similarity = _counter_similarity(
                new_entry.get("_token_counter", Counter()),
                entry.get("_token_counter", Counter()),
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_index = index
        if best_index >= 0 and best_similarity >= 0.92:
            existing = self.entries[best_index]
            existing["visit_count"] = int(existing.get("visit_count", 0) or 0) + 1
            existing["updated_at"] = now_timestamp()
            existing["scene_description"] = str(scene_description or existing.get("scene_description", ""))
            existing["semantic_text"] = str(semantic_text or existing.get("semantic_text", ""))
            existing["doorway_summary"] = str(_as_dict(doorway_runtime).get("summary", "") or existing.get("doorway_summary", ""))
            existing["target_house_match_state"] = str(
                _as_dict(reference_match_runtime).get("match_state", existing.get("target_house_match_state", "unknown"))
            )
            existing["action_hint"] = str(_as_dict(current_plan).get("semantic_subgoal", existing.get("action_hint", "")) or "")
            existing["outcome_status"] = str(outcome_status or existing.get("outcome_status", "active"))
            current_entry = dict(existing)
        else:
            self.entries.append(new_entry)
            if len(self.entries) > self.max_entries:
                self.entries = self.entries[-self.max_entries :]
            current_entry = dict(new_entry)
        current_entry.pop("_token_counter", None)
        return self.get_state(
            mission_id=mission_id,
            query_text=f"{scene_description} {semantic_text}",
            current_entry=current_entry,
        )

    def _retrieve(self, *, mission_id: str, query_text: str) -> List[Tuple[float, Dict[str, Any]]]:
        query_counter = Counter(_tokenize(query_text))
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for entry in self.entries:
            if str(entry.get("mission_id", "")) != str(mission_id or ""):
                continue
            similarity = _counter_similarity(query_counter, entry.get("_token_counter", Counter()))
            if similarity <= 0.0:
                continue
            sanitized = dict(entry)
            sanitized.pop("_token_counter", None)
            scored.append((similarity, sanitized))
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[: self.retrieval_limit]

    def get_state(
        self,
        *,
        mission_id: str = "",
        query_text: str = "",
        current_entry: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        scored = self._retrieve(mission_id=mission_id, query_text=query_text) if mission_id and query_text else []
        top_matches = []
        for similarity, entry in scored:
            top_matches.append(
                {
                    "archive_id": str(entry.get("archive_id", "")),
                    "stage_label": str(entry.get("stage_label", "")),
                    "scene_description": str(entry.get("scene_description", "")),
                    "semantic_text": str(entry.get("semantic_text", "")),
                    "action_hint": str(entry.get("action_hint", "")),
                    "outcome_status": str(entry.get("outcome_status", "")),
                    "similarity": round(float(similarity), 4),
                    "visit_count": int(entry.get("visit_count", 0) or 0),
                    "updated_at": str(entry.get("updated_at", "")),
                }
            )
        summary = "empty semantic archive"
        if current_entry:
            summary = (
                f"stage={current_entry.get('stage_label', 'unknown')} "
                f"visits={int(current_entry.get('visit_count', 1) or 1)} "
                f"matches={len(top_matches)}"
            )
        elif self.entries:
            summary = f"entries={len(self.entries)} retrieval_limit={self.retrieval_limit}"
        return build_semantic_archive_runtime_state(
            mission_id=str(mission_id or ""),
            status="ok" if self.entries else "idle",
            source="local_text_archive",
            entry_count=len(self.entries),
            current_entry=current_entry or {},
            top_matches=top_matches,
            summary=summary,
        )
