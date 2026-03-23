"""
Language-first search memory helpers for person-search missions.

This module keeps a compact natural-language summary of the current search
state, region-level observations, and recent evidence/planner events. The goal
is to complement the structured archive runtime with a planner-friendly text
memory rather than replace it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from runtime_interfaces import build_language_search_memory_state, build_search_region, now_timestamp


def _normalize_region_label(label: Any) -> str:
    return str(label or "").strip().lower()


def _coerce_region(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    return build_search_region(
        region_id=str(raw.get("region_id", raw.get("id", ""))),
        region_label=str(raw.get("region_label", raw.get("label", ""))),
        region_type=str(raw.get("region_type", "area")),
        room_type=str(raw.get("room_type", "")),
        priority=int(raw.get("priority", 0) or 0),
        status=str(raw.get("status", "unobserved")),
        rationale=str(raw.get("rationale", raw.get("reason", ""))),
    )


class LanguageSearchMemory:
    """Maintain lightweight language summaries for house-search episodes."""

    def __init__(self, *, recent_limit: int = 8, region_limit: int = 8) -> None:
        self.recent_limit = max(4, int(recent_limit))
        self.region_limit = max(4, int(region_limit))
        self._note_counter = 0
        self._last_plan_signature = ""
        self._last_focus_region_label = ""
        self.state = build_language_search_memory_state()
        self._region_notes_by_label: Dict[str, Dict[str, Any]] = {}
        self._recent_notes: List[Dict[str, Any]] = []

    def reset(self, *, mission_id: str, mission_type: str, task_label: str) -> Dict[str, Any]:
        self._note_counter = 0
        self._last_plan_signature = ""
        self._last_focus_region_label = ""
        self._region_notes_by_label = {}
        self._recent_notes = []
        self.state = build_language_search_memory_state(
            mission_id=mission_id,
            mission_type=mission_type,
            task_label=task_label,
            global_summary="Language search memory initialized.",
        )
        if task_label:
            self._append_recent_note(
                note_type="mission_reset",
                text=f"Mission reset for task: {task_label}.",
            )
        return self.get_state()

    def _append_recent_note(
        self,
        *,
        note_type: str,
        text: str,
        region_label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        note_text = str(text or "").strip()
        if not note_text:
            return
        region_label_text = str(region_label or "").strip()
        if self._recent_notes:
            last_note = self._recent_notes[-1]
            if (
                str(last_note.get("note_type", "")) == str(note_type or "")
                and str(last_note.get("text", "")) == note_text
                and str(last_note.get("region_label", "")) == region_label_text
            ):
                return
        self._note_counter += 1
        note = {
            "note_id": f"langmem_{self._note_counter:05d}",
            "timestamp": timestamp or now_timestamp(),
            "note_type": str(note_type or ""),
            "region_label": region_label_text,
            "text": note_text,
            "metadata": metadata or {},
        }
        self._recent_notes.append(note)
        if len(self._recent_notes) > self.recent_limit:
            self._recent_notes = self._recent_notes[-self.recent_limit :]

    def _pick_focus_region(
        self,
        *,
        mission: Dict[str, Any],
        search_runtime: Dict[str, Any],
        current_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        runtime_priority = _coerce_region(search_runtime.get("priority_region"))
        if runtime_priority and (
            runtime_priority.get("region_label") or runtime_priority.get("region_id")
        ):
            return runtime_priority
        if isinstance(current_plan, dict):
            plan_priority = _coerce_region(current_plan.get("priority_region"))
            if plan_priority and (
                plan_priority.get("region_label") or plan_priority.get("region_id")
            ):
                return plan_priority
        mission_regions = mission.get("priority_regions") if isinstance(mission.get("priority_regions"), list) else []
        for raw_region in mission_regions:
            region = _coerce_region(raw_region)
            if region and (region.get("region_label") or region.get("region_id")):
                return region
        return {}

    def _build_global_summary(
        self,
        *,
        mission: Dict[str, Any],
        search_runtime: Dict[str, Any],
        person_evidence_runtime: Dict[str, Any],
        search_result: Dict[str, Any],
        focus_region: Dict[str, Any],
    ) -> str:
        mission_type = str(mission.get("mission_type", "semantic_navigation") or "semantic_navigation")
        task_label = str(mission.get("task_label", "") or self.state.get("task_label", "")).strip()
        current_subgoal = str(search_runtime.get("current_search_subgoal", "idle") or "idle")
        focus_label = str(focus_region.get("region_label", "") or "current local view")
        visited_count = int(search_runtime.get("visited_region_count", 0))
        suspect_count = int(person_evidence_runtime.get("suspect_count", 0))
        present_count = int(person_evidence_runtime.get("confirm_present_count", 0))
        absent_count = int(person_evidence_runtime.get("confirm_absent_count", 0))
        result_status = str(search_result.get("result_status", "unknown") or "unknown")
        summary_parts = [
            f"Mission type={mission_type}",
            f"task={task_label or 'idle'}",
            f"subgoal={current_subgoal}",
            f"focus={focus_label}",
            f"visited={visited_count}",
            f"suspect={suspect_count}",
            f"confirmed_present={present_count}",
            f"confirmed_absent={absent_count}",
            f"result={result_status}",
        ]
        return "; ".join(summary_parts) + "."

    def _build_focus_summary(
        self,
        *,
        focus_region: Dict[str, Any],
        search_runtime: Dict[str, Any],
        person_evidence_runtime: Dict[str, Any],
        search_result: Dict[str, Any],
        archive_state: Dict[str, Any],
        current_plan: Optional[Dict[str, Any]] = None,
    ) -> str:
        focus_label = str(focus_region.get("region_label", "") or "current local view")
        focus_status = str(focus_region.get("status", "unobserved") or "unobserved")
        search_subgoal = str(search_runtime.get("current_search_subgoal", "idle") or "idle")
        evidence_status = str(person_evidence_runtime.get("evidence_status", "idle") or "idle")
        result_status = str(search_result.get("result_status", "unknown") or "unknown")
        archive_top = archive_state.get("top_cells") if isinstance(archive_state.get("top_cells"), list) else []
        archive_hint = ""
        if archive_top:
            archive_hint = str(archive_top[0].get("semantic_subgoal", "") or "")
        rationale = str(focus_region.get("rationale", "") or "")
        waypoint_strategy = ""
        if isinstance(current_plan, dict):
            plan_debug = current_plan.get("debug") if isinstance(current_plan.get("debug"), dict) else {}
            waypoint_strategy = str(plan_debug.get("waypoint_strategy", "") or "")
        parts = [
            f"{focus_label} status={focus_status}",
            f"subgoal={search_subgoal}",
            f"evidence={evidence_status}",
            f"result={result_status}",
        ]
        if archive_hint:
            parts.append(f"archive_hint={archive_hint}")
        if waypoint_strategy:
            parts.append(f"waypoint_strategy={waypoint_strategy}")
        if rationale:
            parts.append(f"why={rationale}")
        return "; ".join(parts) + "."

    def sync(
        self,
        *,
        mission: Dict[str, Any],
        search_runtime: Dict[str, Any],
        person_evidence_runtime: Dict[str, Any],
        search_result: Dict[str, Any],
        archive_state: Optional[Dict[str, Any]] = None,
        current_plan: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        archive_snapshot = archive_state or {}
        mission_id = str(mission.get("mission_id", self.state.get("mission_id", "")) or "")
        mission_type = str(mission.get("mission_type", self.state.get("mission_type", "semantic_navigation")) or "semantic_navigation")
        task_label = str(mission.get("task_label", self.state.get("task_label", "")) or self.state.get("task_label", ""))
        focus_region = self._pick_focus_region(
            mission=mission,
            search_runtime=search_runtime,
            current_plan=current_plan,
        )
        focus_label = str(focus_region.get("region_label", "") or "")
        focus_summary = self._build_focus_summary(
            focus_region=focus_region,
            search_runtime=search_runtime,
            person_evidence_runtime=person_evidence_runtime,
            search_result=search_result,
            archive_state=archive_snapshot,
            current_plan=current_plan,
        )
        global_summary = self._build_global_summary(
            mission=mission,
            search_runtime=search_runtime,
            person_evidence_runtime=person_evidence_runtime,
            search_result=search_result,
            focus_region=focus_region,
        )
        if focus_label:
            region_key = _normalize_region_label(focus_label)
            region_note = {
                "region_label": focus_label,
                "status": str(focus_region.get("status", "unobserved") or "unobserved"),
                "summary": focus_summary,
                "last_updated_at": now_timestamp(),
            }
            self._region_notes_by_label[region_key] = region_note
            if len(self._region_notes_by_label) > self.region_limit:
                ordered = sorted(
                    self._region_notes_by_label.items(),
                    key=lambda item: str(item[1].get("last_updated_at", "")),
                    reverse=True,
                )[: self.region_limit]
                self._region_notes_by_label = {key: value for key, value in ordered}
            if region_key != self._last_focus_region_label:
                self._append_recent_note(
                    note_type="focus_region_update",
                    text=f"Planner focus shifted to {focus_label}.",
                    region_label=focus_label,
                )
                self._last_focus_region_label = region_key
        self.state = build_language_search_memory_state(
            mission_id=mission_id,
            mission_type=mission_type,
            task_label=task_label,
            global_summary=global_summary,
            current_focus_region=focus_region,
            current_focus_summary=focus_summary,
            region_notes=list(self._region_notes_by_label.values()),
            recent_notes=list(self._recent_notes),
            note_count=len(self._recent_notes),
            region_note_count=len(self._region_notes_by_label),
        )
        return self.get_state()

    def record_plan(self, *, plan: Dict[str, Any], search_runtime: Dict[str, Any]) -> Dict[str, Any]:
        search_subgoal = str(plan.get("search_subgoal", search_runtime.get("current_search_subgoal", "idle")) or "idle")
        semantic_subgoal = str(plan.get("semantic_subgoal", "idle") or "idle")
        priority_region = _coerce_region(plan.get("priority_region"))
        focus_label = str(priority_region.get("region_label", "") or "current local view")
        signature = "|".join([search_subgoal, semantic_subgoal, focus_label])
        if signature != self._last_plan_signature:
            self._append_recent_note(
                note_type="planner_update",
                text=f"Planner set subgoal={search_subgoal} semantic={semantic_subgoal} focus={focus_label}.",
                region_label=focus_label,
            )
            self._last_plan_signature = signature
        return self.get_state()

    def record_evidence(
        self,
        *,
        event_type: str,
        note: str,
        region: Optional[Dict[str, Any]],
        confidence: float,
        search_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        safe_region = _coerce_region(region)
        region_label = str(safe_region.get("region_label", "") or "current local view")
        result_status = str(search_result.get("result_status", "unknown") or "unknown")
        base_text = f"Evidence {event_type} at {region_label} conf={float(confidence):.2f} result={result_status}."
        if note:
            base_text += f" note={str(note).strip()}."
        self._append_recent_note(
            note_type=f"evidence_{event_type}",
            text=base_text,
            region_label=region_label,
            metadata={"result_status": result_status},
        )
        return self.get_state()

    def get_state(self) -> Dict[str, Any]:
        self.state["recent_notes"] = list(self._recent_notes)
        self.state["region_notes"] = list(self._region_notes_by_label.values())
        self.state["note_count"] = len(self._recent_notes)
        self.state["region_note_count"] = len(self._region_notes_by_label)
        self.state["last_updated_at"] = now_timestamp()
        return build_language_search_memory_state(
            mission_id=str(self.state.get("mission_id", "")),
            mission_type=str(self.state.get("mission_type", "semantic_navigation")),
            task_label=str(self.state.get("task_label", "")),
            global_summary=str(self.state.get("global_summary", "")),
            current_focus_region=self.state.get("current_focus_region", {}) if isinstance(self.state.get("current_focus_region"), dict) else {},
            current_focus_summary=str(self.state.get("current_focus_summary", "")),
            region_notes=list(self._region_notes_by_label.values()),
            recent_notes=list(self._recent_notes),
            note_count=len(self._recent_notes),
            region_note_count=len(self._region_notes_by_label),
            last_updated_at=str(self.state.get("last_updated_at", now_timestamp())),
        )
