from __future__ import annotations

import copy
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_ENTRY_SEARCH_MEMORY_PATH = os.path.join(
    PROJECT_ROOT,
    "phase2_multimodal_fusion_analysis",
    "entry_search_memory.json",
)
DEFAULT_HOUSES_CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "UAV-Flow-Eval",
    "houses_config.json",
)

DEFAULT_VERSION = "v1"
DEFAULT_RECENT_ACTIONS_LIMIT = 5
DEFAULT_RECENT_DECISIONS_LIMIT = 5
DEFAULT_TOP_CANDIDATES_LIMIT = 3
DEFAULT_EPISODIC_LIMIT = 64
DEFAULT_SECTOR_IDS = [
    "front_left",
    "front_center",
    "front_right",
    "left_side",
    "right_side",
]


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _deep_merge_dict(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge_dict(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def _default_working_memory() -> Dict[str, Any]:
    return {
        "target_house_id": "",
        "current_house_id": "",
        "last_best_entry_id": "",
        "recent_actions": [],
        "recent_target_decisions": [],
        "top_candidates": [],
    }


def _default_semantic_memory() -> Dict[str, Any]:
    return {
        "entry_search_status": "not_started",
        "last_best_entry_id": "",
        "search_summary": {
            "observed_sector_count": 0,
            "candidate_entry_count": 0,
            "approachable_entry_count": 0,
            "blocked_entry_count": 0,
            "rejected_entry_count": 0,
        },
        "searched_sectors": {
            sector_id: {
                "observed": False,
                "observation_count": 0,
                "last_visit_time": None,
                "best_entry_state": "",
                "best_target_conditioned_subgoal": "",
                "best_target_match_score": 0.0,
            }
            for sector_id in DEFAULT_SECTOR_IDS
        },
        "candidate_entries": [],
    }


def _default_house_memory(house_id: str, house_name: str = "", house_status: str = "UNSEARCHED") -> Dict[str, Any]:
    timestamp = _now_text()
    return {
        "house_id": str(house_id),
        "house_name": str(house_name or house_id),
        "house_status": str(house_status or "UNSEARCHED"),
        "target_match_active": False,
        "created_at": timestamp,
        "updated_at": timestamp,
        "working_memory": _default_working_memory(),
        "episodic_memory": [],
        "semantic_memory": _default_semantic_memory(),
    }


class EntrySearchMemoryStore:
    def __init__(
        self,
        path: str = DEFAULT_ENTRY_SEARCH_MEMORY_PATH,
        *,
        recent_actions_limit: int = DEFAULT_RECENT_ACTIONS_LIMIT,
        recent_decisions_limit: int = DEFAULT_RECENT_DECISIONS_LIMIT,
        top_candidates_limit: int = DEFAULT_TOP_CANDIDATES_LIMIT,
        episodic_limit: int = DEFAULT_EPISODIC_LIMIT,
    ) -> None:
        self.path = os.path.abspath(path)
        self.recent_actions_limit = max(1, int(recent_actions_limit))
        self.recent_decisions_limit = max(1, int(recent_decisions_limit))
        self.top_candidates_limit = max(1, int(top_candidates_limit))
        self.episodic_limit = max(1, int(episodic_limit))
        self.data: Dict[str, Any] = {
            "version": DEFAULT_VERSION,
            "updated_at": _now_text(),
            "current_target_house_id": "",
            "memories": {},
        }

    def load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            self.data = {
                "version": DEFAULT_VERSION,
                "updated_at": _now_text(),
                "current_target_house_id": "",
                "memories": {},
            }
            return self.data
        with open(self.path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        self.data = raw if isinstance(raw, dict) else {}
        self.data.setdefault("version", DEFAULT_VERSION)
        self.data.setdefault("updated_at", _now_text())
        self.data.setdefault("current_target_house_id", "")
        memories = self.data.get("memories")
        self.data["memories"] = memories if isinstance(memories, dict) else {}
        return self.data

    def save(self) -> str:
        self.data["updated_at"] = _now_text()
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2, ensure_ascii=False)
        return self.path

    def to_dict(self) -> Dict[str, Any]:
        return copy.deepcopy(self.data)

    def set_current_target_house(self, house_id: str) -> None:
        self.data["current_target_house_id"] = str(house_id or "")
        target_house_id = str(house_id or "")
        for memory_house_id, memory in self.memories.items():
            if isinstance(memory, dict):
                memory["target_match_active"] = memory_house_id == target_house_id
                memory["updated_at"] = _now_text()

    @property
    def memories(self) -> Dict[str, Dict[str, Any]]:
        memories = self.data.setdefault("memories", {})
        return memories if isinstance(memories, dict) else {}

    def ensure_house(
        self,
        house_id: str,
        *,
        house_name: str = "",
        house_status: str = "UNSEARCHED",
    ) -> Dict[str, Any]:
        hid = str(house_id or "").strip()
        if not hid:
            raise ValueError("house_id is required")
        memory = self.memories.get(hid)
        if not isinstance(memory, dict):
            memory = _default_house_memory(hid, house_name=house_name, house_status=house_status)
            self.memories[hid] = memory
        else:
            memory.setdefault("house_id", hid)
            memory.setdefault("house_name", str(house_name or hid))
            memory.setdefault("house_status", str(house_status or "UNSEARCHED"))
            memory.setdefault("target_match_active", False)
            memory.setdefault("created_at", _now_text())
            memory.setdefault("updated_at", _now_text())
            if not isinstance(memory.get("working_memory"), dict):
                memory["working_memory"] = _default_working_memory()
            if not isinstance(memory.get("episodic_memory"), list):
                memory["episodic_memory"] = []
            if not isinstance(memory.get("semantic_memory"), dict):
                memory["semantic_memory"] = _default_semantic_memory()
        memory["house_name"] = str(house_name or memory.get("house_name") or hid)
        memory["house_status"] = str(house_status or memory.get("house_status") or "UNSEARCHED")
        self._normalize_house_memory(memory)
        return memory

    def ensure_from_houses_config(self, houses_config_path: str = DEFAULT_HOUSES_CONFIG_PATH) -> Dict[str, Dict[str, Any]]:
        path = os.path.abspath(houses_config_path)
        if not os.path.exists(path):
            return self.memories
        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)
        houses = raw.get("houses", [])
        if isinstance(houses, list):
            for house in houses:
                if not isinstance(house, dict):
                    continue
                self.ensure_house(
                    str(house.get("id", "") or ""),
                    house_name=str(house.get("name", "") or ""),
                    house_status=str(house.get("status", "UNSEARCHED") or "UNSEARCHED"),
                )
        current_target_id = str(raw.get("current_target_id", "") or "")
        if current_target_id:
            self.set_current_target_house(current_target_id)
        return self.memories

    def get_house_memory(self, house_id: str, *, ensure: bool = False) -> Optional[Dict[str, Any]]:
        hid = str(house_id or "").strip()
        if not hid:
            return None
        memory = self.memories.get(hid)
        if isinstance(memory, dict):
            self._normalize_house_memory(memory)
            return memory
        if ensure:
            return self.ensure_house(hid)
        return None

    def update_working_memory(self, house_id: str, patch: Dict[str, Any], *, deep_merge: bool = True) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        working_memory = memory["working_memory"]
        if deep_merge:
            _deep_merge_dict(working_memory, patch)
        else:
            for key, value in patch.items():
                working_memory[key] = copy.deepcopy(value)
        self._normalize_working_memory(memory)
        memory["updated_at"] = _now_text()
        return memory

    def update_semantic_memory(self, house_id: str, patch: Dict[str, Any], *, deep_merge: bool = True) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        semantic_memory = memory["semantic_memory"]
        if deep_merge:
            _deep_merge_dict(semantic_memory, patch)
        else:
            for key, value in patch.items():
                semantic_memory[key] = copy.deepcopy(value)
        self._normalize_semantic_memory(memory)
        memory["updated_at"] = _now_text()
        return memory

    def append_recent_action(self, house_id: str, action_name: str) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        recent_actions = memory["working_memory"].setdefault("recent_actions", [])
        if not isinstance(recent_actions, list):
            recent_actions = []
            memory["working_memory"]["recent_actions"] = recent_actions
        recent_actions.append(str(action_name or ""))
        if len(recent_actions) > self.recent_actions_limit:
            del recent_actions[:-self.recent_actions_limit]
        memory["updated_at"] = _now_text()
        return memory

    def append_recent_target_decision(self, house_id: str, decision: Dict[str, Any]) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        recent_decisions = memory["working_memory"].setdefault("recent_target_decisions", [])
        if not isinstance(recent_decisions, list):
            recent_decisions = []
            memory["working_memory"]["recent_target_decisions"] = recent_decisions
        payload = copy.deepcopy(decision)
        payload.setdefault("timestamp", _now_text())
        recent_decisions.append(payload)
        if len(recent_decisions) > self.recent_decisions_limit:
            del recent_decisions[:-self.recent_decisions_limit]
        memory["updated_at"] = _now_text()
        return memory

    def set_top_candidates(self, house_id: str, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        trimmed = [copy.deepcopy(candidate) for candidate in list(candidates or [])[: self.top_candidates_limit]]
        memory["working_memory"]["top_candidates"] = trimmed
        if trimmed:
            best_candidate_id = str(trimmed[0].get("candidate_id", "") or "")
            memory["working_memory"]["last_best_entry_id"] = best_candidate_id
            memory["semantic_memory"]["last_best_entry_id"] = best_candidate_id
        self._normalize_working_memory(memory)
        self._normalize_semantic_memory(memory)
        memory["updated_at"] = _now_text()
        return memory

    def set_entry_search_status(self, house_id: str, status: str) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        memory["semantic_memory"]["entry_search_status"] = str(status or "not_started")
        self._normalize_semantic_memory(memory)
        memory["updated_at"] = _now_text()
        return memory

    def update_sector(
        self,
        house_id: str,
        sector_id: str,
        *,
        observed: bool = True,
        best_entry_state: Optional[str] = None,
        best_target_conditioned_subgoal: Optional[str] = None,
        best_target_match_score: Optional[float] = None,
        visit_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        searched_sectors = memory["semantic_memory"].setdefault("searched_sectors", {})
        if not isinstance(searched_sectors, dict):
            searched_sectors = {}
            memory["semantic_memory"]["searched_sectors"] = searched_sectors
        sector_key = str(sector_id or "").strip() or "unknown_sector"
        sector = searched_sectors.get(sector_key)
        if not isinstance(sector, dict):
            sector = {
                "observed": False,
                "observation_count": 0,
                "last_visit_time": None,
                "best_entry_state": "",
                "best_target_conditioned_subgoal": "",
                "best_target_match_score": 0.0,
            }
            searched_sectors[sector_key] = sector
        sector["observed"] = bool(observed)
        sector["observation_count"] = int(sector.get("observation_count", 0) or 0) + 1
        sector["last_visit_time"] = visit_time if visit_time is not None else datetime.now().timestamp()
        if best_entry_state is not None:
            sector["best_entry_state"] = str(best_entry_state or "")
        if best_target_conditioned_subgoal is not None:
            sector["best_target_conditioned_subgoal"] = str(best_target_conditioned_subgoal or "")
        if best_target_match_score is not None:
            sector["best_target_match_score"] = float(best_target_match_score)
        self._normalize_semantic_memory(memory)
        memory["updated_at"] = _now_text()
        return memory

    def upsert_candidate_entry(self, house_id: str, candidate: Dict[str, Any]) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        entries = memory["semantic_memory"].setdefault("candidate_entries", [])
        if not isinstance(entries, list):
            entries = []
            memory["semantic_memory"]["candidate_entries"] = entries
        candidate_id = str(candidate.get("candidate_id", "") or "").strip()
        if not candidate_id:
            raise ValueError("candidate_id is required for candidate entry upsert")
        payload = copy.deepcopy(candidate)
        payload["candidate_id"] = candidate_id
        existing_index = next(
            (index for index, item in enumerate(entries) if isinstance(item, dict) and str(item.get("candidate_id", "") or "") == candidate_id),
            None,
        )
        if existing_index is None:
            entries.append(payload)
        else:
            merged = entries[existing_index]
            if not isinstance(merged, dict):
                merged = {}
            _deep_merge_dict(merged, payload)
            entries[existing_index] = merged
        self._normalize_semantic_memory(memory)
        memory["updated_at"] = _now_text()
        return memory

    def append_episodic_snapshot(self, house_id: str, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        memory = self.ensure_house(house_id)
        episodic_memory = memory.setdefault("episodic_memory", [])
        if not isinstance(episodic_memory, list):
            episodic_memory = []
            memory["episodic_memory"] = episodic_memory
        payload = copy.deepcopy(snapshot)
        payload.setdefault("house_id", str(house_id))
        payload.setdefault("timestamp", datetime.now().timestamp())
        snapshot_id = str(payload.get("snapshot_id", "") or "").strip()
        if snapshot_id:
            existing_index = next(
                (index for index, item in enumerate(episodic_memory) if isinstance(item, dict) and str(item.get("snapshot_id", "") or "") == snapshot_id),
                None,
            )
            if existing_index is None:
                episodic_memory.append(payload)
            else:
                merged = episodic_memory[existing_index]
                if not isinstance(merged, dict):
                    merged = {}
                _deep_merge_dict(merged, payload)
                episodic_memory[existing_index] = merged
        else:
            episodic_memory.append(payload)
        if len(episodic_memory) > self.episodic_limit:
            del episodic_memory[:-self.episodic_limit]
        memory["updated_at"] = _now_text()
        return memory

    def _normalize_house_memory(self, memory: Dict[str, Any]) -> None:
        if not isinstance(memory.get("working_memory"), dict):
            memory["working_memory"] = _default_working_memory()
        if not isinstance(memory.get("episodic_memory"), list):
            memory["episodic_memory"] = []
        if not isinstance(memory.get("semantic_memory"), dict):
            memory["semantic_memory"] = _default_semantic_memory()
        self._normalize_working_memory(memory)
        self._normalize_semantic_memory(memory)

    def _normalize_working_memory(self, memory: Dict[str, Any]) -> None:
        working_memory = memory.get("working_memory", {})
        if not isinstance(working_memory, dict):
            working_memory = _default_working_memory()
            memory["working_memory"] = working_memory
        for key, value in _default_working_memory().items():
            working_memory.setdefault(key, copy.deepcopy(value))
        recent_actions = working_memory.get("recent_actions", [])
        working_memory["recent_actions"] = list(recent_actions)[-self.recent_actions_limit :] if isinstance(recent_actions, list) else []
        recent_decisions = working_memory.get("recent_target_decisions", [])
        working_memory["recent_target_decisions"] = (
            list(recent_decisions)[-self.recent_decisions_limit :] if isinstance(recent_decisions, list) else []
        )
        top_candidates = working_memory.get("top_candidates", [])
        working_memory["top_candidates"] = list(top_candidates)[: self.top_candidates_limit] if isinstance(top_candidates, list) else []

    def _normalize_semantic_memory(self, memory: Dict[str, Any]) -> None:
        semantic_memory = memory.get("semantic_memory", {})
        if not isinstance(semantic_memory, dict):
            semantic_memory = _default_semantic_memory()
            memory["semantic_memory"] = semantic_memory
        defaults = _default_semantic_memory()
        for key, value in defaults.items():
            semantic_memory.setdefault(key, copy.deepcopy(value))
        searched_sectors = semantic_memory.get("searched_sectors", {})
        if not isinstance(searched_sectors, dict):
            searched_sectors = {}
            semantic_memory["searched_sectors"] = searched_sectors
        for sector_id, sector_default in defaults["searched_sectors"].items():
            sector = searched_sectors.get(sector_id)
            if not isinstance(sector, dict):
                searched_sectors[sector_id] = copy.deepcopy(sector_default)
                continue
            for key, value in sector_default.items():
                sector.setdefault(key, copy.deepcopy(value))
        candidate_entries = semantic_memory.get("candidate_entries", [])
        semantic_memory["candidate_entries"] = list(candidate_entries) if isinstance(candidate_entries, list) else []
        observed_sector_count = 0
        approachable_entry_count = 0
        blocked_entry_count = 0
        rejected_entry_count = 0
        for sector in searched_sectors.values():
            if isinstance(sector, dict) and bool(sector.get("observed", False)):
                observed_sector_count += 1
        for entry in semantic_memory["candidate_entries"]:
            if not isinstance(entry, dict):
                continue
            status = str(entry.get("status", "") or "")
            if status == "approachable":
                approachable_entry_count += 1
            elif status in {"blocked_temporary", "blocked_confirmed"}:
                blocked_entry_count += 1
            elif status in {"non_target", "window_rejected"}:
                rejected_entry_count += 1
        semantic_memory["search_summary"] = {
            "observed_sector_count": observed_sector_count,
            "candidate_entry_count": len(semantic_memory["candidate_entries"]),
            "approachable_entry_count": approachable_entry_count,
            "blocked_entry_count": blocked_entry_count,
            "rejected_entry_count": rejected_entry_count,
        }


__all__ = [
    "DEFAULT_ENTRY_SEARCH_MEMORY_PATH",
    "DEFAULT_HOUSES_CONFIG_PATH",
    "DEFAULT_SECTOR_IDS",
    "EntrySearchMemoryStore",
]
