from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from .entry_search_memory import EntrySearchMemoryStore
except Exception:
    from entry_search_memory import EntrySearchMemoryStore


DEFAULT_SESSIONS_ROOT = (
    Path(__file__).resolve().parents[1]
    / "captures_remote"
    / "memory_collection_sessions"
)


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def discover_sessions(root: Path) -> List[Path]:
    if (root / "memory_fusion_captures").is_dir():
        return [root]
    return sorted([item for item in root.iterdir() if item.is_dir() and (item / "memory_fusion_captures").is_dir()])


def discover_labeling_dirs(session_dir: Path) -> List[Path]:
    captures_root = session_dir / "memory_fusion_captures"
    if not captures_root.is_dir():
        return []
    output: List[Path] = []
    for capture_dir in sorted(captures_root.iterdir(), key=lambda path: path.name):
        labeling_dir = capture_dir / "labeling"
        if labeling_dir.is_dir():
            output.append(labeling_dir)
    return output


def get_fusion_payload(labeling_dir: Path) -> Dict[str, Any]:
    payload = read_json(labeling_dir / "fusion_result.json")
    fusion = payload.get("fusion", {}) if isinstance(payload.get("fusion"), dict) else {}
    return fusion


def get_target_house_id(labeling_dir: Path, fusion: Dict[str, Any]) -> str:
    target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
    sample = read_json(labeling_dir / "sample_metadata.json")
    temporal = read_json(labeling_dir / "temporal_context.json")
    return str(
        target_context.get("target_house_id")
        or sample.get("target_house_id")
        or temporal.get("current_target_house_id")
        or ""
    ).strip()


def extract_carry(memory_root: Dict[str, Any], house_id: str) -> Dict[str, Any]:
    memories = memory_root.get("memories", {}) if isinstance(memory_root.get("memories"), dict) else {}
    house_memory = memories.get(house_id, {}) if isinstance(memories.get(house_id), dict) else {}
    semantic_memory = house_memory.get("semantic_memory", {}) if isinstance(house_memory.get("semantic_memory"), dict) else {}
    return {
        "perimeter_coverage": copy.deepcopy(semantic_memory.get("perimeter_coverage", {})),
        "search_completion_evidence": copy.deepcopy(semantic_memory.get("search_completion_evidence", {})),
        "entry_search_status": str(semantic_memory.get("entry_search_status") or ""),
    }


def inject_carry(memory_root: Dict[str, Any], house_id: str, carry: Dict[str, Any]) -> Dict[str, Any]:
    if not carry:
        return memory_root
    memories = memory_root.setdefault("memories", {})
    if not isinstance(memories, dict):
        memories = {}
        memory_root["memories"] = memories
    house_memory = memories.get(house_id)
    if not isinstance(house_memory, dict):
        return memory_root
    semantic_memory = house_memory.setdefault("semantic_memory", {})
    if not isinstance(semantic_memory, dict):
        semantic_memory = {}
        house_memory["semantic_memory"] = semantic_memory
    if isinstance(carry.get("perimeter_coverage"), dict) and carry["perimeter_coverage"]:
        semantic_memory["perimeter_coverage"] = copy.deepcopy(carry["perimeter_coverage"])
    if isinstance(carry.get("search_completion_evidence"), dict) and carry["search_completion_evidence"]:
        semantic_memory["search_completion_evidence"] = copy.deepcopy(carry["search_completion_evidence"])
    if str(carry.get("entry_search_status") or "") == "no_entry_found_after_full_coverage":
        semantic_memory["entry_search_status"] = "no_entry_found_after_full_coverage"
    return memory_root


def clear_backfilled_fields(memory_root: Dict[str, Any], house_id: str) -> Dict[str, Any]:
    memories = memory_root.get("memories", {}) if isinstance(memory_root.get("memories"), dict) else {}
    house_memory = memories.get(house_id, {}) if isinstance(memories.get(house_id), dict) else {}
    semantic_memory = house_memory.get("semantic_memory", {}) if isinstance(house_memory.get("semantic_memory"), dict) else {}
    if not semantic_memory:
        return memory_root
    semantic_memory.pop("perimeter_coverage", None)
    semantic_memory.pop("search_completion_evidence", None)
    if str(semantic_memory.get("entry_search_status") or "") == "no_entry_found_after_full_coverage":
        semantic_memory["entry_search_status"] = "searching_entry"
    registry = memory_root.get("house_registry", {}) if isinstance(memory_root.get("house_registry"), dict) else {}
    registry_entry = registry.get(house_id, {}) if isinstance(registry.get(house_id), dict) else {}
    if registry_entry:
        registry_entry.pop("visited_coverage_ratio", None)
        registry_entry.pop("observed_coverage_ratio", None)
        registry_entry.pop("no_entry_after_full_coverage", None)
        if str(registry_entry.get("entry_search_status") or "") == "no_entry_found_after_full_coverage":
            registry_entry["entry_search_status"] = "searching_entry"
        if str(registry_entry.get("search_status") or "") == "NO_ENTRY_FOUND":
            registry_entry["search_status"] = "OBSERVING"
            registry_entry["searched"] = False
    return memory_root


def capture_visit_time(labeling_dir: Path) -> Optional[float]:
    temporal = read_json(labeling_dir / "temporal_context.json")
    sample = read_json(labeling_dir / "sample_metadata.json")
    value = str(temporal.get("capture_time") or sample.get("capture_time") or "").strip()
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except Exception:
        return None


def update_memory_root(
    memory_root: Dict[str, Any],
    *,
    house_id: str,
    target_context: Dict[str, Any],
    carry: Dict[str, Any],
    visit_time: Optional[float] = None,
) -> Dict[str, Any]:
    memory_root = clear_backfilled_fields(memory_root, house_id)
    memory_root = inject_carry(memory_root, house_id, carry)
    store = EntrySearchMemoryStore()
    store.data = memory_root if isinstance(memory_root, dict) else {}
    store.ensure_house(house_id)
    store.update_perimeter_coverage(house_id, target_context, visit_time=visit_time)
    return store.to_dict()


def build_entry_search_memory_section(memory_root: Dict[str, Any], house_id: str) -> Dict[str, Any]:
    store = EntrySearchMemoryStore()
    store.data = memory_root if isinstance(memory_root, dict) else {}
    store.ensure_house(house_id)
    memory = store.get_house_memory(house_id, ensure=True) or {}
    root = store.to_dict()
    semantic_memory = memory.get("semantic_memory", {}) if isinstance(memory.get("semantic_memory"), dict) else {}
    return {
        "status": "ok",
        "memory_path": "",
        "house_id": house_id,
        "sector_id": None,
        "entry_search_status": str(semantic_memory.get("entry_search_status") or ""),
        "house_registry_entry": copy.deepcopy(root.get("house_registry", {}).get(house_id, {})),
        "planner_context": copy.deepcopy(root.get("planner_context", {})),
        "semantic_memory": copy.deepcopy(semantic_memory),
    }


def backfill_labeling_dir(labeling_dir: Path, carry_by_house: Dict[str, Dict[str, Any]], *, dry_run: bool = False) -> Dict[str, Any]:
    fusion = get_fusion_payload(labeling_dir)
    target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
    house_id = get_target_house_id(labeling_dir, fusion)
    if not house_id or not target_context:
        return {"labeling_dir": str(labeling_dir), "status": "skip", "reason": "missing_target_context"}

    carry = carry_by_house.get(house_id, {})
    visit_time = capture_visit_time(labeling_dir)
    before_path = labeling_dir / "entry_search_memory_snapshot_before.json"
    after_path = labeling_dir / "entry_search_memory_snapshot_after.json"

    if before_path.exists() and not dry_run:
        before_payload = read_json(before_path)
        before_memory = before_payload.get("memory", {}) if isinstance(before_payload.get("memory"), dict) else {}
        before_payload["memory"] = inject_carry(clear_backfilled_fields(before_memory, house_id), house_id, carry)
        write_json(before_path, before_payload)

    if not after_path.exists():
        return {"labeling_dir": str(labeling_dir), "status": "skip", "reason": "missing_after_snapshot"}
    after_payload = read_json(after_path)
    after_memory = after_payload.get("memory", {}) if isinstance(after_payload.get("memory"), dict) else {}
    updated_memory = update_memory_root(
        after_memory,
        house_id=house_id,
        target_context=target_context,
        carry=carry,
        visit_time=visit_time,
    )
    carry_by_house[house_id] = extract_carry(updated_memory, house_id)
    if not dry_run:
        after_payload["memory"] = updated_memory
        write_json(after_path, after_payload)
        fusion_result_path = labeling_dir / "fusion_result.json"
        fusion_payload = read_json(fusion_result_path)
        if isinstance(fusion_payload.get("fusion"), dict):
            fusion_payload["fusion"]["entry_search_memory"] = build_entry_search_memory_section(updated_memory, house_id)
            write_json(fusion_result_path, fusion_payload)

    evidence = carry_by_house[house_id].get("search_completion_evidence", {})
    return {
        "labeling_dir": str(labeling_dir),
        "status": "ok",
        "house_id": house_id,
        "entry_search_status": carry_by_house[house_id].get("entry_search_status", ""),
        "visited_coverage_ratio": evidence.get("visited_coverage_ratio"),
        "observed_coverage_ratio": evidence.get("observed_coverage_ratio"),
        "no_entry_after_full_coverage": evidence.get("no_entry_after_full_coverage"),
    }


def backfill_session(session_dir: Path, *, dry_run: bool = False) -> Dict[str, Any]:
    carry_by_house: Dict[str, Dict[str, Any]] = {}
    results: List[Dict[str, Any]] = []
    for labeling_dir in discover_labeling_dirs(session_dir):
        results.append(backfill_labeling_dir(labeling_dir, carry_by_house, dry_run=dry_run))

    stop_snapshot = session_dir / "entry_search_memory_snapshot_stop.json"
    if stop_snapshot.exists() and not dry_run:
        payload = read_json(stop_snapshot)
        memory_root = payload.get("memory", {}) if isinstance(payload.get("memory"), dict) else {}
        for house_id, carry in carry_by_house.items():
            memory_root = inject_carry(memory_root, house_id, carry)
        store = EntrySearchMemoryStore()
        store.data = memory_root
        store.to_dict()
        payload["memory"] = store.to_dict()
        write_json(stop_snapshot, payload)

    ok_count = sum(1 for item in results if item.get("status") == "ok")
    completed_count = sum(1 for item in results if item.get("no_entry_after_full_coverage"))
    summary = {
        "status": "ok",
        "session_dir": str(session_dir),
        "dry_run": bool(dry_run),
        "labeling_count": len(results),
        "ok_count": ok_count,
        "no_entry_completed_count": completed_count,
        "final_carry": carry_by_house,
        "results": results,
    }
    if not dry_run:
        write_json(session_dir / "memory_perimeter_backfill_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill perimeter coverage evidence into memory collection sessions.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--session_dir", type=Path, default=None)
    group.add_argument("--sessions_root", type=Path, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--summary_json", type=Path, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = args.session_dir or args.sessions_root or DEFAULT_SESSIONS_ROOT
    sessions = discover_sessions(root.resolve())
    summaries = [backfill_session(session, dry_run=bool(args.dry_run)) for session in sessions]
    summary = {
        "status": "ok",
        "dry_run": bool(args.dry_run),
        "session_count": len(summaries),
        "sessions": summaries,
    }
    if args.summary_json:
        write_json(args.summary_json, summary)
    print(f"[memory-perimeter-backfill] sessions={len(summaries)} dry_run={int(bool(args.dry_run))}")
    for item in summaries:
        print(
            "  - "
            f"{Path(item.get('session_dir', '')).name}: "
            f"labeling={item.get('labeling_count')} ok={item.get('ok_count')} "
            f"no_entry_completed={item.get('no_entry_completed_count')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
