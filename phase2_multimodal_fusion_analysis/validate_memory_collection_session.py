from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_SESSIONS_ROOT = (
    Path(__file__).resolve().parents[1]
    / "captures_remote"
    / "memory_collection_sessions"
)

REQUIRED_LABELING_FILES = [
    "sample_metadata.json",
    "temporal_context.json",
    "state.json",
    "yolo_result.json",
    "depth_result.json",
    "fusion_result.json",
    "fusion_overlay.png",
    "entry_search_memory_snapshot_before.json",
    "entry_search_memory_snapshot_after.json",
]

ESSENTIAL_JSON_FILES = [
    "sample_metadata.json",
    "fusion_result.json",
    "entry_search_memory_snapshot_after.json",
]

DOORLIKE_CLASSES = {
    "door",
    "open door",
    "open_door",
    "close door",
    "close_door",
    "closed door",
    "closed_door",
}

WINDOW_CLASSES = {"window"}


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }


def read_json(path: Path, issues: Optional[List[Issue]] = None) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        if issues is not None:
            issues.append(
                Issue(
                    "FAIL",
                    "json_read_error",
                    f"Failed to read JSON: {exc}",
                    str(path),
                )
            )
        return {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def as_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def normalize_class_name(value: Any) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def compact_counter(counter: Counter) -> Dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counter.items()) if key}


def status_from_issues(issues: Iterable[Issue]) -> str:
    severities = {issue.severity for issue in issues}
    if "FAIL" in severities:
        return "FAIL"
    if "WARN" in severities:
        return "WARN"
    return "PASS"


def get_memory_for_house(memory_root: Dict[str, Any], house_id: str) -> Dict[str, Any]:
    memories = memory_root.get("memories", {}) if isinstance(memory_root.get("memories"), dict) else {}
    memory = memories.get(str(house_id or ""))
    return memory if isinstance(memory, dict) else {}


def get_registry_for_house(memory_root: Dict[str, Any], house_id: str) -> Dict[str, Any]:
    registry = memory_root.get("house_registry", {}) if isinstance(memory_root.get("house_registry"), dict) else {}
    entry = registry.get(str(house_id or ""))
    return entry if isinstance(entry, dict) else {}


def candidate_entries_for_house(memory_root: Dict[str, Any], house_id: str) -> List[Dict[str, Any]]:
    house_memory = get_memory_for_house(memory_root, house_id)
    semantic_memory = house_memory.get("semantic_memory", {}) if isinstance(house_memory.get("semantic_memory"), dict) else {}
    entries = semantic_memory.get("candidate_entries", [])
    return [entry for entry in as_list(entries) if isinstance(entry, dict)]


def best_entries(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [entry for entry in entries if truthy(entry.get("is_best_candidate", False))]


def choose_best_entry(entries: List[Dict[str, Any]], registry: Dict[str, Any]) -> Dict[str, Any]:
    best_entry_id = str(registry.get("best_entry_id") or "").strip()
    if best_entry_id:
        for entry in entries:
            entry_id = str(entry.get("entry_id") or entry.get("candidate_id") or "").strip()
            if entry_id == best_entry_id:
                return entry
    marked = best_entries(entries)
    return marked[0] if marked else {}


def summarize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    evidence = entry.get("association_evidence", {}) if isinstance(entry.get("association_evidence"), dict) else {}
    source_frames = as_list(entry.get("source_frames"))
    bbox_history = as_list(entry.get("bbox_history"))
    return {
        "entry_id": str(entry.get("entry_id") or entry.get("candidate_id") or ""),
        "entry_type": str(entry.get("entry_type") or entry.get("semantic_class") or entry.get("class_name") or ""),
        "status": str(entry.get("status") or entry.get("entry_state") or ""),
        "observation_count": safe_int(entry.get("observation_count")),
        "source_frame_count": len(source_frames),
        "bbox_history_count": len(bbox_history),
        "target_match_score": round(safe_float(entry.get("target_match_score")), 4),
        "association_confidence": round(safe_float(entry.get("association_confidence")), 4),
        "association_evidence": {
            "distance_score": round(safe_float(evidence.get("distance_score")), 4),
            "view_consistency_score": round(safe_float(evidence.get("view_consistency_score")), 4),
            "appearance_score": round(safe_float(evidence.get("appearance_score")), 4),
            "language_score": round(safe_float(evidence.get("language_score")), 4),
            "geometry_score": round(safe_float(evidence.get("geometry_score")), 4),
            "memory_similarity_score": round(safe_float(evidence.get("memory_similarity_score")), 4),
        },
        "is_best_candidate": truthy(entry.get("is_best_candidate", False)),
        "is_searched": truthy(entry.get("is_searched", False)),
        "is_entered": truthy(entry.get("is_entered", False)),
    }


def association_evidence_nonzero(entry: Dict[str, Any]) -> bool:
    evidence = entry.get("association_evidence", {}) if isinstance(entry.get("association_evidence"), dict) else {}
    for key in ("distance_score", "view_consistency_score", "appearance_score", "geometry_score"):
        if safe_float(evidence.get(key), 0.0) > 0.0:
            return True
    return False


def infer_target_house_id(stop_snapshot: Dict[str, Any], capture_summaries: List[Dict[str, Any]]) -> str:
    house_mission = stop_snapshot.get("house_mission", {}) if isinstance(stop_snapshot.get("house_mission"), dict) else {}
    memory = stop_snapshot.get("memory", {}) if isinstance(stop_snapshot.get("memory"), dict) else {}
    planner_context = memory.get("planner_context", {}) if isinstance(memory.get("planner_context"), dict) else {}
    candidates = [
        house_mission.get("target_house_id"),
        memory.get("current_target_house_id"),
        planner_context.get("target_house_id"),
    ]
    for summary in reversed(capture_summaries):
        candidates.append(summary.get("target_house_id"))
    for value in candidates:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def validate_capture(capture_dir: Path) -> Dict[str, Any]:
    labeling_dir = capture_dir / "labeling"
    missing_files = [
        name
        for name in REQUIRED_LABELING_FILES
        if not (labeling_dir / name).exists()
    ]
    essential_missing = [
        name
        for name in ESSENTIAL_JSON_FILES
        if not (labeling_dir / name).exists()
    ]
    local_issues: List[Issue] = []
    sample = read_json(labeling_dir / "sample_metadata.json", local_issues) if (labeling_dir / "sample_metadata.json").exists() else {}
    fusion_payload = read_json(labeling_dir / "fusion_result.json", local_issues) if (labeling_dir / "fusion_result.json").exists() else {}
    after_snapshot = (
        read_json(labeling_dir / "entry_search_memory_snapshot_after.json", local_issues)
        if (labeling_dir / "entry_search_memory_snapshot_after.json").exists()
        else {}
    )
    fusion = fusion_payload.get("fusion", {}) if isinstance(fusion_payload.get("fusion"), dict) else {}
    memory_root = after_snapshot.get("memory", {}) if isinstance(after_snapshot.get("memory"), dict) else {}
    target_context = fusion.get("target_context", {}) if isinstance(fusion.get("target_context"), dict) else {}
    target_house_id = str(sample.get("target_house_id") or target_context.get("target_house_id") or "").strip()
    registry = get_registry_for_house(memory_root, target_house_id) if target_house_id else {}
    entries = candidate_entries_for_house(memory_root, target_house_id) if target_house_id else []
    best_count = len(best_entries(entries))
    best_entry = choose_best_entry(entries, registry)
    return {
        "capture_dir": str(capture_dir),
        "name": capture_dir.name,
        "labeling_dir": str(labeling_dir),
        "step_index": safe_int(sample.get("memory_step_index", sample.get("step_index"))),
        "capture_source": str(sample.get("capture_source") or ""),
        "target_house_id": target_house_id,
        "current_house_id": str(sample.get("current_house_id") or target_context.get("current_house_id") or ""),
        "target_conditioned_state": str(fusion.get("target_conditioned_state") or ""),
        "target_conditioned_subgoal": str(fusion.get("target_conditioned_subgoal") or ""),
        "target_conditioned_action_hint": str(fusion.get("target_conditioned_action_hint") or ""),
        "search_status": str(registry.get("search_status") or ""),
        "entry_search_status": str(registry.get("entry_search_status") or ""),
        "candidate_count": len(entries),
        "best_true_count": best_count,
        "best_entry": summarize_entry(best_entry) if best_entry else {},
        "missing_files": missing_files,
        "essential_missing": essential_missing,
        "json_read_errors": [issue.to_dict() for issue in local_issues],
        "valid_for_training": not essential_missing and not local_issues,
    }


def validate_session(
    session_dir: Path,
    *,
    min_observation_count: int = 2,
    min_source_frames: int = 2,
    min_bbox_history: int = 2,
) -> Dict[str, Any]:
    session_dir = session_dir.resolve()
    issues: List[Issue] = []
    if not session_dir.exists():
        issues.append(Issue("FAIL", "session_missing", "Session directory does not exist.", str(session_dir)))
        return {
            "status": "FAIL",
            "session_dir": str(session_dir),
            "issues": [issue.to_dict() for issue in issues],
        }

    start_snapshot = session_dir / "entry_search_memory_snapshot_start.json"
    stop_snapshot = session_dir / "entry_search_memory_snapshot_stop.json"
    if not start_snapshot.exists():
        issues.append(Issue("WARN", "missing_start_snapshot", "Missing episode start snapshot.", str(start_snapshot)))
    if not stop_snapshot.exists():
        issues.append(Issue("WARN", "missing_stop_snapshot", "Missing episode stop snapshot.", str(stop_snapshot)))

    captures_root = session_dir / "memory_fusion_captures"
    if not captures_root.exists():
        issues.append(Issue("FAIL", "captures_root_missing", "memory_fusion_captures directory is missing.", str(captures_root)))
        capture_dirs: List[Path] = []
    else:
        capture_dirs = sorted([path for path in captures_root.iterdir() if path.is_dir()], key=lambda p: p.name)
    if not capture_dirs:
        issues.append(Issue("FAIL", "no_captures", "No memory_capture_* directories found.", str(captures_root)))

    capture_summaries = [validate_capture(path) for path in capture_dirs]
    valid_capture_count = sum(1 for item in capture_summaries if item.get("valid_for_training"))
    if capture_dirs and valid_capture_count == 0:
        issues.append(Issue("FAIL", "no_valid_captures", "No capture has the essential JSON files.", str(captures_root)))

    for capture in capture_summaries:
        if capture.get("missing_files"):
            issues.append(
                Issue(
                    "WARN",
                    "capture_missing_files",
                    f"{capture['name']} missing files: {', '.join(capture['missing_files'])}",
                    capture["labeling_dir"],
                )
            )
        for item in capture.get("json_read_errors", []):
            issues.append(Issue(item["severity"], item["code"], item["message"], item.get("path", "")))

    stop_data = read_json(stop_snapshot, issues) if stop_snapshot.exists() else {}
    fallback_after = {}
    if not stop_data and capture_summaries:
        last_after = Path(capture_summaries[-1]["labeling_dir"]) / "entry_search_memory_snapshot_after.json"
        if last_after.exists():
            fallback_after = read_json(last_after, issues)

    final_snapshot = stop_data or fallback_after
    memory_root = final_snapshot.get("memory", {}) if isinstance(final_snapshot.get("memory"), dict) else {}
    target_house_id = infer_target_house_id(final_snapshot, capture_summaries)
    if not target_house_id:
        issues.append(Issue("FAIL", "target_house_missing", "Could not infer target_house_id.", str(session_dir)))

    registry = get_registry_for_house(memory_root, target_house_id) if target_house_id else {}
    entries = candidate_entries_for_house(memory_root, target_house_id) if target_house_id else []
    marked_best = best_entries(entries)
    best_count = len(marked_best)
    if entries and best_count != 1:
        issues.append(
            Issue(
                "FAIL",
                "best_entry_not_unique",
                f"Expected exactly one best entry, found {best_count}.",
                str(session_dir),
            )
        )
    if target_house_id and not get_memory_for_house(memory_root, target_house_id):
        issues.append(
            Issue(
                "FAIL",
                "target_memory_missing",
                f"Target house memory missing for house_id={target_house_id}.",
                str(session_dir),
            )
        )
    if target_house_id and not registry:
        issues.append(
            Issue(
                "WARN",
                "target_registry_missing",
                f"Target house registry entry missing for house_id={target_house_id}.",
                str(session_dir),
            )
        )
    if target_house_id and not entries:
        issues.append(
            Issue(
                "WARN",
                "no_candidate_entries",
                f"No candidate entries found for target house {target_house_id}.",
                str(session_dir),
            )
        )

    best_entry = choose_best_entry(entries, registry)
    best_summary = summarize_entry(best_entry) if best_entry else {}
    if best_entry:
        best_type = normalize_class_name(best_summary.get("entry_type"))
        planner_context = memory_root.get("planner_context", {}) if isinstance(memory_root.get("planner_context"), dict) else {}
        decision_hint = str(registry.get("decision_hint") or planner_context.get("decision_hint") or "")
        should_be_door = str(registry.get("entry_search_status") or "") == "entry_found" or decision_hint == "approach_target_entry"
        if best_type not in DOORLIKE_CLASSES:
            severity = "FAIL" if should_be_door else "WARN"
            issues.append(
                Issue(
                    severity,
                    "best_entry_not_doorlike",
                    f"Best entry type is {best_summary.get('entry_type')!r}, expected door-like.",
                    str(session_dir),
                )
            )
        if best_summary["observation_count"] < int(min_observation_count):
            issues.append(
                Issue(
                    "WARN",
                    "best_entry_low_observation_count",
                    f"Best entry observation_count={best_summary['observation_count']} < {min_observation_count}.",
                    str(session_dir),
                )
            )
        if best_summary["source_frame_count"] < int(min_source_frames):
            issues.append(
                Issue(
                    "WARN",
                    "best_entry_low_source_frames",
                    f"Best entry source_frame_count={best_summary['source_frame_count']} < {min_source_frames}.",
                    str(session_dir),
                )
            )
        if best_summary["bbox_history_count"] < int(min_bbox_history):
            issues.append(
                Issue(
                    "WARN",
                    "best_entry_low_bbox_history",
                    f"Best entry bbox_history_count={best_summary['bbox_history_count']} < {min_bbox_history}.",
                    str(session_dir),
                )
            )
        if not association_evidence_nonzero(best_entry):
            issues.append(
                Issue(
                    "WARN",
                    "best_entry_zero_association_evidence",
                    "Best entry association evidence is all zero.",
                    str(session_dir),
                )
            )

    target_ids = Counter(str(item.get("target_house_id") or "") for item in capture_summaries)
    nonempty_target_ids = [key for key in target_ids if key]
    if len(nonempty_target_ids) > 1:
        issues.append(
            Issue(
                "WARN",
                "multiple_target_house_ids",
                f"Multiple target house ids found in captures: {', '.join(nonempty_target_ids)}.",
                str(session_dir),
            )
        )

    capture_sources = Counter(str(item.get("capture_source") or "") for item in capture_summaries)
    states = Counter(str(item.get("target_conditioned_state") or "") for item in capture_summaries)
    subgoals = Counter(str(item.get("target_conditioned_subgoal") or "") for item in capture_summaries)
    actions = Counter(str(item.get("target_conditioned_action_hint") or "") for item in capture_summaries)
    missing_file_capture_count = sum(1 for item in capture_summaries if item.get("missing_files"))
    summary = {
        "status": status_from_issues(issues),
        "session_dir": str(session_dir),
        "episode_id": str(final_snapshot.get("episode_id") or session_dir.name),
        "episode_label": str(final_snapshot.get("episode_label") or ""),
        "final_step_index": safe_int(final_snapshot.get("step_index")),
        "has_start_snapshot": start_snapshot.exists(),
        "has_stop_snapshot": stop_snapshot.exists(),
        "capture_count": len(capture_summaries),
        "valid_capture_count": valid_capture_count,
        "missing_file_capture_count": missing_file_capture_count,
        "target_house_id": target_house_id,
        "registry_entry": registry,
        "planner_context": memory_root.get("planner_context", {}) if isinstance(memory_root.get("planner_context"), dict) else {},
        "candidate_entry_count": len(entries),
        "best_true_count": best_count,
        "best_entry": best_summary,
        "capture_source_distribution": compact_counter(capture_sources),
        "target_state_distribution": compact_counter(states),
        "target_subgoal_distribution": compact_counter(subgoals),
        "target_action_distribution": compact_counter(actions),
        "issues": [issue.to_dict() for issue in issues],
        "captures": capture_summaries,
    }
    return summary


def find_sessions(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([path for path in root.iterdir() if path.is_dir() and path.name.startswith("memory_episode_")])


def aggregate_batch(session_results: List[Dict[str, Any]], sessions_root: Path) -> Dict[str, Any]:
    status_counts = Counter(str(item.get("status") or "UNKNOWN") for item in session_results)
    target_counts = Counter(str(item.get("target_house_id") or "") for item in session_results)
    total_captures = sum(safe_int(item.get("capture_count")) for item in session_results)
    total_valid_captures = sum(safe_int(item.get("valid_capture_count")) for item in session_results)
    issue_counts: Counter = Counter()
    for item in session_results:
        for issue in item.get("issues", []):
            if isinstance(issue, dict):
                issue_counts[str(issue.get("code") or "")] += 1
    return {
        "status": "FAIL" if status_counts.get("FAIL", 0) else ("WARN" if status_counts.get("WARN", 0) else "PASS"),
        "sessions_root": str(sessions_root.resolve()),
        "session_count": len(session_results),
        "status_counts": compact_counter(status_counts),
        "target_house_distribution": compact_counter(target_counts),
        "total_captures": total_captures,
        "total_valid_captures": total_valid_captures,
        "issue_counts": compact_counter(issue_counts),
        "sessions": session_results,
    }


def print_session_summary(summary: Dict[str, Any]) -> None:
    print(f"[memory-validator] session={summary.get('session_dir')}")
    print(
        "[memory-validator] "
        f"status={summary.get('status')} "
        f"captures={summary.get('capture_count')} "
        f"valid_captures={summary.get('valid_capture_count')} "
        f"target={summary.get('target_house_id') or 'n/a'}"
    )
    print(
        "[memory-validator] "
        f"start={int(bool(summary.get('has_start_snapshot')))} "
        f"stop={int(bool(summary.get('has_stop_snapshot')))} "
        f"final_step={summary.get('final_step_index')}"
    )
    registry = summary.get("registry_entry", {}) if isinstance(summary.get("registry_entry"), dict) else {}
    print(
        "[memory-validator] "
        f"search_status={registry.get('search_status', 'n/a')} "
        f"entry_status={registry.get('entry_search_status', 'n/a')} "
        f"best_entry_id={registry.get('best_entry_id', 'n/a')}"
    )
    best = summary.get("best_entry", {}) if isinstance(summary.get("best_entry"), dict) else {}
    if best:
        evidence = best.get("association_evidence", {}) if isinstance(best.get("association_evidence"), dict) else {}
        print(
            "[memory-validator] "
            f"best={best.get('entry_id')}:{best.get('entry_type')} "
            f"obs={best.get('observation_count')} "
            f"frames={best.get('source_frame_count')} "
            f"bbox={best.get('bbox_history_count')} "
            f"assoc={best.get('association_confidence')} "
            f"dist={evidence.get('distance_score')} "
            f"view={evidence.get('view_consistency_score')} "
            f"app={evidence.get('appearance_score')} "
            f"geom={evidence.get('geometry_score')}"
        )
    issues = summary.get("issues", [])
    if issues:
        print("[memory-validator] issues:")
        for issue in issues:
            print(
                f"  - {issue.get('severity')} {issue.get('code')}: "
                f"{issue.get('message')} {issue.get('path') or ''}"
            )


def print_batch_summary(summary: Dict[str, Any]) -> None:
    print(f"[memory-validator] sessions_root={summary.get('sessions_root')}")
    print(
        "[memory-validator] "
        f"status={summary.get('status')} "
        f"sessions={summary.get('session_count')} "
        f"captures={summary.get('total_captures')} "
        f"valid_captures={summary.get('total_valid_captures')}"
    )
    print(f"[memory-validator] status_counts={summary.get('status_counts')}")
    if summary.get("issue_counts"):
        print(f"[memory-validator] issue_counts={summary.get('issue_counts')}")
    for item in summary.get("sessions", []):
        print(
            "  - "
            f"{item.get('status')} "
            f"{Path(str(item.get('session_dir'))).name} "
            f"captures={item.get('capture_count')} "
            f"target={item.get('target_house_id') or 'n/a'} "
            f"best={item.get('best_entry', {}).get('entry_id', 'n/a') if isinstance(item.get('best_entry'), dict) else 'n/a'}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate memory collection session outputs.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--session_dir", type=Path, help="Single memory_episode_* directory to validate.")
    group.add_argument("--sessions_root", type=Path, help="Directory containing memory_episode_* sessions.")
    parser.add_argument("--output_json", type=Path, default=None, help="Optional path to write the validation summary JSON.")
    parser.add_argument("--min_observation_count", type=int, default=2)
    parser.add_argument("--min_source_frames", type=int, default=2)
    parser.add_argument("--min_bbox_history", type=int, default=2)
    parser.add_argument("--quiet", action="store_true", help="Suppress text summary; still writes --output_json if provided.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.session_dir:
        summary = validate_session(
            args.session_dir,
            min_observation_count=args.min_observation_count,
            min_source_frames=args.min_source_frames,
            min_bbox_history=args.min_bbox_history,
        )
        if args.output_json:
            write_json(args.output_json, summary)
        if not args.quiet:
            print_session_summary(summary)
        return 1 if summary.get("status") == "FAIL" else 0

    sessions_root = args.sessions_root or DEFAULT_SESSIONS_ROOT
    sessions = find_sessions(sessions_root)
    results = [
        validate_session(
            session,
            min_observation_count=args.min_observation_count,
            min_source_frames=args.min_source_frames,
            min_bbox_history=args.min_bbox_history,
        )
        for session in sessions
    ]
    summary = aggregate_batch(results, sessions_root)
    if args.output_json:
        write_json(args.output_json, summary)
    if not args.quiet:
        print_batch_summary(summary)
    return 1 if summary.get("status") == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
