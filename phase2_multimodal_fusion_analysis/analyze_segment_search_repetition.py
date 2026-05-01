from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from batch_segment_search_memory_replay import DEFAULT_OUTPUT_DIR, discover_session_dirs, write_json
from segment_search_memory_replay import DEFAULT_SESSIONS_ROOT


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except Exception:
        return default


def parse_segment_id(update: Dict[str, Any], event: Dict[str, Any]) -> Tuple[str, str, str, int]:
    segment_id = str(update.get("segment_id", "") or "")
    face_id = str(update.get("face_id", "") or "")
    segment_index = safe_int(update.get("segment_index"), -1)
    house_id = str(event.get("target_house_id", "") or "")
    if segment_id.startswith("H"):
        parts = segment_id.split("_")
        if parts:
            house_id = parts[0][1:] or house_id
        if len(parts) >= 2 and not face_id:
            face_id = parts[1]
        if len(parts) >= 3 and segment_index < 0:
            segment_index = safe_int(parts[2], -1)
    return house_id, face_id, segment_id, segment_index


def likely_issue(first_state: str, repeat_count: int, step_span: int, evidence_counts: Counter) -> str:
    if repeat_count <= 0:
        return "no_repeat"
    if first_state == "entry_found_open":
        return "open_entry_not_used_as_stop_or_approach_gate"
    if first_state == "searched_closed_entry":
        return "closed_entry_revisited_after_completion"
    if first_state == "searched_window_only":
        return "window_or_wall_segment_revisited"
    if first_state == "searched_no_entry":
        return "no_entry_segment_revisited"
    if evidence_counts.get("needs_review", 0) > 0:
        return "rule_conflict_needs_review"
    if step_span >= 30:
        return "long_span_repeat_possible_memory_prompt_or_control_issue"
    return "completed_segment_revisited"


def analyze_report(report_path: Path, *, min_repeat_count: int = 1) -> Dict[str, Any]:
    report = read_json(report_path)
    session_dir = Path(str(report.get("session_dir", report_path.parent)))
    session_name = session_dir.name
    grouped: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    timeline = report.get("timeline", []) if isinstance(report.get("timeline"), list) else []

    for event_index, event in enumerate(timeline):
        if not isinstance(event, dict):
            continue
        capture_name = str(event.get("capture_name", "") or "")
        step_index = safe_int(event.get("step_index"), event_index)
        pose = event.get("pose", {}) if isinstance(event.get("pose"), dict) else {}
        yolo_counts = event.get("yolo_class_counts", {}) if isinstance(event.get("yolo_class_counts"), dict) else {}
        for update in event.get("updates", []) or []:
            if not isinstance(update, dict):
                continue
            house_id, face_id, segment_id, segment_index = parse_segment_id(update, event)
            key = (house_id, face_id, segment_index)
            item = grouped.setdefault(
                key,
                {
                    "session_name": session_name,
                    "session_dir": str(session_dir),
                    "report_path": str(report_path),
                    "house_id": house_id,
                    "face_id": face_id,
                    "segment_index": segment_index,
                    "segment_id": segment_id,
                    "first_complete_capture": "",
                    "first_complete_step": None,
                    "first_complete_state": "",
                    "first_complete_evidence": "",
                    "first_complete_pose": {},
                    "final_state": "",
                    "last_capture": "",
                    "last_step": None,
                    "last_pose": {},
                    "observation_count": 0,
                    "repeat_count": 0,
                    "repeat_captures": [],
                    "repeat_steps": [],
                    "requested_state_counts": Counter(),
                    "new_state_counts": Counter(),
                    "evidence_type_counts": Counter(),
                    "yolo_class_counts": Counter(),
                },
            )
            requested_state = str(update.get("requested_state", "") or "")
            new_state = str(update.get("new_state", "") or "")
            evidence_type = str(update.get("evidence_type", "") or "")
            item["observation_count"] += 1
            item["last_capture"] = capture_name
            item["last_step"] = step_index
            item["last_pose"] = pose
            item["final_state"] = new_state
            if requested_state:
                item["requested_state_counts"][requested_state] += 1
            if new_state:
                item["new_state_counts"][new_state] += 1
            if evidence_type:
                item["evidence_type_counts"][evidence_type] += 1
            for class_name, count in yolo_counts.items():
                item["yolo_class_counts"][str(class_name)] += safe_int(count, 0)
            if bool(update.get("is_search_complete", False)) and not item["first_complete_capture"]:
                item["first_complete_capture"] = capture_name
                item["first_complete_step"] = step_index
                item["first_complete_state"] = new_state
                item["first_complete_evidence"] = evidence_type
                item["first_complete_pose"] = pose
            if bool(update.get("repeat_completed_observation", False)):
                item["repeat_count"] += 1
                item["repeat_captures"].append(capture_name)
                item["repeat_steps"].append(step_index)

    repeated_segments: List[Dict[str, Any]] = []
    all_segments: List[Dict[str, Any]] = []
    for item in grouped.values():
        first_step = item.get("first_complete_step")
        last_step = item.get("last_step")
        step_span = 0
        if first_step is not None and last_step is not None:
            step_span = max(0, safe_int(last_step) - safe_int(first_step))
        row = {
            **{key: value for key, value in item.items() if not isinstance(value, Counter)},
            "step_span_after_first_complete": step_span,
            "requested_state_counts": dict(item["requested_state_counts"]),
            "new_state_counts": dict(item["new_state_counts"]),
            "evidence_type_counts": dict(item["evidence_type_counts"]),
            "yolo_class_counts": dict(item["yolo_class_counts"]),
            "repeat_capture_preview": item["repeat_captures"][:8],
            "repeat_step_preview": item["repeat_steps"][:12],
        }
        row["likely_issue"] = likely_issue(
            str(row.get("first_complete_state", "") or ""),
            safe_int(row.get("repeat_count"), 0),
            step_span,
            item["evidence_type_counts"],
        )
        all_segments.append(row)
        if safe_int(row.get("repeat_count"), 0) >= min_repeat_count:
            repeated_segments.append(row)

    repeated_segments = sorted(
        repeated_segments,
        key=lambda row: (
            safe_int(row.get("repeat_count"), 0),
            safe_int(row.get("step_span_after_first_complete"), 0),
        ),
        reverse=True,
    )
    all_segments = sorted(
        all_segments,
        key=lambda row: (
            safe_int(row.get("repeat_count"), 0),
            str(row.get("session_name", "")),
            str(row.get("house_id", "")),
            str(row.get("face_id", "")),
            safe_int(row.get("segment_index"), 0),
        ),
        reverse=True,
    )
    issue_counts = Counter(str(row.get("likely_issue", "")) for row in repeated_segments)
    return {
        "session_name": session_name,
        "session_dir": str(session_dir),
        "report_path": str(report_path),
        "capture_count": safe_int(report.get("capture_count"), 0),
        "segment_count": len(all_segments),
        "repeated_segment_count": len(repeated_segments),
        "repeat_observation_count": sum(safe_int(row.get("repeat_count"), 0) for row in repeated_segments),
        "issue_counts": dict(issue_counts),
        "repeated_segments": repeated_segments,
        "all_observed_segments": all_segments,
    }


def discover_report_paths(sessions_root: Path, pattern: str) -> List[Path]:
    paths: List[Path] = []
    for session_dir in discover_session_dirs(sessions_root, pattern=pattern):
        report_path = session_dir / "segment_search_memory_report.json"
        if report_path.is_file():
            paths.append(report_path)
    return paths


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_repetition_diagnostics(
    *,
    sessions_root: Path,
    output_dir: Path,
    pattern: str,
    min_repeat_count: int,
    top_k: int,
) -> Dict[str, Any]:
    report_paths = discover_report_paths(sessions_root, pattern)
    session_reports: List[Dict[str, Any]] = []
    repeated_rows: List[Dict[str, Any]] = []
    issue_counts: Counter = Counter()
    for report_path in report_paths:
        session_report = analyze_report(report_path, min_repeat_count=min_repeat_count)
        session_reports.append(
            {
                key: value
                for key, value in session_report.items()
                if key not in {"repeated_segments", "all_observed_segments"}
            }
        )
        for row in session_report.get("repeated_segments", []):
            repeated_rows.append(row)
            issue_counts[str(row.get("likely_issue", ""))] += 1

    repeated_rows = sorted(
        repeated_rows,
        key=lambda row: (
            safe_int(row.get("repeat_count"), 0),
            safe_int(row.get("step_span_after_first_complete"), 0),
        ),
        reverse=True,
    )
    session_reports = sorted(
        session_reports,
        key=lambda row: safe_int(row.get("repeat_observation_count"), 0),
        reverse=True,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir.resolve()
    json_path = output_dir / f"segment_search_repetition_diagnostics_{run_id}.json"
    repeated_csv_path = output_dir / f"segment_search_repetition_segments_{run_id}.csv"
    sessions_csv_path = output_dir / f"segment_search_repetition_sessions_{run_id}.csv"

    payload = {
        "version": "segment_search_repetition_diagnostics_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sessions_root": str(sessions_root.resolve()),
        "pattern": pattern,
        "min_repeat_count": min_repeat_count,
        "report_count": len(report_paths),
        "aggregate": {
            "repeated_segment_count": len(repeated_rows),
            "repeat_observation_count": sum(safe_int(row.get("repeat_count"), 0) for row in repeated_rows),
            "issue_counts": dict(issue_counts),
        },
        "top_repeated_segments": repeated_rows[:top_k],
        "sessions": session_reports,
        "outputs": {
            "diagnostics_json": str(json_path),
            "repeated_segments_csv": str(repeated_csv_path),
            "sessions_csv": str(sessions_csv_path),
        },
    }

    write_json(json_path, payload)
    write_csv(
        repeated_csv_path,
        repeated_rows,
        [
            "session_name",
            "house_id",
            "face_id",
            "segment_index",
            "segment_id",
            "repeat_count",
            "step_span_after_first_complete",
            "first_complete_step",
            "first_complete_capture",
            "first_complete_state",
            "first_complete_evidence",
            "final_state",
            "last_step",
            "last_capture",
            "likely_issue",
            "repeat_capture_preview",
            "repeat_step_preview",
        ],
    )
    write_csv(
        sessions_csv_path,
        session_reports,
        [
            "session_name",
            "capture_count",
            "segment_count",
            "repeated_segment_count",
            "repeat_observation_count",
            "issue_counts",
            "report_path",
        ],
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze deterministic segment search memory reports and locate repeated completed segments."
    )
    parser.add_argument("--sessions_root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pattern", default="memory_episode_*")
    parser.add_argument("--min_repeat_count", type=int, default=1)
    parser.add_argument("--top_k", type=int, default=50)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_repetition_diagnostics(
        sessions_root=Path(args.sessions_root),
        output_dir=Path(args.output_dir),
        pattern=str(args.pattern or "memory_episode_*"),
        min_repeat_count=max(1, int(args.min_repeat_count)),
        top_k=max(1, int(args.top_k)),
    )
    aggregate = report.get("aggregate", {})
    print(
        "[segment-repeat] done "
        f"reports={report.get('report_count', 0)} "
        f"segments={aggregate.get('repeated_segment_count', 0)} "
        f"repeat_obs={aggregate.get('repeat_observation_count', 0)}"
    )
    outputs = report.get("outputs", {})
    print(f"[segment-repeat] diagnostics -> {outputs.get('diagnostics_json')}")
    print(f"[segment-repeat] segments csv -> {outputs.get('repeated_segments_csv')}")
    print(f"[segment-repeat] sessions csv -> {outputs.get('sessions_csv')}")


if __name__ == "__main__":
    main()
