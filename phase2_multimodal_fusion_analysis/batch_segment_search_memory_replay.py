from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from segment_search_memory_replay import (
    DEFAULT_HOUSES_CONFIG,
    DEFAULT_SEGMENT_COUNT_PER_FACE,
    DEFAULT_SESSIONS_ROOT,
    COMPLETED_STATES,
    discover_labeling_dirs,
    replay_session,
)


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results" / "segment_search_memory_batch"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def discover_session_dirs(root: Path, pattern: str = "memory_episode_*") -> List[Path]:
    if not root.exists():
        return []
    return sorted(
        [path for path in root.glob(pattern) if path.is_dir()],
        key=lambda path: path.name,
    )


def flatten_house_state_counts(report: Dict[str, Any]) -> Counter:
    counts: Counter = Counter()
    memories = report.get("segment_memories", {})
    if not isinstance(memories, dict):
        return counts
    for memory in memories.values():
        summary = memory.get("summary", {}) if isinstance(memory, dict) else {}
        state_counts = summary.get("state_counts", {}) if isinstance(summary, dict) else {}
        if not isinstance(state_counts, dict):
            continue
        for state, value in state_counts.items():
            try:
                counts[str(state)] += int(value)
            except Exception:
                counts[str(state)] += 0
    return counts


def flatten_house_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    memories = report.get("segment_memories", {})
    if not isinstance(memories, dict):
        return rows
    session_name = Path(str(report.get("session_dir", ""))).name
    for house_id, memory in sorted(memories.items(), key=lambda item: str(item[0])):
        if not isinstance(memory, dict):
            continue
        summary = memory.get("summary", {}) if isinstance(memory.get("summary"), dict) else {}
        state_counts = summary.get("state_counts", {}) if isinstance(summary.get("state_counts"), dict) else {}
        rows.append(
            {
                "session_name": session_name,
                "house_id": str(house_id),
                "total_segments": int(summary.get("total_segments", 0) or 0),
                "complete_segment_count": int(summary.get("complete_segment_count", 0) or 0),
                "completion_ratio": float(summary.get("completion_ratio", 0.0) or 0.0),
                "entry_found_segments": len(summary.get("entry_found_segments", []) or []),
                "needs_review": int(state_counts.get("needs_review", 0) or 0),
                "blocked_or_occluded": int(state_counts.get("blocked_or_occluded", 0) or 0),
                "searched_no_entry": int(state_counts.get("searched_no_entry", 0) or 0),
                "searched_closed_entry": int(state_counts.get("searched_closed_entry", 0) or 0),
                "searched_window_only": int(state_counts.get("searched_window_only", 0) or 0),
                "entry_found_open": int(state_counts.get("entry_found_open", 0) or 0),
                "unseen": int(state_counts.get("unseen", 0) or 0),
            }
        )
    return rows


def count_timeline_updates(report: Dict[str, Any]) -> Counter:
    counts: Counter = Counter()
    timeline = report.get("timeline", [])
    if not isinstance(timeline, list):
        return counts
    for event in timeline:
        if not isinstance(event, dict):
            continue
        if event.get("skip_reason"):
            counts[f"skip:{event.get('skip_reason')}"] += 1
        for update in event.get("updates", []) or []:
            if not isinstance(update, dict):
                continue
            state = str(update.get("new_state", "") or "")
            if state:
                counts[f"update:{state}"] += 1
            if update.get("repeat_completed_observation"):
                counts["repeat_completed_observation"] += 1
    return counts


def build_session_summary(report: Dict[str, Any], session_dir: Path) -> Dict[str, Any]:
    summary = report.get("summary", {}) if isinstance(report.get("summary"), dict) else {}
    state_counts = flatten_house_state_counts(report)
    timeline_counts = count_timeline_updates(report)
    house_rows = flatten_house_rows(report)
    total_segments = sum(int(row.get("total_segments", 0) or 0) for row in house_rows)
    complete_segments = sum(int(row.get("complete_segment_count", 0) or 0) for row in house_rows)
    completion_ratio = float(complete_segments / total_segments) if total_segments else 0.0
    return {
        "session_name": session_dir.name,
        "session_dir": str(session_dir),
        "report_path": str(report.get("output_path", "")),
        "capture_count": int(report.get("capture_count", 0) or 0),
        "processed_event_count": int(report.get("processed_event_count", 0) or 0),
        "house_count": int(summary.get("house_count", 0) or 0),
        "total_segments": total_segments,
        "complete_segment_count": complete_segments,
        "completion_ratio": completion_ratio,
        "repeated_completed_segment_observations": int(summary.get("repeated_completed_segment_observations", 0) or 0),
        "skipped_capture_count": int(summary.get("skipped_capture_count", 0) or 0),
        "skip_reasons": summary.get("skip_reasons", {}),
        "state_counts": dict(state_counts),
        "timeline_update_counts": dict(timeline_counts),
        "needs_review_count": int(state_counts.get("needs_review", 0) or 0),
        "entry_found_open_count": int(state_counts.get("entry_found_open", 0) or 0),
        "completed_state_count": sum(int(state_counts.get(state, 0) or 0) for state in COMPLETED_STATES),
    }


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_batch_report(
    *,
    sessions_root: Path,
    output_dir: Path,
    houses_config_path: Path,
    pattern: str,
    target_house_id: str,
    segment_count: int,
    max_captures: int,
    limit: int,
    overwrite_reports: bool,
) -> Dict[str, Any]:
    session_dirs = discover_session_dirs(sessions_root, pattern=pattern)
    if limit > 0:
        session_dirs = session_dirs[:limit]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir.resolve()
    batch_json_path = output_dir / f"segment_search_memory_batch_summary_{run_id}.json"
    session_csv_path = output_dir / f"segment_search_memory_batch_sessions_{run_id}.csv"
    house_csv_path = output_dir / f"segment_search_memory_batch_houses_{run_id}.csv"

    session_rows: List[Dict[str, Any]] = []
    house_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for index, session_dir in enumerate(session_dirs, start=1):
        report_path = session_dir / "segment_search_memory_report.json"
        try:
            capture_count = len(discover_labeling_dirs(session_dir, max_captures=max_captures))
            if capture_count == 0:
                errors.append(
                    {
                        "session_name": session_dir.name,
                        "session_dir": str(session_dir),
                        "error": "no_labeling_fusion_results",
                    }
                )
                print(f"[{index}/{len(session_dirs)}] skip -> {session_dir.name} (no labeling fusion_result.json)")
                continue
            if overwrite_reports or not report_path.exists():
                report = replay_session(
                    session_dir,
                    houses_config_path=houses_config_path,
                    output_path=report_path,
                    target_house_id=target_house_id,
                    segment_count=segment_count,
                    max_captures=max_captures,
                )
            else:
                report = json.loads(report_path.read_text(encoding="utf-8-sig"))
                report["output_path"] = str(report_path)
            session_summary = build_session_summary(report, session_dir)
            session_rows.append(session_summary)
            house_rows.extend(flatten_house_rows(report))
            print(
                f"[{index}/{len(session_dirs)}] ok -> {session_dir.name} "
                f"captures={session_summary['capture_count']} "
                f"complete={session_summary['complete_segment_count']}/{session_summary['total_segments']} "
                f"repeat={session_summary['repeated_completed_segment_observations']} "
                f"review={session_summary['needs_review_count']}"
            )
        except Exception as exc:
            errors.append(
                {
                    "session_name": session_dir.name,
                    "session_dir": str(session_dir),
                    "error": str(exc),
                }
            )
            print(f"[{index}/{len(session_dirs)}] error -> {session_dir.name}: {exc}")

    aggregate_state_counts: Counter = Counter()
    aggregate_skip_reasons: Counter = Counter()
    for row in session_rows:
        aggregate_state_counts.update(row.get("state_counts", {}) or {})
        aggregate_skip_reasons.update(row.get("skip_reasons", {}) or {})

    batch_report = {
        "version": "segment_search_memory_batch_summary_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sessions_root": str(sessions_root.resolve()),
        "houses_config_path": str(houses_config_path.resolve()),
        "pattern": pattern,
        "target_house_override": target_house_id,
        "segment_count_per_face": segment_count,
        "max_captures": max_captures,
        "session_count_discovered": len(session_dirs),
        "session_count_processed": len(session_rows),
        "error_count": len(errors),
        "aggregate": {
            "capture_count": sum(int(row.get("capture_count", 0) or 0) for row in session_rows),
            "house_count": sum(int(row.get("house_count", 0) or 0) for row in session_rows),
            "total_segments": sum(int(row.get("total_segments", 0) or 0) for row in session_rows),
            "complete_segment_count": sum(int(row.get("complete_segment_count", 0) or 0) for row in session_rows),
            "repeated_completed_segment_observations": sum(
                int(row.get("repeated_completed_segment_observations", 0) or 0) for row in session_rows
            ),
            "skipped_capture_count": sum(int(row.get("skipped_capture_count", 0) or 0) for row in session_rows),
            "needs_review_count": sum(int(row.get("needs_review_count", 0) or 0) for row in session_rows),
            "entry_found_open_count": sum(int(row.get("entry_found_open_count", 0) or 0) for row in session_rows),
            "state_counts": dict(aggregate_state_counts),
            "skip_reasons": dict(aggregate_skip_reasons),
        },
        "sessions": session_rows,
        "houses": house_rows,
        "errors": errors,
        "outputs": {
            "batch_json": str(batch_json_path),
            "session_csv": str(session_csv_path),
            "house_csv": str(house_csv_path),
        },
    }

    write_json(batch_json_path, batch_report)
    write_csv(
        session_csv_path,
        session_rows,
        [
            "session_name",
            "capture_count",
            "processed_event_count",
            "house_count",
            "total_segments",
            "complete_segment_count",
            "completion_ratio",
            "repeated_completed_segment_observations",
            "skipped_capture_count",
            "needs_review_count",
            "entry_found_open_count",
            "report_path",
        ],
    )
    write_csv(
        house_csv_path,
        house_rows,
        [
            "session_name",
            "house_id",
            "total_segments",
            "complete_segment_count",
            "completion_ratio",
            "entry_found_segments",
            "needs_review",
            "blocked_or_occluded",
            "searched_no_entry",
            "searched_closed_entry",
            "searched_window_only",
            "entry_found_open",
            "unseen",
        ],
    )
    return batch_report


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch replay memory collection sessions with deterministic segment search memory."
    )
    parser.add_argument("--sessions_root", default=str(DEFAULT_SESSIONS_ROOT))
    parser.add_argument("--houses_config", default=str(DEFAULT_HOUSES_CONFIG))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--pattern", default="memory_episode_*")
    parser.add_argument("--target_house_id", default="")
    parser.add_argument("--segment_count_per_face", type=int, default=DEFAULT_SEGMENT_COUNT_PER_FACE)
    parser.add_argument("--max_captures", type=int, default=0, help="Only process first N captures in each session.")
    parser.add_argument("--limit", type=int, default=0, help="Only process first N sessions for quick tests.")
    parser.add_argument("--overwrite_reports", action="store_true", help="Regenerate each per-session report.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = build_batch_report(
        sessions_root=Path(args.sessions_root),
        output_dir=Path(args.output_dir),
        houses_config_path=Path(args.houses_config),
        pattern=str(args.pattern or "memory_episode_*"),
        target_house_id=str(args.target_house_id or "").strip(),
        segment_count=max(1, int(args.segment_count_per_face)),
        max_captures=max(0, int(args.max_captures)),
        limit=max(0, int(args.limit)),
        overwrite_reports=bool(args.overwrite_reports),
    )
    aggregate = report.get("aggregate", {})
    print(
        "[segment-batch] done "
        f"sessions={report.get('session_count_processed', 0)}/{report.get('session_count_discovered', 0)} "
        f"captures={aggregate.get('capture_count', 0)} "
        f"repeat={aggregate.get('repeated_completed_segment_observations', 0)} "
        f"review={aggregate.get('needs_review_count', 0)} "
        f"errors={report.get('error_count', 0)}"
    )
    outputs = report.get("outputs", {})
    print(f"[segment-batch] summary -> {outputs.get('batch_json')}")
    print(f"[segment-batch] sessions csv -> {outputs.get('session_csv')}")
    print(f"[segment-batch] houses csv -> {outputs.get('house_csv')}")


if __name__ == "__main__":
    main()
