"""
Offline replay / summary tool for Phase 3 reflex capture bundles.

This script does not drive the simulator. It reads saved capture bundle JSON
files and produces:
- a chronological replay-style text summary
- aggregate action/risk/waypoint statistics

It is meant to validate that the current capture metadata is already useful for
later reflex training and debugging.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List


def load_bundle(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid bundle payload: {path}")
    payload["_bundle_path"] = path
    return payload


def discover_bundles(capture_dir: str) -> List[str]:
    candidates: List[str] = []
    for entry in os.listdir(capture_dir):
        if entry.endswith("_bundle.json"):
            candidates.append(os.path.join(capture_dir, entry))
    candidates.sort()
    return candidates


def iter_bundles(paths: Iterable[str]) -> List[Dict[str, Any]]:
    records = [load_bundle(path) for path in paths]
    records.sort(key=lambda item: (str(item.get("capture_time", "")), str(item.get("capture_id", ""))))
    return records


def summarize_bundles(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    action_counter: Counter = Counter()
    suggested_counter: Counter = Counter()
    task_counter: Counter = Counter()
    retrieval_counter: Counter = Counter()
    risk_values: List[float] = []
    waypoint_distances: List[float] = []
    per_task: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"count": 0, "avg_risk_sum": 0.0, "avg_wp_sum": 0.0})

    for record in records:
        task_label = str(record.get("task_label", "") or "idle")
        action_label = str(record.get("action_label", "") or "idle")
        reflex_sample = record.get("reflex_sample", {}) if isinstance(record.get("reflex_sample"), dict) else {}

        task_counter[task_label] += 1
        action_counter[action_label] += 1
        suggested_counter[str(reflex_sample.get("suggested_action", "idle"))] += 1
        retrieval_cell_id = str(reflex_sample.get("retrieval_cell_id", "") or "none")
        retrieval_counter[retrieval_cell_id] += 1

        risk = float(reflex_sample.get("risk_score", 0.0))
        wp_distance = float(reflex_sample.get("waypoint_distance_cm", 0.0))
        risk_values.append(risk)
        waypoint_distances.append(wp_distance)
        per_task[task_label]["count"] += 1
        per_task[task_label]["avg_risk_sum"] += risk
        per_task[task_label]["avg_wp_sum"] += wp_distance

    avg_risk = sum(risk_values) / len(risk_values) if risk_values else 0.0
    avg_wp_distance = sum(waypoint_distances) / len(waypoint_distances) if waypoint_distances else 0.0

    task_summary: Dict[str, Dict[str, Any]] = {}
    for task_label, values in per_task.items():
        count = max(1, int(values["count"]))
        task_summary[task_label] = {
            "count": int(values["count"]),
            "avg_risk": values["avg_risk_sum"] / count,
            "avg_waypoint_distance_cm": values["avg_wp_sum"] / count,
        }

    return {
        "bundle_count": len(records),
        "task_counts": dict(task_counter),
        "executed_action_counts": dict(action_counter),
        "suggested_action_counts": dict(suggested_counter),
        "retrieval_cell_counts": dict(retrieval_counter),
        "avg_risk_score": avg_risk,
        "avg_waypoint_distance_cm": avg_wp_distance,
        "per_task": task_summary,
    }


def render_replay_lines(records: List[Dict[str, Any]], limit: int) -> List[str]:
    lines: List[str] = []
    for idx, record in enumerate(records[: max(0, int(limit))], start=1):
        reflex_sample = record.get("reflex_sample", {}) if isinstance(record.get("reflex_sample"), dict) else {}
        lines.append(
            f"{idx:03d}. "
            f"time={record.get('capture_time', 'n/a')} "
            f"task={record.get('task_label', 'idle')} "
            f"exec={record.get('action_label', 'idle')} "
            f"suggest={reflex_sample.get('suggested_action', 'idle')} "
            f"wp={float(reflex_sample.get('waypoint_distance_cm', 0.0)):.1f}cm "
            f"yaw_err={float(reflex_sample.get('yaw_error_deg', 0.0)):.1f}deg "
            f"risk={float(reflex_sample.get('risk_score', 0.0)):.2f} "
            f"retrieval={reflex_sample.get('retrieval_cell_id', 'none') or 'none'}"
        )
    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline replay/summary for Phase 3 reflex capture bundles")
    parser.add_argument("--bundle", action="append", default=[], help="Specific bundle JSON path (can be repeated)")
    parser.add_argument("--capture_dir", default="./captures_remote", help="Directory containing *_bundle.json files")
    parser.add_argument("--limit", type=int, default=20, help="How many chronological replay lines to print")
    parser.add_argument("--task_filter", default="", help="Only include bundles whose task label contains this string")
    parser.add_argument("--export_json", default="", help="Optional path to save the summary JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_paths = list(args.bundle)
    if not bundle_paths:
        bundle_paths = discover_bundles(args.capture_dir)

    records = iter_bundles(bundle_paths)
    if args.task_filter:
        needle = args.task_filter.lower()
        records = [record for record in records if needle in str(record.get("task_label", "")).lower()]

    if not records:
        raise SystemExit("No matching capture bundles found.")

    summary = summarize_bundles(records)
    replay_lines = render_replay_lines(records, limit=args.limit)

    print("=== Reflex Replay Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    print("=== Replay Lines ===")
    for line in replay_lines:
        print(line)

    if args.export_json:
        export_payload = {
            "summary": summary,
            "replay_lines": replay_lines,
            "bundle_paths": [record.get("_bundle_path", "") for record in records],
        }
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(export_payload, f, ensure_ascii=False, indent=2)
        print()
        print(f"Saved replay summary to: {args.export_json}")


if __name__ == "__main__":
    main()
