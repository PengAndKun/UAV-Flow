"""
Coverage report for Phase 3 capture bundles or reflex dataset JSONL.

This focuses on two gaps we want to close next:
- richer multi-task data coverage
- broader multi-action coverage for local policy training
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

from reflex_policy_model import load_dataset_jsonl, normalize_action_name, resolve_target_action
from reflex_replay import discover_bundles, iter_bundles


CANONICAL_ACTIONS = [
    "forward",
    "backward",
    "left",
    "right",
    "up",
    "down",
    "yaw_left",
    "yaw_right",
]

def iter_dataset_records(args: argparse.Namespace) -> List[Dict[str, Any]]:
    if args.dataset_jsonl:
        return load_dataset_jsonl(args.dataset_jsonl)
    bundle_paths = discover_bundles(args.capture_dir)
    records = iter_bundles(bundle_paths)
    dataset_records: List[Dict[str, Any]] = []
    for record in records:
        reflex_sample = record.get("reflex_sample", {}) if isinstance(record.get("reflex_sample"), dict) else {}
        dataset_records.append(
            {
                "task_label": record.get("task_label", "idle"),
                "executed_action": record.get("action_label", "idle"),
                "target_action": normalize_action_name(record.get("action_label", "idle")),
                "suggested_action": reflex_sample.get("suggested_action", "idle"),
                "risk_score": reflex_sample.get("risk_score", 0.0),
                "policy_mode": reflex_sample.get("policy_mode", ""),
            }
        )
    return dataset_records


def build_report(records: Iterable[Dict[str, Any]], *, target_per_action: int) -> Dict[str, Any]:
    samples = list(records)
    action_counts: Counter = Counter()
    target_counts: Counter = Counter()
    suggested_counts: Counter = Counter()
    task_counts: Counter = Counter()
    task_action_counts: Dict[str, Counter] = defaultdict(Counter)

    for record in samples:
        task_label = str(record.get("task_label", "") or "idle")
        executed_action = normalize_action_name(record.get("executed_action", "idle"))
        target_action = resolve_target_action(record)
        suggested_action = normalize_action_name(record.get("suggested_action", "idle"))
        task_counts[task_label] += 1
        action_counts[executed_action] += 1
        target_counts[target_action] += 1
        suggested_counts[suggested_action] += 1
        task_action_counts[task_label][executed_action] += 1

    missing_global = [action for action in CANONICAL_ACTIONS if action_counts.get(action, 0) == 0]
    per_task_missing = {
        task_label: [action for action in CANONICAL_ACTIONS if counter.get(action, 0) == 0]
        for task_label, counter in sorted(task_action_counts.items())
    }
    low_coverage_actions = {
        action: max(0, int(target_per_action) - int(action_counts.get(action, 0)))
        for action in CANONICAL_ACTIONS
        if int(action_counts.get(action, 0)) < int(target_per_action)
    }
    recommendations: List[str] = []
    if missing_global:
        recommendations.append(f"Collect at least one sample for missing executed actions: {', '.join(missing_global)}")
    if low_coverage_actions:
        recommendations.append(
            "Increase per-action coverage toward the target count: "
            + ", ".join(f"{action}+={missing}" for action, missing in low_coverage_actions.items())
        )
    if len(task_counts) < 4:
        recommendations.append("Add more task labels so the dataset spans multiple semantic intents, not only one motion command.")
    if not recommendations:
        recommendations.append("Coverage looks healthy for the configured target_per_action threshold.")

    return {
        "sample_count": len(samples),
        "unique_task_count": len(task_counts),
        "executed_action_counts": dict(action_counts),
        "target_action_counts": dict(target_counts),
        "suggested_action_counts": dict(suggested_counts),
        "task_counts": dict(task_counts),
        "task_action_counts": {task: dict(counter) for task, counter in task_action_counts.items()},
        "missing_global_actions": missing_global,
        "per_task_missing_actions": per_task_missing,
        "low_coverage_actions": low_coverage_actions,
        "target_per_action": int(target_per_action),
        "recommendations": recommendations,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Coverage report for Phase 3 multi-task / multi-action reflex data")
    parser.add_argument("--dataset_jsonl", default="", help="Optional Phase 3 dataset JSONL")
    parser.add_argument("--capture_dir", default="./captures_remote", help="Capture directory used when dataset_jsonl is omitted")
    parser.add_argument("--target_per_action", type=int, default=12, help="Desired minimum sample count for each action")
    parser.add_argument("--export_json", default="", help="Optional path to save the report JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = iter_dataset_records(args)
    if not records:
        raise SystemExit("No Phase 3 records found for coverage reporting.")
    report = build_report(records, target_per_action=int(args.target_per_action))
    print("=== Reflex Coverage Report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.export_json:
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Saved report to: {args.export_json}")


if __name__ == "__main__":
    main()
