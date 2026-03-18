"""
Analyze Phase 3 archive retrieval quality from capture bundles.

This report focuses on whether retrieval is showing non-default, goal-aligned
signals that are strong enough to support later archive-distilled policy work.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any, Dict, List

from reflex_replay import discover_bundles, iter_bundles


def should_keep_record(record: Dict[str, Any], *, schema_prefix: str, task_filter: str) -> bool:
    schema_version = str(record.get("dataset_schema_version", "") or "")
    if schema_prefix and not schema_version.startswith(schema_prefix):
        return False
    if task_filter and task_filter.lower() not in str(record.get("task_label", "")).lower():
        return False
    return True


def summarize_retrieval(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    retrieval_count = 0
    same_task_count = 0
    same_subgoal_count = 0
    should_execute_count = 0
    positive_progress_count = 0
    retrieval_scores: List[float] = []
    retrieval_risks: List[float] = []
    current_risks: List[float] = []
    visit_counts: List[int] = []
    current_visit_counts: List[int] = []

    task_breakdown: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {
            "samples": 0,
            "retrieval_hits": 0,
            "same_task_hits": 0,
            "same_subgoal_hits": 0,
            "avg_retrieval_score_sum": 0.0,
        }
    )
    retrieval_cell_counter: Counter[str] = Counter()
    subgoal_counter: Counter[str] = Counter()

    for record in records:
        task_label = str(record.get("task_label", "") or "idle")
        reflex_sample = record.get("reflex_sample", {}) if isinstance(record.get("reflex_sample"), dict) else {}
        archive = record.get("archive", {}) if isinstance(record.get("archive"), dict) else {}
        active_retrieval = archive.get("active_retrieval", {}) if isinstance(archive.get("active_retrieval"), dict) else {}
        current_cell = archive.get("current_cell", {}) if isinstance(archive.get("current_cell"), dict) else {}
        planner_subgoal = str(reflex_sample.get("planner_subgoal", "") or (record.get("plan", {}) or {}).get("semantic_subgoal", ""))
        retrieval_subgoal = str(reflex_sample.get("retrieval_semantic_subgoal", "") or active_retrieval.get("semantic_subgoal", ""))
        retrieval_cell_id = str(reflex_sample.get("retrieval_cell_id", "") or "none")
        retrieval_score = float(reflex_sample.get("retrieval_score", 0.0) or active_retrieval.get("retrieval_score", 0.0) or 0.0)
        progress_cm = float(reflex_sample.get("progress_to_waypoint_cm", 0.0))

        task_breakdown[task_label]["samples"] += 1
        subgoal_counter[planner_subgoal or "idle"] += 1
        current_risks.append(float(reflex_sample.get("risk_score", 0.0)))
        current_visit_counts.append(int(current_cell.get("visit_count", 0) or 0))

        if bool(reflex_sample.get("should_execute", False)):
            should_execute_count += 1
        if progress_cm > 0.0:
            positive_progress_count += 1

        if retrieval_cell_id != "none":
            retrieval_count += 1
            retrieval_cell_counter[retrieval_cell_id] += 1
            retrieval_scores.append(retrieval_score)
            retrieval_risks.append(float(active_retrieval.get("risk_score", 0.0)))
            visit_counts.append(int(active_retrieval.get("visit_count", 0) or 0))
            task_breakdown[task_label]["retrieval_hits"] += 1
            task_breakdown[task_label]["avg_retrieval_score_sum"] += retrieval_score

            retrieval_task = str(active_retrieval.get("task_label", "") or task_label)
            if retrieval_task == task_label:
                same_task_count += 1
                task_breakdown[task_label]["same_task_hits"] += 1
            if retrieval_subgoal and planner_subgoal and retrieval_subgoal == planner_subgoal:
                same_subgoal_count += 1
                task_breakdown[task_label]["same_subgoal_hits"] += 1

    total = len(records)
    per_task: Dict[str, Dict[str, Any]] = {}
    for task_label, stats in task_breakdown.items():
        retrieval_hits = max(1, int(stats["retrieval_hits"])) if int(stats["retrieval_hits"]) > 0 else 1
        per_task[task_label] = {
            "samples": int(stats["samples"]),
            "retrieval_hits": int(stats["retrieval_hits"]),
            "retrieval_hit_rate": float(stats["retrieval_hits"] / stats["samples"]) if stats["samples"] else 0.0,
            "same_task_hit_rate": float(stats["same_task_hits"] / stats["retrieval_hits"]) if stats["retrieval_hits"] else 0.0,
            "same_subgoal_hit_rate": float(stats["same_subgoal_hits"] / stats["retrieval_hits"]) if stats["retrieval_hits"] else 0.0,
            "avg_retrieval_score": float(stats["avg_retrieval_score_sum"] / retrieval_hits) if stats["retrieval_hits"] else 0.0,
        }

    return {
        "schema_version": "phase3.retrieval_quality_report.v1",
        "sample_count": total,
        "retrieval_count": retrieval_count,
        "retrieval_hit_rate": float(retrieval_count / total) if total else 0.0,
        "same_task_hit_rate": float(same_task_count / retrieval_count) if retrieval_count else 0.0,
        "same_subgoal_hit_rate": float(same_subgoal_count / retrieval_count) if retrieval_count else 0.0,
        "should_execute_rate": float(should_execute_count / total) if total else 0.0,
        "positive_progress_rate": float(positive_progress_count / total) if total else 0.0,
        "avg_retrieval_score": float(sum(retrieval_scores) / len(retrieval_scores)) if retrieval_scores else 0.0,
        "avg_retrieval_risk": float(sum(retrieval_risks) / len(retrieval_risks)) if retrieval_risks else 0.0,
        "avg_current_risk": float(sum(current_risks) / len(current_risks)) if current_risks else 0.0,
        "avg_retrieval_visit_count": float(sum(visit_counts) / len(visit_counts)) if visit_counts else 0.0,
        "avg_current_visit_count": float(sum(current_visit_counts) / len(current_visit_counts)) if current_visit_counts else 0.0,
        "planner_subgoal_counts": dict(subgoal_counter),
        "top_retrieval_cells": retrieval_cell_counter.most_common(10),
        "per_task": per_task,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze retrieval quality from Phase 3 capture bundles")
    parser.add_argument("--bundle", action="append", default=[], help="Specific bundle JSON path (can be repeated)")
    parser.add_argument("--capture_dir", default="./captures_remote", help="Directory containing *_bundle.json files")
    parser.add_argument("--task_filter", default="", help="Only include bundles whose task label contains this string")
    parser.add_argument("--schema_prefix", default="phase3.capture_bundle.v1", help="Only include bundles whose schema starts with this")
    parser.add_argument("--export_json", default="", help="Optional path to save the report JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle_paths = list(args.bundle) or discover_bundles(args.capture_dir)
    records = iter_bundles(bundle_paths)
    records = [
        record
        for record in records
        if should_keep_record(record, schema_prefix=str(args.schema_prefix or ""), task_filter=str(args.task_filter or ""))
    ]
    if not records:
        raise SystemExit("No matching Phase 3 capture bundles found.")

    report = summarize_retrieval(records)
    print("=== Retrieval Quality Report ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.export_json:
        with open(args.export_json, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print()
        print(f"Saved retrieval report to: {args.export_json}")


if __name__ == "__main__":
    main()
