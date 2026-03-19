"""
Build episode-level manifests and training-ready JSONL from Phase 3 capture bundles.

This keeps the current manual capture workflow intact while producing:
- episode manifests grouped by task and capture time gap
- a flat JSONL file suitable for later reflex training/debugging
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from reflex_policy_model import normalize_action_name
from reflex_replay import discover_bundles, iter_bundles


def slugify_label(text: str, default: str = "idle") -> str:
    raw = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text or "").strip())
    compact = "_".join(part for part in raw.split("_") if part)
    return compact or default


def parse_capture_datetime(record: Dict[str, Any]) -> Optional[datetime]:
    capture_time = str(record.get("capture_time", "")).strip()
    if not capture_time:
        return None
    for fmt in ("%Y%m%d_%H%M%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(capture_time, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(capture_time.replace("Z", "+00:00"))
    except ValueError:
        return None


def should_keep_record(record: Dict[str, Any], *, schema_prefix: str, task_filter: str) -> bool:
    schema_version = str(record.get("dataset_schema_version", "") or "")
    if schema_prefix and not schema_version.startswith(schema_prefix):
        return False
    if task_filter and task_filter.lower() not in str(record.get("task_label", "")).lower():
        return False
    return True


def build_sample_record(record: Dict[str, Any], *, episode_id: str, step_index: int) -> Dict[str, Any]:
    reflex_sample = record.get("reflex_sample", {}) if isinstance(record.get("reflex_sample"), dict) else {}
    archive = record.get("archive", {}) if isinstance(record.get("archive"), dict) else {}
    archive_current = archive.get("current_cell", {}) if isinstance(archive.get("current_cell"), dict) else {}
    plan = record.get("plan", {}) if isinstance(record.get("plan"), dict) else {}
    pose = record.get("pose", {}) if isinstance(record.get("pose"), dict) else {}
    executed_action = str(record.get("action_label", "idle") or "idle")
    target_action = normalize_action_name(executed_action)
    suggested_action = str(reflex_sample.get("suggested_action", "idle") or "idle")

    return {
        "schema_version": "phase3.dataset_sample.v1",
        "episode_id": episode_id,
        "step_index": int(step_index),
        "capture_id": record.get("capture_id", ""),
        "capture_time": record.get("capture_time", ""),
        "capture_label": record.get("label", ""),
        "task_label": record.get("task_label", ""),
        "executed_action": executed_action,
        "executed_action_canonical": target_action,
        "target_action": target_action,
        "planner_name": plan.get("planner_name", ""),
        "planner_subgoal": plan.get("semantic_subgoal", ""),
        "planner_confidence": float(plan.get("planner_confidence", 0.0)),
        "pose": {
            "x": float(pose.get("x", 0.0)),
            "y": float(pose.get("y", 0.0)),
            "z": float(pose.get("z", 0.0)),
            "yaw": float(pose.get("yaw", 0.0)),
            "command_yaw": float(pose.get("command_yaw", 0.0)),
            "task_yaw": float(pose.get("task_yaw", 0.0)),
            "uav_yaw": float(pose.get("uav_yaw", 0.0)),
        },
        "depth": record.get("depth", {}),
        "target_waypoint": reflex_sample.get("target_waypoint", {}),
        "current_waypoint": reflex_sample.get("current_waypoint", {}),
        "suggested_action": suggested_action,
        "teacher_action": normalize_action_name(suggested_action),
        "policy_mode": reflex_sample.get("policy_mode", ""),
        "policy_name": reflex_sample.get("policy_name", ""),
        "policy_source": reflex_sample.get("policy_source", ""),
        "policy_confidence": float(reflex_sample.get("policy_confidence", 0.0)),
        "should_execute": bool(reflex_sample.get("should_execute", False)),
        "waypoint_distance_cm": float(reflex_sample.get("waypoint_distance_cm", 0.0)),
        "yaw_error_deg": float(reflex_sample.get("yaw_error_deg", 0.0)),
        "vertical_error_cm": float(reflex_sample.get("vertical_error_cm", 0.0)),
        "progress_to_waypoint_cm": float(reflex_sample.get("progress_to_waypoint_cm", 0.0)),
        "risk_score": float(reflex_sample.get("risk_score", 0.0)),
        "retrieval_cell_id": reflex_sample.get("retrieval_cell_id", "") or "none",
        "retrieval_score": float(reflex_sample.get("retrieval_score", 0.0)),
        "retrieval_semantic_subgoal": reflex_sample.get("retrieval_semantic_subgoal", ""),
        "archive_cell_id": reflex_sample.get("archive_cell_id", archive.get("current_cell_id", "")),
        "archive_visit_count": int(archive_current.get("visit_count", 0) or 0),
        "bundle_path": record.get("_bundle_path", ""),
        "rgb_image_path": record.get("rgb_image_path", ""),
        "depth_image_path": record.get("depth_image_path", ""),
        "depth_preview_path": record.get("depth_preview_path", ""),
        "camera_info_path": record.get("camera_info_path", ""),
    }


def group_records_into_episodes(records: Iterable[Dict[str, Any]], *, gap_seconds: float) -> List[List[Dict[str, Any]]]:
    episodes: List[List[Dict[str, Any]]] = []
    current_episode: List[Dict[str, Any]] = []
    previous_dt: Optional[datetime] = None
    previous_task = ""

    for record in records:
        capture_dt = parse_capture_datetime(record)
        task_label = str(record.get("task_label", "") or "idle")
        new_episode = not current_episode
        if current_episode:
            if task_label != previous_task:
                new_episode = True
            elif capture_dt is not None and previous_dt is not None:
                time_gap = (capture_dt - previous_dt).total_seconds()
                if time_gap > float(gap_seconds):
                    new_episode = True

        if new_episode and current_episode:
            episodes.append(current_episode)
            current_episode = []

        current_episode.append(record)
        previous_dt = capture_dt
        previous_task = task_label

    if current_episode:
        episodes.append(current_episode)
    return episodes


def build_episode_manifest(records: List[Dict[str, Any]], *, episode_index: int) -> Dict[str, Any]:
    task_label = str(records[0].get("task_label", "") or "idle")
    task_slug = slugify_label(task_label)
    episode_id = f"episode_{episode_index:04d}_{task_slug}"
    sample_records = [build_sample_record(record, episode_id=episode_id, step_index=idx) for idx, record in enumerate(records, start=1)]

    action_counts = Counter(sample["target_action"] for sample in sample_records)
    suggested_counts = Counter(sample["suggested_action"] for sample in sample_records)
    retrieval_hits = [sample for sample in sample_records if sample["retrieval_cell_id"] != "none"]
    unique_archive_cells = sorted({sample["archive_cell_id"] for sample in sample_records if sample["archive_cell_id"]})
    unique_retrieval_cells = sorted({sample["retrieval_cell_id"] for sample in sample_records if sample["retrieval_cell_id"] != "none"})
    waypoint_distances = [float(sample["waypoint_distance_cm"]) for sample in sample_records]
    risk_scores = [float(sample["risk_score"]) for sample in sample_records]

    start_dt = parse_capture_datetime(records[0])
    end_dt = parse_capture_datetime(records[-1])
    duration_sec = 0.0
    if start_dt is not None and end_dt is not None:
        duration_sec = max(0.0, (end_dt - start_dt).total_seconds())

    manifest = {
        "schema_version": "phase3.episode_manifest.v1",
        "episode_id": episode_id,
        "task_label": task_label,
        "task_slug": task_slug,
        "sample_count": len(sample_records),
        "start_time": records[0].get("capture_time", ""),
        "end_time": records[-1].get("capture_time", ""),
        "duration_sec": duration_sec,
        "bundle_paths": [record.get("_bundle_path", "") for record in records],
        "action_counts": dict(action_counts),
        "target_action_counts": dict(action_counts),
        "suggested_action_counts": dict(suggested_counts),
        "retrieval_hit_count": len(retrieval_hits),
        "retrieval_hit_rate": float(len(retrieval_hits) / len(sample_records)) if sample_records else 0.0,
        "unique_archive_cell_count": len(unique_archive_cells),
        "unique_retrieval_cell_count": len(unique_retrieval_cells),
        "unique_archive_cells": unique_archive_cells,
        "unique_retrieval_cells": unique_retrieval_cells,
        "avg_waypoint_distance_cm": sum(waypoint_distances) / len(waypoint_distances) if waypoint_distances else 0.0,
        "avg_risk_score": sum(risk_scores) / len(risk_scores) if risk_scores else 0.0,
        "samples": sample_records,
    }
    return manifest


def copy_sample_assets(samples: List[Dict[str, Any]], *, episode_dir: Path) -> None:
    assets_dir = episode_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    copied: set[str] = set()
    for sample in samples:
        for key in ("bundle_path", "rgb_image_path", "depth_image_path", "depth_preview_path", "camera_info_path"):
            raw_path = sample.get(key, "")
            if not raw_path:
                continue
            src = Path(raw_path)
            if not src.exists():
                continue
            if str(src) in copied:
                continue
            shutil.copy2(src, assets_dir / src.name)
            copied.add(str(src))


def write_outputs(episode_manifests: List[Dict[str, Any]], *, output_dir: str, copy_assets: bool) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)

    dataset_jsonl_path = out_dir / "reflex_dataset.jsonl"
    index_path = out_dir / "episode_index.json"

    dataset_lines: List[str] = []
    index_payload = {
        "schema_version": "phase3.episode_index.v1",
        "episode_count": len(episode_manifests),
        "episodes": [],
    }

    for manifest in episode_manifests:
        episode_dir = episodes_dir / manifest["episode_id"]
        episode_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = episode_dir / "episode_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        if copy_assets:
            copy_sample_assets(manifest["samples"], episode_dir=episode_dir)

        for sample in manifest["samples"]:
            dataset_lines.append(json.dumps(sample, ensure_ascii=False))

        index_payload["episodes"].append(
            {
                "episode_id": manifest["episode_id"],
                "task_label": manifest["task_label"],
                "sample_count": manifest["sample_count"],
                "duration_sec": manifest["duration_sec"],
                "manifest_path": str(manifest_path),
            }
        )

    dataset_jsonl_path.write_text("\n".join(dataset_lines) + ("\n" if dataset_lines else ""), encoding="utf-8")
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "output_dir": str(out_dir),
        "episode_index_path": str(index_path),
        "dataset_jsonl_path": str(dataset_jsonl_path),
        "episode_count": len(episode_manifests),
        "sample_count": len(dataset_lines),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build episode manifests and JSONL from Phase 3 capture bundles")
    parser.add_argument("--bundle", action="append", default=[], help="Specific bundle JSON path (can be repeated)")
    parser.add_argument("--capture_dir", default="./captures_remote", help="Directory containing *_bundle.json files")
    parser.add_argument("--output_dir", default="./phase3_dataset_export", help="Where to save episode manifests and JSONL")
    parser.add_argument("--gap_seconds", type=float, default=30.0, help="Start a new episode when the time gap exceeds this")
    parser.add_argument("--task_filter", default="", help="Only include bundles whose task label contains this string")
    parser.add_argument("--schema_prefix", default="phase3.capture_bundle.v1", help="Only include bundles whose schema starts with this")
    parser.add_argument("--copy_assets", action="store_true", help="Copy referenced RGB/depth/camera files into episode folders")
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

    grouped = group_records_into_episodes(records, gap_seconds=float(args.gap_seconds))
    manifests = [build_episode_manifest(group, episode_index=index) for index, group in enumerate(grouped, start=1)]
    output_payload = write_outputs(manifests, output_dir=args.output_dir, copy_assets=bool(args.copy_assets))

    print("=== Reflex Dataset Export ===")
    print(json.dumps(output_payload, ensure_ascii=False, indent=2))
    print()
    print("=== Episode Summary ===")
    episode_summary = [
        {
            "episode_id": manifest["episode_id"],
            "task_label": manifest["task_label"],
            "sample_count": manifest["sample_count"],
            "duration_sec": manifest["duration_sec"],
            "retrieval_hit_rate": manifest["retrieval_hit_rate"],
            "avg_waypoint_distance_cm": manifest["avg_waypoint_distance_cm"],
        }
        for manifest in manifests
    ]
    print(json.dumps(episode_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
