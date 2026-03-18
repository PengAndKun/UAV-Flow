"""
Train a minimal prototype-based local policy artifact from Phase 3 JSONL data.

This is a lightweight bridge from the current capture/replay stack to a
replaceable learned local-policy service.
"""

from __future__ import annotations

import argparse
import json

from reflex_policy_model import load_dataset_jsonl, save_artifact, train_prototype_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a prototype-based reflex local policy artifact")
    parser.add_argument("--dataset_jsonl", required=True, help="Path to phase3 reflex dataset JSONL")
    parser.add_argument("--output_path", required=True, help="Where to save the trained artifact JSON")
    parser.add_argument("--policy_name", default="prototype_reflex_policy", help="Policy name stored in the artifact")
    parser.add_argument("--task_filter", default="", help="Only train on samples whose task label contains this string")
    parser.add_argument("--min_count_per_action", type=int, default=1, help="Minimum number of samples required for an action prototype")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_dataset_jsonl(args.dataset_jsonl)
    if args.task_filter:
        needle = args.task_filter.lower()
        records = [record for record in records if needle in str(record.get("task_label", "")).lower()]
    if not records:
        raise SystemExit("No matching samples found for training.")

    artifact = train_prototype_artifact(
        records,
        policy_name=args.policy_name,
        min_count_per_action=int(args.min_count_per_action),
    )
    save_artifact(artifact, args.output_path)

    print("=== Reflex Policy Training Summary ===")
    print(json.dumps({
        "output_path": args.output_path,
        "policy_name": artifact.get("policy_name", ""),
        "sample_count": artifact.get("training_summary", {}).get("sample_count", 0),
        "action_counts": artifact.get("training_summary", {}).get("action_counts", {}),
        "default_action": artifact.get("default_action", ""),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
