"""
Train a minimal prototype-based local policy artifact from Phase 3 JSONL data.

This is a lightweight bridge from the current capture/replay stack to a
replaceable learned local-policy service.
"""

from __future__ import annotations

import argparse
import json

from reflex_policy_model import (
    load_dataset_jsonl,
    save_artifact,
    train_mlp_artifact,
    train_prototype_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a prototype-based reflex local policy artifact")
    parser.add_argument("--dataset_jsonl", required=True, help="Path to phase3 reflex dataset JSONL")
    parser.add_argument("--output_path", required=True, help="Where to save the trained artifact JSON")
    parser.add_argument("--policy_name", default="prototype_reflex_policy", help="Policy name stored in the artifact")
    parser.add_argument("--task_filter", default="", help="Only train on samples whose task label contains this string")
    parser.add_argument("--model_type", default="prototype", choices=["prototype", "mlp"], help="Which lightweight local policy baseline to train")
    parser.add_argument("--min_count_per_action", type=int, default=1, help="Minimum number of samples required for an action prototype")
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden width used by the MLP classifier")
    parser.add_argument("--epochs", type=int, default=250, help="Training epochs for the MLP classifier")
    parser.add_argument("--learning_rate", type=float, default=0.02, help="Learning rate for the MLP classifier")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization for the MLP classifier")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for MLP initialization")
    parser.add_argument("--class_weight_power", type=float, default=0.0, help="Inverse-frequency class weighting power used by the MLP loss")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_dataset_jsonl(args.dataset_jsonl)
    if args.task_filter:
        needle = args.task_filter.lower()
        records = [record for record in records if needle in str(record.get("task_label", "")).lower()]
    if not records:
        raise SystemExit("No matching samples found for training.")

    if args.model_type == "prototype":
        artifact = train_prototype_artifact(
            records,
            policy_name=args.policy_name,
            min_count_per_action=int(args.min_count_per_action),
        )
    else:
        artifact = train_mlp_artifact(
            records,
            policy_name=args.policy_name,
            hidden_dim=int(args.hidden_dim),
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            seed=int(args.seed),
            class_weight_power=float(args.class_weight_power),
        )
    save_artifact(artifact, args.output_path)

    print("=== Reflex Policy Training Summary ===")
    print(json.dumps({
        "output_path": args.output_path,
        "policy_name": artifact.get("policy_name", ""),
        "model_type": artifact.get("model_type", "prototype"),
        "target_label_field": "target_action",
        "sample_count": artifact.get("training_summary", {}).get("sample_count", 0),
        "action_counts": artifact.get("training_summary", {}).get("action_counts", {}),
        "default_action": artifact.get("default_action", ""),
        "train_accuracy": artifact.get("training_summary", {}).get("train_accuracy", None),
        "final_loss": artifact.get("training_summary", {}).get("final_loss", None),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
