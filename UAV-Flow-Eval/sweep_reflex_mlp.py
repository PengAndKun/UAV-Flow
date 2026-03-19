"""
Lightweight hyperparameter sweep for the Phase 3 MLP reflex baseline.

The goal is not huge AutoML, just a repeatable way to try a small grid on a
held-out split and pick a better MLP configuration than the current default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from evaluate_reflex_policy import evaluate_records
from reflex_policy_model import load_dataset_jsonl, save_artifact, train_mlp_artifact, resolve_target_action


def _parse_numeric_grid(text: str, cast) -> List[Any]:
    return [cast(part.strip()) for part in str(text or "").split(",") if part.strip()]


def _stratified_split(records: Sequence[Dict[str, Any]], validation_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    import random

    rng = random.Random(int(seed))
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(resolve_target_action(record), []).append(record)

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    for group in grouped.values():
        items = list(group)
        rng.shuffle(items)
        val_count = max(1, int(round(len(items) * float(validation_ratio))))
        if len(items) <= 2:
            val_count = 1
        val_records.extend(items[:val_count])
        train_records.extend(items[val_count:] or items[:1])
    return train_records, val_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep small MLP hyperparameter grids for Phase 3 reflex training")
    parser.add_argument("--dataset_jsonl", required=True, help="Path to phase3 reflex dataset JSONL")
    parser.add_argument("--hidden_dims", default="16,32,64", help="Comma-separated hidden widths")
    parser.add_argument("--learning_rates", default="0.005,0.01,0.02", help="Comma-separated learning rates")
    parser.add_argument("--weight_decays", default="0.0,1e-4,5e-4", help="Comma-separated weight decays")
    parser.add_argument("--class_weight_powers", default="0.0,0.5,1.0", help="Comma-separated class-weight powers")
    parser.add_argument("--epochs", type=int, default=250, help="Training epochs per trial")
    parser.add_argument("--validation_ratio", type=float, default=0.25, help="Held-out split ratio")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for split and initialization")
    parser.add_argument("--top_k", type=int, default=5, help="How many top trials to print")
    parser.add_argument("--export_json", default="", help="Optional path to save the sweep report JSON")
    parser.add_argument("--best_artifact_path", default="", help="Optional path to retrain and save the best config on the full dataset")
    parser.add_argument("--best_policy_name", default="mlp_reflex_policy_sweep_best", help="Policy name for the retrained best artifact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_dataset_jsonl(args.dataset_jsonl)
    if len(records) < 8:
        raise SystemExit("Not enough samples for a meaningful sweep.")

    train_records, val_records = _stratified_split(records, validation_ratio=float(args.validation_ratio), seed=int(args.seed))
    hidden_dims = _parse_numeric_grid(args.hidden_dims, int)
    learning_rates = _parse_numeric_grid(args.learning_rates, float)
    weight_decays = _parse_numeric_grid(args.weight_decays, float)
    class_weight_powers = _parse_numeric_grid(args.class_weight_powers, float)

    trials: List[Dict[str, Any]] = []
    for hidden_dim in hidden_dims:
        for learning_rate in learning_rates:
            for weight_decay in weight_decays:
                for class_weight_power in class_weight_powers:
                    artifact = train_mlp_artifact(
                        train_records,
                        policy_name=f"sweep_h{hidden_dim}_lr{learning_rate}_wd{weight_decay}_cw{class_weight_power}",
                        hidden_dim=int(hidden_dim),
                        epochs=int(args.epochs),
                        learning_rate=float(learning_rate),
                        weight_decay=float(weight_decay),
                        seed=int(args.seed),
                        class_weight_power=float(class_weight_power),
                    )
                    report = evaluate_records(val_records, artifact)
                    trials.append(
                        {
                            "hidden_dim": int(hidden_dim),
                            "learning_rate": float(learning_rate),
                            "weight_decay": float(weight_decay),
                            "class_weight_power": float(class_weight_power),
                            "val_action_accuracy": float(report.get("action_accuracy", 0.0)),
                            "val_should_execute_accuracy": float(report.get("should_execute_accuracy", 0.0)),
                            "val_avg_confidence": float(report.get("avg_confidence", 0.0)),
                            "train_accuracy": float(artifact.get("training_summary", {}).get("train_accuracy", 0.0)),
                            "final_loss": float(artifact.get("training_summary", {}).get("final_loss", 0.0)),
                        }
                    )

    ranked = sorted(
        trials,
        key=lambda trial: (trial["val_action_accuracy"], trial["val_avg_confidence"], -trial["final_loss"]),
        reverse=True,
    )
    best_trial = ranked[0]

    best_artifact_summary: Dict[str, Any] = {}
    if args.best_artifact_path:
        best_artifact = train_mlp_artifact(
            records,
            policy_name=str(args.best_policy_name),
            hidden_dim=int(best_trial["hidden_dim"]),
            epochs=int(args.epochs),
            learning_rate=float(best_trial["learning_rate"]),
            weight_decay=float(best_trial["weight_decay"]),
            seed=int(args.seed),
            class_weight_power=float(best_trial["class_weight_power"]),
        )
        save_artifact(best_artifact, args.best_artifact_path)
        best_artifact_summary = {
            "best_artifact_path": str(args.best_artifact_path),
            "policy_name": str(args.best_policy_name),
            "train_accuracy": float(best_artifact.get("training_summary", {}).get("train_accuracy", 0.0)),
            "final_loss": float(best_artifact.get("training_summary", {}).get("final_loss", 0.0)),
        }

    report = {
        "schema_version": "phase3.reflex_mlp_sweep_report.v1",
        "dataset_jsonl": args.dataset_jsonl,
        "sample_count": len(records),
        "train_count": len(train_records),
        "validation_count": len(val_records),
        "epochs": int(args.epochs),
        "best_trial": best_trial,
        "top_trials": ranked[: max(1, int(args.top_k))],
        "best_artifact": best_artifact_summary,
    }

    print("=== Reflex MLP Sweep ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.export_json:
        Path(args.export_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved sweep report to: {args.export_json}")


if __name__ == "__main__":
    main()
