from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from phase2_5_representation_distillation.evaluate_representation_ablation import (  # type: ignore
        evaluate_representation_ablation,
    )
    from phase2_5_representation_distillation.trainer import (  # type: ignore
        DistillationTrainerConfig,
        train_representation_distillation,
    )
else:
    from .evaluate_representation_ablation import evaluate_representation_ablation
    from .trainer import DistillationTrainerConfig, train_representation_distillation


DEFAULT_BASE_EXPORT_DIR = Path(
    r"E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\phase2_5_memory_aware_dataset_v3_20260427_20260427_141237"
)
DEFAULT_CONFIG_PATH = Path(r"E:\github\UAV-Flow\phase2_5_representation_distillation\configs\base_config.json")
DEFAULT_RUNS_ROOT = Path(r"E:\github\UAV-Flow\phase2_5_representation_distillation\runs")
DEFAULT_ABLATIONS_ROOT = Path(r"E:\github\UAV-Flow\phase2_5_representation_distillation\ablations")
DEFAULT_RANDOM_SPLIT_ROOT = Path(r"E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports")
DEFAULT_EXPERIMENT_NAME = "memory_aware_v5_formal_ablation"
DEFAULT_ABLATIONS = ("none", "zero_memory", "zero_candidates", "zero_memory_and_candidates")
CORE_METRICS = (
    "target_conditioned_state_accuracy",
    "target_conditioned_subgoal_accuracy",
    "target_conditioned_action_hint_accuracy",
    "target_conditioned_target_candidate_id_accuracy",
    "no_entry_mean_prob_when_true",
    "no_entry_mean_prob_when_false",
    "no_entry_separation_margin",
    "no_entry_false_high_prob_count",
    "no_entry_true_low_prob_count",
)


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else 0.0


def _variance(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    mean_value = _mean(values)
    return float(sum((float(value) - mean_value) ** 2 for value in values) / len(values))


def _metric_stats(values: Sequence[float]) -> Dict[str, Any]:
    value_list = [float(value) for value in values]
    variance = _variance(value_list)
    return {
        "mean": _mean(value_list),
        "variance": variance,
        "std": variance ** 0.5,
        "values": value_list,
    }


def _load_base_config(path: Path) -> Dict[str, Any]:
    payload = _read_json(path.resolve())
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return payload


def _load_export_rows(export_dir: Path) -> List[Dict[str, Any]]:
    rows = _read_jsonl(export_dir / "train.jsonl") + _read_jsonl(export_dir / "val.jsonl")
    rows = [dict(row) for row in rows]
    rows.sort(key=lambda item: str(item.get("sample_id") or ""))
    return rows


def _group_key(row: Mapping[str, Any]) -> str:
    value = str(row.get("target_conditioned_state") or "").strip()
    return value or "unknown"


def _class_counts(rows: Sequence[Mapping[str, Any]], field: str) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        value = str(row.get(field) or "").strip()
        if value:
            counter[value] += 1
    return dict(sorted(counter.items(), key=lambda pair: pair[0]))


def _split_rows(rows: Sequence[Mapping[str, Any]], *, val_ratio: float, seed: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = random.Random(int(seed))
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_group_key(row)].append(dict(row))

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        group = sorted(grouped[key], key=lambda item: str(item.get("sample_id") or ""))
        rng.shuffle(group)
        if len(group) == 1:
            train.extend(group)
            continue
        val_count = max(1, int(round(len(group) * float(val_ratio))))
        if val_count >= len(group):
            val_count = len(group) - 1
        val.extend(group[:val_count])
        train.extend(group[val_count:])

    train.sort(key=lambda item: str(item.get("sample_id") or ""))
    val.sort(key=lambda item: str(item.get("sample_id") or ""))
    for row in train:
        row["split"] = "train"
    for row in val:
        row["split"] = "val"
    return train, val


def _make_random_split_export(
    *,
    base_export_dir: Path,
    output_root: Path,
    split_index: int,
    split_seed: int,
    val_ratio: float,
) -> Path:
    rows = _load_export_rows(base_export_dir)
    train_rows, val_rows = _split_rows(rows, val_ratio=float(val_ratio), seed=int(split_seed))

    split_dir = output_root / f"split_{split_index:02d}_seed_{int(split_seed)}"
    split_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(split_dir / "train.jsonl", train_rows)
    _write_jsonl(split_dir / "val.jsonl", val_rows)
    (split_dir / "train_ids.txt").write_text(
        "\n".join(str(row.get("sample_id") or "") for row in train_rows),
        encoding="utf-8",
    )
    (split_dir / "val_ids.txt").write_text(
        "\n".join(str(row.get("sample_id") or "") for row in val_rows),
        encoding="utf-8",
    )

    manifest = {
        "export_dir": str(split_dir),
        "base_export_dir": str(base_export_dir),
        "split_index": int(split_index),
        "split_seed": int(split_seed),
        "val_ratio": float(val_ratio),
        "total_exported": len(rows),
        "train_count": len(train_rows),
        "val_count": len(val_rows),
        "target_conditioned_state_counts": _class_counts(rows, "target_conditioned_state"),
        "train_target_conditioned_state_counts": _class_counts(train_rows, "target_conditioned_state"),
        "val_target_conditioned_state_counts": _class_counts(val_rows, "target_conditioned_state"),
        "target_conditioned_subgoal_counts": _class_counts(rows, "target_conditioned_subgoal"),
        "train_target_conditioned_subgoal_counts": _class_counts(train_rows, "target_conditioned_subgoal"),
        "val_target_conditioned_subgoal_counts": _class_counts(val_rows, "target_conditioned_subgoal"),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(split_dir / "manifest.json", manifest)
    _write_json(split_dir / "quality_report.json", manifest)
    return split_dir


def _build_trainer_config(
    *,
    base_config: Mapping[str, Any],
    export_dir: Path,
    output_root: Path,
    run_name: str,
    seed: int,
) -> DistillationTrainerConfig:
    return DistillationTrainerConfig(
        export_dir=str(export_dir.resolve()),
        output_root=str(output_root.resolve()),
        run_name=str(run_name),
        top_k=int(base_config.get("top_k", 3)),
        batch_size=int(base_config.get("batch_size", 16)),
        epochs=int(base_config.get("epochs", 80)),
        learning_rate=float(base_config.get("learning_rate", 1e-3)),
        weight_decay=float(base_config.get("weight_decay", 1e-4)),
        patience=int(base_config.get("patience", 12)),
        device=str(base_config.get("device", "cpu")),
        num_workers=int(base_config.get("num_workers", 0)),
        dropout=float(base_config.get("dropout", 0.1)),
        global_hidden_dim=int(base_config.get("global_hidden_dim", 64)),
        candidate_hidden_dim=int(base_config.get("candidate_hidden_dim", 64)),
        fused_hidden_dim=int(base_config.get("fused_hidden_dim", 256)),
        representation_dim=int(base_config.get("representation_dim", 128)),
        save_every_epoch=bool(base_config.get("save_every_epoch", False)),
        stage1_epochs=int(base_config.get("stage1_epochs", 5)),
        stage2_epochs=int(base_config.get("stage2_epochs", 12)),
        seed=int(seed),
    )


def _extract_core_metrics(ablation_summary: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    aggregate = ablation_summary.get("aggregate_results", {})
    if not isinstance(aggregate, dict):
        aggregate = {}
    output: Dict[str, Dict[str, float]] = {}
    for mode, mode_payload in aggregate.items():
        if not isinstance(mode_payload, dict):
            continue
        output[str(mode)] = {}
        for metric_name in CORE_METRICS:
            metric_payload = mode_payload.get(metric_name, {})
            if isinstance(metric_payload, dict):
                output[str(mode)][metric_name] = float(metric_payload.get("mean", 0.0))
            else:
                output[str(mode)][metric_name] = float(metric_payload or 0.0)
    return output


def _extract_delta_metrics(ablation_summary: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    aggregate = ablation_summary.get("aggregate_deltas_vs_none", {})
    if not isinstance(aggregate, dict):
        aggregate = {}
    output: Dict[str, Dict[str, float]] = {}
    for mode, mode_payload in aggregate.items():
        if not isinstance(mode_payload, dict):
            continue
        output[str(mode)] = {}
        for metric_name, metric_payload in mode_payload.items():
            if isinstance(metric_payload, dict):
                output[str(mode)][str(metric_name)] = float(metric_payload.get("mean", 0.0))
    return output


def _aggregate_runs(run_records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    metric_values: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    delta_values: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
    for record in run_records:
        experiment_type = str(record.get("experiment_type") or "")
        metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), dict) else {}
        deltas = record.get("deltas_vs_none", {}) if isinstance(record.get("deltas_vs_none"), dict) else {}
        for mode, mode_metrics in metrics.items():
            if not isinstance(mode_metrics, dict):
                continue
            for metric_name, value in mode_metrics.items():
                metric_values.setdefault(experiment_type, {}).setdefault(str(mode), {}).setdefault(
                    str(metric_name),
                    [],
                ).append(float(value))
        for mode, mode_deltas in deltas.items():
            if not isinstance(mode_deltas, dict):
                continue
            for metric_name, value in mode_deltas.items():
                delta_values.setdefault(experiment_type, {}).setdefault(str(mode), {}).setdefault(
                    str(metric_name),
                    [],
                ).append(float(value))

    return {
        "metrics": {
            exp_type: {
                mode: {
                    metric_name: _metric_stats(values)
                    for metric_name, values in sorted(mode_metrics.items())
                }
                for mode, mode_metrics in sorted(mode_payload.items())
            }
            for exp_type, mode_payload in sorted(metric_values.items())
        },
        "deltas_vs_none": {
            exp_type: {
                mode: {
                    metric_name: _metric_stats(values)
                    for metric_name, values in sorted(mode_metrics.items())
                }
                for mode, mode_metrics in sorted(mode_payload.items())
            }
            for exp_type, mode_payload in sorted(delta_values.items())
        },
    }


def _write_runs_csv(path: Path, run_records: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "experiment_type",
        "repeat_index",
        "train_seed",
        "split_seed",
        "export_dir",
        "checkpoint_path",
        "ablation_mode",
        *CORE_METRICS,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in run_records:
            metrics = record.get("metrics", {}) if isinstance(record.get("metrics"), dict) else {}
            for mode, mode_metrics in metrics.items():
                row = {
                    "experiment_type": record.get("experiment_type", ""),
                    "repeat_index": record.get("repeat_index", ""),
                    "train_seed": record.get("train_seed", ""),
                    "split_seed": record.get("split_seed", ""),
                    "export_dir": record.get("export_dir", ""),
                    "checkpoint_path": record.get("checkpoint_path", ""),
                    "ablation_mode": mode,
                }
                if isinstance(mode_metrics, dict):
                    for metric_name in CORE_METRICS:
                        row[metric_name] = mode_metrics.get(metric_name, "")
                writer.writerow(row)


def run_formal_ablation_experiments(
    *,
    base_export_dir: Path,
    config_path: Path,
    runs_root: Path,
    ablations_root: Path,
    random_split_root: Path,
    experiment_name: str,
    num_repeats: int,
    start_seed: int,
    val_ratio: float,
    random_split_training_seed: int,
    experiment_modes: Sequence[str],
    ablations: Sequence[str],
    inference_repeats: int,
) -> Dict[str, Any]:
    base_config = _load_base_config(config_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_root_name = f"{experiment_name}_{timestamp}"
    run_root = runs_root.resolve() / experiment_root_name
    ablation_root = ablations_root.resolve() / experiment_root_name
    split_root = random_split_root.resolve() / f"{experiment_root_name}_random_splits"
    run_root.mkdir(parents=True, exist_ok=True)
    ablation_root.mkdir(parents=True, exist_ok=True)
    split_root.mkdir(parents=True, exist_ok=True)

    run_records: List[Dict[str, Any]] = []
    modes = {str(mode).strip() for mode in experiment_modes}

    if "seed_retrain" in modes:
        for repeat_idx in range(int(num_repeats)):
            train_seed = int(start_seed) + repeat_idx
            run_name = f"seed_retrain_{repeat_idx:02d}_seed_{train_seed}"
            print(f"[formal-ablation] seed_retrain {repeat_idx + 1}/{num_repeats}: seed={train_seed}")
            trainer_summary = train_representation_distillation(
                _build_trainer_config(
                    base_config=base_config,
                    export_dir=base_export_dir,
                    output_root=run_root / "seed_retrain",
                    run_name=run_name,
                    seed=train_seed,
                )
            )
            checkpoint_path = Path(str(trainer_summary["checkpoints_dir"])) / "best.pt"
            ablation_summary = evaluate_representation_ablation(
                checkpoint_path=checkpoint_path,
                export_dir=base_export_dir,
                output_dir=ablation_root / "seed_retrain" / run_name,
                split="val",
                ablations=ablations,
                batch_size=int(base_config.get("batch_size", 16)),
                device=str(base_config.get("device", "cpu")),
                num_workers=int(base_config.get("num_workers", 0)),
                repeats=int(inference_repeats),
            )
            run_records.append(
                {
                    "experiment_type": "seed_retrain",
                    "repeat_index": repeat_idx,
                    "train_seed": train_seed,
                    "split_seed": None,
                    "export_dir": str(base_export_dir.resolve()),
                    "training_summary_path": str(Path(str(trainer_summary["output_dir"])) / "training_summary.json"),
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "ablation_summary_path": str(Path(str(ablation_summary["output_dir"])) / "ablation_summary.json"),
                    "metrics": _extract_core_metrics(ablation_summary),
                    "deltas_vs_none": _extract_delta_metrics(ablation_summary),
                }
            )

    if "random_split" in modes:
        for repeat_idx in range(int(num_repeats)):
            split_seed = int(start_seed) + 1000 + repeat_idx
            run_name = f"random_split_{repeat_idx:02d}_split_seed_{split_seed}"
            print(
                f"[formal-ablation] random_split {repeat_idx + 1}/{num_repeats}: "
                f"split_seed={split_seed} train_seed={random_split_training_seed}"
            )
            split_export_dir = _make_random_split_export(
                base_export_dir=base_export_dir.resolve(),
                output_root=split_root,
                split_index=repeat_idx,
                split_seed=split_seed,
                val_ratio=float(val_ratio),
            )
            trainer_summary = train_representation_distillation(
                _build_trainer_config(
                    base_config=base_config,
                    export_dir=split_export_dir,
                    output_root=run_root / "random_split",
                    run_name=run_name,
                    seed=int(random_split_training_seed),
                )
            )
            checkpoint_path = Path(str(trainer_summary["checkpoints_dir"])) / "best.pt"
            ablation_summary = evaluate_representation_ablation(
                checkpoint_path=checkpoint_path,
                export_dir=split_export_dir,
                output_dir=ablation_root / "random_split" / run_name,
                split="val",
                ablations=ablations,
                batch_size=int(base_config.get("batch_size", 16)),
                device=str(base_config.get("device", "cpu")),
                num_workers=int(base_config.get("num_workers", 0)),
                repeats=int(inference_repeats),
            )
            run_records.append(
                {
                    "experiment_type": "random_split",
                    "repeat_index": repeat_idx,
                    "train_seed": int(random_split_training_seed),
                    "split_seed": split_seed,
                    "export_dir": str(split_export_dir.resolve()),
                    "training_summary_path": str(Path(str(trainer_summary["output_dir"])) / "training_summary.json"),
                    "checkpoint_path": str(checkpoint_path.resolve()),
                    "ablation_summary_path": str(Path(str(ablation_summary["output_dir"])) / "ablation_summary.json"),
                    "metrics": _extract_core_metrics(ablation_summary),
                    "deltas_vs_none": _extract_delta_metrics(ablation_summary),
                }
            )

    aggregate = _aggregate_runs(run_records)
    summary = {
        "experiment_name": experiment_name,
        "experiment_root_name": experiment_root_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_export_dir": str(base_export_dir.resolve()),
        "config_path": str(config_path.resolve()),
        "runs_root": str(run_root),
        "ablations_root": str(ablation_root),
        "random_split_root": str(split_root),
        "num_repeats": int(num_repeats),
        "start_seed": int(start_seed),
        "val_ratio": float(val_ratio),
        "random_split_training_seed": int(random_split_training_seed),
        "experiment_modes": list(experiment_modes),
        "ablations": list(ablations),
        "inference_repeats": int(inference_repeats),
        "run_count": len(run_records),
        "runs": run_records,
        "aggregate": aggregate,
    }
    summary_path = ablation_root / "formal_ablation_summary.json"
    runs_path = ablation_root / "formal_ablation_runs.jsonl"
    csv_path = ablation_root / "formal_ablation_table.csv"
    _write_json(summary_path, summary)
    _write_jsonl(runs_path, run_records)
    _write_runs_csv(csv_path, run_records)
    print(f"[formal-ablation] summary -> {summary_path}")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run formal memory ablation experiments with repeated training seeds and random splits."
    )
    parser.add_argument("--base_export_dir", type=Path, default=DEFAULT_BASE_EXPORT_DIR)
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--runs_root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--ablations_root", type=Path, default=DEFAULT_ABLATIONS_ROOT)
    parser.add_argument("--random_split_root", type=Path, default=DEFAULT_RANDOM_SPLIT_ROOT)
    parser.add_argument("--experiment_name", type=str, default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--num_repeats", type=int, default=10)
    parser.add_argument("--start_seed", type=int, default=202604270)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--random_split_training_seed", type=int, default=202604279)
    parser.add_argument(
        "--experiment_modes",
        nargs="+",
        default=["seed_retrain", "random_split"],
        choices=("seed_retrain", "random_split"),
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=list(DEFAULT_ABLATIONS),
        choices=DEFAULT_ABLATIONS,
    )
    parser.add_argument("--inference_repeats", type=int, default=1)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = run_formal_ablation_experiments(
        base_export_dir=args.base_export_dir.resolve(),
        config_path=args.config_path.resolve(),
        runs_root=args.runs_root.resolve(),
        ablations_root=args.ablations_root.resolve(),
        random_split_root=args.random_split_root.resolve(),
        experiment_name=str(args.experiment_name),
        num_repeats=int(args.num_repeats),
        start_seed=int(args.start_seed),
        val_ratio=float(args.val_ratio),
        random_split_training_seed=int(args.random_split_training_seed),
        experiment_modes=[str(value) for value in args.experiment_modes],
        ablations=[str(value) for value in args.ablations],
        inference_repeats=int(args.inference_repeats),
    )
    print(
        "[formal-ablation] done: "
        f"runs={summary['run_count']} "
        f"summary={Path(summary['ablations_root']) / 'formal_ablation_summary.json'}"
    )


if __name__ == "__main__":
    main()
