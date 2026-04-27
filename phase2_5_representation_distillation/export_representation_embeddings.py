from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from phase2_5_representation_distillation.dataset import DistillationDataset  # type: ignore
    from phase2_5_representation_distillation.evaluator import (  # type: ignore
        TensorizedEvalDataset,
        _collate_examples,
        _load_checkpoint,
        _move_batch_to_device,
    )
    from phase2_5_representation_distillation.feature_builder import build_training_example  # type: ignore
    from phase2_5_representation_distillation.label_schema import (  # type: ignore
        ACTION_HINT_LABELS,
        ENTRY_STATE_LABELS,
        SUBGOAL_LABELS,
        TARGET_CONDITIONED_ACTION_HINT_LABELS,
        TARGET_CONDITIONED_STATE_LABELS,
        TARGET_CONDITIONED_SUBGOAL_LABELS,
        decode_target_candidate_id,
        target_candidate_num_classes,
    )
    from phase2_5_representation_distillation.model import (  # type: ignore
        EntryRepresentationStudent,
        StudentModelConfig,
    )
else:
    from .dataset import DistillationDataset
    from .evaluator import TensorizedEvalDataset, _collate_examples, _load_checkpoint, _move_batch_to_device
    from .feature_builder import build_training_example
    from .label_schema import (
        ACTION_HINT_LABELS,
        ENTRY_STATE_LABELS,
        SUBGOAL_LABELS,
        TARGET_CONDITIONED_ACTION_HINT_LABELS,
        TARGET_CONDITIONED_STATE_LABELS,
        TARGET_CONDITIONED_SUBGOAL_LABELS,
        decode_target_candidate_id,
        target_candidate_num_classes,
    )
    from .model import EntryRepresentationStudent, StudentModelConfig

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]


DEFAULT_CHECKPOINT_PATH = Path(
    r"E:\github\UAV-Flow\phase2_5_representation_distillation\runs\memory_aware_v5_pilot_20260427\checkpoints\best.pt"
)
DEFAULT_EXPORT_DIR = Path(
    r"E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\phase2_5_memory_aware_dataset_v3_20260427_20260427_141237"
)
DEFAULT_OUTPUT_ROOT = Path(r"E:\github\UAV-Flow\phase2_5_representation_distillation\embeddings")


HEAD_SPECS: Tuple[Tuple[str, str, str, Tuple[str, ...]], ...] = (
    ("entry_state", "entry_state_logits", "entry_state", ENTRY_STATE_LABELS),
    ("subgoal", "subgoal_logits", "subgoal", SUBGOAL_LABELS),
    ("action_hint", "action_hint_logits", "action_hint", ACTION_HINT_LABELS),
    (
        "target_conditioned_state",
        "target_state_logits",
        "target_conditioned_state",
        TARGET_CONDITIONED_STATE_LABELS,
    ),
    (
        "target_conditioned_subgoal",
        "target_subgoal_logits",
        "target_conditioned_subgoal",
        TARGET_CONDITIONED_SUBGOAL_LABELS,
    ),
    (
        "target_conditioned_action_hint",
        "target_action_logits",
        "target_conditioned_action_hint",
        TARGET_CONDITIONED_ACTION_HINT_LABELS,
    ),
)


def _require_runtime() -> None:
    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required to export representation embeddings.")
    if np is None:
        raise ImportError("NumPy is required to write embeddings.npy.")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _append_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _load_examples_for_split(export_root: Path, split: str, *, top_k: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    dataset = DistillationDataset.from_export_dir(export_root, split, lazy=False)
    raw_samples = [dataset[idx] for idx in range(len(dataset))]
    examples = [build_training_example(sample, top_k=top_k) for sample in raw_samples]
    return raw_samples, examples


def _load_requested_examples(
    export_root: Path,
    split: str,
    *,
    top_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    split_name = str(split).strip().lower()
    if split_name == "all":
        raw_all: List[Dict[str, Any]] = []
        examples_all: List[Dict[str, Any]] = []
        for part in ("train", "val"):
            raw_samples, examples = _load_examples_for_split(export_root, part, top_k=top_k)
            raw_all.extend(raw_samples)
            examples_all.extend(examples)
        return raw_all, examples_all
    if split_name not in {"train", "val"}:
        raise ValueError(f"split must be one of train, val, all; got '{split}'")
    return _load_examples_for_split(export_root, split_name, top_k=top_k)


def _label_or_none(label_id: int, labels: Sequence[str]) -> str | None:
    value = int(label_id)
    if 0 <= value < len(labels):
        return str(labels[value])
    return None


def _candidate_label_names(*, top_k: int) -> List[str]:
    return [f"candidate_{idx}" for idx in range(int(top_k))] + ["null_candidate"]


def _decode_candidate_label(label_id: int, *, top_k: int) -> str:
    decoded = decode_target_candidate_id(int(label_id), top_k=int(top_k))
    return "null_candidate" if int(decoded) < 0 else f"candidate_{decoded}"


def _probability_map(values: Sequence[float], labels: Sequence[str]) -> Dict[str, float]:
    return {str(label): float(values[idx]) for idx, label in enumerate(labels)}


def _safe_get_memory_features(raw_sample: Mapping[str, Any]) -> Dict[str, Any]:
    record = raw_sample.get("record", {}) if isinstance(raw_sample.get("record"), dict) else {}
    memory_features = record.get("memory_features", {}) if isinstance(record.get("memory_features"), dict) else {}
    return dict(memory_features)


def _source_session_from_sample(raw_sample: Mapping[str, Any]) -> str:
    record = raw_sample.get("record", {}) if isinstance(raw_sample.get("record"), dict) else {}
    source_run_dir = str(record.get("source_run_dir") or "").strip()
    if not source_run_dir:
        return ""
    path = Path(source_run_dir)
    for parent in [path, *path.parents]:
        if parent.name.startswith("memory_episode_"):
            return parent.name
    return ""


def _argmax(values: Sequence[float]) -> int:
    if not values:
        return -1
    return max(range(len(values)), key=lambda idx: float(values[idx]))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def export_representation_embeddings(
    *,
    checkpoint_path: Path | str,
    export_dir: Path | str,
    output_dir: Path | str,
    split: str = "all",
    batch_size: int = 32,
    device: str = "cpu",
    num_workers: int = 0,
) -> Dict[str, Any]:
    _require_runtime()

    checkpoint_file = Path(checkpoint_path).resolve()
    export_root = Path(export_dir).resolve()
    output_root = Path(output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint = _load_checkpoint(checkpoint_file, str(device))
    model_config = StudentModelConfig(**checkpoint["model_config"])
    model = EntryRepresentationStudent(model_config)
    model.load_state_dict(checkpoint["model_state_dict"])

    run_device = torch.device(str(device))
    model.to(run_device)
    model.eval()

    split_name = str(split).strip().lower()
    raw_samples, examples = _load_requested_examples(export_root, split_name, top_k=int(model_config.top_k))
    dataloader = DataLoader(
        TensorizedEvalDataset(examples),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=_collate_examples,
    )

    embeddings: List[List[float]] = []
    metadata_rows: List[Dict[str, Any]] = []
    cursor = 0

    state_no_entry_idx = TARGET_CONDITIONED_STATE_LABELS.index("target_house_no_entry_after_full_coverage")
    subgoal_no_entry_idx = TARGET_CONDITIONED_SUBGOAL_LABELS.index("complete_no_entry_search")
    candidate_labels = _candidate_label_names(top_k=int(model_config.top_k))

    with torch.no_grad():
        for raw_batch in dataloader:
            batch = _move_batch_to_device(raw_batch, run_device)
            outputs = model(
                batch["global_features"],
                batch["candidate_features"],
                batch["candidate_valid_mask"],
            )

            z_batch = outputs["z_entry"].detach().cpu().tolist()
            prob_batches: Dict[str, List[List[float]]] = {}
            for head_name, logits_key, _label_key, _labels in HEAD_SPECS:
                prob_batches[head_name] = torch.softmax(outputs[logits_key], dim=-1).detach().cpu().tolist()
            prob_batches["target_conditioned_target_candidate_id"] = (
                torch.softmax(outputs["target_candidate_logits"], dim=-1).detach().cpu().tolist()
            )

            batch_size_actual = len(z_batch)
            for local_idx in range(batch_size_actual):
                raw_sample = raw_samples[cursor + local_idx]
                example = examples[cursor + local_idx]
                labels = example["labels"]
                record = raw_sample.get("record", {}) if isinstance(raw_sample.get("record"), dict) else {}
                paths = raw_sample.get("paths", {}) if isinstance(raw_sample.get("paths"), dict) else {}
                memory_features = _safe_get_memory_features(raw_sample)

                row: Dict[str, Any] = {
                    "sample_id": str(example.get("sample_id") or raw_sample.get("sample_id") or ""),
                    "split": str(example.get("split") or raw_sample.get("split") or record.get("split") or ""),
                    "representation_index": len(embeddings),
                    "source_session": _source_session_from_sample(raw_sample),
                    "source_run_dir": str(record.get("source_run_dir") or ""),
                    "task_label": str(record.get("task_label") or ""),
                    "target_house_id": record.get("target_house_id", example.get("metadata", {}).get("target_house_id")),
                    "current_house_id": example.get("metadata", {}).get("current_house_id"),
                    "memory_available": example.get("metadata", {}).get("memory_available"),
                    "memory_source": example.get("metadata", {}).get("memory_source"),
                    "entry_search_status": memory_features.get("entry_search_status", ""),
                    "raw_entry_search_status": memory_features.get("raw_entry_search_status", ""),
                    "no_entry_after_full_coverage": bool(memory_features.get("no_entry_after_full_coverage", 0)),
                    "full_coverage_ready": bool(memory_features.get("full_coverage_ready", 0)),
                    "has_reliable_entry": bool(memory_features.get("has_reliable_entry", 0)),
                    "visited_coverage_ratio": float(memory_features.get("visited_coverage_ratio", 0.0) or 0.0),
                    "observed_coverage_ratio": float(memory_features.get("observed_coverage_ratio", 0.0) or 0.0),
                    "candidate_entry_count": int(memory_features.get("candidate_entry_count", 0) or 0),
                    "best_entry_id": str(memory_features.get("last_best_entry_id") or ""),
                    "best_entry_status": str(memory_features.get("last_best_entry_status") or ""),
                    "paths": {
                        "entry_state_path": str(paths.get("entry_state_path") or ""),
                        "teacher_output_path": str(paths.get("teacher_output_path") or ""),
                        "teacher_validation_path": str(paths.get("teacher_validation_path") or ""),
                        "metadata_path": str(paths.get("metadata_path") or ""),
                    },
                }

                for head_name, _logits_key, label_key, label_names in HEAD_SPECS:
                    probs = prob_batches[head_name][local_idx]
                    pred_id = _argmax(probs)
                    true_id = int(labels.get(label_key, -1))
                    row[f"true_{head_name}"] = _label_or_none(true_id, label_names)
                    row[f"pred_{head_name}"] = _label_or_none(pred_id, label_names)
                    row[f"{head_name}_prob"] = float(probs[pred_id]) if pred_id >= 0 else 0.0
                    row[f"{head_name}_probs"] = _probability_map(probs, label_names)

                candidate_probs = prob_batches["target_conditioned_target_candidate_id"][local_idx]
                candidate_pred_id = _argmax(candidate_probs)
                candidate_true_id = int(labels.get("target_conditioned_target_candidate_id", -1))
                row["true_target_conditioned_target_candidate_id"] = _decode_candidate_label(
                    candidate_true_id,
                    top_k=int(model_config.top_k),
                )
                row["pred_target_conditioned_target_candidate_id"] = _decode_candidate_label(
                    candidate_pred_id,
                    top_k=int(model_config.top_k),
                )
                row["target_conditioned_target_candidate_id_prob"] = (
                    float(candidate_probs[candidate_pred_id]) if candidate_pred_id >= 0 else 0.0
                )
                row["target_conditioned_target_candidate_id_probs"] = _probability_map(
                    candidate_probs,
                    candidate_labels,
                )
                row["no_entry_completion_prob"] = float(
                    prob_batches["target_conditioned_state"][local_idx][state_no_entry_idx]
                )
                row["complete_no_entry_search_prob"] = float(
                    prob_batches["target_conditioned_subgoal"][local_idx][subgoal_no_entry_idx]
                )

                embeddings.append([float(value) for value in z_batch[local_idx]])
                metadata_rows.append(row)

            cursor += batch_size_actual

    embeddings_array = np.asarray(embeddings, dtype=np.float32)
    np.save(output_root / "embeddings.npy", embeddings_array)
    _append_jsonl(output_root / "metadata.jsonl", metadata_rows)

    split_counts: Dict[str, int] = {}
    state_counts: Dict[str, int] = {}
    subgoal_counts: Dict[str, int] = {}
    state_correct = 0
    subgoal_correct = 0
    action_correct = 0
    candidate_correct = 0
    no_entry_true_probs: List[float] = []
    no_entry_false_probs: List[float] = []

    for row in metadata_rows:
        split_counts[str(row.get("split") or "")] = split_counts.get(str(row.get("split") or ""), 0) + 1
        true_state = str(row.get("true_target_conditioned_state") or "")
        true_subgoal = str(row.get("true_target_conditioned_subgoal") or "")
        state_counts[true_state] = state_counts.get(true_state, 0) + 1
        subgoal_counts[true_subgoal] = subgoal_counts.get(true_subgoal, 0) + 1
        state_correct += int(true_state == str(row.get("pred_target_conditioned_state") or ""))
        subgoal_correct += int(true_subgoal == str(row.get("pred_target_conditioned_subgoal") or ""))
        action_correct += int(
            str(row.get("true_target_conditioned_action_hint") or "")
            == str(row.get("pred_target_conditioned_action_hint") or "")
        )
        candidate_correct += int(
            str(row.get("true_target_conditioned_target_candidate_id") or "")
            == str(row.get("pred_target_conditioned_target_candidate_id") or "")
        )
        if true_state == "target_house_no_entry_after_full_coverage":
            no_entry_true_probs.append(float(row.get("no_entry_completion_prob") or 0.0))
        else:
            no_entry_false_probs.append(float(row.get("no_entry_completion_prob") or 0.0))

    sample_count = len(metadata_rows)
    summary = {
        "checkpoint_path": str(checkpoint_file),
        "export_dir": str(export_root),
        "output_dir": str(output_root),
        "split": split_name,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "sample_count": sample_count,
        "embedding_dim": int(embeddings_array.shape[1]) if embeddings_array.ndim == 2 else 0,
        "model_config": checkpoint.get("model_config", {}),
        "split_counts": split_counts,
        "target_conditioned_state_counts": state_counts,
        "target_conditioned_subgoal_counts": subgoal_counts,
        "target_conditioned_state_accuracy": state_correct / max(1, sample_count),
        "target_conditioned_subgoal_accuracy": subgoal_correct / max(1, sample_count),
        "target_conditioned_action_hint_accuracy": action_correct / max(1, sample_count),
        "target_conditioned_target_candidate_id_accuracy": candidate_correct / max(1, sample_count),
        "no_entry_true_count": len(no_entry_true_probs),
        "no_entry_false_count": len(no_entry_false_probs),
        "mean_no_entry_prob_when_true": _mean(no_entry_true_probs),
        "mean_no_entry_prob_when_false": _mean(no_entry_false_probs),
        "files": {
            "embeddings_npy": str(output_root / "embeddings.npy"),
            "metadata_jsonl": str(output_root / "metadata.jsonl"),
            "embedding_summary_json": str(output_root / "embedding_summary.json"),
        },
    }
    _write_json(output_root / "embedding_summary.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export z_entry embeddings from a trained Phase 2.5 representation distillation model."
    )
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--export_dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--run_name", type=str, default=f"embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--split", type=str, default="all", choices=("train", "val", "all"))
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = args.output_dir.resolve() if args.output_dir else args.output_root.resolve() / str(args.run_name)
    summary = export_representation_embeddings(
        checkpoint_path=args.checkpoint_path.resolve(),
        export_dir=args.export_dir.resolve(),
        output_dir=output_dir,
        split=str(args.split),
        batch_size=int(args.batch_size),
        device=str(args.device),
        num_workers=int(args.num_workers),
    )
    print(
        "[rep-distill-export] done: "
        f"split={summary['split']} samples={summary['sample_count']} "
        f"dim={summary['embedding_dim']} output={summary['output_dir']}"
    )


if __name__ == "__main__":
    main()
