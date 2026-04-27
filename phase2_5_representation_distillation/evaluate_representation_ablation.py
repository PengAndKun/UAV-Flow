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
        _classification_report,
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
        target_candidate_num_classes,
    )
    from phase2_5_representation_distillation.model import EntryRepresentationStudent, StudentModelConfig  # type: ignore
else:
    from .dataset import DistillationDataset
    from .evaluator import (
        TensorizedEvalDataset,
        _classification_report,
        _collate_examples,
        _load_checkpoint,
        _move_batch_to_device,
    )
    from .feature_builder import build_training_example
    from .label_schema import (
        ACTION_HINT_LABELS,
        ENTRY_STATE_LABELS,
        SUBGOAL_LABELS,
        TARGET_CONDITIONED_ACTION_HINT_LABELS,
        TARGET_CONDITIONED_STATE_LABELS,
        TARGET_CONDITIONED_SUBGOAL_LABELS,
        target_candidate_num_classes,
    )
    from .model import EntryRepresentationStudent, StudentModelConfig

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
DEFAULT_OUTPUT_ROOT = Path(r"E:\github\UAV-Flow\phase2_5_representation_distillation\ablations")


HEAD_SPECS: Tuple[Tuple[str, str, str, Tuple[str, ...], str], ...] = (
    ("entry_state", "entry_state_logits", "entry_state", ENTRY_STATE_LABELS, "teacher_available"),
    ("subgoal", "subgoal_logits", "subgoal", SUBGOAL_LABELS, "teacher_available"),
    ("action_hint", "action_hint_logits", "action_hint", ACTION_HINT_LABELS, "teacher_available"),
    (
        "target_conditioned_state",
        "target_state_logits",
        "target_conditioned_state",
        TARGET_CONDITIONED_STATE_LABELS,
        "target_conditioned_teacher_available",
    ),
    (
        "target_conditioned_subgoal",
        "target_subgoal_logits",
        "target_conditioned_subgoal",
        TARGET_CONDITIONED_SUBGOAL_LABELS,
        "target_conditioned_teacher_available",
    ),
    (
        "target_conditioned_action_hint",
        "target_action_logits",
        "target_conditioned_action_hint",
        TARGET_CONDITIONED_ACTION_HINT_LABELS,
        "target_conditioned_teacher_available",
    ),
)


def _require_torch() -> None:
    if torch is None or DataLoader is None:
        raise ImportError("PyTorch is required to run representation ablation evaluation.")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else 0.0


def _copy_example(example: Mapping[str, Any]) -> Dict[str, Any]:
    copied = dict(example)
    copied["global_features"] = [float(value) for value in example.get("global_features", [])]
    copied["candidate_features"] = [
        [float(value) for value in candidate]
        for candidate in example.get("candidate_features", [])
        if isinstance(candidate, list)
    ]
    copied["candidate_valid_mask"] = [float(value) for value in example.get("candidate_valid_mask", [])]
    copied["labels"] = dict(example.get("labels", {}))
    copied["label_masks"] = dict(example.get("label_masks", {}))
    copied["label_confidences"] = dict(example.get("label_confidences", {}))
    copied["metadata"] = dict(example.get("metadata", {}))
    copied["global_feature_names"] = list(example.get("global_feature_names", []))
    copied["candidate_feature_names"] = list(example.get("candidate_feature_names", []))
    copied["memory_feature_names"] = list(example.get("memory_feature_names", []))
    copied["memory_features"] = [float(value) for value in example.get("memory_features", [])]
    return copied


def _apply_ablation(example: Mapping[str, Any], ablation: str) -> Dict[str, Any]:
    output = _copy_example(example)
    mode = str(ablation).strip().lower()
    if mode == "none":
        return output

    if mode == "zero_memory":
        names = output.get("global_feature_names", [])
        global_features = output.get("global_features", [])
        for idx, name in enumerate(names):
            if str(name).startswith("memory_") and idx < len(global_features):
                global_features[idx] = 0.0
        output["memory_features"] = [0.0 for _ in output.get("memory_features", [])]
        output.setdefault("metadata", {})["memory_available"] = 0
        output.setdefault("metadata", {})["memory_source"] = "ablated_zero_memory"
        return output

    if mode == "zero_candidates":
        output["candidate_features"] = [
            [0.0 for _ in candidate]
            for candidate in output.get("candidate_features", [])
            if isinstance(candidate, list)
        ]
        output["candidate_valid_mask"] = [0.0 for _ in output.get("candidate_valid_mask", [])]
        return output

    if mode == "zero_memory_and_candidates":
        without_memory = _apply_ablation(output, "zero_memory")
        return _apply_ablation(without_memory, "zero_candidates")

    raise ValueError(f"Unknown ablation mode: {ablation}")


def _load_examples(export_root: Path, split: str, *, top_k: int) -> List[Dict[str, Any]]:
    dataset = DistillationDataset.from_export_dir(export_root, split, lazy=False)
    return [build_training_example(sample, top_k=top_k) for sample in dataset]


def _collect_metrics(
    *,
    model: "EntryRepresentationStudent",
    examples: Sequence[Mapping[str, Any]],
    device: "torch.device",
    batch_size: int,
    num_workers: int,
    top_k: int,
) -> Dict[str, Any]:
    dataloader = DataLoader(
        TensorizedEvalDataset(list(examples)),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=_collate_examples,
    )

    aggregated: Dict[str, Dict[str, List[int]]] = {
        name: {"true": [], "pred": []}
        for name, *_rest in HEAD_SPECS
    }
    aggregated["target_conditioned_target_candidate_id"] = {"true": [], "pred": []}
    no_entry_idx = TARGET_CONDITIONED_STATE_LABELS.index("target_house_no_entry_after_full_coverage")
    true_no_entry_probs: List[float] = []
    false_no_entry_probs: List[float] = []
    false_high_samples: List[Dict[str, Any]] = []
    true_low_samples: List[Dict[str, Any]] = []

    sample_cursor = 0
    with torch.no_grad():
        for raw_batch in dataloader:
            batch = _move_batch_to_device(raw_batch, device)
            outputs = model(
                batch["global_features"],
                batch["candidate_features"],
                batch["candidate_valid_mask"],
            )
            target_mask = batch["label_masks"]["target_conditioned_teacher_available"] > 0

            for head_name, logits_key, label_key, labels, mask_key in HEAD_SPECS:
                mask_tensor = batch["label_masks"][mask_key] > 0
                preds = torch.argmax(outputs[logits_key], dim=-1)
                valid_idx = mask_tensor.nonzero(as_tuple=False).flatten()
                if int(valid_idx.numel()) == 0:
                    continue
                aggregated[head_name]["true"].extend(
                    batch["labels"][label_key][valid_idx].detach().cpu().tolist()
                )
                aggregated[head_name]["pred"].extend(preds[valid_idx].detach().cpu().tolist())

            candidate_preds = torch.argmax(outputs["target_candidate_logits"], dim=-1)
            candidate_idx = target_mask.nonzero(as_tuple=False).flatten()
            if int(candidate_idx.numel()) > 0:
                aggregated["target_conditioned_target_candidate_id"]["true"].extend(
                    batch["labels"]["target_conditioned_target_candidate_id"][candidate_idx].detach().cpu().tolist()
                )
                aggregated["target_conditioned_target_candidate_id"]["pred"].extend(
                    candidate_preds[candidate_idx].detach().cpu().tolist()
                )

            state_probs = torch.softmax(outputs["target_state_logits"], dim=-1).detach().cpu()
            target_labels = batch["labels"]["target_conditioned_state"].detach().cpu()
            batch_sample_ids = list(raw_batch["sample_ids"])
            for local_idx in range(len(batch_sample_ids)):
                prob = float(state_probs[local_idx, no_entry_idx])
                is_no_entry = int(target_labels[local_idx]) == no_entry_idx
                if is_no_entry:
                    true_no_entry_probs.append(prob)
                    if prob < 0.5:
                        true_low_samples.append({"sample_id": batch_sample_ids[local_idx], "prob": prob})
                else:
                    false_no_entry_probs.append(prob)
                    if prob >= 0.5:
                        false_high_samples.append(
                            {
                                "sample_id": batch_sample_ids[local_idx],
                                "prob": prob,
                                "target_state_label_id": int(target_labels[local_idx]),
                            }
                        )
            sample_cursor += len(batch_sample_ids)

    reports: Dict[str, Any] = {}
    for head_name, _logits_key, _label_key, label_names, _mask_key in HEAD_SPECS:
        reports[head_name] = _classification_report(
            aggregated[head_name]["true"],
            aggregated[head_name]["pred"],
            label_names,
        )
    candidate_class_names = [f"candidate_{idx}" for idx in range(int(top_k))] + ["null_candidate"]
    reports["target_conditioned_target_candidate_id"] = _classification_report(
        aggregated["target_conditioned_target_candidate_id"]["true"],
        aggregated["target_conditioned_target_candidate_id"]["pred"],
        candidate_class_names,
    )

    return {
        "sample_count": len(examples),
        "reports": reports,
        "no_entry": {
            "true_count": len(true_no_entry_probs),
            "false_count": len(false_no_entry_probs),
            "mean_prob_when_true": _mean(true_no_entry_probs),
            "mean_prob_when_false": _mean(false_no_entry_probs),
            "separation_margin_mean": _mean(true_no_entry_probs) - _mean(false_no_entry_probs),
            "false_high_prob_count": len(false_high_samples),
            "true_low_prob_count": len(true_low_samples),
            "false_high_prob_samples": false_high_samples[:30],
            "true_low_prob_samples": true_low_samples[:30],
        },
    }


def evaluate_representation_ablation(
    *,
    checkpoint_path: Path | str,
    export_dir: Path | str,
    output_dir: Path | str,
    split: str = "val",
    ablations: Sequence[str] = ("none", "zero_memory"),
    batch_size: int = 32,
    device: str = "cpu",
    num_workers: int = 0,
) -> Dict[str, Any]:
    _require_torch()
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

    base_examples = _load_examples(export_root, str(split), top_k=int(model_config.top_k))
    results: Dict[str, Any] = {}
    for mode in ablations:
        ablated_examples = [_apply_ablation(example, mode) for example in base_examples]
        results[mode] = _collect_metrics(
            model=model,
            examples=ablated_examples,
            device=run_device,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            top_k=int(model_config.top_k),
        )

    baseline = results.get("none", {})
    baseline_reports = baseline.get("reports", {}) if isinstance(baseline.get("reports"), dict) else {}
    deltas: Dict[str, Any] = {}
    for mode, payload in results.items():
        if mode == "none":
            continue
        mode_reports = payload.get("reports", {}) if isinstance(payload.get("reports"), dict) else {}
        deltas[mode] = {}
        for key in (
            "target_conditioned_state",
            "target_conditioned_subgoal",
            "target_conditioned_action_hint",
            "target_conditioned_target_candidate_id",
        ):
            base_acc = float(baseline_reports.get(key, {}).get("accuracy", 0.0))
            mode_acc = float(mode_reports.get(key, {}).get("accuracy", 0.0))
            deltas[mode][f"{key}_accuracy_delta"] = mode_acc - base_acc
        base_no_entry = baseline.get("no_entry", {}) if isinstance(baseline.get("no_entry"), dict) else {}
        mode_no_entry = payload.get("no_entry", {}) if isinstance(payload.get("no_entry"), dict) else {}
        deltas[mode]["no_entry_separation_margin_delta"] = float(
            mode_no_entry.get("separation_margin_mean", 0.0)
        ) - float(base_no_entry.get("separation_margin_mean", 0.0))

    summary = {
        "checkpoint_path": str(checkpoint_file),
        "export_dir": str(export_root),
        "output_dir": str(output_root),
        "split": str(split),
        "ablations": list(ablations),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "model_config": checkpoint.get("model_config", {}),
        "results": results,
        "deltas_vs_none": deltas,
    }
    _write_json(output_root / "ablation_summary.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run representation ablation evaluation for memory-aware distillation checkpoints."
    )
    parser.add_argument("--checkpoint_path", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--export_dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--run_name", type=str, default=f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"))
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=["none", "zero_memory"],
        choices=("none", "zero_memory", "zero_candidates", "zero_memory_and_candidates"),
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = args.output_dir.resolve() if args.output_dir else args.output_root.resolve() / str(args.run_name)
    summary = evaluate_representation_ablation(
        checkpoint_path=args.checkpoint_path.resolve(),
        export_dir=args.export_dir.resolve(),
        output_dir=output_dir,
        split=str(args.split),
        ablations=[str(value) for value in args.ablations],
        batch_size=int(args.batch_size),
        device=str(args.device),
        num_workers=int(args.num_workers),
    )

    none_result = summary["results"].get("none", {})
    zero_memory = summary["results"].get("zero_memory", {})
    none_state = none_result.get("reports", {}).get("target_conditioned_state", {}).get("accuracy", 0.0)
    zero_state = zero_memory.get("reports", {}).get("target_conditioned_state", {}).get("accuracy", 0.0)
    print(
        "[rep-distill-ablation] done: "
        f"split={summary['split']} "
        f"none_state_acc={float(none_state):.4f} "
        f"zero_memory_state_acc={float(zero_state):.4f} "
        f"output={summary['output_dir']}"
    )


if __name__ == "__main__":
    main()
