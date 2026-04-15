from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .dataset import DistillationDataset
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
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - local env may not have torch
    torch = None  # type: ignore[assignment]
    DataLoader = Any  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is required for phase2_5_representation_distillation.evaluator. "
            "Install torch in the training environment before running evaluation."
        )


def _load_checkpoint(path: Path, device: str) -> Dict[str, Any]:
    _require_torch()
    try:
        return torch.load(path, map_location=str(device), weights_only=True)
    except TypeError:
        return torch.load(path, map_location=str(device))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if torch is not None:

    class TensorizedEvalDataset(Dataset):
        def __init__(self, examples: Sequence[Dict[str, Any]]) -> None:
            self.examples = list(examples)

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            return self.examples[index]


def _collate_examples(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    _require_torch()
    if not batch:
        raise ValueError("batch must not be empty")
    return {
        "sample_ids": [str(item["sample_id"]) for item in batch],
        "global_features": torch.tensor([item["global_features"] for item in batch], dtype=torch.float32),
        "candidate_features": torch.tensor([item["candidate_features"] for item in batch], dtype=torch.float32),
        "candidate_valid_mask": torch.tensor([item["candidate_valid_mask"] for item in batch], dtype=torch.float32),
        "labels": {
            key: torch.tensor([item["labels"][key] for item in batch], dtype=torch.long)
            for key in batch[0]["labels"].keys()
        },
        "label_masks": {
            key: torch.tensor([item["label_masks"][key] for item in batch], dtype=torch.float32)
            for key in batch[0]["label_masks"].keys()
        },
    }


def _move_batch_to_device(batch: Dict[str, Any], device: "torch.device") -> Dict[str, Any]:
    _require_torch()
    return {
        "sample_ids": batch["sample_ids"],
        "global_features": batch["global_features"].to(device),
        "candidate_features": batch["candidate_features"].to(device),
        "candidate_valid_mask": batch["candidate_valid_mask"].to(device),
        "labels": {key: value.to(device) for key, value in batch["labels"].items()},
        "label_masks": {key: value.to(device) for key, value in batch["label_masks"].items()},
    }


def _load_examples(dataset: DistillationDataset, *, top_k: int) -> List[Dict[str, Any]]:
    return [build_training_example(sample, top_k=top_k) for sample in dataset]


def _classification_report(
    true_labels: Sequence[int],
    pred_labels: Sequence[int],
    class_names: Sequence[str],
) -> Dict[str, Any]:
    num_classes = len(class_names)
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for truth, pred in zip(true_labels, pred_labels):
        if 0 <= int(truth) < num_classes and 0 <= int(pred) < num_classes:
            matrix[int(truth)][int(pred)] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    correct = 0
    total = 0
    macro_precision = 0.0
    macro_recall = 0.0
    macro_f1 = 0.0

    for idx, class_name in enumerate(class_names):
        tp = matrix[idx][idx]
        fp = sum(matrix[row][idx] for row in range(num_classes) if row != idx)
        fn = sum(matrix[idx][col] for col in range(num_classes) if col != idx)
        support = sum(matrix[idx])
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall / (precision + recall))
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        macro_precision += precision
        macro_recall += recall
        macro_f1 += f1
        correct += tp
        total += support

    class_count = max(1, num_classes)
    return {
        "accuracy": correct / max(1, total),
        "macro_precision": macro_precision / class_count,
        "macro_recall": macro_recall / class_count,
        "macro_f1": macro_f1 / class_count,
        "support": total,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "class_names": list(class_names),
    }


def _save_confusion_matrix_csv(path: Path, matrix: Sequence[Sequence[int]], class_names: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true/pred", *class_names])
        for class_name, row in zip(class_names, matrix):
            writer.writerow([class_name, *row])


def _save_confusion_matrix_png(path: Path, matrix: Sequence[Sequence[int]], class_names: Sequence[str], title: str) -> None:
    if plt is None or np is None:
        return
    arr = np.array(matrix, dtype=float)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(arr, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            ax.text(j, i, int(arr[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _evaluate_single_head(
    outputs: Sequence[int],
    targets: Sequence[int],
    class_names: Sequence[str],
    output_dir: Path,
    head_name: str,
) -> Dict[str, Any]:
    report = _classification_report(targets, outputs, class_names)
    _write_json(output_dir / f"{head_name}_report.json", report)
    _save_confusion_matrix_csv(
        output_dir / f"{head_name}_confusion_matrix.csv",
        report["confusion_matrix"],
        report["class_names"],
    )
    _save_confusion_matrix_png(
        output_dir / f"{head_name}_confusion_matrix.png",
        report["confusion_matrix"],
        report["class_names"],
        f"{head_name} confusion matrix",
    )
    return report


def evaluate_representation_distillation(
    *,
    checkpoint_path: Path | str,
    export_dir: Path | str,
    output_dir: Path | str,
    split: str = "val",
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
    eval_device = torch.device(str(device))
    model.to(eval_device)
    model.eval()

    split_name = str(split).strip().lower()
    dataset = DistillationDataset.from_export_dir(export_root, split_name, lazy=False)
    examples = _load_examples(dataset, top_k=int(model_config.top_k))
    dataloader = DataLoader(
        TensorizedEvalDataset(examples),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=int(num_workers),
        collate_fn=_collate_examples,
    )

    aggregated: Dict[str, List[int]] = {
        "entry_state_true": [],
        "entry_state_pred": [],
        "subgoal_true": [],
        "subgoal_pred": [],
        "action_hint_true": [],
        "action_hint_pred": [],
        "target_state_true": [],
        "target_state_pred": [],
        "target_subgoal_true": [],
        "target_subgoal_pred": [],
        "target_action_true": [],
        "target_action_pred": [],
        "target_candidate_true": [],
        "target_candidate_pred": [],
    }

    with torch.no_grad():
        for raw_batch in dataloader:
            batch = _move_batch_to_device(raw_batch, eval_device)
            outputs = model(
                batch["global_features"],
                batch["candidate_features"],
                batch["candidate_valid_mask"],
            )

            teacher_mask = batch["label_masks"]["teacher_available"] > 0
            target_mask = batch["label_masks"]["target_conditioned_teacher_available"] > 0

            def collect(name: str, logits_key: str, label_key: str, mask_tensor: "Tensor") -> None:
                preds = torch.argmax(outputs[logits_key], dim=-1)
                valid_idx = mask_tensor.nonzero(as_tuple=False).flatten()
                if int(valid_idx.numel()) == 0:
                    return
                aggregated[f"{name}_true"].extend(
                    batch["labels"][label_key][valid_idx].detach().cpu().tolist()
                )
                aggregated[f"{name}_pred"].extend(preds[valid_idx].detach().cpu().tolist())

            collect("entry_state", "entry_state_logits", "entry_state", teacher_mask)
            collect("subgoal", "subgoal_logits", "subgoal", teacher_mask)
            collect("action_hint", "action_hint_logits", "action_hint", teacher_mask)
            collect("target_state", "target_state_logits", "target_conditioned_state", target_mask)
            collect(
                "target_subgoal",
                "target_subgoal_logits",
                "target_conditioned_subgoal",
                target_mask,
            )
            collect(
                "target_action",
                "target_action_logits",
                "target_conditioned_action_hint",
                target_mask,
            )
            collect(
                "target_candidate",
                "target_candidate_logits",
                "target_conditioned_target_candidate_id",
                target_mask,
            )

    reports = {
        "entry_state": _evaluate_single_head(
            aggregated["entry_state_pred"],
            aggregated["entry_state_true"],
            ENTRY_STATE_LABELS,
            output_root,
            "entry_state",
        ),
        "subgoal": _evaluate_single_head(
            aggregated["subgoal_pred"],
            aggregated["subgoal_true"],
            SUBGOAL_LABELS,
            output_root,
            "subgoal",
        ),
        "action_hint": _evaluate_single_head(
            aggregated["action_hint_pred"],
            aggregated["action_hint_true"],
            ACTION_HINT_LABELS,
            output_root,
            "action_hint",
        ),
        "target_conditioned_state": _evaluate_single_head(
            aggregated["target_state_pred"],
            aggregated["target_state_true"],
            TARGET_CONDITIONED_STATE_LABELS,
            output_root,
            "target_conditioned_state",
        ),
        "target_conditioned_subgoal": _evaluate_single_head(
            aggregated["target_subgoal_pred"],
            aggregated["target_subgoal_true"],
            TARGET_CONDITIONED_SUBGOAL_LABELS,
            output_root,
            "target_conditioned_subgoal",
        ),
        "target_conditioned_action_hint": _evaluate_single_head(
            aggregated["target_action_pred"],
            aggregated["target_action_true"],
            TARGET_CONDITIONED_ACTION_HINT_LABELS,
            output_root,
            "target_conditioned_action_hint",
        ),
        "target_conditioned_target_candidate_id": _evaluate_single_head(
            aggregated["target_candidate_pred"],
            aggregated["target_candidate_true"],
            [f"candidate_{i}" for i in range(target_candidate_num_classes(top_k=int(model_config.top_k)) - 1)]
            + ["null_candidate"],
            output_root,
            "target_conditioned_target_candidate_id",
        ),
    }

    summary = {
        "checkpoint_path": str(checkpoint_file),
        "export_dir": str(export_root),
        "split": split_name,
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "sample_count": len(dataset),
        "reports": reports,
    }
    _write_json(output_root / "evaluation_summary.json", summary)
    return summary
