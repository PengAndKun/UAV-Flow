from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .dataset import DistillationDataset
from .feature_builder import build_training_example
from .label_schema import (
    ACTION_HINT_SPACE,
    ENTRY_STATE_SPACE,
    SUBGOAL_SPACE,
    TARGET_ACTION_HINT_SPACE,
    TARGET_STATE_SPACE,
    TARGET_SUBGOAL_SPACE,
    target_candidate_num_classes,
)
from .losses import MultiHeadLossConfig, compute_balanced_class_weights, compute_multi_head_loss
from .model import EntryRepresentationStudent, StudentModelConfig, build_default_config

try:
    import torch
    from torch import Tensor
    from torch.optim import AdamW
    from torch.utils.data import DataLoader, Dataset
except ImportError:  # pragma: no cover - local env may not have torch
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    AdamW = None  # type: ignore[assignment]
    DataLoader = Any  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or AdamW is None:
        raise ImportError(
            "PyTorch is required for phase2_5_representation_distillation.trainer. "
            "Install torch in the training environment before starting training."
        )


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass(frozen=True)
class DistillationTrainerConfig:
    export_dir: str
    output_root: str
    run_name: str
    top_k: int = 3
    batch_size: int = 16
    epochs: int = 80
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 12
    device: str = "cpu"
    num_workers: int = 0
    dropout: float = 0.1
    global_hidden_dim: int = 64
    candidate_hidden_dim: int = 64
    fused_hidden_dim: int = 256
    representation_dim: int = 128
    save_every_epoch: bool = False
    stage1_epochs: int = 5
    stage2_epochs: int = 12


if torch is not None:

    class TensorizedTrainingDataset(Dataset):
        def __init__(self, examples: Sequence[Dict[str, Any]]) -> None:
            self.examples = list(examples)

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> Dict[str, Any]:
            return self.examples[index]


def _load_examples(dataset: DistillationDataset, *, top_k: int) -> List[Dict[str, Any]]:
    return [build_training_example(sample, top_k=top_k) for sample in dataset]


def _collate_examples(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    _require_torch()
    if not batch:
        raise ValueError("batch must not be empty")

    return {
        "sample_ids": [str(item["sample_id"]) for item in batch],
        "global_features": torch.tensor(
            [item["global_features"] for item in batch],
            dtype=torch.float32,
        ),
        "candidate_features": torch.tensor(
            [item["candidate_features"] for item in batch],
            dtype=torch.float32,
        ),
        "candidate_valid_mask": torch.tensor(
            [item["candidate_valid_mask"] for item in batch],
            dtype=torch.float32,
        ),
        "labels": {
            key: torch.tensor([item["labels"][key] for item in batch], dtype=torch.long)
            for key in batch[0]["labels"].keys()
        },
        "label_masks": {
            key: torch.tensor([item["label_masks"][key] for item in batch], dtype=torch.float32)
            for key in batch[0]["label_masks"].keys()
        },
        "label_confidences": {
            key: torch.tensor([item["label_confidences"][key] for item in batch], dtype=torch.float32)
            for key in batch[0]["label_confidences"].keys()
        },
        "metadata": [dict(item.get("metadata", {})) for item in batch],
    }


def _move_batch_to_device(batch: Dict[str, Any], device: "torch.device") -> Dict[str, Any]:
    _require_torch()
    moved = dict(batch)
    moved["global_features"] = batch["global_features"].to(device)
    moved["candidate_features"] = batch["candidate_features"].to(device)
    moved["candidate_valid_mask"] = batch["candidate_valid_mask"].to(device)
    moved["labels"] = {key: value.to(device) for key, value in batch["labels"].items()}
    moved["label_masks"] = {key: value.to(device) for key, value in batch["label_masks"].items()}
    moved["label_confidences"] = {
        key: value.to(device) for key, value in batch["label_confidences"].items()
    }
    return moved


def _infer_model_config(examples: Sequence[Mapping[str, Any]], trainer_cfg: DistillationTrainerConfig) -> StudentModelConfig:
    if not examples:
        raise ValueError("Cannot infer model config from an empty example list")
    first = examples[0]
    global_dim = len(first["global_features"])
    candidate_dim = len(first["candidate_features"][0])
    return StudentModelConfig(
        global_input_dim=global_dim,
        candidate_input_dim=candidate_dim,
        top_k=int(trainer_cfg.top_k),
        global_hidden_dim=int(trainer_cfg.global_hidden_dim),
        candidate_hidden_dim=int(trainer_cfg.candidate_hidden_dim),
        fused_hidden_dim=int(trainer_cfg.fused_hidden_dim),
        representation_dim=int(trainer_cfg.representation_dim),
        dropout=float(trainer_cfg.dropout),
    )


def _build_class_weights(train_examples: Sequence[Mapping[str, Any]], device: "torch.device") -> Dict[str, "Tensor"]:
    _require_torch()
    return {
        "entry_state": compute_balanced_class_weights(
            (item["labels"]["entry_state"] for item in train_examples),
            num_classes=ENTRY_STATE_SPACE.size,
        ).to(device),
        "subgoal": compute_balanced_class_weights(
            (item["labels"]["subgoal"] for item in train_examples),
            num_classes=SUBGOAL_SPACE.size,
        ).to(device),
        "action_hint": compute_balanced_class_weights(
            (item["labels"]["action_hint"] for item in train_examples),
            num_classes=ACTION_HINT_SPACE.size,
        ).to(device),
        "target_conditioned_state": compute_balanced_class_weights(
            (
                item["labels"]["target_conditioned_state"]
                for item in train_examples
                if item["label_masks"]["target_conditioned_teacher_available"] > 0
            ),
            num_classes=TARGET_STATE_SPACE.size,
        ).to(device),
        "target_conditioned_subgoal": compute_balanced_class_weights(
            (
                item["labels"]["target_conditioned_subgoal"]
                for item in train_examples
                if item["label_masks"]["target_conditioned_teacher_available"] > 0
            ),
            num_classes=TARGET_SUBGOAL_SPACE.size,
        ).to(device),
        "target_conditioned_action_hint": compute_balanced_class_weights(
            (
                item["labels"]["target_conditioned_action_hint"]
                for item in train_examples
                if item["label_masks"]["target_conditioned_teacher_available"] > 0
            ),
            num_classes=TARGET_ACTION_HINT_SPACE.size,
        ).to(device),
        "target_conditioned_target_candidate_id": compute_balanced_class_weights(
            (
                item["labels"]["target_conditioned_target_candidate_id"]
                for item in train_examples
                if item["label_masks"]["target_conditioned_teacher_available"] > 0
            ),
            num_classes=target_candidate_num_classes(top_k=len(train_examples[0]["candidate_features"])),
        ).to(device),
    }


def _curriculum_stage_name(epoch_index: int, cfg: DistillationTrainerConfig) -> str:
    stage1_end = max(1, int(cfg.stage1_epochs))
    stage2_end = max(stage1_end, int(cfg.stage2_epochs))
    if epoch_index <= stage1_end:
        return "stage1_state_only"
    if epoch_index <= stage2_end:
        return "stage2_add_subgoals"
    return "stage3_full_multihead"


def _loss_config_for_epoch(epoch_index: int, cfg: DistillationTrainerConfig) -> MultiHeadLossConfig:
    stage_name = _curriculum_stage_name(epoch_index, cfg)
    if stage_name == "stage1_state_only":
        return MultiHeadLossConfig(
            entry_state=1.0,
            subgoal=0.0,
            action_hint=0.0,
            target_state=1.2,
            target_subgoal=0.0,
            target_action=0.0,
            target_candidate=0.0,
        )
    if stage_name == "stage2_add_subgoals":
        return MultiHeadLossConfig(
            entry_state=1.0,
            subgoal=1.0,
            action_hint=0.0,
            target_state=1.2,
            target_subgoal=1.2,
            target_action=0.0,
            target_candidate=0.0,
        )
    return MultiHeadLossConfig()


def _selection_metric(val_metrics: Mapping[str, float]) -> float:
    return (
        1.60 * (1.0 - float(val_metrics["acc_target_state"]))
        + 1.30 * (1.0 - float(val_metrics["acc_target_subgoal"]))
        + 0.70 * (1.0 - float(val_metrics["acc_target_action"]))
        + 0.70 * (1.0 - float(val_metrics["acc_target_candidate"]))
        + 0.80 * (1.0 - float(val_metrics["acc_entry_state"]))
        + 0.40 * (1.0 - float(val_metrics["acc_subgoal"]))
        + 0.20 * (1.0 - float(val_metrics["acc_action_hint"]))
        + 0.10 * float(val_metrics["total_loss"])
    )


def _masked_accuracy(logits: "Tensor", targets: "Tensor", mask: "Tensor") -> float:
    _require_torch()
    valid = mask > 0
    if int(valid.sum().item()) == 0:
        return 0.0
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets) & valid
    return float(correct.sum().item() / max(1, int(valid.sum().item())))


def _run_epoch(
    *,
    model: EntryRepresentationStudent,
    dataloader: "DataLoader",
    optimizer: Optional["AdamW"],
    device: "torch.device",
    class_weights: Mapping[str, "Tensor"],
    loss_config: MultiHeadLossConfig,
    training: bool,
) -> Dict[str, float]:
    _require_torch()
    model.train(mode=training)

    loss_totals: Dict[str, float] = {
        "total_loss": 0.0,
        "loss_entry_state": 0.0,
        "loss_subgoal": 0.0,
        "loss_action_hint": 0.0,
        "loss_target_state": 0.0,
        "loss_target_subgoal": 0.0,
        "loss_target_action": 0.0,
        "loss_target_candidate": 0.0,
    }
    metric_totals: Dict[str, float] = {
        "acc_entry_state": 0.0,
        "acc_subgoal": 0.0,
        "acc_action_hint": 0.0,
        "acc_target_state": 0.0,
        "acc_target_subgoal": 0.0,
        "acc_target_action": 0.0,
        "acc_target_candidate": 0.0,
    }

    num_batches = 0
    for raw_batch in dataloader:
        batch = _move_batch_to_device(raw_batch, device)
        outputs = model(
            batch["global_features"],
            batch["candidate_features"],
            batch["candidate_valid_mask"],
        )
        loss_dict = compute_multi_head_loss(
            outputs,
            batch["labels"],
            batch["label_masks"],
            config=loss_config,
            class_weights=class_weights,
        )

        if training:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            loss_dict["total_loss"].backward()
            optimizer.step()

        teacher_mask = batch["label_masks"]["teacher_available"]
        target_mask = batch["label_masks"]["target_conditioned_teacher_available"]

        metric_totals["acc_entry_state"] += _masked_accuracy(
            outputs["entry_state_logits"], batch["labels"]["entry_state"], teacher_mask
        )
        metric_totals["acc_subgoal"] += _masked_accuracy(
            outputs["subgoal_logits"], batch["labels"]["subgoal"], teacher_mask
        )
        metric_totals["acc_action_hint"] += _masked_accuracy(
            outputs["action_hint_logits"], batch["labels"]["action_hint"], teacher_mask
        )
        metric_totals["acc_target_state"] += _masked_accuracy(
            outputs["target_state_logits"], batch["labels"]["target_conditioned_state"], target_mask
        )
        metric_totals["acc_target_subgoal"] += _masked_accuracy(
            outputs["target_subgoal_logits"], batch["labels"]["target_conditioned_subgoal"], target_mask
        )
        metric_totals["acc_target_action"] += _masked_accuracy(
            outputs["target_action_logits"], batch["labels"]["target_conditioned_action_hint"], target_mask
        )
        metric_totals["acc_target_candidate"] += _masked_accuracy(
            outputs["target_candidate_logits"],
            batch["labels"]["target_conditioned_target_candidate_id"],
            target_mask,
        )

        for key in loss_totals.keys():
            loss_totals[key] += float(loss_dict[key].detach().cpu().item())
        num_batches += 1

    if num_batches == 0:
        raise RuntimeError("Dataloader produced zero batches")

    summary = {key: value / num_batches for key, value in loss_totals.items()}
    summary.update({key: value / num_batches for key, value in metric_totals.items()})
    return summary


def _save_checkpoint(
    path: Path,
    *,
    model: EntryRepresentationStudent,
    optimizer: Optional["AdamW"],
    epoch_index: int,
    trainer_config: DistillationTrainerConfig,
    model_config: StudentModelConfig,
    metrics: Dict[str, Any],
) -> None:
    _require_torch()
    payload = {
        "epoch": int(epoch_index),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "trainer_config": asdict(trainer_config),
        "model_config": asdict(model_config),
        "metrics": metrics,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(payload, path)


def train_representation_distillation(config: DistillationTrainerConfig) -> Dict[str, Any]:
    _require_torch()
    export_dir = Path(config.export_dir).resolve()
    output_root = Path(config.output_root).resolve()
    output_dir = output_root / str(config.run_name)
    checkpoints_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = DistillationDataset.from_export_dir(export_dir, "train", lazy=False)
    val_dataset = DistillationDataset.from_export_dir(export_dir, "val", lazy=False)

    train_examples = _load_examples(train_dataset, top_k=int(config.top_k))
    val_examples = _load_examples(val_dataset, top_k=int(config.top_k))
    if not train_examples:
        raise RuntimeError("Training set is empty after loading examples")
    if not val_examples:
        raise RuntimeError("Validation set is empty after loading examples")

    model_config = _infer_model_config(train_examples, config)
    device = torch.device(str(config.device))
    model = EntryRepresentationStudent(model_config).to(device)

    train_loader = DataLoader(
        TensorizedTrainingDataset(train_examples),
        batch_size=int(config.batch_size),
        shuffle=True,
        num_workers=int(config.num_workers),
        collate_fn=_collate_examples,
    )
    val_loader = DataLoader(
        TensorizedTrainingDataset(val_examples),
        batch_size=int(config.batch_size),
        shuffle=False,
        num_workers=int(config.num_workers),
        collate_fn=_collate_examples,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    class_weights = _build_class_weights(train_examples, device)

    history: List[Dict[str, Any]] = []
    best_metric = float("inf")
    best_epoch = -1
    best_epoch_record: Dict[str, Any] | None = None
    patience_counter = 0

    _write_json(
        output_dir / "run_config.json",
        {
            "trainer_config": asdict(config),
            "model_config": asdict(model_config),
            "train_examples": len(train_examples),
            "val_examples": len(val_examples),
            "selection_metric_mode": "target_priority_weighted_error_plus_loss",
        },
    )

    for epoch in range(1, int(config.epochs) + 1):
        stage_name = _curriculum_stage_name(epoch, config)
        loss_config = _loss_config_for_epoch(epoch, config)
        train_metrics = _run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            class_weights=class_weights,
            loss_config=loss_config,
            training=True,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model=model,
                dataloader=val_loader,
                optimizer=None,
                device=device,
                class_weights=class_weights,
                loss_config=loss_config,
                training=False,
            )

        current_selection_metric = _selection_metric(val_metrics)
        epoch_record = {
            "epoch": epoch,
            "stage_name": stage_name,
            "loss_config": asdict(loss_config),
            "selection_metric": current_selection_metric,
            "train": train_metrics,
            "val": val_metrics,
        }
        history.append(epoch_record)
        _write_json(output_dir / "metrics.json", {"history": history})

        _save_checkpoint(
            checkpoints_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch_index=epoch,
            trainer_config=config,
            model_config=model_config,
            metrics=epoch_record,
        )
        if bool(config.save_every_epoch):
            _save_checkpoint(
                checkpoints_dir / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                epoch_index=epoch,
                trainer_config=config,
                model_config=model_config,
                metrics=epoch_record,
            )

        if current_selection_metric < best_metric:
            best_metric = current_selection_metric
            best_epoch = epoch
            best_epoch_record = epoch_record
            patience_counter = 0
            _save_checkpoint(
                checkpoints_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch_index=epoch,
                trainer_config=config,
                model_config=model_config,
                metrics=epoch_record,
            )
        else:
            patience_counter += 1

        if patience_counter >= int(config.patience):
            break

    summary = {
        "output_dir": str(output_dir),
        "checkpoints_dir": str(checkpoints_dir),
        "best_epoch": best_epoch,
        "best_selection_metric": best_metric,
        "selection_metric_mode": "target_priority_weighted_error_plus_loss",
        "epochs_completed": len(history),
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "history": history,
    }
    if best_epoch_record is not None:
        summary["best_epoch_stage_name"] = best_epoch_record.get("stage_name")
        summary["best_epoch_val_total_loss"] = float(best_epoch_record["val"]["total_loss"])
        summary["best_epoch_val_acc_target_state"] = float(best_epoch_record["val"]["acc_target_state"])
        summary["best_epoch_val_acc_target_subgoal"] = float(best_epoch_record["val"]["acc_target_subgoal"])
        summary["best_epoch_val_acc_entry_state"] = float(best_epoch_record["val"]["acc_entry_state"])
    _write_json(output_dir / "training_summary.json", summary)
    return summary
