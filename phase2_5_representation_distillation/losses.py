from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional

try:
    import torch
    import torch.nn.functional as F
    from torch import Tensor
except ImportError:  # pragma: no cover - local env may not have torch
    torch = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]


def _require_torch() -> None:
    if torch is None or F is None:
        raise ImportError(
            "PyTorch is required for phase2_5_representation_distillation.losses. "
            "Install torch in the training environment before computing losses."
        )


@dataclass(frozen=True)
class MultiHeadLossConfig:
    entry_state: float = 1.0
    subgoal: float = 1.0
    action_hint: float = 0.7
    target_state: float = 1.2
    target_subgoal: float = 1.2
    target_action: float = 0.8
    target_candidate: float = 0.8


def compute_balanced_class_weights(
    label_ids: Iterable[int],
    *,
    num_classes: int,
    min_count: int = 1,
    normalize_mean: bool = True,
) -> "Tensor":
    _require_torch()
    counts = torch.zeros(int(num_classes), dtype=torch.float32)
    for value in label_ids:
        idx = int(value)
        if 0 <= idx < int(num_classes):
            counts[idx] += 1.0

    counts = torch.clamp(counts, min=float(min_count))
    weights = counts.sum() / counts
    if normalize_mean:
        weights = weights / weights.mean()
    return weights


def _masked_cross_entropy(
    logits: "Tensor",
    targets: "Tensor",
    mask: "Tensor",
    *,
    class_weight: Optional["Tensor"] = None,
) -> "Tensor":
    _require_torch()
    if logits.ndim != 2:
        raise ValueError(f"logits must be [B, C], got shape={tuple(logits.shape)}")

    safe_targets = targets.long().clone()
    safe_targets = torch.where(mask > 0, safe_targets, torch.zeros_like(safe_targets))
    per_sample = F.cross_entropy(logits, safe_targets, weight=class_weight, reduction="none")
    masked = per_sample * mask.to(per_sample.dtype)
    denom = torch.clamp(mask.sum().to(per_sample.dtype), min=1.0)
    return masked.sum() / denom


def compute_multi_head_loss(
    outputs: Mapping[str, "Tensor"],
    labels: Mapping[str, "Tensor"],
    label_masks: Mapping[str, "Tensor"],
    *,
    config: MultiHeadLossConfig | None = None,
    class_weights: Optional[Mapping[str, "Tensor"]] = None,
) -> Dict[str, "Tensor"]:
    _require_torch()
    cfg = config or MultiHeadLossConfig()
    class_weights = dict(class_weights or {})

    teacher_mask = label_masks["teacher_available"].float()
    target_mask = label_masks["target_conditioned_teacher_available"].float()

    loss_entry_state = _masked_cross_entropy(
        outputs["entry_state_logits"],
        labels["entry_state"],
        teacher_mask,
        class_weight=class_weights.get("entry_state"),
    )
    loss_subgoal = _masked_cross_entropy(
        outputs["subgoal_logits"],
        labels["subgoal"],
        teacher_mask,
        class_weight=class_weights.get("subgoal"),
    )
    loss_action_hint = _masked_cross_entropy(
        outputs["action_hint_logits"],
        labels["action_hint"],
        teacher_mask,
        class_weight=class_weights.get("action_hint"),
    )
    loss_target_state = _masked_cross_entropy(
        outputs["target_state_logits"],
        labels["target_conditioned_state"],
        target_mask,
        class_weight=class_weights.get("target_conditioned_state"),
    )
    loss_target_subgoal = _masked_cross_entropy(
        outputs["target_subgoal_logits"],
        labels["target_conditioned_subgoal"],
        target_mask,
        class_weight=class_weights.get("target_conditioned_subgoal"),
    )
    loss_target_action = _masked_cross_entropy(
        outputs["target_action_logits"],
        labels["target_conditioned_action_hint"],
        target_mask,
        class_weight=class_weights.get("target_conditioned_action_hint"),
    )
    loss_target_candidate = _masked_cross_entropy(
        outputs["target_candidate_logits"],
        labels["target_conditioned_target_candidate_id"],
        target_mask,
        class_weight=class_weights.get("target_conditioned_target_candidate_id"),
    )

    total_loss = (
        cfg.entry_state * loss_entry_state
        + cfg.subgoal * loss_subgoal
        + cfg.action_hint * loss_action_hint
        + cfg.target_state * loss_target_state
        + cfg.target_subgoal * loss_target_subgoal
        + cfg.target_action * loss_target_action
        + cfg.target_candidate * loss_target_candidate
    )

    return {
        "total_loss": total_loss,
        "loss_entry_state": loss_entry_state,
        "loss_subgoal": loss_subgoal,
        "loss_action_hint": loss_action_hint,
        "loss_target_state": loss_target_state,
        "loss_target_subgoal": loss_target_subgoal,
        "loss_target_action": loss_target_action,
        "loss_target_candidate": loss_target_candidate,
    }

