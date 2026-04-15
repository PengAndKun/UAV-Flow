from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .label_schema import (
    ACTION_HINT_SPACE,
    ENTRY_STATE_SPACE,
    SUBGOAL_SPACE,
    TARGET_ACTION_HINT_SPACE,
    TARGET_STATE_SPACE,
    TARGET_SUBGOAL_SPACE,
    target_candidate_num_classes,
)

try:
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - local env may not have torch
    torch = None  # type: ignore[assignment]
    Tensor = Any  # type: ignore[misc,assignment]
    nn = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required for phase2_5_representation_distillation.model. "
            "Install torch in the training environment before instantiating the model."
        )


@dataclass(frozen=True)
class StudentModelConfig:
    global_input_dim: int
    candidate_input_dim: int
    top_k: int = 3
    global_hidden_dim: int = 64
    candidate_hidden_dim: int = 64
    fused_hidden_dim: int = 256
    representation_dim: int = 128
    dropout: float = 0.1


if nn is not None:

    class GlobalEncoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x)


    class CandidateEncoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x)


    class EntryRepresentationStudent(nn.Module):
        def __init__(self, config: StudentModelConfig) -> None:
            super().__init__()
            self.config = config
            self.global_encoder = GlobalEncoder(
                input_dim=int(config.global_input_dim),
                hidden_dim=int(config.global_hidden_dim),
                dropout=float(config.dropout),
            )
            self.candidate_encoder = CandidateEncoder(
                input_dim=int(config.candidate_input_dim),
                hidden_dim=int(config.candidate_hidden_dim),
                dropout=float(config.dropout),
            )

            fused_input_dim = int(config.global_hidden_dim) + int(config.top_k) * int(config.candidate_hidden_dim)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(fused_input_dim, int(config.fused_hidden_dim)),
                nn.LayerNorm(int(config.fused_hidden_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(float(config.dropout)),
                nn.Linear(int(config.fused_hidden_dim), int(config.representation_dim)),
                nn.LayerNorm(int(config.representation_dim)),
                nn.ReLU(inplace=True),
            )

            repr_dim = int(config.representation_dim)
            self.entry_state_head = nn.Linear(repr_dim, ENTRY_STATE_SPACE.size)
            self.subgoal_head = nn.Linear(repr_dim, SUBGOAL_SPACE.size)
            self.action_hint_head = nn.Linear(repr_dim, ACTION_HINT_SPACE.size)
            self.target_state_head = nn.Linear(repr_dim, TARGET_STATE_SPACE.size)
            self.target_subgoal_head = nn.Linear(repr_dim, TARGET_SUBGOAL_SPACE.size)
            self.target_action_head = nn.Linear(repr_dim, TARGET_ACTION_HINT_SPACE.size)
            self.target_candidate_head = nn.Linear(
                repr_dim,
                target_candidate_num_classes(top_k=int(config.top_k)),
            )

        def forward(
            self,
            global_features: Tensor,
            candidate_features: Tensor,
            candidate_valid_mask: Optional[Tensor] = None,
        ) -> Dict[str, Tensor]:
            """
            Args:
                global_features: [B, Dg]
                candidate_features: [B, K, Dc]
                candidate_valid_mask: [B, K]
            """
            global_repr = self.global_encoder(global_features)

            batch_size, top_k, candidate_dim = candidate_features.shape
            encoded_candidates = self.candidate_encoder(
                candidate_features.reshape(batch_size * top_k, candidate_dim)
            ).reshape(batch_size, top_k, self.config.candidate_hidden_dim)

            if candidate_valid_mask is not None:
                mask = candidate_valid_mask.unsqueeze(-1).to(encoded_candidates.dtype)
                encoded_candidates = encoded_candidates * mask

            fused_input = torch.cat(
                [global_repr, encoded_candidates.reshape(batch_size, top_k * self.config.candidate_hidden_dim)],
                dim=-1,
            )
            z_entry = self.fusion_mlp(fused_input)

            return {
                "z_entry": z_entry,
                "entry_state_logits": self.entry_state_head(z_entry),
                "subgoal_logits": self.subgoal_head(z_entry),
                "action_hint_logits": self.action_hint_head(z_entry),
                "target_state_logits": self.target_state_head(z_entry),
                "target_subgoal_logits": self.target_subgoal_head(z_entry),
                "target_action_logits": self.target_action_head(z_entry),
                "target_candidate_logits": self.target_candidate_head(z_entry),
            }


else:

    class GlobalEncoder:  # pragma: no cover - fallback only
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


    class CandidateEncoder:  # pragma: no cover - fallback only
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


    class EntryRepresentationStudent:  # pragma: no cover - fallback only
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            _require_torch()


def build_default_config(global_input_dim: int, candidate_input_dim: int, *, top_k: int = 3) -> StudentModelConfig:
    return StudentModelConfig(
        global_input_dim=int(global_input_dim),
        candidate_input_dim=int(candidate_input_dim),
        top_k=int(top_k),
    )

