from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


ENTRY_STATE_LABELS: Tuple[str, ...] = (
    "enterable_open_door",
    "enterable_door",
    "visible_but_blocked_entry",
    "front_blocked_detour",
    "window_visible_keep_search",
    "geometric_opening_needs_confirmation",
    "no_entry_confirmed",
)

SUBGOAL_LABELS: Tuple[str, ...] = (
    "keep_search",
    "approach_entry",
    "align_entry",
    "detour_left",
    "detour_right",
    "cross_entry",
    "backoff_and_reobserve",
)

ACTION_HINT_LABELS: Tuple[str, ...] = (
    "forward",
    "yaw_left",
    "yaw_right",
    "left",
    "right",
    "backward",
    "hold",
    "switch_to_next_house",
)

TARGET_CONDITIONED_STATE_LABELS: Tuple[str, ...] = (
    "target_house_not_in_view",
    "target_house_entry_visible",
    "target_house_entry_approachable",
    "target_house_entry_blocked",
    "non_target_house_entry_visible",
    "target_house_geometric_opening_needs_confirmation",
    "target_house_no_entry_after_full_coverage",
)

TARGET_CONDITIONED_SUBGOAL_LABELS: Tuple[str, ...] = (
    "reorient_to_target_house",
    "keep_search_target_house",
    "approach_target_entry",
    "align_target_entry",
    "detour_left_to_target_entry",
    "detour_right_to_target_entry",
    "cross_target_entry",
    "ignore_non_target_entry",
    "backoff_and_reobserve",
    "complete_no_entry_search",
)

TARGET_CONDITIONED_ACTION_HINT_LABELS: Tuple[str, ...] = ACTION_HINT_LABELS

DEFAULT_TOP_K_CANDIDATES = 3


LEGACY_TARGET_STATE_ALIASES: Mapping[str, str] = {
    "target_house_visible_keep_search": "target_house_entry_visible",
}


class LabelEncodingError(ValueError):
    """Raised when a label cannot be encoded or decoded."""


@dataclass(frozen=True)
class LabelSpace:
    name: str
    labels: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.labels:
            raise ValueError(f"Label space '{self.name}' must not be empty")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(f"Label space '{self.name}' contains duplicate labels")

    @property
    def size(self) -> int:
        return len(self.labels)

    @property
    def label_to_id(self) -> Dict[str, int]:
        return {label: idx for idx, label in enumerate(self.labels)}

    @property
    def id_to_label(self) -> Dict[int, str]:
        return {idx: label for idx, label in enumerate(self.labels)}

    def has(self, label: str) -> bool:
        return label in self.label_to_id

    def encode(
        self,
        label: Optional[str],
        *,
        allow_empty: bool = False,
        default: Optional[int] = None,
    ) -> int:
        value = str(label or "").strip()
        if not value:
            if allow_empty and default is not None:
                return int(default)
            raise LabelEncodingError(f"{self.name}: empty label is not allowed")
        if value not in self.label_to_id:
            raise LabelEncodingError(f"{self.name}: unknown label '{value}'")
        return self.label_to_id[value]

    def decode(self, label_id: int) -> str:
        if label_id not in self.id_to_label:
            raise LabelEncodingError(f"{self.name}: unknown id '{label_id}'")
        return self.id_to_label[label_id]


ENTRY_STATE_SPACE = LabelSpace("entry_state", ENTRY_STATE_LABELS)
SUBGOAL_SPACE = LabelSpace("subgoal", SUBGOAL_LABELS)
ACTION_HINT_SPACE = LabelSpace("action_hint", ACTION_HINT_LABELS)
TARGET_STATE_SPACE = LabelSpace("target_conditioned_state", TARGET_CONDITIONED_STATE_LABELS)
TARGET_SUBGOAL_SPACE = LabelSpace(
    "target_conditioned_subgoal",
    TARGET_CONDITIONED_SUBGOAL_LABELS,
)
TARGET_ACTION_HINT_SPACE = LabelSpace(
    "target_conditioned_action_hint",
    TARGET_CONDITIONED_ACTION_HINT_LABELS,
)


def canonical_target_state(label: Optional[str]) -> str:
    value = str(label or "").strip()
    if not value:
        return ""
    return LEGACY_TARGET_STATE_ALIASES.get(value, value)


def encode_entry_state(label: Optional[str]) -> int:
    return ENTRY_STATE_SPACE.encode(label)


def decode_entry_state(label_id: int) -> str:
    return ENTRY_STATE_SPACE.decode(label_id)


def encode_subgoal(label: Optional[str]) -> int:
    return SUBGOAL_SPACE.encode(label)


def decode_subgoal(label_id: int) -> str:
    return SUBGOAL_SPACE.decode(label_id)


def encode_action_hint(label: Optional[str]) -> int:
    return ACTION_HINT_SPACE.encode(label)


def decode_action_hint(label_id: int) -> str:
    return ACTION_HINT_SPACE.decode(label_id)


def encode_target_state(label: Optional[str]) -> int:
    return TARGET_STATE_SPACE.encode(canonical_target_state(label))


def decode_target_state(label_id: int) -> str:
    return TARGET_STATE_SPACE.decode(label_id)


def encode_target_subgoal(label: Optional[str]) -> int:
    return TARGET_SUBGOAL_SPACE.encode(label)


def decode_target_subgoal(label_id: int) -> str:
    return TARGET_SUBGOAL_SPACE.decode(label_id)


def encode_target_action(label: Optional[str]) -> int:
    return TARGET_ACTION_HINT_SPACE.encode(label)


def decode_target_action(label_id: int) -> str:
    return TARGET_ACTION_HINT_SPACE.decode(label_id)


def encode_target_candidate_id(
    candidate_id: Optional[int],
    *,
    top_k: int = DEFAULT_TOP_K_CANDIDATES,
    null_class: bool = True,
) -> int:
    value = -1 if candidate_id is None else int(candidate_id)
    if 0 <= value < int(top_k):
        return value
    if null_class:
        return int(top_k)
    raise LabelEncodingError(
        f"target_conditioned_target_candidate_id: candidate id '{value}' is outside [0, {top_k})"
    )


def decode_target_candidate_id(
    label_id: int,
    *,
    top_k: int = DEFAULT_TOP_K_CANDIDATES,
    null_class: bool = True,
) -> int:
    value = int(label_id)
    if 0 <= value < int(top_k):
        return value
    if null_class and value == int(top_k):
        return -1
    raise LabelEncodingError(
        f"target_conditioned_target_candidate_id: encoded id '{value}' is invalid for top_k={top_k}"
    )


def target_candidate_num_classes(*, top_k: int = DEFAULT_TOP_K_CANDIDATES, null_class: bool = True) -> int:
    return int(top_k) + (1 if null_class else 0)


def summarize_label_spaces(*, top_k: int = DEFAULT_TOP_K_CANDIDATES) -> Dict[str, object]:
    return {
        "entry_state": list(ENTRY_STATE_LABELS),
        "subgoal": list(SUBGOAL_LABELS),
        "action_hint": list(ACTION_HINT_LABELS),
        "target_conditioned_state": list(TARGET_CONDITIONED_STATE_LABELS),
        "target_conditioned_subgoal": list(TARGET_CONDITIONED_SUBGOAL_LABELS),
        "target_conditioned_action_hint": list(TARGET_CONDITIONED_ACTION_HINT_LABELS),
        "target_conditioned_target_candidate_num_classes": target_candidate_num_classes(top_k=top_k),
    }


def validate_known_labels(labels: Iterable[str], space: LabelSpace) -> List[str]:
    unknown: List[str] = []
    mapping = space.label_to_id
    for label in labels:
        value = str(label or "").strip()
        if value and value not in mapping:
            unknown.append(value)
    return unknown
