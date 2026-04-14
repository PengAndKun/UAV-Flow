from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


RESULTS_ROOT = Path(__file__).resolve().parent / "results"
EXPORTS_ROOT = Path(__file__).resolve().parent / "exports"
DEFAULT_EXPORT_NAME = "phase2_5_distillation_dataset"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _discover_labeling_dirs(results_root: Path) -> List[Path]:
    output: List[Path] = []
    for child in sorted(results_root.iterdir()):
        if not child.is_dir():
            continue
        labeling_dir = child / "labeling"
        if labeling_dir.is_dir():
            output.append(labeling_dir)
    return output


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _load_sample_record(labeling_dir: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    entry_state_path = labeling_dir / "entry_state.json"
    teacher_output_path = labeling_dir / "teacher_output.json"
    teacher_validation_path = labeling_dir / "teacher_validation.json"

    if not entry_state_path.exists():
        return None, "missing_entry_state"
    if not teacher_output_path.exists():
        return None, "missing_teacher_output"
    if not teacher_validation_path.exists():
        return None, "missing_teacher_validation"

    try:
        entry_state = _read_json(entry_state_path)
        teacher_output = _read_json(teacher_output_path)
        teacher_validation = _read_json(teacher_validation_path)
    except Exception as exc:
        return None, f"json_read_error:{exc}"

    teacher_targets = entry_state.get("teacher_targets", {}) if isinstance(entry_state.get("teacher_targets"), dict) else {}
    normalized = teacher_validation.get("normalized_teacher_output", {}) if isinstance(teacher_validation.get("normalized_teacher_output"), dict) else {}
    metadata = entry_state.get("metadata", {}) if isinstance(entry_state.get("metadata"), dict) else {}
    global_state = entry_state.get("global_state", {}) if isinstance(entry_state.get("global_state"), dict) else {}
    candidates = entry_state.get("candidates", []) if isinstance(entry_state.get("candidates"), list) else []

    sample = {
        "sample_id": str(entry_state.get("sample_id") or labeling_dir.parent.name),
        "run_dir": str(labeling_dir.parent),
        "labeling_dir": str(labeling_dir),
        "entry_state_path": str(entry_state_path),
        "teacher_output_path": str(teacher_output_path),
        "teacher_validation_path": str(teacher_validation_path),
        "entry_state": entry_state,
        "teacher_output": teacher_output,
        "teacher_validation": teacher_validation,
        "teacher_targets": teacher_targets,
        "normalized_teacher_output": normalized,
        "metadata": metadata,
        "global_state": global_state,
        "candidates": candidates,
    }
    return sample, None


def _filter_sample(
    sample: Dict[str, Any],
    *,
    min_teacher_confidence: float,
    allow_weak_valid: bool,
) -> Tuple[bool, str]:
    teacher_validation = sample["teacher_validation"]
    status = str(teacher_validation.get("status") or "").strip()
    score = _safe_float(teacher_validation.get("score"), 0.0)
    teacher_targets = sample.get("teacher_targets", {})
    confidence = _safe_float(teacher_targets.get("confidence"), 0.0)
    teacher_available = _safe_int(teacher_targets.get("teacher_available"), 0)

    if teacher_available != 1:
        return False, "teacher_unavailable"
    if status == "invalid":
        return False, "teacher_invalid"
    if status == "weak_valid" and not allow_weak_valid:
        return False, "teacher_weak_valid_excluded"
    if confidence < float(min_teacher_confidence):
        return False, "teacher_confidence_too_low"
    if score <= 0.0:
        return False, "teacher_score_nonpositive"
    return True, "accepted"


def _class_counts(items: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        teacher_targets = item.get("teacher_targets", {})
        value = str(teacher_targets.get(field) or "").strip()
        if value:
            counter[value] += 1
    return dict(sorted(counter.items(), key=lambda pair: pair[0]))


def _teacher_field_counts(items: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for item in items:
        teacher_targets = item.get("teacher_targets", {})
        value = str(teacher_targets.get(field) or "").strip()
        if value:
            counter[value] += 1
    return dict(sorted(counter.items(), key=lambda pair: pair[0]))


def _teacher_available_count(items: List[Dict[str, Any]], field: str) -> int:
    total = 0
    for item in items:
        teacher_targets = item.get("teacher_targets", {})
        if _safe_int(teacher_targets.get(field), 0) == 1:
            total += 1
    return total


def _missing_teacher_field_count(items: List[Dict[str, Any]], field: str) -> int:
    total = 0
    for item in items:
        teacher_targets = item.get("teacher_targets", {})
        value = str(teacher_targets.get(field) or "").strip()
        if not value:
            total += 1
    return total


def _group_key(sample: Dict[str, Any]) -> str:
    teacher_targets = sample.get("teacher_targets", {})
    entry_state = str(teacher_targets.get("entry_state") or "").strip()
    return entry_state or "unknown"


def _split_samples(samples: List[Dict[str, Any]], val_ratio: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not samples:
        return [], []

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[_group_key(sample)].append(sample)

    train: List[Dict[str, Any]] = []
    val: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        group = sorted(grouped[key], key=lambda item: item["sample_id"])
        if len(group) == 1:
            train.extend(group)
            continue
        val_count = max(1, int(round(len(group) * float(val_ratio))))
        if val_count >= len(group):
            val_count = len(group) - 1
        val.extend(group[:val_count])
        train.extend(group[val_count:])
    return train, val


def _make_export_record(sample: Dict[str, Any], sample_dir: Path, split: str) -> Dict[str, Any]:
    teacher_targets = sample.get("teacher_targets", {})
    metadata = sample.get("metadata", {})
    global_state = sample.get("global_state", {})
    return {
        "sample_id": sample["sample_id"],
        "split": split,
        "entry_state_path": str(sample_dir / "entry_state.json"),
        "teacher_output_path": str(sample_dir / "teacher_output.json"),
        "teacher_validation_path": str(sample_dir / "teacher_validation.json"),
        "metadata_path": str(sample_dir / "metadata.json"),
        "teacher_available": _safe_int(teacher_targets.get("teacher_available"), 0),
        "target_candidate_id": _safe_int(teacher_targets.get("target_candidate_id"), -1),
        "entry_state": str(teacher_targets.get("entry_state") or ""),
        "subgoal": str(teacher_targets.get("subgoal") or ""),
        "action_hint": str(teacher_targets.get("action_hint") or ""),
        "risk_level": str(teacher_targets.get("risk_level") or ""),
        "target_conditioned_teacher_available": _safe_int(
            teacher_targets.get("target_conditioned_teacher_available"), 0
        ),
        "target_conditioned_target_candidate_id": _safe_int(
            teacher_targets.get("target_conditioned_target_candidate_id"), -1
        ),
        "target_conditioned_state": str(teacher_targets.get("target_conditioned_state") or ""),
        "target_conditioned_subgoal": str(teacher_targets.get("target_conditioned_subgoal") or ""),
        "target_conditioned_action_hint": str(
            teacher_targets.get("target_conditioned_action_hint") or ""
        ),
        "target_conditioned_confidence": _safe_float(
            teacher_targets.get("target_conditioned_confidence"), 0.0
        ),
        "target_house_id": _safe_int(global_state.get("target_house_id"), -1),
        "target_house_in_fov": _safe_int(global_state.get("target_house_in_fov"), 0),
        "target_conditioning_enabled": _safe_int(
            sample.get("metadata", {}).get("target_conditioning_enabled"), 0
        ),
        "task_label": str(metadata.get("task_label") or ""),
        "source_run_dir": str(metadata.get("source_run_dir") or sample["run_dir"]),
    }


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_distillation_dataset(
    *,
    results_root: Path,
    output_root: Path,
    export_name: str,
    val_ratio: float,
    min_teacher_confidence: float,
    allow_weak_valid: bool,
) -> Dict[str, Any]:
    results_root = results_root.resolve()
    output_root = output_root.resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    export_dir = output_root / f"{export_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    samples_dir = export_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    labeling_dirs = _discover_labeling_dirs(results_root)
    print(f"[dataset-export] discovered {len(labeling_dirs)} labeling directories")

    loaded_samples: List[Dict[str, Any]] = []
    skipped_counter: Counter[str] = Counter()
    for idx, labeling_dir in enumerate(labeling_dirs, start=1):
        sample_name = labeling_dir.parent.name
        print(f"[{idx}/{len(labeling_dirs)}] inspect -> {sample_name}")
        sample, error_reason = _load_sample_record(labeling_dir)
        if sample is None:
            skipped_counter[error_reason or "unknown_load_error"] += 1
            print(f"[{idx}/{len(labeling_dirs)}] skip -> {sample_name} ({error_reason})")
            continue
        accepted, reason = _filter_sample(
            sample,
            min_teacher_confidence=float(min_teacher_confidence),
            allow_weak_valid=bool(allow_weak_valid),
        )
        if not accepted:
            skipped_counter[reason] += 1
            print(f"[{idx}/{len(labeling_dirs)}] skip -> {sample_name} ({reason})")
            continue
        loaded_samples.append(sample)
        print(
            f"[{idx}/{len(labeling_dirs)}] accept -> {sample_name} "
            f"({sample['teacher_targets'].get('entry_state', '')}; "
            f"{sample['teacher_targets'].get('subgoal', '')})"
        )

    loaded_samples = sorted(loaded_samples, key=lambda item: item["sample_id"])
    train_samples, val_samples = _split_samples(loaded_samples, val_ratio=float(val_ratio))

    train_records: List[Dict[str, Any]] = []
    val_records: List[Dict[str, Any]] = []
    export_class_counts = {
        "entry_state": _teacher_field_counts(loaded_samples, "entry_state"),
        "subgoal": _teacher_field_counts(loaded_samples, "subgoal"),
        "action_hint": _teacher_field_counts(loaded_samples, "action_hint"),
        "target_conditioned_state": _teacher_field_counts(loaded_samples, "target_conditioned_state"),
        "target_conditioned_subgoal": _teacher_field_counts(loaded_samples, "target_conditioned_subgoal"),
        "target_conditioned_action_hint": _teacher_field_counts(
            loaded_samples, "target_conditioned_action_hint"
        ),
    }
    target_conditioned_teacher_available_count = _teacher_available_count(
        loaded_samples, "target_conditioned_teacher_available"
    )
    missing_target_conditioned_fields = {
        "target_conditioned_state": _missing_teacher_field_count(
            loaded_samples, "target_conditioned_state"
        ),
        "target_conditioned_subgoal": _missing_teacher_field_count(
            loaded_samples, "target_conditioned_subgoal"
        ),
        "target_conditioned_action_hint": _missing_teacher_field_count(
            loaded_samples, "target_conditioned_action_hint"
        ),
    }
    val_sample_ids = {item["sample_id"] for item in val_samples}

    for sample_index, sample in enumerate(loaded_samples, start=1):
        sample_dir = samples_dir / f"sample_{sample_index:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        _write_json(sample_dir / "entry_state.json", sample["entry_state"])
        _write_json(sample_dir / "teacher_output.json", sample["teacher_output"])
        _write_json(sample_dir / "teacher_validation.json", sample["teacher_validation"])
        _write_json(
            sample_dir / "metadata.json",
            {
                "sample_id": sample["sample_id"],
                "run_dir": sample["run_dir"],
                "labeling_dir": sample["labeling_dir"],
                "task_label": sample["metadata"].get("task_label"),
                "source_run_dir": sample["metadata"].get("source_run_dir"),
                "rgb_path": sample["metadata"].get("rgb_path"),
                "depth_cm_path": sample["metadata"].get("depth_cm_path"),
                "depth_preview_path": sample["metadata"].get("depth_preview_path"),
                "target_house_id": sample["global_state"].get("target_house_id"),
                "current_house_id": sample["global_state"].get("current_house_id"),
                "target_house_in_fov": sample["global_state"].get("target_house_in_fov"),
                "target_conditioning_enabled": sample["metadata"].get("target_conditioning_enabled", 1),
            },
        )
        split = "val" if sample["sample_id"] in val_sample_ids else "train"
        record = _make_export_record(sample, sample_dir, split)
        if split == "train":
            train_records.append(record)
        else:
            val_records.append(record)

    _write_jsonl(export_dir / "train.jsonl", train_records)
    _write_jsonl(export_dir / "val.jsonl", val_records)
    (export_dir / "train_ids.txt").write_text("\n".join(record["sample_id"] for record in train_records), encoding="utf-8")
    (export_dir / "val_ids.txt").write_text("\n".join(record["sample_id"] for record in val_records), encoding="utf-8")

    quality_report = {
        "results_root": str(results_root),
        "inspected_labeling_dirs": len(labeling_dirs),
        "exported_samples": len(loaded_samples),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "skipped_counts": dict(sorted(skipped_counter.items(), key=lambda pair: pair[0])),
        "allow_weak_valid": bool(allow_weak_valid),
        "min_teacher_confidence": float(min_teacher_confidence),
        "target_conditioned_enabled": True,
        "target_conditioned_teacher_available_count": target_conditioned_teacher_available_count,
        "missing_target_conditioned_fields_count": missing_target_conditioned_fields,
    }
    _write_json(export_dir / "quality_report.json", quality_report)

    manifest = {
        "export_dir": str(export_dir),
        "results_root": str(results_root),
        "total_exported": len(loaded_samples),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "entry_state_counts": export_class_counts["entry_state"],
        "subgoal_counts": export_class_counts["subgoal"],
        "action_hint_counts": export_class_counts["action_hint"],
        "target_conditioned_state_counts": export_class_counts["target_conditioned_state"],
        "target_conditioned_subgoal_counts": export_class_counts["target_conditioned_subgoal"],
        "target_conditioned_action_hint_counts": export_class_counts["target_conditioned_action_hint"],
        "allow_weak_valid": bool(allow_weak_valid),
        "min_teacher_confidence": float(min_teacher_confidence),
        "target_conditioned_enabled": True,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    _write_json(export_dir / "manifest.json", manifest)

    return {
        "export_dir": str(export_dir),
        "manifest_path": str(export_dir / "manifest.json"),
        "quality_report_path": str(export_dir / "quality_report.json"),
        "total_exported": len(loaded_samples),
        "train_count": len(train_records),
        "val_count": len(val_records),
        "entry_state_counts": export_class_counts["entry_state"],
        "subgoal_counts": export_class_counts["subgoal"],
        "action_hint_counts": export_class_counts["action_hint"],
        "target_conditioned_state_counts": export_class_counts["target_conditioned_state"],
        "target_conditioned_subgoal_counts": export_class_counts["target_conditioned_subgoal"],
        "target_conditioned_action_hint_counts": export_class_counts["target_conditioned_action_hint"],
        "skipped_counts": quality_report["skipped_counts"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export Phase 2.5 distillation train/val datasets from validated fusion labeling packages."
    )
    parser.add_argument("--results_root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--output_root", type=Path, default=EXPORTS_ROOT)
    parser.add_argument("--export_name", type=str, default=DEFAULT_EXPORT_NAME)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--min_teacher_confidence", type=float, default=0.55)
    parser.add_argument("--allow_weak_valid", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = export_distillation_dataset(
        results_root=args.results_root,
        output_root=args.output_root,
        export_name=str(args.export_name),
        val_ratio=float(args.val_ratio),
        min_teacher_confidence=float(args.min_teacher_confidence),
        allow_weak_valid=bool(args.allow_weak_valid),
    )
    print(
        f"[dataset-export] done: exported={summary['total_exported']} "
        f"train={summary['train_count']} val={summary['val_count']}"
    )
    print(f"[dataset-export] manifest -> {summary['manifest_path']}")
    print(f"[dataset-export] quality_report -> {summary['quality_report_path']}")


if __name__ == "__main__":
    main()
