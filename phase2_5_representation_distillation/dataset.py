from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass(frozen=True)
class ExportRecord:
    sample_id: str
    split: str
    entry_state_path: Path
    teacher_output_path: Path
    teacher_validation_path: Path
    metadata_path: Path
    raw_record: Dict[str, Any]


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _record_from_row(row: Dict[str, Any]) -> ExportRecord:
    sample_id = str(row.get("sample_id") or "").strip()
    split = str(row.get("split") or "").strip()
    if not sample_id:
        raise ValueError("JSONL row is missing sample_id")
    if split not in {"train", "val"}:
        raise ValueError(f"JSONL row '{sample_id}' has invalid split '{split}'")

    return ExportRecord(
        sample_id=sample_id,
        split=split,
        entry_state_path=Path(str(row["entry_state_path"])).resolve(),
        teacher_output_path=Path(str(row["teacher_output_path"])).resolve(),
        teacher_validation_path=Path(str(row["teacher_validation_path"])).resolve(),
        metadata_path=Path(str(row["metadata_path"])).resolve(),
        raw_record=dict(row),
    )


class DistillationDataset:
    """Reader for exported Phase 2.5 distillation train/val datasets."""

    def __init__(self, records: List[ExportRecord], *, lazy: bool = True) -> None:
        self.records = list(records)
        self.lazy = bool(lazy)
        self._cache: Dict[int, Dict[str, Any]] = {}
        if not self.lazy:
            for idx in range(len(self.records)):
                self._cache[idx] = self._load_sample(idx)

    @classmethod
    def from_jsonl(cls, jsonl_path: Path | str, *, lazy: bool = True) -> "DistillationDataset":
        path = Path(jsonl_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")
        rows = _read_jsonl(path)
        records = [_record_from_row(row) for row in rows]
        return cls(records, lazy=lazy)

    @classmethod
    def from_export_dir(
        cls,
        export_dir: Path | str,
        split: str,
        *,
        lazy: bool = True,
    ) -> "DistillationDataset":
        split_name = str(split).strip().lower()
        if split_name not in {"train", "val"}:
            raise ValueError(f"split must be 'train' or 'val', got '{split}'")
        export_root = Path(export_dir).resolve()
        jsonl_path = export_root / f"{split_name}.jsonl"
        return cls.from_jsonl(jsonl_path, lazy=lazy)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if index < 0 or index >= len(self.records):
            raise IndexError(index)
        if index not in self._cache:
            self._cache[index] = self._load_sample(index)
        return self._cache[index]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for idx in range(len(self)):
            yield self[idx]

    def _load_sample(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        missing_paths = [
            path
            for path in (
                record.entry_state_path,
                record.teacher_output_path,
                record.teacher_validation_path,
                record.metadata_path,
            )
            if not path.exists()
        ]
        if missing_paths:
            joined = ", ".join(str(path) for path in missing_paths)
            raise FileNotFoundError(f"Sample '{record.sample_id}' is missing files: {joined}")

        entry_state = _read_json(record.entry_state_path)
        teacher_output = _read_json(record.teacher_output_path)
        teacher_validation = _read_json(record.teacher_validation_path)
        metadata = _read_json(record.metadata_path)

        teacher_targets = (
            entry_state.get("teacher_targets", {})
            if isinstance(entry_state.get("teacher_targets"), dict)
            else {}
        )
        global_state = (
            entry_state.get("global_state", {})
            if isinstance(entry_state.get("global_state"), dict)
            else {}
        )
        candidates = (
            entry_state.get("candidates", [])
            if isinstance(entry_state.get("candidates"), list)
            else []
        )

        return {
            "sample_id": record.sample_id,
            "split": record.split,
            "record": record.raw_record,
            "paths": {
                "entry_state_path": str(record.entry_state_path),
                "teacher_output_path": str(record.teacher_output_path),
                "teacher_validation_path": str(record.teacher_validation_path),
                "metadata_path": str(record.metadata_path),
            },
            "entry_state": entry_state,
            "teacher_output": teacher_output,
            "teacher_validation": teacher_validation,
            "metadata": metadata,
            "global_state": global_state,
            "candidates": candidates,
            "teacher_targets": teacher_targets,
        }

    def sample_ids(self) -> List[str]:
        return [record.sample_id for record in self.records]

    def summarize(self) -> Dict[str, Any]:
        return {
            "num_samples": len(self.records),
            "splits": sorted({record.split for record in self.records}),
            "sample_ids": self.sample_ids(),
        }

