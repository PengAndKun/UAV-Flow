from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None  # type: ignore[assignment]


DEFAULT_EMBEDDING_DIR = Path(
    r"E:\github\UAV-Flow\phase2_5_representation_distillation\embeddings\memory_aware_v5_pilot_20260427"
)
NO_ENTRY_STATE = "target_house_no_entry_after_full_coverage"
NO_ENTRY_SUBGOAL = "complete_no_entry_search"


def _require_numpy() -> None:
    if np is None:
        raise ImportError("NumPy is required to analyze representation embeddings.")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return float(default)
        return number
    except (TypeError, ValueError):
        return float(default)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _percentile(values: Sequence[float], q: float) -> float:
    if not values or np is None:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), float(q)))


def _stats(values: Sequence[float]) -> Dict[str, float]:
    return {
        "count": float(len(values)),
        "mean": _mean(values),
        "min": min(values) if values else 0.0,
        "p25": _percentile(values, 25),
        "median": _percentile(values, 50),
        "p75": _percentile(values, 75),
        "max": max(values) if values else 0.0,
    }


def _state_of(row: Mapping[str, Any]) -> str:
    return str(row.get("true_target_conditioned_state") or "unknown")


def _subgoal_of(row: Mapping[str, Any]) -> str:
    return str(row.get("true_target_conditioned_subgoal") or "unknown")


def _sample_key(row: Mapping[str, Any]) -> str:
    return str(row.get("sample_id") or row.get("representation_index") or "")


def _extract_step(row: Mapping[str, Any]) -> int:
    text = f"{row.get('sample_id') or ''} {row.get('source_run_dir') or ''}"
    match = re.search(r"step(\d+)", text)
    if match:
        return int(match.group(1))
    return int(row.get("representation_index") or 0)


def _normalize_rows(rows: List[Dict[str, Any]], embeddings: "np.ndarray") -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        rep_idx = int(row.get("representation_index", idx))
        if rep_idx < 0 or rep_idx >= int(embeddings.shape[0]):
            rep_idx = idx
        item = dict(row)
        item["representation_index"] = rep_idx
        item["sequence_step"] = _extract_step(item)
        normalized.append(item)
    return normalized


def _centroid_distance_rows(embeddings: "np.ndarray", rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    states = sorted({_state_of(row) for row in rows})
    centroids: Dict[str, Any] = {}
    counts: Dict[str, int] = {}
    for state in states:
        indices = [int(row["representation_index"]) for row in rows if _state_of(row) == state]
        if not indices:
            continue
        centroids[state] = embeddings[indices].mean(axis=0)
        counts[state] = len(indices)

    output: List[Dict[str, Any]] = []
    for left in states:
        for right in states:
            if left not in centroids or right not in centroids:
                continue
            diff = centroids[left] - centroids[right]
            output.append(
                {
                    "state_a": left,
                    "state_b": right,
                    "count_a": counts[left],
                    "count_b": counts[right],
                    "euclidean_distance": float(np.linalg.norm(diff)),
                    "cosine_similarity": float(
                        np.dot(centroids[left], centroids[right])
                        / max(1e-8, np.linalg.norm(centroids[left]) * np.linalg.norm(centroids[right]))
                    ),
                }
            )
    return output


def _write_centroid_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "state_a",
        "state_b",
        "count_a",
        "count_b",
        "euclidean_distance",
        "cosine_similarity",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _nearest_neighbor_rows(
    embeddings: "np.ndarray",
    rows: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
) -> List[Dict[str, Any]]:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / np.maximum(norms, 1e-8)
    similarity = normalized @ normalized.T

    output: List[Dict[str, Any]] = []
    for row_idx, row in enumerate(rows):
        rep_idx = int(row["representation_index"])
        scores = similarity[rep_idx]
        neighbor_indices = np.argsort(-scores)
        neighbors: List[Dict[str, Any]] = []
        for neighbor_idx in neighbor_indices:
            n_idx = int(neighbor_idx)
            if n_idx == rep_idx:
                continue
            if len(neighbors) >= int(top_k):
                break
            neighbor_row = rows[n_idx]
            neighbors.append(
                {
                    "sample_id": _sample_key(neighbor_row),
                    "split": str(neighbor_row.get("split") or ""),
                    "source_session": str(neighbor_row.get("source_session") or ""),
                    "state": _state_of(neighbor_row),
                    "subgoal": _subgoal_of(neighbor_row),
                    "cosine_similarity": float(scores[n_idx]),
                    "euclidean_distance": float(np.linalg.norm(embeddings[rep_idx] - embeddings[n_idx])),
                }
            )
        output.append(
            {
                "sample_id": _sample_key(row),
                "split": str(row.get("split") or ""),
                "source_session": str(row.get("source_session") or ""),
                "state": _state_of(row),
                "subgoal": _subgoal_of(row),
                "neighbors": neighbors,
            }
        )
    return output


def _no_entry_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    true_rows = [row for row in rows if _state_of(row) == NO_ENTRY_STATE]
    false_rows = [row for row in rows if _state_of(row) != NO_ENTRY_STATE]
    true_probs = [_safe_float(row.get("no_entry_completion_prob")) for row in true_rows]
    false_probs = [_safe_float(row.get("no_entry_completion_prob")) for row in false_rows]
    complete_probs = [_safe_float(row.get("complete_no_entry_search_prob")) for row in true_rows]

    false_high = [
        {
            "sample_id": _sample_key(row),
            "state": _state_of(row),
            "subgoal": _subgoal_of(row),
            "prob": _safe_float(row.get("no_entry_completion_prob")),
            "full_coverage_ready": bool(row.get("full_coverage_ready")),
            "has_reliable_entry": bool(row.get("has_reliable_entry")),
            "source_session": str(row.get("source_session") or ""),
        }
        for row in false_rows
        if _safe_float(row.get("no_entry_completion_prob")) >= 0.5
    ]
    true_low = [
        {
            "sample_id": _sample_key(row),
            "state": _state_of(row),
            "subgoal": _subgoal_of(row),
            "prob": _safe_float(row.get("no_entry_completion_prob")),
            "full_coverage_ready": bool(row.get("full_coverage_ready")),
            "has_reliable_entry": bool(row.get("has_reliable_entry")),
            "source_session": str(row.get("source_session") or ""),
        }
        for row in true_rows
        if _safe_float(row.get("no_entry_completion_prob")) < 0.5
    ]

    return {
        "no_entry_state": NO_ENTRY_STATE,
        "true_count": len(true_rows),
        "false_count": len(false_rows),
        "true_no_entry_prob": _stats(true_probs),
        "false_no_entry_prob": _stats(false_probs),
        "true_complete_no_entry_search_prob": _stats(complete_probs),
        "false_high_prob_count": len(false_high),
        "true_low_prob_count": len(true_low),
        "false_high_prob_samples": false_high[:30],
        "true_low_prob_samples": true_low[:30],
        "separation_margin_mean": _mean(true_probs) - _mean(false_probs),
    }


def _episode_temporal_report(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_session: Dict[str, List[Mapping[str, Any]]] = {}
    for row in rows:
        session = str(row.get("source_session") or "unknown")
        by_session.setdefault(session, []).append(row)

    session_reports: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    for session, session_rows in sorted(by_session.items()):
        ordered = sorted(session_rows, key=lambda item: (int(item.get("sequence_step") or 0), _sample_key(item)))
        states = [_state_of(row) for row in ordered]
        subgoals = [_subgoal_of(row) for row in ordered]
        state_changes = sum(1 for idx in range(1, len(states)) if states[idx] != states[idx - 1])
        no_entry_indices = [idx for idx, row in enumerate(ordered) if _state_of(row) == NO_ENTRY_STATE]
        early_no_entry = [
            {
                "sample_id": _sample_key(row),
                "sequence_step": int(row.get("sequence_step") or 0),
                "full_coverage_ready": bool(row.get("full_coverage_ready")),
                "no_entry_after_full_coverage": bool(row.get("no_entry_after_full_coverage")),
                "prob": _safe_float(row.get("no_entry_completion_prob")),
            }
            for row in ordered
            if _state_of(row) == NO_ENTRY_STATE
            and (not bool(row.get("full_coverage_ready")) or not bool(row.get("no_entry_after_full_coverage")))
        ]
        if early_no_entry:
            warnings.append(
                {
                    "source_session": session,
                    "warning": "no_entry_state_without_full_coverage_evidence",
                    "samples": early_no_entry[:10],
                }
            )

        session_reports.append(
            {
                "source_session": session,
                "sample_count": len(ordered),
                "first_step": int(ordered[0].get("sequence_step") or 0) if ordered else 0,
                "last_step": int(ordered[-1].get("sequence_step") or 0) if ordered else 0,
                "state_changes": state_changes,
                "state_change_ratio": state_changes / max(1, len(ordered) - 1),
                "first_state": states[0] if states else "",
                "last_state": states[-1] if states else "",
                "unique_states": sorted(set(states)),
                "unique_subgoals": sorted(set(subgoals)),
                "no_entry_count": len(no_entry_indices),
                "first_no_entry_index": no_entry_indices[0] if no_entry_indices else -1,
                "first_no_entry_step": (
                    int(ordered[no_entry_indices[0]].get("sequence_step") or 0) if no_entry_indices else -1
                ),
                "max_no_entry_completion_prob": max(
                    [_safe_float(row.get("no_entry_completion_prob")) for row in ordered],
                    default=0.0,
                ),
                "early_no_entry_warning_count": len(early_no_entry),
            }
        )

    return {
        "session_count": len(session_reports),
        "sessions": session_reports,
        "warning_count": len(warnings),
        "warnings": warnings,
    }


def _projection_2d(embeddings: "np.ndarray") -> "np.ndarray":
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def _write_projection_png(path: Path, embeddings: "np.ndarray", rows: Sequence[Mapping[str, Any]]) -> bool:
    if plt is None:
        return False
    coords = _projection_2d(embeddings)
    states = sorted({_state_of(row) for row in rows})
    color_map = {state: idx for idx, state in enumerate(states)}

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    for state in states:
        indices = [idx for idx, row in enumerate(rows) if _state_of(row) == state]
        if not indices:
            continue
        ax.scatter(
            coords[indices, 0],
            coords[indices, 1],
            s=28,
            alpha=0.78,
            label=state,
            c=[color_map[state]] * len(indices),
            cmap="tab10",
            vmin=0,
            vmax=max(1, len(states) - 1),
        )
    ax.set_title("Memory-aware z_entry projection (PCA/SVD)")
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def analyze_representation_embeddings(
    *,
    embedding_dir: Path | str,
    output_dir: Path | str | None = None,
    nearest_k: int = 5,
) -> Dict[str, Any]:
    _require_numpy()

    embedding_root = Path(embedding_dir).resolve()
    analysis_root = Path(output_dir).resolve() if output_dir else embedding_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)

    embeddings_path = embedding_root / "embeddings.npy"
    metadata_path = embedding_root / "metadata.jsonl"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings.npy: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata.jsonl: {metadata_path}")

    embeddings = np.load(embeddings_path)
    rows = _normalize_rows(_read_jsonl(metadata_path), embeddings)
    if int(embeddings.shape[0]) != len(rows):
        raise ValueError(f"Embedding row count mismatch: embeddings={embeddings.shape[0]} metadata={len(rows)}")

    centroid_rows = _centroid_distance_rows(embeddings, rows)
    nearest_rows = _nearest_neighbor_rows(embeddings, rows, top_k=int(nearest_k))
    no_entry = _no_entry_report(rows)
    temporal = _episode_temporal_report(rows)
    projection_written = _write_projection_png(analysis_root / "embedding_projection.png", embeddings, rows)

    _write_centroid_csv(analysis_root / "state_centroid_distance.csv", centroid_rows)
    _write_jsonl(analysis_root / "nearest_neighbors.jsonl", nearest_rows)
    _write_json(analysis_root / "no_entry_probability_report.json", no_entry)
    _write_json(analysis_root / "episode_temporal_consistency_report.json", temporal)

    summary = {
        "embedding_dir": str(embedding_root),
        "output_dir": str(analysis_root),
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "sample_count": len(rows),
        "embedding_dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
        "state_count": len({_state_of(row) for row in rows}),
        "nearest_k": int(nearest_k),
        "no_entry_true_count": no_entry["true_count"],
        "no_entry_false_count": no_entry["false_count"],
        "mean_no_entry_prob_when_true": no_entry["true_no_entry_prob"]["mean"],
        "mean_no_entry_prob_when_false": no_entry["false_no_entry_prob"]["mean"],
        "no_entry_separation_margin_mean": no_entry["separation_margin_mean"],
        "temporal_warning_count": temporal["warning_count"],
        "projection_written": projection_written,
        "files": {
            "state_centroid_distance_csv": str(analysis_root / "state_centroid_distance.csv"),
            "nearest_neighbors_jsonl": str(analysis_root / "nearest_neighbors.jsonl"),
            "no_entry_probability_report_json": str(analysis_root / "no_entry_probability_report.json"),
            "episode_temporal_consistency_report_json": str(
                analysis_root / "episode_temporal_consistency_report.json"
            ),
            "embedding_projection_png": str(analysis_root / "embedding_projection.png"),
            "analysis_summary_json": str(analysis_root / "analysis_summary.json"),
        },
    }
    _write_json(analysis_root / "analysis_summary.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze exported z_entry embeddings for memory-aware representation quality."
    )
    parser.add_argument("--embedding_dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--nearest_k", type=int, default=5)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    summary = analyze_representation_embeddings(
        embedding_dir=args.embedding_dir.resolve(),
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        nearest_k=int(args.nearest_k),
    )
    print(
        "[rep-distill-analyze] done: "
        f"samples={summary['sample_count']} dim={summary['embedding_dim']} "
        f"no_entry_margin={summary['no_entry_separation_margin_mean']:.4f} "
        f"output={summary['output_dir']}"
    )


if __name__ == "__main__":
    main()
