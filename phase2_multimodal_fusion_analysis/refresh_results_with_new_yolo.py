from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from entry_state_builder import build_entry_state_for_labeling_dir
from fusion_entry_analysis import find_latest_phase2_weights, reprocess_phase2_fusion_run
from teacher_validator import validate_labeling_dir


RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _discover_run_dirs(results_root: Path) -> List[Path]:
    output: List[Path] = []
    for child in sorted(results_root.iterdir()):
        if child.is_dir() and child.name.startswith("fusion_"):
            output.append(child)
    return output


def _parse_labeling_summary(path: Path) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    if not path.exists():
        return parsed
    for raw_line in _read_text(path).splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    parsed["sample_id"] = str(parsed.get("sample_id", path.parent.parent.name)).strip()
    parsed["task_label"] = str(parsed.get("task_label", "")).strip()
    parsed["fusion_state"] = str(parsed.get("fusion_state", "")).strip()
    parsed["top_yolo"] = str(parsed.get("top_yolo", "")).strip()
    return parsed


def _stage_group(active_stage: str) -> str:
    stage = str(active_stage or "").strip().lower()
    if stage in {"approach_entry", "cross_entry"}:
        return "entry"
    if stage in {"entry_search", "outside_localization", "target_house_verification"}:
        return "search"
    if stage in {"inside_house", "indoor_room_search", "mission_report"}:
        return "other"
    return "other"


def _infer_expected_from_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    fusion_state = str(summary.get("fusion_state", "")).strip()
    top_yolo = str(summary.get("top_yolo", "")).strip().lower()
    top_yolo_class = top_yolo.split("(", 1)[0].strip()

    enterable_states = {"enterable_open_door", "enterable_door"}
    blocked_states = {"visible_but_blocked_entry", "front_blocked_detour"}
    search_states = {"window_visible_keep_search", "geometric_opening_needs_confirmation", "no_entry_confirmed"}

    expected_entry_visible = fusion_state in (enterable_states | blocked_states)
    if fusion_state == "window_visible_keep_search" or top_yolo_class.startswith("window"):
        expected_entry_visible = False
    expected_entry_traversable = fusion_state in enterable_states

    expected_stage_group = "search"
    if fusion_state in enterable_states:
        expected_stage_group = "entry"
    elif fusion_state in blocked_states:
        expected_stage_group = "blocked"
    elif fusion_state in search_states:
        expected_stage_group = "search"

    return {
        "fusion_state": fusion_state,
        "top_yolo_class": top_yolo_class,
        "expected_entry_visible": expected_entry_visible,
        "expected_entry_traversable": expected_entry_traversable,
        "expected_stage_group": expected_stage_group,
    }


def _compare_summary_and_llm(summary: Dict[str, Any], llm_payload: Dict[str, Any]) -> Dict[str, Any]:
    parsed = llm_payload.get("parsed", {}) if isinstance(llm_payload.get("parsed"), dict) else {}
    expected = _infer_expected_from_summary(summary)
    actual_stage_group = _stage_group(str(parsed.get("active_stage", "")))

    entry_visible_match = bool(parsed.get("entry_door_visible", False)) == bool(expected["expected_entry_visible"])
    entry_traversable_match = bool(parsed.get("entry_door_traversable", False)) == bool(
        expected["expected_entry_traversable"]
    )

    stage_match = False
    if expected["expected_stage_group"] == "entry":
        stage_match = actual_stage_group == "entry"
    elif expected["expected_stage_group"] == "search":
        stage_match = actual_stage_group == "search"
    elif expected["expected_stage_group"] == "blocked":
        stage_match = actual_stage_group != "entry"

    issues: List[str] = []
    if not entry_visible_match:
        issues.append("entry_visible_mismatch")
    if not entry_traversable_match:
        issues.append("entry_traversable_mismatch")
    if not stage_match:
        issues.append("stage_group_mismatch")

    overall = "consistent"
    if issues and len(issues) == 1:
        overall = "partially_consistent"
    elif len(issues) >= 2:
        overall = "conflict"

    return {
        "expected": expected,
        "actual": {
            "scene_state": parsed.get("scene_state"),
            "active_stage": parsed.get("active_stage"),
            "entry_door_visible": parsed.get("entry_door_visible"),
            "entry_door_traversable": parsed.get("entry_door_traversable"),
            "confidence": parsed.get("confidence"),
        },
        "agreement": {
            "entry_visible_match": entry_visible_match,
            "entry_traversable_match": entry_traversable_match,
            "stage_group_match": stage_match,
            "overall": overall,
            "issues": issues,
        },
    }


def _discover_anthropic_result_files(labeling_dir: Path) -> List[Path]:
    return sorted(
        [
            path
            for path in labeling_dir.glob("anthropic*_scene_result.json")
            if "_vs_labeling_compare" not in path.name
        ],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )


def _compare_path_for_result(result_path: Path) -> Path:
    name = result_path.name
    if name.endswith("_scene_result.json"):
        return result_path.with_name(name.replace("_scene_result.json", "_vs_labeling_compare.json"))
    return result_path.with_name(result_path.stem + "_vs_labeling_compare.json")


def refresh_existing_compare_results(labeling_dir: Path) -> List[Path]:
    summary_path = labeling_dir / "labeling_summary.txt"
    summary = _parse_labeling_summary(summary_path)
    if not summary:
        return []

    compare_paths: List[Path] = []
    for result_path in _discover_anthropic_result_files(labeling_dir):
        llm_payload = _read_json(result_path)
        compare_path = _compare_path_for_result(result_path)
        compare_payload = {
            "sample_id": summary.get("sample_id", labeling_dir.parent.name),
            "labeling_dir": str(labeling_dir),
            "task_label": summary.get("task_label", ""),
            "model_name": llm_payload.get("model_name", ""),
            "summary_reference": summary,
            "llm_reference": {
                "result_json": str(result_path),
                "prompt_log_path": llm_payload.get("prompt_log_path", ""),
            },
            "comparison": _compare_summary_and_llm(summary, llm_payload),
            "refreshed_at": datetime.now().isoformat(timespec="seconds"),
        }
        _write_json(compare_path, compare_payload)
        compare_paths.append(compare_path)
    return compare_paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Re-run Phase 2 YOLO+fusion analysis for existing results folders using a new YOLO checkpoint, "
            "then refresh downstream compare/teacher/state files."
        )
    )
    parser.add_argument("--results_root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--only_dir", type=str, default="", help="Only process one fusion_xxx directory.")
    parser.add_argument("--weights", type=Path, default=None, help="Path to the new YOLO weights.")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--skip_compare_refresh", action="store_true")
    parser.add_argument("--skip_teacher_refresh", action="store_true")
    parser.add_argument("--skip_entry_state_refresh", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    results_root = args.results_root.resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    run_dirs = _discover_run_dirs(results_root)
    if args.only_dir:
        run_dirs = [path for path in run_dirs if path.name == args.only_dir]
        if not run_dirs:
            raise FileNotFoundError(f"Could not find run directory named: {args.only_dir}")

    weights = args.weights.resolve() if args.weights else find_latest_phase2_weights()
    total = len(run_dirs)
    print(f"[refresh-yolo] discovered {total} fusion run directories")
    print(f"[refresh-yolo] weights -> {weights}")

    summary_items: List[Dict[str, Any]] = []
    ok_count = 0
    error_count = 0

    for idx, run_dir in enumerate(run_dirs, start=1):
        print(f"[{idx}/{total}] start -> {run_dir.name}")
        try:
            result = reprocess_phase2_fusion_run(
                run_dir,
                weights=weights,
                conf=float(args.conf),
                imgsz=int(args.imgsz),
                device=str(args.device),
            )
            labeling_dir = run_dir / "labeling"

            compare_paths: List[Path] = []
            teacher_summary: Optional[Dict[str, Any]] = None
            entry_state_summary: Optional[Dict[str, Any]] = None

            if not args.skip_compare_refresh:
                compare_paths = refresh_existing_compare_results(labeling_dir)
            if not args.skip_teacher_refresh:
                teacher_summary = validate_labeling_dir(labeling_dir)
            if not args.skip_entry_state_refresh:
                entry_state_summary = build_entry_state_for_labeling_dir(labeling_dir)

            fusion = result.get("fusion", {}) if isinstance(result.get("fusion"), dict) else {}
            print(
                f"[{idx}/{total}] done -> {run_dir.name} "
                f"({fusion.get('final_entry_state', 'unknown')}; "
                f"compare={len(compare_paths)}; "
                f"teacher={teacher_summary.get('status') if isinstance(teacher_summary, dict) else 'skip'}; "
                f"state={'ok' if isinstance(entry_state_summary, dict) else 'skip'})"
            )

            summary_items.append(
                {
                    "run_dir": str(run_dir),
                    "status": "ok",
                    "weights": str(weights),
                    "final_entry_state": fusion.get("final_entry_state"),
                    "recommended_subgoal": fusion.get("recommended_subgoal"),
                    "recommended_action_hint": fusion.get("recommended_action_hint"),
                    "crossing_ready": fusion.get("crossing_ready"),
                    "compare_refreshed_count": len(compare_paths),
                    "compare_paths": [str(path) for path in compare_paths],
                    "teacher_status": teacher_summary.get("status") if isinstance(teacher_summary, dict) else None,
                    "teacher_validation_path": teacher_summary.get("teacher_validation_path")
                    if isinstance(teacher_summary, dict)
                    else None,
                    "entry_state_path": entry_state_summary.get("entry_state_path")
                    if isinstance(entry_state_summary, dict)
                    else None,
                    "fusion_result_path": str(run_dir / "fusion" / "fusion_result.json"),
                }
            )
            ok_count += 1
        except Exception as exc:
            print(f"[{idx}/{total}] error -> {run_dir.name}: {exc}")
            summary_items.append(
                {
                    "run_dir": str(run_dir),
                    "status": "error",
                    "error": str(exc),
                }
            )
            error_count += 1

    summary_path = results_root / f"refresh_results_with_new_yolo_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _write_json(
        summary_path,
        {
            "results_root": str(results_root),
            "weights": str(weights),
            "ok_count": ok_count,
            "error_count": error_count,
            "items": summary_items,
        },
    )
    print(f"[refresh-yolo] finished: ok={ok_count} error={error_count}")
    print(f"[refresh-yolo] summary -> {summary_path}")


if __name__ == "__main__":
    main()
