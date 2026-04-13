from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fusion_entry_analysis import (
    find_latest_phase2_weights,
    reprocess_phase2_fusion_run,
)


RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def _discover_run_dirs(results_root: Path) -> List[Path]:
    output: List[Path] = []
    for child in sorted(results_root.iterdir()):
        if child.is_dir() and child.name.startswith("fusion_"):
            output.append(child)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reprocess existing Phase 2 fusion result folders with the updated RGB-first depth fusion logic."
    )
    parser.add_argument("--results_root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--only_dir", type=str, default="", help="Only reprocess a single fusion_xxx directory name.")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    results_root = args.results_root.resolve()
    if not results_root.exists():
        raise FileNotFoundError(f"Results root not found: {results_root}")

    all_run_dirs = _discover_run_dirs(results_root)
    if args.only_dir:
        all_run_dirs = [path for path in all_run_dirs if path.name == args.only_dir]
        if not all_run_dirs:
            raise FileNotFoundError(f"Could not find run directory named: {args.only_dir}")

    weights = args.weights.resolve() if args.weights else find_latest_phase2_weights()
    total = len(all_run_dirs)
    print(f"[reprocess] discovered {total} fusion run directories")
    print(f"[reprocess] weights -> {weights}")

    summary: List[Dict[str, Any]] = []
    ok_count = 0
    error_count = 0
    for idx, run_dir in enumerate(all_run_dirs, start=1):
        print(f"[{idx}/{total}] start -> {run_dir.name}")
        try:
            result = reprocess_phase2_fusion_run(
                run_dir,
                weights=weights,
                conf=float(args.conf),
                imgsz=int(args.imgsz),
                device=str(args.device),
            )
            fusion = result.get("fusion", {}) if isinstance(result.get("fusion"), dict) else {}
            print(
                f"[{idx}/{total}] done -> {run_dir.name} "
                f"({fusion.get('final_entry_state', 'unknown')}; "
                f"subgoal={fusion.get('recommended_subgoal', 'n/a')}; "
                f"cross_ready={int(bool(fusion.get('crossing_ready', False)))})"
            )
            summary.append(
                {
                    "run_dir": str(run_dir),
                    "status": "ok",
                    "final_entry_state": fusion.get("final_entry_state"),
                    "recommended_subgoal": fusion.get("recommended_subgoal"),
                    "recommended_action_hint": fusion.get("recommended_action_hint"),
                    "crossing_ready": fusion.get("crossing_ready"),
                    "decision_reason": fusion.get("decision_reason"),
                    "fusion_result_path": str(run_dir / "fusion" / "fusion_result.json"),
                }
            )
            ok_count += 1
        except Exception as exc:
            print(f"[{idx}/{total}] error -> {run_dir.name}: {exc}")
            summary.append(
                {
                    "run_dir": str(run_dir),
                    "status": "error",
                    "error": str(exc),
                }
            )
            error_count += 1

    summary_path = results_root / f"reprocess_phase2_fusion_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    summary_path.write_text(
        json.dumps(
            {
                "results_root": str(results_root),
                "weights": str(weights),
                "ok_count": ok_count,
                "error_count": error_count,
                "items": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"[reprocess] finished: ok={ok_count} error={error_count}")
    print(f"[reprocess] summary -> {summary_path}")


if __name__ == "__main__":
    main()
