from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from phase2_5_representation_distillation.evaluator import evaluate_representation_distillation  # type: ignore
else:
    from .evaluator import evaluate_representation_distillation


DEFAULT_EXPORT_DIR = Path(
    r"E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\phase2_5_distillation_dataset_20260414_163838"
)
DEFAULT_OUTPUT_ROOT = Path(r"E:\github\UAV-Flow\phase2_5_representation_distillation\evals")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Phase 2.5 representation distillation student."
    )
    parser.add_argument("--checkpoint_path", type=Path, required=True)
    parser.add_argument("--export_dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", type=str, default=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output_dir = args.output_root.resolve() / str(args.run_name)
    summary = evaluate_representation_distillation(
        checkpoint_path=args.checkpoint_path.resolve(),
        export_dir=args.export_dir.resolve(),
        output_dir=output_dir,
        split=str(args.split),
        batch_size=int(args.batch_size),
        device=str(args.device),
        num_workers=int(args.num_workers),
    )
    print(
        f"[rep-distill-eval] done: split={summary['split']} samples={summary['sample_count']} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()

