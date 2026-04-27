from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from phase2_5_representation_distillation.trainer import (  # type: ignore
        DistillationTrainerConfig,
        train_representation_distillation,
    )
else:
    from .trainer import DistillationTrainerConfig, train_representation_distillation

DEFAULT_EXPORT_DIR = Path(
    r"E:\github\UAV-Flow\phase2_multimodal_fusion_analysis\exports\phase2_5_distillation_dataset_20260414_163838"
)
DEFAULT_OUTPUT_ROOT = Path(r"E:\github\UAV-Flow\phase2_5_representation_distillation\runs")
DEFAULT_CONFIG_PATH = Path(r"E:\github\UAV-Flow\phase2_5_representation_distillation\configs\base_config.json")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_config_file(path: Path) -> dict:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Config file not found: {resolved}")
    payload = _read_json(resolved)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object: {resolved}")
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the Phase 2.5 representation distillation student on exported target-conditioned data."
    )
    parser.add_argument("--config_path", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--export_dir", type=Path, default=None)
    parser.add_argument("--output_root", type=Path, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--global_hidden_dim", type=int, default=None)
    parser.add_argument("--candidate_hidden_dim", type=int, default=None)
    parser.add_argument("--fused_hidden_dim", type=int, default=None)
    parser.add_argument("--representation_dim", type=int, default=None)
    parser.add_argument("--stage1_epochs", type=int, default=None)
    parser.add_argument("--stage2_epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save_every_epoch", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    file_cfg = _load_config_file(args.config_path)
    run_name = (
        str(args.run_name)
        if args.run_name
        else str(file_cfg.get("run_name") or f"distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    )
    config = DistillationTrainerConfig(
        export_dir=str(
            (args.export_dir if args.export_dir is not None else Path(str(file_cfg.get("export_dir", DEFAULT_EXPORT_DIR)))).resolve()
        ),
        output_root=str(
            (args.output_root if args.output_root is not None else Path(str(file_cfg.get("output_root", DEFAULT_OUTPUT_ROOT)))).resolve()
        ),
        run_name=run_name,
        top_k=int(args.top_k if args.top_k is not None else file_cfg.get("top_k", 3)),
        batch_size=int(args.batch_size if args.batch_size is not None else file_cfg.get("batch_size", 16)),
        epochs=int(args.epochs if args.epochs is not None else file_cfg.get("epochs", 80)),
        learning_rate=float(
            args.learning_rate if args.learning_rate is not None else file_cfg.get("learning_rate", 1e-3)
        ),
        weight_decay=float(
            args.weight_decay if args.weight_decay is not None else file_cfg.get("weight_decay", 1e-4)
        ),
        patience=int(args.patience if args.patience is not None else file_cfg.get("patience", 12)),
        device=str(args.device if args.device is not None else file_cfg.get("device", "cpu")),
        num_workers=int(args.num_workers if args.num_workers is not None else file_cfg.get("num_workers", 0)),
        dropout=float(args.dropout if args.dropout is not None else file_cfg.get("dropout", 0.1)),
        global_hidden_dim=int(
            args.global_hidden_dim if args.global_hidden_dim is not None else file_cfg.get("global_hidden_dim", 64)
        ),
        candidate_hidden_dim=int(
            args.candidate_hidden_dim
            if args.candidate_hidden_dim is not None
            else file_cfg.get("candidate_hidden_dim", 64)
        ),
        fused_hidden_dim=int(
            args.fused_hidden_dim if args.fused_hidden_dim is not None else file_cfg.get("fused_hidden_dim", 256)
        ),
        representation_dim=int(
            args.representation_dim
            if args.representation_dim is not None
            else file_cfg.get("representation_dim", 128)
        ),
        stage1_epochs=int(args.stage1_epochs if args.stage1_epochs is not None else file_cfg.get("stage1_epochs", 5)),
        stage2_epochs=int(args.stage2_epochs if args.stage2_epochs is not None else file_cfg.get("stage2_epochs", 12)),
        seed=int(args.seed if args.seed is not None else file_cfg.get("seed", 0)),
        save_every_epoch=bool(args.save_every_epoch or file_cfg.get("save_every_epoch", False)),
    )
    summary = train_representation_distillation(config)
    print(
        f"[rep-distill] done: epochs={summary['epochs_completed']} "
        f"best_epoch={summary['best_epoch']} best_selection_metric={summary['best_selection_metric']:.6f}"
    )
    print(f"[rep-distill] output -> {summary['output_dir']}")


if __name__ == "__main__":
    main()
