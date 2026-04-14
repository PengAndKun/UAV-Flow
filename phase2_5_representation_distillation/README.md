# Phase 2.5 Representation Distillation

This directory contains the first implementation blocks for the Phase 2.5
representation distillation pipeline.

Current modules:

- `label_schema.py`
  - fixed label spaces and encoding/decoding helpers
- `dataset.py`
  - export-dataset reader for `train.jsonl` / `val.jsonl`

The implementation order follows:

1. label schema
2. dataset reader
3. feature builder
4. student model
5. loss builder
6. trainer
7. evaluator

