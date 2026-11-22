# Changelog

## [2.0.0] - 2026-11-22

Kaggle score 0.97598 private / 0.97554 public with a ResNet-18 + Vision Transformer ensemble. Full notes: [docs/releases/v2.0.0.md](docs/releases/v2.0.0.md).

### Added
- Vision Transformer with a convolutional patch stem, AdamW, warmup + cosine schedule.
- Weight averaging (EMA), mixed precision chosen automatically, gradient accumulation.
- Confusion-aware loss and case-confusion metrics; most-confused pairs logged to MLflow.
- Ensemble prediction with weights and shift test-time augmentation.
- Per-model Hydra recipes in `configs/model/`, including `efficientnet_lowmem`.
- Labeled hold-out test split (`data.test_ratio`).

### Changed
- Elastic augmentation now warps strokes smoothly and runs about 6× faster.
- Narrower scale augmentation; shear up to 10°.
- Models are exported on CPU so `.pt2` files load on any machine.
- `predict.py` takes `model.model_paths=[...]`; models are selected with `model=<name>`.
- `val_acc` is plain accuracy.
- PyTorch comes from PyPI.

### Fixed
- Early stopping used `mode="min"` on `val_acc`.
- `trainer.test()` crashed on its logging call.
- Dataset download skipped an existing empty `data/` folder and failed silently on Kaggle authentication errors.
- Training was not seeded; checkpoints did not store hyperparameters; `num_workers=0` crashed.

## [1.0.0]

First release: ResNet-18 with 0.968 Kaggle public score.
