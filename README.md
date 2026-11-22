# Armenian Letter Recognition

Image classification of handwritten Armenian letters on the [Mashtots dataset](https://www.kaggle.com/competitions/mashtots-dataset-v2): grayscale `64 x 64` images, 78 classes (39 uppercase letters with labels `0-38`, and the same 39 letters in lowercase with labels `39-77`).

The repository contains a complete training and inference pipeline:

- PyTorch Lightning training, configured with Hydra and tracked with MLflow.
- Six architectures: ResNet-18, EfficientNet, a Vision Transformer (ViT), ResNet with an ArcFace head, and two small CNNs.
- Handwriting-specific augmentation, a confusion-aware loss for look-alike letters, mixed precision, and weight averaging (EMA).
- Export to Torch Export (`.pt2`) and ONNX, and single-model or ensemble prediction with optional test-time augmentation.

## Results

Kaggle leaderboard scores (accuracy) and accuracy on the held-out validation split (7,006 images):

| Model | Validation | Kaggle private | Kaggle public |
| --- | ---: | ---: | ---: |
| **Ensemble: ResNet-18 + ViT, shift TTA** | – | **0.97598** | **0.97554** |
| ResNet-18 | 0.9719 | 0.97071 | 0.97101 |
| Vision Transformer | 0.9669 | 0.96954 | 0.96961 |
| ResNet-18 from v1.0.0 (previous release) | 0.967 | – | 0.968 |

The ensemble cuts the private-leaderboard error of the best single model from 2.93% to 2.40%, about 18% fewer mistakes. Single models were scored without test-time augmentation and the ensemble with it, so its gain combines both.

### Download

Pretrained models from the [v2.0.0 release](https://github.com/arkarapetyan/mashtots/releases/tag/v2.0.0):

| Model | Torch Export | ONNX |
| --- | --- | --- |
| ResNet-18 | [mashtots-resnet.pt2](https://github.com/arkarapetyan/mashtots/releases/download/v2.0.0/mashtots-resnet.pt2) | [mashtots-resnet.onnx](https://github.com/arkarapetyan/mashtots/releases/download/v2.0.0/mashtots-resnet.onnx) |
| Vision Transformer | [mashtots-vit.pt2](https://github.com/arkarapetyan/mashtots/releases/download/v2.0.0/mashtots-vit.pt2) | [mashtots-vit.onnx](https://github.com/arkarapetyan/mashtots/releases/download/v2.0.0/mashtots-vit.onnx) |

Both formats run on CPU or GPU. To reproduce the best submission, put the two `.pt2` files in `models/` and run the ensemble command under [Prediction and Ensembles](#prediction-and-ensembles).

### Where the models still fail

The most frequent validation mistakes of the ResNet-18 at its best epoch (197 errors in total):

| True → predicted | Count | Share of the true class |
| --- | ---: | ---: |
| Ձ → Ռ | 13 | 14.8% |
| ժ → յ | 7 | 7.5% |
| յ → ժ | 6 | 6.5% |
| Գ → գ | 6 | 6.5% |
| Ֆ → ֆ | 5 | 5.7% |
| Ռ → Ձ | 5 | 5.9% |

Most errors are pairs of letters that look nearly identical in handwriting (Ձ/Ռ, ժ/յ, հ/ի) and upper/lowercase forms of the same letter (Գ/գ, Ֆ/ֆ). Predicting the right letter in the wrong case accounts for 0.59% of validation images, about a fifth of the ResNet's errors; ignoring case, its accuracy is 97.77%.

## Project Structure

```text
.
├── configs/
│   ├── train_config.yaml          # Shared training settings
│   ├── model/                     # One recipe per architecture, selected with model=<name>
│   │   ├── resnet.yaml
│   │   ├── efficientnet.yaml
│   │   ├── efficientnet_lowmem.yaml
│   │   ├── vit.yaml
│   │   ├── resnet_arcface.yaml
│   │   ├── lightnet.yaml
│   │   └── baseconvnet.yaml
│   └── test_config.yaml           # Prediction / ensemble settings
├── scripts/
│   ├── data/
│   │   ├── dataset.py             # Dataset classes (image folders and CSV)
│   │   ├── data_module.py         # LightningDataModule: download, splits, augmentation
│   │   └── kaggle_download.py     # Kaggle download helper
│   ├── utils/
│   │   ├── callbacks.py           # Confusion matrix + most-confused pairs logging
│   │   ├── losses.py              # Confusion-aware loss
│   │   ├── modules.py             # Building blocks (residual, MBConv, transformer, ArcFace)
│   │   ├── reproducibility.py     # Seeding helpers
│   │   ├── results_saving.py      # Submission CSV writer
│   │   └── transforms.py          # Fast elastic warp, noise, OpenCV helpers
│   ├── model.py                   # Model registry and architectures
│   ├── ensemble.py                # Ensemble and test-time augmentation
│   ├── serialized_model.py        # Torch Export / ONNX export and inference
│   ├── train_config.py            # Training config validation (Pydantic)
│   └── test_config.py             # Prediction config validation (Pydantic)
├── EDA.ipynb                      # Exploratory data analysis
├── train.py                       # Training entry point
└── predict.py                     # Prediction entry point
```

## Setup

Requires Python `>=3.12` and [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

PyTorch is installed from PyPI; on Linux these are the CUDA 12.8 builds, which run on any NVIDIA driver that supports CUDA 12.x (driver 525 or newer) and fall back to CPU without a GPU.

### Kaggle Credentials

The dataset is downloaded with the Kaggle API on first use. Put your credentials in a `.env` file in the repository root (see `env.example`):

```bash
KAGGLE_API_TOKEN=your_api_token      # from https://www.kaggle.com/settings/api
# or, with a legacy kaggle.json key:
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_key
```

You also need to join the competition and accept its rules on Kaggle.

## Dataset

When `data.data_dir` does not contain the extracted dataset, it is downloaded and extracted automatically (also into an existing empty folder):

```text
data/
├── Train/Train/<class>/*.png     # 70,060 labeled images, one folder per class
└── new_test/new_test.csv         # unlabeled Kaggle test set, one row per image
```

The labeled images are split with a seeded, stratified split into 63,054 training and 7,006 validation images (`data.val_ratio: 0.1`). Setting `data.test_ratio > 0` also holds out a labeled test set, evaluated with the best checkpoint after training. The Kaggle CSV is only used for prediction; it is read by column name, using an `Id` column and a label column when present.

## Experiment Tracking

Training logs to MLflow at `logger.tracking_uri` (default `http://localhost:5000`). Start the server from the repository root before training:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlflow.db --artifacts-destination ./mlartifacts
```

Then open http://127.0.0.1:5000. Each run records its hyperparameters, per-epoch metrics, learning rate, a confusion-matrix heatmap and a list of the most-confused letter pairs every 5 epochs, and the exported model.

## Training

Shared settings live in `configs/train_config.yaml`. Each file in `configs/model/` is a complete recipe for one architecture and is selected with `model=<name>`:

```bash
uv run python train.py model=resnet
uv run python train.py model=vit
uv run python train.py model=efficientnet_lowmem
```

| Recipe | Optimizer | Peak LR | Weight decay | Head dropout | Epochs | Weight averaging |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `resnet` | SGD (Nesterov) + OneCycle | 0.1 | 5e-4 | 0.4 | 100 | off |
| `efficientnet` | SGD (Nesterov) + OneCycle | 0.1 | 5e-4 | 0.2 | 100 | off |
| `efficientnet_lowmem` | same, run as 2 × 64 images per step | 0.1 | 5e-4 | 0.2 | 100 | off |
| `vit` | AdamW + warmup/cosine | 5e-4 | 0.05 | 0.1 | 200 | on (decay 0.9998) |
| `resnet_arcface` | SGD (Nesterov) + OneCycle | 0.1 | 5e-4 | 0.4 | 100 | off |
| `lightnet`, `baseconvnet` | SGD (Nesterov) + OneCycle | 0.1 | 5e-4 | 0.3 | 100 | off |

All recipes use batch size 128 and label smoothing 0.1. Command-line values override both the recipe and the shared settings, e.g. `trainer.max_epochs=150 data.batch_size=256`.

Useful settings:

- `model.model_kwargs.*`: architecture-specific arguments, e.g. `model.model_kwargs.depth=12` for the ViT or `model.model_kwargs.arcface_m=0.3` for ArcFace.
- `data.augmentation.*`, `loss.*`, `callback.weight_averaging.*`: see the sections below.
- `trainer.precision`, `trainer.deterministic`, `trainer.accumulate_grad_batches`: see [Speed and Memory](#speed-and-memory).
- `export.export_format` (`torch` or `onnx`) and `export.filename`.

The checkpoint with the best validation accuracy is kept in `model_checkpoints/` and exported to `models/` after training. Exports are made on CPU, so they load on machines with or without a GPU. Early stopping (`callback.early_stop_patience: 100`) only ends a run after 100 epochs without improvement, because both learning-rate schedules reach their best accuracy near the end.

### Augmentation

Training images go through a random affine transform (rotation up to 10°, shift up to 10%, scaling 0.95–1.05, shear up to 10° for writer slant), a smooth elastic warp of the strokes (`alpha=40`, `sigma=5`, applied to half of the images), and light pixel noise. Flips are never used: they turn letters into other letters or non-letters. The scaling range is kept narrow because letter size is one of the few cues that separates some uppercase and lowercase forms.

The elastic warp uses `FastElasticTransform`, which produces the same result as torchvision's `ElasticTransform` about 6× faster by blurring the random displacement field with two 1-D passes instead of one 2-D pass.

### Loss and Case Confusion

The loss is cross-entropy with label smoothing plus a pairwise term that pushes each letter's score above the scores of its usual "rivals":

- Every letter starts with one rival: the same letter in the other case (`loss.case_pair_offset: 39`).
- With `loss.adaptive: true`, after each epoch the training confusion matrix adds the `loss.top_k` letters each class is most often mistaken for, and gives error-prone classes more weight (`loss.class_weight_strength`, capped by `loss.max_class_weight`). Only training data is used, so the validation set stays clean for model selection.

Set `loss.pair_weight=0 loss.adaptive=false` for plain cross-entropy. Two metrics track case confusion directly:

- `val_letter_acc`: accuracy when upper/lowercase are not distinguished.
- `val_case_swap`: fraction of validation images predicted as the right letter in the wrong case.

### Weight Averaging

`callback.weight_averaging.enabled=true` keeps an exponential moving average of the weights (decay `callback.weight_averaging.decay`). Validation, checkpoint selection, and the exported model then all use the averaged weights. The `vit` recipe turns it on with decay `0.9998`.

### Speed and Memory

- `trainer.precision: auto` (default) trains with mixed precision: `bf16-mixed` on GPUs with native bfloat16 (RTX 30xx and newer, A100, L4), `16-mixed` on older CUDA GPUs, and `32` on CPU. Weights stay in fp32, so checkpoints and exports are fp32.
- `trainer.deterministic: true` (default) makes runs reproducible; `false` lets cuDNN pick faster kernels.
- At batch size 128, ResNet and the ViT store about 1.5–2 GB of activations, but EfficientNet stores about 7 GB because its early layers run at full resolution. Use `model=efficientnet_lowmem`, and avoid running EfficientNet in parallel with another training. `trainer.accumulate_grad_batches` works the same way for any model: halve `data.batch_size` and double the accumulation to keep training equivalent.

## Prediction and Ensembles

Prediction is configured in `configs/test_config.yaml`. Give one model, or several to average their class probabilities; `.pt2` and `.onnx` files can be mixed. CSVs are written to `results/`.

```bash
# Single models
uv run python predict.py 'model.model_paths=[models/mashtots-resnet.pt2]' output.filename=resnet.csv
uv run python predict.py 'model.model_paths=[models/mashtots-vit.pt2]' output.filename=vit.csv

# Best submission: ResNet-18 + ViT with shift test-time augmentation
uv run python predict.py \
  'model.model_paths=[models/mashtots-resnet.pt2,models/mashtots-vit.pt2]' \
  model.tta=true output.filename=ensemble_resnet_vit.csv

# Weighted ensemble
uv run python predict.py \
  'model.model_paths=[models/mashtots-resnet.pt2,models/mashtots-vit.pt2]' \
  'model.weights=[1.5,1.0]' output.filename=weighted.csv
```

With `model.tta=true`, every model also sees the images shifted by `model.tta_shift_px` (default 2) pixels in four directions. For ensembles, the script prints how often each model agrees with the ensemble.

Output format:

```csv
Id,Category
1,12
2,45
3,7
```

## Reproducing v2.0.0

```bash
uv run python train.py model=resnet export.filename=mashtots-resnet logger.experiment_name=mashtots
uv run python train.py model=vit export.filename=mashtots-vit logger.experiment_name=mashtots
uv run python predict.py \
  'model.model_paths=[models/mashtots-resnet.pt2,models/mashtots-vit.pt2]' \
  model.tta=true output.filename=ensemble_resnet_vit.csv
```

Training took about 4 hours for the ResNet (100 epochs) and 7 hours for the ViT, which reached its best validation accuracy at epoch 75 and was stopped by early stopping at epoch 175. An EfficientNet run (`efficientnet_lowmem`) was stopped after 62 of 100 epochs at 0.9587 validation accuracy and is not part of the release.

## Models

| Config (`model=`) | Registered name | Description |
| --- | --- | --- |
| `resnet` | `resnet` | ResNet-18 adapted to 64 × 64 grayscale images: 3 × 3 stride-1 stem, no max-pooling. 11.3M parameters. |
| `vit` | `vit` | Vision Transformer with a convolutional patch stem (4 × 4 patches, 256 tokens), 8 encoder blocks, width 192, 6 heads, average pooling. 3.7M parameters. |
| `efficientnet`, `efficientnet_lowmem` | `efficientnet` | EfficientNet-B0-style network of MBConv blocks with squeeze-and-excitation, kept at higher resolution for small images. 4.1M parameters. |
| `resnet_arcface` | `resnet+arcface` | ResNet-18 embedding with an ArcFace (additive angular margin) head. |
| `lightnet` | `lightnet` | Small residual CNN. |
| `baseconvnet` | `baseconvnet` | Baseline CNN. |

## Export Formats

| Format | Extension | Backend |
| --- | --- | --- |
| `torch` | `.pt2` | `torch.export` |
| `onnx` | `.onnx` | ONNX Runtime (CUDA provider when `onnxruntime-gpu` is installed, otherwise CPU) |

## License

GNU General Public License v3.0; see [LICENSE](LICENSE).
