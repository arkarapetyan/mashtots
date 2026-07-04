# Armenian Letter Recognition

This project trains and runs image classification models for Armenian handwritten letter recognition using the Mashtots dataset. The task is to classify grayscale `64 x 64` character images into one of 78 Armenian character classes.

The codebase includes:

- PyTorch and Lightning training loops.
- A Lightning `DataModule` for downloading, splitting, augmenting, and loading the Mashtots dataset.
- Several model architectures, including a baseline CNN, ResNet-style model, EfficientNet-style model, and EfficientNet with ArcFace.
- Export support for Torch Export (`.pt2`) and ONNX (`.onnx`) models.
- Prediction code that writes Kaggle-style submission CSV files.

## Best Result

| Model | Epoch | Validation Loss | Validation Accuracy | Test Accuracy (Kaggle Public) | Export Format | 
| --- | ---: | ---: | ---: | ---: | --- |
| ResNet | 93 | 0.86 | 0.967 | 0.968  | Torch Export / ONNX | 


Exported versions are available for download as:

```text
[models/mashtots-best-0.97.pt2](https://github.com/arkarapetyan/mashtots/releases/download/v1.0.0/mashtots-best-0.97.pt2)
[models/mashtots-best-0.97.onnx](https://github.com/arkarapetyan/mashtots/releases/download/v1.0.0/mashtots-best-0.97.onnx)
```

## Project Structure

```text
.
├── configs/
│   ├── train_config.yaml      # Training configuration
│   └── test_config.yaml       # Prediction configuration
├── scripts/
│   ├── data/
│   │   ├── dataset.py         # Dataset and LightningDataModule
│   │   └── kaggle_download.py # Kaggle download helper
│   ├── utils/
│   │   ├── metrics.py         # Accuracy metric
│   │   ├── modules.py         # Neural network building blocks
│   │   └── transforms.py      # OpenCV preprocessing helpers
│   ├── model.py               # Model registry and architectures
│   ├── serialized_model.py    # Torch/ONNX export and inference
│   ├── train_config.py        # Training config validation
│   └── test_config.py         # Prediction config validation
├── train.py                   # Training entry point
├── predict.py                 # Prediction entry point
└── README.md
```


## Setup

This project requires Python `>=3.12`.

Install dependencies with `uv`:

```bash
uv sync
```

Or install from `pyproject.toml` with your preferred Python environment manager.

## Kaggle Credentials

The dataset download uses the Kaggle API. Create a `.env` file or export these variables in your shell:

```bash
KAGGLE_USERNAME=your_username
KAGGLE_API_TOKEN=your_token
```

The expected format is shown in `.env.example`.

## Dataset

Training uses the Mashtots Kaggle competition dataset:

```text
mashtots-dataset-v2
```

When the configured `data.data_dir` does not exist, the `MashtotsDataModule` downloads and extracts the competition files automatically.

Expected extracted layout:

```text
data/
├── Train/
│   └── Train/
│       ├── 0/
│       ├── 1/
│       └── ...
└── new_test/
    └── new_test.csv
```

## Training

Training is configured through `configs/train_config.yaml`.

Run:

```bash
uv run python train.py --config-path configs --config-name train_config
```

Useful config values:

- `model.model_name`: one of `lightnet`, `efficientnet`, `resnet`, or `resnet+arcface`.
- `model.num_classes`: number of target classes. The current config uses `78`.
- `data.batch_size`: training batch size.
- `data.val_ratio`: validation split ratio.
- `trainer.max_epochs`: maximum training epochs.
- `export.export_format`: `torch` or `onnx`.

Example override:

```bash
uv run python train.py --config-path configs --config-name train_config \
  model.model_name=efficientnet \
  trainer.max_epochs=100 \
  data.batch_size=128
```

Training logs metrics to MLflow. If using the current `logger.tracking_uri` value, start an MLflow server before training:

```bash
uv run mlflow server --host 127.0.0.1 --port 5000
```

Checkpoints are written to `model_checkpoints/`, and exported models are written to `models/`.

## Prediction

Prediction is configured through `configs/test_config.yaml`.

Run:

```bash
uv run python predict.py --config-name test_config
```

The prediction script:

1. Loads prediction data from `data/new_test/new_test.csv`.
2. Loads the serialized model from `model.model_path`.
3. Runs inference with the selected export backend.
4. Writes a Kaggle-style CSV to `results/`.

Example override:

```bash
uv run python predict.py --config-name test_config \
  model.model_path=models/mashtots-best-0.97.onnx \
  output.filename=predictions.csv
```

The output file has this format:

```csv
Id,Category
1,12
2,45
3,7
```

## Models

Registered model names:

| Name | Description |
| --- | --- |
| `lightnet` | Small residual convolutional classifier. |
| `resnet` | ResNet-18-style classifier adapted for small grayscale images. |
| `efficientnet` | EfficientNet-style classifier using MBConv blocks. |
| `resnet+arcface` | ResNet-18 embedding model with an ArcFace head. |

## Export Formats

| Format | Extension | Backend |
| --- | --- | --- |
| `torch` | `.pt2` | `torch.export` |
| `onnx` | `.onnx` | ONNX Runtime |

