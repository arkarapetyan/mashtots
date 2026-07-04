"""Pydantic models for validating training configuration."""

from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator
from urllib.parse import urlparse

from .model import list_available_models
from .serialized_model import list_available_formats


class ModelConfig(BaseModel):
    """Model and optimizer settings used during training.

    Attributes:
        model_name: Registered model architecture name.
        in_channels: Number of image input channels.
        num_classes: Number of output classes.
        dropout_rate: Dropout probability used by supported models.
        optimizer_lr: Learning rate passed to the optimizer and scheduler.
        optimizer_weight_decay: Weight decay passed to the optimizer.
        lr_scheduler_factor: Multiplicative scheduler factor.
        lr_scheduler_patience: Scheduler patience in epochs.
    """

    model_name: str
    in_channels: int = Field(gt=0)
    num_classes: int = Field(gt=1)
    dropout_rate: float = Field(ge=0.0, le=1.0)
    optimizer_lr: float = Field(gt=0)
    optimizer_weight_decay: float = Field(gt=0)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate that the configured model name is registered.

        Args:
            v: Model name from the configuration.

        Returns:
            The validated model name.

        Raises:
            ValueError: If the model name is not registered.
        """

        if v not in list_available_models():
            raise ValueError(f"Invalid Model Name: {v}")
        return v


class DataConfig(BaseModel):
    """Dataset and data loader settings for training.

    Attributes:
        data_dir: Root directory containing or receiving the dataset.
        batch_size: Number of samples per training batch.
        num_workers: Number of data loader worker processes.
        val_ratio: Fraction of the training data reserved for validation.
        pin_memory: Whether data loaders should pin CPU memory.
    """

    data_dir: Path
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    val_ratio: float = Field(gt=0, lt=1)
    pin_memory: bool = Field()


class CallbackConfig(BaseModel):
    """Callback settings for checkpointing and early stopping.

    Attributes:
        checkpoints_dir: Directory where model checkpoints are written.
        early_stop_patience: Number of validation checks without improvement
            before early stopping.
    """

    checkpoints_dir: Path
    early_stop_patience: int = Field(ge=1)
    class_names: list[str]


class LoggerConfig(BaseModel):
    """MLflow logger settings.

    Attributes:
        experiment_name: MLflow experiment name.
        tracking_uri: MLflow tracking URI.
    """

    experiment_name: str
    tracking_uri: str

    @field_validator("tracking_uri")
    @classmethod
    def validate_tracking_uri(cls, v: str) -> str:
        """Validate supported MLflow tracking URI formats.

        Args:
            v: Tracking URI from the configuration.

        Returns:
            The validated tracking URI.

        Raises:
            ValueError: If the URI format is unsupported.
        """

        parsed = urlparse(v)

        # Accept file-based URIs
        if v.startswith("file:"):
            return v

        # Accept sqlite URIs
        if v.startswith("sqlite:"):
            return v

        # Accept databricks
        if v == "databricks":
            return v

        # Accept http/https with netloc
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return v

        raise ValueError("Invalid MLflow tracking URI")


class TrainerConfig(BaseModel):
    """Lightning trainer settings.

    Attributes:
        max_epochs: Maximum number of training epochs.
        precision: Numeric precision setting accepted by Lightning.
    """

    max_epochs: int = Field(ge=1)
    precision: Literal["32", "16-mixed", "bf16-mixed"]


class ExportConfig(BaseModel):
    """Model export settings.

    Attributes:
        model_dir: Directory where the exported model is saved.
        filename: Optional exported model filename without extension.
        export_format: Registered export format name.
    """

    model_dir: Path
    filename: str | None
    export_format: str

    @field_validator("export_format")
    @classmethod
    def validate_export_format(cls, v: str) -> str:
        """Validate that the configured export format is registered.

        Args:
            v: Export format name from the configuration.

        Returns:
            The validated export format name.

        Raises:
            ValueError: If the export format is not registered.
        """

        if v not in list_available_formats():
            raise ValueError(f"Invalid Export Format: {v}")
        return v


class TrainConfig(BaseModel):
    """Complete training configuration tree."""

    seed: int = Field(ge=0)
    model: ModelConfig
    data: DataConfig
    callback: CallbackConfig
    logger: LoggerConfig
    trainer: TrainerConfig
    export: ExportConfig
