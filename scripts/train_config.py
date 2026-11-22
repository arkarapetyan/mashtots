"""Pydantic models for validating training configuration."""

from pathlib import Path
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from urllib.parse import urlparse

from .model import list_available_models
from .serialized_model import list_available_formats


class ModelConfig(BaseModel):
    """Model and optimizer settings used during training.

    The architecture-specific values live in ``configs/model/<name>.yaml``;
    pick one with ``model=<name>`` on the command line.

    Attributes:
        model_name: Registered model architecture name.
        in_channels: Number of image input channels.
        num_classes: Number of output classes.
        dropout_rate: Dropout probability used by supported models.
        optimizer_lr: Learning rate passed to the optimizer and scheduler.
        optimizer_weight_decay: Weight decay passed to the optimizer.
        model_kwargs: Extra architecture-specific keyword arguments forwarded
            to the model constructor, e.g. ``depth`` for ``vit`` or
            ``arcface_m`` for ``resnet+arcface``.
    """

    model_name: str
    in_channels: int = Field(gt=0)
    num_classes: int = Field(gt=1)
    dropout_rate: float = Field(ge=0.0, le=1.0)
    optimizer_lr: float = Field(gt=0)
    optimizer_weight_decay: float = Field(gt=0)
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("model_kwargs")
    @classmethod
    def validate_model_kwargs(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Reject extra arguments that clash with the named model settings.

        Args:
            v: Extra model keyword arguments from the configuration.

        Returns:
            The validated keyword arguments.

        Raises:
            ValueError: If a key duplicates an argument ``train.py`` already
                passes to the model.
        """

        reserved = {
            "model_name",
            "in_channels",
            "num_classes",
            "dropout_rate",
            "optimizer_lr",
            "optimizer_weight_decay",
            "max_epochs",
            "loss",
        }
        clashes = reserved.intersection(v)
        if clashes:
            raise ValueError(
                f"{sorted(clashes)} already have their own config fields "
                "(under `model`, `loss` or `trainer`); remove them from `model_kwargs`"
            )
        return v

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


class AugmentationConfig(BaseModel):
    """Training augmentation settings (see ``build_train_transform``).

    Attributes:
        degrees: Maximum rotation in degrees.
        translate: Maximum horizontal and vertical shift as image fractions.
        scale: Range of random scaling. Kept narrow because letter size helps
            tell some upper/lowercase pairs apart.
        shear: Maximum shear in degrees.
        elastic_alpha: Elastic warp strength.
        elastic_sigma: Smoothness of the elastic warp in pixels.
        elastic_p: Probability of applying the elastic warp (0 disables it).
        noise_std: Standard deviation of additive pixel noise (0 disables it).
    """

    degrees: float = Field(default=10.0, ge=0)
    translate: tuple[float, float] = (0.1, 0.1)
    scale: tuple[float, float] = (0.95, 1.05)
    shear: float = Field(default=10.0, ge=0)
    elastic_alpha: float = Field(default=40.0, ge=0)
    elastic_sigma: float = Field(default=5.0, gt=0)
    elastic_p: float = Field(default=0.5, ge=0, le=1)
    noise_std: float = Field(default=0.02, ge=0)


class DataConfig(BaseModel):
    """Dataset and data loader settings for training.

    Attributes:
        data_dir: Root directory containing or receiving the dataset.
        batch_size: Number of samples per training batch.
        num_workers: Number of data loader worker processes.
        val_ratio: Fraction of the labeled images reserved for validation.
        test_ratio: Fraction of the labeled images held out as a test set and
            evaluated once after training. ``0`` skips testing.
        pin_memory: Whether data loaders should pin CPU memory.
        augmentation: Training augmentation settings.
    """

    data_dir: Path
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    val_ratio: float = Field(gt=0, lt=1)
    test_ratio: float = Field(default=0.0, ge=0, lt=1)
    pin_memory: bool = Field()
    augmentation: AugmentationConfig = Field(default_factory=AugmentationConfig)

    @model_validator(mode="after")
    def validate_split(self) -> "DataConfig":
        """Ensure some data is left for training.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If ``val_ratio + test_ratio >= 1``.
        """

        if self.val_ratio + self.test_ratio >= 1:
            raise ValueError("val_ratio + test_ratio must be below 1")
        return self


class LossConfig(BaseModel):
    """Settings for ``ConfusionAwareLoss``.

    Attributes:
        label_smoothing: Label smoothing of the cross-entropy term.
        pair_weight: Weight of the pairwise term that separates each letter
            from its usual rivals. ``0`` gives plain cross-entropy.
        case_pair_offset: Label distance between a letter's upper and lower
            case (39 for Mashtots). Enables the case-twin rival and the
            ``*_letter_acc`` / ``*_case_swap`` metrics. ``None`` disables both.
        case_prior: Weight of the case twin among each letter's rivals,
            relative to the observed confusions (which weigh 1 together).
        adaptive: Update class weights and rivals from the training confusion
            matrix after every epoch.
        class_weight_strength: How much more weight error-prone letters get.
            ``0`` keeps uniform class weights.
        max_class_weight: Cap on a letter's weight relative to an error-free
            letter.
        top_k: Number of most-confused classes kept as rivals per letter.
        momentum: Smoothing of the confusion statistics across epochs.
        adapt_start_epoch: First epoch whose statistics are used.
    """

    label_smoothing: float = Field(default=0.1, ge=0, lt=1)
    pair_weight: float = Field(default=0.0, ge=0)
    case_pair_offset: Optional[int] = Field(default=None, gt=0)
    case_prior: float = Field(default=1.0, ge=0)
    adaptive: bool = False
    class_weight_strength: float = Field(default=1.0, ge=0)
    max_class_weight: float = Field(default=3.0, ge=1)
    top_k: int = Field(default=3, ge=1)
    momentum: float = Field(default=0.7, ge=0, lt=1)
    adapt_start_epoch: int = Field(default=5, ge=0)


class WeightAveragingConfig(BaseModel):
    """Exponential moving average (EMA) of the model weights.

    When enabled, validation, checkpoints and the exported model all use the
    averaged weights.

    Attributes:
        enabled: Whether to keep an EMA of the weights.
        decay: EMA decay per optimizer step. ``0.9995`` averages over roughly
            the last 2,000 steps (about 4 epochs at batch size 128).
    """

    enabled: bool = False
    decay: float = Field(default=0.9995, gt=0, lt=1)


class CallbackConfig(BaseModel):
    """Callback settings for checkpointing, early stopping and diagnostics.

    Attributes:
        checkpoints_dir: Directory where model checkpoints are written.
        early_stop_patience: Number of validation checks without improvement
            before early stopping.
        confusion_matrix_every_n_epochs: How often the validation confusion
            matrix and most-confused pairs are logged to MLflow.
        weight_averaging: EMA settings.
        class_names: Display name of every class, in label order.
    """

    checkpoints_dir: Path
    early_stop_patience: int = Field(ge=1)
    confusion_matrix_every_n_epochs: int = Field(default=5, ge=1)
    weight_averaging: WeightAveragingConfig = Field(
        default_factory=WeightAveragingConfig
    )
    class_names: list[str]


class LoggerConfig(BaseModel):
    """MLflow logger settings.

    Attributes:
        experiment_name: MLflow experiment name.
        tracking_uri: MLflow tracking URI.
        run_name: MLflow run name. Defaults to the model name.
    """

    experiment_name: str
    tracking_uri: str
    run_name: Optional[str] = None

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
        precision: Numeric precision. ``"auto"`` picks ``"bf16-mixed"`` on
            GPUs with native bfloat16 (Ampere and newer), ``"16-mixed"`` on
            older CUDA GPUs, and ``"32"`` on CPU.
        deterministic: Use deterministic algorithms (reproducible, but can be
            slower, e.g. for attention kernels).
        accumulate_grad_batches: Number of batches whose gradients are summed
            before each optimizer step. ``data.batch_size=64`` with ``2`` trains
            like batch size 128 with half the activation memory.
    """

    max_epochs: int = Field(ge=1)
    precision: Literal["auto", "32", "16-mixed", "bf16-mixed"] = "auto"
    deterministic: bool = True
    accumulate_grad_batches: int = Field(default=1, ge=1)


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
    loss: LossConfig = Field(default_factory=LossConfig)
    callback: CallbackConfig
    logger: LoggerConfig
    trainer: TrainerConfig
    export: ExportConfig

    @model_validator(mode="after")
    def validate_class_layout(self) -> "TrainConfig":
        """Check that class names and case pairs match the number of classes.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If the settings disagree with ``model.num_classes``.
        """

        num_classes = self.model.num_classes
        if len(self.callback.class_names) != num_classes:
            raise ValueError(
                f"callback.class_names has {len(self.callback.class_names)} "
                f"entries for {num_classes} classes"
            )
        offset = self.loss.case_pair_offset
        if offset is not None and 2 * offset != num_classes:
            raise ValueError(
                f"loss.case_pair_offset={offset} requires model.num_classes="
                f"{2 * offset}, got {num_classes}"
            )
        return self
