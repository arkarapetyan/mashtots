"""Pydantic models for validating prediction configuration."""

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from .serialized_model import list_available_extensions


class ModelConfig(BaseModel):
    """Serialized model settings used for prediction.

    Attributes:
        model_paths: One or more exported model files (``.pt2`` or
            ``.onnx``). Several paths form an ensemble whose class
            probabilities are averaged.
        weights: Optional weight per model for the probability average.
            Defaults to equal weights.
        tta: Whether to average each model over small image shifts.
        tta_shift_px: Shift distance in pixels used by TTA.
    """

    model_paths: list[Path] = Field(min_length=1)
    weights: Optional[list[float]] = None
    tta: bool = False
    tta_shift_px: int = Field(default=2, ge=1, le=8)

    @field_validator("model_paths")
    @classmethod
    def validate_model_paths(cls, paths: list[Path]) -> list[Path]:
        """Validate that every serialized model exists and has a known format.

        Args:
            paths: Model paths from the configuration.

        Returns:
            The validated paths, made absolute.

        Raises:
            FileNotFoundError: If a path does not exist.
            ValueError: If a file extension is not registered.
        """

        resolved = []
        for path in paths:
            path = path.expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(f"File Does Not Exist: {path!r}")
            if path.suffix not in list_available_extensions():
                raise ValueError(f"Invalid File Extension: {path.suffix}")
            resolved.append(path)
        return resolved

    @model_validator(mode="after")
    def validate_weights(self) -> "ModelConfig":
        """Check that weights, when given, match the models and are usable.

        Returns:
            The validated configuration.

        Raises:
            ValueError: If the weights are invalid.
        """

        if self.weights is not None:
            if len(self.weights) != len(self.model_paths):
                raise ValueError(
                    f"Got {len(self.weights)} weights for "
                    f"{len(self.model_paths)} models"
                )
            if min(self.weights) < 0 or sum(self.weights) <= 0:
                raise ValueError("weights must be non-negative and not all zero")
        return self


class DataConfig(BaseModel):
    """Prediction dataset settings.

    Attributes:
        data_dir: Root directory containing the dataset used for prediction.
        batch_size: Number of samples per prediction batch.
        num_workers: Number of data loader worker processes.
        pin_memory: Whether data loaders should pin CPU memory.
    """

    data_dir: Path
    batch_size: int = Field(gt=0)
    num_workers: int = Field(ge=0)
    pin_memory: bool = Field()


class OutputConfig(BaseModel):
    """Prediction output settings.

    Attributes:
        output_dir: Directory where the predictions CSV is saved.
        filename: Name of the predictions CSV file.
    """

    output_dir: Path
    filename: str


class TestConfig(BaseModel):
    """Complete prediction configuration tree."""

    seed: int = Field(ge=0)
    model: ModelConfig
    data: DataConfig
    output: OutputConfig
