"""Pydantic models for validating prediction configuration."""

from pathlib import Path
from pydantic import BaseModel, Field, field_validator

from .serialized_model import list_available_extensions


class ModelConfig(BaseModel):
    """Serialized model settings used for prediction.

    Attributes:
        export_format: Registered serialization format name.
        model_path: Path to the exported model file.
    """

    model_path: Path

    @field_validator("model_path")
    @classmethod
    def validate_model_path(cls, v: Path) -> Path:
        """Validate that the serialized model path exists.

        Args:
            v: Model path from the configuration.

        Returns:
            The validated model path.

        Raises:
            FileNotFoundError: If the path does not exist.
            ValueError: If the File Extension is not registered.
        """

        if not v.exists():
            raise FileNotFoundError(f"File Does Not Exist: {v!r}")
        if v.suffix not in list_available_extensions():
            raise ValueError(f"Invalid File Extension: {v.suffix}")
        return v


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
