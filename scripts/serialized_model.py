"""Serialization backends for exporting and loading trained models."""

from __future__ import annotations

import os
import torch
import onnxruntime as ort

from torch.export import Dim
from lightning import LightningModule
from typing import Type, Callable
from pathlib import Path


_SERIALIZATION_FORMATS: dict[str, Type[BaseSerializedModel]] = {}

_FILE_EXTENSIONS: dict[str, Type[BaseSerializedModel]] = {}


def register_format(
    name: str,
    extensions: list[str],
) -> Callable[[Type], Type]:
    """Register a serialized model backend under a format name.

    Args:
        name: Format name used in configuration, such as ``"torch"`` or
            ``"onnx"``.

    Returns:
        Decorator that registers and returns the backend class.
    """

    def decorator(cls: Type[BaseSerializedModel]) -> Type[BaseSerializedModel]:
        """Add a backend class to the serialization registry.

        Args:
            cls: Serialized model backend class.

        Returns:
            The same class, unchanged.
        """

        _SERIALIZATION_FORMATS[name] = cls
        for ext in extensions:
            _FILE_EXTENSIONS[ext] = cls

        return cls

    return decorator


def get_export_format(name: str) -> Type[BaseSerializedModel]:
    """Return the serialized model backend registered for a format.

    Args:
        name: Registered export format name.

    Returns:
        Serialized model backend class.

    Raises:
        ValueError: If no backend is registered under ``name``.
    """

    if name not in _SERIALIZATION_FORMATS:
        raise ValueError(f"Invalid Model Name: {name}")
    return _SERIALIZATION_FORMATS[name]


def list_available_formats() -> frozenset[str]:
    """List names of registered serialization formats.

    Returns:
        Frozen set of available format names.
    """

    return frozenset(_SERIALIZATION_FORMATS.keys())


def get_extension_format(ext: str) -> Type[BaseSerializedModel]:
    """Return the serialized model backend registered for a file extension.

    Args:
        ext: Registered file extension.

    Returns:
        Serialized model backend class.

    Raises:
        ValueError: If no backend is registered under ``ext``.
    """

    if ext not in _FILE_EXTENSIONS:
        raise ValueError(f"Invalid File Extension: {ext}")
    return _FILE_EXTENSIONS[ext]


def list_available_extensions() -> frozenset[str]:
    """List names of registered file extensions.

    Returns:
        Frozen set of available file extensions.
    """

    return frozenset(_FILE_EXTENSIONS.keys())


class BaseSerializedModel:
    """Base interface for serialized model backends.

    Args:
        model_path: Path to the serialized model file.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
    """

    def __init__(self, model_path: Path):
        """Validate that a serialized model path exists.

        Args:
            model_path: Path to the serialized model file.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
        """

        if not model_path.exists():
            raise FileNotFoundError(f"File Not Found: {model_path!r}")

    def export_model(
        model: LightningModule,
        example_input: torch.Tensor,
        save_dir: Path,
        filename: str,
        **kwargs,
    ) -> Path:
        """Export a Lightning model to a backend-specific serialized file.

        Args:
            model: Trained Lightning model to export.
            example_input: Example input batch used to trace/export shapes.
            save_dir: Directory where the serialized file is written.
            filename: Filename without extension.
            **kwargs: Backend-specific export options.

        Returns:
            Path to the exported model file.
        """

        pass

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """Run inference with a serialized model backend.

        Args:
            input_data: Input image batch.

        Returns:
            Model logits for the input batch.
        """

        pass


@register_format("torch", [".pt", ".pt2"])
class TorchSerializedModel(BaseSerializedModel):
    """Torch Export backend for loading ``.pt2`` models and inference."""

    def __init__(self, model_path: Path):
        """Load a Torch Export model from disk.

        Args:
            model_path: Path to a ``torch.export`` serialized model.
        """

        super(TorchSerializedModel, self).__init__(model_path)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        exported_program = torch.export.load(model_path)
        self._model = exported_program.module()
        self._model.to(self._device)

    def export_model(
        model: LightningModule,
        example_input: torch.Tensor,
        save_dir: Path,
        filename: str,
        **kwargs,
    ) -> Path:
        """Export a Lightning model using ``torch.export``.

        Args:
            model: Trained Lightning model to export.
            example_input: Example input batch used for export.
            save_dir: Directory where the ``.pt2`` file is written.
            filename: Filename without extension.
            **kwargs: Unused backend-specific options.

        Returns:
            Path to the exported ``.pt2`` file.
        """

        if not save_dir.exists():
            save_dir.mkdir()
        save_path = save_dir / f"{filename}.pt2"
        model.eval()
        example_input = example_input.to(model.device)
        exported = torch.export.export(
            model,
            args=(example_input,),
            dynamic_shapes={"x": {0: Dim("batch_size", min=1, max=1024)}},
        )
        torch.export.save(
            exported,
            save_path,
        )
        return save_path

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """Run inference with the loaded Torch Export model.

        Args:
            input_data: Input image batch.

        Returns:
            Model logits on the configured device.
        """

        input_data = input_data.to(self._device)
        with torch.no_grad():
            logits = self._model(input_data)
        return logits


@register_format("onnx", [".onnx"])
class OnnxSerializedModel(BaseSerializedModel):
    """ONNX Runtime backend for loading ``.onnx`` models and inference."""

    def __init__(self, model_path: Path):
        """Create an ONNX Runtime inference session.

        Args:
            model_path: Path to an ONNX model file.
        """

        super(OnnxSerializedModel, self).__init__(model_path)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = os.cpu_count()
        self._ort_session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

    def export_model(
        model: LightningModule,
        example_input: torch.Tensor,
        save_dir: Path,
        filename: str,
        **kwargs,
    ) -> Path:
        """Export a Lightning model to ONNX.

        Args:
            model: Trained Lightning model to export.
            example_input: Example input batch used for ONNX tracing.
            save_dir: Directory where the ``.onnx`` file is written.
            filename: Filename without extension.
            **kwargs: Unused backend-specific options.

        Returns:
            Path to the exported ONNX file.
        """

        if not save_dir.exists():
            save_dir.mkdir()
        save_path = save_dir / f"{filename}.onnx"
        model.eval()
        example_input = example_input.to(model.device)
        torch.onnx.export(
            model,
            example_input,
            save_path,
            export_params=True,
            opset_version=14,
            dynamo=False,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            external_data=False,
        )
        return save_path

    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """Run inference with ONNX Runtime.

        Args:
            input_data: Input image batch on CPU.

        Returns:
            Model logits converted to a PyTorch tensor.
        """

        ort_inputs = {self._ort_session.get_inputs()[0].name: input_data.numpy()}
        ort_outputs = self._ort_session.run(None, ort_inputs)
        return torch.Tensor(ort_outputs)[0]
