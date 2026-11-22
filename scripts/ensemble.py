"""Ensemble inference over one or more serialized models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

from .serialized_model import BaseSerializedModel, get_extension_format


def shift_images(images: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Translate images by whole pixels, filling uncovered pixels with background.

    Each image's minimum value is used as its background, which is the
    normalized value of a black pixel for every Mashtots image.

    Args:
        images: Image batch with shape ``(N, C, H, W)``.
        dx: Horizontal shift in pixels (positive moves right).
        dy: Vertical shift in pixels (positive moves down).

    Returns:
        Shifted image batch with the same shape.
    """

    if dx == 0 and dy == 0:
        return images
    height, width = images.shape[-2:]
    background = images.amin(dim=(-2, -1), keepdim=True)
    shifted = torch.roll(images, shifts=(dy, dx), dims=(-2, -1))

    uncovered = torch.zeros(height, width, dtype=torch.bool, device=images.device)
    if dy > 0:
        uncovered[:dy, :] = True
    elif dy < 0:
        uncovered[dy:, :] = True
    if dx > 0:
        uncovered[:, :dx] = True
    elif dx < 0:
        uncovered[:, dx:] = True
    return torch.where(uncovered, background, shifted)


class EnsemblePredictor:
    """Average class probabilities over several serialized models.

    Each model may be a ``.pt2`` (Torch Export) or ``.onnx`` file. Logits are
    turned into probabilities with a softmax and combined with a weighted
    mean. A single model path gives ordinary single-model inference.

    With ``tta=True`` every model also sees the batch shifted by
    ``tta_shift_px`` pixels up, down, left and right, and its probabilities
    are averaged over those five views. Flips and rotations are not used:
    they change or invalidate letters.

    Args:
        model_paths: Serialized model files.
        weights: Optional non-negative weight per model; normalized to sum
            to one. Defaults to equal weights.
        tta: Whether to use shift test-time augmentation.
        tta_shift_px: Shift distance in pixels for TTA.

    Raises:
        ValueError: If no paths are given or ``weights`` is invalid.
    """

    def __init__(
        self,
        model_paths: Sequence[Path],
        weights: Optional[Sequence[float]] = None,
        tta: bool = False,
        tta_shift_px: int = 2,
    ):
        """Load every model. See the class docstring for the arguments."""

        if not model_paths:
            raise ValueError("At least one model path is required")
        if weights is None:
            weights = [1.0] * len(model_paths)
        if len(weights) != len(model_paths) or min(weights) < 0 or sum(weights) <= 0:
            raise ValueError(
                "weights must be non-negative, sum to more than zero, and "
                "have one entry per model"
            )

        self.model_paths = [Path(p) for p in model_paths]
        self.models: list[BaseSerializedModel] = [
            get_extension_format(p.suffix)(model_path=p) for p in self.model_paths
        ]
        total = float(sum(weights))
        self.weights = [w / total for w in weights]
        self.shifts = [(0, 0)]
        if tta:
            s = tta_shift_px
            self.shifts += [(s, 0), (-s, 0), (0, s), (0, -s)]

    def model_probabilities(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Return each model's class probabilities, averaged over TTA views.

        Args:
            images: Normalized image batch with shape ``(N, C, H, W)``.

        Returns:
            One ``(N, num_classes)`` CPU tensor per model.
        """

        per_model = []
        for model in self.models:
            views = [
                F.softmax(
                    model.predict(shift_images(images, dx, dy)).float().cpu(), dim=1
                )
                for dx, dy in self.shifts
            ]
            per_model.append(torch.stack(views).mean(dim=0))
        return per_model

    def predict_proba(self, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Return ensemble probabilities and the per-model probabilities.

        Args:
            images: Normalized image batch with shape ``(N, C, H, W)``.

        Returns:
            The weighted-mean probabilities ``(N, num_classes)`` and the list of
            per-model probabilities.
        """

        per_model = self.model_probabilities(images)
        ensemble = sum(w * p for w, p in zip(self.weights, per_model))
        return ensemble, per_model
