"""Loss functions for Mashtots classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def case_twin_indices(num_classes: int, case_pair_offset: int | None) -> torch.Tensor | None:
    """Return, for every class, the index of the same letter in the other case.

    Mashtots labels uppercase letters ``0 .. offset - 1`` and the matching
    lowercase letters ``offset .. 2 * offset - 1`` in the same order, so the
    twin of class ``c`` is ``(c + offset) % (2 * offset)``.

    Args:
        num_classes: Number of classes.
        case_pair_offset: Distance between a letter and its other-case twin,
            or ``None`` when the label set has no case pairs.

    Returns:
        Long tensor of shape ``(num_classes,)``, or ``None``.

    Raises:
        ValueError: If ``num_classes != 2 * case_pair_offset``.
    """

    if case_pair_offset is None:
        return None
    if num_classes != 2 * case_pair_offset:
        raise ValueError(
            f"case_pair_offset={case_pair_offset} requires num_classes="
            f"{2 * case_pair_offset}, got {num_classes}"
        )
    return (torch.arange(num_classes) + case_pair_offset) % num_classes


class ConfusionAwareLoss(nn.Module):
    """Cross-entropy with extra weight on the letters the model confuses most.

    For a sample with label ``y`` and logits ``z`` the loss is::

        w[y] * (CE(z, y) + pair_weight * sum_j P[y, j] * softplus(z_j - z_y))

    averaged over the batch with the weights ``w`` in the denominator.

    * ``softplus(z_j - z_y) = -log(p_y / (p_y + p_j))`` is a binary
      cross-entropy that asks the model to separate ``y`` from a specific rival
      class ``j``. It is close to zero once ``z_y`` is well above ``z_j``, so
      it only acts on pairs the model actually mixes up.
    * ``P`` (rows sum to one) lists each class's rivals. It starts with the
      other-case twin of every letter (``case_pair_offset``) and, when
      ``adaptive`` is on, adds the ``top_k`` classes each letter is most often
      mistaken for.
    * ``w`` (mean one) gives classes with a higher error rate a larger share of
      the loss when ``adaptive`` is on and ``class_weight_strength > 0``:
      ``1 + class_weight_strength * error / mean_error``, capped at
      ``max_class_weight`` and then rescaled to mean one.

    Adaptive statistics come from the *training* confusion matrix passed to
    :meth:`update_from_confusion` once per epoch, so the validation set stays
    untouched for model selection. They are kept as non-persistent buffers:
    they are not saved in checkpoints and are rebuilt after one epoch when
    training resumes.

    Args:
        num_classes: Number of classes.
        label_smoothing: Label smoothing for the cross-entropy term.
        pair_weight: Weight of the pairwise term. ``0`` disables it.
        case_pair_offset: Index distance between a letter's two cases, or
            ``None`` to skip the case-twin prior.
        case_prior: Weight of the case twin in each row of ``P`` relative to
            the observed confusions, which together weigh ``1``.
        adaptive: Whether :meth:`update_from_confusion` changes ``w`` and ``P``.
        class_weight_strength: How strongly error rates move ``w``. ``0`` keeps
            uniform class weights.
        max_class_weight: Cap on a class's weight relative to an error-free
            class, so a few hard classes cannot dominate the loss.
        top_k: Number of observed rivals kept per class in ``P``.
        momentum: Exponential smoothing of the confusion statistics across
            epochs (``0`` uses only the latest epoch).
        adapt_start_epoch: First epoch (zero-based) whose confusion matrix is
            used; earlier epochs mostly reflect an untrained model.
    """

    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.1,
        pair_weight: float = 0.0,
        case_pair_offset: int | None = None,
        case_prior: float = 1.0,
        adaptive: bool = False,
        class_weight_strength: float = 1.0,
        max_class_weight: float = 3.0,
        top_k: int = 3,
        momentum: float = 0.7,
        adapt_start_epoch: int = 5,
    ):
        """Initialize the loss. See the class docstring for the arguments."""

        super().__init__()
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
        self.pair_weight = pair_weight
        self.case_prior = case_prior
        self.adaptive = adaptive
        self.class_weight_strength = class_weight_strength
        self.max_class_weight = max_class_weight
        self.top_k = top_k
        self.momentum = momentum
        self.adapt_start_epoch = adapt_start_epoch

        twins = case_twin_indices(num_classes, case_pair_offset)
        case_matrix = torch.zeros(num_classes, num_classes)
        if twins is not None:
            case_matrix[torch.arange(num_classes), twins] = 1.0

        self.register_buffer("case_matrix", case_matrix, persistent=False)
        self.register_buffer("pair_matrix", case_matrix.clone(), persistent=False)
        self.register_buffer("class_weights", torch.ones(num_classes), persistent=False)
        self.register_buffer(
            "confusion_ema", torch.zeros(num_classes, num_classes), persistent=False
        )
        self._has_stats = False

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits: Class logits with shape ``(N, num_classes)``.
            target: Class indices with shape ``(N,)``.

        Returns:
            Scalar loss.
        """

        logits = logits.float()
        loss = F.cross_entropy(
            logits,
            target,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )

        if self.pair_weight > 0:
            true_logit = logits.gather(1, target.unsqueeze(1))
            rivals = self.pair_matrix[target]  # (N, C), zero on the true class
            pairwise = (rivals * F.softplus(logits - true_logit)).sum(dim=1)
            loss = loss + self.pair_weight * pairwise

        weights = self.class_weights[target]
        return (weights * loss).sum() / weights.sum()

    @torch.no_grad()
    def update_from_confusion(self, confusion: torch.Tensor, epoch: int) -> None:
        """Update class weights and rival classes from a confusion matrix.

        Args:
            confusion: Raw counts with shape ``(num_classes, num_classes)``;
                rows are true labels and columns predictions.
            epoch: Zero-based index of the epoch the counts come from.
        """

        if not self.adaptive or epoch < self.adapt_start_epoch:
            return

        confusion = confusion.to(self.confusion_ema.device, torch.float32)
        rates = confusion / confusion.sum(dim=1, keepdim=True).clamp_min(1.0)
        if self._has_stats:
            self.confusion_ema.mul_(self.momentum).add_(rates, alpha=1 - self.momentum)
        else:
            self.confusion_ema.copy_(rates)
            self._has_stats = True

        error = 1.0 - self.confusion_ema.diagonal()
        if self.class_weight_strength > 0:
            weights = 1.0 + self.class_weight_strength * error / error.mean().clamp_min(1e-8)
            weights = weights.clamp(max=self.max_class_weight)
            self.class_weights.copy_(weights / weights.mean())

        if self.pair_weight > 0:
            observed = self.confusion_ema.clone()
            observed.fill_diagonal_(0.0)
            values, indices = observed.topk(min(self.top_k, self.num_classes - 1), dim=1)
            observed = torch.zeros_like(observed).scatter_(1, indices, values)
            observed = observed / observed.sum(dim=1, keepdim=True).clamp_min(1e-12)

            pairs = self.case_prior * self.case_matrix + observed
            self.pair_matrix.copy_(pairs / pairs.sum(dim=1, keepdim=True).clamp_min(1e-12))
