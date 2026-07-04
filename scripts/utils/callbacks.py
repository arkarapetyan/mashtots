from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import lightning as L
import seaborn as sns
from torchmetrics.classification import MulticlassConfusionMatrix
from lightning.pytorch.loggers import MLFlowLogger


class ConfusionMatrixCallback(L.Callback):
    """Logs a validation confusion matrix heatmap every N epochs.

    Accumulates predictions/targets across validation batches with a
    `torchmetrics.MulticlassConfusionMatrix` (safe under multi-GPU/DDP),
    then renders a heatmap sized to stay legible for large class counts
    and logs it to MLflow.

    Args:
        num_classes: Total number of classes.
        every_n_epochs: Log the confusion matrix every N validation
            epochs. Defaults to 1 (every epoch).
        class_names: Optional axis tick labels. Defaults to class indices.
        normalize: Normalization for `MulticlassConfusionMatrix`
            ("true", "pred", "all", or None/"none" for raw counts).
            Defaults to "true", recommended when classes are imbalanced.
        preds_key: Key in the `validation_step` output dict holding
            predicted class indices or raw logits.
        target_key: Key in the `validation_step` output dict holding
            ground-truth class indices.

    Example:
        In the LightningModule::

            def validation_step(self, batch, batch_idx):
                x, y = batch
                logits = self(x)
                self.log("val_loss", F.cross_entropy(logits, y))
                return {"preds": logits, "target": y}

        Attached to the Trainer::

            trainer = pl.Trainer(callbacks=[
                ConfusionMatrixCallback(num_classes=78, every_n_epochs=5)
            ])
    """

    _VALID_NORMALIZE = {None, "none", "true", "pred", "all"}

    def __init__(
        self,
        num_classes: int,
        every_n_epochs: int = 1,
        class_names: Optional[List[str]] = None,
        normalize: Optional[str] = "true",
        preds_key: str = "preds",
        target_key: str = "target",
    ):
        super().__init__()
        if class_names is not None and len(class_names) != num_classes:
            raise ValueError("len(class_names) must equal num_classes.")
        if normalize not in self._VALID_NORMALIZE:
            raise ValueError(
                f"normalize must be one of "
                f"{sorted(str(v) for v in self._VALID_NORMALIZE)}, got {normalize!r}."
            )

        self.num_classes = num_classes
        self.every_n_epochs = every_n_epochs
        self.class_names = class_names or [str(i) for i in range(num_classes)]
        self.normalize = normalize
        self.preds_key = preds_key
        self.target_key = target_key
        self.confmat = MulticlassConfusionMatrix(
            num_classes=num_classes, normalize=normalize
        )

    def _should_log(self, trainer: L.Trainer) -> bool:
        """Checks whether the current epoch is a logging epoch."""
        return (trainer.current_epoch + 1) % self.every_n_epochs == 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        """Accumulates batch predictions/targets when this is a logging epoch."""
        if not self._should_log(trainer):
            return
        if (
            not outputs
            or self.preds_key not in outputs
            or self.target_key not in outputs
        ):
            raise KeyError(
                f"ConfusionMatrixCallback expected validation_step's return dict to "
                f"contain '{self.preds_key}' and '{self.target_key}', got: {outputs!r}. "
                "Silently skipping this would leave the confusion matrix all zeros."
            )
        preds = outputs[self.preds_key]
        target = outputs[self.target_key]
        if preds.ndim > 1:  # raw logits/probabilities -> class indices
            preds = preds.argmax(dim=-1)
        self.confmat.update(preds.detach().cpu(), target.detach().cpu())

    def on_validation_epoch_end(self, trainer, pl_module):
        """Renders and logs the confusion matrix, then resets accumulated state."""
        if not self._should_log(trainer):
            return

        cm = self.confmat.compute().numpy()
        fig = self._plot(cm, trainer.current_epoch)
        self._log_figure(trainer, fig)
        plt.close(fig)
        self.confmat.reset()

    def _plot(self, cm: np.ndarray, epoch: int) -> plt.Figure:
        """Builds a heatmap sized and styled to stay legible with many classes."""
        n = len(self.class_names)
        side = max(14, n * 0.22)  # scale canvas with class count
        fig, ax = plt.subplots(figsize=(side, side))

        is_normalized = self.normalize not in (None, "none")
        # Pin the color scale instead of letting seaborn auto-range it to
        # [cm.min(), cm.max()] on every call. Without this, a diagonal-heavy
        # matrix maps almost every off-diagonal cell to the same pale color,
        # and each epoch's plot is rescaled independently, so two very
        # different matrices can render as near-identical images.
        vmin, vmax = (0, 1) if is_normalized else (0, max(cm.max(), 1))

        sns.heatmap(
            cm,
            ax=ax,
            cmap="Blues",
            square=True,
            cbar=True,
            vmin=vmin,
            vmax=vmax,
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            annot=n <= 100,  # per-cell numbers get unreadable past ~30 classes
            fmt=".2f" if is_normalized else ".0f",
            linewidths=0.1,
            linecolor="lightgray",
            annot_kws={"size": 6},
        )
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Validation Confusion Matrix (Epoch {epoch})")

        tick_fontsize = max(4, min(8, 400 / n))
        plt.setp(ax.get_xticklabels(), rotation=90, fontsize=tick_fontsize)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=tick_fontsize)
        fig.tight_layout()
        return fig

    def _log_figure(self, trainer: L.Trainer, fig: plt.Figure) -> None:
        """Logs the confusion matrix to MLflow."""

        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                logger.experiment.log_figure(
                    run_id=logger.run_id,
                    figure=fig,
                    artifact_file=f"confusion_matrix/epoch_{trainer.current_epoch}.png",
                )
                return

        raise RuntimeError("No MLFlowLogger attached to the trainer.")
