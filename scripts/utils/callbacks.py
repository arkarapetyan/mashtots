"""Lightning callbacks for training diagnostics."""

from typing import List, Optional

import matplotlib

matplotlib.use("Agg")  # render figures without a display (training servers)
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
    and logs it to MLflow together with a text file listing the most
    frequently confused class pairs. Nothing is logged when no MLflow logger
    is attached.

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
        annotate_max_classes: Write the value in each cell only when there
            are at most this many classes; past that the numbers are
            unreadable and slow to draw.
        top_pairs: Number of most-confused (true, predicted) pairs to list.

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
        annotate_max_classes: int = 30,
        top_pairs: int = 20,
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
        self.annotate_max_classes = annotate_max_classes
        self.top_pairs = top_pairs
        # Raw counts; normalized on demand so both views come from one matrix.
        self.confmat = MulticlassConfusionMatrix(num_classes=num_classes)

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

        counts = self.confmat.compute().numpy().astype(np.float64)
        self.confmat.reset()

        logger = self._mlflow_logger(trainer)
        if logger is None or trainer.sanity_checking:
            return

        fig = self._plot(self._normalize(counts), trainer.current_epoch)
        logger.experiment.log_figure(
            run_id=logger.run_id,
            figure=fig,
            artifact_file=f"confusion_matrix/epoch_{trainer.current_epoch}.png",
        )
        plt.close(fig)
        logger.experiment.log_text(
            run_id=logger.run_id,
            text=self._top_pairs_report(counts),
            artifact_file=f"confusion_matrix/top_pairs_epoch_{trainer.current_epoch}.txt",
        )

    def _normalize(self, counts: np.ndarray) -> np.ndarray:
        """Applies the configured normalization to a matrix of raw counts."""
        if self.normalize in (None, "none"):
            return counts
        if self.normalize == "true":
            return counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)
        if self.normalize == "pred":
            return counts / np.clip(counts.sum(axis=0, keepdims=True), 1, None)
        return counts / max(counts.sum(), 1)

    def _top_pairs_report(self, counts: np.ndarray) -> str:
        """Lists the most frequent (true, predicted) mistakes as a table."""
        off_diagonal = counts.copy()
        np.fill_diagonal(off_diagonal, 0)
        row_totals = np.clip(counts.sum(axis=1), 1, None)
        total_errors = max(off_diagonal.sum(), 1)

        order = np.argsort(off_diagonal, axis=None)[::-1][: self.top_pairs]
        lines = [
            f"Total validation errors: {int(off_diagonal.sum())}",
            "true -> predicted | count | % of true class | % of all errors",
        ]
        for flat_idx in order:
            true_idx, pred_idx = np.unravel_index(flat_idx, counts.shape)
            n = off_diagonal[true_idx, pred_idx]
            if n == 0:
                break
            lines.append(
                f"{self.class_names[true_idx]} ({true_idx}) -> "
                f"{self.class_names[pred_idx]} ({pred_idx}) | {int(n)} | "
                f"{100 * n / row_totals[true_idx]:.1f}% | {100 * n / total_errors:.1f}%"
            )
        return "\n".join(lines)

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
            annot=n <= self.annotate_max_classes,
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

    @staticmethod
    def _mlflow_logger(trainer: L.Trainer) -> Optional[MLFlowLogger]:
        """Returns the trainer's MLflow logger, if any."""
        for logger in trainer.loggers:
            if isinstance(logger, MLFlowLogger):
                return logger
        return None
