"""Model registry and Lightning image classification architectures."""

from __future__ import annotations
import math
from typing import Callable, Type
import torch
import torch.nn as nn
import lightning as L


from torchmetrics import MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
)

from .utils.modules import (
    MBConvLayer,
    ResidualBlock,
    ArcMarginProduct,
    PatchEmbedding,
    TransformerEncoderBlock,
)
from .utils.losses import ConfusionAwareLoss


_REGISTRY = {}


def register_model(
    name: str,
) -> Callable[[Type[ImageClassificationModel]], Type[ImageClassificationModel]]:
    """Register a model class under a configuration name.

    Args:
        name: Name used to select the model from configuration.

    Returns:
        Decorator that registers and returns the model class.
    """

    def decorator(
        cls: Type[ImageClassificationModel],
    ) -> Type[ImageClassificationModel]:
        """Add a model class to the registry.

        Args:
            cls: Model class to register.

        Returns:
            The same model class, unchanged.
        """

        _REGISTRY[name] = cls
        return cls

    return decorator


def get_model_type(name: str) -> Type[ImageClassificationModel]:
    """Return the model class registered for ``name``.

    Args:
        name: Registered model name.

    Returns:
        Model class associated with ``name``.

    Raises:
        ValueError: If no model is registered under ``name``.
    """

    if name not in _REGISTRY:
        raise ValueError(f"Invalid Model Name: {name}")
    return _REGISTRY[name]


def list_available_models() -> frozenset[str]:
    """List registered model names.

    Returns:
        Frozen set of available model names.
    """

    return frozenset(_REGISTRY.keys())


class ImageClassificationModel(L.LightningModule):
    """Base Lightning module for image classification models.

    Handles the loss, metrics, training/validation/test steps and the default
    optimizer. Subclasses only define the architecture and ``forward``.

    Logged metrics, per stage (``train``, ``val``, ``test``):

    * ``<stage>_acc``: plain accuracy (fraction correct), the Kaggle metric.
    * ``<stage>_f1``: macro-averaged F1.
    * ``<stage>_letter_acc``: accuracy when upper/lowercase are not
      distinguished (only with ``loss.case_pair_offset``).
    * ``<stage>_case_swap``: fraction of samples predicted as the same letter
      in the other case (only with ``loss.case_pair_offset``).

    Args:
        in_channels: Number of image input channels.
        num_classes: Number of target classes.
        **kwargs: Extra hyperparameters used by subclasses and optimizer
            configuration. ``loss`` holds the keyword arguments of
            :class:`ConfusionAwareLoss`.
    """

    def __init__(self, in_channels: int, num_classes: int, **kwargs):
        """Initialize common loss, metrics and hyperparameter storage.

        Args:
            in_channels: Number of image input channels.
            num_classes: Number of target classes.
            **kwargs: Extra hyperparameters used by subclasses and optimizer
                configuration.
        """

        super(ImageClassificationModel, self).__init__()
        # Store every constructor argument in the checkpoint so
        # ``load_from_checkpoint`` rebuilds the same architecture on its own.
        self.save_hyperparameters(
            {"in_channels": in_channels, "num_classes": num_classes, **kwargs}
        )
        self._kwargs = kwargs
        self.num_classes = num_classes

        loss_config = dict(kwargs.get("loss") or {})
        self.criterion = ConfusionAwareLoss(num_classes, **loss_config)
        self.case_pair_offset = loss_config.get("case_pair_offset")

        self.train_metrics = self._make_metrics()
        self.val_metrics = self._make_metrics()
        self.test_metrics = self._make_metrics()

        # Training-set confusion counts that drive the adaptive loss.
        self.train_confmat = (
            MulticlassConfusionMatrix(num_classes=num_classes)
            if self.criterion.adaptive
            else None
        )

    def _make_metrics(self) -> nn.ModuleDict:
        """Create the metric collection for one stage.

        Returns:
            Module dict mapping metric names to torchmetrics objects.
        """

        metrics = {
            "acc": MulticlassAccuracy(num_classes=self.num_classes, average="micro"),
            "f1": MulticlassF1Score(num_classes=self.num_classes, average="macro"),
        }
        if self.case_pair_offset is not None:
            metrics["letter_acc"] = MulticlassAccuracy(
                num_classes=self.case_pair_offset, average="micro"
            )
            metrics["case_swap"] = MeanMetric()
        return nn.ModuleDict(metrics)

    def _update_and_log_metrics(
        self,
        stage: str,
        metrics: nn.ModuleDict,
        preds: torch.Tensor,
        target: torch.Tensor,
    ) -> None:
        """Update a stage's metrics and register them for epoch-level logging.

        Args:
            stage: Metric name prefix, e.g. ``"val"``.
            metrics: Metrics created by :meth:`_make_metrics`.
            preds: Predicted class indices.
            target: True class indices.
        """

        metrics["acc"].update(preds, target)
        metrics["f1"].update(preds, target)
        if self.case_pair_offset is not None:
            offset = self.case_pair_offset
            metrics["letter_acc"].update(preds % offset, target % offset)
            twins = (target + offset) % self.num_classes
            metrics["case_swap"].update((preds == twins).float())

        for name, metric in metrics.items():
            self.log(
                f"{stage}_{name}",
                metric,
                on_step=False,
                on_epoch=True,
                logger=True,
            )

    def _evaluation_step(
        self,
        stage: str,
        metrics: nn.ModuleDict,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> dict:
        """Shared validation/test step.

        Args:
            stage: Metric name prefix.
            metrics: Metrics for the stage.
            batch: Batch containing input tensors and class labels.

        Returns:
            Predictions and targets, consumed by the confusion matrix callback.
        """

        x, y_true = batch
        logits = self(x)
        loss = self.criterion(logits, y_true)
        y_pred = logits.argmax(dim=1)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, logger=True)
        self._update_and_log_metrics(stage, metrics, y_pred, y_true)
        return {"preds": y_pred, "target": y_true}

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        """Run one Lightning training step.

        Args:
            batch: Batch containing input tensors and class labels.
            batch_idx: Index of the current batch.

        Returns:
            Loss for the batch.
        """

        x, y_true = batch
        logits = self(x, y_true)
        loss = self.criterion(logits, y_true)
        y_pred = logits.argmax(dim=1)

        self.log("train_loss", loss, on_step=False, on_epoch=True, logger=True)
        self._update_and_log_metrics("train", self.train_metrics, y_pred, y_true)
        if self.train_confmat is not None:
            self.train_confmat.update(y_pred, y_true)
        return loss

    def on_train_epoch_end(self) -> None:
        """Feed this epoch's training confusion matrix to the adaptive loss."""

        if self.train_confmat is None:
            return
        counts = self.train_confmat.compute()
        self.train_confmat.reset()
        self.criterion.update_from_confusion(counts, self.current_epoch)
        self.log(
            "loss_max_class_weight",
            self.criterion.class_weights.max(),
            on_step=False,
            on_epoch=True,
        )

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """Run one Lightning validation step.

        Args:
            batch: Batch containing input tensors and class labels.
            batch_idx: Index of the current batch.

        Returns:
            Predictions and targets for the confusion matrix callback.
        """

        return self._evaluation_step("val", self.val_metrics, batch)

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """Run one Lightning test step.

        Args:
            batch: Batch containing input tensors and class labels.
            batch_idx: Index of the current batch.

        Returns:
            Predictions and targets.
        """

        return self._evaluation_step("test", self.test_metrics, batch)

    def configure_optimizers(self):
        """Configure the optimizer and learning-rate scheduler.

        Returns:
            Lightning optimizer configuration using SGD and OneCycleLR.
        """

        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self._kwargs.get("optimizer_lr", 1e-2),
            momentum=0.9,
            weight_decay=self._kwargs.get("optimizer_weight_decay", 5e-4),
            nesterov=True,
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self._kwargs.get("optimizer_lr", 1e-2),
            total_steps=int(self.trainer.estimated_stepping_batches),
            pct_start=0.1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


@register_model("lightnet")
class LightNet(ImageClassificationModel):
    """Small residual convolutional classifier for grayscale character images."""

    def __init__(self, in_channels: int, num_classes: int, **kwargs):
        """Initialize the LightNet architecture.

        Args:
            in_channels: Number of image input channels.
            num_classes: Number of output classes.
            **kwargs: Extra hyperparameters, including ``dropout_rate``.
        """

        super(LightNet, self).__init__(in_channels, num_classes, **kwargs)

        self.features = nn.Sequential(
            # Block 1 (64 -> 32)
            ResidualBlock(in_channels, 32),
            nn.MaxPool2d(2),
            # Block 2 (32 -> 16)
            ResidualBlock(32, 64),
            nn.MaxPool2d(2),
            # Block 3 (16 -> 8)
            ResidualBlock(64, 128),
            nn.MaxPool2d(2),
            # Block 4 (8 -> 4)
            ResidualBlock(128, 384),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(384, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(self._kwargs.get("dropout_rate", 0.2)),
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute class logits.

        Args:
            x: Input image batch with shape ``(N, C, H, W)``.
            labels: Optional Labels of the Input During Training with shape ``(N,)``.

        Returns:
            Class logits with shape ``(N, num_classes)``.
        """

        x = self.features(x)
        x = self.pool(x)
        x = self.embedding(x)
        logits = self.classifier(x)
        return logits


@register_model("efficientnet")
class EfficientNet(ImageClassificationModel):
    """EfficientNet-style classifier built from MBConv blocks."""

    def __init__(self, in_channels: int, num_classes: int, **kwargs):
        """Initialize the EfficientNet-style architecture.

        Args:
            in_channels: Number of image input channels.
            num_classes: Number of output classes.
            **kwargs: Extra hyperparameters, including ``dropout_rate``.
        """

        super(EfficientNet, self).__init__(in_channels, num_classes, **kwargs)
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels,
                32,
                3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )

        self.mb_block = nn.Sequential(
            self._make_blocks(),
        )

        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.SiLU(),
        )

        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(self._kwargs.get("dropout_rate", 0.2)),
            nn.Linear(1280, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute class logits.

        Args:
            x: Input image batch with shape ``(N, C, H, W)``.
            labels: Optional  Labels of the Input During Training with shape ``(N,)``.

        Returns:
            Class logits with shape ``(N, num_classes)``.
        """

        x = self.stem(x)
        x = self.mb_block(x)
        x = self.head(x)
        logits = self.fc(x)
        return logits

    def _make_blocks(self) -> list[MBConvLayer]:
        """Build the MBConv stack used by the EfficientNet backbone.

        Returns:
            Sequential module containing configured MBConv layers.
        """

        blocks = []
        parameters = [
            [1, 32, 16, 3, 1, 1],
            [2, 16, 24, 3, 1, 6],
            [2, 24, 40, 5, 2, 6],
            [3, 40, 80, 3, 2, 6],
            [3, 80, 112, 5, 1, 6],
            [4, 112, 192, 5, 1, 6],
            [1, 192, 320, 3, 1, 6],
        ]

        for param_list in parameters:
            cnt = param_list[0]
            params = param_list[1:]
            blocks.append(
                MBConvLayer(*params),
            )
            params[0] = params[1]
            params[3] = 1

            for _ in range(cnt - 1):
                blocks.append(
                    MBConvLayer(*params),
                )

        return nn.Sequential(
            *blocks,
        )


class ResNet18Base(ImageClassificationModel):
    """ResNet-18-style classifier base without the final layer."""

    def __init__(self, in_channels: int, num_classes: int, **kwargs):
        """Initialize the ResNet-style architecture wtihout the final layer.

        Args:
            in_channels: Number of image input channels.
            num_classes: Number of output classes.
            **kwargs: Extra hyperparameters, including ``dropout_rate``.
        """

        super(ResNet18Base, self).__init__(
            in_channels,
            num_classes,
            **kwargs,
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                64,
                3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.in_channels = 64

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self._kwargs.get("dropout_rate", 0.4)),
        )

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Create one residual stage.

        Args:
            out_channels: Number of channels produced by the stage.
            num_blocks: Number of residual blocks in the stage.
            stride: Stride for the first residual block.

        Returns:
            Sequential residual stage.
        """

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(
                ResidualBlock(
                    self.in_channels,
                    out_channels,
                    stride,
                )
            )
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute Embedded Vector.

        Args:
            x: Input image batch with shape ``(N, C, H, W)``.
            labels: Optional Labels of the Input During Training with shape ``(N,)``.

        Returns:
            Vector with shape ``(N, 256)``.
        """

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.fc(x)
        return x


@register_model("resnet")
class ResNet18(ResNet18Base):
    """ResNet-18 model adapted for small character images"""

    def __init__(self, in_channels: int, num_classes: int, **kwargs):
        """Initialize the ResNet-style architecture wtih a final linear layer.

        Args:
            in_channels: Number of image input channels.
            num_classes: Number of output classes.
            **kwargs: Extra hyperparameters, including ``dropout_rate``.
        """
        super(ResNet18, self).__init__(
            in_channels,
            num_classes,
            **kwargs,
        )
        self.final = nn.Linear(256, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Class Logits.

        Args:
            x: Input image batch with shape ``(N, C, H, W)``.
            labels: Optional Labels of the Input During Training with shape ``(N,)``.

        Returns:
            Logits with shape ``(N, num_classes)``.
        """

        x = super(ResNet18, self).forward(x)
        x = self.final(x)
        return x


@register_model("resnet+arcface")
class ResNetArcFace(ResNet18Base):
    """ResNet-18 model adapted for arcface loss"""

    def __init__(self, in_channels: int, num_classes: int, **kwargs):
        """Initialize the ResNet-style architecture wtih a final linear layer.

        Args:
            in_channels: Number of image input channels.
            num_classes: Number of output classes.
            **kwargs: Extra hyperparameters, including ``dropout_rate``.
        """
        super(ResNetArcFace, self).__init__(
            in_channels,
            num_classes,
            **kwargs,
        )
        self.final = ArcMarginProduct(
            in_features=256,
            out_features=num_classes,
            s=kwargs.get("arcface_s", 18.0),
            m=kwargs.get("arcface_m", 0.25),
        )

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute Class Logits.

        Args:
            x: Input image batch with shape ``(N, C, H, W)``.
            labels: Optional Labels of the Input During Training with shape ``(N,)``.

        Returns:
            Logits with shape ``(N, num_classes)``.
        """

        x = super(ResNetArcFace, self).forward(x)
        x = self.final(x, labels)
        return x


@register_model("vit")
class VisionTransformer(ImageClassificationModel):
    """Vision Transformer (ViT) adapted for small grayscale character images.

    Follows Dosovitskiy et al. (2021) with changes that matter when training
    from scratch on ~70k images of size ``64 x 64``:

    * Convolutional patch stem by default (see ``PatchEmbedding``).
    * Global average pooling over tokens by default instead of a class token.
    * Stochastic depth that increases linearly with block depth.
    * AdamW with linear warmup and cosine decay; no weight decay on biases,
      norm parameters, position embeddings, or the class token.
    * Gradient-norm clipping.

    The shared ``dropout_rate``, ``optimizer_lr`` and ``optimizer_weight_decay``
    hyperparameters come from the model config as for the other models. The
    transformer-specific ones are read from ``**kwargs`` (set them under
    ``model.model_kwargs`` in the training config):

    ============================  ========  =====================================
    Key                           Default   Meaning
    ============================  ========  =====================================
    ``img_size``                  64        Input height and width.
    ``patch_size``                4         Patch side / total stem stride.
    ``patch_stem``                "conv"    ``"conv"`` or ``"linear"``.
    ``embed_dim``                 192       Token dimension.
    ``depth``                     8         Number of encoder blocks.
    ``num_heads``                 6         Attention heads per block.
    ``mlp_ratio``                 4.0       MLP hidden size / ``embed_dim``.
    ``pooling``                   "mean"    ``"mean"`` or ``"cls"``.
    ``drop_path_rate``            0.1       Stochastic depth of the last block.
    ``attn_dropout_rate``         0.0       Dropout on attention weights.
    ``mlp_dropout_rate``          0.0       Dropout inside blocks.
    ``warmup_epochs``             5         Linear LR warmup length.
    ``min_lr``                    1e-6      LR at the end of the cosine decay.
    ``grad_clip_norm``            1.0       Max gradient norm (0 disables).
    ============================  ========  =====================================
    """

    def __init__(self, in_channels: int, num_classes: int, **kwargs):
        """Initialize the Vision Transformer.

        Args:
            in_channels: Number of image input channels.
            num_classes: Number of output classes.
            **kwargs: Shared hyperparameters (``dropout_rate``,
                ``optimizer_lr``, ``optimizer_weight_decay``, ``max_epochs``)
                and the transformer hyperparameters listed in the class
                docstring.

        Raises:
            ValueError: If ``img_size`` is not divisible by ``patch_size`` or
                ``pooling`` is unknown.
        """

        super().__init__(in_channels, num_classes, **kwargs)

        img_size = kwargs.get("img_size", 64)
        patch_size = kwargs.get("patch_size", 4)
        embed_dim = kwargs.get("embed_dim", 192)
        depth = kwargs.get("depth", 8)
        num_heads = kwargs.get("num_heads", 6)
        mlp_ratio = kwargs.get("mlp_ratio", 4.0)
        drop_path_rate = kwargs.get("drop_path_rate", 0.1)
        self.pooling = kwargs.get("pooling", "mean")

        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size ({img_size}) must be divisible by patch_size "
                f"({patch_size})"
            )
        if self.pooling not in ("mean", "cls"):
            raise ValueError(
                f"Unknown pooling: {self.pooling!r} (use 'mean' or 'cls')"
            )

        self.patch_embed = PatchEmbedding(
            in_channels,
            embed_dim,
            patch_size,
            stem=kwargs.get("patch_stem", "conv"),
        )
        num_tokens = (img_size // patch_size) ** 2

        if self.pooling == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            num_tokens += 1
        else:
            self.register_parameter("cls_token", None)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))

        # Stochastic depth grows linearly from 0 at the first block.
        drop_path_rates = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    dropout=kwargs.get("mlp_dropout_rate", 0.0),
                    attn_dropout=kwargs.get("attn_dropout_rate", 0.0),
                    drop_path=rate,
                )
                for rate in drop_path_rates
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(kwargs.get("dropout_rate", 0.1)),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following common ViT practice."""

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute class logits.

        Args:
            x: Input image batch with shape ``(N, C, H, W)``.
            labels: Optional Labels of the Input During Training with shape ``(N,)``.
                Unused; accepted for interface compatibility.

        Returns:
            Class logits with shape ``(N, num_classes)``.
        """

        x = self.patch_embed(x)
        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.blocks(x)
        x = self.norm(x)
        x = x[:, 0] if self.pooling == "cls" else x.mean(dim=1)
        return self.head(x)

    def configure_optimizers(self):
        """Configure AdamW with linear warmup followed by cosine decay.

        Returns:
            Lightning optimizer configuration with a per-step scheduler.
        """

        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim <= 1 or name in ("pos_embed", "cls_token"):
                no_decay.append(param)
            else:
                decay.append(param)

        base_lr = self._kwargs.get("optimizer_lr", 1e-3)
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": decay,
                    "weight_decay": self._kwargs.get("optimizer_weight_decay", 0.05),
                },
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=base_lr,
            betas=(0.9, 0.999),
        )

        total_steps = max(1, int(self.trainer.estimated_stepping_batches))
        max_epochs = max(1, self.trainer.max_epochs or 1)
        warmup_steps = int(
            total_steps * self._kwargs.get("warmup_epochs", 5) / max_epochs
        )
        min_lr_ratio = self._kwargs.get("min_lr", 1e-6) / base_lr

        def lr_lambda(step: int) -> float:
            """Return the LR multiplier for an optimizer step."""

            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = min(
                1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)
            )
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip gradients by norm unless the Trainer already sets clipping.

        Args:
            optimizer: Optimizer whose gradients are clipped.
            gradient_clip_val: Value passed by the Trainer, if configured.
            gradient_clip_algorithm: Algorithm passed by the Trainer, if
                configured.
        """

        clip_val = gradient_clip_val
        if clip_val is None:
            clip_val = self._kwargs.get("grad_clip_norm", 1.0)
        if clip_val:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )


@register_model("baseconvnet")
class BaseConvolutionalNet(ImageClassificationModel):
    """Baseline convolutional classifier for character images."""

    def __init__(self, in_channels: int, num_classes: int, **kwargs):
        """Initialize the baseline convolutional network.

        Args:
            in_channels: Number of image input channels.
            num_classes: Number of output classes.
            **kwargs: Extra hyperparameters.
        """

        super(BaseConvolutionalNet, self).__init__(
            in_channels,
            num_classes,
            **kwargs,
        )

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
        )

        self.layer1 = self._make_block(32, 64, 0.2)
        self.layer2 = self._make_block(64, 128, 0.25)
        self.layer3 = self._make_block(128, 256, 0.3)
        self.pool = nn.MaxPool2d(2)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.Dropout1d(0.4),
            nn.Linear(512, num_classes),
        )

    def _make_block(
        self,
        in_channels: int,
        out_channels: int,
        dropout_rate: int,
    ) -> nn.Sequential:
        """Create a convolutional block for the baseline network.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            dropout_rate: Dropout probability for the block.

        Returns:
            Sequential convolutional block.
        """

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, padding_mode="zeros"),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout_rate),
        )

    def forward(
        self,
        x: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute class logits.

        Args:
            x: Input image batch with shape ``(N, C, H, W)``.
            labels: Optional Labels of the Input During Training with shape ``(N,)``.

        Returns:
            Class logits with shape ``(N, num_classes)``.
        """

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = self.fc(x)
        return x
