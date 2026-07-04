"""Model registry and Lightning image classification architectures."""

from __future__ import annotations
from typing import Callable, Type
import torch
import torch.nn as nn
import lightning as L


from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

from .utils.modules import MBConvLayer, ResidualBlock, ArcMarginProduct


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

    Args:
        in_channels: Number of image input channels.
        num_classes: Number of target classes.
        **kwargs: Extra hyperparameters used by subclasses and optimizer
            configuration.
    """

    def __init__(self, in_channels: int, num_classes: int, **kwargs):
        """Initialize common loss and hyperparameter storage.

        Args:
            in_channels: Number of image input channels.
            num_classes: Number of target classes.
            **kwargs: Extra hyperparameters used by subclasses and optimizer
                configuration.
        """

        super(ImageClassificationModel, self).__init__()

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self._kwargs = kwargs

        self.train_acc = MulticlassAccuracy(num_classes=num_classes).to(self.device)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes).to(self.device)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes).to(self.device)

        self.train_f1 = MulticlassF1Score(num_classes=num_classes).to(self.device)
        self.val_f1 = MulticlassF1Score(num_classes=num_classes).to(self.device)
        self.test_f1 = MulticlassF1Score(num_classes=num_classes).to(self.device)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> float:
        """Run one Lightning training step.

        Args:
            batch: Batch containing input tensors and class labels.
            batch_idx: Index of the current batch.

        Returns:
            Cross-entropy loss for the batch.
        """

        x, y_true = batch
        y_pred = self.forward(x, y_true)
        loss = self.criterion(y_pred, y_true)

        y_pred = y_pred.argmax(dim=1)
        self.train_acc.update(y_pred, y_true)
        self.train_f1.update(y_pred, y_true)

        self.log(
            "train_loss",
            loss.detach().item(),
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "train_acc",
            self.train_acc,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "train_f1",
            self.train_f1,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """Run one Lightning validation step.

        Args:
            batch: Batch containing input tensors and class labels.
            batch_idx: Index of the current batch.
        """

        x, y_true = batch
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y_true)

        y_pred = y_pred.argmax(dim=1)
        self.val_acc.update(y_pred, y_true)
        self.val_f1.update(y_pred, y_true)

        self.log(
            "val_loss",
            loss.detach().item(),
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "val_acc",
            self.val_acc,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )
        self.log(
            "val_f1",
            self.val_f1,
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=False,
        )

        return {"preds": y_pred, "target": y_true}

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> dict:
        """Run one Lightning test step.

        Args:
            batch: Batch containing input tensors and class labels.
            batch_idx: Index of the current batch.
        """

        x, y_true = batch
        y_pred = self.forward(x)

        y_pred = y_pred.argmax(dim=1)
        self.test_acc.update(y_pred, y_true)
        self.test_f1.update(y_pred, y_true)

        self.log(
            "test_acc",
            self.test_acc,
            on_step=False,
            on_epoch=False,
            sync_dist=False,
        )
        self.log(
            "test_f1",
            self.test_f1,
            on_step=False,
            on_epoch=False,
            sync_dist=False,
        )

        return {"preds": y_pred, "target": y_true}

    def configure_optimizers(self):
        """Configure the optimizer and learning-rate scheduler.

        Returns:
            Lightning optimizer configuration using SGD and OneCycleLR.
        """

        max_epochs = self._kwargs.get("max_epochs", 100)
        steps_per_epoch = self.trainer.estimated_stepping_batches // max_epochs
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
            steps_per_epoch=steps_per_epoch,
            epochs=max_epochs,
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
            ResidualBlock(1, 32),
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
