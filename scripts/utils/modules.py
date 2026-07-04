"""Reusable neural network modules for image classification models."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MBConvLayer(nn.Module):
    """Mobile inverted bottleneck convolution block with squeeze-excitation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Depthwise convolution kernel size.
        stride: Depthwise convolution stride.
        expansion: Channel expansion factor for the hidden representation.
        se_ratio: Ratio used to compute squeeze-excitation hidden channels.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expansion: int,
        se_ratio: float = 0.25,
    ):
        """Initialize the MBConv layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Depthwise convolution kernel size.
            stride: Depthwise convolution stride.
            expansion: Channel expansion factor.
            se_ratio: Ratio used for squeeze-excitation hidden channels.
        """

        super(MBConvLayer, self).__init__()
        hidden_dim = in_channels * expansion
        layers = []

        if expansion != 1:
            layers.extend(
                [
                    nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(),
                ]
            )
        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=hidden_dim,
                    bias=False,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
            ]
        )
        self.conv = nn.Sequential(*layers)

        se_hidden = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, se_hidden, 1),
            nn.SiLU(),
            nn.Conv2d(se_hidden, hidden_dim, 1),
            nn.Sigmoid(),
        )

        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.use_residual = in_channels == out_channels and stride == 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MBConv block.

        Args:
            x: Input feature map with shape ``(N, C, H, W)``.

        Returns:
            Output feature map after convolution, squeeze-excitation, projection,
            and optional residual addition.
        """

        identity = x
        x = self.conv(x)
        x = x * self.se(x)
        x = self.project(x)

        if self.use_residual:
            x = x + identity
        return x


class ResidualBlock(nn.Module):
    """Two-layer residual convolution block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride used by the first convolution and projection shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
    ):
        """Initialize the residual block.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            stride: Stride used by the first convolution and projection
                shortcut.
        """

        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, 3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.activation = nn.ReLU()

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block.

        Args:
            x: Input feature map with shape ``(N, C, H, W)``.

        Returns:
            Output feature map after residual addition and activation.
        """

        identity = self.skip(x)
        x = self.conv_block(x)
        x = x + identity
        x = self.activation(x)
        return x


class ArcMarginProduct(nn.Module):
    """ArcFace Layer Implementation

    source (modified): https://github.com/wujiyang/Face_Pytorch/blob/master/margin/ArcMarginProduct.py

    Args:

    """

    def __init__(
        self,
        in_features=256,
        out_features=78,
        s=30.0,
        m=0.5,
        easy_margin=False,
    ):
        super(ArcMarginProduct, self).__init__()
        self.in_feature = in_features
        self.out_feature = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        # make the function cos(theta+m) monotonic decreasing while theta in [0°,180°]
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels: torch.Tensor | None = None):
        if labels is not None:
            return self.train_forward(x, labels)
        else:
            return F.linear(F.normalize(x), F.normalize(self.weight)) * self.s

    def train_forward(self, x, labels):
        # cos(theta)
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        # cos(theta + m)
        # MODIFIED to avoid negative value under the root
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=1e-7))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        return output
