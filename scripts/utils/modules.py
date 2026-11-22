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


class DropPath(nn.Module):
    """Stochastic depth: randomly drop a residual branch for whole samples.

    During training each sample's branch output is zeroed with probability
    ``drop_prob`` and the survivors are rescaled by ``1 / (1 - drop_prob)``.
    At inference it is the identity.

    Args:
        drop_prob: Probability of dropping the branch for a sample.
    """

    def __init__(self, drop_prob: float = 0.0):
        """Initialize the drop-path layer.

        Args:
            drop_prob: Probability of dropping the branch for a sample.
        """

        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-sample stochastic depth.

        Args:
            x: Residual branch output with the batch on dimension 0.

        Returns:
            Tensor with the same shape as ``x``.
        """

        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob)
        return x * mask / keep_prob


class PatchEmbedding(nn.Module):
    """Split an image into patches and project each patch to a token.

    Two stems are supported:

    * ``"linear"``: the original ViT patchify layer, a single
      ``patch_size x patch_size`` convolution with the same stride.
    * ``"conv"``: a small stack of overlapping ``3 x 3`` stride-2 convolutions
      followed by a ``1 x 1`` projection (Xiao et al., 2021, "Early
      Convolutions Help Transformers See Better"). It gives the transformer a
      locality prior, which matters when training from scratch on a dataset
      this size. Requires ``patch_size`` to be a power of two.

    Args:
        in_channels: Number of image input channels.
        embed_dim: Token embedding dimension.
        patch_size: Side length of the square patch, i.e. the total stride.
        stem: ``"conv"`` or ``"linear"``.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        stem: str = "conv",
    ):
        """Initialize the patch embedding.

        Args:
            in_channels: Number of image input channels.
            embed_dim: Token embedding dimension.
            patch_size: Side length of the square patch, i.e. the total stride.
            stem: ``"conv"`` or ``"linear"``.

        Raises:
            ValueError: If ``stem`` is unknown, or ``patch_size`` is not a
                power of two when ``stem="conv"``.
        """

        super().__init__()

        if stem == "linear":
            self.proj = nn.Conv2d(
                in_channels, embed_dim, patch_size, stride=patch_size
            )
        elif stem == "conv":
            num_downsamples = int(math.log2(patch_size))
            if 2**num_downsamples != patch_size:
                raise ValueError(
                    f"patch_size must be a power of two for the conv stem, "
                    f"got {patch_size}"
                )
            layers: list[nn.Module] = []
            channels = in_channels
            for i in range(num_downsamples):
                # Channel width doubles each stage and ends at embed_dim / 2.
                out_channels = max(16, embed_dim // 2 ** (num_downsamples - i))
                layers.extend(
                    [
                        nn.Conv2d(
                            channels,
                            out_channels,
                            3,
                            stride=2,
                            padding=1,
                            bias=False,
                        ),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(),
                    ]
                )
                channels = out_channels
            layers.append(nn.Conv2d(channels, embed_dim, 1))
            self.proj = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown patch stem: {stem!r} (use 'conv' or 'linear')")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed an image batch as a token sequence.

        Args:
            x: Input image batch with shape ``(N, C, H, W)``.

        Returns:
            Tokens with shape ``(N, (H / patch_size) * (W / patch_size), D)``.
        """

        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention over a token sequence.

    Uses ``F.scaled_dot_product_attention`` so PyTorch can pick a fused
    kernel on GPU; it also exports through both ``torch.export`` and ONNX.

    Args:
        dim: Token embedding dimension.
        num_heads: Number of attention heads. Must divide ``dim``.
        qkv_bias: Whether the query/key/value projection has a bias.
        attn_dropout: Dropout probability on the attention weights.
        proj_dropout: Dropout probability after the output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
    ):
        """Initialize the attention layer.

        Args:
            dim: Token embedding dimension.
            num_heads: Number of attention heads. Must divide ``dim``.
            qkv_bias: Whether the query/key/value projection has a bias.
            attn_dropout: Dropout probability on the attention weights.
            proj_dropout: Dropout probability after the output projection.

        Raises:
            ValueError: If ``num_heads`` does not divide ``dim``.
        """

        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_dropout = attn_dropout

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.

        Args:
            x: Tokens with shape ``(N, L, D)``.

        Returns:
            Tokens with shape ``(N, L, D)``.
        """

        batch_size, seq_len, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, seq_len, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (N, heads, L, hd)
        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        x = x.transpose(1, 2).reshape(batch_size, seq_len, dim)
        return self.proj_dropout(self.proj(x))


class TransformerEncoderBlock(nn.Module):
    """Pre-norm transformer encoder block (attention + MLP) with drop path.

    Args:
        dim: Token embedding dimension.
        num_heads: Number of attention heads.
        mlp_ratio: Hidden size of the MLP relative to ``dim``.
        qkv_bias: Whether the query/key/value projection has a bias.
        dropout: Dropout probability inside the MLP and after attention.
        attn_dropout: Dropout probability on the attention weights.
        drop_path: Stochastic depth probability for both residual branches.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        attn_dropout: float = 0.0,
        drop_path: float = 0.0,
    ):
        """Initialize the encoder block.

        Args:
            dim: Token embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Hidden size of the MLP relative to ``dim``.
            qkv_bias: Whether the query/key/value projection has a bias.
            dropout: Dropout probability inside the MLP and after attention.
            attn_dropout: Dropout probability on the attention weights.
            drop_path: Stochastic depth probability for both residual
                branches.
        """

        super().__init__()
        hidden_dim = int(dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            attn_dropout=attn_dropout,
            proj_dropout=dropout,
        )
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the encoder block.

        Args:
            x: Tokens with shape ``(N, L, D)``.

        Returns:
            Tokens with shape ``(N, L, D)``.
        """

        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
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

    def _cosine(self, x):
        # The margin math is sensitive to rounding near cos = 1, so it runs in
        # fp32 even when the matmul runs in fp16/bf16 under mixed precision.
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        return cosine.float().clamp(-1.0, 1.0)

    def forward(self, x, labels: torch.Tensor | None = None):
        if labels is not None:
            return self.train_forward(x, labels)
        else:
            return self._cosine(x) * self.s

    def train_forward(self, x, labels):
        # cos(theta)
        cosine = self._cosine(x)
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
