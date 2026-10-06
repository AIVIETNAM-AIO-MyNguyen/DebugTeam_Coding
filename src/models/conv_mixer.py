"""
ConvMixer and Res-ConvMixer Architectures.

References:
- ConvMixer: Patches Are All You Need?
  (Trockman & Kolter, ICLR 2022 / TMLR 2023, https://arxiv.org/abs/2201.09792)
- ResMLP: Feedforward networks for image classification with data-efficient training
  (Touvron et al., NeurIPS 2021, https://arxiv.org/abs/2105.03404)

Key Innovations & Combination:
1. ConvMixer: Uses patch embeddings + Depthwise Separable Convolutions (depthwise for
   spatial mixing, pointwise for channel mixing) in an isotropic architecture.
2. Res-ConvMixer (Hybrid): Combines ResMLP's learnable Affine transformations (scale & shift)
   and Channel MLP expansion with ConvMixer's depthwise spatial convolutions. This eliminates
   ResMLP's rigid S x S spatial matrix while providing superior 2D inductive bias and higher
   accuracy on low-resolution datasets like CIFAR-100.
"""

from typing import Optional, Union
import torch
import torch.nn as nn
from src.models.base import BaseClassifier, register_model


class Affine2d(nn.Module):
    """
    Channel-wise learnable affine transformation for 4D feature maps (B, C, H, W).
    y = alpha * x + beta, where alpha, beta are (1, C, 1, 1).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + self.beta


class Residual(nn.Module):
    """Residual wrapper: y = x + fn(x)."""
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(x)


class ConvMixerBlock(nn.Module):
    """
    Standard ConvMixer Block (Trockman & Kolter, 2022).
    - Spatial mixing: Residual Depthwise Conv (kernel_size x kernel_size, groups=dim) + GELU + BatchNorm
    - Channel mixing: Pointwise Conv (1 x 1) + GELU + BatchNorm
    """
    def __init__(self, dim: int, kernel_size: int = 7):
        super().__init__()
        self.spatial = Residual(
            nn.Sequential(
                nn.Conv2d(
                    dim, dim, kernel_size=kernel_size, groups=dim,
                    padding=kernel_size // 2, bias=True
                ),
                nn.GELU(),
                nn.BatchNorm2d(dim),
            )
        )
        self.channel = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.channel(self.spatial(x))


class ResConvMixerBlock(nn.Module):
    """
    Hybrid Res-ConvMixer Block:
    Combines ResMLP's Affine layers and Channel MLP with ConvMixer's Depthwise Spatial Conv.
    - Spatial Sub-block:
        y = Affine(x)
        y = DepthwiseConv(y, kernel_size)
        y = GELU(y)
        y = Affine(y)
        x = x + y
    - Channel Sub-block:
        y = Affine(x)
        y = PointwiseConv(dim -> expansion_factor * dim)
        y = GELU(y)
        y = PointwiseConv(expansion_factor * dim -> dim)
        y = Affine(y)
        x = x + y
    """
    def __init__(self, dim: int, kernel_size: int = 7, expansion_factor: int = 2):
        super().__init__()
        # Spatial Mixing (Depthwise Conv + Affine)
        self.affine_spatial_in = Affine2d(dim)
        self.depthwise = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, groups=dim,
            padding=kernel_size // 2, bias=True
        )
        self.act_spatial = nn.GELU()
        self.affine_spatial_out = Affine2d(dim)

        # Channel Mixing (MLP with Expansion + Affine)
        hidden_dim = dim * expansion_factor
        self.affine_channel_in = Affine2d(dim)
        self.channel_fc1 = nn.Conv2d(dim, hidden_dim, kernel_size=1, bias=True)
        self.act_channel = nn.GELU()
        self.channel_fc2 = nn.Conv2d(hidden_dim, dim, kernel_size=1, bias=True)
        self.affine_channel_out = Affine2d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Spatial mixing sub-block
        s = self.affine_spatial_in(x)
        s = self.act_spatial(self.depthwise(s))
        x = x + self.affine_spatial_out(s)

        # 2. Channel mixing sub-block
        c = self.affine_channel_in(x)
        c = self.act_channel(self.channel_fc1(c))
        c = self.channel_fc2(c)
        x = x + self.affine_channel_out(c)
        return x


@register_model("conv_mixer")
class ConvMixer(BaseClassifier):
    """
    ConvMixer Vision Architecture.
    Reference: Trockman & Kolter, ICLR 2022 / TMLR 2023.

    Args:
        in_channels: Number of input image channels (e.g., 3 for RGB).
        image_size: Input spatial resolution (e.g., 32 for CIFAR).
        patch_size: Patch size for embedding (e.g., 2 for CIFAR-100, 7 for ImageNet).
        dim: Embedding dimension (e.g., 256).
        depth: Number of ConvMixer blocks (e.g., 8).
        kernel_size: Depthwise convolution kernel size (e.g., 7 or 9).
        num_classes: Number of output classification categories.
    """
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 2,
        dim: int = 256,
        depth: int = 8,
        kernel_size: int = 7,
        num_classes: int = 100,
        num_features: Optional[int] = None,
        num_layers: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        # Support aliases
        if num_features is not None:
            dim = num_features
        if num_layers is not None:
            depth = num_layers

        self.dim = dim
        self.depth = depth
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.num_classes = num_classes

        # 1. Patch Embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim),
        )

        # 2. Stack of ConvMixer Blocks
        self.blocks = nn.ModuleList([
            ConvMixerBlock(dim=dim, kernel_size=kernel_size)
            for _ in range(depth)
        ])

        # 3. Classifier Head
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding: (B, C, H, W) -> (B, dim, H/p, W/p)
        x = self.patch_embed(x)

        # Repeated mixing
        for block in self.blocks:
            x = block(x)

        # Global pooling & classification
        x = self.pooling(x).flatten(1)
        return self.head(x)


@register_model("res_conv_mixer")
class ResConvMixer(BaseClassifier):
    """
    Res-ConvMixer: Hybrid Architecture combining ResMLP and ConvMixer.

    Innovations:
    - Replaces ResMLP's rigid S x S global linear cross-patch transpose with
      translation-invariant Depthwise Convolutions.
    - Employs ResMLP's Affine transformations (scale & shift) instead of standard BatchNorm
      to eliminate batch-size sensitivity.
    - Uses ResMLP's inverted bottleneck Channel MLP (expansion factor 2x/4x) for enriched
      channel feature mixing.

    Args:
        in_channels: Number of input channels (3).
        image_size: Input spatial resolution (32).
        patch_size: Patch size (2).
        dim: Embedding feature dimension (256).
        depth: Number of layers (8).
        kernel_size: Depthwise spatial kernel size (7).
        expansion_factor: Expansion factor for channel MLP (2).
        num_classes: Number of classification classes (100).
    """
    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 2,
        dim: int = 256,
        depth: int = 8,
        kernel_size: int = 7,
        expansion_factor: int = 2,
        num_classes: int = 100,
        num_features: Optional[int] = None,
        num_layers: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if num_features is not None:
            dim = num_features
        if num_layers is not None:
            depth = num_layers

        self.dim = dim
        self.depth = depth
        self.patch_size = patch_size
        self.kernel_size = kernel_size
        self.expansion_factor = expansion_factor
        self.num_classes = num_classes

        # Patch Embedding
        self.patch_embed = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            Affine2d(dim),
        )

        # Res-ConvMixer Blocks
        self.blocks = nn.ModuleList([
            ResConvMixerBlock(dim=dim, kernel_size=kernel_size, expansion_factor=expansion_factor)
            for _ in range(depth)
        ])

        # Classification Head
        self.norm = Affine2d(dim)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.pooling(x).flatten(1)
        return self.head(x)
