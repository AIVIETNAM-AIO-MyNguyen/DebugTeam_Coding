"""
Pyramid-ResMLP: Hierarchical ResMLP with Overlapping Patching & RepTokenMix.

This architecture solves the core limitations of standard ResMLP by introducing:
1. Overlapping Convolutional Patch Embedding:
   Eliminates boundary artifacts compared to harsh non-overlapping patch cuts.
2. Hierarchical / Pyramid Multi-Stage Architecture:
   Transitions from high-resolution fine features (Stage 1) to low-resolution
   high-level semantics (Stage 3/4) via Convolutional Downsampling layers.
3. Re-parameterizable Spatial Mixing (RepTokenMix):
   Combines global Linear spatial mixing with parallel local Depthwise Convolutions
   (3x3 + 1x1) during training, restoring translation equivariance and local priors.
   At deploy time (locality_injection), conv branches are mathematically fused into
   the linear layer for zero extra inference latency.
"""

from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base import BaseClassifier, register_model
from src.models.res_mlp import AffineTransform, ChannelMixLayer
from src.models.res_mlp_rep_token import RepTokenMixLayer


class OverlappingPatchEmbed(nn.Module):
    """
    Overlapping Convolutional Patch Embedding (Stem).

    Uses a convolutional layer with padding so neighboring patches overlap,
    preventing edge discontinuities across patch boundaries.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 64,
        patch_size: int = 2,
    ):
        super().__init__()
        self.patch_size = patch_size

        if patch_size == 2:
            # 2x downsample with 1-pixel overlap (e.g. 32x32 -> 16x16)
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        elif patch_size == 4:
            # 4x downsample with 3-pixel overlap (e.g. 64x64 -> 16x16 or 224x224 -> 56x56)
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=4, padding=2)
        else:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride=patch_size)

        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        # x: (B, C_in, H, W)
        x = self.norm(self.proj(x))
        H_out, W_out = x.shape[2], x.shape[3]
        # Flatten to tokens: (B, N, C_out)
        tokens = x.flatten(2).transpose(1, 2)
        return tokens, H_out, W_out


class ConvDownsample(nn.Module):
    """
    Hierarchical Transition Layer between Pyramid Stages.

    Reduces spatial resolution by 2x and projects channel dimension from C_in to C_out.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        # x: (B, N, C)
        B, N, C = x.shape
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        out_2d = self.norm(self.conv(x_2d))
        H_new, W_new = out_2d.shape[2], out_2d.shape[3]
        out_tokens = out_2d.flatten(2).transpose(1, 2)
        return out_tokens, H_new, W_new


class PyramidResMLPBlock(nn.Module):
    """
    A single block within a Pyramid stage combining RepTokenMixLayer and ChannelMixLayer.
    """

    def __init__(
        self,
        features: int,
        patches: int,
        grid_h: int,
        grid_w: int,
        expansion_factor: int = 4,
        reparam_conv_k: Sequence[int] = (1, 3),
        deploy: bool = False,
    ):
        super().__init__()
        # We pass image_size=grid_h*2, patch_size=2 so that image_size//patch_size == grid_h
        self.token_mix = RepTokenMixLayer(
            features=features,
            patches=patches,
            image_size=grid_h * 2,
            patch_size=2,
            reparam_conv_k=reparam_conv_k,
            deploy=deploy,
        )
        self.channel_mix = ChannelMixLayer(features, expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_mix(x)
        x = self.channel_mix(x)
        return x


@register_model("pyramid_res_mlp")
@register_model("pyramid_rep_res_mlp")
@register_model("res_mlp_pyramid")
class PyramidResMLP(BaseClassifier):
    """
    Pyramid-ResMLP: Hierarchical Vision Model with Multi-Scale Stages & RepTokenMix.

    Args:
        image_size: Input image resolution (e.g. 32 for CIFAR, 64 for Tiny-ImageNet).
        patch_size: Initial stem downsampling factor (2 for CIFAR, 4 for Tiny-ImageNet).
        in_channels: Input channels (3 for RGB).
        num_classes: Classification categories.
        channels: Channel dimensions for each pyramid stage (default: (64, 128, 256)).
        num_blocks: Number of blocks in each stage (default: (2, 2, 4)).
        expansion_factor: MLP expansion factor in ChannelMix (default: 4).
        reparam_conv_k: Kernel sizes for parallel local DW convs in RepTokenMix (default: (1, 3)).
        deploy: If True, builds directly in fused deploy mode.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 3,
        num_classes: int = 100,
        channels: Sequence[int] = (64, 128, 256),
        num_blocks: Sequence[int] = (2, 2, 4),
        expansion_factor: int = 4,
        reparam_conv_k: Sequence[int] = (1, 3),
        deploy: bool = False,
        **kwargs,
    ):
        super().__init__()
        assert len(channels) == len(num_blocks), "Length of channels and num_blocks must match."

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_stages = len(channels)
        self.deploy = deploy

        # 1. Overlapping Patch Embedding (Stem)
        self.patch_embed = OverlappingPatchEmbed(
            in_channels=in_channels,
            out_channels=channels[0],
            patch_size=patch_size,
        )

        curr_H = image_size // patch_size
        curr_W = image_size // patch_size

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.stage_resolutions = []

        # 2. Build Pyramid Stages
        for i in range(self.num_stages):
            C = channels[i]
            N = curr_H * curr_W
            self.stage_resolutions.append((curr_H, curr_W))

            stage_blocks = nn.ModuleList([
                PyramidResMLPBlock(
                    features=C,
                    patches=N,
                    grid_h=curr_H,
                    grid_w=curr_W,
                    expansion_factor=expansion_factor,
                    reparam_conv_k=reparam_conv_k,
                    deploy=deploy,
                )
                for _ in range(num_blocks[i])
            ])
            self.stages.append(stage_blocks)

            # Downsampling transition between stages
            if i < self.num_stages - 1:
                self.downsamples.append(ConvDownsample(channels[i], channels[i + 1]))
                curr_H = curr_H // 2
                curr_W = curr_W // 2

        # 3. Final Norm and Classifier Head
        self.norm = AffineTransform(channels[-1])
        self.head = nn.Linear(channels[-1], num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens, H, W = self.patch_embed(x)

        for i, stage in enumerate(self.stages):
            curr_H, curr_W = self.stage_resolutions[i]
            for block in stage:
                tokens = block(tokens)

            if i < len(self.downsamples):
                tokens, H, W = self.downsamples[i](tokens, curr_H, curr_W)

        tokens = self.norm(tokens)
        return tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward_features(x)
        # Global Average Pooling over tokens
        embedding = torch.mean(features, dim=1)
        logits = self.head(embedding)
        return logits

    def locality_injection(self):
        """
        Performs Structural Re-parameterization across all RepTokenMixLayers
        in all stages, fusing local convs into pure linear spatial operations.
        """
        for m in self.modules():
            if hasattr(m, "local_inject") and callable(m.local_inject):
                m.local_inject()
        self.deploy = True
