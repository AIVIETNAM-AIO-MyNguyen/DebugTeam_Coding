"""
Pyramid-ResMLP Pure (Ablation Model):
Hierarchical ResMLP with Overlapping Patching & Pure Linear Spatial Mixing (No Parallel CNN Branches).

This model serves as an Ablation Baseline to scientifically isolate the contribution
of the parallel Depthwise Convolution branches in PyramidResMLP.
- Keeps: Overlapping Patch Embedding (Stem) and Convolutional Downsamplers between stages.
- Removes: All parallel DW Conv branches inside the TokenMix layers (pure Linear(N, N) spatial mixing).
"""

from typing import Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from src.models.base import BaseClassifier, register_model
from src.models.res_mlp import AffineTransform, ChannelMixLayer
from src.models.pyramid_res_mlp import OverlappingPatchEmbed, ConvDownsample
from src.models.layers import DropPath


class PureTokenMixLayer(nn.Module):
    """
    Pure ResMLP TokenMixLayer: Global Linear(N, N) spatial mixing without any parallel conv branches.
    Supports DropPath for regularization.
    """

    def __init__(self, features: int, patches: int, drop_path: float = 0.0):
        super().__init__()
        self.features = features
        self.patches = patches
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.aff1 = AffineTransform(features=features)
        self.fc1 = nn.Linear(patches, patches)
        self.aff2 = AffineTransform(features=features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        residual = x
        x = self.aff1(x)
        # Spatial mixing: transpose to (B, C, N), apply Linear(N, N), transpose back
        x = self.fc1(x.transpose(1, 2)).transpose(1, 2)
        x = self.aff2(x)
        return residual + self.drop_path(x)


class PurePyramidResMLPBlock(nn.Module):
    """
    Block inside Pure Pyramid Stage: PureTokenMixLayer + ChannelMixLayer with DropPath.
    """

    def __init__(
        self,
        features: int,
        patches: int,
        expansion_factor: int = 4,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.token_mix = PureTokenMixLayer(features=features, patches=patches, drop_path=drop_path)
        self.channel_mix = ChannelMixLayer(features, expansion_factor, drop_path=drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_mix(x)
        x = self.channel_mix(x)
        return x


@register_model("pyramid_res_mlp_pure")
@register_model("pure_pyramid_res_mlp")
@register_model("pyramid_resmlp_pure")
class PyramidResMLPPure(BaseClassifier):
    """
    Pyramid-ResMLP Pure (Ablation): Hierarchical ResMLP with pure Linear TokenMix.

    Args:
        image_size: Input resolution (e.g. 32 for CIFAR).
        patch_size: Stem downsampling factor (default: 2).
        in_channels: Input channels (default: 3).
        num_classes: Classification categories (default: 100).
        channels: Channel dimensions for each stage (default: (64, 128, 256)).
        num_blocks: Blocks per stage (default: (2, 2, 4)).
        expansion_factor: MLP expansion factor in ChannelMix (default: 4).
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
        drop_path_rate: float = 0.0,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        assert len(channels) == len(num_blocks), "Length of channels and num_blocks must match."

        # Support alias drop_path in kwargs
        if "drop_path" in kwargs and drop_path_rate == 0.0:
            drop_path_rate = float(kwargs["drop_path"])

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_stages = len(channels)
        self.drop_path_rate = drop_path_rate
        self.deploy = True  # Already purely linear, no conv branches to fuse

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

        # Stochastic depth decay rule (linear schedule across all blocks)
        total_blocks = sum(num_blocks)
        dpr = (
            [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
            if drop_path_rate > 0.0
            else [0.0] * total_blocks
        )

        cur_block_idx = 0
        # 2. Build Pure Pyramid Stages
        for i in range(self.num_stages):
            C = channels[i]
            N = curr_H * curr_W
            self.stage_resolutions.append((curr_H, curr_W))

            stage_blocks = nn.ModuleList([
                PurePyramidResMLPBlock(
                    features=C,
                    patches=N,
                    expansion_factor=expansion_factor,
                    drop_path=dpr[cur_block_idx + j],
                )
                for j in range(num_blocks[i])
            ])
            cur_block_idx += num_blocks[i]
            self.stages.append(stage_blocks)

            if i < self.num_stages - 1:
                self.downsamples.append(ConvDownsample(channels[i], channels[i + 1]))
                curr_H = curr_H // 2
                curr_W = curr_W // 2

        # 3. Final Norm, Dropout, and Classifier Head
        self.norm = AffineTransform(channels[-1])
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
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
        embedding = torch.mean(features, dim=1)
        embedding = self.dropout(embedding)
        logits = self.head(embedding)
        return logits

    def locality_injection(self):
        """No-op since this pure ablation model has no conv branches inside blocks."""
        self.deploy = True
