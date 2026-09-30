"""
MLP-Mixer: An all-MLP Architecture for Vision.
Reference: Tolstikhin et al., NeurIPS 2021 (https://arxiv.org/abs/2105.01601)

Features:
- Token-mixing MLP (transposed over sequence dimension)
- Channel-mixing MLP (over feature channel dimension)
- LayerNorm and skip connections
- Fully configurable for CIFAR-100 (32x32), Tiny-ImageNet (64x64), and ImageNet (224x224)
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
from src.models.base import BaseClassifier, register_model


class PatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches and linearly projects each patch to hidden_dim.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_dim: int = 256,
    ):
        super().__init__()
        assert image_size % patch_size == 0, f"image_size ({image_size}) must be divisible by patch_size ({patch_size})"

        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, hidden_dim, H/P, W/P)
        x = x.flatten(2)  # (B, hidden_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, hidden_dim)
        return x


class MLPBlock(nn.Module):
    """
    Standard Feed-Forward Network (FFN) used for both token and channel mixing.
    LayerNorm is applied prior to mixing (Pre-Norm architecture).
    """

    def __init__(self, in_features: int, hidden_features: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MixerBlock(nn.Module):
    """
    Single MLP-Mixer block consisting of:
    1. Token-mixing MLP: Communicates information across spatial locations (patches).
    2. Channel-mixing MLP: Communicates information across channels/features.
    Both stages use LayerNorm and residual connections.
    """

    def __init__(
        self,
        num_patches: int,
        hidden_dim: int,
        tokens_mlp_dim: int,
        channels_mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Token mixing
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.token_mlp = MLPBlock(num_patches, tokens_mlp_dim, dropout)

        # Channel mixing
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.channel_mlp = MLPBlock(hidden_dim, channels_mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, num_patches, hidden_dim)

        # 1. Token Mixing: across spatial tokens
        residual = x
        y = self.norm1(x)
        y = y.transpose(1, 2)  # (B, hidden_dim, num_patches)
        y = self.token_mlp(y)  # (B, hidden_dim, num_patches)
        y = y.transpose(1, 2)  # (B, num_patches, hidden_dim)
        x = residual + y

        # 2. Channel Mixing: across feature channels
        residual = x
        y = self.norm2(x)
        y = self.channel_mlp(y)  # (B, num_patches, hidden_dim)
        x = residual + y

        return x


@register_model("mlp_mixer")
class MLPMixer(BaseClassifier):
    """
    MLP-Mixer architecture for image classification.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 100,
        hidden_dim: int = 256,
        num_blocks: int = 8,
        tokens_mlp_dim: int = 128,
        channels_mlp_dim: int = 512,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Mixer Blocks
        self.blocks = nn.ModuleList([
            MixerBlock(
                num_patches=num_patches,
                hidden_dim=hidden_dim,
                tokens_mlp_dim=tokens_mlp_dim,
                channels_mlp_dim=channels_mlp_dim,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        # Pre-head LayerNorm & Classification Head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """Initializes weights using truncated normal / standard Xavier."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts representations before classification head."""
        x = self.patch_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Global Average Pooling across patch tokens
        x = x.mean(dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        logits = self.head(feat)
        return logits
