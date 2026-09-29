"""
ResMLP: Feedforward Networks for Image Classification with Data-Efficient Training.

Adapted for CIFAR-100 (32x32).

Main characteristics:
- Patch embedding using non-overlapping image patches
- Cross-patch communication using a single Linear layer
- Cross-channel communication using a two-layer MLP with GELU
- Learned affine transformations instead of LayerNorm
- Residual connections
"""

import torch
import torch.nn as nn

from src.models.base import BaseClassifier, register_model

# Implement PatchEmbedding similar to mlp_mixer.py
class PatchEmbedding(nn.Module):
    """
    Splits image into non-overlapping patches and linearly projects
    each patch to hidden_dim.

    Input:
        (B, C, H, W)

    Output:
        (B, num_patches, hidden_dim)
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_dim: int = 384,
    ):
        super().__init__()

        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible "
            f"by patch_size ({patch_size})"
        )

        self.num_patches = (image_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)

        x = self.proj(x)
        # (B, hidden_dim, H/P, W/P)

        x = x.flatten(2)
        # (B, hidden_dim, num_patches)

        x = x.transpose(1, 2)
        # (B, num_patches, hidden_dim)

        return x


class Affine(nn.Module):
    """
    Learnable affine transformation used by ResMLP.

    Performs:

        Aff(x) = alpha * x + beta

    Unlike LayerNorm, this operation does not calculate
    statistics from the input.
    """

    def __init__(self, dim: int):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + self.beta


class CrossPatchLinear(nn.Module):
    """
    Cross-patch communication.

    ResMLP uses a single Linear layer across patch positions,
    rather than an MLP.

    Input:
        (B, num_patches, hidden_dim)

    The tensor is transposed so that Linear operates on
    the patch dimension.
    """

    def __init__(self, num_patches: int):
        super().__init__()

        self.linear = nn.Linear(num_patches, num_patches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, num_patches, hidden_dim)

        x = x.transpose(1, 2)
        # (B, hidden_dim, num_patches)

        x = self.linear(x)
        # Linear operates across patches

        x = x.transpose(1, 2)
        # (B, num_patches, hidden_dim)

        return x


class CrossChannelMLP(nn.Module):
    """
    Cross-channel communication.

    Two-layer MLP:

        hidden_dim
            ↓
        mlp_dim
            ↓
          GELU
            ↓
        hidden_dim

    This operates independently on every patch.
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x


class ResMLPBlock(nn.Module):
    """
    Single ResMLP block consisting of:

    1. Cross-patch communication
       Affine -> Linear across patches -> residual

    2. Cross-channel communication
       Affine -> MLP -> residual

    Input/output:
        (B, num_patches, hidden_dim)
    """

    def __init__(
        self,
        num_patches: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Cross-patch
        self.affine1 = Affine(hidden_dim)
        self.cross_patch = CrossPatchLinear(num_patches)

        # Cross-channel
        self.affine2 = Affine(hidden_dim)
        self.cross_channel = CrossChannelMLP(
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # --------------------------------
        # 1. Cross-patch communication
        # --------------------------------

        residual = x

        y = self.affine1(x)
        y = self.cross_patch(y)

        x = residual + y

        # --------------------------------
        # 2. Cross-channel communication
        # --------------------------------

        residual = x

        y = self.affine2(x)
        y = self.cross_channel(y)

        x = residual + y

        return x


@register_model("resmlp")
class ResMLP(BaseClassifier):
    """
    ResMLP architecture adapted for image classification.

    Default configuration targets CIFAR-100:
        image size  : 32x32
        patch size  : 4x4
        patches     : 8x8 = 64
        hidden dim  : 384
        blocks      : 12
        classes     : 100
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 100,
        hidden_dim: int = 384,
        num_blocks: int = 12,
        mlp_dim: int = 1536,
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

        # ResMLP Blocks
        self.blocks = nn.ModuleList([
            ResMLPBlock(
                num_patches=num_patches,
                hidden_dim=hidden_dim,
                mlp_dim=mlp_dim,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        # Final affine transformation
        self.norm = Affine(hidden_dim)

        # Classification head
        self.head = nn.Linear(
            hidden_dim,
            num_classes,
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialize Linear and Conv2d layers.
        """

        for m in self.modules():

            if isinstance(m, nn.Linear):

                nn.init.trunc_normal_(
                    m.weight,
                    std=0.02,
                )

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.Conv2d):

                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                )

                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract representation before classification head.
        """

        # Patch embedding
        x = self.patch_embed(x)

        # ResMLP blocks
        for block in self.blocks:
            x = block(x)

        # Final affine
        x = self.norm(x)

        # Global Average Pooling across patches
        x = x.mean(dim=1)

        return x

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        feat = self.forward_features(x)

        logits = self.head(feat)

        return logits