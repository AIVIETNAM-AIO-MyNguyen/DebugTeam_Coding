"""
ResMLP: Feedforward Networks for Image Classification
with Data-Efficient Training.

Reference:
Touvron et al.
https://arxiv.org/abs/2105.03404

This implementation follows the original ResMLP architecture,
adapted for CIFAR-100 (32x32).

Main characteristics:
- Non-overlapping image patches
- Linear patch embedding
- Cross-patch communication using a single Linear layer
- Cross-channel communication using a two-layer MLP
- GELU activation only in the channel MLP
- Affine transformation instead of LayerNorm
- LayerScale on residual branches
- No positional embeddings
- No self-attention
- No convolutional feature extraction
"""

import torch
import torch.nn as nn

from src.models.base import BaseClassifier, register_model


# ============================================================
# Patch Embedding
# ============================================================

class PatchEmbedding(nn.Module):
    """
    Split the image into non-overlapping patches and linearly
    project every flattened patch into hidden_dim.

    Example for CIFAR-100:

        Input:
            (B, 3, 32, 32)

        patch_size = 4

        Number of patches:
            (32 / 4)^2 = 64

        Patch dimension:
            3 * 4 * 4 = 48

        Output:
            (B, 64, hidden_dim)

    We explicitly use patch extraction + nn.Linear rather than
    Conv2d to make the all-MLP nature of ResMLP clear.
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
            f"image_size ({image_size}) must be divisible by "
            f"patch_size ({patch_size})"
        )

        self.image_size = image_size
        self.patch_size = patch_size

        patches_per_side = image_size // patch_size
        self.num_patches = patches_per_side ** 2

        self.patch_dim = (
            in_channels
            * patch_size
            * patch_size
        )

        # Linear projection:
        #
        # flattened patch -> embedding
        #
        # CIFAR example:
        # 48 -> 384
        self.proj = nn.Linear(
            self.patch_dim,
            hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:
        # (B, C, H, W)

        p = self.patch_size

        # Extract non-overlapping patches
        x = x.unfold(
            dimension=2,
            size=p,
            step=p,
        )

        x = x.unfold(
            dimension=3,
            size=p,
            step=p,
        )

        # Shape:
        # (B, C, H/P, W/P, P, P)

        x = x.permute(
            0, 2, 3, 1, 4, 5
        )

        # Shape:
        # (B, H/P, W/P, C, P, P)

        B = x.shape[0]

        x = x.reshape(
            B,
            self.num_patches,
            self.patch_dim,
        )

        # Shape:
        # (B, num_patches, patch_dim)

        x = self.proj(x)

        # Shape:
        # (B, num_patches, hidden_dim)

        return x


# ============================================================
# Affine Transformation
# ============================================================

class Affine(nn.Module):
    """
    Learnable affine transformation used by ResMLP.

        Aff(x) = alpha * x + beta

    alpha and beta operate independently on every feature
    channel.

    Unlike LayerNorm:
    - no mean calculation
    - no variance calculation
    - no input-dependent normalization

    Original ResMLP initialization:

        alpha = 1
        beta  = 0

    Therefore, at initialization:

        Aff(x) = x
    """

    def __init__(
        self,
        dim: int,
    ):
        super().__init__()

        self.alpha = nn.Parameter(
            torch.ones(dim)
        )

        self.beta = nn.Parameter(
            torch.zeros(dim)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.alpha * x + self.beta


# ============================================================
# Channel MLP
# ============================================================

class ChannelMLP(nn.Module):
    """
    Cross-channel communication.

    Each patch is processed independently using:

        hidden_dim
            ↓
        4 * hidden_dim
            ↓
          GELU
            ↓
        hidden_dim

    Unlike the cross-patch operation, this sublayer contains
    a nonlinear activation.
    """

    def __init__(
        self,
        hidden_dim: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()

        mlp_dim = int(
            hidden_dim * mlp_ratio
        )

        self.fc1 = nn.Linear(
            hidden_dim,
            mlp_dim,
        )

        self.act = nn.GELU()

        self.drop1 = nn.Dropout(dropout)

        self.fc2 = nn.Linear(
            mlp_dim,
            hidden_dim,
        )

        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)

        return x


# ============================================================
# ResMLP Block
# ============================================================

class ResMLPBlock(nn.Module):
    """
    One ResMLP block.

    It contains two residual sublayers:

    1. Cross-patch communication

        x
        ↓
        Affine
        ↓
        transpose
        ↓
        Linear across patches
        ↓
        transpose
        ↓
        LayerScale
        ↓
        residual addition


    2. Cross-channel communication

        x
        ↓
        Affine
        ↓
        Linear
        ↓
        GELU
        ↓
        Linear
        ↓
        LayerScale
        ↓
        residual addition


    Tensor representation:

        (B, num_patches, hidden_dim)
    """

    def __init__(
        self,
        num_patches: int,
        hidden_dim: int,
        mlp_ratio: float = 4.0,
        layerscale_init: float = 1e-4,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Cross-patch branch
        # ----------------------------------------------------

        self.affine1 = Affine(hidden_dim)

        self.cross_patch = nn.Linear(
            num_patches,
            num_patches,
        )

        # ----------------------------------------------------
        # Cross-channel branch
        # ----------------------------------------------------

        self.affine2 = Affine(hidden_dim)

        self.channel_mlp = ChannelMLP(
            hidden_dim=hidden_dim,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # ----------------------------------------------------
        # LayerScale
        # ----------------------------------------------------

        self.gamma1 = nn.Parameter(
            layerscale_init
            * torch.ones(hidden_dim)
        )

        self.gamma2 = nn.Parameter(
            layerscale_init
            * torch.ones(hidden_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # ====================================================
        # 1. Cross-patch communication
        # ====================================================

        residual = x

        # Pre-Affine
        y = self.affine1(x)

        # (B, patches, channels)
        #
        # Linear must operate over patches,
        # therefore transpose:
        #
        # (B, channels, patches)

        y = y.transpose(1, 2)

        y = self.cross_patch(y)

        y = y.transpose(1, 2)

        # LayerScale
        y = self.gamma1 * y

        # Residual connection
        x = residual + y

        # ====================================================
        # 2. Cross-channel communication
        # ====================================================

        residual = x

        # Pre-Affine
        y = self.affine2(x)

        # MLP operates over hidden_dim
        y = self.channel_mlp(y)

        # LayerScale
        y = self.gamma2 * y

        # Residual connection
        x = residual + y

        return x


# ============================================================
# ResMLP
# ============================================================

@register_model("resmlp")
class ResMLP(BaseClassifier):
    """
    ResMLP image classifier.

    Default configuration is based on ResMLP-S12,
    adapted to CIFAR-100.

    Original S12:
        hidden_dim = 384
        depth      = 12
        mlp_ratio  = 4

    CIFAR adaptation:
        image_size = 32
        patch_size = 4
        classes    = 100

    Resulting representation:

        (B, 3, 32, 32)

                ↓

        64 patches of 4x4

                ↓

        (B, 64, 384)

                ↓

        12 ResMLP blocks

                ↓

        (B, 64, 384)

                ↓

        Affine

                ↓

        Average pooling over patches

                ↓

        (B, 384)

                ↓

        Linear classifier

                ↓

        (B, 100)
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 100,
        hidden_dim: int = 384,
        num_blocks: int = 12,
        mlp_ratio: float = 4.0,
        layerscale_init: float = 1e-4,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # ----------------------------------------------------
        # Patch embedding
        # ----------------------------------------------------

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
        )

        num_patches = (
            self.patch_embed.num_patches
        )

        # ----------------------------------------------------
        # ResMLP blocks
        # ----------------------------------------------------

        self.blocks = nn.ModuleList([
            ResMLPBlock(
                num_patches=num_patches,
                hidden_dim=hidden_dim,
                mlp_ratio=mlp_ratio,
                layerscale_init=layerscale_init,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        # ----------------------------------------------------
        # Final Affine
        # ----------------------------------------------------

        self.norm = Affine(hidden_dim)

        # ----------------------------------------------------
        # Classification head
        # ----------------------------------------------------

        self.head = nn.Linear(
            hidden_dim,
            num_classes,
        )

        self._init_weights()

    # ========================================================
    # Weight initialization
    # ========================================================

    def _init_weights(self):
        """
        Initialize Linear layers following the standard
        initialization used by ResMLP/ViT implementations.
        """

        for module in self.modules():

            if isinstance(
                module,
                nn.Linear,
            ):

                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02,
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

    # ========================================================
    # Feature extraction
    # ========================================================

    def forward_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # ----------------------------------------------------
        # Patch embedding
        # ----------------------------------------------------

        x = self.patch_embed(x)

        # Shape:
        # (B, num_patches, hidden_dim)

        # ----------------------------------------------------
        # ResMLP backbone
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(x)

        # ----------------------------------------------------
        # Final Affine
        # ----------------------------------------------------

        x = self.norm(x)

        # ----------------------------------------------------
        # Global average pooling over patch tokens
        # ----------------------------------------------------

        x = x.mean(dim=1)

        # Shape:
        # (B, hidden_dim)

        return x

    # ========================================================
    # Classification
    # ========================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        features = self.forward_features(x)

        logits = self.head(features)

        return logits