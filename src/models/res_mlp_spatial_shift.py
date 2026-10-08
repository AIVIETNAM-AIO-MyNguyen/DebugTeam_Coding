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

    CIFAR-100 example:

        Input:
            (B, 3, 32, 32)

        patch_size = 4

        Patch grid:
            8 x 8

        Number of patches:
            64

        Output:
            (B, 64, hidden_dim)
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        hidden_dim: int = 384,
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size

        self.patches_per_side = image_size // patch_size
        self.num_patches = self.patches_per_side ** 2

        self.patch_dim = (
            in_channels
            * patch_size
            * patch_size
        )

        # Pure linear patch projection.
        # No Conv2d is used.
        self.proj = nn.Linear(
            self.patch_dim,
            hidden_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # x:
        # (B, C, H, W)

        p = self.patch_size

        # ----------------------------------------------------
        # Extract non-overlapping patches
        # ----------------------------------------------------

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

        # (B, C, H/P, W/P, P, P)

        x = x.permute(
            0, 2, 3, 1, 4, 5
        )

        # (B, H/P, W/P, C, P, P)

        B = x.shape[0]

        x = x.reshape(
            B,
            self.num_patches,
            self.patch_dim,
        )

        # (B, N, patch_dim)

        x = self.proj(x)

        # (B, N, hidden_dim)

        return x


# ============================================================
# Affine Transformation
# ============================================================

class Affine(nn.Module):
    """
    Learnable affine transformation:

        Aff(x) = alpha * x + beta

    Same operation as the original ResMLP implementation.
    """

    def __init__(self, dim: int):
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
# Spatial Shift
# ============================================================

class SpatialShift(nn.Module):
    """
    Parameter-free spatial communication.

    Input:
        (B, N, C)

    where:

        N = H * W

    For CIFAR-100 with patch_size=4:

        N = 64
        H = W = 8

    The token sequence is reshaped back into its spatial grid:

        (B, N, C)
            ->
        (B, H, W, C)

    Channels are divided into four groups:

        Group 1 -> shift left
        Group 2 -> shift right
        Group 3 -> shift up
        Group 4 -> shift down

    Then the feature map is flattened back into tokens:

        (B, H, W, C)
            ->
        (B, N, C)

    The spatial-shift operation contains NO learnable parameters.
    """

    def __init__(
        self,
        grid_size: int,
    ):
        super().__init__()

        self.grid_size = grid_size

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # x:
        # (B, N, C)

        B, N, C = x.shape

        H = self.grid_size
        W = self.grid_size

        assert N == H * W, (
            f"Number of patches ({N}) does not match "
            f"grid size ({H} x {W})."
        )

        assert C % 4 == 0, (
            f"hidden_dim ({C}) must be divisible by 4 "
            "for the four spatial-shift groups."
        )

        # ----------------------------------------------------
        # Restore 2D spatial structure
        # ----------------------------------------------------

        x = x.reshape(
            B,
            H,
            W,
            C,
        )

        # Avoid modifying x in-place.
        shifted = torch.zeros_like(x)

        group_size = C // 4

        # ----------------------------------------------------
        # Group 1: shift LEFT
        # ----------------------------------------------------
        #
        # Original:
        #
        # A B C
        # D E F
        #
        # After left shift:
        #
        # B C 0
        # E F 0

        shifted[
            :,
            :,
            :-1,
            0:group_size,
        ] = x[
            :,
            :,
            1:,
            0:group_size,
        ]

        # ----------------------------------------------------
        # Group 2: shift RIGHT
        # ----------------------------------------------------

        shifted[
            :,
            :,
            1:,
            group_size:2 * group_size,
        ] = x[
            :,
            :,
            :-1,
            group_size:2 * group_size,
        ]

        # ----------------------------------------------------
        # Group 3: shift UP
        # ----------------------------------------------------

        shifted[
            :,
            :-1,
            :,
            2 * group_size:3 * group_size,
        ] = x[
            :,
            1:,
            :,
            2 * group_size:3 * group_size,
        ]

        # ----------------------------------------------------
        # Group 4: shift DOWN
        # ----------------------------------------------------

        shifted[
            :,
            1:,
            :,
            3 * group_size:,
        ] = x[
            :,
            :-1,
            :,
            3 * group_size:,
        ]

        # ----------------------------------------------------
        # Flatten spatial grid back into tokens
        # ----------------------------------------------------

        shifted = shifted.reshape(
            B,
            N,
            C,
        )

        return shifted


# ============================================================
# Channel MLP
# ============================================================

class ChannelMLP(nn.Module):
    """
    Cross-channel communication.

        hidden_dim
            ↓
        4 * hidden_dim
            ↓
          GELU
            ↓
        hidden_dim

    This is unchanged from ResMLP.
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
# Shift-ResMLP Block
# ============================================================

class ShiftResMLPBlock(nn.Module):
    """
    Modified ResMLP block.

    The original ResMLP cross-patch linear operation:

        Affine
            ↓
        transpose
            ↓
        Linear(N -> N)
            ↓
        transpose

    is replaced with:

        Affine
            ↓
        Spatial Shift

    Everything else remains equivalent to the original
    ResMLP block.


    Block structure:

    Spatial branch:

        x
        ↓
      Affine
        ↓
    SpatialShift
        ↓
    LayerScale
        ↓
       + x


    Channel branch:

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
       + x
    """

    def __init__(
        self,
        grid_size: int,
        hidden_dim: int,
        mlp_ratio: float = 4.0,
        layerscale_init: float = 1e-4,
        dropout: float = 0.0,
    ):
        super().__init__()

        # ----------------------------------------------------
        # Spatial communication branch
        # ----------------------------------------------------

        self.affine1 = Affine(hidden_dim)

        self.spatial_shift = SpatialShift(
            grid_size=grid_size,
        )

        # ----------------------------------------------------
        # Channel communication branch
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
        # 1. Spatial communication
        # ====================================================

        residual = x

        # Pre-Affine
        y = self.affine1(x)

        # Spatial shift:
        #
        # (B, N, C)
        #     ↓
        # (B, H, W, C)
        #     ↓
        # shift feature groups
        #     ↓
        # (B, N, C)

        y = self.spatial_shift(y)

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

        # Channel MLP
        y = self.channel_mlp(y)

        # LayerScale
        y = self.gamma2 * y

        # Residual connection
        x = residual + y

        return x


# ============================================================
# Shift-ResMLP
# ============================================================

@register_model("shift_resmlp")
@register_model("res_mlp_spatial_shift")
class ShiftResMLP(BaseClassifier):
    """
    Shift-ResMLP.

    Controlled modification of ResMLP where:

        Cross-Patch Linear
                ↓
          Spatial Shift

    All other major components remain the same:

    - linear patch embedding
    - Affine transformations
    - channel MLP
    - GELU only in channel MLP
    - LayerScale
    - residual connections
    - global average pooling
    - linear classification head
    - no self-attention
    - no convolutional feature extraction


    Default CIFAR-100 configuration:

        Input:
            (B, 3, 32, 32)

        patch_size:
            4

        Patch grid:
            8 x 8

        Tokens:
            64

        hidden_dim:
            384

        blocks:
            12
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

        assert image_size % patch_size == 0

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes

        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks

        # ----------------------------------------------------
        # Patch grid
        # ----------------------------------------------------

        self.grid_size = (
            image_size // patch_size
        )

        # ----------------------------------------------------
        # Patch embedding
        # ----------------------------------------------------

        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
        )

        # ----------------------------------------------------
        # Shift-ResMLP blocks
        # ----------------------------------------------------

        self.blocks = nn.ModuleList([
            ShiftResMLPBlock(
                grid_size=self.grid_size,
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

        # Patch embedding
        x = self.patch_embed(x)

        # (B, N, C)

        # Shift-ResMLP backbone
        for block in self.blocks:
            x = block(x)

        # Final Affine
        x = self.norm(x)

        # Global average pooling over patch tokens
        x = x.mean(dim=1)

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