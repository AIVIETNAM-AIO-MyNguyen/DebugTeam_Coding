"""
ResMLP-LocalStem (Idea 3): Hybrid ResMLP with Local Perceptron Stem.

This architecture adopts the modern "Local -> Global" design paradigm:
- Early Layers (Stage 1: Stem): Uses Local Perceptron layers (DW 3x3 + DW 1x1 + Linear)
  to aggressively extract low-level local patterns (edges, textures) from the patch grid.
- Deep Layers (Stage 2: Body): Uses standard pure ResMLP layers (Global Token-Mix + Channel-Mix)
  to integrate global semantic relationships across the entire image.

During deployment (locality_injection):
- The Local Perceptron Stem blocks are re-parameterized into pure linear transformations.
- The entire model achieves the inference efficiency of pure ResMLP while retaining
  the superior convergence and representation power of early local feature extraction.
"""

from typing import Optional, Sequence
import torch
from torch import nn
from src.models.base import BaseClassifier, register_model
from src.models.res_mlp import ResMLPLayer, check_sizes
from src.models.res_mlp_rep_token import ResMLPRepTokenLayer


@register_model("res_mlp_local_stem")
@register_model("resmlp_local_stem")
@register_model("res_mlp_rep_stem")
class ResMLPLocalStem(BaseClassifier):
    """
    ResMLP with Local Stem Architecture (Idea 3).

    Args:
        image_size: Input resolution (e.g. 32 for CIFAR, 64 for Tiny-ImageNet).
        patch_size: Patch size (e.g. 4).
        in_channels: Input channels (3 for RGB).
        num_features: Channel dimension (e.g. 128).
        expansion_factor: Expansion factor in ChannelMixLayer (default: 2).
        num_layers: Total number of blocks in the network (default: 8).
        num_stem_blocks: Number of early blocks equipped with Local Perceptron (default: 2).
        num_classes: Number of output classification categories.
        reparam_conv_k: Kernel sizes for local conv branches in the stem (default: (1, 3)).
        deploy: Whether to construct in deployment mode.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_features: int = 128,
        expansion_factor: int = 2,
        num_layers: int = 8,
        num_stem_blocks: int = 2,
        num_classes: int = 100,
        reparam_conv_k: Sequence[int] = (1, 3),
        deploy: bool = False,
        **kwargs,
    ):
        super().__init__()
        num_patches = check_sizes(image_size, patch_size)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_stem_blocks = min(num_stem_blocks, num_layers)
        self.deploy = deploy

        self.patcher = nn.Conv2d(
            in_channels, num_features, kernel_size=patch_size, stride=patch_size
        )

        blocks = []
        for i in range(num_layers):
            if i < self.num_stem_blocks:
                # Stage 1: Local Stem (Local Perceptron + Channel-Mix)
                blocks.append(
                    ResMLPRepTokenLayer(
                        num_features=num_features,
                        num_patches=num_patches,
                        image_size=image_size,
                        patch_size=patch_size,
                        expansion_factor=expansion_factor,
                        reparam_conv_k=reparam_conv_k,
                        deploy=deploy,
                    )
                )
            else:
                # Stage 2: Global Body (Standard ResMLP: Token-Mix + Channel-Mix)
                blocks.append(
                    ResMLPLayer(
                        num_features=num_features,
                        num_patches=num_patches,
                        expansion_factor=expansion_factor,
                    )
                )

        self.mlps = nn.Sequential(*blocks)
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patcher(x)
        batch_size, num_features, _, _ = patches.shape
        patches = patches.permute(0, 2, 3, 1)
        patches = patches.view(batch_size, -1, num_features)

        embedding = self.mlps(patches)
        embedding = torch.mean(embedding, dim=1)
        logits = self.classifier(embedding)
        return logits

    def locality_injection(self):
        """Fuses all Local Stem blocks into deploy state."""
        for m in self.modules():
            if hasattr(m, "local_inject") and callable(m.local_inject):
                m.local_inject()
        self.deploy = True
