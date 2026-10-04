"""
ResMLP-RepToken (Idea 1): ResMLP with Re-parameterizable Token Mixing.

In this architecture, EVERY TokenMixLayer in ResMLP is augmented with parallel
local Depthwise Convolution branches (DW 3x3 + BN and DW 1x1 + BN) alongside
the global Linear(N, N) spatial mixer.

During training:
    y_spatial = GlobalLinear(x) + DW3x3(x_2d) + DW1x1(x_2d)

During deployment (locality_injection):
    All convolution branches are converted into equivalent spatial matrices and
    analytically fused into a single channel-specific spatial transformation:
        y_spatial = x @ W_deploy.T + b_deploy
    Zero extra latency, zero extra convolution ops, but with strong learned local priors!
"""

from typing import Optional, Sequence, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F
from src.models.base import BaseClassifier, register_model
from src.models.res_mlp import AffineTransform, ChannelMixLayer, check_sizes


from src.models.layers import DropPath


class RepTokenMixLayer(nn.Module):
    """
    Re-parameterizable Token Mixing Layer.

    Combines global Linear spatial mixing with local Depthwise Convolution branches.
    At deploy time, all branches are merged into a single linear operator.
    Supports DropPath (Stochastic Depth) for regularization.
    """

    def __init__(
        self,
        features: int,
        patches: int,
        image_size: int,
        patch_size: int,
        reparam_conv_k: Sequence[int] = (1, 3),
        drop_path: float = 0.0,
        deploy: bool = False,
    ):
        super().__init__()
        self.features = features
        self.patches = patches
        self.Hp = image_size // patch_size
        self.Wp = image_size // patch_size
        assert self.Hp * self.Wp == patches, f"Patches ({patches}) must equal Hp*Wp ({self.Hp}*{self.Wp})"
        self.deploy = deploy
        self.reparam_conv_k = reparam_conv_k
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.aff1 = AffineTransform(features=features)
        self.aff2 = AffineTransform(features=features)

        if deploy:
            self.register_buffer("fused_weight", torch.zeros(features, patches, patches))
            self.register_buffer("fused_bias", torch.zeros(features, patches))
        else:
            self.fc1 = nn.Linear(patches, patches)
            self.conv_branches = nn.ModuleList()
            for k in reparam_conv_k:
                conv = nn.Conv2d(
                    in_channels=features,
                    out_channels=features,
                    kernel_size=k,
                    stride=1,
                    padding=k // 2,
                    groups=features,
                    bias=False,
                )
                bn = nn.BatchNorm2d(features)
                self.conv_branches.append(nn.Sequential(conv, bn))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C)
        residual = x
        x = self.aff1(x)
        x_trans = x.transpose(1, 2)  # (B, C, N)

        if self.deploy:
            # Fused inference: single tensor multiplication
            # x_trans is (B, C, N), fused_weight is (C, N, N)
            x_spatial = torch.einsum("bcn,cmn->bcm", x_trans, self.fused_weight) + self.fused_bias
        else:
            global_out = self.fc1(x_trans)  # (B, C, N)
            x_2d = x_trans.reshape(-1, self.features, self.Hp, self.Wp)
            conv_out = sum(branch(x_2d) for branch in self.conv_branches).flatten(2)
            x_spatial = global_out + conv_out

        x = self.aff2(x_spatial.transpose(1, 2))  # (B, N, C)
        return residual + self.drop_path(x)

    def local_inject(self):
        """Merges all local conv branches into the linear spatial operator."""
        if self.deploy:
            return

        N = self.patches
        C = self.features
        device = self.fc1.weight.device
        dtype = self.fc1.weight.dtype
        eye = torch.eye(N, device=device, dtype=dtype).reshape(N, 1, self.Hp, self.Wp)

        W_conv = torch.zeros(C, N, N, device=device, dtype=dtype)
        b_conv = torch.zeros(C, N, device=device, dtype=dtype)

        for branch in self.conv_branches:
            conv = branch[0]
            bn = branch[1]
            std = (bn.running_var + bn.eps).sqrt()
            t = (bn.weight / std).reshape(-1, 1, 1, 1)
            w = conv.weight * t
            b = bn.bias - bn.running_mean * bn.weight / std
            k = conv.kernel_size[0]

            for c in range(C):
                k_c = w[c : c + 1]
                m = F.conv2d(eye, k_c, padding=k // 2).reshape(N, N).t()
                W_conv[c] += m
                b_conv[c] += b[c].repeat(N)

        W_deploy = self.fc1.weight.unsqueeze(0) + W_conv  # (C, N, N)
        b_deploy = self.fc1.bias.unsqueeze(0) + b_conv    # (C, N)

        self.register_buffer("fused_weight", W_deploy)
        self.register_buffer("fused_bias", b_deploy)

        del self.fc1
        del self.conv_branches
        self.deploy = True


class ResMLPRepTokenLayer(nn.Module):
    """ResMLP Block with RepTokenMixLayer."""

    def __init__(
        self,
        num_features: int,
        num_patches: int,
        image_size: int,
        patch_size: int,
        expansion_factor: int = 4,
        reparam_conv_k: Sequence[int] = (1, 3),
        deploy: bool = False,
    ):
        super().__init__()
        self.token_mix = RepTokenMixLayer(
            features=num_features,
            patches=num_patches,
            image_size=image_size,
            patch_size=patch_size,
            reparam_conv_k=reparam_conv_k,
            deploy=deploy,
        )
        self.channel_mix = ChannelMixLayer(num_features, expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_mix(x)
        return self.channel_mix(x)


@register_model("res_mlp_rep_token")
@register_model("resmlp_rep_token")
@register_model("res_mlp_reptoken")
class ResMLPRepToken(BaseClassifier):
    """
    ResMLP-RepToken Architecture (Idea 1).

    ResMLP where every block uses RepTokenMixLayer to inject local inductive bias
    into the spatial mixing step.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_features: int = 128,
        expansion_factor: int = 2,
        num_layers: int = 8,
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
        self.deploy = deploy

        self.patcher = nn.Conv2d(
            in_channels, num_features, kernel_size=patch_size, stride=patch_size
        )
        self.mlps = nn.Sequential(
            *[
                ResMLPRepTokenLayer(
                    num_features=num_features,
                    num_patches=num_patches,
                    image_size=image_size,
                    patch_size=patch_size,
                    expansion_factor=expansion_factor,
                    reparam_conv_k=reparam_conv_k,
                    deploy=deploy,
                )
                for _ in range(num_layers)
            ]
        )
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
        """Fuses all RepTokenMixLayers across all blocks into deploy state."""
        for m in self.modules():
            if hasattr(m, "local_inject") and callable(m.local_inject):
                m.local_inject()
        self.deploy = True
