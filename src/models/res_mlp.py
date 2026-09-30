"""
ResMLP Architecture with Local Perceptron (Structural Re-parameterization).

References:
- ResMLP: Feedforward networks for image classification with data-efficient training
  (Touvron et al., NeurIPS 2021, https://arxiv.org/abs/2105.03404)
- RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition
  (Ding et al., CVPR 2022, https://arxiv.org/abs/2105.01883)

Key Innovations:
1. Affine Transformations: Replaces LayerNorm/BatchNorm with learnable scale & shift.
2. Cross-Patch Linear Communication: Global spatial token mixing via full FC layer.
3. Local Perceptron (Locality Injection): Parallel 1x1 and 3x3 depthwise conv branches
   during training to inject 2D inductive bias.
4. Structural Re-parameterization: Fuses conv branches into the cross-patch Linear matrix
   via `model.locality_injection()`, yielding a pure MLP with 0 extra FLOPs at inference.
"""

from typing import Sequence, Tuple
import torch
from torch import nn
from torch.nn import functional as F
from src.models.base import BaseClassifier, register_model


class AffineTransform(nn.Module):
    """Element-wise affine transformation: y = alpha * x + beta."""
    def __init__(self, features: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, features))
        self.beta = nn.Parameter(torch.zeros(1, 1, features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * x + self.beta


class LocalPerceptron(nn.Module):
    """
    Local Perceptron: Re-parameterizable local convolution branches.
    Injects 2D spatial inductive bias during training via parallel 1x1 and 3x3 convs + BN.
    At inference, converts conv kernels into equivalent FC weights and fuses into fc1.
    """
    def __init__(
        self,
        grid_h: int,
        grid_w: int,
        reparam_conv_k: Sequence[int] = (1, 3),
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.num_patches = grid_h * grid_w
        self.reparam_conv_k = reparam_conv_k
        self.deploy = False

        if reparam_conv_k is not None:
            for k in reparam_conv_k:
                pad = k // 2
                conv = nn.Conv2d(1, 1, kernel_size=k, stride=1, padding=pad, bias=False)
                bn = nn.BatchNorm2d(1)
                self.add_module(f"conv{k}", conv)
                self.add_module(f"bn{k}", bn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, num_patches, features)
        Returns: Tensor of shape (B, num_patches, features)
        """
        if self.deploy or self.reparam_conv_k is None:
            return 0

        B, S, C = x.shape
        # Reshape to spatial 2D feature map per channel: (B * C, 1, grid_h, grid_w)
        x_2d = x.permute(0, 2, 1).reshape(B * C, 1, self.grid_h, self.grid_w)

        out = 0
        for k in self.reparam_conv_k:
            conv = getattr(self, f"conv{k}")
            bn = getattr(self, f"bn{k}")
            out = out + bn(conv(x_2d))

        # Reshape back to (B, S, C)
        out = out.reshape(B, C, S).permute(0, 2, 1)
        return out

    def get_equivalent_fc(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Derives equivalent (W_fc, b_fc) of shape (S, S) and (S,) using impulse response.
        """
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        fused_kernels = []
        for k in self.reparam_conv_k:
            conv = getattr(self, f"conv{k}")
            bn = getattr(self, f"bn{k}")
            std = (bn.running_var + bn.eps).sqrt()
            w = conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1)
            b = bn.bias - bn.running_mean * bn.weight / std
            fused_kernels.append((k, w, b))

        largest_k = max(self.reparam_conv_k)
        total_w = 0
        total_b = 0
        for k, w, b in fused_kernels:
            if k < largest_k:
                pad = (largest_k - k) // 2
                w = F.pad(w, [pad, pad, pad, pad])
            total_w = total_w + w
            total_b = total_b + b

        # Impulse response through identity matrix
        I = torch.eye(self.num_patches, device=device, dtype=dtype).reshape(
            self.num_patches, 1, self.grid_h, self.grid_w
        )
        conv_I = F.conv2d(I, total_w, padding=largest_k // 2)
        W_conv = conv_I.reshape(self.num_patches, self.num_patches).t()
        b_conv = total_b.repeat(self.num_patches)
        return W_conv, b_conv


class TokenMixLayer(nn.Module):
    """
    Cross-patch communication sublayer with optional Local Perceptron injection.
    """
    def __init__(
        self,
        features: int,
        patches: int,
        grid_h: int = 8,
        grid_w: int = 8,
        use_local_perceptron: bool = True,
        reparam_conv_k: Sequence[int] = (1, 3),
    ):
        super().__init__()
        self.features = features
        self.patches = patches
        self.use_local_perceptron = use_local_perceptron

        self.aff1 = AffineTransform(features=features)
        self.fc1 = nn.Linear(patches, patches)
        self.aff2 = AffineTransform(features=features)

        if use_local_perceptron:
            self.local_perceptron = LocalPerceptron(
                grid_h=grid_h,
                grid_w=grid_w,
                reparam_conv_k=reparam_conv_k,
            )
        else:
            self.local_perceptron = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.aff1(x)

        # Cross-patch linear
        fc_out = self.fc1(x.transpose(1, 2)).transpose(1, 2)

        # Local perceptron conv branches (during training)
        if self.local_perceptron is not None and not self.local_perceptron.deploy:
            fc_out = fc_out + self.local_perceptron(x)

        x = self.aff2(fc_out)
        return x + residual

    def local_inject(self):
        """Fuses Local Perceptron convolution branches into fc1."""
        if self.local_perceptron is not None and not self.local_perceptron.deploy:
            w_lp, b_lp = self.local_perceptron.get_equivalent_fc()
            self.fc1.weight.data = self.fc1.weight.data + w_lp
            if self.fc1.bias is None:
                self.fc1.bias = nn.Parameter(b_lp)
            else:
                self.fc1.bias.data = self.fc1.bias.data + b_lp
            self.local_perceptron.deploy = True


class ChannelMixLayer(nn.Module):
    """Cross-channel feed-forward sublayer (FCN)."""
    def __init__(self, features: int, expansion_factor: int):
        super().__init__()
        num_hidden = features * expansion_factor
        self.aff1 = AffineTransform(features=features)
        self.fc1 = nn.Linear(features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, features)
        self.aff2 = AffineTransform(features=features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.aff1(x)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = self.aff2(x)
        return x + residual


class ResMLPLayer(nn.Module):
    """A full ResMLP layer containing TokenMix and ChannelMix sublayers."""
    def __init__(
        self,
        num_features: int,
        num_patches: int,
        expansion_factor: int,
        grid_h: int = 8,
        grid_w: int = 8,
        use_local_perceptron: bool = True,
        reparam_conv_k: Sequence[int] = (1, 3),
    ):
        super().__init__()
        self.token_mix = TokenMixLayer(
            features=num_features,
            patches=num_patches,
            grid_h=grid_h,
            grid_w=grid_w,
            use_local_perceptron=use_local_perceptron,
            reparam_conv_k=reparam_conv_k,
        )
        self.channel_mix = ChannelMixLayer(num_features, expansion_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_mix(x)
        return self.channel_mix(x)

    def local_inject(self):
        self.token_mix.local_inject()


def check_sizes(image_size: int, patch_size: int) -> Tuple[int, int, int]:
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisible by `patch_size`"
    num_patches = sqrt_num_patches ** 2
    return num_patches, sqrt_num_patches, sqrt_num_patches


@register_model("res_mlp")
@register_model("resmlp")
class ResMLP(BaseClassifier):
    """
    ResMLP Vision Classifier with optional Local Perceptron (Locality Injection).
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
        use_local_perceptron: bool = True,
        reparam_conv_k: Sequence[int] = (1, 3),
        **kwargs
    ):
        super().__init__()
        num_patches, grid_h, grid_w = check_sizes(image_size, patch_size)
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.use_local_perceptron = use_local_perceptron
        self.deploy = False

        self.patcher = nn.Conv2d(
            in_channels, num_features, kernel_size=patch_size, stride=patch_size
        )
        self.mlps = nn.Sequential(
            *[
                ResMLPLayer(
                    num_features=num_features,
                    num_patches=num_patches,
                    expansion_factor=expansion_factor,
                    grid_h=grid_h,
                    grid_w=grid_w,
                    use_local_perceptron=use_local_perceptron,
                    reparam_conv_k=reparam_conv_k,
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
        # patches.shape == (batch_size, num_patches, num_features)
        embedding = self.mlps(patches)
        embedding = torch.mean(embedding, dim=1)
        logits = self.classifier(embedding)
        return logits

    def locality_injection(self):
        """
        Merges all Local Perceptron convolution branches into the cross-patch Linear layers.
        Switches model to deploy mode.
        """
        for layer in self.mlps:
            if hasattr(layer, "local_inject"):
                layer.local_inject()
        self.deploy = True