"""
RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition /
RepMLPNet: Hierarchical Vision MLP with Re-parameterized Locality.
Reference: Ding et al., CVPR 2022 (https://arxiv.org/abs/2112.11081)
Source Repository: D:\\RepMLP (https://github.com/DingXiaoH/RepMLP)

Features:
- Structural Re-parameterization (Locality Injection): Injects local inductive bias
  via parallel 1x1 and 3x3 depthwise/grouped convolutions during training.
- Zero-cost Inference: Mathematical fusion of Conv kernels and BatchNorm parameters into
  the FC weight matrix via `model.locality_injection()`, yielding a pure MLP at test time.
- Global Perceptron: Squeeze-and-excitation style channel attention.
- Hierarchical Stage Design: Flexible downsampling stages compatible with CIFAR-100 (32x32),
  Tiny-ImageNet (64x64), and ImageNet (224x224 / 256x256).
"""

from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from src.models.base import BaseClassifier, register_model


def conv_bn(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]],
    groups: int = 1,
) -> nn.Sequential:
    """Creates a Conv2d + BatchNorm2d block without bias."""
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


def conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]],
    groups: int = 1,
) -> nn.Sequential:
    """Creates a Conv2d + BatchNorm2d + ReLU block."""
    result = conv_bn(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        groups=groups,
    )
    result.add_module("relu", nn.ReLU(inplace=True))
    return result


def fuse_bn(conv_or_fc: nn.Module, bn: nn.BatchNorm2d) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fuses BatchNorm parameters into the preceding Convolution or FC layer weights and bias.
    """
    std = (bn.running_var + bn.eps).sqrt()
    t = bn.weight / std
    t = t.reshape(-1, 1, 1, 1)

    if len(t) == conv_or_fc.weight.size(0):
        return conv_or_fc.weight * t, bn.bias - bn.running_mean * bn.weight / std
    else:
        repeat_times = conv_or_fc.weight.size(0) // len(t)
        repeated = t.repeat_interleave(repeat_times, 0)
        return (
            conv_or_fc.weight * repeated,
            (bn.bias - bn.running_mean * bn.weight / std).repeat_interleave(repeat_times, 0),
        )


class GlobalPerceptron(nn.Module):
    """
    Global Perceptron: Global channel attention mechanism similar to Squeeze-and-Excitation.
    Captures global context across channels with lightweight 1x1 convolutions.
    """

    def __init__(self, input_channels: int, internal_neurons: int):
        super().__init__()
        internal_neurons = max(1, internal_neurons)
        self.fc1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=internal_neurons,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.fc2 = nn.Conv2d(
            in_channels=internal_neurons,
            out_channels=input_channels,
            kernel_size=1,
            stride=1,
            bias=True,
        )
        self.input_channels = input_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = torch.sigmoid(x)
        x = x.view(-1, self.input_channels, 1, 1)
        return x


class RepMLPBlock(nn.Module):
    """
    RepMLP Block with Locality Injection:
    - Training mode (`deploy=False`):
        Includes both FC3 (channel/spatial matrix multiplication) and parallel
        re-parameterizable convolution branches (`repconv1`, `repconv3`).
    - Deploy mode (`deploy=True`):
        Conv branches are mathematically merged into FC3 via `local_inject()`.
        At inference, only pure FC computation is performed.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        h: int,
        w: int,
        reparam_conv_k: Sequence[int] = (1, 3),
        globalperceptron_reduce: int = 4,
        num_sharesets: int = 1,
        deploy: bool = False,
    ):
        super().__init__()
        assert in_channels == out_channels, f"in_channels ({in_channels}) must equal out_channels ({out_channels})"
        assert in_channels % num_sharesets == 0, f"in_channels ({in_channels}) must be divisible by num_sharesets ({num_sharesets})"

        self.C = in_channels
        self.O = out_channels
        self.S = num_sharesets
        self.h = h
        self.w = w
        self.deploy = deploy

        self.gp = GlobalPerceptron(
            input_channels=in_channels,
            internal_neurons=in_channels // globalperceptron_reduce,
        )

        self.fc3 = nn.Conv2d(
            self.h * self.w * num_sharesets,
            self.h * self.w * num_sharesets,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=deploy,
            groups=num_sharesets,
        )

        if deploy:
            self.fc3_bn = nn.Identity()
        else:
            self.fc3_bn = nn.BatchNorm2d(num_sharesets)

        self.reparam_conv_k = reparam_conv_k
        if not deploy and reparam_conv_k is not None:
            for k in reparam_conv_k:
                conv_branch = conv_bn(
                    num_sharesets,
                    num_sharesets,
                    kernel_size=k,
                    stride=1,
                    padding=k // 2,
                    groups=num_sharesets,
                )
                self.add_module(f"repconv{k}", conv_branch)

    def partition(self, x: torch.Tensor, h_parts: int, w_parts: int) -> torch.Tensor:
        x = x.reshape(-1, self.C, h_parts, self.h, w_parts, self.w)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x

    def partition_affine(self, x: torch.Tensor, h_parts: int, w_parts: int) -> torch.Tensor:
        fc_inputs = x.reshape(-1, self.S * self.h * self.w, 1, 1)
        out = self.fc3(fc_inputs)
        out = out.reshape(-1, self.S, self.h, self.w)
        out = self.fc3_bn(out)
        out = out.reshape(-1, h_parts, w_parts, self.S, self.h, self.w)
        return out

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        global_vec = self.gp(inputs)

        origin_shape = inputs.size()
        h_parts = origin_shape[2] // self.h
        w_parts = origin_shape[3] // self.w

        partitions = self.partition(inputs, h_parts, w_parts)
        fc3_out = self.partition_affine(partitions, h_parts, w_parts)

        # Local Perceptron: Conv branches during training
        if self.reparam_conv_k is not None and not self.deploy:
            conv_inputs = partitions.reshape(-1, self.S, self.h, self.w)
            conv_out = 0
            for k in self.reparam_conv_k:
                conv_branch = getattr(self, f"repconv{k}")
                conv_out = conv_out + conv_branch(conv_inputs)
            conv_out = conv_out.reshape(-1, h_parts, w_parts, self.S, self.h, self.w)
            fc3_out = fc3_out + conv_out

        fc3_out = fc3_out.permute(0, 3, 1, 4, 2, 5)  # (N, O, h_parts, out_h, w_parts, out_w)
        out = fc3_out.reshape(*origin_shape)
        out = out * global_vec
        return out

    def get_equivalent_fc3(self) -> Tuple[torch.Tensor, torch.Tensor]:
        fc_weight, fc_bias = fuse_bn(self.fc3, self.fc3_bn)
        if self.reparam_conv_k is not None:
            largest_k = max(self.reparam_conv_k)
            largest_branch = getattr(self, f"repconv{largest_k}")
            total_kernel, total_bias = fuse_bn(largest_branch.conv, largest_branch.bn)
            for k in self.reparam_conv_k:
                if k != largest_k:
                    k_branch = getattr(self, f"repconv{k}")
                    kernel, bias = fuse_bn(k_branch.conv, k_branch.bn)
                    total_kernel = total_kernel + F.pad(kernel, [(largest_k - k) // 2] * 4)
                    total_bias = total_bias + bias
            rep_weight, rep_bias = self._convert_conv_to_fc(total_kernel, total_bias)
            final_fc3_weight = rep_weight.reshape_as(fc_weight) + fc_weight
            final_fc3_bias = rep_bias + fc_bias
        else:
            final_fc3_weight = fc_weight
            final_fc3_bias = fc_bias
        return final_fc3_weight, final_fc3_bias

    def local_inject(self):
        """Merges convolution branches into the FC3 layer and deletes the conv branches."""
        if self.deploy:
            return
        self.deploy = True
        fc3_weight, fc3_bias = self.get_equivalent_fc3()
        if self.reparam_conv_k is not None:
            for k in self.reparam_conv_k:
                if hasattr(self, f"repconv{k}"):
                    delattr(self, f"repconv{k}")
        delattr(self, "fc3")
        delattr(self, "fc3_bn")
        self.fc3 = nn.Conv2d(
            self.S * self.h * self.w,
            self.S * self.h * self.w,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=self.S,
        )
        self.fc3_bn = nn.Identity()
        self.fc3.weight.data = fc3_weight
        self.fc3.bias.data = fc3_bias

    def _convert_conv_to_fc(self, conv_kernel: torch.Tensor, conv_bias: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        I = (
            torch.eye(self.h * self.w, device=conv_kernel.device)
            .repeat(1, self.S)
            .reshape(self.h * self.w, self.S, self.h, self.w)
        )
        fc_k = F.conv2d(
            I,
            conv_kernel,
            padding=(conv_kernel.size(2) // 2, conv_kernel.size(3) // 2),
            groups=self.S,
        )
        fc_k = fc_k.reshape(self.h * self.w, self.S * self.h * self.w).t()
        fc_bias = conv_bias.repeat_interleave(self.h * self.w)
        return fc_k, fc_bias


class FFNBlock(nn.Module):
    """Feed-Forward Network block using 1x1 convs and GELU."""

    def __init__(self, in_channels: int, hidden_channels: Optional[int] = None, out_channels: Optional[int] = None):
        super().__init__()
        out_features = out_channels or in_channels
        hidden_features = hidden_channels or in_channels
        self.ffn_fc1 = conv_bn(in_channels, hidden_features, 1, 1, 0)
        self.ffn_fc2 = conv_bn(hidden_features, out_features, 1, 1, 0)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn_fc2(self.act(self.ffn_fc1(x)))


class RepMLPNetUnit(nn.Module):
    """A building unit combining RepMLPBlock, FFNBlock, Pre-BatchNorm, and skip connections."""

    def __init__(
        self,
        channels: int,
        h: int,
        w: int,
        reparam_conv_k: Sequence[int] = (1, 3),
        globalperceptron_reduce: int = 4,
        ffn_expand: int = 4,
        num_sharesets: int = 1,
        deploy: bool = False,
    ):
        super().__init__()
        self.repmlp_block = RepMLPBlock(
            in_channels=channels,
            out_channels=channels,
            h=h,
            w=w,
            reparam_conv_k=reparam_conv_k,
            globalperceptron_reduce=globalperceptron_reduce,
            num_sharesets=num_sharesets,
            deploy=deploy,
        )
        self.ffn_block = FFNBlock(channels, channels * ffn_expand)
        self.prebn1 = nn.BatchNorm2d(channels)
        self.prebn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.repmlp_block(self.prebn1(x))
        z = y + self.ffn_block(self.prebn2(y))
        return z


# Pre-defined architectural configurations from Ding et al. (CVPR 2022)
REPMLPNET_PRESETS = {
    "repmlpnet-t224": {
        "image_size": 224,
        "patch_size": 4,
        "channels": (64, 128, 256, 512),
        "hs": (56, 28, 14, 7),
        "ws": (56, 28, 14, 7),
        "num_blocks": (2, 2, 6, 2),
        "sharesets_nums": (1, 4, 16, 128),
    },
    "repmlpnet-b224": {
        "image_size": 224,
        "patch_size": 4,
        "channels": (96, 192, 384, 768),
        "hs": (56, 28, 14, 7),
        "ws": (56, 28, 14, 7),
        "num_blocks": (2, 2, 12, 2),
        "sharesets_nums": (1, 4, 32, 128),
    },
    "repmlpnet-t256": {
        "image_size": 256,
        "patch_size": 4,
        "channels": (64, 128, 256, 512),
        "hs": (64, 32, 16, 8),
        "ws": (64, 32, 16, 8),
        "num_blocks": (2, 2, 6, 2),
        "sharesets_nums": (1, 4, 16, 128),
    },
    "repmlpnet-b256": {
        "image_size": 256,
        "patch_size": 4,
        "channels": (96, 192, 384, 768),
        "hs": (64, 32, 16, 8),
        "ws": (64, 32, 16, 8),
        "num_blocks": (2, 2, 12, 2),
        "sharesets_nums": (1, 4, 32, 128),
    },
    "repmlpnet-d256": {
        "image_size": 256,
        "patch_size": 4,
        "channels": (80, 160, 320, 640),
        "hs": (64, 32, 16, 8),
        "ws": (64, 32, 16, 8),
        "num_blocks": (2, 2, 18, 2),
        "sharesets_nums": (1, 4, 16, 128),
    },
    "repmlpnet-l256": {
        "image_size": 256,
        "patch_size": 4,
        "channels": (96, 192, 384, 768),
        "hs": (64, 32, 16, 8),
        "ws": (64, 32, 16, 8),
        "num_blocks": (2, 2, 18, 2),
        "sharesets_nums": (1, 4, 32, 256),
    },
}


@register_model("rep_mlp")
@register_model("repmlp")
@register_model("repmlpnet")
class RepMLP(BaseClassifier):
    """
    RepMLP / RepMLPNet Vision Classifier.

    Compatible with:
    - Predefined architectures: 'RepMLPNet-T224', 'RepMLPNet-B224', etc.
    - Small datasets: CIFAR-100 (image_size=32), Tiny-ImageNet (image_size=64).
    - Custom hierarchical stage configurations.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 2,
        in_channels: int = 3,
        num_classes: int = 100,
        channels: Optional[Sequence[int]] = None,
        num_blocks: Optional[Sequence[int]] = None,
        sharesets_nums: Optional[Sequence[int]] = None,
        hs: Optional[Sequence[int]] = None,
        ws: Optional[Sequence[int]] = None,
        reparam_conv_k: Sequence[int] = (1, 3),
        globalperceptron_reduce: int = 4,
        ffn_expand: int = 4,
        use_checkpoint: bool = False,
        deploy: bool = False,
        arch: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()

        # 1. Resolve presets if arch is explicitly given
        if arch is not None:
            arch_key = arch.lower().strip()
            if arch_key in REPMLPNET_PRESETS:
                preset = REPMLPNET_PRESETS[arch_key]
                image_size = preset["image_size"]
                patch_size = preset["patch_size"]
                channels = preset["channels"]
                hs = preset["hs"]
                ws = preset["ws"]
                num_blocks = preset["num_blocks"]
                sharesets_nums = preset["sharesets_nums"]

        # 2. Derive sensible defaults based on image_size if channels/blocks are not provided
        if channels is None or num_blocks is None:
            if image_size <= 32:
                # CIFAR-100 defaults
                patch_size = kwargs.get("patch_size", 2)
                channels = (64, 128, 256)
                num_blocks = (2, 4, 2)
                sharesets_nums = (1, 4, 16)
            elif image_size <= 64:
                # Tiny-ImageNet defaults
                patch_size = kwargs.get("patch_size", 4)
                channels = (64, 128, 256)
                num_blocks = (2, 4, 2)
                sharesets_nums = (1, 4, 16)
            else:
                # Default 224x224 (T224 preset)
                patch_size = kwargs.get("patch_size", 4)
                channels = (64, 128, 256, 512)
                num_blocks = (2, 2, 6, 2)
                sharesets_nums = (1, 4, 16, 128)

        # 3. Derive spatial stage sizes (hs, ws) if not provided
        num_stages = len(num_blocks)
        if hs is None or ws is None:
            curr_h = image_size // patch_size
            curr_w = image_size // patch_size
            hs_list, ws_list = [], []
            for _ in range(num_stages):
                hs_list.append(curr_h)
                ws_list.append(curr_w)
                curr_h = max(1, curr_h // 2)
                curr_w = max(1, curr_w // 2)
            hs = tuple(hs_list)
            ws = tuple(ws_list)

        if sharesets_nums is None:
            sharesets_nums = tuple(max(1, min(ch, 2 ** (2 * i))) for i, ch in enumerate(channels))

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.deploy = deploy

        # 4. Patch Embedding Layer
        self.conv_embedding = conv_bn_relu(
            in_channels=in_channels,
            out_channels=channels[0],
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # 5. Build Hierarchical Stages
        stages = []
        embeds = []
        for stage_idx in range(num_stages):
            stage_blocks = [
                RepMLPNetUnit(
                    channels=channels[stage_idx],
                    h=hs[stage_idx],
                    w=ws[stage_idx],
                    reparam_conv_k=reparam_conv_k,
                    globalperceptron_reduce=globalperceptron_reduce,
                    ffn_expand=ffn_expand,
                    num_sharesets=sharesets_nums[stage_idx],
                    deploy=deploy,
                )
                for _ in range(num_blocks[stage_idx])
            ]
            stages.append(nn.ModuleList(stage_blocks))

            # Downsampling transition between stages
            if stage_idx < num_stages - 1:
                embeds.append(
                    conv_bn_relu(
                        in_channels=channels[stage_idx],
                        out_channels=channels[stage_idx + 1],
                        kernel_size=2,
                        stride=2,
                        padding=0,
                    )
                )

        self.stages = nn.ModuleList(stages)
        self.embeds = nn.ModuleList(embeds)
        self.head_norm = nn.BatchNorm2d(channels[-1])
        self.head = nn.Linear(channels[-1], num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts features before classification head."""
        x = self.conv_embedding(x)
        for i, stage in enumerate(self.stages):
            for block in stage:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(block, x)
                else:
                    x = block(x)
            if i < len(self.stages) - 1:
                embed = self.embeds[i]
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(embed, x)
                else:
                    x = embed(x)
        x = self.head_norm(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.forward_features(x)
        logits = self.head(feat)
        return logits

    def locality_injection(self):
        """
        Conducts Structural Re-parameterization across all RepMLP blocks in the model.
        Fuses all parallel conv branches into the FC3 layers and removes conv weights.
        Switches the model to deploy mode.
        """
        for m in self.modules():
            if hasattr(m, "local_inject") and callable(m.local_inject):
                m.local_inject()
        self.deploy = True
