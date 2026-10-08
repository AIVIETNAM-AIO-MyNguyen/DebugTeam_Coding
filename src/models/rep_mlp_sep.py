"""
RepMLP-Sep: RepMLP with Per-Branch Depthwise-Separable Local Perceptron (Cách 2).

In this architecture, EVERY convolution branch k in `reparam_conv_k` (e.g. k=1, k=3)
is directly constructed as a Depthwise-Separable Convolution block:
    repconv{k} = DW (k×k, groups=S) + BN  -->  PW (1×1, groups=1) + BN

Each scale k gets its own dedicated cross-channel Pointwise mixing matrix.
There is NO activation function between DW and PW to preserve strict linearity,
allowing all branches to be analytically re-parameterized into a single FC3
linear transformation during `locality_injection()`.

At inference (deploy=True):
- All DW and PW branches are fused together with FC3 into a pure Linear layer.
- Zero extra computational cost or latency at test time.
"""

from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from src.models.base import BaseClassifier, register_model


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

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
    Global Perceptron: Global channel attention mechanism (Squeeze-and-Excitation).
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


# ---------------------------------------------------------------------------
# Depthwise-Separable Branch Module (Cách 2)
# ---------------------------------------------------------------------------

class SeparableConvBranch(nn.Module):
    """
    Individual Depthwise-Separable Conv branch for a specific kernel size k:
        DW (k×k, groups=S) + BN --> PW (1×1, groups=1) + BN

    No activation between DW and PW to preserve strict linearity for Re-parameterization.
    """

    def __init__(self, num_sharesets: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.num_sharesets = num_sharesets

        self.dw = conv_bn(
            in_channels=num_sharesets,
            out_channels=num_sharesets,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=num_sharesets,
        )
        self.pw = conv_bn(
            in_channels=num_sharesets,
            out_channels=num_sharesets,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,  # cross-shareset channel mixing
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


# ---------------------------------------------------------------------------
# RepMLP Block with Per-Branch Separable Conv
# ---------------------------------------------------------------------------

class RepMLPBlockSep(nn.Module):
    """
    RepMLP Block with Per-Branch Separable Locality Injection.

    Each branch k in `reparam_conv_k` is an independent SeparableConvBranch (DW_k -> PW_k).
    During training:
      - FC3 (channel/spatial affine)
      - repconv{k}: SeparableConvBranch for each k in reparam_conv_k (default: 1, 3)
    During deploy:
      - All Separable branches are converted into equivalent full matrices and fused into FC3.
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
        use_separable: bool = True,
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
        self.use_separable = use_separable

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
                if use_separable:
                    # Cách 2: Every branch is a Depthwise-Separable block (DW_k -> PW_k)
                    branch = SeparableConvBranch(num_sharesets=num_sharesets, kernel_size=k)
                else:
                    # Fallback to standard DW-only
                    branch = conv_bn(
                        num_sharesets,
                        num_sharesets,
                        kernel_size=k,
                        stride=1,
                        padding=k // 2,
                        groups=num_sharesets,
                    )
                self.add_module(f"repconv{k}", branch)

    def partition(self, x: torch.Tensor, h_parts: int, w_parts: int) -> torch.Tensor:
        x = x.reshape(-1, self.C, h_parts, self.h, w_parts, self.w)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x

    def partition_affine(self, x: torch.Tensor, h_parts: int, w_parts: int) -> torch.Tensor:
        fc_inputs = x.reshape(-1, self.S * self.h * self.w, 1, 1)
        if isinstance(self.fc3, nn.Linear):
            out = self.fc3(fc_inputs.squeeze(-1).squeeze(-1))
            out = out.reshape(-1, self.S, self.h, self.w)
        else:
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
        if not self.deploy and self.reparam_conv_k is not None:
            conv_inputs = partitions.reshape(-1, self.S, self.h, self.w)
            conv_out = 0
            for k in self.reparam_conv_k:
                branch = getattr(self, f"repconv{k}")
                conv_out = conv_out + branch(conv_inputs)
            conv_out = conv_out.reshape(-1, h_parts, w_parts, self.S, self.h, self.w)
            fc3_out = fc3_out + conv_out

        fc3_out = fc3_out.permute(0, 3, 1, 4, 2, 5)
        out = fc3_out.reshape(*origin_shape)
        out = out * global_vec
        return out

    def _convert_sep_branch_to_fc(self, branch: SeparableConvBranch) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a SeparableConvBranch (DW_k + BN -> PW_k + BN) to an equivalent full FC matrix.
        Output weight shape: (S*hw, S*hw), bias shape: (S*hw,)
        """
        hw = self.h * self.w

        # 1. Fuse BN into DW conv
        dw_kernel_fused, dw_bias_fused = fuse_bn(branch.dw.conv, branch.dw.bn)

        # 2. Fuse BN into PW conv
        pw_kernel_fused, pw_bias_fused = fuse_bn(branch.pw.conv, branch.pw.bn)

        # 3. Convert DW to block-diagonal FC
        dw_fc_grouped, dw_fc_bias = self._convert_conv_to_fc(dw_kernel_fused, dw_bias_fused)

        # Expand DW to full (S*hw, S*hw) block-diagonal matrix
        dw_fc_full = torch.zeros(
            self.S * hw, self.S * hw,
            device=dw_fc_grouped.device, dtype=dw_fc_grouped.dtype,
        )
        for g in range(self.S):
            dw_fc_full[g * hw:(g + 1) * hw, g * hw:(g + 1) * hw] = (
                dw_fc_grouped[g * hw:(g + 1) * hw, :]
            )

        # 4. Convert PW 1×1 to Kronecker product FC: W_pw ⊗ I_{hw}
        pw_matrix = pw_kernel_fused.squeeze(-1).squeeze(-1)  # (S, S)
        pw_fc_weight = torch.zeros(
            self.S * hw, self.S * hw,
            device=pw_matrix.device, dtype=pw_matrix.dtype,
        )
        eye_hw = torch.eye(hw, device=pw_matrix.device, dtype=pw_matrix.dtype)
        for i in range(self.S):
            for j in range(self.S):
                pw_fc_weight[i * hw:(i + 1) * hw, j * hw:(j + 1) * hw] = (
                    pw_matrix[i, j] * eye_hw
                )
        pw_fc_bias = pw_bias_fused.repeat_interleave(hw)

        # 5. Multiply: W_equiv = PW_fc @ DW_fc, b_equiv = PW_fc @ b_dw + b_pw
        equiv_weight = pw_fc_weight @ dw_fc_full
        equiv_bias = pw_fc_weight @ dw_fc_bias + pw_fc_bias

        return equiv_weight, equiv_bias

    def get_equivalent_fc3(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Analytically merges all Separable branches into FC3.
        When use_separable=True, cross-channel connections exist, so FC3 becomes a full Linear layer.
        """
        hw = self.h * self.w

        # Fuse FC3 + BN
        fc_weight_grouped, fc_bias = fuse_bn(self.fc3, self.fc3_bn)

        if self.use_separable:
            # Expand grouped FC3 to full (S*hw, S*hw) matrix
            full_fc_weight = torch.zeros(
                self.S * hw, self.S * hw,
                device=fc_weight_grouped.device, dtype=fc_weight_grouped.dtype,
            )
            for g in range(self.S):
                full_fc_weight[g * hw:(g + 1) * hw, g * hw:(g + 1) * hw] = (
                    fc_weight_grouped[g * hw:(g + 1) * hw, :, 0, 0]
                )
            full_fc_bias = fc_bias

            # Convert and sum each Separable branch
            if self.reparam_conv_k is not None:
                for k in self.reparam_conv_k:
                    branch = getattr(self, f"repconv{k}")
                    b_weight, b_bias = self._convert_sep_branch_to_fc(branch)
                    full_fc_weight = full_fc_weight + b_weight
                    full_fc_bias = full_fc_bias + b_bias

            return full_fc_weight, full_fc_bias
        else:
            # Fallback for standard DW-only
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
                final_fc3_weight = rep_weight.reshape_as(fc_weight_grouped) + fc_weight_grouped
                final_fc3_bias = rep_bias + fc_bias
            else:
                final_fc3_weight = fc_weight_grouped
                final_fc3_bias = fc_bias
            return final_fc3_weight, final_fc3_bias

    def local_inject(self):
        """
        Fuses all Separable branches into FC3 and deletes them for zero-cost inference.
        """
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

        if self.use_separable:
            # Full Linear layer for cross-channel mixing
            self.fc3 = nn.Linear(self.S * self.h * self.w, self.S * self.h * self.w, bias=True)
            self.fc3.weight.data = fc3_weight
            self.fc3.bias.data = fc3_bias
        else:
            self.fc3 = nn.Conv2d(
                self.S * self.h * self.w,
                self.S * self.h * self.w,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
                groups=self.S,
            )
            self.fc3.weight.data = fc3_weight
            self.fc3.bias.data = fc3_bias

        self.fc3_bn = nn.Identity()

    def _convert_conv_to_fc(
        self, conv_kernel: torch.Tensor, conv_bias: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Converts grouped DW conv kernel to an equivalent FC matrix."""
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


# ---------------------------------------------------------------------------
# FFN Block
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# RepMLPNet Unit with Per-Branch Separable Conv
# ---------------------------------------------------------------------------

class RepMLPNetUnitSep(nn.Module):
    """A building unit combining RepMLPBlockSep, FFNBlock, Pre-BatchNorm, and skip connections."""

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
        use_separable: bool = True,
    ):
        super().__init__()
        self.repmlp_block = RepMLPBlockSep(
            in_channels=channels,
            out_channels=channels,
            h=h,
            w=w,
            reparam_conv_k=reparam_conv_k,
            globalperceptron_reduce=globalperceptron_reduce,
            num_sharesets=num_sharesets,
            deploy=deploy,
            use_separable=use_separable,
        )
        self.ffn_block = FFNBlock(channels, channels * ffn_expand)
        self.prebn1 = nn.BatchNorm2d(channels)
        self.prebn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.repmlp_block(self.prebn1(x))
        z = y + self.ffn_block(self.prebn2(y))
        return z


# ---------------------------------------------------------------------------
# Top-level Classifier: RepMLPSep
# ---------------------------------------------------------------------------

@register_model("rep_mlp_sep")
@register_model("repmlp_sep")
@register_model("rep_mlp_ds2")
class RepMLPSep(BaseClassifier):
    """
    RepMLP-Sep: Vision Classifier with Per-Branch Depthwise-Separable Local Perceptron.

    Features:
    - Each branch k in reparam_conv_k is an independent Depthwise-Separable block:
        repconv{k} = DW (k×k) -> PW (1×1)
    - Full mathematical fusion into a single Linear layer at deploy time.
    - Zero inference overhead.
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
        use_separable: bool = True,
        **kwargs,
    ):
        super().__init__()

        # Derive sensible defaults based on image_size
        if channels is None or num_blocks is None:
            if image_size <= 32:
                patch_size = kwargs.get("patch_size", 2)
                channels = (64, 128, 256)
                num_blocks = (2, 4, 2)
                sharesets_nums = (1, 4, 16)
            elif image_size <= 64:
                patch_size = kwargs.get("patch_size", 4)
                channels = (64, 128, 256)
                num_blocks = (2, 4, 2)
                sharesets_nums = (1, 4, 16)
            else:
                patch_size = kwargs.get("patch_size", 4)
                channels = (64, 128, 256, 512)
                num_blocks = (2, 2, 6, 2)
                sharesets_nums = (1, 4, 16, 128)

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

        # Patch Embedding Layer
        self.conv_embedding = conv_bn_relu(
            in_channels=in_channels,
            out_channels=channels[0],
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        # Build Hierarchical Stages
        stages = []
        embeds = []
        for stage_idx in range(num_stages):
            stage_blocks = [
                RepMLPNetUnitSep(
                    channels=channels[stage_idx],
                    h=hs[stage_idx],
                    w=ws[stage_idx],
                    reparam_conv_k=reparam_conv_k,
                    globalperceptron_reduce=globalperceptron_reduce,
                    ffn_expand=ffn_expand,
                    num_sharesets=sharesets_nums[stage_idx],
                    deploy=deploy,
                    use_separable=use_separable,
                )
                for _ in range(num_blocks[stage_idx])
            ]
            stages.append(nn.ModuleList(stage_blocks))

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
        Conducts Structural Re-parameterization across all RepMLP-Sep blocks.
        Fuses all Separable branches into FC3 layers and switches to deploy mode.
        """
        for m in self.modules():
            if hasattr(m, "local_inject") and callable(m.local_inject):
                m.local_inject()
        self.deploy = True
