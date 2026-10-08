"""
RepMLP-DWS: RepMLP with Depthwise-Separable Local Perceptron.

Extends the original RepMLP (Ding et al., CVPR 2022) by adding a parallel
Depthwise-Separable Convolution branch (DW 3×3 → PW 1×1, no activation
between) to the Local Perceptron. This enables cross-shareset channel mixing
during training while preserving full re-parameterizability into a pure MLP
at inference time.

Key differences from rep_mlp.py:
  - RepMLPBlockDWS adds a `dw_pw` branch: DW 3×3 (groups=S) → BN → PW 1×1 (groups=1) → BN
  - No activation between DW and PW → both are linear → can be fused into
    a single FC matrix and merged into FC3 via `local_inject()`.
  - The `_convert_dwpw_to_fc()` method handles the 2-step fusion:
    Step 1: Fuse BN into DW and PW separately
    Step 2: Convert DW to FC matrix
    Step 3: Multiply PW (as FC) × DW (as FC) → equivalent FC
    Step 4: Add equivalent FC into FC3

Usage:
    from src.models.rep_mlp_dws import RepMLPDWS

    # Training
    model = RepMLPDWS(image_size=32, num_classes=100)
    out = model(x)  # uses DW, PW, and FC3 branches

    # Deploy (fuse all conv branches into FC3 → pure MLP)
    model.locality_injection()
    out = model(x)  # only FC3, mathematically identical
"""

from typing import List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from src.models.base import BaseClassifier, register_model


# ---------------------------------------------------------------------------
# Utility functions (same as rep_mlp.py)
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


# ---------------------------------------------------------------------------
# DW-Separable branch module
# ---------------------------------------------------------------------------

class DWSeparableBranch(nn.Module):
    """
    Depthwise-Separable branch: DW k×k (groups=S) + BN → PW 1×1 (groups=1) + BN.
    No activation between DW and PW to preserve re-parameterizability.

    Args:
        num_sharesets: Number of sharesets (= channel dim of the partitioned input).
        kernel_size: Spatial kernel size for the depthwise conv.
    """

    def __init__(self, num_sharesets: int, kernel_size: int):
        super().__init__()
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
            groups=1,  # ← cross-channel mixing between sharesets
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # DW → BN → PW → BN  (no activation between — both are linear)
        return self.pw(self.dw(x))


# ---------------------------------------------------------------------------
# Modified RepMLP Block with DW-Separable branch
# ---------------------------------------------------------------------------

class RepMLPBlockDWS(nn.Module):
    """
    RepMLP Block with Depthwise-Separable Locality Injection.

    Compared to the original RepMLPBlock, this adds a parallel DW-Separable
    branch (`dw_pw_branch`) that provides cross-shareset channel mixing
    during training. At deploy time, all branches (including DW-Sep) are
    mathematically fused into FC3.

    Training branches:
      - FC3 + BN (channel/spatial matrix multiplication)
      - repconv1: DW 1×1 + BN (original)
      - repconv3: DW 3×3 + BN (original)
      - dw_pw_branch: DW 3×3 + BN → PW 1×1 + BN (NEW — cross-channel mixing)

    Deploy mode:
      - All branches fused into FC3 → pure MLP inference.
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
        use_dw_sep: bool = True,
        dw_sep_kernel: int = 3,
    ):
        super().__init__()
        assert in_channels == out_channels, (
            f"in_channels ({in_channels}) must equal out_channels ({out_channels})"
        )
        assert in_channels % num_sharesets == 0, (
            f"in_channels ({in_channels}) must be divisible by num_sharesets ({num_sharesets})"
        )

        self.C = in_channels
        self.O = out_channels
        self.S = num_sharesets
        self.h = h
        self.w = w
        self.deploy = deploy
        self.use_dw_sep = use_dw_sep

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

        # Original DW conv branches (repconv1, repconv3)
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

        # NEW: Depthwise-Separable branch (DW → PW, no activation)
        if not deploy and use_dw_sep and num_sharesets > 1:
            # Only add DW-Sep when S > 1, because with S=1 the PW 1×1 is trivially a scalar
            self.dw_pw_branch = DWSeparableBranch(
                num_sharesets=num_sharesets,
                kernel_size=dw_sep_kernel,
            )
        else:
            self.dw_pw_branch = None

    def partition(self, x: torch.Tensor, h_parts: int, w_parts: int) -> torch.Tensor:
        x = x.reshape(-1, self.C, h_parts, self.h, w_parts, self.w)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x

    def partition_affine(self, x: torch.Tensor, h_parts: int, w_parts: int) -> torch.Tensor:
        fc_inputs = x.reshape(-1, self.S * self.h * self.w, 1, 1)
        if isinstance(self.fc3, nn.Linear):
            # Deploy mode with DW-Sep: FC3 is nn.Linear (full matrix)
            out = self.fc3(fc_inputs.squeeze(-1).squeeze(-1))
            out = out.reshape(-1, self.S, self.h, self.w)
        else:
            # Training mode or deploy without DW-Sep: FC3 is Conv2d
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
        if not self.deploy:
            conv_inputs = partitions.reshape(-1, self.S, self.h, self.w)
            conv_out = 0

            # Original DW branches
            if self.reparam_conv_k is not None:
                for k in self.reparam_conv_k:
                    conv_branch = getattr(self, f"repconv{k}")
                    conv_out = conv_out + conv_branch(conv_inputs)

            # NEW: DW-Separable branch (cross-channel mixing)
            if self.dw_pw_branch is not None:
                conv_out = conv_out + self.dw_pw_branch(conv_inputs)

            conv_out = conv_out.reshape(-1, h_parts, w_parts, self.S, self.h, self.w)
            fc3_out = fc3_out + conv_out

        fc3_out = fc3_out.permute(0, 3, 1, 4, 2, 5)  # (N, O, h_parts, out_h, w_parts, out_w)
        out = fc3_out.reshape(*origin_shape)
        out = out * global_vec
        return out

    def get_equivalent_fc3(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the equivalent FC3 weight and bias after fusing all branches.

        When DW-Sep branch is present (cross-channel PW), the result is a FULL
        (S*h*w, S*h*w) matrix (not block-diagonal), so FC3 must be deployed
        with groups=1.

        When DW-Sep is absent, the result stays block-diagonal and FC3 can
        remain grouped (groups=S), same as the original RepMLP.
        """
        hw = self.h * self.w
        has_dw_sep = self.dw_pw_branch is not None

        # --- Convert FC3 (grouped) + BN to full FC matrix ---
        fc_weight_grouped, fc_bias = fuse_bn(self.fc3, self.fc3_bn)
        # fc_weight_grouped shape: (S*hw, hw, 1, 1) when groups=S
        # Expand to full (S*hw, S*hw) matrix (block-diagonal)
        if has_dw_sep:
            full_fc_weight = torch.zeros(
                self.S * hw, self.S * hw,
                device=fc_weight_grouped.device, dtype=fc_weight_grouped.dtype,
            )
            for g in range(self.S):
                # Each group: out [g*hw : (g+1)*hw], in [g*hw : (g+1)*hw]
                full_fc_weight[g * hw:(g + 1) * hw, g * hw:(g + 1) * hw] = (
                    fc_weight_grouped[g * hw:(g + 1) * hw, :, 0, 0]
                )
        else:
            full_fc_weight = fc_weight_grouped

        # --- Fuse original DW conv branches ---
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
            # rep_weight is (S*hw, hw) — block-diagonal contribution
            if has_dw_sep:
                # Expand to full matrix
                for g in range(self.S):
                    full_fc_weight[g * hw:(g + 1) * hw, g * hw:(g + 1) * hw] += (
                        rep_weight[g * hw:(g + 1) * hw, :]
                    )
                fc_bias = rep_bias + fc_bias
            else:
                full_fc_weight = rep_weight.reshape_as(full_fc_weight) + full_fc_weight
                fc_bias = rep_bias + fc_bias

        # --- NEW: Fuse DW-Separable branch (full matrix) ---
        if has_dw_sep:
            dws_weight, dws_bias = self._convert_dwpw_to_fc(self.dw_pw_branch)
            # dws_weight is (S*hw, S*hw) — full matrix (cross-channel)
            full_fc_weight = dws_weight + full_fc_weight
            fc_bias = dws_bias + fc_bias

        return full_fc_weight, fc_bias

    def _convert_dwpw_to_fc(
        self, branch: DWSeparableBranch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts a DW-Separable branch (DW+BN → PW+BN) into an equivalent FC matrix.

        Since DW conv is grouped, _convert_conv_to_fc returns (S*hw, hw) — a
        block-diagonal matrix. We first expand it to full (S*hw, S*hw), then
        multiply with the PW FC matrix (also S*hw × S*hw via Kronecker product).
        """
        hw = self.h * self.w

        # Step 1: Fuse BN into DW conv
        dw_kernel_fused, dw_bias_fused = fuse_bn(branch.dw.conv, branch.dw.bn)

        # Step 2: Fuse BN into PW conv
        pw_kernel_fused, pw_bias_fused = fuse_bn(branch.pw.conv, branch.pw.bn)

        # Step 3: Convert fused DW conv to FC
        # Returns (S*hw, hw) — grouped/block-diagonal
        dw_fc_grouped, dw_fc_bias = self._convert_conv_to_fc(dw_kernel_fused, dw_bias_fused)

        # Expand DW FC to full (S*hw, S*hw) block-diagonal matrix
        dw_fc_full = torch.zeros(
            self.S * hw, self.S * hw,
            device=dw_fc_grouped.device, dtype=dw_fc_grouped.dtype,
        )
        for g in range(self.S):
            dw_fc_full[g * hw:(g + 1) * hw, g * hw:(g + 1) * hw] = (
                dw_fc_grouped[g * hw:(g + 1) * hw, :]
            )

        # Step 4: Convert PW 1×1 conv to FC matrix
        # PW is (S, S, 1, 1) → Kronecker product: pw_matrix ⊗ I_{hw}
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

        # Step 5: Compose — equiv = PW @ DW
        equiv_weight = pw_fc_weight @ dw_fc_full
        # Step 6: Compose biases — b_equiv = PW @ b_dw + b_pw
        equiv_bias = pw_fc_weight @ dw_fc_bias + pw_fc_bias

        return equiv_weight, equiv_bias

    def local_inject(self):
        """Merges all branches (including DW-Sep) into FC3 and deletes them."""
        if self.deploy:
            return
        self.deploy = True
        has_dw_sep = self.dw_pw_branch is not None
        fc3_weight, fc3_bias = self.get_equivalent_fc3()

        # Delete original conv branches
        if self.reparam_conv_k is not None:
            for k in self.reparam_conv_k:
                if hasattr(self, f"repconv{k}"):
                    delattr(self, f"repconv{k}")

        # Delete DW-Sep branch
        if self.dw_pw_branch is not None:
            delattr(self, "dw_pw_branch")
            self.dw_pw_branch = None

        delattr(self, "fc3")
        delattr(self, "fc3_bn")

        if has_dw_sep:
            # DW-Sep introduces cross-channel mixing → FC3 must be ungrouped
            self.fc3 = nn.Linear(self.S * self.h * self.w, self.S * self.h * self.w, bias=True)
            self.fc3.weight.data = fc3_weight
            self.fc3.bias.data = fc3_bias
        else:
            # No DW-Sep → block-diagonal, keep grouped conv like original
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
        """Converts a grouped depthwise conv kernel to an equivalent FC weight matrix."""
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
# FFN Block (same as rep_mlp.py)
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
# RepMLPNet Unit with DWS
# ---------------------------------------------------------------------------

class RepMLPNetUnitDWS(nn.Module):
    """A building unit combining RepMLPBlockDWS, FFNBlock, Pre-BatchNorm, and skip connections."""

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
        use_dw_sep: bool = True,
        dw_sep_kernel: int = 3,
    ):
        super().__init__()
        self.repmlp_block = RepMLPBlockDWS(
            in_channels=channels,
            out_channels=channels,
            h=h,
            w=w,
            reparam_conv_k=reparam_conv_k,
            globalperceptron_reduce=globalperceptron_reduce,
            num_sharesets=num_sharesets,
            deploy=deploy,
            use_dw_sep=use_dw_sep,
            dw_sep_kernel=dw_sep_kernel,
        )
        self.ffn_block = FFNBlock(channels, channels * ffn_expand)
        self.prebn1 = nn.BatchNorm2d(channels)
        self.prebn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x + self.repmlp_block(self.prebn1(x))
        z = y + self.ffn_block(self.prebn2(y))
        return z


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@register_model("rep_mlp_dws")
@register_model("repmlp_dws")
class RepMLPDWS(BaseClassifier):
    """
    RepMLP-DWS: RepMLP with Depthwise-Separable Local Perceptron.

    Adds a parallel DW-Separable branch to the Local Perceptron for
    cross-shareset channel mixing during training. Fully re-parameterizable
    into a pure MLP at inference time.

    Compatible with:
    - CIFAR-100 (image_size=32), Tiny-ImageNet (image_size=64)
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
        use_dw_sep: bool = True,
        dw_sep_kernel: int = 3,
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
                RepMLPNetUnitDWS(
                    channels=channels[stage_idx],
                    h=hs[stage_idx],
                    w=ws[stage_idx],
                    reparam_conv_k=reparam_conv_k,
                    globalperceptron_reduce=globalperceptron_reduce,
                    ffn_expand=ffn_expand,
                    num_sharesets=sharesets_nums[stage_idx],
                    deploy=deploy,
                    use_dw_sep=use_dw_sep,
                    dw_sep_kernel=dw_sep_kernel,
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
        Conducts Structural Re-parameterization across all RepMLP-DWS blocks.
        Fuses all parallel conv branches (including DW-Sep) into FC3 layers.
        Switches the model to deploy mode → pure MLP inference.
        """
        for m in self.modules():
            if hasattr(m, "local_inject") and callable(m.local_inject):
                m.local_inject()
        self.deploy = True
