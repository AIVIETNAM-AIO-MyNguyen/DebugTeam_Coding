"""
Common Neural Network Layers and Utilities.

Includes:
- DropPath (Stochastic Depth) for vision transformers and modern MLPs
"""

import torch
import torch.nn as nn


def drop_path(
    x: torch.Tensor,
    drop_prob: float = 0.0,
    training: bool = False,
    scale_by_keep: bool = True,
) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (applied in main path of residual blocks).

    Args:
        x: Input tensor.
        drop_prob: Probability of dropping path.
        training: Whether currently in training mode.
        scale_by_keep: Whether to scale surviving paths by (1 / (1 - drop_prob)).
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    # Work with any tensor dimension, e.g. (B, N, C) or (B, C, H, W)
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """
    DropPath (Stochastic Depth) module.
    """

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = float(drop_prob)
        self.scale_by_keep = scale_by_keep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self) -> str:
        return f"drop_prob={self.drop_prob:.4f}"
