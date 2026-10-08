"""
Unit tests for common neural network layers (DropPath / Stochastic Depth).
"""

import torch
import pytest
from src.models.layers import DropPath, drop_path


def test_drop_path_identity_when_zero_or_eval():
    x = torch.randn(8, 16, 64)

    # drop_prob = 0.0 in training
    dp = DropPath(0.0)
    dp.train()
    assert torch.equal(dp(x), x)

    # drop_prob > 0.0 in eval
    dp = DropPath(0.5)
    dp.eval()
    assert torch.equal(dp(x), x)


def test_drop_path_active_in_train():
    torch.manual_seed(42)
    x = torch.ones(1000, 10, 10)
    dp = DropPath(0.2)
    dp.train()
    out = dp(x)

    # Some batch items should be dropped to zero completely
    batch_zeros = (out.sum(dim=(1, 2)) == 0).float()
    zero_ratio = batch_zeros.mean().item()
    # Expect roughly 20% drops
    assert 0.15 < zero_ratio < 0.25

    # Surviving items should be scaled by 1 / (1 - 0.2) = 1.25
    surviving = out[batch_zeros == 0]
    assert torch.allclose(surviving, torch.tensor(1.25))

    # Mean over large batch should approximately equal input mean (1.0)
    assert abs(out.mean().item() - 1.0) < 0.05


def test_drop_path_backward():
    x = torch.randn(4, 16, 32, requires_grad=True)
    dp = DropPath(0.3)
    dp.train()
    out = dp(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert x.grad.shape == x.shape
