"""
Unit tests for deterministic seed control.
"""

import random
import numpy as np
import pytest
import torch
from src.utils.seed import get_generator, set_seed


def test_seed_determinism_numpy():
    set_seed(42, deterministic=True)
    val1 = np.random.rand(5)

    set_seed(42, deterministic=True)
    val2 = np.random.rand(5)

    np.testing.assert_array_equal(val1, val2)


def test_seed_determinism_python_random():
    set_seed(42, deterministic=True)
    val1 = [random.random() for _ in range(5)]

    set_seed(42, deterministic=True)
    val2 = [random.random() for _ in range(5)]

    assert val1 == val2


def test_seed_determinism_torch():
    set_seed(42, deterministic=True)
    t1 = torch.randn(3, 3)

    set_seed(42, deterministic=True)
    t2 = torch.randn(3, 3)

    torch.testing.assert_close(t1, t2)


def test_generator_determinism():
    g1 = get_generator(123)
    g2 = get_generator(123)

    t1 = torch.rand(4, generator=g1)
    t2 = torch.rand(4, generator=g2)

    torch.testing.assert_close(t1, t2)
