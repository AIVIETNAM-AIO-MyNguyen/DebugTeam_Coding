"""
Unit tests for MLP architectures and registry.
"""

import pytest
import torch
from src.config.parser import ConfigDict
from src.models import build_model, list_models
from src.models.mlp_mixer import MLPMixer


def test_registry_contains_mlp_mixer():
    models = list_models()
    assert "mlp_mixer" in models


def test_mlp_mixer_cifar_forward():
    model = MLPMixer(
        image_size=32,
        patch_size=4,
        num_classes=100,
        hidden_dim=128,
        num_blocks=4,
        tokens_mlp_dim=64,
        channels_mlp_dim=256,
    )
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)


def test_mlp_mixer_tiny_imagenet_forward():
    model = MLPMixer(
        image_size=64,
        patch_size=8,
        num_classes=200,
        hidden_dim=128,
        num_blocks=4,
        tokens_mlp_dim=64,
        channels_mlp_dim=256,
    )
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 200)


def test_build_model_from_config():
    cfg = ConfigDict({
        "model": {
            "name": "mlp_mixer",
            "image_size": 32,
            "patch_size": 4,
            "num_classes": 100,
            "hidden_dim": 64,
            "num_blocks": 2,
            "tokens_mlp_dim": 32,
            "channels_mlp_dim": 128,
        }
    })
    model = build_model(cfg)
    assert isinstance(model, MLPMixer)
    assert model.num_parameters() > 0
