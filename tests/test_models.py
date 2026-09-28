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


def test_registry_contains_rep_mlp():
    models = list_models()
    assert "rep_mlp" in models
    assert "repmlp" in models
    assert "repmlpnet" in models


def test_rep_mlp_cifar_forward():
    from src.models.rep_mlp import RepMLP

    model = RepMLP(
        image_size=32,
        patch_size=2,
        num_classes=100,
        channels=(64, 128, 256),
        num_blocks=(2, 2, 2),
    )
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)


def test_rep_mlp_tiny_imagenet_forward():
    from src.models.rep_mlp import RepMLP

    model = RepMLP(
        image_size=64,
        patch_size=4,
        num_classes=200,
        channels=(64, 128, 256),
        num_blocks=(2, 2, 2),
    )
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 200)


def test_rep_mlp_locality_injection_equivalence():
    from src.models.rep_mlp import RepMLP

    model = RepMLP(
        image_size=32,
        patch_size=2,
        num_classes=10,
        channels=(32, 64),
        num_blocks=(1, 1),
    )
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_train = model(x)

    # Perform structural re-parameterization
    model.locality_injection()
    assert model.deploy is True

    with torch.no_grad():
        out_deploy = model(x)

    diff = (out_train - out_deploy).abs().max().item()
    assert diff < 1e-4, f"Locality injection mismatch: max diff = {diff}"


def test_build_rep_mlp_from_config():
    from src.models.rep_mlp import RepMLP

    cfg = ConfigDict({
        "model": {
            "name": "rep_mlp",
            "image_size": 32,
            "patch_size": 2,
            "num_classes": 100,
        },
        "data": {
            "dataset": "cifar100",
            "image_size": 32,
        },
    })
    model = build_model(cfg)
    assert isinstance(model, RepMLP)
    assert model.num_parameters() > 0
