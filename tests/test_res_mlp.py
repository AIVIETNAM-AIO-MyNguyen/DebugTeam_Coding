"""
Unit tests for ResMLP architecture with Local Perceptron and Structural Re-parameterization.
"""

import pytest
import torch
from src.config.parser import ConfigDict
from src.models import build_model, list_models
from src.models.res_mlp import ResMLP, LocalPerceptron, AffineTransform


def test_registry_contains_res_mlp():
    models = list_models()
    assert "res_mlp" in models
    assert "resmlp" in models


def test_affine_transform():
    aff = AffineTransform(128)
    x = torch.randn(2, 64, 128)
    out = aff(x)
    assert out.shape == (2, 64, 128)
    diff = (x - out).abs().max().item()
    assert diff == 0.0  # Initialized as identity


def test_res_mlp_cifar_forward():
    model = ResMLP(
        image_size=32,
        patch_size=4,
        num_classes=100,
        num_features=128,
        expansion_factor=2,
        num_layers=4,
        use_local_perceptron=True,
    )
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)


def test_res_mlp_tiny_imagenet_forward():
    model = ResMLP(
        image_size=64,
        patch_size=8,
        num_classes=200,
        num_features=128,
        expansion_factor=2,
        num_layers=4,
        use_local_perceptron=True,
    )
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 200)


def test_res_mlp_locality_injection_equivalence():
    model = ResMLP(
        image_size=32,
        patch_size=4,
        num_classes=10,
        num_features=64,
        expansion_factor=2,
        num_layers=2,
        use_local_perceptron=True,
    )
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_train = model(x)

    model.locality_injection()
    assert model.deploy is True

    with torch.no_grad():
        out_deploy = model(x)

    diff = (out_train - out_deploy).abs().max().item()
    assert diff < 1e-4, f"ResMLP locality injection mismatch: max diff = {diff}"


def test_build_res_mlp_from_config():
    cfg = ConfigDict({
        "model": {
            "name": "res_mlp",
            "image_size": 32,
            "patch_size": 4,
            "num_classes": 100,
            "num_features": 64,
            "expansion_factor": 2,
            "num_layers": 2,
            "use_local_perceptron": True,
        },
        "data": {
            "dataset": "cifar100",
            "image_size": 32,
        },
    })
    model = build_model(cfg)
    assert isinstance(model, ResMLP)
    assert model.num_parameters() > 0
