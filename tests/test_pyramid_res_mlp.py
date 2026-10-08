"""
Unit tests for Pyramid-ResMLP:
- Overlapping patch embedding (Stem)
- Hierarchical multi-stage pyramid structure with convolutional downsamplers
- RepTokenMixLayer with local DW convs
- Locality injection equivalence (Re-parameterization diff < 1e-4)
"""

import pytest
import torch
from src.models import build_model, list_models
from src.models.pyramid_res_mlp import PyramidResMLP


def test_registry_contains_pyramid_res_mlp():
    models = list_models()
    assert "pyramid_res_mlp" in models
    assert "pyramid_rep_res_mlp" in models
    assert "res_mlp_pyramid" in models


def test_pyramid_res_mlp_cifar_forward():
    model = PyramidResMLP(
        image_size=32,
        patch_size=2,
        num_classes=100,
        channels=(32, 64, 128),
        num_blocks=(2, 2, 2),
    )
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)


def test_pyramid_res_mlp_tiny_imagenet_forward():
    model = PyramidResMLP(
        image_size=64,
        patch_size=4,
        num_classes=200,
        channels=(32, 64, 128),
        num_blocks=(2, 2, 2),
    )
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 200)


def test_pyramid_res_mlp_locality_injection():
    """Verify that fusing all RepTokenMixLayers across all pyramid stages is exact."""
    model = PyramidResMLP(
        image_size=32,
        patch_size=2,
        num_classes=10,
        channels=(32, 64, 128),
        num_blocks=(1, 1, 1),
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
    assert diff < 1e-4, f"Locality injection mismatch for PyramidResMLP: max diff = {diff}"


def test_pyramid_res_mlp_param_count():
    model = PyramidResMLP(
        image_size=32,
        patch_size=2,
        num_classes=100,
        channels=(64, 128, 256),
        num_blocks=(2, 2, 2),
        expansion_factor=4,
    )
    params = model.num_parameters()
    print(f"Pyramid-ResMLP (2, 2, 2) params: {params:,}")
    assert params > 1_500_000


def test_build_from_config():
    from src.config.parser import ConfigDict

    cfg = ConfigDict({
        "model": {
            "name": "pyramid_res_mlp",
            "image_size": 32,
            "patch_size": 2,
            "num_classes": 100,
            "channels": [64, 128, 256],
            "num_blocks": [2, 2, 2],
        },
        "data": {"dataset": "cifar100", "image_size": 32},
    })
    m = build_model(cfg)
    assert isinstance(m, PyramidResMLP)
    assert m.num_stages == 3


def test_load_and_build_from_yaml_configs():
    from src.config.parser import load_config

    # Test CIFAR-100 YAML config
    cfg_cifar = load_config(
        config_path="configs/cifar100_pyramid_res_mlp.yaml",
        default_config_path="configs/default.yaml",
    )
    m_cifar = build_model(cfg_cifar)
    assert isinstance(m_cifar, PyramidResMLP)
    assert m_cifar.image_size == 32
    assert m_cifar.num_classes == 100
    assert m_cifar.num_stages == 3

    # Test Tiny-ImageNet YAML config
    cfg_tiny = load_config(
        config_path="configs/tiny_imagenet_pyramid_res_mlp.yaml",
        default_config_path="configs/default.yaml",
    )
    m_tiny = build_model(cfg_tiny)
    assert isinstance(m_tiny, PyramidResMLP)
    assert m_tiny.image_size == 64
    assert m_tiny.num_classes == 200
    assert m_tiny.num_stages == 3
    assert m_tiny.drop_path_rate == 0.1


def test_pyramid_res_mlp_drop_path_and_dropout():
    model = PyramidResMLP(
        image_size=32,
        patch_size=2,
        num_classes=10,
        channels=(32, 64, 128),
        num_blocks=(2, 2, 2),
        drop_path_rate=0.2,
        dropout=0.1,
    )
    # Forward in train mode with backward pass
    model.train()
    x = torch.randn(4, 3, 32, 32, requires_grad=True)
    out = model(x)
    assert out.shape == (4, 10)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None

    # In eval mode, DropPath and Dropout are deactivated, locality injection is exact
    model.eval()
    with torch.no_grad():
        out_train_mode = model(x)
    model.locality_injection()
    with torch.no_grad():
        out_fused = model(x)
    diff = (out_train_mode - out_fused).abs().max().item()
    assert diff < 1e-4, f"Locality injection mismatch with drop_path: max diff = {diff}"
