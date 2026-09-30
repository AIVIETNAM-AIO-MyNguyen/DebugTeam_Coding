"""
Unit tests for Pyramid-ResMLP Pure (Ablation model without parallel CNN branches).
"""

import pytest
import torch
from src.models import build_model, list_models
from src.models.pyramid_res_mlp_pure import PyramidResMLPPure
from src.models.pyramid_res_mlp import PyramidResMLP


def test_registry_contains_pyramid_res_mlp_pure():
    models = list_models()
    assert "pyramid_res_mlp_pure" in models
    assert "pure_pyramid_res_mlp" in models
    assert "pyramid_resmlp_pure" in models


def test_pyramid_res_mlp_pure_cifar_forward():
    model = PyramidResMLPPure(
        image_size=32,
        patch_size=2,
        num_classes=100,
        channels=(32, 64, 128),
        num_blocks=(2, 2, 2),
    )
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)


def test_pyramid_res_mlp_pure_tiny_imagenet_forward():
    model = PyramidResMLPPure(
        image_size=64,
        patch_size=4,
        num_classes=200,
        channels=(32, 64, 128),
        num_blocks=(2, 2, 2),
    )
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 200)


def test_pyramid_res_mlp_pure_param_comparison():
    """
    Pure model should have slightly fewer parameters during training because it has
    no DW 3x3 and DW 1x1 conv branches + BNs in the TokenMix layers.
    """
    kwargs = dict(
        image_size=32,
        patch_size=2,
        num_classes=100,
        channels=(64, 128, 256),
        num_blocks=(2, 2, 2),
        expansion_factor=4,
    )
    m_pure = PyramidResMLPPure(**kwargs)
    m_full = PyramidResMLP(**kwargs)

    p_pure = m_pure.num_parameters()
    p_full = m_full.num_parameters()

    print(f"Pyramid-ResMLP Pure: {p_pure:,} params")
    print(f"Pyramid-ResMLP Full: {p_full:,} params (+{p_full - p_pure:,})")
    assert p_pure < p_full


def test_build_from_config():
    from src.config.parser import ConfigDict

    cfg = ConfigDict({
        "model": {
            "name": "pyramid_res_mlp_pure",
            "image_size": 32,
            "patch_size": 2,
            "num_classes": 100,
            "channels": [64, 128, 256],
            "num_blocks": [2, 2, 2],
        },
        "data": {"dataset": "cifar100", "image_size": 32},
    })
    m = build_model(cfg)
    assert isinstance(m, PyramidResMLPPure)
    assert m.num_stages == 3


def test_load_and_build_from_yaml_configs():
    from src.config.parser import load_config

    cfg_cifar = load_config(
        config_path="configs/cifar100_pyramid_res_mlp_pure.yaml",
        default_config_path="configs/default.yaml",
    )
    m_cifar = build_model(cfg_cifar)
    assert isinstance(m_cifar, PyramidResMLPPure)
    assert m_cifar.image_size == 32
    assert m_cifar.num_classes == 100

    cfg_tiny = load_config(
        config_path="configs/tiny_imagenet_pyramid_res_mlp_pure.yaml",
        default_config_path="configs/default.yaml",
    )
    m_tiny = build_model(cfg_tiny)
    assert isinstance(m_tiny, PyramidResMLPPure)
    assert m_tiny.image_size == 64
    assert m_tiny.num_classes == 200
