"""
Unit tests for RepMLP-Sep (Cách 2: Per-Branch Depthwise-Separable variant).
Tests forward pass, registry, locality injection equivalence, and parameter count.
"""

import pytest
import torch
from src.models import build_model, list_models
from src.models.rep_mlp_sep import RepMLPSep


def test_registry_contains_rep_mlp_sep():
    models = list_models()
    assert "rep_mlp_sep" in models
    assert "repmlp_sep" in models
    assert "rep_mlp_ds2" in models


def test_rep_mlp_sep_cifar_forward():
    model = RepMLPSep(
        image_size=32,
        patch_size=2,
        num_classes=100,
        channels=(64, 128, 256),
        num_blocks=(2, 2, 2),
    )
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)


def test_rep_mlp_sep_tiny_imagenet_forward():
    model = RepMLPSep(
        image_size=64,
        patch_size=4,
        num_classes=200,
        channels=(64, 128, 256),
        num_blocks=(2, 2, 2),
    )
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 200)


def test_rep_mlp_sep_locality_injection_equivalence():
    """
    Core correctness test: verify that fusing all per-branch Separable Conv blocks
    into FC3 produces numerically identical outputs (diff < 1e-4).
    """
    model = RepMLPSep(
        image_size=32,
        patch_size=2,
        num_classes=10,
        channels=(32, 64),
        num_blocks=(1, 1),
        sharesets_nums=(1, 4),
        use_separable=True,
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


def test_rep_mlp_sep_params_comparison():
    """RepMLP-Sep should have PW layers for both k=1 and k=3 branches."""
    from src.models.rep_mlp import RepMLP

    kwargs = dict(
        image_size=32,
        patch_size=2,
        num_classes=100,
        channels=(64, 128, 256),
        num_blocks=(2, 2, 2),
        sharesets_nums=(1, 4, 16),
    )
    orig = RepMLP(**kwargs)
    sep = RepMLPSep(**kwargs, use_separable=True)

    orig_p = orig.num_parameters()
    sep_p = sep.num_parameters()
    assert sep_p > orig_p
    print(f"Original RepMLP: {orig_p:,} params")
    print(f"RepMLP-Sep (Cách 2): {sep_p:,} params (+{sep_p - orig_p:,} params)")


def test_rep_mlp_sep_without_separable_matches_original_equivalence():
    """With use_separable=False, fallback locality injection works properly."""
    model = RepMLPSep(
        image_size=32,
        patch_size=2,
        num_classes=10,
        channels=(32, 64),
        num_blocks=(1, 1),
        sharesets_nums=(1, 4),
        use_separable=False,
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
    assert diff < 1e-4, f"Fallback injection mismatch: max diff = {diff}"


def test_build_rep_mlp_sep_from_config():
    from src.config.parser import ConfigDict

    cfg = ConfigDict({
        "model": {
            "name": "rep_mlp_sep",
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
    assert isinstance(model, RepMLPSep)
    assert model.num_parameters() > 0


def test_load_and_build_from_yaml_configs():
    from src.config.parser import load_config

    # Test CIFAR-100 YAML config
    cfg_cifar = load_config(
        config_path="configs/cifar100_rep_mlp_sep.yaml",
        default_config_path="configs/default.yaml",
    )
    model_cifar = build_model(cfg_cifar)
    assert isinstance(model_cifar, RepMLPSep)
    assert model_cifar.image_size == 32
    assert model_cifar.num_classes == 100

    # Test Tiny-ImageNet YAML config
    cfg_tiny = load_config(
        config_path="configs/tiny_imagenet_rep_mlp_sep.yaml",
        default_config_path="configs/default.yaml",
    )
    model_tiny = build_model(cfg_tiny)
    assert isinstance(model_tiny, RepMLPSep)
    assert model_tiny.image_size == 64
    assert model_tiny.num_classes == 200
