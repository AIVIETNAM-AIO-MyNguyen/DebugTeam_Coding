"""
Unit tests for RepMLP-DWS (Depthwise-Separable variant).
Tests forward pass, registry, locality injection equivalence, and param comparison.
"""

import pytest
import torch
from src.models import build_model, list_models
from src.models.rep_mlp_dws import RepMLPDWS


def test_registry_contains_rep_mlp_dws():
    models = list_models()
    assert "rep_mlp_dws" in models
    assert "repmlp_dws" in models


def test_rep_mlp_dws_cifar_forward():
    model = RepMLPDWS(
        image_size=32,
        patch_size=2,
        num_classes=100,
        channels=(64, 128, 256),
        num_blocks=(2, 2, 2),
    )
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)


def test_rep_mlp_dws_tiny_imagenet_forward():
    model = RepMLPDWS(
        image_size=64,
        patch_size=4,
        num_classes=200,
        channels=(64, 128, 256),
        num_blocks=(2, 2, 2),
    )
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 200)


def test_rep_mlp_dws_locality_injection_equivalence():
    """
    The core test: verify that fusing DW-Sep branches into FC3
    produces mathematically identical outputs (within numerical tolerance).
    """
    model = RepMLPDWS(
        image_size=32,
        patch_size=2,
        num_classes=10,
        channels=(32, 64),
        num_blocks=(1, 1),
        sharesets_nums=(1, 4),
        use_dw_sep=True,
    )
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_train = model(x)

    # Perform structural re-parameterization (fuse DW, PW, DW-Sep into FC3)
    model.locality_injection()
    assert model.deploy is True

    with torch.no_grad():
        out_deploy = model(x)

    diff = (out_train - out_deploy).abs().max().item()
    assert diff < 1e-4, f"Locality injection mismatch: max diff = {diff}"


def test_rep_mlp_dws_has_more_params_than_original():
    """DWS variant should have slightly more params due to PW branches."""
    from src.models.rep_mlp import RepMLP

    kwargs = dict(
        image_size=32,
        patch_size=2,
        num_classes=100,
        channels=(64, 128, 256),
        num_blocks=(2, 2, 2),
        sharesets_nums=(1, 4, 16),
    )
    original = RepMLP(**kwargs)
    dws = RepMLPDWS(**kwargs, use_dw_sep=True)

    orig_params = original.num_parameters()
    dws_params = dws.num_parameters()

    # DWS should have more params (due to PW 1×1 branches in stages with S > 1)
    assert dws_params > orig_params, (
        f"Expected DWS ({dws_params}) > Original ({orig_params})"
    )
    print(f"Original: {orig_params:,} params")
    print(f"DWS:      {dws_params:,} params")
    print(f"Overhead:  {dws_params - orig_params:,} params (+{(dws_params - orig_params) / orig_params * 100:.2f}%)")


def test_rep_mlp_dws_without_dwsep_matches_original_behavior():
    """With use_dw_sep=False, DWS variant should behave like original."""
    model = RepMLPDWS(
        image_size=32,
        patch_size=2,
        num_classes=10,
        channels=(32, 64),
        num_blocks=(1, 1),
        use_dw_sep=False,
    )
    model.eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        out_train = model(x)

    model.locality_injection()
    with torch.no_grad():
        out_deploy = model(x)

    diff = (out_train - out_deploy).abs().max().item()
    assert diff < 1e-4, f"Mismatch with use_dw_sep=False: max diff = {diff}"


def test_build_rep_mlp_dws_from_config():
    from src.config.parser import ConfigDict

    cfg = ConfigDict({
        "model": {
            "name": "rep_mlp_dws",
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
    assert isinstance(model, RepMLPDWS)
    assert model.num_parameters() > 0


def test_load_and_build_from_yaml_configs():
    from src.config.parser import load_config

    # Test CIFAR-100 YAML config
    cfg_cifar = load_config(
        config_path="configs/cifar100_rep_mlp_dws.yaml",
        default_config_path="configs/default.yaml",
    )
    model_cifar = build_model(cfg_cifar)
    assert isinstance(model_cifar, RepMLPDWS)
    assert model_cifar.image_size == 32
    assert model_cifar.num_classes == 100

    # Test Tiny-ImageNet YAML config
    cfg_tiny = load_config(
        config_path="configs/tiny_imagenet_rep_mlp_dws.yaml",
        default_config_path="configs/default.yaml",
    )
    model_tiny = build_model(cfg_tiny)
    assert isinstance(model_tiny, RepMLPDWS)
    assert model_tiny.image_size == 64
    assert model_tiny.num_classes == 200
