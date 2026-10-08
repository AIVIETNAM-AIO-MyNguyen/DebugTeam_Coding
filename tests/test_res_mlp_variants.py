"""
Unit tests for ResMLP variants:
- ResMLP-RepToken (Idea 1: Local injection in all TokenMix layers)
- ResMLP-LocalStem (Idea 3: Local injection in the early Stem layers)
"""

import pytest
import torch
from src.models import build_model, list_models
from src.models.res_mlp import ResMLP
from src.models.res_mlp_rep_token import ResMLPRepToken
from src.models.res_mlp_local_stem import ResMLPLocalStem


def test_registry_contains_new_res_mlp_variants():
    models = list_models()
    assert "res_mlp_rep_token" in models
    assert "resmlp_rep_token" in models
    assert "res_mlp_reptoken" in models
    assert "res_mlp_local_stem" in models
    assert "resmlp_local_stem" in models
    assert "res_mlp_rep_stem" in models


def test_res_mlp_rep_token_forward():
    model = ResMLPRepToken(
        image_size=32,
        patch_size=4,
        num_classes=100,
        num_features=64,
        num_layers=4,
    )
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)


def test_res_mlp_rep_token_locality_injection():
    """Verify that fusing conv branches into Linear produces identical output."""
    model = ResMLPRepToken(
        image_size=32,
        patch_size=4,
        num_classes=10,
        num_features=32,
        num_layers=2,
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
    assert diff < 1e-4, f"Locality injection mismatch for ResMLPRepToken: max diff = {diff}"


def test_res_mlp_local_stem_forward():
    model = ResMLPLocalStem(
        image_size=32,
        patch_size=4,
        num_classes=100,
        num_features=64,
        num_layers=4,
        num_stem_blocks=2,
    )
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)


def test_res_mlp_local_stem_locality_injection():
    """Verify that fusing stem conv branches produces identical output."""
    model = ResMLPLocalStem(
        image_size=32,
        patch_size=4,
        num_classes=10,
        num_features=32,
        num_layers=4,
        num_stem_blocks=2,
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
    assert diff < 1e-4, f"Locality injection mismatch for ResMLPLocalStem: max diff = {diff}"


def test_res_mlp_variants_param_hierarchy():
    """
    Param hierarchy:
    ResMLP (Base) < ResMLPLocalStem (Local in stem only) < ResMLPRepToken (Local in all layers)
    """
    kwargs = dict(
        image_size=32,
        patch_size=4,
        num_classes=100,
        num_features=64,
        num_layers=6,
    )
    base = ResMLP(**kwargs)
    stem = ResMLPLocalStem(**kwargs, num_stem_blocks=2)
    token = ResMLPRepToken(**kwargs)

    p_base = base.num_parameters()
    p_stem = stem.num_parameters()
    p_token = token.num_parameters()

    assert p_base < p_stem < p_token
    print(f"ResMLP Base:       {p_base:,} params")
    print(f"ResMLP LocalStem:  {p_stem:,} params (+{p_stem - p_base:,})")
    print(f"ResMLP RepToken:   {p_token:,} params (+{p_token - p_base:,})")


def test_build_from_config():
    from src.config.parser import ConfigDict

    # Config for RepToken
    cfg1 = ConfigDict({
        "model": {"name": "res_mlp_rep_token", "image_size": 32, "patch_size": 4, "num_classes": 100},
        "data": {"dataset": "cifar100", "image_size": 32},
    })
    m1 = build_model(cfg1)
    assert isinstance(m1, ResMLPRepToken)

    # Config for LocalStem
    cfg2 = ConfigDict({
        "model": {"name": "res_mlp_local_stem", "image_size": 32, "patch_size": 4, "num_classes": 100, "num_stem_blocks": 2},
        "data": {"dataset": "cifar100", "image_size": 32},
    })
    m2 = build_model(cfg2)
    assert isinstance(m2, ResMLPLocalStem)
    assert m2.num_stem_blocks == 2


def test_load_and_build_from_yaml_configs():
    from src.config.parser import load_config

    # Test CIFAR-100 YAML configs
    cfg_rep_token = load_config(
        config_path="configs/cifar100_res_mlp_rep_token.yaml",
        default_config_path="configs/default.yaml",
    )
    m_rep_token = build_model(cfg_rep_token)
    assert isinstance(m_rep_token, ResMLPRepToken)
    assert m_rep_token.image_size == 32
    assert m_rep_token.num_classes == 100

    cfg_local_stem = load_config(
        config_path="configs/cifar100_res_mlp_local_stem.yaml",
        default_config_path="configs/default.yaml",
    )
    m_local_stem = build_model(cfg_local_stem)
    assert isinstance(m_local_stem, ResMLPLocalStem)
    assert m_local_stem.image_size == 32
    assert m_local_stem.num_classes == 100
    assert m_local_stem.num_stem_blocks == 2

    # Test Tiny-ImageNet YAML configs
    cfg_tiny_stem = load_config(
        config_path="configs/tiny_imagenet_res_mlp_local_stem.yaml",
        default_config_path="configs/default.yaml",
    )
    m_tiny_stem = build_model(cfg_tiny_stem)
    assert isinstance(m_tiny_stem, ResMLPLocalStem)
    assert m_tiny_stem.image_size == 64
    assert m_tiny_stem.num_classes == 200
