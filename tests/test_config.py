"""
Unit tests for configuration system.
"""

from pathlib import Path
import pytest
from src.config.parser import ConfigDict, deep_merge, load_config, parse_cli_overrides


def test_config_dict_attribute_access():
    cfg = ConfigDict({"model": {"name": "mlp_mixer", "hidden_dim": 256}})
    assert cfg.model.name == "mlp_mixer"
    assert cfg.model.hidden_dim == 256

    # Test modification
    cfg.model.hidden_dim = 512
    assert cfg.model.hidden_dim == 512


def test_deep_merge():
    base = {"training": {"lr": 0.001, "epochs": 100}, "data": {"batch_size": 128}}
    update = {"training": {"lr": 0.0005}, "data": {"num_workers": 8}}

    merged = deep_merge(base, update)
    assert merged["training"]["lr"] == 0.0005
    assert merged["training"]["epochs"] == 100
    assert merged["data"]["batch_size"] == 128
    assert merged["data"]["num_workers"] == 8


def test_parse_cli_overrides():
    cli_args = [
        "--training.lr", "5e-4",
        "--training.epochs", "50",
        "--training.deterministic", "true",
        "--model.hidden_dim", "384",
    ]
    overrides = parse_cli_overrides(cli_args)
    assert overrides["training"]["lr"] == 5e-4
    assert overrides["training"]["epochs"] == 50
    assert overrides["training"]["deterministic"] is True
    assert overrides["model"]["hidden_dim"] == 384


def test_load_default_and_cifar_config():
    cfg = load_config(
        config_path="configs/cifar100_mlp_mixer.yaml",
        default_config_path="configs/default.yaml",
    )
    assert cfg.data.dataset == "cifar100"
    assert cfg.model.name == "mlp_mixer"
    assert cfg.training.seed == 42
