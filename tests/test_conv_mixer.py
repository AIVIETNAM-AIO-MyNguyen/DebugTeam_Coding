import pytest
import torch
from src.config.parser import ConfigDict
from src.models import build_model, list_models
from src.models.conv_mixer import ConvMixer, ResConvMixer


def test_models_registration():
    models = list_models()
    assert "conv_mixer" in models
    assert "res_conv_mixer" in models


def test_conv_mixer_forward():
    model = ConvMixer(
        image_size=32,
        patch_size=2,
        dim=128,
        depth=4,
        kernel_size=5,
        num_classes=100,
    )
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)
    assert model.num_parameters() > 0


def test_res_conv_mixer_forward():
    model = ResConvMixer(
        image_size=32,
        patch_size=2,
        dim=128,
        depth=4,
        kernel_size=5,
        expansion_factor=2,
        num_classes=100,
    )
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)
    assert model.num_parameters() > 0


def test_build_conv_mixer_from_config():
    cfg = ConfigDict({
        "model": {
            "name": "conv_mixer",
            "image_size": 32,
            "patch_size": 2,
            "dim": 128,
            "depth": 4,
            "kernel_size": 7,
            "num_classes": 100,
        },
        "data": {
            "dataset": "cifar100",
            "num_classes": 100,
            "image_size": 32,
        }
    })
    model = build_model(cfg)
    assert isinstance(model, ConvMixer)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)


def test_build_res_conv_mixer_from_config():
    cfg = ConfigDict({
        "model": {
            "name": "res_conv_mixer",
            "image_size": 32,
            "patch_size": 2,
            "dim": 128,
            "depth": 4,
            "kernel_size": 7,
            "expansion_factor": 2,
            "num_classes": 100,
        },
        "data": {
            "dataset": "cifar100",
            "num_classes": 100,
            "image_size": 32,
        }
    })
    model = build_model(cfg)
    assert isinstance(model, ResConvMixer)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)
