"""
Models Package.

Provides `build_model(cfg)` and exports all registered MLP architectures.
"""

from typing import Any, Dict
import torch.nn as nn

from src.models.base import BaseClassifier, MODEL_REGISTRY, list_models, register_model
# Import all model files to trigger registration
import src.models.mlp_mixer  # noqa: F401


def build_model(cfg) -> BaseClassifier:
    """
    Builds a vision model instance from configuration.

    Args:
        cfg: Configuration dictionary or ConfigDict containing `cfg.model`.

    Returns:
        Instantiated nn.Module subclassing BaseClassifier.
    """
    model_name = cfg.model.name.lower().strip()
    if model_name not in MODEL_REGISTRY:
        available = list_models()
        raise ValueError(
            f"Unknown model name '{model_name}'. Available registered models: {available}"
        )

    model_cls = MODEL_REGISTRY[model_name]

    # Collect parameters: start with all keys under cfg.model, remove 'name'
    model_params = {}
    if hasattr(cfg.model, "to_dict"):
        raw_params = cfg.model.to_dict()
    elif isinstance(cfg.model, dict):
        raw_params = dict(cfg.model)
    else:
        raw_params = {}

    for k, v in raw_params.items():
        if k != "name":
            model_params[k] = v

    # Automatically synchronize num_classes with dataset
    if hasattr(cfg, "data"):
        if "num_classes" in cfg.data and cfg.data.num_classes is not None:
            model_params["num_classes"] = cfg.data.num_classes
        elif cfg.data.dataset == "cifar100":
            model_params.setdefault("num_classes", 100)
        elif "tiny_imagenet" in cfg.data.dataset:
            model_params.setdefault("num_classes", 200)

    if "image_size" not in model_params and hasattr(cfg, "data") and "image_size" in cfg.data:
        if cfg.data.image_size is not None:
            model_params["image_size"] = cfg.data.image_size

    return model_cls(**model_params)


__all__ = [
    "BaseClassifier",
    "MODEL_REGISTRY",
    "list_models",
    "register_model",
    "build_model",
]
