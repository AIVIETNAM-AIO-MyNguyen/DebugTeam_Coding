"""
Base Model and Model Registry for MLP Variants.

Allows teammates to easily register new MLP architectures:
    @register_model("my_custom_mlp")
    class MyCustomMLP(BaseClassifier):
        ...
"""

import abc
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn as nn

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str):
    """Decorator to register a new model architecture."""
    def decorator(cls):
        key = name.lower().strip()
        if key in MODEL_REGISTRY:
            raise ValueError(f"Model '{key}' is already registered.")
        MODEL_REGISTRY[key] = cls
        return cls
    return decorator


def list_models() -> List[str]:
    """Returns a list of all registered model architecture names."""
    return sorted(list(MODEL_REGISTRY.keys()))


class BaseClassifier(nn.Module, abc.ABC):
    """
    Abstract base class for vision classifiers.
    Provides common utilities for parameter counting and feature extraction.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (B, C, H, W).
        Returns:
            Logits of shape (B, num_classes).
        """
        pass

    def num_parameters(self, trainable_only: bool = True) -> int:
        """Computes total number of parameters."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

    def get_summary(self, input_size: tuple = (3, 32, 32)) -> Dict[str, Any]:
        """Returns a summary dictionary of model parameters and architecture name."""
        return {
            "model_class": self.__class__.__name__,
            "total_params": self.num_parameters(trainable_only=False),
            "trainable_params": self.num_parameters(trainable_only=True),
            "input_size": input_size,
        }
