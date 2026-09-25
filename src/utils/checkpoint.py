"""
Model Checkpointing Utilities.

Provides unified checkpoint saving, loading, and resuming mechanisms.
"""

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union
import torch
import torch.nn as nn


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: Union[str, Path],
    filename: str = "last_model.pt",
    best_filename: str = "best_model.pt",
) -> Path:
    """
    Saves checkpoint state to disk. If is_best=True, copies the file to best_filename.

    Args:
        state: Dictionary with keys 'epoch', 'model_state', 'optimizer_state', etc.
        is_best: Boolean indicating if this is the best epoch so far.
        checkpoint_dir: Directory to save checkpoint files in.
        filename: Name of the latest checkpoint file.
        best_filename: Name of the best checkpoint file.

    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    file_path = checkpoint_dir / filename
    torch.save(state, file_path)

    if is_best:
        best_path = checkpoint_dir / best_filename
        shutil.copyfile(file_path, best_path)

    return file_path


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Loads model and training states from a checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        model: nn.Module to load weights into.
        optimizer: Optional optimizer to restore state.
        scheduler: Optional learning rate scheduler to restore state.
        scaler: Optional GradScaler to restore state.
        map_location: Device to load checkpoint on.
        strict: Whether to strictly enforce key matching in state_dict.

    Returns:
        Checkpoint dictionary containing metadata (e.g. epoch, best_metric, config).
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.save if False else torch.load(checkpoint_path, map_location=map_location, weights_only=False)

    # Handle model state dict (support raw model or nested under 'model_state' / 'state_dict')
    if "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Strip potential 'module.' prefixes from DistributedDataParallel
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    model.load_state_dict(clean_state_dict, strict=strict)

    # Restore optimizer state
    if optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    # Restore scheduler state
    if scheduler is not None and "scheduler_state" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state"])

    # Restore scaler state
    if scaler is not None and "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])

    return checkpoint
