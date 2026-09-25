"""
Deterministic Seed Control Utilities.

Provides unified seed control across Python's random, NumPy, PyTorch (CPU & CUDA),
and PyTorch DataLoader multi-processing workers.
"""

import os
import random
from typing import Optional
import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Sets seed across all libraries to ensure reproducible experiments.

    Args:
        seed: Random seed integer.
        deterministic: If True, configures cuDNN to use deterministic algorithms.
            Note: deterministic=True may have a small performance penalty on GPU.
    """
    # 1. Python built-in random and hash seed
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2. NumPy
    np.random.seed(seed)

    # 3. PyTorch (CPU & CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 4. CuDNN backend
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Benchmark mode optimizes CUDA kernels for fixed input shapes
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """
    Worker initialization function for PyTorch DataLoader.
    Ensures each multiprocessing DataLoader worker has a distinct yet deterministic seed.

    Usage:
        DataLoader(dataset, worker_init_fn=seed_worker, generator=get_generator(seed))
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_generator(seed: Optional[int] = None) -> torch.Generator:
    """
    Creates and seeds a PyTorch Generator for DataLoader shuffling.

    Args:
        seed: Seed integer. If None, uses a default seed of 42.

    Returns:
        torch.Generator seeded with `seed`.
    """
    generator = torch.Generator()
    generator.manual_seed(seed if seed is not None else 42)
    return generator
