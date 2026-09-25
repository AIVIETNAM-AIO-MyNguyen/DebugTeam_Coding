"""
Distributed Training Utilities (DDP).

Enables multi-GPU training (e.g. 2x NVIDIA T4 on Kaggle, A100/H100 clusters).
Safe for single-GPU and CPU environments.
"""

import os
from typing import Optional, Tuple
import torch
import torch.distributed as dist


def init_distributed_mode() -> Tuple[bool, int, int, int]:
    """
    Initializes DistributedDataParallel (DDP) environment if launched via torchrun.

    Returns:
        (is_distributed, rank, local_rank, world_size)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        dist.barrier()
        return True, rank, local_rank, world_size
    else:
        # Single GPU or CPU
        return False, 0, 0, 1


def cleanup_distributed() -> None:
    """Cleans up the distributed process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    """Returns True if the current process is the main process (rank 0)."""
    return not dist.is_initialized() or dist.get_rank() == 0


def get_rank() -> int:
    """Returns global rank (0 if not distributed)."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Returns world size (1 if not distributed)."""
    return dist.get_world_size() if dist.is_initialized() else 1


def reduce_tensor(tensor: torch.Tensor, average: bool = True) -> torch.Tensor:
    """
    Reduces tensor across all distributed processes (e.g. for gathering loss/accuracy).
    """
    if not dist.is_initialized():
        return tensor

    reduced = tensor.clone()
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    if average:
        reduced /= get_world_size()
    return reduced
