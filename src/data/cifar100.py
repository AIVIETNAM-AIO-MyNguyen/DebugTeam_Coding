"""
CIFAR-100 Dataset Pipeline.

100 classes, 60,000 32x32 color images (50,000 train, 10,000 test).
Supports automatic download, deterministic worker seeding, and train/val/test splits.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import CIFAR100
from torchvision.datasets.utils import download_and_extract_archive

# High-speed HuggingFace mirror for CIFAR-100 (bypasses 67kB/s rate-limiting on Toronto server)
HUGGINGFACE_CIFAR100_URL = (
    "https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/cifar-100-python.tar.gz"
)
TORONTO_CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"


class FastCIFAR100(CIFAR100):
    """
    Subclass of torchvision CIFAR100 using a high-speed HuggingFace CDN mirror
    with automatic fallback to the official University of Toronto server.
    """
    url = HUGGINGFACE_CIFAR100_URL

    def download(self) -> None:
        if self._check_integrity():
            return

        import os
        os.makedirs(self.root, exist_ok=True)
        try:
            print(f"Downloading CIFAR-100 from high-speed HuggingFace mirror: {HUGGINGFACE_CIFAR100_URL}")
            download_and_extract_archive(
                HUGGINGFACE_CIFAR100_URL,
                self.root,
                filename=self.filename,
                md5=self.tgz_md5,
            )
        except Exception as e:
            print(f"HuggingFace mirror failed ({e}), falling back to official Toronto server...")
            download_and_extract_archive(
                TORONTO_CIFAR100_URL,
                self.root,
                filename=self.filename,
                md5=self.tgz_md5,
            )


# Monkey-patch global torchvision CIFAR100.url so any direct calls also use the fast mirror
CIFAR100.url = HUGGINGFACE_CIFAR100_URL

from src.data.transforms import MixupCutmixCollate, get_transforms
from src.utils.seed import get_generator, seed_worker


def get_cifar100_datasets(
    data_dir: Union[str, Path] = "data/cifar100",
    image_size: Optional[int] = None,
    val_split: float = 0.1,
    seed: int = 42,
    auto_augment: bool = False,
    rand_augment: bool = False,
    random_erasing_prob: float = 0.0,
    color_jitter: float = 0.0,
    download: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Downloads and prepares CIFAR-100 train, validation, and test datasets.

    Args:
        data_dir: Path to store/load CIFAR-100.
        image_size: Optional resize (e.g. 32, 64, or 224). If None, keeps 32x32.
        val_split: Fraction of training set to use for validation (e.g. 0.1 = 5,000 images).
                   If 0.0, uses the test set as validation.
        seed: Random seed for deterministic train/val split.
        auto_augment: Enable AutoAugment on training set.
        rand_augment: Enable RandAugment on training set.
        random_erasing_prob: Probability of RandomErasing.
        color_jitter: Color jitter factor.
        download: Whether to download if not present.

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Check for pre-mounted Kaggle/Colab datasets to bypass slow Toronto HTTP download
    kaggle_candidates = [
        Path("/kaggle/input/cifar100-python"),
        Path("/kaggle/input/cifar-100-python"),
        Path("/kaggle/input/cifar100"),
        Path("/kaggle/input/cifar-100"),
    ]
    for cand in kaggle_candidates:
        if cand.is_dir():
            if (cand / "cifar-100-python").is_dir():
                data_dir = cand
                download = False
                break
            elif (cand / "train").is_file():
                data_dir = cand.parent
                download = False
                break

    # If dataset already exists locally, disable download to avoid re-triggering slow server check
    if (data_dir / "cifar-100-python" / "train").is_file():
        download = False

    train_transform = get_transforms(
        dataset_name="cifar100",
        image_size=image_size,
        is_train=True,
        auto_augment=auto_augment,
        rand_augment=rand_augment,
        random_erasing_prob=random_erasing_prob,
        color_jitter=color_jitter,
    )

    eval_transform = get_transforms(
        dataset_name="cifar100",
        image_size=image_size,
        is_train=False,
    )

    test_dataset = FastCIFAR100(root=str(data_dir), train=False, download=download, transform=eval_transform)

    if val_split > 0.0:
        # Load two copies of full train data: one with train transforms, one with eval transforms
        full_train_aug = FastCIFAR100(root=str(data_dir), train=True, download=download, transform=train_transform)
        full_train_eval = FastCIFAR100(root=str(data_dir), train=True, download=download, transform=eval_transform)

        total_train = len(full_train_aug)
        val_len = int(total_train * val_split)
        train_len = total_train - val_len

        # Deterministic index split
        generator = torch.Generator().manual_seed(seed)
        indices = torch.randperm(total_train, generator=generator).tolist()
        train_indices = indices[:train_len]
        val_indices = indices[train_len:]

        train_dataset = Subset(full_train_aug, train_indices)
        val_dataset = Subset(full_train_eval, val_indices)
    else:
        train_dataset = FastCIFAR100(root=str(data_dir), train=True, download=download, transform=train_transform)
        val_dataset = test_dataset

    return train_dataset, val_dataset, test_dataset


def get_cifar100_dataloaders(
    data_dir: Union[str, Path] = "data/cifar100",
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: Optional[int] = None,
    val_split: float = 0.1,
    seed: int = 42,
    pin_memory: bool = True,
    use_mixup: bool = False,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    auto_augment: bool = False,
    rand_augment: bool = False,
    random_erasing_prob: float = 0.0,
    color_jitter: float = 0.0,
    download: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs PyTorch DataLoaders for CIFAR-100 with reproducible worker seeding.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_ds, val_ds, test_ds = get_cifar100_datasets(
        data_dir=data_dir,
        image_size=image_size,
        val_split=val_split,
        seed=seed,
        auto_augment=auto_augment,
        rand_augment=rand_augment,
        random_erasing_prob=random_erasing_prob,
        color_jitter=color_jitter,
        download=download,
    )

    collate_fn = None
    if use_mixup:
        collate_fn = MixupCutmixCollate(
            num_classes=100,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
        )

    # Multi-GPU DistributedSampler support
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_ds,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=True,
            seed=seed,
        )
        shuffle = False
    else:
        train_sampler = None
        shuffle = True

    persistent_workers = num_workers > 0
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
        generator=get_generator(seed) if train_sampler is None else None,
        collate_fn=collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=seed_worker,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
