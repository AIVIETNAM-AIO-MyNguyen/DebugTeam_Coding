"""
Unified Data Pipeline Package.

Provides a single function `build_dataloaders(cfg)` to instantiate dataloaders for:
- CIFAR-100
- Tiny-ImageNet-200
- ImageNet Subset (e.g. ImageNet-100)
"""

from typing import Tuple
from torch.utils.data import DataLoader

from src.data.transforms import get_transforms, MixupCutmixCollate, DATASET_STATS
from src.data.cifar100 import get_cifar100_datasets, get_cifar100_dataloaders
from src.data.tiny_imagenet import (
    TinyImageNetDataset,
    get_tiny_imagenet_datasets,
    get_tiny_imagenet_dataloaders,
    download_and_extract_tiny_imagenet,
)
from src.data.imagenet_subset import (
    ImageNetSubsetDataset,
    get_imagenet_subset_datasets,
    get_imagenet_subset_dataloaders,
    generate_mock_imagenet_data,
)


def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs (train_loader, val_loader, test_loader) based on configuration.

    Args:
        cfg: Configuration dictionary or ConfigDict containing `cfg.data` and `cfg.training`.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    dataset_name = cfg.data.dataset.lower().replace("-", "_")
    batch_size = cfg.data.batch_size
    num_workers = cfg.data.num_workers
    pin_memory = cfg.data.get("pin_memory", True)
    image_size = cfg.data.get("image_size", None)
    seed = cfg.training.seed

    # Augmentation flags
    aug = cfg.get("augmentation", {})
    auto_augment = aug.get("auto_augment", False)
    rand_augment = aug.get("rand_augment", False)
    random_erasing_prob = aug.get("random_erasing", 0.0)
    color_jitter = aug.get("color_jitter", 0.0)
    use_mixup = aug.get("use_mixup", False)
    mixup_alpha = aug.get("mixup_alpha", 0.8)
    cutmix_alpha = aug.get("cutmix_alpha", 1.0)

    if dataset_name == "cifar100":
        return get_cifar100_dataloaders(
            data_dir=cfg.data.data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            val_split=cfg.data.get("val_split", 0.1),
            seed=seed,
            pin_memory=pin_memory,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            auto_augment=auto_augment,
            rand_augment=rand_augment,
            random_erasing_prob=random_erasing_prob,
            color_jitter=color_jitter,
            download=cfg.data.get("download", True),
        )

    elif dataset_name in ("tiny_imagenet", "tinyimagenet", "tiny_imagenet_200"):
        return get_tiny_imagenet_dataloaders(
            data_dir=cfg.data.data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            image_size=image_size,
            seed=seed,
            pin_memory=pin_memory,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            auto_augment=auto_augment,
            rand_augment=rand_augment,
            random_erasing_prob=random_erasing_prob,
            color_jitter=color_jitter,
            download=cfg.data.get("download", True),
        )

    elif dataset_name in ("imagenet_subset", "imagenet100", "imagenet_100", "imagenet"):
        return get_imagenet_subset_dataloaders(
            data_dir=cfg.data.data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=cfg.data.get("num_classes", 100),
            class_list=cfg.data.get("class_list", None),
            image_size=image_size or 224,
            seed=seed,
            pin_memory=pin_memory,
            use_mixup=use_mixup,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            auto_augment=auto_augment,
            rand_augment=rand_augment,
            random_erasing_prob=random_erasing_prob,
            color_jitter=color_jitter,
            allow_mock=cfg.data.get("allow_mock", True),
        )

    else:
        raise ValueError(
            f"Unsupported dataset: '{dataset_name}'. "
            f"Supported options: 'cifar100', 'tiny_imagenet', 'imagenet_subset'."
        )


__all__ = [
    "build_dataloaders",
    "get_transforms",
    "MixupCutmixCollate",
    "DATASET_STATS",
    "get_cifar100_datasets",
    "get_cifar100_dataloaders",
    "TinyImageNetDataset",
    "get_tiny_imagenet_datasets",
    "get_tiny_imagenet_dataloaders",
    "download_and_extract_tiny_imagenet",
    "ImageNetSubsetDataset",
    "get_imagenet_subset_datasets",
    "get_imagenet_subset_dataloaders",
    "generate_mock_imagenet_data",
]
