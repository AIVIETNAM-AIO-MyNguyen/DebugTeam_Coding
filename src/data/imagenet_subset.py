"""
ImageNet Subset Pipeline (e.g., ImageNet-100).

Supports:
- Loading a subset of N classes from standard ImageNet (ILSVRC) directory structure.
- Deterministic class selection with seed or custom class list file.
- Remapping selected class synsets to continuous 0..(N-1) labels.
- Mock/synthetic dataset generation for fast smoke testing without 150GB download.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.transforms import MixupCutmixCollate, get_transforms
from src.utils.seed import get_generator, seed_worker


def generate_mock_imagenet_data(
    root_dir: Union[str, Path],
    num_classes: int = 10,
    train_samples_per_class: int = 20,
    val_samples_per_class: int = 5,
    image_size: int = 224,
) -> Path:
    """
    Generates a lightweight mock ImageNet folder structure for testing pipelines.
    Creates colored synthetic images.
    """
    root_dir = Path(root_dir) / "mock_imagenet"
    if (root_dir / "train").is_dir() and (root_dir / "val").is_dir():
        return root_dir

    print(f"Creating mock ImageNet dataset at {root_dir} ({num_classes} classes)...")
    for split, count in [("train", train_samples_per_class), ("val", val_samples_per_class)]:
        for c in range(num_classes):
            class_dir = root_dir / split / f"n{c:08d}"
            class_dir.mkdir(parents=True, exist_ok=True)
            color = (
                (c * 23) % 255,
                (c * 57) % 255,
                (c * 89) % 255,
            )
            for s in range(count):
                img = Image.new("RGB", (image_size, image_size), color=color)
                img.save(class_dir / f"img_{s:04d}.JPEG")

    return root_dir


class ImageNetSubsetDataset(Dataset):
    """
    Dataset that loads a subset of classes from an ImageNet-style directory.
    Directory layout expected:
        root/
            train/
                <class_id>/
                    *.JPEG
            val/
                <class_id>/
                    *.JPEG
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        num_classes: int = 100,
        class_list: Optional[List[str]] = None,
        seed: int = 42,
        transform: Optional[Any] = None,
        allow_mock: bool = True,
    ):
        """
        Args:
            root: Root path containing 'train' and 'val' subdirectories.
            split: 'train' or 'val'.
            num_classes: Number of classes to include in subset (e.g. 100 for ImageNet-100).
            class_list: Explicit list of class folder names. If None, deterministically samples num_classes.
            seed: Seed for sampling classes if class_list is None.
            transform: torchvision transforms.
            allow_mock: If True and dataset directory is empty or missing, generates a mock dataset.
        """
        assert split in ("train", "val"), f"Split must be 'train' or 'val', got {split}"
        self.split = split
        self.transform = transform
        self.root = Path(root)

        # Check if directory exists and has train/val
        if not (self.root / split).is_dir():
            if allow_mock:
                print(f"ImageNet directory '{self.root}' not found. Generating mock dataset for testing...")
                self.root = generate_mock_imagenet_data(self.root.parent, num_classes=min(num_classes, 10))
            else:
                raise FileNotFoundError(f"ImageNet split directory not found at: {self.root / split}")

        split_dir = self.root / split

        # Discover all available class folders
        all_class_dirs = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        if not all_class_dirs:
            if allow_mock:
                self.root = generate_mock_imagenet_data(self.root.parent, num_classes=min(num_classes, 10))
                split_dir = self.root / split
                all_class_dirs = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
            else:
                raise RuntimeError(f"No class folders found in {split_dir}")

        # Determine subset classes
        if class_list is not None:
            self.classes = [c for c in class_list if c in all_class_dirs]
        elif num_classes is not None and num_classes < len(all_class_dirs):
            rng = np.random.RandomState(seed)
            selected_indices = sorted(rng.choice(len(all_class_dirs), size=num_classes, replace=False))
            self.classes = [all_class_dirs[i] for i in selected_indices]
        else:
            self.classes = all_class_dirs

        self.class_to_idx: Dict[str, int] = {c: i for i, c in enumerate(self.classes)}

        # Collect image paths and assigned target indices
        self.samples: List[Tuple[Path, int]] = []
        valid_exts = {".jpeg", ".jpg", ".png"}
        for class_name in self.classes:
            target_idx = self.class_to_idx[class_name]
            class_folder = split_dir / class_name
            if class_folder.is_dir():
                for img_path in class_folder.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in valid_exts:
                        self.samples.append((img_path, target_idx))

        # Sort samples for consistency
        self.samples.sort(key=lambda x: str(x[0]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, target = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_imagenet_subset_datasets(
    data_dir: Union[str, Path] = "data/imagenet",
    num_classes: int = 100,
    class_list: Optional[List[str]] = None,
    image_size: int = 224,
    seed: int = 42,
    auto_augment: bool = False,
    rand_augment: bool = False,
    random_erasing_prob: float = 0.0,
    color_jitter: float = 0.0,
    allow_mock: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Constructs train, validation, and test datasets for ImageNet subset.
    """
    train_transform = get_transforms(
        dataset_name="imagenet_subset",
        image_size=image_size,
        is_train=True,
        auto_augment=auto_augment,
        rand_augment=rand_augment,
        random_erasing_prob=random_erasing_prob,
        color_jitter=color_jitter,
    )

    eval_transform = get_transforms(
        dataset_name="imagenet_subset",
        image_size=image_size,
        is_train=False,
    )

    train_ds = ImageNetSubsetDataset(
        root=data_dir,
        split="train",
        num_classes=num_classes,
        class_list=class_list,
        seed=seed,
        transform=train_transform,
        allow_mock=allow_mock,
    )

    # Use the same class subset for validation
    val_ds = ImageNetSubsetDataset(
        root=data_dir,
        split="val",
        num_classes=num_classes,
        class_list=train_ds.classes,
        seed=seed,
        transform=eval_transform,
        allow_mock=allow_mock,
    )

    # Test set points to val set
    test_ds = val_ds

    return train_ds, val_ds, test_ds


def get_imagenet_subset_dataloaders(
    data_dir: Union[str, Path] = "data/imagenet",
    batch_size: int = 64,
    num_workers: int = 4,
    num_classes: int = 100,
    class_list: Optional[List[str]] = None,
    image_size: int = 224,
    seed: int = 42,
    pin_memory: bool = True,
    use_mixup: bool = False,
    mixup_alpha: float = 0.8,
    cutmix_alpha: float = 1.0,
    auto_augment: bool = False,
    rand_augment: bool = False,
    random_erasing_prob: float = 0.0,
    color_jitter: float = 0.0,
    allow_mock: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs DataLoaders for ImageNet subset.
    """
    train_ds, val_ds, test_ds = get_imagenet_subset_datasets(
        data_dir=data_dir,
        num_classes=num_classes,
        class_list=class_list,
        image_size=image_size,
        seed=seed,
        auto_augment=auto_augment,
        rand_augment=rand_augment,
        random_erasing_prob=random_erasing_prob,
        color_jitter=color_jitter,
        allow_mock=allow_mock,
    )

    actual_num_classes = len(train_ds.classes)
    collate_fn = None
    if use_mixup:
        collate_fn = MixupCutmixCollate(
            num_classes=actual_num_classes,
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

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
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
        worker_init_fn=seed_worker,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
