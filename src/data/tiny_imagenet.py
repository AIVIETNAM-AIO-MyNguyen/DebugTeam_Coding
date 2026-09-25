"""
Tiny-ImageNet-200 Dataset Pipeline.

200 classes, 100,000 train images (500 per class), 10,000 validation images (50 per class).
Image resolution: 64x64.
Stanford CS231n benchmark: http://cs231n.stanford.edu/tiny-imagenet-200.zip (~248MB).

Provides automatic download, zip extraction, val_annotations parsing, and reproducible DataLoaders.
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.data.transforms import MixupCutmixCollate, get_transforms
from src.utils.seed import get_generator, seed_worker

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


class _DownloadProgressBar(tqdm):
    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None) -> None:
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_and_extract_tiny_imagenet(data_dir: Union[str, Path]) -> Path:
    """
    Downloads and extracts Tiny-ImageNet-200 zip if not already present.

    Args:
        data_dir: Directory where tiny-imagenet-200 folder should live.

    Returns:
        Path to the root `tiny-imagenet-200` directory.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    extracted_dir = data_dir / "tiny-imagenet-200"
    if (extracted_dir / "train").is_dir() and (extracted_dir / "val").is_dir():
        return extracted_dir

    # Check if zip exists
    zip_path = data_dir / "tiny-imagenet-200.zip"
    if not zip_path.is_file():
        print(f"Downloading Tiny-ImageNet from {TINY_IMAGENET_URL}...")
        with _DownloadProgressBar(unit="B", unit_scale=True, miniters=1, desc="tiny-imagenet-200.zip") as t:
            urllib.request.urlretrieve(TINY_IMAGENET_URL, filename=zip_path, reporthook=t.update_to)

    print("Extracting tiny-imagenet-200.zip...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_dir)

    print(f"Tiny-ImageNet successfully extracted to {extracted_dir}")
    return extracted_dir


class TinyImageNetDataset(Dataset):
    """
    PyTorch Dataset for Tiny-ImageNet-200.
    Directly handles the `val_annotations.txt` structure without requiring manual file rearrangement.
    """

    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Any] = None,
        download: bool = True,
    ):
        """
        Args:
            root: Root path to datasets directory (containing 'tiny-imagenet-200').
            split: One of 'train', 'val', or 'test'.
            transform: torchvision transforms.
            download: Whether to download if not present.
        """
        assert split in ("train", "val", "test"), f"Invalid split: {split}"
        self.split = split
        self.transform = transform

        root_path = Path(root)
        if not (root_path / "tiny-imagenet-200").is_dir() and not (root_path / "train").is_dir():
            if download:
                self.dataset_root = download_and_extract_tiny_imagenet(root_path)
            else:
                raise FileNotFoundError(
                    f"Tiny-ImageNet not found at {root_path}. Set download=True to download automatically."
                )
        else:
            self.dataset_root = (root_path / "tiny-imagenet-200") if (root_path / "tiny-imagenet-200").is_dir() else root_path

        # 1. Load class list (wnids.txt)
        wnids_file = self.dataset_root / "wnids.txt"
        with open(wnids_file, "r", encoding="utf-8") as f:
            self.classes: List[str] = [line.strip() for line in f if line.strip()]

        self.class_to_idx: Dict[str, int] = {wnid: i for i, wnid in enumerate(self.classes)}

        # 2. Load human-readable names (words.txt)
        self.class_names: Dict[str, str] = {}
        words_file = self.dataset_root / "words.txt"
        if words_file.is_file():
            with open(words_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t", 1)
                    if len(parts) == 2 and parts[0] in self.class_to_idx:
                        self.class_names[parts[0]] = parts[1]

        # 3. Load sample paths and targets
        self.samples: List[Tuple[Path, int]] = []
        if self.split == "train":
            train_dir = self.dataset_root / "train"
            for wnid in self.classes:
                class_idx = self.class_to_idx[wnid]
                img_dir = train_dir / wnid / "images"
                if img_dir.is_dir():
                    for img_file in img_dir.glob("*.JPEG"):
                        self.samples.append((img_file, class_idx))

        elif self.split == "val":
            val_dir = self.dataset_root / "val"
            annot_file = val_dir / "val_annotations.txt"
            with open(annot_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    img_name, wnid = parts[0], parts[1]
                    if wnid in self.class_to_idx:
                        img_path = val_dir / "images" / img_name
                        self.samples.append((img_path, self.class_to_idx[wnid]))

        elif self.split == "test":
            # For testing, standard benchmark uses the validation set since CS231n test set is unlabelled
            val_dir = self.dataset_root / "val"
            annot_file = val_dir / "val_annotations.txt"
            with open(annot_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    img_name, wnid = parts[0], parts[1]
                    if wnid in self.class_to_idx:
                        img_path = val_dir / "images" / img_name
                        self.samples.append((img_path, self.class_to_idx[wnid]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img_path, target = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


def get_tiny_imagenet_datasets(
    data_dir: Union[str, Path] = "data/tiny_imagenet",
    image_size: Optional[int] = None,
    seed: int = 42,
    auto_augment: bool = False,
    rand_augment: bool = False,
    random_erasing_prob: float = 0.0,
    color_jitter: float = 0.0,
    download: bool = True,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Constructs Tiny-ImageNet train, validation, and test datasets.
    """
    train_transform = get_transforms(
        dataset_name="tiny_imagenet",
        image_size=image_size,
        is_train=True,
        auto_augment=auto_augment,
        rand_augment=rand_augment,
        random_erasing_prob=random_erasing_prob,
        color_jitter=color_jitter,
    )

    eval_transform = get_transforms(
        dataset_name="tiny_imagenet",
        image_size=image_size,
        is_train=False,
    )

    train_ds = TinyImageNetDataset(root=data_dir, split="train", transform=train_transform, download=download)
    val_ds = TinyImageNetDataset(root=data_dir, split="val", transform=eval_transform, download=download)
    test_ds = TinyImageNetDataset(root=data_dir, split="test", transform=eval_transform, download=download)

    return train_ds, val_ds, test_ds


def get_tiny_imagenet_dataloaders(
    data_dir: Union[str, Path] = "data/tiny_imagenet",
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: Optional[int] = None,
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
    Constructs DataLoaders for Tiny-ImageNet-200.
    """
    train_ds, val_ds, test_ds = get_tiny_imagenet_datasets(
        data_dir=data_dir,
        image_size=image_size,
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
            num_classes=200,
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
