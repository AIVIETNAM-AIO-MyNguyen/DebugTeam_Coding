"""
Unit tests for data pipeline and transforms.
"""

from pathlib import Path
from PIL import Image
import pytest
import torch
from src.data.transforms import MixupCutmixCollate, get_transforms
from src.data.imagenet_subset import generate_mock_imagenet_data, ImageNetSubsetDataset


def test_transforms_cifar():
    transform = get_transforms("cifar100", image_size=32, is_train=True)
    img = Image.new("RGB", (32, 32), color=(100, 150, 200))
    tensor = transform(img)
    assert tensor.shape == (3, 32, 32)
    assert isinstance(tensor, torch.Tensor)


def test_mixup_cutmix_collate():
    collate = MixupCutmixCollate(num_classes=10, mixup_alpha=0.8, cutmix_alpha=1.0)
    batch = [
        (torch.randn(3, 32, 32), 1),
        (torch.randn(3, 32, 32), 4),
        (torch.randn(3, 32, 32), 7),
        (torch.randn(3, 32, 32), 2),
    ]
    images, targets = collate(batch)
    assert images.shape == (4, 3, 32, 32)
    assert targets.shape == (4, 10)  # smoothed / mixed one-hot distributions


def test_mock_imagenet_generation_and_loading(tmp_path: Path):
    mock_dir = generate_mock_imagenet_data(
        root_dir=tmp_path,
        num_classes=3,
        train_samples_per_class=4,
        val_samples_per_class=2,
        image_size=64,
    )
    dataset = ImageNetSubsetDataset(
        root=mock_dir,
        split="train",
        num_classes=3,
        allow_mock=False,
    )
    assert len(dataset) == 12  # 3 classes * 4 images
    img, target = dataset[0]
    assert target in (0, 1, 2)
