"""
Dataset Download Helper Script.

Usage:
    python scripts/download_data.py --dataset cifar100
    python scripts/download_data.py --dataset tiny_imagenet
    python scripts/download_data.py --dataset all
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from torchvision.datasets import CIFAR100
from src.data.tiny_imagenet import download_and_extract_tiny_imagenet
from src.data.imagenet_subset import generate_mock_imagenet_data


def parse_args():
    parser = argparse.ArgumentParser(description="Download vision datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["cifar100", "tiny_imagenet", "mock_imagenet", "all"],
        help="Dataset to download",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Root directory to store datasets",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("cifar100", "all"):
        print("\n--- Downloading CIFAR-100 ---")
        cifar_dir = data_dir / "cifar100"
        CIFAR100(root=str(cifar_dir), train=True, download=True)
        CIFAR100(root=str(cifar_dir), train=False, download=True)
        print(f"CIFAR-100 downloaded to {cifar_dir}")

    if args.dataset in ("tiny_imagenet", "all"):
        print("\n--- Downloading Tiny-ImageNet-200 ---")
        tiny_dir = data_dir / "tiny_imagenet"
        download_and_extract_tiny_imagenet(tiny_dir)
        print(f"Tiny-ImageNet-200 ready at {tiny_dir}")

    if args.dataset in ("mock_imagenet", "all"):
        print("\n--- Generating Mock ImageNet Subset (for testing) ---")
        mock_dir = generate_mock_imagenet_data(data_dir / "imagenet", num_classes=10)
        print(f"Mock ImageNet ready at {mock_dir}")

    print("\nDataset preparation complete!")


if __name__ == "__main__":
    main()
