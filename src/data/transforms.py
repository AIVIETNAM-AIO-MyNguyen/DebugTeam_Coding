"""
Data Transforms and Augmentation Pipeline.

Supports:
- Standard dataset normalizations (CIFAR-100, Tiny-ImageNet, ImageNet)
- AutoAugment / RandAugment
- RandomErasing / Cutout
- Batch-level Mixup and CutMix collation for MLP training
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
import torchvision.transforms as T


# Standard dataset mean and std statistics
DATASET_STATS = {
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "native_size": (32, 32),
    },
    "tiny_imagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "native_size": (64, 64),
    },
    "imagenet_subset": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "native_size": (224, 224),
    },
    "imagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "native_size": (224, 224),
    },
}


def get_transforms(
    dataset_name: str,
    image_size: Optional[Union[int, Tuple[int, int]]] = None,
    is_train: bool = True,
    auto_augment: bool = False,
    rand_augment: bool = False,
    random_erasing_prob: float = 0.0,
    color_jitter: float = 0.0,
) -> T.Compose:
    """
    Builds torchvision transform pipeline for train or evaluation.

    Args:
        dataset_name: One of 'cifar100', 'tiny_imagenet', 'imagenet_subset'.
        image_size: Target image resolution. If None, uses native dataset resolution.
        is_train: True for training transforms (with augmentations), False for validation/test.
        auto_augment: Whether to apply AutoAugment.
        rand_augment: Whether to apply RandAugment.
        random_erasing_prob: Probability of RandomErasing (Cutout).
        color_jitter: Strength of ColorJitter (brightness, contrast, saturation).

    Returns:
        torchvision.transforms.Compose instance.
    """
    key = dataset_name.lower().replace("-", "_")
    stats = DATASET_STATS.get(key, DATASET_STATS["imagenet"])
    mean = stats["mean"]
    std = stats["std"]
    native_size = stats["native_size"]

    if image_size is None:
        target_size = native_size
    elif isinstance(image_size, int):
        target_size = (image_size, image_size)
    else:
        target_size = image_size

    transform_list = []

    if is_train:
        # Spatial augmentations based on dataset scale
        if target_size == (32, 32):
            transform_list.extend([
                T.RandomCrop(32, padding=4, padding_mode="reflect"),
                T.RandomHorizontalFlip(p=0.5),
            ])
        elif target_size == (64, 64):
            transform_list.extend([
                T.RandomCrop(64, padding=8, padding_mode="reflect"),
                T.RandomHorizontalFlip(p=0.5),
            ])
        else:
            # Standard ImageNet-style random resized crop
            transform_list.extend([
                T.RandomResizedCrop(target_size, scale=(0.08, 1.0), interpolation=T.InterpolationMode.BICUBIC),
                T.RandomHorizontalFlip(p=0.5),
            ])

        # Color Jitter
        if color_jitter > 0.0:
            transform_list.append(
                T.ColorJitter(
                    brightness=color_jitter,
                    contrast=color_jitter,
                    saturation=color_jitter,
                )
            )

        # Advanced Augmentations
        if rand_augment:
            transform_list.append(T.RandAugment(num_ops=2, magnitude=9))
        elif auto_augment:
            if "cifar" in key:
                transform_list.append(T.AutoAugment(T.AutoAugmentPolicy.CIFAR10))
            else:
                transform_list.append(T.AutoAugment(T.AutoAugmentPolicy.IMAGENET))

        transform_list.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

        # Random Erasing (Cutout)
        if random_erasing_prob > 0.0:
            transform_list.append(
                T.RandomErasing(p=random_erasing_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3), value="random")
            )

    else:
        # Evaluation / Validation transform
        if target_size == (32, 32) or target_size == (64, 64):
            if target_size != native_size:
                transform_list.append(T.Resize(target_size, interpolation=T.InterpolationMode.BICUBIC))
        else:
            # For 224x224 (e.g. ImageNet scale), resize slightly larger then center crop
            scale_size = int(target_size[0] / 0.875)
            transform_list.extend([
                T.Resize((scale_size, scale_size), interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(target_size),
            ])

        transform_list.extend([
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    return T.Compose(transform_list)


class MixupCutmixCollate:
    """
    Collate function implementing Mixup and CutMix data augmentation.
    Commonly essential for training MLP vision architectures (like MLP-Mixer and ResMLP).

    Converts integer targets into one-hot smoothed/mixed target distributions.
    """

    def __init__(
        self,
        num_classes: int,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 1.0,
        label_smoothing: float = 0.1,
    ):
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.label_smoothing = label_smoothing

    def _one_hot(self, target: torch.Tensor) -> torch.Tensor:
        y = torch.zeros(target.size(0), self.num_classes, device=target.device)
        y.scatter_(1, target.unsqueeze(1), 1.0)
        if self.label_smoothing > 0.0:
            y = y * (1.0 - self.label_smoothing) + self.label_smoothing / self.num_classes
        return y

    def _rand_bbox(self, size: Tuple[int, ...], lam: float) -> Tuple[int, int, int, int]:
        w, h = size[2], size[3]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)

        cx = np.random.randint(w)
        cy = np.random.randint(h)

        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)

        return bbx1, bby1, bbx2, bby2

    def __call__(self, batch: list) -> Tuple[torch.Tensor, torch.Tensor]:
        images = torch.stack([item[0] for item in batch])
        targets = torch.tensor([item[1] for item in batch], dtype=torch.long)

        one_hot_targets = self._one_hot(targets)

        if np.random.rand() > self.prob:
            return images, one_hot_targets

        lam = 1.0
        use_cutmix = False

        if self.mixup_alpha > 0 and self.cutmix_alpha > 0:
            use_cutmix = np.random.rand() > 0.5

        batch_size = images.size(0)
        rand_index = torch.randperm(batch_size)

        if use_cutmix:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
            images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
            lam = 1.0 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        elif self.mixup_alpha > 0:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            images = lam * images + (1.0 - lam) * images[rand_index]

        mixed_targets = lam * one_hot_targets + (1.0 - lam) * one_hot_targets[rand_index]
        return images, mixed_targets
