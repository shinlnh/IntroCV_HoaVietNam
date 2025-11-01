from pathlib import Path
from typing import Tuple, List

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGENET_STATS = {
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}


def build_transforms(img_size: int, augment_cfg: dict) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms: List = [
        transforms.Resize(256),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
    ]
    rotation = augment_cfg.get("rotation", 0)
    if rotation:
        train_tfms.append(transforms.RandomRotation(rotation))
    if augment_cfg.get("use_randaugment", False):
        train_tfms.append(transforms.RandAugment())
    else:
        jitter = augment_cfg.get("jitter")
        if jitter:
            train_tfms.append(transforms.ColorJitter(*jitter))
    train_tfms.extend([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_STATS["mean"], IMAGENET_STATS["std"]),
    ])

    test_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_STATS["mean"], IMAGENET_STATS["std"]),
    ])
    return transforms.Compose(train_tfms), test_tfms


def build_dataloaders(
    root: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    augment_cfg: dict,
):
    train_tfms, test_tfms = build_transforms(img_size, augment_cfg)
    root_path = Path(root)
    train_ds = datasets.ImageFolder(root_path / "train", transform=train_tfms)
    test_ds = datasets.ImageFolder(root_path / "test", transform=test_tfms)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, test_loader, train_ds.classes
