"""Dataset utilities for the MLOps PyTorch pipeline.
Provides a simple `get_dataloaders` function that returns
training and validation DataLoaders.
The implementation uses torchvision's CIFAR‑10 dataset as a
placeholder - replace with your own data source as needed.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616],
                ),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            ),
        ]
    )


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 1,
) -> tuple[DataLoader, DataLoader]:
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=get_transforms(train=True),
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=get_transforms(train=False),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        multiprocessing_context="spawn" if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
