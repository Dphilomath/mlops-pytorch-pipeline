# src/dataset.py
"""Dataset utilities for the MLOps PyTorch pipeline.
Provides a simple `get_dataloaders` function that returns
training and validation DataLoaders.
The implementation uses torchvision's CIFAR‑10 dataset as a
placeholder – replace with your own data source as needed.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from pathlib import Path

def get_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 64,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    """Create training and validation dataloaders.

    Args:
        data_dir (str): Directory where the dataset will be downloaded.
        batch_size (int): Batch size for both loaders.
        num_workers (int): Number of subprocesses for data loading.
        pin_memory (bool): Whether to pin memory (recommended for CUDA).
    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, val_loader)
    """
    # Simple transforms – normalize CIFAR‑10 images
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                             std=[0.2470, 0.2435, 0.2616]),
    ])

    # Download CIFAR‑10 (or use existing) into the provided directory
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    # Split into training and validation (90/10 split)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
