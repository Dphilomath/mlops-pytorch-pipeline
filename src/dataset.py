"""Dataset utilities for the MLOps PyTorch pipeline.
Provides a simple `get_dataloaders` function that returns
training and validation DataLoaders with fast mirror support.
"""

import os
import tarfile
import urllib.request
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DEFAULT_MIRRORS = [
    "https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz",
    "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
]


def ensure_cifar10_dataset(data_dir: str):
    """Download and extract CIFAR-10 from available mirrors if not already present."""
    data_path = Path(data_dir)
    extracted_dir = data_path / "cifar-10-batches-py"
    tar_path = data_path / "cifar-10-python.tar.gz"

    # If dataset is already extracted and verified, skip downloading
    if extracted_dir.exists() and any(extracted_dir.iterdir()):
        return

    data_path.mkdir(parents=True, exist_ok=True)

    mirrors = []
    env_mirror = os.getenv("CIFAR10_MIRROR_URL")
    if env_mirror:
        mirrors.append(env_mirror)
    mirrors.extend(DEFAULT_MIRRORS)

    if not tar_path.exists() or tar_path.stat().st_size < 100_000_000:
        download_success = False
        for mirror_url in mirrors:
            try:
                print(f"Downloading CIFAR-10 from: {mirror_url} ...", flush=True)
                req = urllib.request.Request(
                    mirror_url, headers={"User-Agent": "Mozilla/5.0"}
                )
                with urllib.request.urlopen(req, timeout=30) as response, open(
                    tar_path, "wb"
                ) as out_file:
                    total_size = int(response.headers.get("Content-Length", 0))
                    chunk_size = 1024 * 1024  # 1MB chunks
                    downloaded = 0
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        downloaded += len(chunk)
                        if total_size > 0:
                            mb_done = downloaded / (1024 * 1024)
                            mb_total = total_size / (1024 * 1024)
                            pct = (downloaded / total_size) * 100
                            print(
                                f"\rProgress: {mb_done:.1f}/{mb_total:.1f} MB ({pct:.1f}%)",
                                end="",
                                flush=True,
                            )
                print("\nDownload complete!", flush=True)
                download_success = True
                break
            except Exception as e:
                print(
                    f"\nMirror failed ({mirror_url}): {e}. Trying next mirror...",
                    flush=True,
                )
                if tar_path.exists():
                    tar_path.unlink()

        if not download_success:
            raise RuntimeError(
                "Failed to download CIFAR-10 dataset from all configured mirrors."
            )

    if not extracted_dir.exists():
        print(
            f"Extracting dataset archive {tar_path.name} to {data_dir}...", flush=True
        )
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        print("Extraction complete.", flush=True)


def get_transforms(
    train: bool = True, strategy: str = "strategy_2"
) -> transforms.Compose:
    transform_list = []

    # Both transfer learning strategies (strategy_2, strategy_3) resize 32x32 images to 224x224
    if strategy in ["strategy_2", "strategy_3"]:
        transform_list.append(transforms.Resize(224))

    if train:
        if strategy == "strategy_3":
            # Advanced data augmentations for tuned transfer learning (strategy_3)
            transform_list.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ColorJitter(
                        brightness=0.2, contrast=0.2, saturation=0.2
                    ),
                ]
            )
        else:
            # Baseline data augmentation
            transform_list.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(
                        224 if strategy in ["strategy_2", "strategy_3"] else 32,
                        padding=4,
                    ),
                ]
            )

    transform_list.append(transforms.ToTensor())

    if strategy in ["strategy_2", "strategy_3"]:
        # Standard ImageNet normalization for transfer learning models
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    else:
        # Standard CIFAR-10 normalization for baseline training from scratch
        transform_list.append(
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616],
            )
        )

    return transforms.Compose(transform_list)


def get_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    num_workers: int = 0,
    strategy: str = "strategy_2",
) -> tuple[DataLoader, DataLoader]:
    ensure_cifar10_dataset(data_dir)

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=False,
        transform=get_transforms(train=True, strategy=strategy),
    )
    val_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=False,
        transform=get_transforms(train=False, strategy=strategy),
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
