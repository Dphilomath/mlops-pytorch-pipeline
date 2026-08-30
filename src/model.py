import torch
from torchvision import models


def get_model(
    architecture: str = "resnet18",
    num_classes: int = 10,
    strategy: str = "transfer_learning",
):
    """Return a torchvision model adapted for the given training strategy.

    Args:
        architecture: Name of the torchvision model (e.g., "resnet18", "vgg16").
        num_classes: Number of output classes for the classifier.
        strategy: Optimization strategy ("transfer_learning", "scratch", etc.).
    """
    # Map supported architectures to torchvision constructors
    arch_map = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "vgg16": models.vgg16,
    }
    if architecture not in arch_map:
        raise ValueError(
            f"Unsupported architecture '{architecture}'. Supported: {list(arch_map.keys())}"
        )

    # Transfer learning strategies use pre-trained ImageNet weights
    if strategy.startswith("transfer_learning") or strategy in [
        "strategy_2",
        "strategy_3",
    ]:
        weights = "DEFAULT"
    else:
        weights = None

    model = arch_map[architecture](weights=weights)

    # Replace the final fully‑connected layer to match the number of classes (all layers fine-tuned)
    if "resnet" in architecture:
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif "vgg" in architecture:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)

    return model
