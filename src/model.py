import torch
from torchvision import models

def get_model(architecture: str = "resnet18", num_classes: int = 10):
    """Return a torchvision model with the given architecture and number of classes.

    Args:
        architecture: Name of the torchvision model (e.g., "resnet18", "vgg16").
        num_classes: Number of output classes for the classifier.
    """
    # Map supported architectures to torchvision constructors
    arch_map = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "vgg16": models.vgg16,
    }
    if architecture not in arch_map:
        raise ValueError(f"Unsupported architecture '{architecture}'. Supported: {list(arch_map.keys())}")
    model = arch_map[architecture](weights=None)
    # Replace the final fully‑connected layer to match the number of classes
    if "resnet" in architecture:
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
    elif "vgg" in architecture:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
    return model
