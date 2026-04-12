"""
models.py — Model factory
Supports: mobilenet, resnet18
"""

import torch.nn as nn
import torchvision.models as tv_models


def get_model(name: str, num_classes: int = 10) -> nn.Module:
    """
    Return a model with the correct output dimension.

    Args:
        name        : 'mobilenet' | 'resnet18'
        num_classes : number of output classes
    Returns:
        nn.Module (weights randomly initialized)
    """
    name = name.lower().replace("-", "").replace("_", "")

    if name == "mobilenet" or name == "mobilenetv2":
        model = tv_models.mobilenet_v2(weights=None)
        # Replace classifier head
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif name == "resnet18":
        model = tv_models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: mobilenet, resnet18")

    return model
