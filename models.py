"""
models.py — Model factory
Hỗ trợ: lenet, mobilenet, resnet18
"""

import torch.nn as nn
import torchvision.models as tv_models


class LeNet(nn.Module):
    """
    LeNet đúng theo iDLG paper gốc:
      - Sigmoid activation
      - hidden=768 cho ảnh 32x32x3
      - hidden=588 cho MNIST 28x28x1
    """
    def __init__(self, channel=3, hidden=768, num_classes=10):
        super().__init__()
        act = nn.Sigmoid
        self.body = nn.Sequential(
            nn.Conv2d(channel, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=2),
            act(),
            nn.Conv2d(12, 12, kernel_size=5, padding=5//2, stride=1),
            act(),
        )
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x):
        out = self.body(x)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def get_model(name: str, num_classes: int = 10,
              img_size: int = 32, channel: int = 3) -> nn.Module:
    """
    Trả về model với output dimension đúng.

    Args:
        name        : 'lenet' | 'mobilenet' | 'resnet18'
        num_classes : số class đầu ra
        img_size    : kích thước ảnh đầu vào (dùng để tính hidden cho LeNet)
    """
    name = name.lower().replace("-", "").replace("_", "")

    if name == "lenet":
        # hidden = 12 * (img_size//4) * (img_size//4)
        # img_size=28(MNIST):7x7=588, img_size=32:8x8=768, img_size=64:16x16=3072
        h = img_size // 4
        hidden = 12 * h * h
        model = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
        # Weight init đúng theo iDLG paper gốc
        for m in model.modules():
            try:
                if hasattr(m, "weight"):
                    m.weight.data.uniform_(-0.5, 0.5)
            except Exception:
                pass
            try:
                if hasattr(m, "bias"):
                    m.bias.data.uniform_(-0.5, 0.5)
            except Exception:
                pass

    elif name in ("mobilenet", "mobilenetv2"):
        model = tv_models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif name == "resnet18":
        model = tv_models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(
            f"Unknown model '{name}'. Choose from: lenet, mobilenet, resnet18")

    return model