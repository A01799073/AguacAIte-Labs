import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import ResNet50_Weights


def build_resnet50(pretrained: bool = True, freeze_backbone: bool = True):
    """
    Configure Restnet50 for binary classification (Normal vs Tuberculosis
    """

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # ResNet uses conv1 directly
    model.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)

    # Replace classfier (binary outputs)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features,1)

    # Freeze feature extractor
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        # Ensure newly added layers remain trainable
        for param in model.conv1.parameters():
            param.requires_grad = True

        for param in model.fc.parameters():
            param.requires_grad = True

    return model