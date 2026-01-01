import torch
import torch.nn as nn
from torchvision import models

def build_efficientnet_b0(pretrained: bool = True, freeze_backbone: bool = True):
    """
    Configure EfficientNet-B0 for binary classification (Normal vs Tuberculosis
    """

    model = models.efficientnet_b0(pretrained=pretrained)

    # EfficientNet first conv is usually features[0][0]
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels=1,
        out_channels=first_conv.out_channels,
        kernel_size=first_conv.kernel_size,
        stride=first_conv.stride,
        padding=first_conv.padding,
        bias=False
    )
    # Replace classfier (binary outputs)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    # Freeze feature extractor
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model