import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import EfficientNet_B0_Weights

def build_efficientnet_b0(pretrained: bool = True, freeze_backbone: bool = True):
    """
    Build an EfficientNet-B0 model adapted for binary classification
    of chest X-ray images (Normal vs Tuberculosis).
    """
    weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # Replace first convolution to accept 1-channel (grayscale) input
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        in_channels = 1,
        out_channels = first_conv.out_channels,
        kernel_size = first_conv.kernel_size,
        stride = first_conv.stride,
        padding = first_conv.padding,
        bias = False
    )
    
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)

    # Freeze feature extractor
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

        # Ensure trainability of newly added layers
        for param in model.features[0][0].parameters():
            param.requires_grad = True

        for param in model.classifier.parameters():
            param.requires_grad = True

    return model