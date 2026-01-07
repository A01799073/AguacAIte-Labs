import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import DenseNet121_Weights

def build_densenet121(pretrained: bool = True, freeze_backbone: bool = True):
    """
    Build a DenseNet121 model adapted for binary classification
    of chest X-ray images (Normal vs Tuberculosis).
    """
    weights = DenseNet121_Weights.DEFAULT if pretrained else None
    model = models.densenet121(weights=weights)

    # DenseNet expect 3-channel input
    model.features.conv0 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)

    # Replace classfier for binary outputs
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features,1)

    # Freeze feature extractor
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        
        # Ensure the new first conv layer is trainable
        for param in model.features.conv0.parameters():
            param.requires_grad = True

    return model