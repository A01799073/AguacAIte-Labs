import torch
import torch.nn as nn
from torchvision import models

def build_densenet121(pretrained: bool = True, freeze_backbone: bool = True):
    """
    Configure DenseNet121 for binary classification (Normal vs Tuberculosis
    """

    model = models.densenet121(pretrained=pretrained)

    # DenseNet expect 3-channel input
    model.features.conv0 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)

    # Replace classfier (binary outputs)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features,1)

    # Freeze feature extractor
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    return model