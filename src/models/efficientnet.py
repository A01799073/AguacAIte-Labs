import torch
import torch.nn as nn
from torchvision import models

def build_efficientnet_b0(pretrained: bool = True, freeze_backbone: bool = True):
    """
    Configure EfficientNet-B0 for binary classification (Normal vs Tuberculosis
    """

    model = models.efficientnet_b0(pretrained=pretrained)

    # DenseNet expect 3-channel input
    model.features[0][0] = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, bias = False)

    # Replace classfier (binary outputs)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features,1)

    # Freeze feature extractor
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False
        
        for param in model.classifier.parameters():
            param.requires_grad = True

    return model