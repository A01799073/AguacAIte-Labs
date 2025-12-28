import torch
import torch.nn as nn
from torchvision import models

def build_densenet121(pretrained: bool = True, freeze_backbone: bool = True):
    """
    Configure Restnet50 for binary classification (Normal vs Tuberculosis
    """

    model = models.resnet50(pretrained=pretrained)

    # DenseNet expect 3-channel input
    model.features.conv1 = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = 7, stride = 2, padding = 3, bias = False)

    # Replace classfier (binary outputs)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features,1)

    # Freeze feature extractor
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        
        #Always keeps classifier trainable
        for param in model.fc.parameters():
            param.requires_grad = True

    return model