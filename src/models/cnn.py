import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    """
    Simple convolutional neural network for binary classification
    of chest X-ray images (Tuberculosis vs Normal)
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Convolutional feature extractor
        self.features = nn.Sequential(

            # First conv block
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 122 x 122

            # Second conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> 61 x 61

            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)   # -> 30 x 30
        )

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 30 * 30, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)  # Binary output (logits)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x