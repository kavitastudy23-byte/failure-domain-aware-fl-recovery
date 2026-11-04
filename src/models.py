import torch
from torch import nn


class CifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class FemnistCNN(nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(64 * 7 * 7, 2048), nn.ReLU(), nn.Linear(2048, num_classes)
        )

    def forward(self, x):
        return self.net(x)
