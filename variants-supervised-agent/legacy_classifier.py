"""CNN architectures for the separate BC policy and version-1 classifier checkpoints."""

import torch
from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False), nn.BatchNorm2d(channels), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False), nn.BatchNorm2d(channels))
        self.skip = nn.Identity() if stride == 1 and in_channels == channels else nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride, bias=False), nn.BatchNorm2d(channels))

    def forward(self, value):
        return torch.relu(self.main(value) + self.skip(value))


def make_encoder(architecture, embedding_dim):
    if architecture == "small":
        return nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(64 * 4 * 4, embedding_dim))
    if architecture != "resnet18":
        raise ValueError("Architecture must be resnet18 or small")
    layers = [nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1)]
    previous = 64
    for channels in (64, 128, 256, 512):
        layers.extend((BasicBlock(previous, channels, 1 if channels == 64 else 2), BasicBlock(channels, channels)))
        previous = channels
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, embedding_dim))
