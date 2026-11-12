"""Image-pair classifier shared by skip-gram training and milestone extraction."""

from pathlib import Path

import numpy as np
import torch
from torch import nn


def frames_to_tensor(frames, device="cpu"):
    """Convert uint8 RGB [B,84,84,3] observations to float [B,3,84,84]."""
    frames = np.asarray(frames)
    if frames.ndim != 4 or frames.shape[1:] != (84, 84, 3):
        raise ValueError(f"Expected [B,84,84,3] RGB frames, got {frames.shape}")
    if frames.dtype != np.uint8:
        raise ValueError(f"Expected uint8 frames in [0,255], got {frames.dtype}")
    frames = np.ascontiguousarray(frames)
    if not frames.flags.writeable:
        frames = frames.copy()
    return torch.from_numpy(frames).to(
        device=device, dtype=torch.float32
    ).permute(0, 3, 1, 2).contiguous().div_(255.0)


class BasicBlock(nn.Module):
    """Standard two-convolution residual block for the optional ResNet-18."""

    def __init__(self, in_channels, channels, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.skip = nn.Identity() if stride == 1 and in_channels == channels else nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, stride, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return torch.relu(self.main(x) + self.skip(x))


def make_encoder(architecture, embedding_dim):
    if architecture == "small":
        # Keep spatial information: avatar displacement matters within a room.
        return nn.Sequential(
            nn.Conv2d(3, 16, 5, 2, 2), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(64 * 4 * 4, embedding_dim),
        )
    if architecture == "resnet18":
        layers = [
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1),
        ]
        previous = 64
        for channels in (64, 128, 256, 512):
            layers.extend((BasicBlock(previous, channels, 1 if channels == 64 else 2),
                           BasicBlock(channels, channels)))
            previous = channels
        layers.extend((nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(512, embedding_dim)))
        return nn.Sequential(*layers)
    raise ValueError(f"Unknown architecture: {architecture}")


class PairClassifier(nn.Module):
    """Shared visual encoder and symmetric binary similarity head (logits)."""

    def __init__(self, architecture="small", embedding_dim=128):
        super().__init__()
        self.model_config = {"architecture": architecture, "embedding_dim": embedding_dim}
        self.encoder = make_encoder(architecture, embedding_dim)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128), nn.ReLU(inplace=True), nn.Linear(128, 1)
        )

    def encode(self, frames):
        return self.encoder(frames)

    def pair_logits(self, embedding_a, embedding_b):
        features = torch.cat(((embedding_a - embedding_b).abs(), embedding_a * embedding_b), -1)
        return self.head(features).squeeze(-1)

    def forward(self, frame_a, frame_b):
        # A single encoder call also treats both branches equally in BatchNorm.
        embeddings = self.encode(torch.cat((frame_a, frame_b), 0))
        return self.pair_logits(*embeddings.chunk(2, 0))


def load_classifier(path, device="cpu"):
    """Load a trained checkpoint, returning an evaluation model and its metadata."""
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported milestone classifier checkpoint format")
    model = PairClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint
