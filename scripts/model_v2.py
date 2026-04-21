"""
Improved from-scratch CNN for cat breed recognition.

Key upgrades over CatDogCNN (model v1):
  - Residual blocks: skip connections let gradients flow through deeper networks
    and let the model learn "what to add" rather than "what to output"
  - Global Average Pooling: instead of flattening 256×8×8=16k values,
    average each feature map to one number — far less overfitting
  - Deeper: more residual blocks = more capacity without degradation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Two conv layers with a skip connection around them.

    If in_channels != out_channels (or stride > 1), the shortcut is projected
    via a 1×1 conv so the shapes match before adding.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3,
                               stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Shortcut projection when dimensions change
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class CatCNNv2(nn.Module):
    """
    Architecture overview (128×128 input):

    Stem:    Conv(3→64, 3×3) → BN → ReLU                    → 128×128×64
    Stage 1: ResBlock(64→64)  × 2  + stride-2 ResBlock       →  64×64×64
    Stage 2: ResBlock(64→128) × 2  + stride-2 ResBlock       →  32×32×128
    Stage 3: ResBlock(128→256)× 2  + stride-2 ResBlock       →  16×16×256
    Stage 4: ResBlock(256→512)× 2  + stride-2 ResBlock       →   8×8×512
    GAP:     AdaptiveAvgPool(1×1) → Flatten                  →  512
    Head:    Dropout → Linear(512, num_classes)
    """
    def __init__(self, num_classes):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.stage1 = nn.Sequential(
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64, stride=2),    # 128→64
        )
        self.stage2 = nn.Sequential(
            ResBlock(64, 128),
            ResBlock(128, 128),
            ResBlock(128, 128, stride=2),  # 64→32
        )
        self.stage3 = nn.Sequential(
            ResBlock(128, 256),
            ResBlock(256, 256),
            ResBlock(256, 256, stride=2),  # 32→16
        )
        self.stage4 = nn.Sequential(
            ResBlock(256, 512),
            ResBlock(512, 512),
            ResBlock(512, 512, stride=2),  # 16→8
        )

        # Global Average Pooling: collapses 8×8 spatial dims to 1×1
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        return self.classifier(x)


def mixup_batch(images, labels, num_classes, alpha=0.3):
    """Blend pairs of images and return soft labels.

    alpha controls how strongly the two classes mix. alpha=0.3 means most
    blends are close to one class (λ ~ 0.85 on average) rather than 50/50.

    Returns mixed images and soft label vectors (not hard indices).
    """
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    batch_size = images.size(0)
    idx = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[idx]

    # Convert hard labels to one-hot, then blend
    y1 = F.one_hot(labels, num_classes).float()
    y2 = F.one_hot(labels[idx], num_classes).float()
    mixed_labels = lam * y1 + (1 - lam) * y2

    return mixed_images, mixed_labels
