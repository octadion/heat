"""
ResNet-18 for CIFAR-10.

This is NOT torchvision.models.resnet18 (which is for ImageNet 224x224).
CIFAR-10 ResNet-18 uses 3x3 first conv with stride 1, no maxpool.

The model exposes intermediate features after each stage. HEAT needs these
to compute hierarchical energy E_l per stage, all evaluated through the same
classifier head h.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------------

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


# ----------------------------------------------------------------------------
# ResNet-18 with stage outputs exposed
# ----------------------------------------------------------------------------

class ResNet18CIFAR(nn.Module):
    """
    Standard CIFAR-10 ResNet-18.

    Stages produce features with channels [64, 128, 256, 512] and
    progressively halved spatial size [32, 16, 8, 4].

    The forward pass returns:
        logits: (B, num_classes) — final prediction
        stages: list of 4 tensors — features after stage 1..4
                (only when return_stages=True)

    For HEAT, we evaluate energy on every stage via the SAME classifier
    head `self.linear`. This requires per-stage projection because stage
    channels differ from `linear`'s input dim (512). The projection is
    fixed (orthogonal, frozen) and lives in HEATWrapper, not here — keeping
    this model class clean and reusable for baselines.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes

        # Stem: 3x3 conv stride 1 (CIFAR-10), NOT 7x7 stride 2 (ImageNet)
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # 4 stages, [2, 2, 2, 2] basic blocks each
        self.stage1 = self._make_stage(64,  num_blocks=2, stride=1)
        self.stage2 = self._make_stage(128, num_blocks=2, stride=2)
        self.stage3 = self._make_stage(256, num_blocks=2, stride=2)
        self.stage4 = self._make_stage(512, num_blocks=2, stride=2)

        self.linear = nn.Linear(512, num_classes)

        # Channel sizes per stage — used by HEATWrapper
        self.stage_channels = [64, 128, 256, 512]

    def _make_stage(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, return_stages: bool = False):
        out = F.relu(self.bn1(self.conv1(x)))
        s1 = self.stage1(out)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        # Standard prediction: GAP over s4 → linear
        feat = F.adaptive_avg_pool2d(s4, 1).flatten(1)
        logits = self.linear(feat)

        if return_stages:
            return logits, [s1, s2, s3, s4]
        return logits

    # Convenience: just return stages without logits
    def forward_stages(self, x: torch.Tensor):
        _, stages = self.forward(x, return_stages=True)
        return stages


def resnet18_cifar(num_classes: int = 10) -> ResNet18CIFAR:
    return ResNet18CIFAR(num_classes=num_classes)
