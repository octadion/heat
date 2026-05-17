"""
WideResNet-28-10 for CIFAR-10/100 with stage-level outputs.

Mirrors the API of resnet_cifar.ResNet18CIFAR:
  - forward(x, return_stages=False)
  - .stage_channels   tuple of channel widths at each stage output
  - .linear           classifier head (Linear(stage_out_dim, num_classes))

This is the architecture used by RobustBench `Standard` and by TEA paper Tab 7.

Stage definition (analogous to ResNet18CIFAR):
  stem -> block1 (stage 0) -> block2 (stage 1) -> block3 (stage 2) -> head
We expose 3 stages — fewer than ResNet18 (which has 4) because WRN-28-10 has
only 3 residual groups. This is intentional: the cross-architecture story
should hold WITH DIFFERENT stage counts, demonstrating that where-emergence
is not a stage-count artifact.

Channel widths at stages: (160, 320, 640) for k=10. Final stage = 640 = head input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _BasicBlock(nn.Module):
    """Pre-activation basic block used by WideResNet."""

    def __init__(self, in_planes: int, out_planes: int, stride: int,
                 dropout: float = 0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = (
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                      bias=False)
            if stride != 1 or in_planes != out_planes
            else nn.Identity()
        )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        sc = self.shortcut(out) if not isinstance(self.shortcut, nn.Identity) else x
        out = self.conv1(out)
        out = F.relu(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + sc


class _NetworkBlock(nn.Module):
    def __init__(self, n: int, in_planes: int, out_planes: int,
                 stride: int, dropout: float = 0.0):
        super().__init__()
        layers = []
        for i in range(n):
            in_p = in_planes if i == 0 else out_planes
            s = stride if i == 0 else 1
            layers.append(_BasicBlock(in_p, out_planes, s, dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class WideResNet(nn.Module):
    """
    WRN-depth-widen for CIFAR. Default WRN-28-10.

    For depth=28: each NetworkBlock has n = (depth - 4) // 6 = 4 blocks.
    """

    def __init__(self, depth: int = 28, widen_factor: int = 10,
                 num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth must be 6n+4"
        n = (depth - 4) // 6
        widths = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        self.conv1 = nn.Conv2d(3, widths[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.block1 = _NetworkBlock(n, widths[0], widths[1], stride=1,
                                    dropout=dropout)
        self.block2 = _NetworkBlock(n, widths[1], widths[2], stride=2,
                                    dropout=dropout)
        self.block3 = _NetworkBlock(n, widths[2], widths[3], stride=2,
                                    dropout=dropout)
        self.bn1 = nn.BatchNorm2d(widths[3])
        self.linear = nn.Linear(widths[3], num_classes)
        self.num_classes = num_classes

        # Stage channel widths exposed for HEAT projections.
        # (block1_out, block2_out, block3_out)
        self.stage_channels = (widths[1], widths[2], widths[3])

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_stages: bool = False):
        s0 = self.conv1(x)
        s1 = self.block1(s0)              # stage 0 output
        s2 = self.block2(s1)              # stage 1 output
        s3 = self.block3(s2)              # stage 2 output
        out = F.relu(self.bn1(s3))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        logits = self.linear(out)

        if return_stages:
            # Return raw stage outputs (before final BN+ReLU+pool) — matching
            # the convention of resnet_cifar so HEAT projections work uniformly.
            return logits, [s1, s2, s3]
        return logits


def wrn_28_10_cifar(num_classes: int = 10) -> WideResNet:
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes)
