"""
ViT-S/16 wrapper for CIFAR-10 / CIFAR-100 TTA.

Backbone: timm `vit_small_patch16_224` (ImageNet-pretrained).
  - embed_dim = 384
  - depth = 12 transformer blocks
  - patch size = 16, input resolution = 224
The pretrained 1000-class head is replaced with `nn.Linear(384, num_classes)`.

CIFAR images (native 32x32) MUST be resized to 224x224 (bicubic) and
normalized with ImageNet stats (resolved from timm) by the data pipeline.

HEAT contract — this wrapper is designed so `src/methods/heat.py` works
unchanged. HEAT needs:
  - `model(x, return_stages=True)` → `(logits, stages)` where `stages` is a
    list of 4D `[B, C, H, W]` tensors (HEAT applies `F.adaptive_avg_pool2d(z,1)`).
  - `model.stage_channels` — list of per-stage channel counts.
  - `model.linear` — the classification head; HEAT reads `linear.in_features`
    and applies `linear` to projected features.

We expose 4 stages from transformer blocks [2, 5, 8, 11]. For each block,
the token sequence `[B, N, 384]` is mean-pooled over tokens to `[B, 384]`,
then reshaped to `[B, 384, 1, 1]`. This way HEAT's
`F.adaptive_avg_pool2d(z, 1).flatten(1)` becomes a no-op producing `[B, 384]`,
exactly as it does for CNN stage features.

Because every entry of `stage_channels` equals `linear.in_features` (= 384),
HEAT's `identity_at_match=True` automatically uses `torch.eye(384)` for every
projection — no custom projection code is needed.

The standard forward path (the `model(x)` call used by source / tent / tea /
epotta / retta and evaluation) is timm's own; we do NOT re-implement it.
Stage capture is done with forward hooks so a single forward pass yields
both `logits` and the 4 stage features.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


VIT_S_STAGE_BLOCKS = (2, 5, 8, 11)  # 0-indexed transformer blocks to capture
VIT_S_EMBED_DIM = 384


class ViTSCifar(nn.Module):
    """
    timm ViT-S/16 wrapped with HEAT-compatible stage-feature extraction.

    Args:
        num_classes: 10 for CIFAR-10, 100 for CIFAR-100.
        pretrained: load ImageNet-pretrained weights from timm.
        timm_model_name: timm model id; default 'vit_small_patch16_224'.
            For a specific pretrained tag pass e.g.
            'vit_small_patch16_224.augreg_in21k_ft_in1k'.
    """

    def __init__(
        self,
        num_classes: int = 10,
        pretrained: bool = True,
        timm_model_name: str = "vit_small_patch16_224",
    ):
        super().__init__()

        # Lazy import so non-ViT runs do not require timm.
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "timm is required for vit_s. Install with `pip install timm`."
            ) from e

        # Build with `num_classes=0` so timm's head is an identity; we attach
        # our own. Keeping our head separate makes the HEAT contract
        # (`self.linear`) explicit.
        self.backbone = timm.create_model(
            timm_model_name,
            pretrained=pretrained,
            num_classes=0,  # disables timm head; backbone returns features
        )
        if getattr(self.backbone, "embed_dim", VIT_S_EMBED_DIM) != VIT_S_EMBED_DIM:
            raise ValueError(
                f"Expected embed_dim={VIT_S_EMBED_DIM}, got "
                f"{self.backbone.embed_dim}. Pass a ViT-S/16 model name."
            )

        self.num_classes = num_classes
        self.linear = nn.Linear(VIT_S_EMBED_DIM, num_classes)

        # HEAT contract: 4 stages, all 384-dim → identity_at_match=True kicks
        # in and HEAT builds `torch.eye(384)` projections automatically.
        self.stage_channels = [VIT_S_EMBED_DIM] * len(VIT_S_STAGE_BLOCKS)

        self._stage_blocks = VIT_S_STAGE_BLOCKS

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def _capture_block_outputs(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Run a forward through the backbone while capturing intermediate
        transformer block outputs at self._stage_blocks via forward hooks.

        Returns:
            features: backbone output features (post final norm / pre-head),
                shape `[B, embed_dim]` — these come from timm's
                `forward_features` + `forward_head(pre_logits=True)` pipeline.
            stage_features_4d: list of 4 tensors each `[B, 384, 1, 1]`.
        """
        captured: dict[int, torch.Tensor] = {}

        def make_hook(idx: int):
            def _hook(_module, _inp, out):
                # Block output is `[B, N, embed_dim]`. Mean-pool over tokens
                # (the full sequence including CLS — CLS is 1/N of the
                # average, negligible) → `[B, embed_dim]`. Then reshape
                # `[B, embed_dim, 1, 1]` to satisfy HEAT's 4D contract.
                pooled = out.mean(dim=1)
                captured[idx] = pooled.unsqueeze(-1).unsqueeze(-1)
            return _hook

        handles = []
        for idx in self._stage_blocks:
            block = self.backbone.blocks[idx]
            handles.append(block.register_forward_hook(make_hook(idx)))
        try:
            # timm's standard inference path: produces `[B, embed_dim]` after
            # final norm + CLS-token selection (with num_classes=0 the head
            # is an identity, so this is the pre-logits feature).
            features = self.backbone(x)
        finally:
            for h in handles:
                h.remove()

        stage_features_4d = [captured[i] for i in self._stage_blocks]
        return features, stage_features_4d

    def forward(self, x: torch.Tensor, return_stages: bool = False):
        features, stage_features = self._capture_block_outputs(x)
        logits = self.linear(features)
        if return_stages:
            return logits, stage_features
        return logits

    def forward_stages(self, x: torch.Tensor) -> List[torch.Tensor]:
        _, stages = self.forward(x, return_stages=True)
        return stages


def vit_s_cifar(num_classes: int = 10, pretrained: bool = True) -> ViTSCifar:
    return ViTSCifar(num_classes=num_classes, pretrained=pretrained)


# ---------------------------------------------------------------------------
# Data config helper (resolved from timm rather than hardcoded)
# ---------------------------------------------------------------------------

def get_vit_s_data_config(
    timm_model_name: str = "vit_small_patch16_224",
    pretrained: bool = True,
) -> dict:
    """
    Return the timm data config (input_size, mean, std, interpolation) for
    the ViT-S/16 model. Used by the data pipeline to build the eval/train
    transforms without hardcoding ImageNet mean/std.
    """
    import timm
    from timm.data import resolve_model_data_config

    m = timm.create_model(timm_model_name, pretrained=pretrained, num_classes=0)
    cfg = resolve_model_data_config(m)
    del m
    return cfg
