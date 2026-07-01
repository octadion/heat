"""
AdaContrast ResNet-50 wrapper for DomainNet-126, exposed for HEAT.

NEW FILE (DomainNet-126 smoke integration). Does NOT modify any existing model.

The DomainNet-126 source checkpoints (DianCh/AdaContrast, seeds 2020/2021/2022)
are a SHOT/AdaContrast-style net, NOT vanilla torchvision:

    x -> ResNet-50 backbone -> GAP(2048)
      -> bottleneck:  Linear(2048 -> 256) + BatchNorm1d(256)
      -> classifier:  weight_norm(Linear(256 -> 126))   -> logits

The AdaContrast `Classifier` stores this as:
    encoder = Sequential(resnet50_with_fc_replaced_by_Linear(2048->256),
                         BatchNorm1d(256))
    fc      = weight_norm(Linear(256, 126), dim=0)        # keys fc.weight_g/v, fc.bias
    forward: x -> encoder -> flatten -> fc

So the checkpoint state_dict keys look like:
    encoder.0.<resnet keys incl encoder.0.fc.{weight,bias}>   # bottleneck Linear
    encoder.1.{weight,bias,running_mean,running_var,...}      # bottleneck BN1d
    fc.weight_g, fc.weight_v, fc.bias                          # weight-normed head

HEAT contract (src/methods/heat.py, UNCHANGED) requires:
    model.linear              -- the head; HEAT reads `model.linear.in_features`
                                 and calls `model.linear(z_proj)` per stage.
    model.stage_channels      -- [256, 512, 1024, 2048] (ResNet-50 layer1..4).
    model(x, return_stages=True) -> (logits, [z1, z2, z3, z4])  (maps BEFORE GAP).

KEY DESIGN DECISION (documented per the task):
HEAT projects each stage's GAP feature with a frozen W_l to `head_in_dim`, then
applies `model.linear`. The shared head is (bottleneck -> classifier), whose
INPUT is the 2048-dim backbone feature. Therefore **head_in_dim must be 2048**
(NOT 126, NOT the 256-dim bottleneck). We expose `model.linear` as a callable
head whose `.in_features == 2048` and that internally applies
bottleneck(Linear+BN) then the weight-normed classifier. heat.py is NOT touched:
its `model.linear.in_features` reads 2048, and `identity_at_match` makes the
final stage (2048 ch) project through identity, reproducing the real logits.

The head is a plain callable (not an nn.Module) that REFERENCES the wrapper's
real submodules, so its parameters live once under backbone.fc / bottleneck_bn /
classifier and are NOT duplicated in state_dict.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


class _AdaContrastHead:
    """Callable HEAT sees as `model.linear`: 2048-dim GAP feature -> 126 logits
    via bottleneck (Linear 2048->256, BN1d 256) + weight-normed classifier
    (256->126). Not an nn.Module, so it does not duplicate parameters in the
    wrapper's state_dict (it references the real submodules)."""

    def __init__(self, bottleneck_linear: nn.Linear, bottleneck_bn: nn.Module,
                 classifier: nn.Module):
        self._bl = bottleneck_linear
        self._bn = bottleneck_bn
        self._cls = classifier
        self.in_features = int(bottleneck_linear.in_features)  # 2048

    def __call__(self, v: torch.Tensor) -> torch.Tensor:
        return self._cls(self._bn(self._bl(v)))


class ResNet50DomainNet(nn.Module):
    """AdaContrast ResNet-50 for DomainNet-126, HEAT-compatible.

    Submodule names are chosen so the AdaContrast checkpoint maps cleanly:
        encoder.0.*  ->  backbone.*        (resnet incl backbone.fc = 2048->256)
        encoder.1.*  ->  bottleneck_bn.*   (BatchNorm1d 256)
        fc.*         ->  classifier.*      (weight-normed 256->126)
    """

    def __init__(self, num_classes: int = 126, bottleneck_dim: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.bottleneck_dim = bottleneck_dim

        self.backbone = tvm.resnet50(weights=None)
        feat_dim = self.backbone.fc.in_features  # 2048
        self._feat_dim = feat_dim
        # Replace the torchvision classifier with the AdaContrast bottleneck
        # Linear (this is encoder.0.fc in the checkpoint).
        self.backbone.fc = nn.Linear(feat_dim, bottleneck_dim)
        # Bottleneck BatchNorm1d (encoder.1).
        self.bottleneck_bn = nn.BatchNorm1d(bottleneck_dim)
        # Classifier (the checkpoint's `fc`). The AdaContrast `fc` is weight-
        # normed (keys fc.weight_g / fc.weight_v), but we expose the EFFECTIVE
        # plain nn.Linear and FOLD weight_g/weight_v into .weight at load time
        # (see load_domainnet_checkpoint). Rationale: (1) identical forward
        # (weight_norm is just a reparameterization); (2) a live weight_norm hook
        # makes .weight a non-leaf tensor, which breaks copy.deepcopy -- and the
        # method factory deep-copies the base model. A plain Linear is also the
        # standard TTA treatment (HEAT restores/drifts params by name).
        self.classifier = nn.Linear(bottleneck_dim, num_classes)

        # ResNet-50 layer1..4 output channels.
        self.stage_channels = [256, 512, 1024, 2048]

        # Head HEAT uses; .in_features == 2048 (the bottleneck input dim).
        self.linear = _AdaContrastHead(self.backbone.fc, self.bottleneck_bn,
                                       self.classifier)

    # ------------------------------------------------------------------
    def _features(self, x: torch.Tensor):
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        z1 = b.layer1(x)
        z2 = b.layer2(z1)
        z3 = b.layer3(z2)
        z4 = b.layer4(z3)
        return z1, z2, z3, z4

    def forward(self, x: torch.Tensor, return_stages: bool = False):
        z1, z2, z3, z4 = self._features(x)
        feat = F.adaptive_avg_pool2d(z4, 1).flatten(1)   # (B, 2048)
        logits = self.linear(feat)                       # bottleneck + classifier
        if return_stages:
            return logits, [z1, z2, z3, z4]
        return logits


# ----------------------------------------------------------------------------
# Checkpoint loading (AdaContrast -> this wrapper). Reports key matches.
# ----------------------------------------------------------------------------

_REMAP_PREFIXES = (
    ("encoder.0.", "backbone."),
    ("encoder.1.", "bottleneck_bn."),
    ("fc.", "classifier."),
)


def _strip_prefixes(state: dict) -> dict:
    """Strip a leading 'module.' (DataParallel) and unwrap common containers."""
    out = {}
    for k, v in state.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        out[nk] = v
    return out


def _remap_adacontrast_keys(state: dict) -> dict:
    remapped = {}
    for k, v in state.items():
        nk = k
        for src, dst in _REMAP_PREFIXES:
            if k.startswith(src):
                nk = dst + k[len(src):]
                break
        remapped[nk] = v
    return remapped


def _fold_weight_norm(state: dict) -> dict:
    """Fold a weight-normed classifier (classifier.weight_g / classifier.weight_v)
    into an effective classifier.weight, so it loads into a plain nn.Linear.
    weight_norm(dim=0): w = g * v / ||v|| with the norm taken per output row."""
    g = state.pop("classifier.weight_g", None)
    v = state.pop("classifier.weight_v", None)
    if g is not None and v is not None and "classifier.weight" not in state:
        # v: (out, in); g: (out, 1). Norm over the input dim, per output row.
        norm = v.flatten(1).norm(dim=1).view(-1, *([1] * (v.dim() - 1)))
        state["classifier.weight"] = g * v / norm
    return state


def load_domainnet_checkpoint(model: ResNet50DomainNet, ckpt_path: str,
                              device: Optional[torch.device] = None,
                              verbose: bool = True) -> dict:
    """Load an AdaContrast source checkpoint into the wrapper.

    Returns a report dict: {ok, strict, missing, unexpected, n_loaded,
    sample_ckpt_keys}. Tries strict first; on failure loads non-strict and
    reports exactly which keys mismatched (so the remap can be adjusted if a
    given mirror uses different prefixes)."""
    map_loc = device if device is not None else "cpu"
    raw = torch.load(ckpt_path, map_location=map_loc, weights_only=False)

    # Unwrap common containers.
    if isinstance(raw, dict):
        for key in ("state_dict", "model", "net", "classifier"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    state = _strip_prefixes(raw)
    sample = list(state.keys())[:24]
    remapped = _fold_weight_norm(_remap_adacontrast_keys(state))

    if verbose:
        print(f"[ckpt] {ckpt_path}")
        print(f"[ckpt] {len(state)} tensors; sample keys: {sample}")

    try:
        model.load_state_dict(remapped, strict=True)
        if verbose:
            print(f"[ckpt] strict load OK ({len(remapped)} tensors).")
        return {"ok": True, "strict": True, "missing": [], "unexpected": [],
                "n_loaded": len(remapped), "sample_ckpt_keys": sample}
    except RuntimeError as exc:
        if verbose:
            print(f"[ckpt] strict load FAILED: {exc}")
        incompat = model.load_state_dict(remapped, strict=False)
        missing = list(incompat.missing_keys)
        unexpected = list(incompat.unexpected_keys)
        n_loaded = len(remapped) - len(unexpected)
        if verbose:
            print(f"[ckpt] non-strict load: {n_loaded} matched, "
                  f"{len(missing)} missing, {len(unexpected)} unexpected.")
            print(f"[ckpt] missing(head): {missing[:12]}")
            print(f"[ckpt] unexpected(head): {unexpected[:12]}")
        ok = (len(missing) == 0 and len(unexpected) == 0)
        return {"ok": ok, "strict": False, "missing": missing,
                "unexpected": unexpected, "n_loaded": n_loaded,
                "sample_ckpt_keys": sample}


def resnet50_domainnet(num_classes: int = 126, **_) -> ResNet50DomainNet:
    return ResNet50DomainNet(num_classes=num_classes)
