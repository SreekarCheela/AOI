"""Whole-board classifier: MobileNetV3 backbone + pass/fail and defect-type heads.

See ARCHITECTURE.md for the backbone selection rationale.
"""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from labels import NUM_DEFECT_CLASSES

_BACKBONES = {
    "mobilenet_v3_large": (
        models.mobilenet_v3_large,
        models.MobileNet_V3_Large_Weights.IMAGENET1K_V2,
    ),
    "mobilenet_v3_small": (
        models.mobilenet_v3_small,
        models.MobileNet_V3_Small_Weights.IMAGENET1K_V1,
    ),
}


class WholeBoardClassifier(nn.Module):
    """MobileNetV3 feature extractor with two independent classification heads.

    - pass_fail: single logit, BCEWithLogitsLoss target (0=fail, 1=pass)
    - defect_type: one logit per class in labels.DEFECT_CLASSES, multi-label
      (a board may exhibit more than one defect simultaneously)
    """

    def __init__(
        self,
        backbone_name: str = "mobilenet_v3_large",
        pretrained: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if backbone_name not in _BACKBONES:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. Options: {list(_BACKBONES)}"
            )
        ctor, weights_enum = _BACKBONES[backbone_name]
        backbone = ctor(weights=weights_enum if pretrained else None)

        # torchvision MobileNetV3's `classifier` is a small MLP ending in a
        # Linear to 1000 ImageNet classes. We keep everything up to (and
        # including) the penultimate hidden Linear + activation, and replace
        # the final Linear with our two heads sharing that hidden feature.
        in_features = backbone.classifier[-1].in_features
        backbone.classifier = backbone.classifier[:-1]  # drop final 1000-way Linear
        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout)

        self.pass_fail_head = nn.Linear(in_features, 1)
        self.defect_type_head = nn.Linear(in_features, NUM_DEFECT_CLASSES)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        features = self.dropout(features)
        return {
            "pass_fail_logits": self.pass_fail_head(features).squeeze(-1),
            "defect_type_logits": self.defect_type_head(features),
        }


def build_model(config: dict) -> WholeBoardClassifier:
    model_cfg = config.get("model", {})
    return WholeBoardClassifier(
        backbone_name=model_cfg.get("backbone", "mobilenet_v3_large"),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.2),
    )
