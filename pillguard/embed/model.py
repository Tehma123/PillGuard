"""Lightweight embedding network and the ArcFace head used only during training.

* Backbone: torchvision MobileNetV3-Small (default), MobileNetV3-Large or EfficientNet-B0,
  ImageNet-pretrained.
* Head: global average pool -> MLP -> ``EMB_DIM`` -> L2 normalise.
* Loss: additive angular margin softmax (ArcFace) over the *seen* classes. At inference
  the class weights are discarded; recognition is nearest-prototype in embedding space,
  so adding a drug later only needs new prototypes, no retraining.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from pillguard.config import EMB_DIM

BACKBONES = {
    "mobilenet_v3_small": (models.mobilenet_v3_small, models.MobileNet_V3_Small_Weights.IMAGENET1K_V1, 576),
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V1, 960),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
}


class EmbeddingNet(nn.Module):
    def __init__(self, backbone: str = "mobilenet_v3_small", emb_dim: int = EMB_DIM, pretrained: bool = True,
                 hidden: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        ctor, weights, feat_dim = BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)
        self.backbone_name = backbone
        self.features = net.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, emb_dim),
        )
        self.emb_dim = emb_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.pool(self.features(x)).flatten(1)
        return F.normalize(self.proj(f), dim=1)


class ArcFaceHead(nn.Module):
    """Normalised-softmax with additive angular margin (Deng et al. 2019)."""

    def __init__(self, n_classes: int, emb_dim: int = EMB_DIM, scale: float = 30.0, margin: float = 0.30) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, emb_dim))
        nn.init.xavier_uniform_(self.weight)
        self.s, self.m = scale, margin
        self.cos_m, self.sin_m = math.cos(margin), math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, emb: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        cos = F.linear(emb, F.normalize(self.weight, dim=1)).clamp(-1 + 1e-7, 1 - 1e-7)
        if labels is None:
            return cos * self.s
        sin = torch.sqrt(1.0 - cos * cos)
        phi = cos * self.cos_m - sin * self.sin_m          # cos(theta + m)
        phi = torch.where(cos > self.th, phi, cos - self.mm)  # keep monotonic when theta + m > pi
        onehot = F.one_hot(labels, cos.shape[1]).to(cos.dtype)
        return self.s * (onehot * phi + (1 - onehot) * cos)


def build_model(backbone: str = "mobilenet_v3_small", emb_dim: int = EMB_DIM, pretrained: bool = True) -> EmbeddingNet:
    return EmbeddingNet(backbone=backbone, emb_dim=emb_dim, pretrained=pretrained)


def load_embedding_net(ckpt_path, device: str = "cpu") -> EmbeddingNet:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(ck["backbone"], ck["emb_dim"], pretrained=False)
    model.load_state_dict(ck["model"])
    model.eval().to(device)
    return model
