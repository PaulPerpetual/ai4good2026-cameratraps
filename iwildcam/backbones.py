"""Backbone registry: every frozen feature extractor the protocol knows, behind one interface.

Adding a backbone = adding one `Spec` to `REGISTRY`. Nothing downstream (feature store,
classifiers, metrics) depends on which backbone produced the features.

    spec = REGISTRY["dinov2l"]
    model, transform = spec.load()          # model(images) -> (batch, spec.dim)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torchvision.transforms import v2

IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
OPENAI_CLIP = ((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

DINOV2_REPO = "facebookresearch/dinov2:7764ea0f912e53c92e82eb78a2a1631e92725fc8"


@dataclass(frozen=True)
class Spec:
    name: str            # short name used on the command line
    store_name: str      # prefix of its feature-store folders
    size: int            # input side in pixels (images are squashed to size x size)
    dim: int
    normalisation: tuple
    features: str        # what the feature vector is, for config.json
    build: Callable[[], nn.Module] = field(repr=False)

    def transform(self):
        mean, std = self.normalisation
        return v2.Compose([
            v2.Resize((self.size, self.size), interpolation=v2.InterpolationMode.BICUBIC
                      if self.normalisation is OPENAI_CLIP else v2.InterpolationMode.BILINEAR,
                      antialias=True),
            v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean=mean, std=std)])

    def load(self):
        return self.build().eval(), self.transform()

    def folder_name(self, view: str = "crop", cameras: str = "trainval") -> str:
        """Feature-store folder. cameras: trainval | test | testempty."""
        name = f"{self.store_name}_{self.size}" + ("" if view == "whole" else f"_{view}")
        return name + {"trainval": "", "test": "_test", "testempty": "_testempty"}[cameras]


class _DinoV2(nn.Module):
    """[class token, mean of patch tokens], both after the final norm."""

    def __init__(self, hub_name: str):
        super().__init__()
        self.body = torch.hub.load(DINOV2_REPO, hub_name, trust_repo=True, skip_validation=True)

    def forward(self, images):
        tokens = self.body.forward_features(images)
        return torch.cat([tokens["x_norm_clstoken"], tokens["x_norm_patchtokens"].mean(1)], dim=1)


class _OpenClip(nn.Module):
    """The image embedding (after the projection), not normalised."""

    def __init__(self, hub_name: str):
        super().__init__()
        import open_clip
        self.body, _, _ = open_clip.create_model_and_transforms(hub_name)

    def forward(self, images):
        return self.body.encode_image(images)


class _ResNet50(nn.Module):
    """Global average pool before ImageNet's classifier. The starter baseline only."""

    def __init__(self):
        super().__init__()
        import torchvision.models as tvm
        self.body = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
        self.body.fc = nn.Identity()

    def forward(self, images):
        return self.body(images)


REGISTRY: dict[str, Spec] = {s.name: s for s in [
    Spec("dinov2l", "dinov2_vitl14_reg", 448, 2048, IMAGENET,
         "[x_norm_clstoken, mean(x_norm_patchtokens)]", lambda: _DinoV2("dinov2_vitl14_reg")),
    Spec("bioclip2", "bioclip2", 224, 768, OPENAI_CLIP,
         "open_clip encode_image (projected, unnormalised)", lambda: _OpenClip("hf-hub:imageomics/bioclip-2")),
    Spec("resnet50", "resnet50", 448, 2048, IMAGENET,
         "global average pool (fc removed)", _ResNet50),
]}
