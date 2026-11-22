"""The backbone / input size / view / detector combinations of the efficiency and resolution study.

Experiment 3's ResNet-50 and DINOv2-L at 448 px on MegaDetector v5a crops are included; their
features already exist. The others need the weights or detections listed in NEEDS.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Combination:
    backbone: str
    size: int
    view: str = "crop"          # "crop" (MegaDetector box) or "whole" (whole photo)
    detector: str = "md5a"      # whose boxes the crop uses

    @property
    def name(self):
        detector = "" if self.view == "whole" or self.detector == "md5a" else f" {self.detector}"
        return f"{self.backbone} {self.size}px {self.view}{detector}"


# What has to be downloaded or computed before a combination can run.
NEEDS = {"dinov2s": "DINOv2-S weights", "mobilenetv3": "MobileNetV3-Large weights", "md6c": "compact MegaDetector boxes"}

COMBINATIONS = [
    # every backbone, small and large input, on crops and on the whole photo
    *(Combination(b, s, v) for b in ("mobilenetv3", "resnet50", "dinov2s", "dinov2b", "dinov2l")
      for s in (224, 448) for v in ("crop", "whole")),
    # the resolution curve for one CNN and two ViTs
    *(Combination(b, s) for b in ("resnet50", "dinov2s", "dinov2l") for s in (112, 336)),
    # crops from the compact detector
    *(Combination(b, s, detector="md6c") for b, s in (("resnet50", 448), ("dinov2s", 224), ("dinov2l", 448))),
]


def needs(combination: Combination) -> list[str]:
    return [NEEDS[k] for k in (combination.backbone, combination.detector) if k in NEEDS
            and not (k == combination.detector and combination.view == "whole")]
