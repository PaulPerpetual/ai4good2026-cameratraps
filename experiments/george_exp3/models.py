"""The two backbones, the network used for fine-tuning, and LoRA adapters."""
import math

import torch
import torch.nn as nn
import torchvision.models as tvm

from settings import BACKBONES, DINOV2_REPO, N_SPECIES


class Backbone(nn.Module):
    """Photo in, one feature vector out.

    ResNet-50:   the global average pool before ImageNet's classifier (2048 numbers).
    MobileNetV3: the same for MobileNetV3-Large (960).
    DINOv2:      the summary token and the average of the patch tokens (2 x 384 for S,
                 2 x 768 for B, 2 x 1024 for L).
    """

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        if name == "resnet50":
            self.body = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V1)
            self.body.fc = nn.Identity()
        elif name == "mobilenetv3":
            self.body = tvm.mobilenet_v3_large(weights=tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
            self.body.classifier = nn.Identity()
        else:
            self.body = torch.hub.load(DINOV2_REPO, BACKBONES[name], trust_repo=True, skip_validation=True)

    def forward(self, images):
        if self.name in ("resnet50", "mobilenetv3"):
            return self.body(images)
        tokens = self.body.forward_features(images)
        return torch.cat([tokens["x_norm_clstoken"], tokens["x_norm_patchtokens"].mean(1)], dim=1)


class SpeciesNetwork(nn.Module):
    """Backbone, then the features standardised and a linear layer for the 60 species.

    The linear layer starts as the fitted linear discriminant, so before any training the
    network gives exactly the frozen pipeline's predictions.
    """

    def __init__(self, backbone_name: str, discriminant):
        super().__init__()
        self.backbone = Backbone(backbone_name)
        self.register_buffer("feature_mean", torch.tensor(discriminant.feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.tensor(discriminant.feature_std, dtype=torch.float32))
        self.head = nn.Linear(len(discriminant.feature_mean), N_SPECIES)
        self.head.weight.data = torch.tensor(discriminant.weights.T, dtype=torch.float32)
        self.head.bias.data = torch.tensor(discriminant.bias, dtype=torch.float32)

    def forward(self, images):
        features = self.backbone(images)
        return self.head((features.float() - self.feature_mean) / self.feature_std)

    def train(self, mode=True):
        """Training mode, except that BatchNorm keeps ImageNet's statistics: recomputing them
        on our training cameras would tie them to those cameras' lighting."""
        super().train(mode)
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self


# ---------------------------------------------------------------------------------------
# LoRA: small trainable adapters next to frozen layers
# ---------------------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """A frozen linear layer plus a trainable low-rank correction: output = W x + B A x.

    B starts at zero, so the layer starts out unchanged. `strength` scales the correction
    at prediction time (1: as trained, 0: the original layer).
    """

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        self.lora_A = nn.Linear(base.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scale = 2.0          # alpha / rank with alpha = 2 * rank
        self.strength = 1.0

    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x)) * (self.scale * self.strength)


ADAPTED_LAYERS = (["attn", "qkv"], ["attn", "proj"], ["mlp", "fc1"], ["mlp", "fc2"])


def add_lora(network: SpeciesNetwork, rank: int) -> int:
    """Freeze DINOv2 and put an adapter next to the 4 linear layers of each of its 24 blocks.
    Returns how many layers were adapted (96)."""
    dinov2 = network.backbone.body
    for parameter in dinov2.parameters():
        parameter.requires_grad = False
    targets = [name for name, module in dinov2.named_modules()
               if isinstance(module, nn.Linear) and name.startswith("blocks.")
               and name.split(".")[-2:] in ADAPTED_LAYERS]
    for name in targets:
        parent_name, child_name = name.rsplit(".", 1)
        parent = dinov2.get_submodule(parent_name)
        setattr(parent, child_name, LoRALinear(getattr(parent, child_name), rank))
    return len(targets)


def set_adapter_strength(network: nn.Module, strength: float):
    for module in network.modules():
        if isinstance(module, LoRALinear):
            module.strength = strength
