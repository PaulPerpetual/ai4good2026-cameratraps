"""Step 4: fine-tune the backbone on the training cameras' crops.

    python finetune.py --backbone resnet50 --seed 0     # all layers
    python finetune.py --backbone dinov2l --seed 0      # LoRA adapters, DINOv2 itself frozen

Starts from step 2's camera-aware classifier and scores the validation cameras after every
epoch. Test cameras are not read.
"""
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import lda
from data import AUGMENTED, Crops, detector_confidence, load_features
from metrics import score
from models import SpeciesNetwork, add_lora, set_adapter_strength
from settings import CLASSIFIER, FINE_TUNING, N_SPECIES, WARMUP_STEPS, checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["resnet50", "dinov2l"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample", type=int, help="quick check: this many photos per split, 1 epoch, nothing saved")
    args = parser.parse_args()
    plan = FINE_TUNING[args.backbone]
    epochs = 1 if args.sample else plan.epochs
    out = checkpoint_path(args.backbone, args.seed)
    if out.exists() and not args.sample:
        raise SystemExit(f"{out} exists; these are the weights behind the slides")
    torch.manual_seed(args.seed)

    photos, discriminant = starting_point(args.backbone)
    train = photos[photos.official_ood == "train"].reset_index(drop=True)
    val = photos[photos.official_ood == "val"].reset_index(drop=True)
    if args.sample:
        train = train.sample(args.sample, random_state=args.seed).reset_index(drop=True)
        val = val.sample(args.sample, random_state=args.seed).reset_index(drop=True)

    network = SpeciesNetwork(args.backbone, discriminant).cuda()
    if plan.method == "lora":
        print(f"adapters on {add_lora(network, plan.lora_rank)} layers")
        network.cuda()
    trainable = [p for p in network.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{"params": trainable, "lr": plan.learning_rate},
                                   {"params": network.head.parameters(), "lr": plan.head_learning_rate}],
                                  weight_decay=plan.weight_decay)

    species = train.y.to_numpy().astype(int)
    weights = lda.burst_weights(train.burst_id)
    loader = DataLoader(Crops(train.file_name, transform=AUGMENTED, species=species, weights=weights / weights.mean()),
                        batch_size=plan.batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
    schedule = warmup_then_cosine(optimizer, total_steps=epochs * len(loader))
    log_frequency = torch.tensor(np.log(np.maximum(np.bincount(species, weights, N_SPECIES), 1e-9) / weights.sum()),
                                 dtype=torch.float32).cuda() if plan.balanced_loss else 0.0

    best = evaluate(network, val)
    print(f"epoch 0  validation {best['per_photo']:.4f} · {best['per_burst']:.4f}")
    best_state = None
    for epoch in range(1, epochs + 1):
        train_one_epoch(network, loader, optimizer, schedule, log_frequency)
        result = evaluate(network, val)
        print(f"epoch {epoch}  validation {result['per_photo']:.4f} · {result['per_burst']:.4f}")
        if result["per_photo"] + result["per_burst"] > best["per_photo"] + best["per_burst"]:
            best, best_state = result, {k: v.detach().cpu().clone() for k, v in network.state_dict().items()}

    if plan.method == "lora":
        compare_adapter_strengths(network, val)
        best_state = {k: v.cpu() for k, v in network.state_dict().items() if "lora_" in k or k.startswith("head.")}
    if args.sample:
        return
    if best_state is None:
        raise SystemExit("no epoch beat the starting point; nothing saved")
    torch.save(best_state, out)
    print(f"saved {out}")


def starting_point(backbone):
    """Step 2's camera-aware classifier, fitted on the training cameras' saved features."""
    photos, features = load_features(backbone, test=False)
    train = (photos.official_ood == "train").to_numpy()
    discriminant = lda.fit(features[train], photos.y.to_numpy()[train].astype(int), photos.location.to_numpy()[train],
                           photos.burst_id.to_numpy()[train], *CLASSIFIER[(2, backbone, "ood")])
    return photos, discriminant


def train_one_epoch(network, loader, optimizer, schedule, log_frequency):
    """Each photo's loss is weighted so that every burst counts once. With log_frequency, the
    loss is the balanced softmax: common species cannot win just by being common."""
    network.train()
    trainable = [p for group in optimizer.param_groups for p in group["params"]]
    for images, species, weights in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = network(images.cuda(non_blocking=True))
        losses = F.cross_entropy(scores.float() + log_frequency, species.cuda(), reduction="none")
        loss = (losses * weights.cuda().float()).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        schedule.step()


def warmup_then_cosine(optimizer, total_steps):
    """Learning rate rises linearly for the first steps, then falls along a cosine to zero."""
    warmup = min(WARMUP_STEPS, total_steps // 10)

    def factor(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(total_steps - warmup, 1)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def evaluate(network, photos) -> dict:
    """Macro F1 per photo and after the burst vote."""
    network.eval()
    loader = DataLoader(Crops(photos.file_name), batch_size=64, num_workers=16, pin_memory=True)
    predicted = []
    for images, _, _ in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted.append(network(images.cuda(non_blocking=True)).float().argmax(1).cpu().numpy())
    return score(photos.y.to_numpy().astype(int), np.concatenate(predicted), photos.burst_id.to_numpy(),
                 detector_confidence(photos.file_name))


def compare_adapter_strengths(network, val):
    """Validation score with the adapters' change scaled down; 0 is the frozen model."""
    for strength in (0.0, 0.25, 0.5, 0.75, 1.0):
        set_adapter_strength(network, strength)
        result = evaluate(network, val)
        print(f"adapter strength {strength:.2f}  validation {result['per_photo']:.4f} · {result['per_burst']:.4f}")
    set_adapter_strength(network, 1.0)


if __name__ == "__main__":
    main()
