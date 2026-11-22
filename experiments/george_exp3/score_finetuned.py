"""Step 4: score the saved fine-tuned weights on the unseen test cameras (OOD).

    python score_finetuned.py --backbone resnet50     # mean ± std over the 3 seeds
    python score_finetuned.py --backbone dinov2l      # adapters at 25% strength

No ID score: the fine-tuned backbones learned from photos that are also in the seen-camera test.
"""
import argparse
import json

import numpy as np
import torch

from data import labelled_photos
from finetune import evaluate, starting_point
from models import SpeciesNetwork, add_lora, set_adapter_strength
from settings import EXPECTED_VALIDATION_FINETUNED, FINE_TUNING, RESULTS, checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["resnet50", "dinov2l"])
    args = parser.parse_args()
    plan = FINE_TUNING[args.backbone]

    photos, discriminant = starting_point(args.backbone)
    val = photos[photos.official_ood == "val"].reset_index(drop=True)
    test = labelled_photos(test=True)

    results = {}
    for seed in plan.seeds:
        network = load_finetuned(args.backbone, seed, discriminant).cuda()
        validation = evaluate(network, val)
        check_validation(validation, EXPECTED_VALIDATION_FINETUNED[(args.backbone, seed)])
        results[seed] = evaluate(network, test)
        print(f"seed {seed}  OOD {results[seed]['per_photo']:.4f} · {results[seed]['per_burst']:.4f}")

    summary = {key: (float(np.mean([r[key] for r in results.values()])),
                     float(np.std([r[key] for r in results.values()], ddof=1)) if len(results) > 1 else 0.0)
               for key in ("per_photo", "per_burst")}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"finetuned_{args.backbone}.json").write_text(
        json.dumps({"backbone": args.backbone, "seeds": results, "mean_and_std": summary}, indent=2))
    (photo, photo_std), (burst, burst_std) = summary["per_photo"], summary["per_burst"]
    print(f"{args.backbone} fine-tuned   OOD {photo:.3f} ± {photo_std:.3f} · {burst:.3f} ± {burst_std:.3f}")


def load_finetuned(backbone, seed, discriminant) -> SpeciesNetwork:
    """The network with the saved weights; feature mean and std come from the discriminant."""
    plan = FINE_TUNING[backbone]
    network = SpeciesNetwork(backbone, discriminant)
    if plan.method == "lora":
        add_lora(network, plan.lora_rank)
        set_adapter_strength(network, plan.adapter_strength)

    saved = torch.load(checkpoint_path(backbone, seed), map_location="cpu")
    saved = {_current_name(k): v.float() if v.is_floating_point() else v for k, v in saved.items()
             if k not in ("mu", "sd", "feature_mean", "feature_std")}
    missing, unexpected = network.load_state_dict(saved, strict=False)
    assert not unexpected, unexpected
    if plan.method == "lora":      # only the adapters and the head were saved; the rest is DINOv2
        missing = [k for k in missing if "lora_" in k or k.startswith("head.")]
    else:
        missing = [k for k in missing if k not in ("feature_mean", "feature_std")]
    assert not missing, missing
    return network


def _current_name(key: str) -> str:
    """Weights saved by the earlier scripts call the backbone 'body' instead of 'backbone.body'."""
    return "backbone." + key if key.startswith("body.") else key


def check_validation(got, expected):
    for key, value in zip(("per_photo", "per_burst"), expected):
        if abs(got[key] - value) > 0.002:
            raise SystemExit(f"validation {key} {got[key]:.4f}, expected {value}: weights differ, test not scored")


if __name__ == "__main__":
    main()
