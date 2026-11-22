"""Rare species: how they do, and whether two cheap changes to the classifier help them
without hurting the common ones.

    python long_tail.py --backbone dinov2l          # step 2: camera-aware classifier on crops

Species are split into three equal-sized groups by how many training bursts they have
(training cameras only): rare, medium, common. The two changes:
    prior adjustment  add tau * log(species' share of training bursts) to the scores;
                      tau < 0 favours rare species, tau > 0 common ones
    species balance   how much each species, rather than each burst, counts in the
                      within-species variation (0: every burst, 1: every species the same)
Every combination is scored on the validation cameras; the best is then scored once on the
test cameras, next to the plain classifier.
"""
import argparse
import json

import numpy as np
import pandas as pd

import exp3_path  # noqa: F401
import plots
from common import RESULTS, classifier, settings, training_statistics
from metrics import burst_vote, macro_f1
from settings import N_SPECIES

TAUS = (-1.0, -0.5, -0.25, 0.0, 0.25, 0.5)
SPECIES_BALANCES = (0.0, 0.5, 1.0)
GROUPS = ("rare", "medium", "common")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", required=True, choices=["resnet50", "dinov2l"])
    parser.add_argument("--step", type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()

    ood = next(s for s in settings(args.backbone, args.step) if s.name == "ood")
    plain_stats = training_statistics(ood.train)
    group_of, limits = species_groups(plain_stats.bursts_per_species)
    names = species_names(ood)

    # 1. every combination on the validation cameras
    candidates = []
    for balance in SPECIES_BALANCES:
        stats = plain_stats if balance == 0 else training_statistics(ood.train, balance)
        model = classifier(stats, args.backbone, args.step, "ood")
        val_scores = model.scores(ood.val.features)
        for tau in TAUS:
            result = evaluate(adjust(val_scores, stats, tau), ood.val, group_of)
            candidates.append({"tau": tau, "species_balance": balance, **result})
            print(f"tau {tau:+.2f}  balance {balance:.1f}   val {result['per_photo']:.4f} · {result['per_burst']:.4f}"
                  f"   rare {result['groups']['rare']:.3f}  medium {result['groups']['medium']:.3f}"
                  f"  common {result['groups']['common']:.3f}")
    plain = next(c for c in candidates if c["tau"] == 0 and c["species_balance"] == 0)
    chosen = max(candidates, key=lambda c: c["per_photo"] + c["per_burst"])
    print(f"chosen on validation: tau {chosen['tau']}, species balance {chosen['species_balance']}")

    # 2. the plain and the chosen classifier, once on the test cameras
    test = {}
    for label, choice in (("plain", plain), ("chosen", chosen)):
        stats = training_statistics(ood.train, choice["species_balance"])
        model = classifier(stats, args.backbone, args.step, "ood")
        scores = adjust(model.scores(ood.test.features), stats, choice["tau"])
        test[label] = evaluate(scores, ood.test, group_of)
        test[label]["confusions_of_rare"] = rare_confusions(scores.argmax(1), ood.test.species, group_of, names)
        print(f"test {label}: {test[label]['per_photo']:.4f} · {test[label]['per_burst']:.4f}   "
              + "  ".join(f"{g} {test[label]['groups'][g]:.3f}" for g in GROUPS))

    table = pd.DataFrame({"species": names, "group": [group_of.get(s, "untrained") for s in range(N_SPECIES)],
                          "training_bursts": plain_stats.bursts_per_species,
                          "test_photos": np.bincount(ood.test.species, minlength=N_SPECIES),
                          "f1_plain": test["plain"]["per_species"], "f1_chosen": test["chosen"]["per_species"]})
    table = table[table.test_photos > 0]
    table.attrs["group_limits"] = limits

    RESULTS.mkdir(parents=True, exist_ok=True)
    name = f"long_tail_{args.backbone}_step{args.step}"
    for result in [*candidates, *test.values()]:
        result.pop("per_species", None)
    (RESULTS / f"{name}.json").write_text(json.dumps({
        "backbone": args.backbone, "step": args.step, "group_limits_in_training_bursts": limits,
        "validation": candidates, "chosen": {"tau": chosen["tau"], "species_balance": chosen["species_balance"]},
        "test": test}, indent=2))
    table.to_csv(RESULTS / f"{name}_species.csv", index=False)
    plots.per_species(table, f"Per-species F1, {args.backbone}, unseen cameras", RESULTS / f"{name}.png")


def species_groups(bursts_per_species):
    """Rare / medium / common: the trained species split into thirds by training bursts."""
    trained = np.flatnonzero(bursts_per_species > 0)
    limits = [float(x) for x in np.quantile(bursts_per_species[trained], [1 / 3, 2 / 3])]
    group_of = {}
    for s in trained:
        n = bursts_per_species[s]
        group_of[int(s)] = "rare" if n <= limits[0] else "medium" if n <= limits[1] else "common"
    return group_of, limits


def adjust(scores, stats, tau):
    """Add tau * log(share of training bursts) to every species' score."""
    if tau == 0:
        return scores
    trained = stats.trained_species
    log_share = np.zeros(N_SPECIES)                      # untrained species stay at -inf anyway
    log_share[trained] = np.log(stats.bursts_per_species[trained] / stats.bursts_per_species.sum())
    return scores + tau * log_share


def per_species_f1(true, predicted):
    """F1 of each species present in `true`; NaN for the others."""
    correct = np.bincount(true[true == predicted], minlength=N_SPECIES)
    n_true = np.bincount(true, minlength=N_SPECIES)
    n_predicted = np.bincount(predicted, minlength=N_SPECIES)
    f1 = 2 * correct / np.maximum(n_true + n_predicted, 1)
    return np.where(n_true > 0, f1, np.nan)


def evaluate(scores, photos, group_of) -> dict:
    predicted = scores.argmax(1)
    voted = burst_vote(predicted, photos.bursts, photos.detector_confidence)
    f1 = per_species_f1(photos.species, predicted)
    groups = {g: float(np.nanmean([f1[s] for s, gg in group_of.items() if gg == g])) for g in GROUPS}
    return {"per_photo": macro_f1(photos.species, predicted), "per_burst": macro_f1(photos.species, voted),
            "groups": groups, "per_species": [None if np.isnan(x) else float(x) for x in f1]}


def rare_confusions(predicted, true, group_of, names, top=10):
    """The most frequent mistakes on photos of rare species: (true species, predicted as, photos)."""
    rare = np.array([group_of.get(s) == "rare" for s in true])
    wrong = rare & (predicted != true)
    pairs = pd.Series(list(zip(true[wrong], predicted[wrong]))).value_counts().head(top)
    return [{"species": names[t], "predicted_as": names[p], "photos": int(n)} for (t, p), n in pairs.items()]


def species_names(setting):
    """Species name of each class number, from the photo tables."""
    table = pd.concat([setting.train.table, setting.val.table, setting.test.table])
    names = dict(zip(table.y.astype(int), table.category))
    return [names.get(s, f"species {s}") for s in range(N_SPECIES)]


if __name__ == "__main__":
    main()
