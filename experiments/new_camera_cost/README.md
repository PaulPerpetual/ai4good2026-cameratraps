# How much does a new camera cost, and what reduces it? (Shayan)

Frozen-feature experiments on iWildCam (`official_ood`), training and validation cameras only.
The test cameras are not used by anything in this folder. Deck: "Experiment 4"; team doc:
Approaches 2 to 8.

## What is here

| File | What it does |
|---|---|
| `cache_features.py`, `run_cache.sh` | frozen DINOv2 ViT-B/14 features of every train/val image (448 px, [CLS, mean patch]) |
| `pilot_a.py` | data loading, linear head (L-BFGS), burst pooling, the species-mix pilot (Approach 3) |
| `rare_checks.py`, `val_compare.py` | rare-species heads and their validation check (Approach 4) |
| `gap_measure.py`, `gap_compare.py`, `import_features.py` | cost of a new camera: seen vs unseen model on the same query bursts (Approaches 5 and 6) |
| `gap_decompose.py` | the camera's own bursts vs other cameras' bursts of the same species, at equal class mass (Approach 6) |
| `gap_budget.py`, `local_analysis.py` | up to k labelled bursts per new camera; what the local benefit goes along with (Approach 7) |
| `gap_select.py`, `posthoc11.py` | which bursts to label: random, spread out, most uncertain (Approach 8) |
| `run_*.sh` | the exact launchers used on vinzant |
| `APPROVALS.md` | every run: scope, pre-registered rules, reviews and all reported numbers |
| `tests/` | local checks on synthetic features |
| `notes/` | literature notes and the write-up of the early experiments |

As the repo's `.gitignore` asks, results and logs are not committed. They are on vinzant in
`users/shayan/pilot_a/results/` and `logs/` (one folder per run, named as in `APPROVALS.md`). Feature
caches are on vinzant too: `users/shayan/pilot_a/features/` (ViT-B, BioCLIP imports) and
`/zfs/ai4good/groups/camera-traps/features/` (George's ViT-L and ResNet-50, whole and crop).

## Main results (training cameras, camera-wise 5-fold cross-validation, burst-pooled macro-F1)

- Frozen DINOv2 ViT-B + linear head + burst pooling: 0.82 on the validation cameras (ResNet-50 baseline about 0.54).
- Cost of a new camera, whole photo vs crop: DINOv2 ViT-L 11.7 → 5.3 points, ResNet-50 16.3 → 11.1, BioCLIP 11.6 → 8.5.
- At equal species counts, a camera's own earlier bursts (at least a day before its test bursts) are still worth
  5.8 points (95% interval 1.3 to 8.5) for DINOv2 ViT-L.
- Up to 8 labelled bursts per new camera add 1.5 points with crop features (0.1 to 3.1, inconclusive against a 2-point rule).
- Choosing which bursts to label: spread out vs random +1.0 (inconclusive). Exploratory and fragile: the single most
  uncertain burst per camera +3.6 over random with crops, mostly from two rare species; needs replication.

## Running

The `run_*.sh` launchers are the exact ones used on vinzant and point to `users/shayan/pilot_a` there
(the scripts, the feature caches and the outputs); to rerun from this folder, change `$W/pilot_a` in a
launcher. The Python scripts run from the repo root with the repo's environment (they import
`iwildcam`) and this folder on `PYTHONPATH`; each prints its arguments with `--help`.

## AI use

The code, runs and write-ups in this folder were done with AI assistance: Claude (Anthropic) wrote and
ran the code under Shayan's direction, and GPT-6 Astra (OpenAI, via Codex) reviewed every run plan before
it started and every interpretation afterwards. The review record is in `APPROVALS.md`.
