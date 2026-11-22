# Follow-up: calibration, rare species, compute cost and image resolution

The TA's suggested directions that George took on. Everything builds on `../exp3` (same
features, classifier and metrics); run the scripts from this folder on vinzant.
Results go to `/zfs/ai4good/groups/camera-traps/users/george/followup_results/`.

Rules kept throughout: every choice is made on the validation cameras (or held-out training
cameras), test is scored once per final setup, both backbones (ResNet-50, DINOv2-L) are run,
and no backbone was trained on iWildCam.

## 1. Calibration — `calibration.py`

Can the classifier's confidence be trusted, and can it decide which photos a human checks?
The step-2 classifier's scores become probabilities with a temperature fitted on validation
(photos and bursts each get their own). Reports ECE and NLL before and after, per photo and
per burst, and the macro F1 of the photos kept when the least confident go to a human.

## 2. Rare species — `long_tail.py`

Species split into thirds by training bursts (rare / medium / common). Two changes to the
classifier, all combinations scored on validation, the best scored once on test next to the
plain classifier:
- prior adjustment: add tau × log(species' share of training bursts) to the scores
- species balance: how much each species, rather than each burst, counts in the
  within-species variation

## 3 + 4. Compute cost and image resolution

| Script | What it does | Where it runs |
|---|---|---|
| `efficiency_cost.py` | parameters, GFLOPs, 4-thread CPU latency of each backbone at each input size | CPU |
| `detector_cost.py` | the same for MegaDetector v5a and v6-compact at 1280 px | CPU, `.venv-megadetector` |
| `detect.py` | runs MegaDetector v6-compact over every labelled photo | GPU, `.venv-megadetector` |
| `run_features.py` | features of every combination in `combinations.py` | GPU |
| `efficiency_accuracy.py` | camera-aware classifier per combination (settings chosen on held-out training cameras); OOD and ID macro F1; accuracy by animal size | CPU |
| `efficiency_report.py` | cost of the whole pipeline (detector + backbone) against accuracy, the frontier picked on validation, figures | CPU |

`run_gpu.sh` runs `detect.py` then `run_features.py --all`, started through
`../scripts/wait_for_gpu.sh` so it only uses the shared GPU when it is free.

CPU latency is measured on vinzant's CPU with 4 threads as a stand-in for a 4-core
Raspberry-Pi-class computer; only the comparison between models is meaningful.

MegaDetector runs in its own environment (`users/george/.venv-megadetector`, PytorchWildlife
1.3.1). Note for the report: MegaDetector was trained on public camera-trap collections that
probably include WCS photos, iWildCam's source; this applies to the detector the whole team uses.

## Helpers

`common.py` (the OOD and ID settings), `combinations.py` (what the efficiency study covers),
`plots.py` (figures), `exp3_path.py` (makes `../exp3` importable).
