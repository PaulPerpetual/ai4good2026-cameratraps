# Data and evaluation protocol

Why this exists: by November each of us had a working pipeline, but no two of them scored
the same way (different validation sets, metrics, heads, seeds, and different rules for
touching test), so our numbers could not be put in one table. This protocol fixes the data
handling and the evaluation once; methods plug into it. Every fixed value lives in
`iwildcam/protocol.py`; changing one is a protocol change and gets a line in the changelog
at the bottom.

## 1. Task and split

- iWildCam 2020, the 60 species of `iwildcam.data.load_task()` (junk categories removed).
- Split `official_ood`: 216 training cameras, 32 validation cameras, 46 test cameras. No
  camera appears in two parts.
- Images are read from the full-resolution originals (`raw/train`).

## 2. Input: the animal crop

Most confident MegaDetector **v5a** animal box with confidence ≥ 0.2, made square in pixels,
15% padding per side, shifted to stay inside the frame, cut from the original. Below 0.2 or
no box: the whole frame. (`iwildcam/crops.py`; boxes: Pia's v5a run.)

Why: cropping is the single largest gain we measured that is not a backbone change. Pia,
ResNet-50 end to end: whole frame 0.418 → crop 0.503 (test, 3 seeds). Shayan: the cost of an
unseen camera drops from 11.7 to 5.3 points for DINOv2-L with crops. Tighter SAM masks were
not better than the box crop (Pia).

## 3. Backbones and the feature store

Backbones are frozen. Each is an entry in `iwildcam/backbones.py`; images are squashed to the
backbone's input size.

| name | model | input | dim | role |
|---|---|---|---|---|
| `dinov2l` | DINOv2 ViT-L/14 with registers | 448 | 2048 | default |
| `bioclip2` | BioCLIP 2 (`imageomics/bioclip-2`) | 224 | 768 | second backbone |
| `resnet50` | ImageNet ResNet-50 | 448 | 2048 | starter baseline only |

Features are computed once (`scripts/extract_features.py`) into
`/zfs/ai4good/groups/camera-traps/features/`, three folders per backbone:

| folder | contents | who may read it |
|---|---|---|
| `<name>` | train + val cameras, labelled and empty frames | anyone, any time |
| `<name>_testempty` | empty frames of the test cameras | anyone: unsupervised adaptation, allowed by the task |
| `<name>_test` | labelled test images | only through `iwildcam.ledger` (section 6) |

Each folder records how it was made (`config.json`: weights, size, crop rule, hash of the
detections file, row counts) and is complete only once `manifest.json` exists. Read with
`iwildcam.store.load()`. Do not write into another person's folder; make a new one.

Adding a backbone: one `Spec` in `backbones.py`, then run the extraction for the three folders.

## 4. Choosing settings

Every setting (classifier regularisation, augmentation weight, threshold, epoch...) is chosen
by the **mean over the 5 camera-wise cross-validation folds of the training cameras**:
fit on 4 folds' cameras, score the held-out fold's cameras.

- Folds are fixed in `data/camera_folds.csv` (committed): cameras sorted by region, random
  within region, dealt round-robin (Shayan's scheme). Use `iwildcam.folds.fold_of()`.
- The 32 official validation cameras are reported after the choice, as a check, with a
  camera-bootstrap 95% interval. They are not used to choose. Reason: with 32 cameras the
  interval on per-burst F1 is about ±5 points (DINOv2-L: 0.87–0.97), wider than most
  differences we care about; DINOv2 vs DINOv3 and several of our ablations were inside it.
- Selection metric: mean per-image macro F1 over the folds (per burst is near its ceiling and
  moves in coarse steps; the burst vote is applied on top of the chosen setting anyway).
- Anything stochastic (training, augmentation sampling): at least 3 seeds, report mean ± std.

## 5. Scoring

- **Macro F1** over the species present in the evaluated set (`iwildcam.metrics.macro_f1`).
- Always report two numbers, **per image · per burst**. Per burst uses the
  **detector-weighted burst vote** (Paul): each photo votes for its prediction with weight =
  MegaDetector's animal confidence; the burst's winner is given to all its photos. On
  ResNet-50 this is worth about +10 points over per image; an unweighted vote about +3.
- Intervals: `iwildcam.metrics.camera_bootstrap` resamples whole cameras (images of one
  camera are not independent). For a comparison, bootstrap the difference, not each side.
- Classifier default: the camera-aware linear discriminant (George, `iwildcam/lda.py`),
  fitted on the training cameras only, each burst counting once. A method that needs another
  head reports the LDA result next to its own.

## 6. Test access

- The test cameras are scored **once per final configuration**. A final configuration is a
  JSON file in `configs/final/` produced after section 4; it pins the backbone, the head and
  every chosen value.
- Test features are read only through `iwildcam.ledger.load_test()`, which appends who/when/
  commit to `results/test_ledger.jsonl` and refuses a configuration already scored. The ledger
  is committed. `iwildcam.pipeline --test` also checks that official val reproduces the number
  in the config before opening test.
- Ablations, sweeps and "which is better" questions are answered on CV + official val, never
  on test.
- Results that used test before this protocol (Pia's per-mode test numbers, the
  ensemble one-off, George's exp3) stay valid as reported, but are marked "pre-protocol" and
  are not reused to pick anything.

## 7. Using test-camera empty frames

Allowed (no labels involved). Report such methods as **target adaptation**, separately from
source-only methods, and also run the same method with training-camera empty frames
(source backgrounds), so we see whether the target frames themselves help. (Pia's and Junwu's
designs already do this.) Validation analogue: for CV, the held-out fold's empty frames play
the role of the target frames.

## 8. What is consolidated and what is a separate experiment

**Consolidated** means it runs inside the one pipeline, from one YAML config:

```
crop ──► frozen backbone ──► feature store ──► head ──► burst vote ──► score
MD v5a   dinov2l | bioclip2                 lda |      detector-      CV folds · official val
(Pia)    (George, Paul)     (George)        lda_cam    weighted       · test once via ledger
                                            (George)   (Paul)         (Shayan's folds)

uv run python -m iwildcam.pipeline configs/pipeline/dinov2l_lda_cam.yaml [--test]
```

Every run writes its frozen settings to `configs/final/<name>.json` and a row to
`results/runs.jsonl` (committed), so all pipeline results land in one table.

**Separate experiments** live in `experiments/<person_topic>/` with their own code, imported as
they were. Their numbers were produced before this protocol and are **not directly
comparable** to pipeline numbers. Each one could be brought into the pipeline as follows:

| experiment | what it is | how it would enter the pipeline | effort |
|---|---|---|---|
| `pia_background` | background replacement / SAM cutouts, ResNet-50 end to end | as an **augment** stage: paste animals onto source or target empty frames, embed with the frozen backbone, add to training (same stage as Junwu's) | GPU to embed composites |
| `junwu_copypaste` | copy-paste onto source/target empty frames, frozen DINOv2-L + SGD | the same **augment** stage; weight of synthetic rows chosen on CV | composites exist for DINOv2-L; re-embed for BioCLIP 2 |
| `george_followup` long tail | prior adjustment tau·log(prior) on scores | a **head option** (`prior_tau`), chosen on CV | small, CPU |
| `george_followup` calibration | temperature scaling, ECE | an extra **report** next to F1 | small, CPU |
| `shayan_new_camera_cost` | cost of an unseen camera; k labelled bursts per camera | an **evaluation mode** (k bursts per held-out camera); switch to `data/camera_folds.csv` | moderate, CPU |
| `george_exp3` | the original of the shared core; LoRA fine-tuning | core already consolidated; fine-tuning stays separate (does not fit frozen features and did not help DINOv2-L) | n/a |
| `george_followup` efficiency | compute cost vs accuracy, MD v6-compact | stays separate (a deployment study, not a method) | n/a |
| `paul_burst_vote_backbones` | burst vote, DINOv2 vs v3, BioCLIP fine-tuning | burst vote and backbone choice are consolidated; fine-tuning stays separate | n/a |

Adding a stage: a config key, a function in `iwildcam/pipeline.py`, a line in this table and
in the changelog. Until a method is in the pipeline, report it in its README with the
protocol's scoring (sections 4–5) so it can at least be placed next to pipeline numbers.

## 9. Repository layout

```
iwildcam/             shared code
  protocol.py           every fixed value above
  crops.py              detector boxes, crop rule, Crops dataset
  backbones.py          backbone registry
  store.py              feature store: row selection, reading
  folds.py              camera-wise CV folds
  lda.py                linear discriminant, plain and camera-aware (George)
  metrics.py            macro F1, burst vote, camera bootstrap
  ledger.py             test access
  pipeline.py           the consolidated pipeline (one YAML config per run)
  data.py, train*.py    starter code (unchanged)
scripts/
  extract_features.py   fill the feature store (resumable)
  run_store.sh, when_gpu_free.sh
experiments/<person_topic>/   each person's work, imported as it was, with a README
configs/pipeline/     one YAML per pipeline run
configs/final/        frozen settings chosen by each run
results/runs.jsonl          every pipeline run (committed)
results/test_ledger.jsonl   every test access (committed)
```

A new experiment goes in `experiments/<name>/` with a README stating: the question, which
store folders it reads, how settings were chosen, and its numbers under this protocol.

## 10. Status and open points (2026-11-22)

| | under protocol? | note |
|---|---|---|
| DINOv2-L crop features (train/val, test) | yes | existing store; `_testempty` queued |
| BioCLIP 2 crop features | queued on GPU | `logs/store.log` |
| DINOv2-L + camera-aware LDA | selected (`configs/final/dinov2l_lda_cam.json`), test not scored | CV 0.707 · 0.898; val 0.795 · 0.933 (shrinkage 0.01, camera weight 2) |
| BioCLIP 2 + camera-aware LDA | after extraction | |
| Pia background modes | separate | see section 8 |
| Junwu copy-paste | separate | see section 8 |
| Shayan new-camera cost | already CV-based | uses its own folds; switch to `camera_folds.csv` |
| George follow-ups | separate | see section 8 |

Open:
- Contamination: MegaDetector was probably trained on WCS images (iWildCam's source); check
  whether BioCLIP 2's training data (TreeOfLife-200M) contains iWildCam/LILA camera-trap images.

## Changelog
- 2026-11-22: single entry point `iwildcam/pipeline.py` with YAML configs; section 8 lists what is
  consolidated and what stays a separate experiment.
- 2026-11-22: first version (Paul). Selection by per-image CV macro F1 (decided over per burst,
  which for DINOv2-L would pick shrinkage 0.3 / camera weight 16 instead of 0.01 / 2).
