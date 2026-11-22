# Experiment 3: frozen and fine-tuned backbones on MegaDetector crops

Every Experiment 3 number on the slides comes from one of the commands below. Run them from
this folder on vinzant (`bash run_all.sh` runs all of them in order).

## Pipeline

```
photos ──MegaDetector crop──► backbone ──► 2048 features per photo        features.py   (GPU, once)
                                                │
                    ┌───────────────────────────┴───────────────────────────┐
                    ▼                                                       ▼
   linear discriminant on the saved features              fine-tune the backbone on the crops
   steps 1-3                                frozen.py      step 4         finetune.py  (GPU, hours)
                    │                                                       │
                    ▼                                                       ▼
   OOD + ID macro F1, per photo and per burst            OOD macro F1     score_finetuned.py
```

## Results and the command behind each

Macro F1, per photo · per burst. OOD = the 46 test cameras never seen in training; ID = the
random_burst split, whose test photos come from cameras also seen in training.

| Step | What changes | Command | ResNet-50 OOD | ResNet-50 ID | DINOv2-L OOD | DINOv2-L ID |
|---|---|---|---|---|---|---|
| 1 | crop + plain classifier | `frozen.py --step 1` | 0.531 · 0.739 | 0.630 · 0.792 | 0.732 · 0.902 | 0.775 · 0.925 |
| 2 | camera-aware classifier | `frozen.py --step 2` | 0.532 · 0.754 | 0.627 · 0.797 | 0.734 · 0.916 | 0.761 · 0.931 |
| 3 | + four views averaged | `frozen.py --step 3` | 0.549 · 0.758 | 0.641 · 0.806 | 0.741 · 0.919 | 0.764 · 0.932 |
| 4 | fine-tuned backbone | `score_finetuned.py` | 0.568 · 0.787 | – | 0.740 · 0.919 | – |

Add `--backbone resnet50` or `--backbone dinov2l` to each command. Step 1 with DINOv2-L is
Experiment 3 of presentation 2. Step 4 for ResNet-50 is the mean of 3 seeds (± 0.003 · 0.002).

## Files

| File | What it does |
|---|---|
| `settings.py` | paths, and every fixed setting with where it was chosen |
| `data.py` | which photos, the MegaDetector crop, loading saved features, the two splits |
| `lda.py` | the linear discriminant, plain or camera-aware |
| `metrics.py` | the burst vote and macro F1 |
| `models.py` | the backbones, the network for fine-tuning, LoRA adapters |
| `features.py` | backbone over every crop, saves the features |
| `frozen.py` | steps 1-3 |
| `finetune.py` | step 4: trains and saves the weights |
| `score_finetuned.py` | step 4: scores the saved weights |
| `run_all.sh` | every command, with the expected numbers |

## How the steps work

**Crop.** The most confident MegaDetector animal box (confidence ≥ 0.2), made square, 15%
padding on each side, cut from the full-resolution photo. Below 0.2, the whole photo. Same
rule as Experiment 2.

**Plain classifier (step 1).** Each species gets a centre, its average feature vector. A
photo gets the species with the closest centre, measured against how much photos of one
species normally vary. Each burst counts once when fitting.

**Camera-aware classifier (step 2).** The same, but how each species' centre shifts from
camera to camera is also counted as normal variation, so the classifier relies less on what
changes with the camera (light, background).

**Four views (step 3).** Features of the crop, its mirror image, a tighter crop (5% padding)
and a looser one (40%) are averaged.

**Fine-tuning (step 4).** Starts from step 2's classifier. ResNet-50: all layers, 6 epochs,
learning rate 1e-5, the best epoch on validation kept. DINOv2-L: frozen, with LoRA adapters
(rank 8) on the 96 linear layers of its 24 blocks, about 1% of the weights, 3 epochs; at test
time the adapters' change is used at 25% strength, the best on validation.

**Burst vote.** The photos of a burst vote on its species; each vote counts as much as
MegaDetector's confidence that the photo shows an animal.

## What is not rerun

- **Choosing the settings.** The shrinkage and camera weight in `settings.py` were chosen on
  held-out training cameras (Shayan's `val_compare.py` for step 1, `scripts/explore2.py` for
  steps 2-3); the four views were tested in `scripts/explore3.py`. These searches are not part
  of this folder; their chosen values are.
- **Training.** `finetune.py` refuses to overwrite the weights behind the slides. Training
  again would take hours and give slightly different numbers.
- **Test once.** Each setting was fixed before the test cameras were scored. The scripts here
  reproduce those scores; before scoring test, each checks that validation still gives the
  numbers the settings were chosen with.

Also tried and dropped (not on the slides): full fine-tuning of DINOv2-L, LoRA on ResNet-50,
and leaving photos without a detection out of fitting.
