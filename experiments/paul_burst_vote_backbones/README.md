# Detector-weighted burst vote, frozen vs fine-tuned, and backbone choice (Paul)

Imported from `users/paul/camera-trap/` on vinzant (2026-11-22). Code unchanged; the scripts
expect to run from that folder (they import `linprobe` from it and George's `scripts/score_test.py`).

## 1. Detector-weighted burst vote (`linprobe.py votes`, now `iwildcam/metrics.py`)

A camera trap fires a burst of photos of the same animal; they should get one label. Each
photo votes for its predicted species, weighted by MegaDetector's confidence that it shows an
animal (blurry or empty-looking frames count less); every photo then gets the burst's winner.
Adopted by the whole team as the per-burst score.

ResNet-50 fine-tuned, official OOD test, 3 seeds (`results/votes.csv`):

| rule | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| per image | 0.405 | 0.426 | 0.385 |
| equal vote | 0.410 | 0.460 | 0.416 |
| detector-weighted vote | **0.495** | **0.539** | **0.510** |

The weighting carries almost all of the gain: about +10 points vs +3 for an unweighted vote.

## 2. Frozen vs fine-tuned

- BioCLIP 1 on crops: frozen linear probe 0.547 per image (test, `results/linprobe.csv`);
  LP-FT + WiSE-FT 0.647–0.672 per image, 0.837–0.858 per burst (test, 3 seeds, `results/finetune.jsonl`).
  Fine-tuning helps BioCLIP a lot.
- BioCLIP 2 LP-FT (`finetune_bioclip2.py`): official val 0.746 · 0.930, best epoch = last
  (stopped by the 55-minute cap), so not converged.
- Frozen DINOv2-L + camera-aware LDA (George): official val 0.793 · 0.946; George's LoRA
  fine-tuning did not improve it (test 0.741 → 0.740). SpeciesNet fine-tuning: best epoch 0.
- So: the strongest model needs no fine-tuning, and the frozen-feature setup is what makes
  quick, controlled ablations possible for the rest of the team.

## 3. DINOv2 vs DINOv3 (`v3_vs_v2.py`, `results/v3_vs_v2.txt`)

Same crops and LDA, official val, camera bootstrap: DINOv3-L at 448 px is 0.011 lower per image
(95% CI −0.024 to +0.022) and 0.015 lower per burst (−0.046 to +0.018) than DINOv2-L. DINOv2-g:
higher per image, lower per burst (`results/lda_val.csv`). No reason to move off DINOv2-L, which
is the protocol's default.

Not imported (kept in `users/paul/camera-trap`): SpeciesNet, the SN+DINOv2 ensemble, iNat and
taxonomy LDA.
