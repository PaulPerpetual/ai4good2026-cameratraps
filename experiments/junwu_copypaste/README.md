# Copy-paste augmentation and background debiasing on frozen DINOv2-L features (Junwu)

Imported from `users/junwu/` on vinzant (2026-11-22). README written at import from the scripts
and result files; Junwu, please correct anything misdescribed.

All three lines of work use the shared frozen DINOv2-L crop features
(`features/dinov2_vitl14_reg_448`) and an SGD logistic-regression head; scores are on the
official validation cameras, per image, seed 42.

## background_debiasing/
Estimate a "background direction" as the mean feature of training-camera empty frames and
project it out of every feature with strength lambda. `debias_results.csv`: no effect
(val macro F1 0.591 at lambda 0 vs 0.593 at lambda 1).

## copypaste/
Paste SAM2-segmented animals onto other empty frames, embed the composites, add them to the
training set (versions v1..v6 of the generator; v6 is the one reported).
`baseline_results/` vs `augmented_results/`: 478 synthetic images, val macro F1 0.628 → 0.644.

## domain_adaptation/
Paste animals onto source-camera or target(test)-camera empty frames, at random or matched
to the scene. `results/five_way_comparison.csv` (weight 5): every variant about +1.6 to
+1.9 macro F1 over the baseline, source and target alike, so the gain is not specific to
seeing target backgrounds. `results/comparison.csv` (B0/B1/B2): a small adapter trained
with a consistency loss between an animal and its composite; B2 collapses the features
(`feature_diagnosis.csv`: different-species cosine 0.03 → 0.82).

## Under the protocol (see PROTOCOL.md)
Not yet rerun. To report under the protocol: choose the synthetic weight on the camera-wise
CV folds, report several seeds, report per burst as well, and use the LDA head next to SGD.
Generated images, features and checkpoints stay in `users/junwu/` (not in git).
