# Camera-trap project: idea proposals and literature notes

7 October 2026. Prepared by Claude (Opus 5.5) together with ChatGPT (via Codex), for Shayan / Group A.

AI-use note: everything in this file was produced by AI tools. The course requires that to be disclosed, and you need to be able to defend each claim yourself. Novelty statements below are "we searched and did not find it", not proof.

Marking used below: **[C]** = I (Claude) opened the source myself. **[G]** = read by ChatGPT only; I did not open it.

---

## 1. Status

- **Baseline: already run by a teammate** (5–6 October), checked on 10 October once cluster access worked. Results are in section 1a. No re-run was needed.
- **Research: done.** Three rounds with ChatGPT: independent reviews, a mutual critique, and a final check.
- **Outcome:** two finalist ideas (A and B), one reserve, and a list of directions that are already taken.
- **Update, 10 October:** idea A was piloted on the validation cameras and is not supported in this form; see section 5a. The pilot also produced a much stronger baseline than the ResNet.
- **Update, 20 October:** changing the classifier gave large gains at held-out training cameras but none that could be shown on the validation cameras; see section 5d. Work on the classifier head is paused.
- **Update, 20 October (later):** the new-camera cost measured on the team's other features (section 5e). A bigger backbone helps everywhere without a shown reduction of the gap; MegaDetector crops shrink BioCLIP's gap from 11.6 to 8.5 points.

## 1a. Baseline results (added 10 October)

Read from `results/runs.jsonl` and the saved predictions in the team's shared clone on vinzant (`/zfs/ai4good/groups/camera-traps/camera-trap`). ResNet-50 at 448 px, 12 epochs, 3 seeds per split.

| Split | Test macro-F1 (mean of 3 seeds) | Per seed | Accuracy | README target |
|---|---|---|---|---|
| `official_ood` (unseen cameras) | 0.405 | 0.405 / 0.426 / 0.385 | 0.595 | 0.4005 |
| `random_burst` (same cameras) | 0.618 | 0.622 / 0.623 / 0.610 | 0.753 | 0.6277 |

- **Time:** about 4.5 hours per run with two runs sharing the GPU, 13.5 hours for all six.
- **Seed noise:** the three unseen-camera seeds span 4 points. A method that gains 2 points on 3 seeds has shown nothing yet.
- **Validation peaks early** (epoch 3, 5 and 9 for the three unseen-camera seeds) and then declines.
- **Burst voting is a free gain.** Giving every image the majority prediction of its burst, with no retraining, lifts unseen-camera macro-F1 from 0.405 to 0.429 and accuracy from 0.595 to 0.655. Any new method has to be compared against this.
- **Rare species are where the score is lost.** The 22 classes with fewer than 50 training bursts average F1 0.18; the other 35 average 0.54. Seven classes are below 0.1.
- **Night is not the problem:** accuracy is 0.63 at night and 0.59 by day.
- **What was not saved:** only the test predictions (the chosen class per image). No scores, no validation predictions, no model weights. The pilots for ideas A and B need scores on validation cameras, so they need a new run or cached features.

These are test-set numbers, used here only to describe the baseline.

## 2. What the metadata says

All numbers are for the `official_ood` split, computed from `data/metadata.csv.gz`. Scripts are throwaway; ask me to regenerate them.

Caveat: my first pass used the **test** labels for descriptive statistics. ChatGPT flagged this. I re-ran the important ones on the **validation** cameras, and those are the ones to quote. Keep all further hypothesis selection on validation cameras, and say in the paper that test-set statistics were inspected.

**Each camera sees very few species.**
- Validation cameras: median 4 species (IQR 3–6) out of 60. Training cameras: median 3.
- Cameras with GPS fall in two regions that share no species at all in training: Americas (29 species) and Africa (25).

**How much does knowing the camera's species mix tell you?** Mean log-probability the prior gives to the true species, per burst, on validation cameras with GPS (higher is better; uniform guessing is −4.09):

| Prior | Log-prob | Gain over region |
|---|---|---|
| Global training frequencies | −3.57 | |
| Region (continent) | −2.70 | 0 |
| 10 nearest training cameras by GPS, with shrinkage | −2.33 | 0.37 |
| 5 labelled bursts from the same camera | −2.15 | 0.55 |
| 20 labelled bursts from the same camera | −1.85 | 0.85 |
| The camera's own mix, leave-one-burst-out | −1.70 | 1.00 |

So about one nat of usable information sits in the camera-level mix, and GPS neighbours recover roughly a third of it. A whole-image model very likely already knows the region from the background.

**Cameras are small samples.** Validation cameras have a median of 22 bursts; 12 of 32 have fewer than 10. Any estimate of a camera's mix from its own unlabeled images is a small-sample problem.

**Bursts are the real unit.** Median 10 frames per burst, 99% of images are in bursts of 3 or more, about 98% of bursts contain one species. The 85,265 training images are 6,770 bursts.

**Empties are uneven and mostly daytime.**
- Validation: 25 of 32 cameras have any; counts run 1535, 1419, 415, 228, … down to 3.
- Test: 34 of 46 cameras have any; one camera holds 54% of them; 12 of those 34 have fewer than 20.
- Validation empties are 20% night; validation animal images are 30% night. On test it is 7% versus 22%.
- Training: 186 of 216 cameras have empties (median 56). Another 19 training-fold cameras have empties but no labelled images of the 60 species. (Corrected on 10 October: an earlier version said 205 of 216, which counted those 19.)

**Rare classes and metric noise.**
- 10 classes have fewer than 20 training bursts; 25 have fewer than 50.
- Validation contains only 45 of the 60 species, so model selection gets no feedback on some test classes.
- Test (from the first pass): 57 classes, 7 of them with 2 bursts or fewer. Each class is worth 1.75 macro-F1 points, so a couple of bursts can move the headline number.

**Test cameras sit among training cameras.** Median distance to the nearest training camera is 1.3 km (GPS only, no labels used).

## 3. Literature, by theme

### The benchmark and its leaders
- [WILDS leaderboard](https://wilds.stanford.edu/leaderboard/) **[C]**. iWildCam OOD macro-F1: AutoFT (CLIP ViT-L/14@336) 52.0; FLYP 46.0; model soups 43.3; ERM with CLIP ViT-L 41.4; ERM with targeted augmentation (ResNet-50) 36.5; plain ERM ResNet-50 about 31–32. With unlabeled data: Connect Later 36.8; Noisy Student 32.1; DANN 31.9; CORAL 27.9.
- Same leaderboard: test-time BatchNorm adaptation scores 13.8 and ARM-BN 23.3, far below ERM. Naive per-camera normalisation is destructive here.
- The leaders are generic robust fine-tuning methods for CLIP. None uses camera-trap structure (empties, bursts, location).
- These numbers are **not comparable** with the course protocol (60 classes, re-annotated labels, different training allocation).
- [Sagawa et al., WILDS 2.0](https://arxiv.org/pdf/2112.05090) **[C, search summary]**: methods using unlabeled data had limited success on WILDS.

### Background, crops and copy-paste
- [Beery et al. 2018, Recognition in Terra Incognita](https://arxiv.org/abs/1807.04975) **[C]**: the original demonstration that classifiers fail at new locations.
- [Gao et al. 2023, Targeted Augmentations](https://arxiv.org/abs/2302.11861) **[C]**: copy-paste onto empty backgrounds from training cameras where the same species occurs gives 36.5, versus 34.7 for any background and 30.2 without. Lesson: background carries useful habitat information; erasing it blindly costs accuracy.
- [Mesnage et al. 2025](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12588685/) **[C exists; G for the detail]**: copy-paste improves unseen-location classification; according to ChatGPT's reading of the methods they paste onto empties from the test locations.
- [WildFiT (SenSys 2026)](https://arxiv.org/abs/2409.07796) **[C]**: synthesises training data against local backgrounds and fine-tunes on-device.
- [Gadot et al. 2024, To crop or not to crop](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cvi2.12318) **[G]**: on iWildCam-WILDS, whole-image 49.0, MegaDetector crops 51.9, both combined 54.2 OOD macro-F1 (EfficientNetV2-M, filtered training labels). The authors note MegaDetector was trained on some iWildCam data. Any background-removal idea has to be compared against crops.
- [Yu et al. 2026, Instance-Free Domain Adaptive Object Detection](https://arxiv.org/abs/2602.06484) **[C]**: adapts a detector using target images that contain no objects, with a wildlife benchmark. The setting "target domain gives you only background" is therefore already named, for detection.

### Context: location, time, sequences
- [Pahuja et al., COSMO](https://arxiv.org/abs/2401.00608) **[C]**: location, time and taxonomy in a knowledge graph, but only during training; inference uses the image alone. Leaderboard 32.3, about the same as ERM.
- [Mac Aodha et al. 2019, geographical priors](https://arxiv.org/abs/1906.05272) **[C]**: a separately trained location prior multiplied into the classifier's output at inference. "An ecological prior helps classification" is therefore not new.
- [Haucke et al. 2025, Deep Multi-modal Species Occupancy Modeling](https://www.biorxiv.org/content/10.1101/2025.09.06.674602v1.full) **[C]**: DINOv2 embeddings of one blank daytime camera-trap image per site as predictors of species occupancy. Closest precedent for "empties predict the local species pool"; its goal is occupancy modelling, not classification.
- [Bao & Karaletsos, ContextViT](https://arxiv.org/abs/2305.19402) **[C exists; G for numbers]**: a per-camera context token inferred from that camera's images; iWildCam accuracy gains of about 1.5–2 points.
- [Context R-CNN](https://arxiv.org/abs/1912.03538) **[C]**: attention over a per-camera memory of other frames, for detection.
- [Dussert et al., attention across detections](https://www.biorxiv.org/content/10.1101/2025.07.15.664849v1.full) **[C]**: self-attention across crops in a sequence, tested on held-out locations. ChatGPT also points to Dussert et al. 2024 on sequence averaging and temperature scaling **[G]**.
- DeLoCo (Ecological Informatics, 2025) **[partly verified]**: decoupled species and location heads on iWildCam; we found no use of target-camera empties, but could not read the full text.
- [SpeciesNet](https://github.com/google/cameratrapai) **[G]**: trained on 65M camera-trap images with geographic rules. Almost certainly saw these cameras; do not use it as a clean baseline.
- [WildIng](https://arxiv.org/abs/2601.00993) **[C]**: text descriptions plus BioCLIP for Africa/America shift.

### Label shift and test-time adaptation
- [Sun et al. 2023, TTLSA](https://arxiv.org/abs/2211.15646) **[C]**: train with the nuisance variable, then re-estimate the joint prior with EM at test time. Nearest relative of idea A; it assumes the nuisance values were seen in training.
- [Park et al. 2023, Label Shift Adapter](https://arxiv.org/abs/2308.08810), [Garg et al. 2023, RLSbench](https://arxiv.org/abs/2302.03020v2), [Boudiaf et al. 2022, LAME](https://arxiv.org/pdf/2201.05718) **[C exist]**: adaptation under combined covariate and label shift; LAME adjusts outputs only and suits correlated test streams.
- [Ziegler & Czyż 2024](https://openreview.net/pdf?id=Ft4kHrOawZ), [Ye et al. WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Ye_Label_Shift_Estimation_for_Class-Imbalance_Problem_A_Bayesian_Approach_WACV_2024_paper.html), [dynamic Dirichlet priors](https://arxiv.org/abs/2511.18615) **[C exist]**: Bayesian and MAP label-shift estimation is established.
- [Liu et al. 2023, logit correction](https://arxiv.org/abs/2212.01433v2) **[C]**, [BalPoE](https://arxiv.org/abs/2206.05260) **[G]**, [BoDA](https://arxiv.org/abs/2203.09513v2) **[C]**: prior-in-the-loss training and multi-domain long-tailed recognition.

### Building blocks
- [AnomalyDINO (WACV 2025)](https://arxiv.org/abs/2405.14529) **[C exists]**: few-shot anomaly localisation from a memory bank of DINOv2 patch features of a few normal images.
- Classical camera-trap background subtraction, e.g. [Animal Scanner](https://onlinelibrary.wiley.com/doi/full/10.1002/ece3.4747) **[C exists]**.
- Backbones to consider: DINOv2/DINOv3, [BioCLIP](https://arxiv.org/abs/2311.18803), [BioCLIP 2](https://arxiv.org/abs/2505.23883) **[G]**. Check each checkpoint's training data before claiming it is clean.

## 4. Guidelines we followed for finding ideas

- **Goal-driven, not idea-driven** ([Schulman's guide](https://pengsida.net/files/John_Schulman_How_to_do_research.pdf)): start from the measured failure and the measured headroom, aim high, and climb through small experiments.
- **Novelty is an insight, not complexity** ([Michael Black, Novelty in Science](https://perceiving-systems.blog/en/news/novelty-in-science)): a simple method that connects two fields or explains a failure counts.
- **ChatGPT's four heuristics:** invert an assumption (the target data contain none of the classes); split the problem into causes (appearance nuisance versus ecological information); subtract the nearest prior work and see what is left; prefer ideas that can fail cheaply.
- **Two independent reviewers, then attack each other's ideas.** Both of us arrived at similar top-two ideas independently, which is a warning that others may have too.

## 5. Idea A: factor out the camera's species mix, then re-estimate it at a new camera

**In plain words.** Each camera sees a handful of species. A normal classifier quietly learns "this background means these species", which is wrong at a new camera. Instead, hand the model each training camera's species mix during training, so the image part only has to learn what the animal looks like. At a new camera, plug in a mix estimated for that camera.

**Method.**
1. *Training.* Head `f` on frozen features, with loss `CE(f(x) + log π_c, y)`, where `π_c` is the training camera's burst-level species mix.
   - Estimate `π_c` on a different block of bursts than the one being trained on (cross-fitting), with a floor and shrinkage toward the regional mix.
   - Weight by burst, not by image. Try a tempered offset `τ · log π_c`.
2. *Context expert.* Predict an unseen camera's mix from its empty frames: kernel regression over training cameras, using similarity between mean empty-frame embeddings, optionally combined with GPS neighbours.
3. *Refinement.* Update that prediction with the camera's own unlabeled bursts: one latent species per burst, frames pooled with an effective-sample-size correction, posterior mean rather than MAP.
4. *Prediction.* `f(x) + log π̂_c`, aggregated over the burst. Tune the decision rule for macro-F1 on validation cameras.

**Why it might be new.** The ingredients are known (product-of-experts debiasing, logit correction, EM for label shift, geo priors). What we did not find is the prior being trained inside the loss and then **re-estimated for a camera never seen in training**, from background-only frames plus a small unlabeled stream. ChatGPT also raised the point that makes this more than plumbing: a whole-image classifier already contains a background-driven prior, so multiplying another prior on top counts the habitat twice. Showing that failure and fixing it is the contribution.

**Decisive pilot (validation cameras).** A grid: {ordinary head, prior-in-loss head} × {no prior, GPS prior, empties prior, refined prior, the camera's true mix}, all with identical burst pooling.

**Stop rules.**
- True-mix gain under about 2 macro-F1: drop A.
- Empties prior no better than GPS in held-out log score and macro-F1: drop the empties claim.
- Prior-in-loss head no better than the ordinary head with a prior-ratio correction: drop the method claim.

**Baselines a reviewer will ask for.** Calibrated burst averaging; standard EM and Ye et al.'s MAP estimator; LAME; a GPS prior in the style of Mac Aodha; crop + whole-image fusion; two backbones.

**Main risks.**
- The model mistakes an appearance shift for a change in species mix and becomes confidently wrong.
- Many cameras have fewer than 10 bursts.
- What triggers a camera is not the same as which species live nearby (activity, placement, detectability).

**Cost.** Cache global embeddings once (up to 3 spaced frames per training burst, all validation frames, up to 16 empties per camera). ChatGPT's planning estimate: 6–10 GPU-hours including three head seeds; the inference sweeps run on CPU. Not measured.

**Odds (ChatGPT's estimate, which I share):** 40–55% for a repeatable gain over a strong matched baseline; 35–50% for a workshop paper; 5–10% for a main conference.

## 5a. Pilot result for idea A (10 October 2026)

**Outcome: the pilot does not support idea A in this form.** Scripts, approvals, logs and full results are in `pilot_a/`. Every run was approved by both Claude and Astra (GPT-6 Astra) beforehand; see `pilot_a/APPROVALS.md`.

**Setup.** Frozen DINOv2 ViT-B/14 features of whole images at 448 px, linear heads, all tuning by camera-wise cross-validation inside the 216 training cameras, then one scoring of the 32 validation cameras (14,652 images, 1,109 bursts, 45 species). No test image was opened.

**Validation cameras, macro-F1 / accuracy.**

| Method | Macro-F1 | Accuracy |
|---|---|---|
| Ordinary head, per image, as trained | 0.724 | 0.850 |
| Ordinary head, pooled over bursts and calibrated | 0.824 | 0.936 |
| … with the regional mix | 0.827 | 0.941 |
| … with the mix predicted from GPS neighbours | 0.816 | 0.932 |
| … with the mix predicted from empties | 0.820 | 0.931 |
| … with GPS and empties combined | 0.819 | 0.933 |
| … with a reference mix that uses labels of half the camera's bursts | 0.841 | 0.962 |
| Prior-in-loss head with the regional mix | 0.792 | 0.932 |
| Prior-in-loss head with the same reference mix | 0.844 | 0.959 |

**The three questions, each with one comparison fixed from the training cameras beforehand.**

| Question | Comparison | Difference in macro-F1 | 95% interval | Margin | Outcome |
|---|---|---|---|---|---|
| K1: is the mix worth knowing? | ordinary + reference mix vs ordinary pooled | +0.017 | −0.029 to +0.040 | 0.02 | inconclusive |
| K2: do empties add to GPS? | ordinary + GPS and empties vs ordinary + GPS | +0.003 | −0.014 to +0.008 | 0.01 | not supported |
| K3: is prior-in-loss better? | prior-in-loss + region vs ordinary + region | −0.034 | −0.054 to +0.013 | 0.01 | inconclusive, leaning against |

- **K2.** The prespecified empties combination did not meet the one-point criterion. This is one pipeline with fallbacks, and only 16 of the 32 validation cameras have both GPS and empties.
- **K3.** Prior-in-loss was below the ordinary head for every usable prior (by 1.3 to 14.6 points) and level with it only under the reference mix (+0.3). That justifies setting it aside; it does not prove it fails in general.
- **K1.** Small and uncertain. The prior-in-loss head gains 7.5 points from the reference mix over its own global baseline, so it depends heavily on an accurate mix.

**Exploratory only, not to be claimed.** Adjusting the mix on the camera's own unlabelled bursts raised the ordinary head's validation macro-F1 by 1.1 to 3.5 points, with every interval including zero, but lowered it in cross-validation on the training cameras. The signs disagree.

**What this does and does not show.**
- Whole-image features from one backbone, validation cameras only. Nothing is shown about crop features, where the background is removed and a prior might matter more.
- Why priors added little is not established. That the head already uses the habitat it sees is plausible; noisy prior estimates, calibration, or the likelihood factorisation failing are alternatives.
- Several tuned values sit on the edge of their grids (regularisation at the top, temperature at the bottom for prior-in-loss, EM strength at the top). This may handicap prior-in-loss and understate the baseline.
- Fold and burst-halving seeds were not varied. All but one fit stopped before the iteration limit with gradients up to 1e-4; they are not certified converged.
- The tuning log scores for GPS and empties cover different sets of cameras and are not a controlled comparison.

**The more useful finding: a much stronger baseline.** On the same validation cameras the teammate's fine-tuned ResNet-50 reached about 0.54 macro-F1 per image (its best epoch, so selected on validation). The frozen DINOv2 linear head reaches 0.724 per image, and 0.824 once predictions are pooled over bursts and calibrated. These are complete-pipeline comparisons. They set a stronger practical baseline; they do not show that the cross-camera gap belonged to the ResNet, because seen-camera performance of the new pipeline has not been measured.

**Next steps agreed with Astra, in this order.**
1. Stop developing idea A for now. Keep these results as they are.
2. Measure how large the unseen-camera gap still is with the new pipeline, on matched query bursts with the same classes: train once with other bursts of the query cameras allowed, once with those cameras excluded. This needs no test data.
3. Look at the errors by species rarity, camera, day or night and animal size before deciding on idea B.
4. Only then, with the next method, the baseline and the evaluation plan written down, open the test cameras once.

## 5b. How much an unseen camera still costs (10 October 2026)

A measurement on the 216 training cameras, with the same frozen DINOv2 pipeline. No validation or test camera was used. Script `pilot_a/gap_measure.py`, results in `pilot_a/results/gap_v1/`, approved by Astra and Claude beforehand.

**Setup.** Five camera-wise folds. For every camera with at least two bursts (196 cameras), half the bursts are queries (3,319 bursts). An "unseen" model trains without the query cameras. A "seen" model gets the query cameras' other bursts and gives up the same number of bursts from other cameras. Both are scored on the same queries. "Pooled" is the plain average of a burst's scores, without the calibration behind the 0.824 figure.

| Queries | Scoring | Unseen | Seen | Gap in macro-F1 | 95% interval |
|---|---|---|---|---|---|
| A random half of each camera's bursts | pooled | 0.621 | 0.756 | +0.135 | +0.092 to +0.150 |
| | per image | 0.561 | 0.683 | +0.122 | +0.087 to +0.134 |
| The later half, support strictly earlier | pooled | 0.611 | 0.699 | +0.088 | +0.043 to +0.108 |
| | per image | 0.543 | 0.629 | +0.086 | +0.050 to +0.102 |

- **An unseen-camera cost remains with the stronger backbone:** 13.5 points against the random-burst reference, 8.8 points when only earlier bursts of the camera are available.
- **What the gap contains.** Both models train on the same number of bursts, not the same species mix. The gap is the value of labelled bursts from the camera itself, including what they say about its species mix. It is not a pure background effect.
- **The gap is smaller under the chronological split.** Bursts close in time landing on both sides is one possible explanation; the two splits also differ in difficulty and season, so this is not established.
- **Regularisation.** Raising l2 to 0.03 or 0.1 lowered both models' scores and the gap (+0.124 and +0.081 for random queries). 0.01 was fixed as the headline value beforehand.
- **Do not compare** these numbers with the ResNet's 0.618 versus 0.405, which were measured on different images.

**Exploratory breakdown, one random split, burst level.** These describe where the difference shows up; the groups differ in many ways, so they do not identify causes.
- When the query species also occurs among that camera's support bursts, accuracy is 0.833 unseen and 0.910 seen. When it does not, it is 0.652 unseen and 0.617 seen.
- By how many training bursts the species has: under 20 (10 species) 0.056 unseen and 0.296 seen; 20 to 49, 0.472 and 0.604; 50 to 149, 0.757 and 0.844; 150 or more, 0.895 and 0.943. For rare species the burst budget is not matched per species: holding out a camera removes a large share of their data.
- Night bursts 0.781 and 0.879; day bursts 0.839 and 0.888.
- The seen model is better at 64 cameras, equal at 125 and worse at 7.

**Reading agreed with Astra.** The residual gap can be reported with confidence. The mechanisms are hypotheses. Nothing here shows that using empties cannot help, or whether rare species fail for lack of examples, because of the representation, or because of the decision rule. Sensible next checks, all cheap: per-fold class counts and confusion patterns for the rare species, simple changes to the head (class weighting, logit adjustment, prototypes), and the same pipeline on crops versus whole images with identical heads.

## 5c. Why rare species fail at unseen cameras (11 October 2026)

Diagnostics and controls on the same frozen DINOv2 pipeline, with the pilot's five camera-wise folds over the 216 training cameras, so every prediction is for a camera the model did not train on. No validation or test camera. Script `pilot_a/rare_checks.py`, results in `pilot_a/results/rare_v1/`, approved by Astra and Claude beforehand. Nothing was tuned. "Pooled" is the plain average of a burst's scores.

**Where the ordinary head loses.** Overall pooled macro-F1 0.622 (per image 0.564, accuracy 0.819).

| Training bursts of the species | Species | Mean F1 | Mean recall | Predictions per true image (median) | Share of the macro-F1 shortfall |
|---|---|---|---|---|---|
| Under 20 | 10 | 0.072 | 0.049 | 0.00 | 41% |
| 20 to 49 | 15 | 0.517 | 0.443 | 0.60 | 32% |
| 50 to 149 | 19 | 0.783 | 0.786 | 0.99 | 18% |
| 150 or more | 16 | 0.871 | 0.903 | 1.05 | 9% |

- Of the ten rarest species, eight have an F1 of 0 and six are never predicted at all.
- The 25 species with fewer than 50 training bursts account for 73% of the shortfall.
- Burst recall by the bursts a fold could actually train on: 0.00 with 0 to 4 bursts, 0.06 with 5 to 9, 0.40 with 10 to 19, 0.59 with 20 to 49, 0.81 with 50 to 149, 0.91 with 150 or more.
- Several prominent confusions involve a common close relative: margay taken for ocelot (283 of 313 images), Yucatan brown brocket for red brocket (799 of 1,009), yellow-backed for black-fronted duiker (155 of 201), crested guan for great curassow (300 of 400), goat for cattle. Others cross families or orders. On average about 30% of a rare species' errors stay in its family and about 70% go to a common species.
- Distinctive species do well with few bursts: dromedary 0.996 (37 bursts), spotted hyena 0.934 (30), vulturine guineafowl 0.887 (49), olive baboon 0.871 (31).
- One of the ten rarest classes is "ave desconocida" (unknown bird), an annotator catch-all.

**Simple changes to the classifier.**

| Head | Macro-F1 per image | Pooled | Accuracy | Mean F1, 10 rarest species | Pooled difference from ordinary (95% interval over cameras) |
|---|---|---|---|---|---|
| Ordinary | 0.564 | 0.622 | 0.819 | 0.072 | |
| Minus half the log class frequency | 0.581 | 0.680 | 0.825 | 0.314 | +0.058 (+0.032 to +0.086) |
| Minus the log class frequency | 0.565 | 0.673 | 0.779 | 0.366 | +0.052 (+0.018 to +0.077) |
| Class-balanced loss | 0.570 | 0.670 | 0.815 | 0.307 | +0.049 (+0.025 to +0.070) |
| Prototypes (cosine) | 0.449 | 0.515 | 0.599 | 0.191 | −0.107 (−0.137 to −0.080) |
| Linear discriminant | 0.613 | 0.711 | 0.844 | 0.343 | +0.090 (+0.068 to +0.110) |

- Substantial performance is recoverable from the existing frozen features by changing the classifier or its decision rule.
- This does not show the features are adequate. The discriminant also changes how covariance is modelled, so its gain is not all from removing the bias toward frequent species; against the best frequency adjustment its lead is about 3 points, without an interval of its own.
- The +9 points is against the uncalibrated ordinary head on held-out training cameras, and it is the best of several untuned controls, so it is exploratory.
- Putting the log frequency inside the loss gives the same result as subtracting it afterwards (0.672), as expected with a free intercept. That is a check of the implementation, not a second result.
- The only validation number of this kind so far is from the pilot's different, calibrated pipeline: subtracting the log frequency moved it from 0.824 to 0.837.

**Less labelled support, same species.** Each of the 16 common species was cut down to k training bursts, taken from few cameras, with the ordinary head. Species-averaged recall at held-out cameras: 0.903 uncut (about 220 pure bursts per fold), 0.801 at k = 50 (41 bursts and 23 cameras actually kept), 0.685 at 20, 0.496 at 10, 0.320 at 5. The spread between species is large: F1 at k = 10 runs from 0.14 (ocelot) to 0.95 (elephant). Naturally rare species sit lower than the cut common species at the same fold-specific support (burst recall 0.06 against 0.28 with 5 to 9 bursts, 0.40 against 0.48 with 10 to 19). This is a response curve. Bursts and cameras are cut together and the species differ, so it does not isolate why naturally rare species fail.

**Next, to be fixed before it is run (done, see section 5d).** One comparison on the validation cameras of three heads with the same pooling: ordinary, frequency-adjusted with its strength chosen by cross-validation on the training cameras, and linear discriminant with its shrinkage chosen the same way. Before choosing any fine-grained method, look at the errors that remain after the better head, not the ordinary head's errors. The whole-image versus crop comparison with identical heads remains worthwhile.

## 5d. Do the classifier changes hold on the validation cameras? (run 11 October, reported 20 October 2026)

Script `pilot_a/val_compare.py`, results in `pilot_a/results/val_compare_v1/`, approved by Astra and Claude beforehand (Astra rejected the first version). Three classifiers with identical pooling (a burst's prediction is the argmax of its mean score). Every setting was chosen at held-out training cameras by camera-wise cross-validation, the validation cameras were scored once, and the contrasts, margins, primary contrast and a decision rule were fixed before that. These are prespecified exploratory outcomes, not an untouched confirmation: the pilot had already scored the ordinary head on these cameras, and the classifiers and grid ranges were chosen after the rare-species checks on the same training cameras.

Chosen on the training cameras: ordinary head l2 0.0003; frequency-adjusted head l2 0.01 with tau 0.75; linear discriminant shrinkage 0.3.

| Classifier | Validation macro-F1 (pooled) | Per image | Accuracy | Held-out training cameras, nested |
|---|---|---|---|---|
| Ordinary | 0.854 | 0.713 | 0.949 | 0.625 |
| Frequency-adjusted | 0.847 | 0.697 | 0.930 | 0.689 |
| Linear discriminant | 0.853 | 0.691 | 0.959 | 0.721 |
| Pilot's calibrated pipeline | 0.824 | | 0.936 | |

| Contrast | Validation, 32 cameras | Held-out training cameras, nested, 216 cameras |
|---|---|---|
| V1 (primary): linear discriminant minus ordinary, margin 2 points | −0.001 (−0.048 to +0.039), inconclusive | +0.096 (+0.073 to +0.120) |
| V2: linear discriminant minus frequency-adjusted, margin 1 point | +0.007 (−0.048 to +0.068), inconclusive | +0.033 (+0.009 to +0.060) |
| V3: frequency-adjusted minus ordinary, margin 1 point | −0.007 (−0.044 to +0.024), inconclusive | +0.063 (+0.037 to +0.093) |

- **Outcome:** all three contrasts are inconclusive. By the rule fixed beforehand, the classifier change is not shown on validation and the calibrated ordinary pipeline stays the baseline. The newly tuned ordinary head is kept as a control in later experiments.
- **The training-camera gains did not reproduce on validation.** The validation intervals for V1 and V3 do not include gains as large as the nested training-camera estimates. V2's interval still reaches +6.8 points, and the two sets of cameras have different species.
- **On validation the three classifiers are within 0.7 points of each other.** By training bursts per species, frequency adjustment and the discriminant do better on the rarest species present (5 species under 20 bursts: 0.62 ordinary, 0.71 frequency-adjusted, 0.65 discriminant; 8 species with 20 to 49: 0.82, 0.81, 0.90) and worse on the 17 species with 50 to 149 (0.92, 0.88, 0.86). The bins are small.
- **Descriptive, outside the contrasts:** all three re-tuned heads scored 2.3 to 3.0 points above the pilot's calibrated pipeline, with intervals that include zero. The difference means something else for each: for the ordinary head a different l2 and plain pooling, for the frequency-adjusted head also a different decision rule, for the discriminant a different classifier.
- **Checks:** the pilot's 0.824 was reproduced exactly, the grid points shared with the rare-species checks reproduce them exactly, and no fit reached the optimiser's limits.

**Why training cameras and validation disagree: a possible explanation, exploratory and not established.**
- In five-fold camera-wise cross-validation a held-out burst is scored by a model that has only part of its species' training bursts. Weighted by bursts, the median share over species is 0.61 for the ten rarest species (about 6 of 10.5 bursts) and 0.72 to 0.74 for the others. The models that scored validation had all of them. The rarest species are where a few bursts matter most (section 5c: burst recall 0.06 with 5 to 9 training bursts and 0.40 with 10 to 19, an association across species, not a learning curve for these species). So the cross-validation may overstate how much a rare-species correction helps at deployment.
- To interpret this, per-species validation F1 was joined into the table after the run. That was a post-hoc look at validation errors, outside the plan, and is recorded as such; nothing was selected with it. Two of the five rarest species present in validation, chimpanzee and bushpig, had F1 0 at held-out training cameras with the ordinary head and about 1.0 on validation with all three heads. That fits the explanation but does not separate it from easier validation images.
- Other explanations are open: only 13 validation species have under 50 training bursts, the validation cameras and regions differ, and 32 cameras give wide intervals.
- More folds would not remove the deficit. Computed from the metadata, a 10-fold split raises the median share for the rarest species only from 0.61 to 0.65, and leaving one camera out only to 0.75, because rare species are concentrated in a few cameras. No protocol that keeps the query camera unseen can give back that camera's bursts. The 10-fold check was therefore not run.

**What this changes.**
- No further work on the classifier head for now.
- A rare-species method chosen on training-camera cross-validation has to be checked on validation before it is believed, keeping in mind how small validation is.
- Next: the whole-image versus crop comparison with identical heads, back on the course's main question, the cost of an unseen camera.

**A correction to sections 5a to 5c.** The fit diagnostics there flagged only the optimiser's iteration limit, not its limit on loss evaluations. The pilot's ordinary-head fits were refitted in this run with identical iteration counts and reached neither limit. For the other earlier fits it remains unknown whether the evaluation limit ended any of them.

## 5e. How much a new camera costs with the team's other features (20 October 2026)

The measurement of section 5b, run on more feature sets: DINOv2 ViT-B/14 (ours), DINOv2 ViT-L/14 (George's cache, same images, made with our caching script) and BioCLIP on whole frames and on MegaDetector crops (Paul's features). George and Paul agreed to us reading their files. Scripts `pilot_a/gap_measure.py` (version 2, which also saves every query image's predictions) and `pilot_a/gap_compare.py`, plus `pilot_a/import_features.py` for Paul's files. Approved by Astra and Claude beforehand: Run 6 directly, Run 7 after two rejections over how to prove that Paul's feature rows line up with their file names. Training cameras only. The same seeds give exactly the same query bursts and training bursts for every feature set, so the comparisons are paired; this was asserted.

| Features (random queries, pooled, l2 0.01) | Unseen camera | Seen camera | Cost of a new camera |
|---|---|---|---|
| DINOv2 ViT-B/14, whole image | 0.621 | 0.756 | 13.5 points |
| DINOv2 ViT-L/14, whole image | 0.673 | 0.791 | 11.7 points |
| BioCLIP, whole frame | 0.494 | 0.610 | 11.6 points |
| BioCLIP, crop and fallback | 0.600 | 0.685 | 8.5 points |

- **Reproduction:** the ViT-B re-run reproduces section 5b exactly (results, every fit, saved predictions).
- **Bigger backbone (ViT-L against ViT-B):** better at unseen cameras (+5.3 points, 95% interval +2.7 to +7.1) and at seen cameras (+3.5, +2.1 to +4.8). The point estimate of the gap is smaller, but the difference is inconclusive: −1.8 (−4.2 to +1.3) against the 2-point margin fixed beforehand.
- **Crops (BioCLIP, crop and fallback against whole frame):** the gap is smaller, 8.5 against 11.6 points, a difference of −3.0 (−6.4 to −0.3), which meets the rule fixed beforehand. It comes with better scores on both sides: +10.5 points at unseen cameras and +7.5 at seen ones. Meeting the rule means the point estimate reaches the margin and the interval excludes zero, not that the reduction is known to exceed 2 points. Under the chronological split the estimated reduction is smaller and its interval includes zero (−1.4, −4.0 to +1.4); per image it is −3.5 (−4.9 to −1.6).
- **Across backbones** (secondary, intervals only): BioCLIP on whole frames is far below DINOv2 ViT-B at unseen cameras (−12.6). With crops it comes close (−2.1, −4.8 to +0.1) but is 7.1 points lower at seen cameras, so its smaller gap reflects a larger deficit at seen cameras, not better performance at unseen ones. DINOv2 ViT-L on whole images has the highest observed unseen score among these fixed-protocol pipelines.
- **Caveats:**
  - "Crop" here is Paul's pipeline: the most confident MegaDetector v5a box with confidence of at least 0.2, squared, padded by 15%, cut from the full-resolution originals, and the whole frame for the 22.5 to 23.7% of query images without such a box. Resolution, detection, padding and fallback all change together, so the gain is not attributed to background removal alone.
  - The detector's input resolution is unverified, and its training data may include these cameras.
  - l2 0.01 is fixed for all feature sets and not tuned for any of them. The sensitivity runs show strong dependence on it (higher l2 shrinks every gap, through larger losses at seen cameras), so cross-backbone gaps may partly reflect how well 0.01 suits each feature set.
  - The gap includes what a camera's own bursts say about its species mix. One fold assignment. Frozen features only, not Paul's fine-tuned models.
- **Open question:** does the same crop pipeline also help DINOv2? That needs DINOv2 features of exactly Paul's crops (same box selection, squaring, padding, clipping, source images and fallback), checked against his implementation on a sample, and its own approval. It would still not isolate background removal.

## 6. Idea B: the camera's empties as its background model

**In plain words.** The camera does not move, so its empty frames show the scene without an animal. Compare each patch of a new image with the same camera's empties, in the feature space of a frozen foundation model. Patches that differ more than empties differ from each other are probably the animal. Classify from those patches. No detector, and no training at the new camera.

**Method.**
1. Frozen ViT patch tokens (DINOv2-B/14 at 448 px).
2. Per-camera bank of patch tokens from up to 16 diverse empties, matched by position (small spatial tolerance) and kept separately for day and night.
3. *Reliability.* Compare held-out empties against the bank to learn how much empties differ from each other at that camera. Score an image's patches against that reference distribution, so the weight means "more unusual than an empty would be".
4. Pool tokens with those weights; train the species head on training cameras using the same pooling.
5. Fall back to ordinary pooling when the camera has too few empties or none at the right time of day.

**Ladder of variants, cheapest first.** Subtract the camera's mean empty embedding → project out a low-rank "nuisance" subspace learned from within-camera empty variation (ChatGPT's proposal) → nearest-neighbour patch pooling in the style of AnomalyDINO → reliability-calibrated pooling.

**Why it might be new.** We found no paper that uses a per-camera bank of empties in foundation-feature space to pool foreground for species classification. Merely applying AnomalyDINO to empties would be incremental; the contribution has to be the reliability question: when can an empty reference be trusted?

**Controls.** Wrong-camera empties; 1, 4 and 16 references; day/night mismatch; classical pixel background subtraction; MegaDetector crops and crop + whole-image fusion; a visual audit of a fixed sample.

**Stop rules.**
- Same-camera references no better than wrong-camera references and simple subtraction: stop.
- Crop + whole-image fusion dominates and B has no advantage in supervision, robustness or compute: stop the paper direction.

**Main risks.**
- Lighting and vegetation change patches more than the animal does.
- Empties are 80–93% daytime; 7 of 32 validation cameras and 12 of 46 test cameras have none.
- Do not use burst-mates as background: an animal that stays put becomes "background".

**Cost.** ChatGPT's planning estimate: 8–14 further GPU-hours for patch extraction, matching, controls and heads. Not measured.

**Odds (ChatGPT's estimate):** 20–35% for a repeatable gain over crop + whole-image fusion, clearly higher against detector-free whole-image pooling; 25–40% workshop; 3–7% main conference.

## 7. Reserve, and directions that are taken

**Reserve: synthetic probes.** Paste known animals into a target camera's empties, run the frozen model, and estimate that camera's confusion pattern to calibrate predictions (rather than retraining). It inherits segmentation and compositing problems and sits close to WildFiT and Mesnage.

**Already taken or unpromising.**
- Copy-paste onto target-camera empties: Gao, Mesnage, WildFiT.
- Generic DANN/CORAL or test-time BatchNorm: at or below ERM on the leaderboard.
- A plain geographic prior: Mac Aodha; SpeciesNet.
- Sequence attention or burst averaging as the contribution: Dussert; Context R-CNN.
- A per-camera context token: ContextViT.
- Taxonomy losses: the hierarchy is shallow (60 species in 54 genera), so little is shared.

## 8. What a paper will need

- **Protocol.** Describe the task as re-annotated, filtered iWildCam with the WILDS test cameras. Add the official iWildCam-WILDS benchmark or a second dataset with a fresh camera split.
- **Contamination.** Audit every checkpoint. MegaDetector and SpeciesNet were trained on data from the same source. Include an ImageNet-only control.
- **Information regimes.** Report separately: source only; plus target empties; plus the target camera's unlabeled animal images (transductive).
- **Statistics.** Three seeds measure optimisation noise only. Add paired resampling over cameras and bursts, and per-class changes.
- **Strong baseline first.** Frozen modern backbone, crops, calibrated burst averaging. A gain over the 40.1 ResNet baseline alone will not convince anyone.

## 9. Launching the baseline on vinzant

Assumes the Kaggle archive is downloaded, `prepare.py --height 448` has been run, and the repo is cloned in the group folder. Run inside tmux.

```bash
uv run python -m iwildcam.train --config configs/reference.yaml --split official_ood --seeds 0,1,2
```

```bash
uv run python -m iwildcam.train --config configs/reference.yaml --split random_burst --seeds 0,1,2
```

```bash
uv run python -m iwildcam.summarise
```

Targets from the README: 0.4005 (`official_ood`) and 0.6277 (`random_burst`) for `test_macro_f1_present`, mean of 3 seeds. Each epoch line prints its duration, so the total time is known after the first epoch.

## 10. How the discussion with ChatGPT went

- **Round 1, independent.** ChatGPT browsed the web and proposed five ideas; its top two (nuisance correction from empties; ecological priors from empties) matched mine.
- **Round 2, critique.** I supplied prior work it had missed (Haucke; COSMO's inference detail; Gadot; ContextViT; DeLoCo). It corrected me (crops alone do not beat AutoFT; Mesnage already uses test-location empties), showed that A's broad claim is dead (Mac Aodha; Ye), and proposed the fix for double-counted context.
- **Round 3, final check.** I proposed training the prior into the loss. ChatGPT found no exact match, called the novelty medium-low for the algorithm and stronger if the failure mechanism is demonstrated, and tightened the design (cross-fitted priors, burst weighting, a separate decision rule for macro-F1, a factorial pilot).
