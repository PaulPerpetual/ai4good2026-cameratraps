> Imported from `pdelap-background-20261018/` on vinzant (2026-11-22). Its `iwildcam/` is Pia's
> modified copy of the starter package (adds backgrounds, foregrounds, crop comparison, threshold
> sweep); run the commands below from this folder so that copy is imported instead of the repo's.
> Detections (`data/md-v5a.json`, also the protocol's crop source) and cutouts stay in that folder.
> Results at import: `results_by_config.csv` (one seed per mode except whole, which has 3) and
> `results_crop_vs_sam.json` (3 seeds). Crop beat whole frame by about 8.5 macro F1 points (test).

# Background replacement experiments

Question: does background diversity improve unseen-camera macro F1, and does using
empty deployment-camera frames add anything beyond source-camera augmentation?
Copy-paste is established, including on iWildCam; see
[Gao et al. (2023)](https://proceedings.mlr.press/v202/gao23g.html).
The contribution here is the controlled ablation under this project's protocol.

## Preparation

Prepare JPEGs as described in README.md, then run a fixed MegaDetector checkpoint.
For example, with the [official batch CLI](https://github.com/agentmorris/MegaDetector):

```powershell
uv run --with megadetector python -m megadetector.detection.run_detector_batch MDV5A data/images_h224 data/md.json --recursive --output_relative_filenames
```

Record the resolved detector version. If detector dependencies conflict with this
project's pinned environment, run detection in a separate environment and copy the JSON.
The preparation command requires modern MD JSON (`images[].file`, normalized XYWH
boxes, category `1 = animal`). Legacy absolute-pixel iWildCam detections are rejected.
Use the same image directory for detection, foreground preparation, and training.

Obtain the SAM ViT-B checkpoint using
[Meta's instructions](https://github.com/facebookresearch/segment-anything#model-checkpoints).
SAM is needed only for preparation, not classifier training:

```powershell
uv run --with "segment-anything @ git+https://github.com/facebookresearch/segment-anything.git" python -m iwildcam.prepare_foregrounds --detections data/md.json --image-dir data/images_h224 --output data/foregrounds-v1 --sam-checkpoint data/sam_vit_b_01ec64.pth
```

For crop-only preparation, omit the SAM checkpoint and `--with` argument.
Preparation selects the highest-confidence animal detection at threshold 0.2, prompts
SAM with its box, and clips the mask to that box. Multiple animals are approximated by
the top detection. Inspect cutouts for wrong animals, missing limbs, and background
leakage before a sweep. Thresholds must be selected with validation data.

Frozen inference processes all available labelled images, including evaluation images
for crop/neutral evaluation, without using species labels. Only training-image animals
are composited during training. Each new output directory contains PNG cutouts and a
manifest with detector metadata, input/checkpoint/cutout hashes, and failure counts.
Use versioned artifact directories; do not modify cutouts after preparation.

## Five ablations

The supplied config uses ResNet18 at 224px for iteration. Keep backbone, schedule,
resolution, probability, and seeds fixed across ablations:

```powershell
uv run python -m iwildcam.train --config configs/background.yaml --background-mode whole --seeds 0,1,2
uv run python -m iwildcam.train --config configs/background.yaml --background-mode crop --seeds 0,1,2
uv run python -m iwildcam.train --config configs/background.yaml --background-mode neutral --seeds 0,1,2
uv run python -m iwildcam.train --config configs/background.yaml --background-mode source --seeds 0,1,2
uv run python -m iwildcam.train --config configs/background.yaml --background-mode target --seeds 0,1,2
uv run python -m iwildcam.summarise
```

Override artifact location with `--foreground-manifest PATH`. For final experiments,
copy the background settings into a config with the reference backbone and schedule.

| Mode | Training input | Evaluation input |
|---|---|---|
| whole | Original | Original |
| crop | Detector crop | Detector crop |
| neutral | Centered cutout on grey canvas | Centered cutout on grey canvas |
| source | Original or animal on source empty frame | Original |
| target | Original or animal on target empty frame | Original |

Replacement probability defaults to 0.5. Source backgrounds use only empty rows with
`official_ood=train`; target backgrounds use only empty rows with `official_ood=test`.
Validation backgrounds are excluded. Both modes require `official_ood`. Sampling is
uniform over cameras, then frames; placement is random and original relative box size
is retained. Lighting and ecological plausibility are not matched. An empty background
pool fails explicitly. Randomness uses seeded torch generators, including loader workers.

Missing detections/empty masks fall back to the original full image in every mode.
No evaluation examples are dropped; one shared classifier handles fallbacks.

## Reporting and limitations

Report mean/std `test_macro_f1_present` across seeds for all five methods.
Results include per-camera metrics, preprocessing success counts per split, sampled
background camera IDs, and `test_empty_coverage`: separate scores for cameras with and
without available empty frames. Compare methods within each subgroup; the groups can
have different species distributions. Manifest digests enter experiment identities,
and prediction files include run keys to avoid overwriting other ablations.

Report target adaptation separately from source-only domain generalization. It uses
test-camera empty frames intentionally, as allowed by this project. Select settings
with validation scores, never test scores. This implementation does not simulate
adaptation to validation-camera backgrounds; add that experiment before extensive
target-adaptation tuning. Segmentation errors, unrealistic composites, and loss of
useful context can reduce performance; no improvement is assumed.
