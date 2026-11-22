#!/usr/bin/env bash
# Every number of Experiment 3 on the slides, in order. Run from this folder on vinzant:
#   source /zfs/ai4good/groups/camera-traps/env.sh && bash run_all.sh
set -euo pipefail
PY=/zfs/ai4good/groups/camera-traps/users/george/.venv/bin/python

# 1. Features (GPU, once; the folders already exist, so these stop right away)
for backbone in resnet50 dinov2l; do
    for view in crop flip pad05 pad40; do
        $PY features.py --backbone $backbone --view $view        || true
        $PY features.py --backbone $backbone --view $view --test || true
    done
done

# 2. Steps 1-3: frozen backbone (CPU, seconds)       expected  OOD photo · burst   ID photo · burst
CUDA_VISIBLE_DEVICES="" $PY frozen.py --backbone resnet50 --step 1   # 0.531 · 0.739   0.630 · 0.792
CUDA_VISIBLE_DEVICES="" $PY frozen.py --backbone dinov2l  --step 1   # 0.732 · 0.902   0.775 · 0.925
CUDA_VISIBLE_DEVICES="" $PY frozen.py --backbone resnet50 --step 2   # 0.532 · 0.754   0.627 · 0.797
CUDA_VISIBLE_DEVICES="" $PY frozen.py --backbone dinov2l  --step 2   # 0.734 · 0.916   0.761 · 0.931
CUDA_VISIBLE_DEVICES="" $PY frozen.py --backbone resnet50 --step 3   # 0.549 · 0.758   0.641 · 0.806
CUDA_VISIBLE_DEVICES="" $PY frozen.py --backbone dinov2l  --step 3   # 0.741 · 0.919   0.764 · 0.932

# 3. Step 4: fine-tuning (GPU, hours). The saved weights exist, so training stops right away;
#    delete or move them to train again.
for seed in 0 1 2; do $PY finetune.py --backbone resnet50 --seed $seed || true; done
$PY finetune.py --backbone dinov2l --seed 0 || true

$PY score_finetuned.py --backbone resnet50    # 0.568 ± 0.003 · 0.787 ± 0.002
$PY score_finetuned.py --backbone dinov2l     # 0.740 · 0.919
