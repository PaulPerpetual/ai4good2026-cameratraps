#!/bin/bash
# DINOv3 ViT-L and ViT-B crop features at 448 px, then George's heads on val. Test is not read.
# Logged to logs/dinov3_448.log.
set -uo pipefail
source /zfs/ai4good/groups/camera-traps/env.sh
cd "$G/users/paul/camera-trap"
export HF_HOME="$G/users/paul/.cache/huggingface" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=16 IWILDCAM_DATA_ROOT=$G/camera-trap/data
exec > >(tee -a logs/dinov3_448.log) 2>&1
.venv/bin/python cache_dinov3_448.py --model vit_large_patch16_dinov3.lvd1689m --crop --out feats/dinov3_vitl16_448_crop || exit 1
.venv/bin/python cache_dinov3_448.py --model vit_base_patch16_dinov3.lvd1689m --crop --out feats/dinov3_vitb16_448_crop || exit 1
CUDA_VISIBLE_DEVICES="" .venv/bin/python lda_val.py dinov2L_448=$G/features/dinov2_vitl14_reg_448_crop dinov3L_256=feats/dinov3_vitl16_crop dinov3L_448=feats/dinov3_vitl16_448_crop dinov3B_448=feats/dinov3_vitb16_448_crop | tee results/lda_val_dinov3_448.txt
echo "== $(date) done"
