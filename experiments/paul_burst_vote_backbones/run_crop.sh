#!/bin/bash
# BioCLIP on MegaDetector crops: extract features on the GPU, then probe and analyse on the CPU.
# Run from the repo root inside tmux; everything is logged to logs/crop-pipeline.log.
set -euo pipefail
source /zfs/ai4good/groups/camera-traps/env.sh
cd "$G/users/paul/camera-trap"
export HF_HOME="$G/users/paul/.cache/huggingface"
exec > >(tee -a logs/crop-pipeline.log) 2>&1

echo "== $(date) extract bioclip crops"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
uv run python linprobe.py extract --backbone bioclip --view crop --num-workers 12

# the rest needs no GPU; keep the CPU footprint modest on the shared node
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16
echo "== $(date) probe"
uv run python linprobe.py probe --backbone bioclip --view crop
echo "== $(date) analyse"
uv run python linprobe.py analyse --backbone bioclip --view crop --num-workers 16
echo "== $(date) votes"
uv run python linprobe.py votes --feats bioclip bioclip-crop resnet50
echo "== $(date) done"
