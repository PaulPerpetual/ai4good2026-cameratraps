#!/usr/bin/env bash
# Run 9: how much of the new-camera cost is the camera's own images, and how much its species
# (gap_decompose.py), on seven read-only feature caches with the same index: DINOv2 ViT-L whole
# and crop (George), DINOv2 ViT-B whole (Shayan), ResNet-50 whole and crop (George), BioCLIP
# whole and crop (Paul, imported). Every cache is checked against its earlier gap run.
# Needs the approval of Claude and GPT-6 Astra recorded in APPROVALS.md before it is started.
# Start it inside tmux on vinzant:  tmux new -d -s shayan-decomp ./run_decompose.sh
set -uo pipefail
G=/zfs/ai4good/groups/camera-traps
source "$G/env.sh"
W=$G/users/shayan
F=$W/pilot_a/features
R=$W/pilot_a/results
cd "$W/ai4good2026-cameratraps"
NAME=decompose_v1
if [ -e "$R/$NAME" ]; then echo "results/$NAME exists; not overwriting"; exit 9; fi
LOG=$W/pilot_a/logs/${NAME}_$(date +%F_%H%M%S).log
.venv/bin/python "$W/pilot_a/gap_decompose.py" \
    --cache vitl="$G/features/dinov2_vitl14_reg_448"           --check-queries vitl="$R/gap_vitl14_v1/query_images.csv.gz" \
    --cache vitl_crop="$G/features/dinov2_vitl14_reg_448_crop" --check-queries vitl_crop="$R/gap_vitl14_crop_v1/query_images.csv.gz" \
    --cache vitb="$F/dinov2_vitb14_reg_448"                    --check-queries vitb="$R/gap_vitb14_v2/query_images.csv.gz" \
    --cache resnet="$G/features/resnet50_448"                  --check-queries resnet="$R/gap_resnet50_v1/query_images.csv.gz" \
    --cache resnet_crop="$G/features/resnet50_448_crop"        --check-queries resnet_crop="$R/gap_resnet50_crop_v1/query_images.csv.gz" \
    --cache bioclip="$F/bioclip_whole_paul"                    --check-queries bioclip="$R/gap_bioclip_whole_v1/query_images.csv.gz" \
    --cache bioclip_crop="$F/bioclip_crop_paul"                --check-queries bioclip_crop="$R/gap_bioclip_crop_v1/query_images.csv.gz" \
    --primary vitl --pair vitl_crop=vitl --pair resnet_crop=resnet --pair bioclip_crop=bioclip \
    --repo . --out "$R/$NAME" --threads 32 --device cuda --max-iter 2000 --max-minutes 150 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "EXIT=$status" | tee -a "$LOG"
exit "$status"
