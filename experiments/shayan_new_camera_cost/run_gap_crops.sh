#!/usr/bin/env bash
# Run 8: the new-camera cost (gap_measure.py) on George's crop features and the ResNet-50
# whole-frame control, all with the same index as Shayan's cache (read only):
#   DINOv2 ViT-L/14 on MegaDetector crops   (whole-frame partner: results/gap_vitl14_v1, Run 6)
#   ResNet-50 on whole frames and on crops   (a third backbone for the crop contrast)
# The crop rule is Paul's (most confident MD v5a box >= 0.2, squared, 15% margin, whole-frame
# fallback), the same one as the BioCLIP crops of Run 7.
# Needs the approval of Claude and GPT-6 Astra recorded in APPROVALS.md before it is started.
# Start it inside tmux on vinzant:  tmux new -d -s shayan-gap ./run_gap_crops.sh
set -uo pipefail
G=/zfs/ai4good/groups/camera-traps
source "$G/env.sh"
W=$G/users/shayan
cd "$W/ai4good2026-cameratraps"
for spec in "$G/features/dinov2_vitl14_reg_448_crop gap_vitl14_crop_v1" \
            "$G/features/resnet50_448 gap_resnet50_v1" \
            "$G/features/resnet50_448_crop gap_resnet50_crop_v1"; do
    set -- $spec
    CACHE=$1; NAME=$2
    if [ -e "$W/pilot_a/results/$NAME" ]; then echo "results/$NAME exists; not overwriting"; exit 9; fi
    LOG=$W/pilot_a/logs/${NAME}_$(date +%F_%H%M%S).log
    .venv/bin/python "$W/pilot_a/gap_measure.py" --cache "$CACHE" --repo . --out "$W/pilot_a/results/$NAME" \
        --threads 32 --device cuda --max-iter 2000 --max-minutes 30 2>&1 | tee "$LOG"
    status=${PIPESTATUS[0]}
    echo "EXIT=$status" | tee -a "$LOG"
    if [ "$status" -ne 0 ]; then echo "run on $CACHE failed; not starting the next one"; exit "$status"; fi
done
