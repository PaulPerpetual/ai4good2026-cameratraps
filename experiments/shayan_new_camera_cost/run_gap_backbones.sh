#!/usr/bin/env bash
# Run 6: the new-camera cost (gap_measure.py) on two feature caches with the same index:
# Shayan's DINOv2 ViT-B/14 (a re-run of Run 3 that also saves every query image's predictions)
# and George's DINOv2 ViT-L/14 (read only, used with his permission).
# Needs the approval of Claude and GPT-6 Astra recorded in APPROVALS.md before it is started.
# Start it inside tmux on vinzant:  tmux new -d -s shayan-gap ./run_gap_backbones.sh
set -uo pipefail
G=/zfs/ai4good/groups/camera-traps
source "$G/env.sh"
W=$G/users/shayan
cd "$W/ai4good2026-cameratraps"
for spec in "$W/pilot_a/features/dinov2_vitb14_reg_448 gap_vitb14_v2" "$G/features/dinov2_vitl14_reg_448 gap_vitl14_v1"; do
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
