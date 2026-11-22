#!/usr/bin/env bash
# Run 11: which bursts to label at a new camera (gap_select.py): random vs spread-out vs most
# uncertain, on DINOv2 ViT-L crop and whole frame (George's caches, read only). Training cameras only.
# Needs the approval of Claude and GPT-6 Astra recorded in APPROVALS.md before it is started.
# Start it inside tmux on vinzant:  tmux new -d -s shayan-select ./run_select.sh
set -uo pipefail
G=/zfs/ai4good/groups/camera-traps
source "$G/env.sh"
W=$G/users/shayan
R=$W/pilot_a/results
cd "$W/ai4good2026-cameratraps"
NAME=select_v1
if [ -e "$R/$NAME" ]; then echo "results/$NAME exists; not overwriting"; exit 9; fi
LOG=$W/pilot_a/logs/${NAME}_$(date +%F_%H%M%S).log
.venv/bin/python "$W/pilot_a/gap_select.py" \
    --cache vitl_crop="$G/features/dinov2_vitl14_reg_448_crop" --cache vitl="$G/features/dinov2_vitl14_reg_448" \
    --primary vitl_crop --check-budget "$R/budget_v1/query_images.csv.gz" \
    --repo . --out "$R/$NAME" --threads 32 --device cuda --max-iter 2000 --max-minutes 60 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "EXIT=$status" | tee -a "$LOG"
exit "$status"
