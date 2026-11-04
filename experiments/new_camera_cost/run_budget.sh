#!/usr/bin/env bash
# Run 10: the annotation budget at a new camera (gap_budget.py) on DINOv2 ViT-L crop and whole
# frame (George's caches, read only), then the CPU-only exploratory analysis of Run 9's
# predictions (local_analysis.py). Training cameras only.
# Needs the approval of Claude and GPT-6 Astra recorded in APPROVALS.md before it is started.
# Start it inside tmux on vinzant:  tmux new -d -s shayan-budget ./run_budget.sh
set -uo pipefail
G=/zfs/ai4good/groups/camera-traps
source "$G/env.sh"
W=$G/users/shayan
R=$W/pilot_a/results
cd "$W/ai4good2026-cameratraps"
for NAME in budget_v1 local_analysis_v1; do
    if [ -e "$R/$NAME" ]; then echo "results/$NAME exists; not overwriting"; exit 9; fi
done
LOG=$W/pilot_a/logs/budget_v1_$(date +%F_%H%M%S).log
.venv/bin/python "$W/pilot_a/gap_budget.py" \
    --cache vitl_crop="$G/features/dinov2_vitl14_reg_448_crop" --cache vitl="$G/features/dinov2_vitl14_reg_448" \
    --primary vitl_crop --check-decompose "$R/decompose_v1/query_images.csv.gz" \
    --repo . --out "$R/budget_v1" --threads 32 --device cuda --max-iter 2000 --max-minutes 60 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "EXIT=$status" | tee -a "$LOG"
[ "$status" -ne 0 ] && exit "$status"
LOG=$W/pilot_a/logs/local_analysis_v1_$(date +%F_%H%M%S).log
.venv/bin/python "$W/pilot_a/local_analysis.py" --decompose "$R/decompose_v1" --cache "$G/features/dinov2_vitl14_reg_448" \
    --sets vitl,vitl_crop --repo . --out "$R/local_analysis_v1" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "EXIT=$status" | tee -a "$LOG"
exit "$status"
