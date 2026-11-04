#!/usr/bin/env bash
# Run 5: three classifiers on the validation cameras, settings chosen on the training cameras.
# Needs the approval of Claude and GPT-6 Astra recorded in APPROVALS.md before it is started.
# Start it inside tmux on vinzant:  tmux new -d -s shayan-valcmp ./run_valcmp.sh
set -uo pipefail
G=/zfs/ai4good/groups/camera-traps
source "$G/env.sh"
W=$G/users/shayan
cd "$W/ai4good2026-cameratraps"
LOG=$W/pilot_a/logs/valcmp_$(date +%F_%H%M%S).log
.venv/bin/python "$W/pilot_a/val_compare.py" \
    --cache "$W/pilot_a/features/dinov2_vitb14_reg_448" --repo . \
    --out "$W/pilot_a/results/val_compare_v1" --threads 32 \
    --device cuda --max-iter 2000 --max-minutes 30 "$@" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "EXIT=$status" | tee -a "$LOG"
exit "$status"
