#!/usr/bin/env bash
# Run 4: rare-species diagnostics and head controls for the frozen-feature pipeline.
# Approved by Claude and GPT-6 Astra on 2026-10-10 (see APPROVALS.md).
# Start it inside tmux on vinzant:  tmux new -d -s shayan-rare ./run_rare.sh
set -uo pipefail
G=/zfs/ai4good/groups/camera-traps
source "$G/env.sh"
W=$G/users/shayan
cd "$W/ai4good2026-cameratraps"
LOG=$W/pilot_a/logs/rare_$(date +%F_%H%M%S).log
.venv/bin/python "$W/pilot_a/rare_checks.py" \
    --cache "$W/pilot_a/features/dinov2_vitb14_reg_448" --repo . \
    --out "$W/pilot_a/results/rare_v1" --threads 32 \
    --device cuda --max-iter 2000 --max-minutes 30 "$@" 2>&1 | tee "$LOG"
echo "EXIT=${PIPESTATUS[0]}" | tee -a "$LOG"
