#!/usr/bin/env bash
# Run 2 of the idea-A pilot: the factorial comparison on cached features. CPU only.
# Approved by Claude and GPT-6 Astra on 2026-10-10 (see APPROVALS.md).
# Start it inside tmux on vinzant:  tmux new -d -s shayan-pilot ./run_pilot.sh
set -uo pipefail
G=/zfs/ai4good/groups/camera-traps
source "$G/env.sh"
W=$G/users/shayan
cd "$W/ai4good2026-cameratraps"
LOG=$W/pilot_a/logs/pilot_$(date +%F_%H%M%S).log
CUDA_VISIBLE_DEVICES="" .venv/bin/python "$W/pilot_a/pilot_a.py" \
    --cache "$W/pilot_a/features/dinov2_vitb14_reg_448" --repo . \
    --out "$W/pilot_a/results/pilot_a_v1" --threads 32 "$@" 2>&1 | tee "$LOG"
echo "EXIT=${PIPESTATUS[0]}" | tee -a "$LOG"
