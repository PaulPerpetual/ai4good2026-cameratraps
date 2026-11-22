#!/usr/bin/env bash
# Run 1 of the idea-A pilot: cache frozen DINOv2 features for the train and validation
# cameras. Approved by Claude and GPT-6 Astra on 2026-10-10 (see APPROVALS.md).
# Start it inside tmux on vinzant:  tmux new -d -s shayan-feat ./run_cache.sh
set -uo pipefail
G=/zfs/ai4good/groups/camera-traps
source "$G/env.sh"
W=$G/users/shayan
cd "$W/ai4good2026-cameratraps"
export IWILDCAM_DATA_ROOT=$G/camera-trap/data
LOG=$W/pilot_a/logs/cache_$(date +%F_%H%M%S).log
.venv/bin/python "$W/pilot_a/cache_features.py" --out "$W/pilot_a/features/dinov2_vitb14_reg_448" "$@" 2>&1 | tee "$LOG"
echo "EXIT=${PIPESTATUS[0]}" | tee -a "$LOG"
