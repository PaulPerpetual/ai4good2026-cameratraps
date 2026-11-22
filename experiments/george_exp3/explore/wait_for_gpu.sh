#!/usr/bin/env bash
# Wait until nobody else is on the GPU, then run a command. If the command reports that the
# GPU was busy (exit 2) or that it gave the GPU back to another job (exit 4), wait again and
# re-run it; cache_features.py resumes from its last checkpoint. Any other exit code ends
# the loop. Start inside tmux:
#   tmux new -d -s george-wait ./scripts/wait_for_gpu.sh ./scripts/run_cache_vitl.sh
set -uo pipefail
INTERVAL=${INTERVAL:-300}                 # seconds between checks
LOG=logs/wait_for_gpu_$(date +%F_%H%M%S).log
mkdir -p logs
say() { echo "$(date "+%F %T")  $*" | tee -a "$LOG"; }

gpu_free() {
    local used apps
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    [ "$used" -lt 2000 ] && [ "$apps" -eq 0 ]
}

say "waiting for a free GPU to run: $*"
checks=0
while true; do
    if gpu_free; then
        say "GPU free, starting"
        "$@"; rc=$?
        case $rc in
            0) say "finished (exit 0)"; exit 0 ;;
            2|4) say "GPU taken again (exit $rc); waiting" ;;
            *) say "command failed (exit $rc); stopping"; exit "$rc" ;;
        esac
    fi
    checks=$((checks + 1))
    [ $((checks % 12)) -eq 0 ] && say "still waiting ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader) in use)"
    sleep "$INTERVAL"
done
