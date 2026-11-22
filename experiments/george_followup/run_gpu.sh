#!/usr/bin/env bash
# GPU part of the efficiency study: compact MegaDetector over every labelled photo, then the
# features of every combination. Started through scripts/wait_for_gpu.sh, so it only begins
# once the shared GPU is free:
#   tmux new -d -s george-followup-gpu scripts/wait_for_gpu.sh followup/run_gpu.sh
set -o pipefail
G=/zfs/ai4good/groups/camera-traps
source $G/env.sh
export IWILDCAM_DATA_ROOT=$G/camera-trap/data PYTHONDONTWRITEBYTECODE=1
cd $G/users/george/followup
LOG=$G/users/george/logs/followup_gpu.log
if [ ! -f $G/users/george/detections/md6c.json ]; then
    ../.venv-megadetector/bin/python -W ignore detect.py 2>&1 | grep --line-buffered -E "^[0-9,]+ /|wrote|Error" | tee -a $LOG
    [ "${PIPESTATUS[0]}" -ne 0 ] && exit 1
fi
../.venv/bin/python run_features.py --all 2>&1 | grep --line-buffered -vE "xFormers|warnings.warn" | tee -a $LOG
exit "${PIPESTATUS[0]}"
