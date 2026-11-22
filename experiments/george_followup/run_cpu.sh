#!/usr/bin/env bash
# CPU part of the follow-up: re-check Experiment 3, then calibration and long tail.
G=/zfs/ai4good/groups/camera-traps
source $G/env.sh
export IWILDCAM_DATA_ROOT=$G/camera-trap/data CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 PYTHONDONTWRITEBYTECODE=1
PY=$G/users/george/.venv/bin/python
LOG=$G/users/george/logs/followup_cpu.log
{
cd $G/users/george/exp3
for b in resnet50 dinov2l; do for s in 1 2 3; do $PY frozen.py --backbone $b --step $s; done; done
cd $G/users/george/followup
for b in resnet50 dinov2l; do $PY calibration.py --backbone $b; done
for b in resnet50 dinov2l; do $PY long_tail.py --backbone $b; done
echo "cpu part done"
} 2>&1 | tee $LOG
